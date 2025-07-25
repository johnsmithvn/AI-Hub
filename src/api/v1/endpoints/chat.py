

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncIterator
import time
import json
import asyncio
import torch
from uuid import uuid4
from loguru import logger

from src.core.custom_model_manager import CustomModelManager
from ....core.redis_client import get_cache
from ....schemas.chat import (
    ChatCompletionRequest, ChatCompletionResponse, 
    ChatMessage, ChatCompletionChoice
)

router = APIRouter()

async def get_model_manager(request: Request) -> CustomModelManager:
    """Dependency to get custom model manager"""
    return request.app.state.model_manager

@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    model_manager: CustomModelManager = Depends(get_model_manager)
):
    """
    Create a chat completion (OpenAI-compatible)
    Supports both streaming and non-streaming responses
    """
    start_time = time.time()
    
    try:
        # Model selection and switching
        model_name = request.model
        if not model_name or model_name == "auto":
            # Intelligent model selection
            model_name = await model_manager.get_best_model_for_task(
                task_type="chat",
                language="en",  # Could be detected from input
                complexity="medium"
            )
        
        if not model_name:
            raise HTTPException(status_code=400, detail="No suitable model available")
        
        # Switch to the selected model
        success = await model_manager.switch_model(model_name)
        if not success:
            raise HTTPException(status_code=503, detail=f"Failed to load model {model_name}")
        
        # Process the request
        if request.stream:
            return StreamingResponse(
                generate_streaming_response(request, model_manager, model_name, start_time),
                media_type="text/plain"
            )
        else:
            return await generate_complete_response(request, model_manager, model_name, start_time)
    
    except Exception as e:
        response_time = time.time() - start_time
        await log_request_error(request, str(e), response_time)
        raise HTTPException(status_code=500, detail=str(e))

async def generate_complete_response(
    request: ChatCompletionRequest,
    model_manager: CustomModelManager,
    model_name: str,
    start_time: float
) -> ChatCompletionResponse:
    """Generate complete (non-streaming) chat response"""
    
    # Check if model exists in registry
    if model_name not in model_manager.model_registry:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model_info = model_manager.model_registry[model_name]
    
    # Generate response based on provider
    if model_info.provider == "ollama":
        # Check if ollama client is available
        if not hasattr(model_manager, 'ollama_client') or model_manager.ollama_client is None:
            logger.warning(f"Ollama client not configured, falling back to HuggingFace for model {model_name}")
            response_text = await generate_hf_response(request, model_manager, model_name)
        else:
            response_text = await generate_ollama_response(request, model_manager, model_name)
    elif model_info.provider in ["huggingface", "local"]:
        response_text = await generate_hf_response(request, model_manager, model_name)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {model_info.provider}")
    
    # Calculate metrics
    response_time = time.time() - start_time
    model_manager.record_request(model_name, response_time)
    
    # Build response
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex[:29]}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": estimate_tokens(request.messages),
            "completion_tokens": estimate_tokens([{"content": response_text}]),
            "total_tokens": estimate_tokens(request.messages) + estimate_tokens([{"content": response_text}])
        }
    )
    
    # Cache conversation if session_id provided
    if hasattr(request, 'session_id') and request.session_id:
        await cache_conversation_turn(request.session_id, request.messages, response_text)
    
    return response

async def generate_streaming_response(
    request: ChatCompletionRequest,
    model_manager: CustomModelManager,
    model_name: str,
    start_time: float
) -> AsyncIterator[str]:
    """Generate streaming chat response"""
    
    try:
        if model_name not in model_manager.model_registry:
            error_chunk = {
                "error": {"message": f"Model {model_name} not found", "type": "model_not_found"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            return
            
        model_info = model_manager.model_registry[model_name]
        
        if model_info.provider == "ollama" and hasattr(model_manager, 'ollama_client') and model_manager.ollama_client:
            async for chunk in generate_ollama_streaming(request, model_manager, model_name):
                yield f"data: {json.dumps(chunk)}\n\n"
        else:
            # For non-Ollama models or when Ollama client is unavailable, simulate streaming by chunking the response
            response_text = await generate_hf_response(request, model_manager, model_name)
            
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = {
                    "id": f"chatcmpl-{uuid4().hex[:29]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": word + " " if i < len(words) - 1 else word},
                        "finish_reason": None if i < len(words) - 1 else "stop"
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)  # Small delay for realistic streaming
        
        # Send final chunk
        yield "data: [DONE]\n\n"
        
        # Record metrics
        response_time = time.time() - start_time
        model_manager.record_request(model_name, response_time)
        
    except Exception as e:
        error_chunk = {
            "error": {"message": str(e), "type": "server_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

async def generate_ollama_response(
    request: ChatCompletionRequest,
    model_manager: CustomModelManager,
    model_name: str
) -> str:
    """Generate response using Ollama"""
    
    # Validate ollama client exists
    if not hasattr(model_manager, 'ollama_client') or model_manager.ollama_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Ollama client not available. Please configure Ollama or use HuggingFace models."
        )
    
    # Convert messages to Ollama format
    ollama_messages = []
    for msg in request.messages:
        ollama_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    try:
        # Make request to Ollama
        response = await model_manager.ollama_client.chat(
            model=model_name,
            messages=ollama_messages,
            options={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
            }
        )
        
        return response['message']['content']
        
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        raise HTTPException(status_code=503, detail=f"Ollama service error: {str(e)}")

async def generate_ollama_streaming(
    request: ChatCompletionRequest,
    model_manager: CustomModelManager,
    model_name: str
) -> AsyncIterator[Dict]:
    """Generate streaming response using Ollama"""
    
    # Validate ollama client exists
    if not hasattr(model_manager, 'ollama_client') or model_manager.ollama_client is None:
        yield {
            "error": {"message": "Ollama client not available", "type": "service_unavailable"}
        }
        return
    
    # Convert messages to Ollama format
    ollama_messages = []
    for msg in request.messages:
        ollama_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    try:
        # Stream from Ollama
        async for chunk in await model_manager.ollama_client.chat(
            model=model_name,
            messages=ollama_messages,
            stream=True,
            options={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
            }
        ):
            if chunk.get('message', {}).get('content'):
                yield {
                    "id": f"chatcmpl-{uuid4().hex[:29]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk['message']['content']},
                        "finish_reason": "stop" if chunk.get('done') else None
                    }]
                }
    except Exception as e:
        logger.error(f"Ollama streaming failed: {e}")
        yield {
            "error": {"message": f"Ollama streaming error: {str(e)}", "type": "service_error"}
        }

async def generate_hf_response(
    request: ChatCompletionRequest,
    model_manager: CustomModelManager,
    model_name: str
) -> str:
    """Generate response using HuggingFace models - delegates to model manager"""
    
    # Format conversation into a single prompt
    conversation_text = ""
    for msg in request.messages:
        role_prefix = {"user": "Human: ", "assistant": "Assistant: ", "system": "System: "}.get(msg.role, f"{msg.role}: ")
        conversation_text += f"{role_prefix}{msg.content}\n"
    
    # Add generation prompt
    conversation_text += "Assistant: "
    
    # Use the centralized generate_response method from model manager
    response = await model_manager.generate_response(
        model_name=model_name,
        prompt=conversation_text,
        max_tokens=request.max_tokens or 512,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=False
    )
    
    return response

# Utility functions
def estimate_tokens(messages: List[Dict]) -> int:
    """Rough token estimation"""
    total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
    return max(1, total_chars // 4)  # Rough approximation

async def cache_conversation_turn(session_id: str, messages: List[ChatMessage], response: str):
    """Cache conversation turn for history"""
    cache = await get_cache()
    
    # Get existing conversation
    conversation = await cache.get(f"conversation:{session_id}") or []
    
    # Add new messages
    conversation.extend([msg.dict() for msg in messages])
    conversation.append({"role": "assistant", "content": response})
    
    # Cache updated conversation
    await cache.set(f"conversation:{session_id}", conversation, expire=86400)  # 24 hours

async def log_request_error(request: ChatCompletionRequest, error: str, response_time: float):
    """Log request error for analytics - Enhanced implementation"""
    try:
        error_data = {
            "timestamp": time.time(),
            "model": request.model,
            "error": error,
            "response_time": response_time,
            "message_count": len(request.messages),
            "user_id": getattr(request, 'user_id', 'anonymous')
        }
        
        # Log to console for now (could be enhanced to store in database)
        logger.error(f"Chat completion error: {error_data}")
        
        # TODO: Store in analytics database when implemented
        # await analytics_service.log_error(error_data)
        
    except Exception as e:
        logger.error(f"Failed to log request error: {e}")

@router.get("/models")
async def list_chat_models(
    model_manager: CustomModelManager = Depends(get_model_manager)
):
    """List available chat models"""
    models = await model_manager.get_available_models()
    
    chat_models = [
        {
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ai-hub",
            "permission": [],
            "root": name,
            "parent": None,
        }
        for name, info in models.items()
        if info.model_type.value in ["chat", "code"]  # Include both chat and code models
    ]
    
    return {"object": "list", "data": chat_models}

@router.post("/switch-model")
async def switch_model(
    model_name: str,
    model_manager: CustomModelManager = Depends(get_model_manager)
):
    """Switch to a different model"""
    success = await model_manager.switch_model(model_name)
    
    if success:
        return {"success": True, "active_model": model_name}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to switch to model {model_name}")

@router.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    cache = await get_cache()
    conversation = await cache.get(f"conversation:{session_id}")
    
    if not conversation:
        return {"session_id": session_id, "messages": []}
    
    return {"session_id": session_id, "messages": conversation}
