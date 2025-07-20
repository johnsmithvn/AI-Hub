"""
Chat endpoints witasync def get_model_manager_dep(request: Request):
    """Dependency to get model manager"""
    return await get_model_manager()

@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    model_manager = Depends(get_model_manager_dep)
):compatible API and advanced features
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncIterator
import time
import json
import asyncio
import torch
from uuid import uuid4

from ....core.custom_model_manager import get_model_manager
from ....core.redis_client import get_cache
from ....schemas.chat import (
    ChatCompletionRequest, ChatCompletionResponse, 
    ChatMessage, ChatCompletionChoice
)

router = APIRouter()

async def get_model_manager(request: Request) -> ModelManager:
    """Dependency to get model manager"""
    return request.app.state.model_manager

@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
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
    model_manager: ModelManager,
    model_name: str,
    start_time: float
) -> ChatCompletionResponse:
    """Generate complete (non-streaming) chat response"""
    
    # Get the loaded model based on provider
    model_info = model_manager.models[model_name]
    
    if model_info.provider == "ollama":
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
    model_manager: ModelManager,
    model_name: str,
    start_time: float
) -> AsyncIterator[str]:
    """Generate streaming chat response"""
    
    try:
        model_info = model_manager.models[model_name]
        
        if model_info.provider == "ollama":
            async for chunk in generate_ollama_streaming(request, model_manager, model_name):
                yield f"data: {json.dumps(chunk)}\n\n"
        else:
            # For non-Ollama models, simulate streaming by chunking the response
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
    model_manager: ModelManager,
    model_name: str
) -> str:
    """Generate response using Ollama"""
    
    # Convert messages to Ollama format
    ollama_messages = []
    for msg in request.messages:
        ollama_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
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

async def generate_ollama_streaming(
    request: ChatCompletionRequest,
    model_manager: ModelManager,
    model_name: str
) -> AsyncIterator[Dict]:
    """Generate streaming response using Ollama"""
    
    # Convert messages to Ollama format
    ollama_messages = []
    for msg in request.messages:
        ollama_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
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

async def generate_hf_response(
    request: ChatCompletionRequest,
    model_manager: ModelManager,
    model_name: str
) -> str:
    """Generate response using HuggingFace models"""
    
    # Get model and tokenizer
    model = model_manager.loaded_models[model_name]
    tokenizer = model_manager.tokenizers[model_name]
    
    # Format conversation
    conversation = []
    for msg in request.messages:
        conversation.append({"role": msg.role, "content": msg.content})
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

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
    """Log request error for analytics"""
    # This could be enhanced to store in database
    pass

@router.get("/models")
async def list_chat_models(
    model_manager: ModelManager = Depends(get_model_manager)
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
    model_manager: ModelManager = Depends(get_model_manager)
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
