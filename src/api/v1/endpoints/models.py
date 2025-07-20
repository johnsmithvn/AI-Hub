"""
Model management endpoints for dynamic loading, switching, and monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio

from ...core.model_manager import ModelManager, ModelInfo, ModelType, ModelStatus

router = APIRouter()

class ModelListResponse(BaseModel):
    models: List[Dict[str, Any]]
    total: int
    loaded: int

class ModelStatusResponse(BaseModel):
    name: str
    status: str
    provider: str
    model_type: str
    vram_usage_gb: float
    load_time: float
    last_used: float
    performance_metrics: Dict[str, float]

class ModelSwitchRequest(BaseModel):
    model_name: str
    force: bool = Field(default=False)

class ModelLoadRequest(BaseModel):
    model_name: str
    force: bool = Field(default=False)
    priority: int = Field(default=5, ge=1, le=10)

async def get_model_manager(request: Request) -> ModelManager:
    """Dependency to get model manager"""
    return request.app.state.model_manager

@router.get("/", response_model=ModelListResponse)
async def list_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    provider: Optional[str] = None,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """List all available models with filtering options"""
    
    models = await model_manager.get_available_models()
    
    # Apply filters
    filtered_models = []
    for name, info in models.items():
        # Filter by type
        if model_type and info.model_type.value != model_type:
            continue
        
        # Filter by status
        if status and info.status.value != status:
            continue
        
        # Filter by provider
        if provider and info.provider != provider:
            continue
        
        model_data = {
            "name": name,
            "display_name": name,
            "provider": info.provider,
            "model_type": info.model_type.value,
            "status": info.status.value,
            "vram_usage_gb": info.vram_usage / (1024**3) if info.vram_usage else 0,
            "load_time": info.load_time,
            "last_used": info.last_used,
            "specialties": info.specialties,
            "languages": info.languages,
            "config": info.config,
            "performance_metrics": info.performance_metrics
        }
        filtered_models.append(model_data)
    
    loaded_count = sum(1 for m in filtered_models if m["status"] == "loaded")
    
    return ModelListResponse(
        models=filtered_models,
        total=len(filtered_models),
        loaded=loaded_count
    )

@router.get("/{model_name}/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get detailed status of a specific model"""
    
    models = await model_manager.get_available_models()
    
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    info = models[model_name]
    
    return ModelStatusResponse(
        name=model_name,
        status=info.status.value,
        provider=info.provider,
        model_type=info.model_type.value,
        vram_usage_gb=info.vram_usage / (1024**3) if info.vram_usage else 0,
        load_time=info.load_time,
        last_used=info.last_used,
        performance_metrics=info.performance_metrics
    )

@router.post("/{model_name}/load")
async def load_model(
    model_name: str,
    request: ModelLoadRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Load a specific model"""
    
    # Start loading in background for better responsiveness
    if request.priority >= 8:  # High priority - load immediately
        success = await model_manager.load_model(model_name, force=request.force)
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}")
        
        return {"success": True, "message": f"Model {model_name} loaded successfully"}
    else:
        # Lower priority - load in background
        background_tasks.add_task(model_manager.load_model, model_name, request.force)
        return {"success": True, "message": f"Model {model_name} loading in background"}

@router.post("/{model_name}/unload")
async def unload_model(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Unload a specific model to free VRAM"""
    
    success = await model_manager.unload_model(model_name)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to unload model {model_name}")
    
    return {"success": True, "message": f"Model {model_name} unloaded successfully"}

@router.post("/switch", response_model=Dict[str, Any])
async def switch_model(
    request: ModelSwitchRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Switch to a different model"""
    
    success = await model_manager.switch_model(request.model_name)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to switch to model {request.model_name}")
    
    # Get updated status
    status = await model_manager.get_status()
    
    return {
        "success": True,
        "active_model": request.model_name,
        "status": status
    }

@router.get("/recommend")
async def recommend_model(
    task_type: str = "chat",
    language: str = "en",
    complexity: str = "medium",
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get model recommendation for a specific task"""
    
    recommended_model = await model_manager.get_best_model_for_task(
        task_type=task_type,
        language=language,
        complexity=complexity
    )
    
    if not recommended_model:
        raise HTTPException(status_code=404, detail="No suitable model found for the task")
    
    models = await model_manager.get_available_models()
    model_info = models[recommended_model]
    
    return {
        "recommended_model": recommended_model,
        "reasoning": {
            "task_type": task_type,
            "language": language,
            "complexity": complexity,
            "model_type": model_info.model_type.value,
            "specialties": model_info.specialties,
            "status": model_info.status.value
        },
        "model_info": {
            "name": recommended_model,
            "provider": model_info.provider,
            "vram_usage_gb": model_info.vram_usage / (1024**3) if model_info.vram_usage else 0,
            "performance_metrics": model_info.performance_metrics
        }
    }

@router.get("/status/system")
async def get_system_status(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get comprehensive system and model status"""
    
    status = await model_manager.get_status()
    return status

@router.post("/optimize")
async def optimize_models(
    target_vram_percentage: float = Field(default=80.0, ge=50.0, le=95.0),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Optimize model loading based on VRAM usage"""
    
    current_status = await model_manager.get_status()
    current_vram_usage = current_status["vram_usage"]["percentage"]
    
    if current_vram_usage <= target_vram_percentage:
        return {
            "success": True,
            "message": "System already optimized",
            "current_vram_usage": current_vram_usage,
            "target": target_vram_percentage
        }
    
    # Find models to unload (least recently used)
    models = await model_manager.get_available_models()
    loaded_models = [
        (name, info) for name, info in models.items() 
        if info.status == ModelStatus.LOADED and name != model_manager.active_model
    ]
    
    # Sort by last used time (oldest first)
    loaded_models.sort(key=lambda x: x[1].last_used)
    
    unloaded_models = []
    for model_name, model_info in loaded_models:
        if current_vram_usage <= target_vram_percentage:
            break
        
        success = await model_manager.unload_model(model_name)
        if success:
            unloaded_models.append(model_name)
            current_vram_usage -= (model_info.vram_usage / model_manager.max_vram) * 100
    
    return {
        "success": True,
        "message": f"Optimized system - unloaded {len(unloaded_models)} models",
        "unloaded_models": unloaded_models,
        "new_vram_usage": current_vram_usage,
        "target": target_vram_percentage
    }

@router.post("/refresh")
async def refresh_model_list(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Refresh the list of available models"""
    
    try:
        # Re-discover models
        await model_manager._discover_models()
        
        models = await model_manager.get_available_models()
        
        return {
            "success": True,
            "message": "Model list refreshed successfully",
            "total_models": len(models),
            "models_by_provider": {
                provider: sum(1 for m in models.values() if m.provider == provider)
                for provider in set(m.provider for m in models.values())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh model list: {str(e)}")

@router.get("/performance/metrics")
async def get_performance_metrics(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get detailed performance metrics for all models"""
    
    status = await model_manager.get_status()
    performance_stats = status.get("performance", {})
    
    # Add additional computed metrics
    for model_name, stats in performance_stats.items():
        if stats.get("request_count", 0) > 0:
            stats["requests_per_hour"] = stats["request_count"] / max(1, status["uptime"] / 3600)
            stats["efficiency_score"] = min(100, max(0, 100 - stats["avg_response_time"]))
    
    return {
        "performance_metrics": performance_stats,
        "system_metrics": {
            "total_requests": sum(stats.get("request_count", 0) for stats in performance_stats.values()),
            "avg_system_response_time": sum(stats.get("avg_response_time", 0) for stats in performance_stats.values()) / max(1, len(performance_stats)),
            "active_models": len([m for m in status.get("loaded_models", [])]),
            "vram_usage": status.get("vram_usage", {})
        }
    }
