"""
Health check and system status endpoints
"""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from typing import Dict, Any
import psutil
import time

from ...core.model_manager import ModelManager
from ...core.database import check_db_health
from ...core.redis_client import check_redis_health

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float
    services: Dict[str, Any]
    system: Dict[str, Any]

async def get_model_manager(request: Request) -> ModelManager:
    """Dependency to get model manager"""
    return request.app.state.model_manager

@router.get("/", response_model=HealthResponse)
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Comprehensive health check"""
    
    # Check core services
    db_healthy = await check_db_health()
    redis_healthy = await check_redis_health()
    
    # Get model manager status
    model_status = await model_manager.get_status()
    
    # System metrics
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total": psutil.disk_usage('/').total,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent
        }
    }
    
    # Overall status
    overall_status = "healthy"
    if not db_healthy or not redis_healthy:
        overall_status = "unhealthy"
    elif model_status.get("status") != "operational":
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        version="1.0.0",
        uptime=time.time() - psutil.Process().create_time(),
        services={
            "database": {"status": "healthy" if db_healthy else "unhealthy"},
            "redis": {"status": "healthy" if redis_healthy else "unhealthy"},
            "models": model_status
        },
        system=system_info
    )

@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    return {"status": "ready"}

@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive"}
