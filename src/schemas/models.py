"""
Pydantic schemas for model management API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ModelInfo(BaseModel):
    """Model information schema"""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Model display name")
    description: Optional[str] = Field(None, description="Model description")
    model_type: str = Field(..., description="Model type (text, image, audio, multimodal)")
    model_family: Optional[str] = Field(None, description="Model family (llama, gpt, etc.)")
    version: Optional[str] = Field(None, description="Model version")
    size_gb: Optional[float] = Field(None, description="Model size in GB")
    context_length: Optional[int] = Field(None, description="Context length in tokens")
    supports_streaming: bool = Field(False, description="Supports streaming responses")
    supports_functions: bool = Field(False, description="Supports function calling")
    supports_vision: bool = Field(False, description="Supports vision/image input")
    supports_audio: bool = Field(False, description="Supports audio input")
    is_local: bool = Field(True, description="Is locally hosted")
    is_active: bool = Field(True, description="Is currently active")
    avg_tokens_per_second: Optional[float] = Field(None, description="Average generation speed")
    vram_usage_gb: Optional[float] = Field(None, description="VRAM usage in GB")

class ModelListResponse(BaseModel):
    """Response schema for listing models"""
    models: List[ModelInfo]
    total: int
    active_count: int
    
class ModelLoadRequest(BaseModel):
    """Request schema for loading a model"""
    model_id: str = Field(..., description="Model ID to load")
    force_reload: bool = Field(False, description="Force reload if already loaded")
    
class ModelLoadResponse(BaseModel):
    """Response schema for model loading"""
    success: bool
    message: str
    model_id: str
    load_time: Optional[float] = Field(None, description="Load time in seconds")
    memory_usage: Optional[float] = Field(None, description="Memory usage in GB")

class ModelUnloadRequest(BaseModel):
    """Request schema for unloading a model"""
    model_id: str = Field(..., description="Model ID to unload")
    
class ModelUnloadResponse(BaseModel):
    """Response schema for model unloading"""
    success: bool
    message: str
    model_id: str
    freed_memory: Optional[float] = Field(None, description="Freed memory in GB")

class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics"""
    model_id: str
    avg_response_time: float = Field(..., description="Average response time in ms")
    tokens_per_second: float = Field(..., description="Tokens generated per second")
    total_requests: int = Field(..., description="Total requests processed")
    error_rate: float = Field(..., description="Error rate percentage")
    memory_usage: float = Field(..., description="Current memory usage in GB")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")

class ModelStatusResponse(BaseModel):
    """Response schema for model status"""
    loaded_models: List[str]
    available_models: List[str]
    memory_usage: Dict[str, float]
    performance_metrics: List[ModelPerformanceMetrics]
    system_info: Dict[str, Any]
