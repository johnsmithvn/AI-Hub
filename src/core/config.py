"""
Core configuration settings for AI Backend Hub
Supports environment-based configuration with intelligent defaults
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "AI Backend Hub"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="Environment (development, production)")
    DEBUG: bool = Field(default=False, description="Debug mode")
    HOST: str = Field(default="127.0.0.1", description="Host address")
    PORT: int = Field(default=8000, description="Port number")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/ai_hub",
        description="PostgreSQL database URL"
    )
    DATABASE_ECHO: bool = Field(default=False, description="SQLAlchemy echo mode")
    
    # Redis Configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    REDIS_MAX_CONNECTIONS: int = Field(default=20, description="Redis connection pool size")
    
    # Model Management - Enhanced for External Models
    MODEL_CACHE_DIR: str = Field(default="./models", description="Local model cache directory")
    EXTERNAL_MODELS_DIR: str = Field(
        default="d:/DEV/All-Project/AI-models", 
        description="External models directory (for GGUF models)"
    )
    MAX_VRAM_GB: float = Field(default=14.0, description="Maximum VRAM to use (GB)")
    MODEL_LOAD_TIMEOUT: int = Field(default=300, description="Model loading timeout (seconds)")
    CONCURRENT_MODELS: int = Field(default=2, description="Maximum concurrent loaded models")
    
    # Model Switching & Performance
    ENABLE_SMART_SWITCHING: bool = Field(default=True, description="Enable intelligent model switching")
    MODEL_KEEPALIVE_MINUTES: int = Field(default=30, description="Keep unused models loaded (minutes)")
    AUTO_UNLOAD_THRESHOLD: float = Field(default=0.90, description="Auto-unload when VRAM > threshold")
    PRELOAD_POPULAR_MODELS: bool = Field(default=True, description="Preload frequently used models")
    
    # Quantization & Optimization
    ENABLE_4BIT_QUANTIZATION: bool = Field(default=True, description="Enable 4-bit quantization")
    ENABLE_8BIT_QUANTIZATION: bool = Field(default=True, description="Enable 8-bit quantization")
    DEFAULT_QUANTIZATION: str = Field(default="4bit", description="Default quantization (4bit/8bit/16bit)")
    ENABLE_FLASH_ATTENTION: bool = Field(default=True, description="Enable Flash Attention if available")
    ENABLE_MODEL_PARALLELISM: bool = Field(default=False, description="Enable model parallelism for large models")
    
    # External Model Provider Configuration
    ENABLE_GGUF_MODELS: bool = Field(default=True, description="Enable GGUF model support")
    GGUF_CONTEXT_LENGTH: int = Field(default=4096, description="Default context length for GGUF models")
    GGUF_GPU_LAYERS: int = Field(default=-1, description="GPU layers for GGUF (-1 = auto)")
    
    # Custom Local Model Management (Replaced Ollama)
    LOCAL_MODELS_DIR: str = Field(default="./local_models", description="Local HuggingFace models directory")
    MAX_VRAM_USAGE: float = Field(default=0.85, description="Maximum VRAM usage (85% of total)")
    
    # HuggingFace Configuration
    HF_TOKEN: Optional[str] = Field(default=None, description="HuggingFace API token")
    HF_CACHE_DIR: str = Field(default="./hf_cache", description="HuggingFace cache directory")
    
    # Training Configuration
    TRAINING_DATA_DIR: str = Field(default="./training_data", description="Training data directory")
    TRAINING_OUTPUT_DIR: str = Field(default="./trained_models", description="Training output directory")
    MAX_TRAINING_JOBS: int = Field(default=2, description="Maximum concurrent training jobs")
    DEFAULT_BATCH_SIZE: int = Field(default=4, description="Default training batch size")
    DEFAULT_LEARNING_RATE: float = Field(default=2e-4, description="Default learning rate")
    DEFAULT_LORA_R: int = Field(default=16, description="Default LoRA rank")
    DEFAULT_LORA_ALPHA: int = Field(default=32, description="Default LoRA alpha")
    
    # Multi-Modal Configuration
    UPLOAD_DIR: str = Field(default="./uploads", description="File upload directory")
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024, description="Maximum file size (bytes)")
    SUPPORTED_AUDIO_FORMATS: List[str] = Field(
        default=["mp3", "wav", "m4a", "flac"],
        description="Supported audio formats"
    )
    SUPPORTED_IMAGE_FORMATS: List[str] = Field(
        default=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        description="Supported image formats"
    )
    SUPPORTED_DOCUMENT_FORMATS: List[str] = Field(
        default=["pdf", "docx", "xlsx", "txt", "md"],
        description="Supported document formats"
    )
    
    # API Security
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT secret key"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=1440,  # 24 hours
        description="Access token expiration time"
    )
    API_KEY: Optional[str] = Field(default=None, description="API key for authentication")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    # Background Tasks
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/1",
        description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/1",
        description="Celery result backend URL"
    )
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PORT: int = Field(default=8001, description="Metrics server port")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Performance
    MAX_WORKERS: int = Field(default=4, description="Maximum worker processes")
    WORKER_CONNECTIONS: int = Field(default=1000, description="Worker connections")
    
    # Model-Specific Settings
    DEFAULT_MODEL: str = Field(default="llama2:7b", description="Default model for chat")
    CODE_MODEL: str = Field(default="codellama:7b", description="Default model for code")
    VISION_MODEL: str = Field(default="llava:7b", description="Default model for vision")
    
    # Device Configuration
    DEVICE: str = Field(default="auto", description="Device to use (auto, cpu, cuda)")
    TORCH_DTYPE: str = Field(default="float16", description="PyTorch data type")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist"""
    directories = [
        settings.MODEL_CACHE_DIR,
        settings.HF_CACHE_DIR,
        settings.TRAINING_DATA_DIR,
        settings.TRAINING_OUTPUT_DIR,
        settings.UPLOAD_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize directories
ensure_directories()
