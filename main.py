"""
AI Backend Hub - Comprehensive Multi-Modal AI System

Main FastAPI application entry point with advanced model management,
training pipeline, and multi-modal processing capabilities.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import settings
from src.core.database import init_db
from src.core.redis_client import init_redis
from src.core.custom_model_manager import get_model_manager
from src.api.v1 import router as api_v1_router
from src.core.middleware import setup_middleware
from src.core.logging_config import setup_logging

# Global instances
model_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting AI Backend Hub...")
    
    # Initialize core services
    await init_db()
    await init_redis()
    
    # Initialize custom model manager
    model_manager = await get_model_manager()
    
    # Store in app state
    app.state.model_manager = model_manager
    
    logger.info("✅ AI Backend Hub started successfully!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI Backend Hub...")
    logger.info("✅ AI Backend Hub shutdown complete!")

# Create FastAPI app
app = FastAPI(
    title="AI Backend Hub",
    description="Comprehensive Multi-Modal AI System with Dynamic Model Management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Include API routes
app.include_router(api_v1_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "message": "AI Backend Hub - Multi-Modal AI System",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check model manager
        if hasattr(app.state, 'model_manager'):
            model_status = await app.state.model_manager.get_status()
        else:
            model_status = {"status": "not_initialized"}
            
        return {
            "status": "healthy",
            "models": model_status,
            "timestamp": "2025-01-20T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,  # Single worker for model management
        log_level="info"
    )
