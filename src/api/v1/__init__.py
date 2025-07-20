"""
API v1 router - Main API endpoints
"""

from fastapi import APIRouter
from .endpoints import (
    chat, models, training, files, health, 
    vision, audio, documents, analytics
)

router = APIRouter()

# Include all endpoint routers
router.include_router(health.router, prefix="/health", tags=["Health"])
router.include_router(chat.router, prefix="/chat", tags=["Chat"])
router.include_router(models.router, prefix="/models", tags=["Models"])
router.include_router(training.router, prefix="/training", tags=["Training"])
router.include_router(files.router, prefix="/files", tags=["Files"])
router.include_router(vision.router, prefix="/vision", tags=["Vision"])
router.include_router(audio.router, prefix="/audio", tags=["Audio"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
