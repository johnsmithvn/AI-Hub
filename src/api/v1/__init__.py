"""
API v1 router - Clean, production-ready endpoints only
"""

from fastapi import APIRouter
from .endpoints import chat, models, training, health
from .endpoints._placeholders import (
    files_router, vision_router, audio_router, 
    documents_router, analytics_router
)

router = APIRouter()

# Active endpoints (fully implemented)
router.include_router(health.router, prefix="/health", tags=["Health"])
router.include_router(chat.router, prefix="/chat", tags=["Chat"])  
router.include_router(models.router, prefix="/models", tags=["Models"])
router.include_router(training.router, prefix="/training", tags=["Training"])

# Placeholder endpoints (return 501 Not Implemented)
router.include_router(files_router, prefix="/files", tags=["Files (Disabled)"])
router.include_router(vision_router, prefix="/vision", tags=["Vision (Disabled)"])
router.include_router(audio_router, prefix="/audio", tags=["Audio (Disabled)"])
router.include_router(documents_router, prefix="/documents", tags=["Documents (Disabled)"])
router.include_router(analytics_router, prefix="/analytics", tags=["Analytics (Disabled)"])
