"""
PLACEHOLDER ENDPOINTS - NOT YET IMPLEMENTED
==========================================

These endpoints are planned features but not yet implemented.
They are temporarily disabled to avoid confusion during development.

TODO: Implement these features when ready:
- Files: Upload, list, and manage files
- Vision: Image analysis and generation  
- Audio: Speech-to-text and text-to-speech
- Documents: Document processing and search
- Analytics: Usage metrics and performance monitoring

To enable any of these endpoints:
1. Move the corresponding file from _disabled/ to endpoints/
2. Implement the actual logic
3. Add to main.py router
"""

from fastapi import APIRouter, HTTPException

# Disabled placeholder endpoints
def create_disabled_router(feature_name: str) -> APIRouter:
    """Create a router that returns 'not implemented' responses"""
    router = APIRouter()
    
    @router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def not_implemented(path: str):
        raise HTTPException(
            status_code=501, 
            detail=f"{feature_name} feature is not yet implemented. See _placeholders.py for details."
        )
    
    return router

# Create disabled routers
files_router = create_disabled_router("Files")
vision_router = create_disabled_router("Vision") 
audio_router = create_disabled_router("Audio")
documents_router = create_disabled_router("Documents")
analytics_router = create_disabled_router("Analytics")
