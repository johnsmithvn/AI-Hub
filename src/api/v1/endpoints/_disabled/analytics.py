from fastapi import APIRouter

router = APIRouter()

@router.get("/usage")
async def get_usage_analytics():
    """Get system usage analytics"""
    return {"message": "Usage analytics endpoint - to be implemented"}

@router.get("/performance")
async def get_performance_analytics():
    """Get performance analytics"""
    return {"message": "Performance analytics endpoint - to be implemented"}
