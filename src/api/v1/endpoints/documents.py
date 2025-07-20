from fastapi import APIRouter

router = APIRouter()

@router.post("/process")
async def process_document():
    """Process document and extract information"""
    return {"message": "Document processing endpoint - to be implemented"}

@router.get("/search")
async def search_documents():
    """Search through processed documents"""
    return {"message": "Document search endpoint - to be implemented"}
