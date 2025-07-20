from fastapi import APIRouter

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio():
    """Transcribe audio to text"""
    return {"message": "Audio transcription endpoint - to be implemented"}

@router.post("/synthesize")
async def synthesize_speech():
    """Synthesize speech from text"""
    return {"message": "Speech synthesis endpoint - to be implemented"}
