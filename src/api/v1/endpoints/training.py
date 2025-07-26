"""
Training pipeline endpoints for LoRA/QLoRA model training
"""

from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from loguru import logger

from src.core.custom_model_manager import CustomModelManager

router = APIRouter()

async def get_model_manager(request: Request) -> CustomModelManager:
    """Dependency to get custom model manager"""
    return request.app.state.model_manager

class TrainingJobRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    base_model: str = Field(..., description="Base model to fine-tune")
    
    # Training configuration
    training_type: str = Field(default="lora", description="Training type: lora, qlora, full")
    epochs: int = Field(default=3, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=32)
    learning_rate: float = Field(default=5e-4, ge=1e-6, le=1e-2)
    
    # LoRA specific parameters
    lora_r: int = Field(default=16, ge=1, le=256, description="LoRA attention dimension")
    lora_alpha: int = Field(default=32, ge=1, le=512, description="LoRA scaling parameter")
    lora_dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    
    # Data configuration
    dataset_format: str = Field(default="alpaca", description="Dataset format: alpaca, sharegpt, conversation")
    validation_split: float = Field(default=0.1, ge=0.0, le=0.3)
    max_seq_length: int = Field(default=2048, ge=256, le=8192)
    
    # Advanced settings
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32)
    warmup_steps: int = Field(default=100, ge=0, le=1000)
    save_steps: int = Field(default=500, ge=100, le=2000)
    logging_steps: int = Field(default=50, ge=10, le=500)
    
    @validator('base_model')
    def validate_base_model(cls, v):
        # Add validation for supported base models
        supported_models = ['llama2:7b', 'mistral:7b', 'codellama:7b']
        # This would be dynamic based on available models
        return v

class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_duration_hours: Optional[float] = None

class TrainingJobStatus(BaseModel):
    job_id: str
    name: str
    status: str
    progress_percentage: float
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    estimated_time_remaining_hours: Optional[float] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks,
    model_manager: CustomModelManager = Depends(get_model_manager)
):
    """Create a new training job using CustomModelManager"""
    
    try:
        # Validate base model exists
        available_models = await model_manager.get_available_models()
        if request.base_model not in available_models:
            raise HTTPException(
                status_code=404, 
                detail=f"Base model '{request.base_model}' not found. Available models: {list(available_models.keys())}"
            )
        
        # Prepare training configuration
        training_config = {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "lora_r": request.lora_r,
            "lora_alpha": request.lora_alpha,
            "lora_dropout": request.lora_dropout,
            "gradient_accumulation": request.gradient_accumulation_steps,
            "save_steps": request.save_steps,
            "logging_steps": request.logging_steps,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]  # Common LoRA targets
        }
        
        # Set up paths
        dataset_path = f"training_data/{request.name}_dataset.json"  # This should be uploaded separately
        output_dir = f"trained_models/{request.name}_{int(time.time())}"
        
        # Check if dataset exists (placeholder - in real implementation, dataset would be uploaded)
        if not os.path.exists(dataset_path):
            # Create a dummy dataset file for demonstration
            os.makedirs("training_data", exist_ok=True)
            dummy_dataset = [
                {"instruction": "Hello", "input": "", "output": "Hi there! How can I help you?"},
                {"instruction": "What is AI?", "input": "", "output": "AI stands for Artificial Intelligence..."}
            ]
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(dummy_dataset, f, ensure_ascii=False, indent=2)
        
        # Start training using CustomModelManager
        job_id = await model_manager.start_training(
            model_name=request.base_model,
            dataset_path=dataset_path,
            output_dir=output_dir,
            training_config=training_config
        )
        
        # Estimate training duration
        estimated_duration = estimate_training_duration(
            dataset_size=1000,  # This would be calculated from actual dataset
            epochs=request.epochs,
            batch_size=request.batch_size,
            model_size="7b"
        )
        
        logger.info(f"Training job {job_id} created for model {request.base_model}")
        
        return TrainingJobResponse(
            job_id=job_id,
            status="pending",
            message=f"Training job '{request.name}' created and started successfully",
            estimated_duration_hours=estimated_duration
        )
        
    except Exception as e:
        logger.error(f"Failed to create training job: {e}")
        raise HTTPException(status_code=500, detail=f"Training job creation failed: {str(e)}")

from fastapi import Query

@router.get("/jobs", response_model=List[TrainingJobStatus])
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0)
):
    """List training jobs with optional filtering"""
    
    jobs_dir = Path("training_jobs")
    if not jobs_dir.exists():
        return []
    
    jobs = []
    for job_file in jobs_dir.glob("*.json"):
        try:
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            
            # Filter by status if provided
            if status and job_data.get("status") != status:
                continue
            
            job_status = TrainingJobStatus(
                job_id=job_data["job_id"],
                name=job_data["name"],
                status=job_data["status"],
                progress_percentage=job_data.get("progress_percentage", 0.0),
                current_epoch=job_data.get("current_epoch", 0),
                total_epochs=job_data["training_config"]["epochs"],
                current_step=job_data.get("current_step", 0),
                total_steps=job_data.get("total_steps", 0),
                training_loss=job_data.get("training_loss"),
                validation_loss=job_data.get("validation_loss"),
                learning_rate=job_data.get("learning_rate"),
                estimated_time_remaining_hours=job_data.get("estimated_time_remaining_hours"),
                created_at=job_data["created_at"],
                started_at=job_data.get("started_at"),
                completed_at=job_data.get("completed_at")
            )
            jobs.append(job_status)
            
        except Exception as e:
            continue  # Skip corrupted job files
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply pagination
    return jobs[offset:offset + limit]

@router.get("/jobs/{job_id}", response_model=TrainingJobStatus)
async def get_training_job(
    job_id: str,
    model_manager: CustomModelManager = Depends(get_model_manager)
):
    """Get details of a specific training job from CustomModelManager"""
    
    # Check if job exists in model manager
    if job_id not in model_manager.training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    try:
        job_data = model_manager.training_jobs[job_id]
        
        # Calculate progress
        progress_percentage = 0.0
        if job_data["status"] == "training":
            progress_percentage = 50.0  # Placeholder - would be real progress
        elif job_data["status"] == "completed":
            progress_percentage = 100.0
        elif job_data["status"] == "failed":
            progress_percentage = 0.0
        
        return TrainingJobStatus(
            job_id=job_id,
            name=job_data.get("model_name", f"Training {job_id}"),
            status=job_data["status"],
            progress_percentage=progress_percentage,
            current_epoch=job_data.get("current_epoch", 0),
            total_epochs=job_data["config"].get("epochs", 3),
            current_step=job_data.get("current_step", 0),
            total_steps=job_data.get("total_steps", 1000),
            training_loss=job_data.get("training_loss"),
            validation_loss=job_data.get("validation_loss"),
            learning_rate=job_data["config"].get("learning_rate"),
            estimated_time_remaining_hours=job_data.get("estimated_time_remaining"),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(job_data["start_time"])),
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(job_data["start_time"])) if job_data.get("start_time") else None,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(job_data["end_time"])) if job_data.get("end_time") else None
        )
        
    except Exception as e:
        logger.error(f"Error getting training job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading job data: {str(e)}")

@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str):
    """Cancel a running training job"""
    
    job_file = Path(f"training_jobs/{job_id}.json")
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Training job not found")
    
    try:
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        
        if job_data["status"] in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel job with status: {job_data['status']}"
            )
        
        # Update job status
        job_data["status"] = "cancelled"
        job_data["completed_at"] = "2025-01-20T00:00:00Z"
        
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        # In production, this would signal the training process to stop
        
        return {"success": True, "message": f"Training job {job_id} cancelled"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {str(e)}")

@router.post("/jobs/{job_id}/restart")
async def restart_training_job(job_id: str, background_tasks: BackgroundTasks):
    """Restart a failed or cancelled training job"""
    
    job_file = Path(f"training_jobs/{job_id}.json")
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Training job not found")
    
    try:
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        
        if job_data["status"] not in ["failed", "cancelled"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot restart job with status: {job_data['status']}"
            )
        
        # Reset job status
        job_data["status"] = "pending"
        job_data["progress_percentage"] = 0.0
        job_data["current_epoch"] = 0
        job_data["current_step"] = 0
        job_data["started_at"] = None
        job_data["completed_at"] = None
        
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        # Restart training in background
        training_config = TrainingJobRequest(**job_data["training_config"])
        background_tasks.add_task(start_training_job, job_id, training_config)
        
        return {"success": True, "message": f"Training job {job_id} restarted"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restarting job: {str(e)}")

@router.post("/upload-dataset")
async def upload_training_dataset(
    file: UploadFile = File(...),
    dataset_name: str = "",
    format: str = "alpaca"
):
    """Upload training dataset"""
    
    if not file.filename.endswith(('.json', '.jsonl', '.txt')):
        raise HTTPException(
            status_code=400,
            detail="Only JSON, JSONL, and TXT files are supported"
        )
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = Path(f"training_data/{filename}")
    file_path.parent.mkdir(exist_ok=True)
    
    # Save uploaded file
    try:
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Validate and analyze dataset
        dataset_info = analyze_dataset(file_path, format)
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "dataset_info": dataset_info
        }
        
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")

@router.get("/datasets")
async def list_datasets():
    """List available training datasets"""
    
    datasets_dir = Path("training_data")
    if not datasets_dir.exists():
        return []
    
    datasets = []
    for file_path in datasets_dir.glob("*"):
        if file_path.is_file() and file_path.suffix in ['.json', '.jsonl', '.txt']:
            try:
                stat = file_path.stat()
                datasets.append({
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "created_at": stat.st_ctime,
                    "modified_at": stat.st_mtime
                })
            except Exception:
                continue
    
    return datasets

# Helper functions
def estimate_training_duration(dataset_size: int, epochs: int, batch_size: int, model_size: str) -> float:
    """Estimate training duration in hours"""
    
    # Base time per sample (rough estimates)
    time_per_sample = {
        "7b": 0.1,    # seconds per sample for 7B model
        "13b": 0.2,   # seconds per sample for 13B model
        "70b": 1.0    # seconds per sample for 70B model
    }
    
    base_time = time_per_sample.get(model_size, 0.1)
    
    # Calculate total training time
    total_samples = dataset_size * epochs
    samples_per_batch = batch_size
    total_batches = total_samples / samples_per_batch
    
    # Account for gradient accumulation and other overheads
    overhead_factor = 1.5
    
    total_seconds = total_batches * base_time * overhead_factor
    total_hours = total_seconds / 3600
    
    return round(total_hours, 2)

def analyze_dataset(file_path: Path, format: str) -> Dict[str, Any]:
    """Analyze uploaded dataset and return statistics"""
    
    try:
        if format == "alpaca":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                sample_count = len(data)
                avg_input_length = sum(len(item.get('input', '')) for item in data) / max(1, sample_count)
                avg_output_length = sum(len(item.get('output', '')) for item in data) / max(1, sample_count)
                
                return {
                    "format": format,
                    "sample_count": sample_count,
                    "avg_input_length": round(avg_input_length),
                    "avg_output_length": round(avg_output_length),
                    "estimated_tokens": round((avg_input_length + avg_output_length) * sample_count / 4)
                }
        
        # Add support for other formats...
        
        return {
            "format": format,
            "sample_count": 0,
            "avg_input_length": 0,
            "avg_output_length": 0,
            "estimated_tokens": 0
        }
        
    except Exception as e:
        return {
            "error": f"Failed to analyze dataset: {str(e)}",
            "format": format
        }

