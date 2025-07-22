"""
Custom Model Manager for AI Backend Hub
Full control over model loading, training, and GPU management
No external dependencies - Pure HuggingFace + PyTorch
Optimized for RTX 4060 Ti 16GB VRAM
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import GPUtil
import psutil
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
    GenerationConfig
)
from peft import (
    LoraConfig, 
    PeftModel, 
    PeftConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset, load_dataset
from loguru import logger

from .config import settings
from .redis_client import get_cache

class ModelType(Enum):
    """Model type enumeration"""
    CHAT = "chat"
    CODE = "code"
    VIETNAMESE = "vietnamese"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

class ModelStatus(Enum):
    """Model loading status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    TRAINING = "training"
    ERROR = "error"

@dataclass
class LocalModelInfo:
    """Unified model information schema - compatible with both old and new APIs"""
    name: str
    model_type: ModelType
    local_path: str
    # Core attributes
    size_gb: float = 0.0
    vram_usage: float = 0.0  # Added for API compatibility
    load_time: float = 0.0
    status: ModelStatus = ModelStatus.UNLOADED
    last_used: float = field(default_factory=time.time)
    
    # Extended attributes for flexibility
    provider: str = "local"
    tokenizer_path: Optional[str] = None
    config_path: Optional[str] = None
    quantization: Optional[str] = None  # Added for API compatibility
    specialties: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
class CustomModelManager:
    """
    Unified Model Manager - Single source of truth for all model operations
    Replaces both CustomModelManager and ModelManager to eliminate conflicts
    
    Features:
    - Load models từ local paths (HuggingFace + PyTorch only, no Ollama)
    - Custom training với LoRA/QLoRA support
    - Intelligent VRAM management cho RTX 4060 Ti 16GB
    - Vietnamese language support + multi-language
    - API compatibility với existing endpoints
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Core storage
        self.model_registry: Dict[str, LocalModelInfo] = {}
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.training_jobs: Dict[str, Any] = {}
        self.active_model: Optional[str] = None
        
        # API compatibility properties
        self.models: Dict[str, LocalModelInfo] = self.model_registry  # Alias for API compatibility
        self.tokenizers: Dict[str, Any] = {}  # Separate tokenizer storage for API compatibility
        
        # Performance tracking
        self.request_counts: Dict[str, int] = {}
        self.response_times: Dict[str, List[float]] = {}

        # RTX 4060 Ti 16GB optimization
        self.total_vram = 16 * 1024  # 16GB in MB
        self.max_vram_usage = 0.85   # Use max 85% = ~13.6GB
        self.reserved_vram = 2 * 1024  # Reserve 2GB for system
        self.max_vram = self.total_vram

        # Local model paths
        self.local_models_dir = Path("local_models")
        self.chat_models_dir = self.local_models_dir / "chat_models"
        self.code_models_dir = self.local_models_dir / "code_models"
        self.vietnamese_models_dir = self.local_models_dir / "vietnamese_models"
        self.custom_models_dir = self.local_models_dir / "custom_models"
        
        # Training output
        self.trained_models_dir = Path("trained_models")
        self.training_data_dir = Path("training_data")
        
        # Initialize
        asyncio.create_task(self._scan_local_models())
        
    async def _scan_local_models(self):
        """Scan local directories for available models"""
        logger.info("🔍 Scanning local models...")
        
        model_dirs = [
            (self.chat_models_dir, ModelType.CHAT),
            (self.code_models_dir, ModelType.CODE),
            (self.vietnamese_models_dir, ModelType.VIETNAMESE),
            (self.custom_models_dir, ModelType.CUSTOM)
        ]
        
        for model_dir, model_type in model_dirs:
            if model_dir.exists():
                for model_path in model_dir.iterdir():
                    if model_path.is_dir():
                        await self._register_local_model(model_path, model_type)
                        
        logger.info(f"📊 Found {len(self.model_registry)} local models")
    
    async def _register_local_model(self, model_path: Path, model_type: ModelType):
        """Register a local model"""
        try:
            model_name = model_path.name
            
            # Check if valid model directory
            if not (model_path / "config.json").exists():
                logger.warning(f"⚠️ No config.json found in {model_path}")
                return
            
            # Calculate model size
            size_gb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024**3)
            
            model_info = LocalModelInfo(
                name=model_name,
                model_type=model_type,
                local_path=str(model_path),
                size_gb=size_gb
            )
            
            self.model_registry[model_name] = model_info
            logger.info(f"✅ Registered {model_name} ({size_gb:.1f}GB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to register {model_path}: {e}")
    
    async def get_available_models(self) -> Dict[str, LocalModelInfo]:
        """Return mapping of model name to info"""
        return self.model_registry.copy()

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Get list of models as dictionaries"""
        models = []
        for name, info in self.model_registry.items():
            models.append({
                "name": name,
                "type": info.model_type.value,
                "size_gb": info.size_gb,
                "status": info.status.value,
                "vram_usage": info.vram_usage,
                "local_path": info.local_path,
            })
        return models
    
    async def load_model(
        self, 
        model_name: str, 
        quantization: str = "4bit",
        max_memory: Optional[Dict] = None
    ) -> bool:
        """
        Load model từ local path với custom configuration
        """
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Model {model_name} not found in registry")
            
            if model_name in self.loaded_models:
                logger.info(f"📦 Model {model_name} already loaded")
                return True
            
            model_info = self.model_registry[model_name]
            model_info.status = ModelStatus.LOADING
            
            # Check VRAM availability
            if not await self._check_vram_available(model_info.size_gb):
                await self._free_vram_space()
            
            logger.info(f"🚀 Loading {model_name} from {model_info.local_path}")
            
            # Configure quantization for RTX 4060 Ti
            bnb_config = None
            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif quantization == "8bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16
                )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_info.local_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load model với memory optimization
            model = AutoModelForCausalLM.from_pretrained(
                model_info.local_path,
                quantization_config=bnb_config,
                device_map="auto" if not max_memory else max_memory,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            # Store loaded model and tokenizer separately for API compatibility
            self.loaded_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "info": model_info,
                "load_time": time.time(),
                "generation_config": GenerationConfig.from_pretrained(model_info.local_path)
            }
            
            # Store tokenizer separately for API compatibility
            self.tokenizers[model_name] = tokenizer
            
            # Update status
            model_info.status = ModelStatus.LOADED
            model_info.vram_usage = self._get_model_vram_usage(model)
            model_info.last_used = time.time()
            model_info.load_time = self.loaded_models[model_name]["load_time"]
            model_info.quantization = quantization  # Store quantization info

            logger.info(f"✅ {model_name} loaded successfully ({model_info.vram_usage:.1f}MB VRAM)")
            self.active_model = model_name

            return True
            
        except Exception as e:
            if model_name in self.model_registry:
                self.model_registry[model_name].status = ModelStatus.ERROR
            logger.error(f"❌ Failed to load {model_name}: {e}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload model để free VRAM"""
        try:
            if model_name not in self.loaded_models:
                logger.warning(f"⚠️ Model {model_name} not loaded")
                return False
            
            # Clear model from GPU memory
            del self.loaded_models[model_name]["model"]
            del self.loaded_models[model_name]["tokenizer"]
            del self.loaded_models[model_name]
            
            # Clear tokenizer from separate storage
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            # Force garbage collection
            torch.cuda.empty_cache()
            
            # Update status
            if model_name in self.model_registry:
                self.model_registry[model_name].status = ModelStatus.UNLOADED
                self.model_registry[model_name].vram_usage = 0.0
                
            if self.active_model == model_name:
                self.active_model = None
            
            
            logger.info(f"🗑️ Model {model_name} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unload {model_name}: {e}")
            return False
    
    async def generate_response(
        self, 
        model_name: str, 
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> str:
        """Generate response với custom model"""
        try:
            if model_name not in self.loaded_models:
                success = await self.load_model(model_name)
                if not success:
                    raise ValueError(f"Failed to load model {model_name}")
            
            model_data = self.loaded_models[model_name]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            # Prepare input
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(inputs, **generation_kwargs)
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Update usage stats
            self.model_registry[model_name].last_used = time.time()
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Generation failed for {model_name}: {e}")
            raise

    async def switch_model(self, model_name: str) -> bool:
        """Switch active model, loading if necessary"""
        if self.active_model == model_name:
            return True

        if self.active_model:
            await self.unload_model(self.active_model)

        success = await self.load_model(model_name)
        if success:
            self.active_model = model_name
        return success

    async def get_best_model_for_task(
        self,
        task_type: str,
        language: str = "en",
        complexity: str = "medium",
    ) -> Optional[str]:
        """Intelligent model selection - enhanced with scoring"""
        logger.info(f"🎯 Finding best model for task: {task_type}, language: {language}")
        
        # Map task types to model types
        task_mapping = {
            "chat": ModelType.CHAT,
            "conversation": ModelType.CHAT,
            "code": ModelType.CODE,
            "programming": ModelType.CODE,
            "vietnamese": ModelType.VIETNAMESE,
            "vi": ModelType.VIETNAMESE,
            "multimodal": ModelType.MULTIMODAL,
            "custom": ModelType.CUSTOM
        }
        
        target_type = task_mapping.get(task_type.lower(), ModelType.CHAT)
        
        # Filter models by type and availability
        suitable_models = []
        for name, info in self.model_registry.items():
            if info.model_type == target_type:
                score = 100  # Base score
                
                # Prefer loaded models
                if info.status == ModelStatus.LOADED:
                    score += 50
                
                # Language preference
                if language in info.languages or not info.languages:
                    score += 30
                
                # Performance history
                if name in self.response_times and self.response_times[name]:
                    avg_time = sum(self.response_times[name]) / len(self.response_times[name])
                    if avg_time < 1000:  # < 1 second
                        score += 20
                
                # Recent usage
                time_since_use = time.time() - info.last_used
                if time_since_use < 3600:  # Used within last hour
                    score += 10
                
                suitable_models.append((name, score))
        
        # Fallback to any chat model if no specific match
        if not suitable_models and target_type != ModelType.CHAT:
            for name, info in self.model_registry.items():
                if info.model_type == ModelType.CHAT:
                    suitable_models.append((name, 50))  # Lower score for fallback
        
        if not suitable_models:
            logger.warning(f"No suitable model found for task: {task_type}")
            return None
        
        # Sort by score and return best
        suitable_models.sort(key=lambda x: x[1], reverse=True)
        best_model = suitable_models[0][0]
        
        logger.info(f"✅ Selected model: {best_model} for task: {task_type}")
        return best_model
    
    # API Compatibility Methods
    async def initialize(self):
        """Initialize method for compatibility with ModelManager API"""
        logger.info("🔧 Initializing Unified Model Manager...")
        await self._scan_local_models()
        logger.info("✅ Model Manager initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Model Manager...")
        
        # Unload all models
        for model_name in list(self.loaded_models.keys()):
            await self.unload_model(model_name)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Model Manager cleanup complete")
    async def start_training(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str,
        training_config: Dict[str, Any]
    ) -> str:
        """
        Start custom training với LoRA/QLoRA
        """
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Model {model_name} not found")
            
            # Load base model for training
            base_model_path = self.model_registry[model_name].local_path
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=training_config.get("lora_r", 16),
                lora_alpha=training_config.get("lora_alpha", 32),
                lora_dropout=training_config.get("lora_dropout", 0.1),
                target_modules=training_config.get("target_modules", ["q_proj", "v_proj"])
            )
            
            # Load dataset
            if dataset_path.endswith(".json"):
                dataset = Dataset.from_json(dataset_path)
            elif dataset_path.endswith(".csv"):
                dataset = Dataset.from_csv(dataset_path)
            else:
                raise ValueError("Unsupported dataset format")
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=training_config.get("epochs", 3),
                per_device_train_batch_size=training_config.get("batch_size", 4),
                gradient_accumulation_steps=training_config.get("gradient_accumulation", 1),
                learning_rate=training_config.get("learning_rate", 2e-4),
                fp16=True,  # For RTX 4060 Ti
                save_steps=training_config.get("save_steps", 500),
                logging_steps=training_config.get("logging_steps", 10),
                remove_unused_columns=False,
            )
            
            job_id = f"training_{int(time.time())}"
            
            # Store training job
            self.training_jobs[job_id] = {
                "model_name": model_name,
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "config": training_config,
                "status": "starting",
                "start_time": time.time()
            }
            
            # Start training in background
            asyncio.create_task(self._run_training(job_id, base_model_path, dataset, lora_config, training_args))
            
            logger.info(f"🎓 Started training job {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start training: {e}")
            raise
    
    async def _run_training(self, job_id: str, model_path: str, dataset, lora_config, training_args):
        """Run training process"""
        try:
            self.training_jobs[job_id]["status"] = "training"
            
            # Load base model for training
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Prepare trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            )
            
            # Start training
            trainer.train()
            
            # Save trained model
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            
            self.training_jobs[job_id]["status"] = "completed"
            self.training_jobs[job_id]["end_time"] = time.time()
            
            logger.info(f"✅ Training job {job_id} completed")
            
        except Exception as e:
            self.training_jobs[job_id]["status"] = "failed"
            self.training_jobs[job_id]["error"] = str(e)
            logger.error(f"❌ Training job {job_id} failed: {e}")
    
    async def _check_vram_available(self, required_gb: float) -> bool:
        """Check if enough VRAM available"""
        try:
            gpu = GPUtil.getGPUs()[0]
            available_mb = gpu.memoryFree
            required_mb = required_gb * 1024
            
            return available_mb >= required_mb
        except:
            return True  # Fallback if GPU info unavailable
    
    async def _free_vram_space(self):
        """Free VRAM by unloading least used models"""
        try:
            # Sort models by last used time
            sorted_models = sorted(
                [(name, data["info"].last_used) for name, data in self.loaded_models.items()],
                key=lambda x: x[1]
            )
            
            # Unload oldest model
            if sorted_models:
                oldest_model = sorted_models[0][0]
                await self.unload_model(oldest_model)
                
        except Exception as e:
            logger.error(f"Failed to free VRAM: {e}")
    
    def _get_model_vram_usage(self, model) -> float:
        """Estimate model VRAM usage"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            # Rough estimate: 4 bytes per parameter for float32, 2 for float16
            bytes_per_param = 2  # Using float16/bfloat16
            vram_mb = (total_params * bytes_per_param) / (1024 * 1024)
            return vram_mb
        except:
            return 0.0
    


    def record_request(self, model_name: str, response_time: float):
        """Record request statistics"""
        if model_name not in self.request_counts:
            self.request_counts[model_name] = 0
            self.response_times[model_name] = []

        self.request_counts[model_name] += 1
        self.response_times[model_name].append(response_time)

        if len(self.response_times[model_name]) > 100:
            self.response_times[model_name] = self.response_times[model_name][-100:]

    def _get_performance_stats(self) -> Dict[str, Any]:
        """Aggregate performance statistics"""
        stats = {}
        for name, times in self.response_times.items():
            if times:
                stats[name] = {
                    "avg_response_time": sum(times) / len(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "request_count": self.request_counts.get(name, 0),
                }
        return stats

    async def get_status(self) -> Dict[str, Any]:
        """Get high level system status"""
        system = await self.get_system_status()
        gpu = system.get("gpu", {})

        return {
            "status": "operational",
            "active_model": self.active_model,
            "loaded_models": list(self.loaded_models.keys()),
            "total_models": len(self.model_registry),
            "vram_usage": {
                "current_gb": gpu.get("memory_used", 0) / 1024,
                "max_gb": gpu.get("memory_total", 0) / 1024,
                "percentage": (gpu.get("memory_used", 0) / gpu.get("memory_total", 1)) * 100 if gpu.get("memory_total") else 0,
            },
            "performance": self._get_performance_stats(),
        }
    

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system resource status"""
        try:
            gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            
            return {
                "gpu": {
                    "name": gpu.name if gpu else "No GPU",
                    "memory_total": gpu.memoryTotal if gpu else 0,
                    "memory_used": gpu.memoryUsed if gpu else 0,
                    "memory_free": gpu.memoryFree if gpu else 0,
                    "utilization": gpu.load if gpu else 0
                },
                "cpu": {
                    "usage": psutil.cpu_percent(),
                    "memory": psutil.virtual_memory().percent
                },
                "loaded_models": len(self.loaded_models),
                "total_models": len(self.model_registry),
                "training_jobs": len([j for j in self.training_jobs.values() if j["status"] == "training"])
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {}

# Global model manager instance
_model_manager = None

async def get_model_manager() -> CustomModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = CustomModelManager()
    return _model_manager
