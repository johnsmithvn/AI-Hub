"""
Custom Model Manager for AI Backend Hub
Hybrid model support: HuggingFace/PyTorch + GGUF external models
Full control over model loading, training, and GPU management
Optimized for RTX 4060 Ti 16GB VRAM
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import GPUtil
import psutil
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling, GenerationConfig
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

# Import hybrid extensions
try:
    from .hybrid_extensions import (
        load_model_hybrid, 
        generate_response_hybrid, 
        unload_model_hybrid,
        _load_gguf_model_internal,
        _load_huggingface_model_internal,
        _generate_hf_response
    )
    HYBRID_EXTENSIONS_AVAILABLE = True
except ImportError:
    HYBRID_EXTENSIONS_AVAILABLE = False
    logger.warning("Hybrid extensions not available")

class ModelType(Enum):
    """Model type enumeration"""
    CHAT = "chat"
    CODE = "code"
    VIETNAMESE = "vietnamese"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"
    EMBEDDING = "embedding"

class ModelStatus(Enum):
    """Model loading status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    TRAINING = "training"
    ERROR = "error"
    SWITCHING = "switching"

class ModelFormat(Enum):
    """Model format types"""
    HUGGINGFACE = "huggingface"
    GGUF = "gguf"
    PYTORCH = "pytorch"
    ONNX = "onnx"

@dataclass
class LocalModelInfo:
    """Unified model information schema - supports both HF and GGUF models"""
    name: str
    model_type: ModelType
    local_path: str
    model_format: ModelFormat = ModelFormat.HUGGINGFACE  # New: format type
    
    # Core attributes
    size_gb: float = 0.0
    vram_usage: float = 0.0
    load_time: float = 0.0
    status: ModelStatus = ModelStatus.UNLOADED
    last_used: float = field(default_factory=time.time)
    
    # Extended attributes for flexibility
    provider: str = "local"
    tokenizer_path: Optional[str] = None
    config_path: Optional[str] = None
    quantization: Optional[str] = None
    specialties: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # GGUF-specific attributes
    context_length: Optional[int] = None
    gpu_layers: Optional[int] = None
class CustomModelManager:
    """
    Hybrid Model Manager - Supports both HuggingFace and GGUF models
    
    Features:
    - HuggingFace models từ local paths (PyTorch + transformers)
    - GGUF models từ external directory (llama.cpp)
    - Intelligent model switching và VRAM management  
    - LoRA/QLoRA training support (HuggingFace only)
    - Vietnamese language support + multi-language
    - API compatibility với existing endpoints
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Core storage - HuggingFace models
        self.model_registry: Dict[str, LocalModelInfo] = {}
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.training_jobs: Dict[str, Any] = {}
        self.active_model: Optional[str] = None
        
        # GGUF models integration
        self.gguf_models: Dict[str, LocalModelInfo] = {}
        self.gguf_loader = None
        
        # API compatibility properties
        self.models: Dict[str, LocalModelInfo] = {}  # Combined registry
        self.tokenizers: Dict[str, Any] = {}  # Separate tokenizer storage
        
        # Performance tracking
        self.request_counts: Dict[str, int] = {}
        self.response_times: Dict[str, List[float]] = {}

        # RTX 4060 Ti 16GB optimization
        self.total_vram = settings.MAX_VRAM_GB * 1024  # Convert to MB
        self.max_vram_usage = settings.MAX_VRAM_USAGE
        self.reserved_vram = 2 * 1024  # Reserve 2GB for system
        self.max_vram = self.total_vram

        # Model paths
        self.local_models_dir = Path(settings.LOCAL_MODELS_DIR)
        self.external_models_dir = Path(settings.EXTERNAL_MODELS_DIR)
        
        # Training output
        self.trained_models_dir = Path(settings.TRAINING_OUTPUT_DIR)
        self.training_data_dir = Path(settings.TRAINING_DATA_DIR)
        
        # Initialize both HF and GGUF models
        asyncio.create_task(self._initialize_all_models())
        
        # Add hybrid methods if available
        if HYBRID_EXTENSIONS_AVAILABLE:
            self.load_model_hybrid = lambda *args, **kwargs: load_model_hybrid(self, *args, **kwargs)
            self.generate_response_hybrid = lambda *args, **kwargs: generate_response_hybrid(self, *args, **kwargs)  
            self.unload_model_hybrid = lambda *args, **kwargs: unload_model_hybrid(self, *args, **kwargs)
            self._load_gguf_model_internal = lambda *args, **kwargs: _load_gguf_model_internal(self, *args, **kwargs)
            self._load_huggingface_model_internal = lambda *args, **kwargs: _load_huggingface_model_internal(self, *args, **kwargs)
            self._generate_hf_response = lambda *args, **kwargs: _generate_hf_response(self, *args, **kwargs)
            
            # Add enum references for hybrid methods
            self.ModelStatus = ModelStatus
            self.ModelFormat = ModelFormat
        
    async def _initialize_all_models(self):
        """Initialize both HuggingFace and GGUF models"""
        logger.info("🔧 Initializing Hybrid Model Manager...")
        
        # Initialize GGUF loader first
        if settings.ENABLE_GGUF_MODELS:
            try:
                from .gguf_loader import get_gguf_loader
                self.gguf_loader = await get_gguf_loader()
                await self._integrate_gguf_models()
            except ImportError:
                logger.warning("GGUF support disabled - llama-cpp-python not available")
        
        # Scan local HuggingFace models
        await self._scan_local_models()
        
        # Combine all models into unified registry
        self._update_combined_registry()
        
        logger.info(f"✅ Initialized {len(self.models)} total models (HF: {len(self.model_registry)}, GGUF: {len(self.gguf_models)})")
        
    async def _integrate_gguf_models(self):
        """Integrate GGUF models into the manager"""
        if not self.gguf_loader:
            return
        
        for model_name, gguf_info in self.gguf_loader.gguf_models.items():
            # Convert GGUFModelInfo to LocalModelInfo
            model_info = LocalModelInfo(
                name=model_name,
                model_type=self._determine_model_type(model_name, gguf_info.provider),
                local_path=gguf_info.model_path,
                model_format=ModelFormat.GGUF,
                size_gb=gguf_info.file_size_gb,
                vram_usage=gguf_info.estimated_vram_mb,
                provider=gguf_info.provider,
                quantization=gguf_info.quantization,
                context_length=gguf_info.context_length,
                gpu_layers=gguf_info.gpu_layers,
                specialties=self._determine_specialties(model_name),
                languages=self._determine_languages(model_name, gguf_info.provider),
                config={
                    "model_format": "gguf",
                    "context_length": gguf_info.context_length,
                    "gpu_layers": gguf_info.gpu_layers,
                    "original_path": gguf_info.model_path
                }
            )
            
            self.gguf_models[model_name] = model_info
    
    def _determine_model_type(self, model_name: str, provider: str) -> ModelType:
        """Determine model type from name and provider"""
        name_lower = model_name.lower()
        
        if provider == "vilm" or "vietnam" in name_lower:
            return ModelType.VIETNAMESE
        elif any(x in name_lower for x in ["code", "coder", "programming", "codellama"]):
            return ModelType.CODE
        elif any(x in name_lower for x in ["vision", "llava", "multimodal"]):
            return ModelType.MULTIMODAL
        elif any(x in name_lower for x in ["embed", "sentence"]):
            return ModelType.EMBEDDING
        else:
            return ModelType.CHAT
    
    def _update_combined_registry(self):
        """Update combined models registry for API compatibility"""
        self.models.clear()
        self.models.update(self.model_registry)  # HuggingFace models
        self.models.update(self.gguf_models)     # GGUF models
    
    async def _scan_local_models(self):
        """Scan local directories for HuggingFace models"""
        logger.info("🔍 Scanning local HuggingFace models...")
        
        if not self.local_models_dir.exists():
            logger.info(f"Creating local models directory: {self.local_models_dir}")
            self.local_models_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Scan subdirectories by type
        model_dirs = [
            (self.local_models_dir / "chat_models", ModelType.CHAT),
            (self.local_models_dir / "code_models", ModelType.CODE),
            (self.local_models_dir / "vietnamese_models", ModelType.VIETNAMESE),
            (self.local_models_dir / "custom_models", ModelType.CUSTOM)
        ]
        
        for model_dir, model_type in model_dirs:
            if model_dir.exists():
                for model_path in model_dir.iterdir():
                    if model_path.is_dir():
                        await self._register_local_model(model_path, model_type)
        
        # Also scan root directory for any models
        for model_path in self.local_models_dir.iterdir():
            if model_path.is_dir() and (model_path / "config.json").exists():
                await self._register_local_model(model_path, ModelType.CUSTOM)
                        
        logger.info(f"📊 Found {len(self.model_registry)} local HuggingFace models")
    async def _register_local_model(self, model_path: Path, model_type: ModelType):
        """Register a local HuggingFace model"""
        try:
            model_name = model_path.name
            
            # Check if valid HuggingFace model directory
            if not (model_path / "config.json").exists():
                logger.warning(f"⚠️ No config.json found in {model_path}")
                return
            
            # Calculate model size
            size_gb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024**3)
            
            model_info = LocalModelInfo(
                name=model_name,
                model_type=model_type,
                local_path=str(model_path),
                model_format=ModelFormat.HUGGINGFACE,
                size_gb=size_gb,
                provider="local",
                specialties=self._determine_specialties(model_name),
                languages=self._determine_languages(model_name, "local")
            )
            
            self.model_registry[model_name] = model_info
            logger.info(f"✅ Registered HF model: {model_name} ({size_gb:.1f}GB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to register {model_path}: {e}")

    def _determine_specialties(self, model_name: str) -> List[str]:
        """Determine model specialties based on name"""
        specialties = []
        name_lower = model_name.lower()
        
        if any(x in name_lower for x in ["chat", "instruct", "conversation"]):
            specialties.append("conversation")
        if any(x in name_lower for x in ["code", "coder", "programming"]):
            specialties.append("coding")
        if any(x in name_lower for x in ["dpo", "roleplay", "storytelling"]):
            specialties.append("creative_writing")
        if "vl" in name_lower or "vision" in name_lower:
            specialties.append("multimodal")
        if any(x in name_lower for x in ["vietnam", "vi", "vietnamese"]):
            specialties.append("vietnamese")
        
        return specialties or ["general"]
    
    def _determine_languages(self, model_name: str, provider: str) -> List[str]:
        """Determine supported languages"""
        languages = ["en"]  # Default English
        name_lower = model_name.lower()
        
        if provider == "vilm" or any(x in name_lower for x in ["vietnam", "vi", "vietnamese"]):
            languages.append("vi")
        if "qwen" in name_lower:
            languages.extend(["zh", "en"])
        if "multilingual" in name_lower:
            languages.extend(["es", "fr", "de", "ja", "ko"])
            
        return languages
    
    async def get_available_models(self) -> Dict[str, LocalModelInfo]:
        """Return combined mapping of all models (HF + GGUF)"""
        return self.models.copy()

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """Get list of all models as dictionaries"""
        models = []
        for name, info in self.models.items():
            models.append({
                "name": name,
                "type": info.model_type.value,
                "format": info.model_format.value,
                "size_gb": info.size_gb,
                "status": info.status.value,
                "vram_usage": info.vram_usage,
                "local_path": info.local_path,
                "provider": info.provider,
                "quantization": info.quantization,
                "specialties": info.specialties,
                "languages": info.languages
            })
        return models
    
    async def load_model(
        self, 
        model_name: str, 
        quantization: str = "4bit",
        max_memory: Optional[Dict] = None
    ) -> bool:
        """
        Load model using hybrid system (HuggingFace or GGUF)
        """
        try:
            # Use hybrid loading if available
            if HYBRID_EXTENSIONS_AVAILABLE and hasattr(self, 'load_model_hybrid'):
                return await self.load_model_hybrid(model_name, quantization, max_memory)
            
            # Fallback to legacy HuggingFace loading
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
        """Unload model using hybrid system (HuggingFace or GGUF)"""
        try:
            # Use hybrid unloading if available
            if HYBRID_EXTENSIONS_AVAILABLE and hasattr(self, 'unload_model_hybrid'):
                return await self.unload_model_hybrid(model_name)
            
            # Fallback to legacy HuggingFace unloading
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
        """Generate response using hybrid system (HuggingFace or GGUF)"""
        try:
            # Use hybrid generation if available
            if HYBRID_EXTENSIONS_AVAILABLE and hasattr(self, 'generate_response_hybrid'):
                return await self.generate_response_hybrid(
                    model_name, prompt, max_tokens, temperature, top_p, stream
                )
            
            # Fallback to legacy HuggingFace generation
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
