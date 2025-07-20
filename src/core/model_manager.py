"""
Advanced Model Manager for AI Backend Hub
Handles dynamic model loading, switching, and intelligent resource management
"""

import asyncio
import torch
import psutil
import GPUtil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
from loguru import logger

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig, pipeline
)
from peft import PeftModel, PeftConfig
import ollama

from .config import settings
from .redis_client import get_cache, cache_model_metadata, get_cached_model_metadata

class ModelType(Enum):
    """Model type enumeration"""
    CHAT = "chat"
    CODE = "code"
    VISION = "vision"
    EMBEDDING = "embedding"
    CUSTOM = "custom"

class ModelStatus(Enum):
    """Model status enumeration"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    SWITCHING = "switching"

@dataclass
class ModelInfo:
    """Model information and metadata"""
    name: str
    model_type: ModelType
    provider: str  # "ollama", "huggingface", "local"
    path: str
    vram_usage: float = 0.0
    load_time: float = 0.0
    status: ModelStatus = ModelStatus.UNLOADED
    last_used: float = field(default_factory=time.time)
    config: Dict[str, Any] = field(default_factory=dict)
    quantization: Optional[str] = None
    specialties: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class ModelManager:
    """Advanced model management with intelligent switching and resource optimization"""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.active_model: Optional[str] = None
        self.loading_lock = asyncio.Lock()
        self.ollama_client = None
        
        # Resource tracking
        self.max_vram = settings.MAX_VRAM_GB * 1024 * 1024 * 1024  # Convert to bytes
        self.current_vram_usage = 0
        
        # Performance tracking
        self.request_counts: Dict[str, int] = {}
        self.response_times: Dict[str, List[float]] = {}
    
    async def initialize(self):
        """Initialize model manager and discover available models"""
        logger.info("🔧 Initializing Model Manager...")
        
        try:
            # Initialize Ollama client
            self.ollama_client = ollama.AsyncClient(host=settings.OLLAMA_HOST)
            
            # Discover available models
            await self._discover_models()
            
            # Load default model
            if settings.DEFAULT_MODEL:
                await self.load_model(settings.DEFAULT_MODEL)
            
            logger.info("✅ Model Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Model Manager initialization failed: {e}")
            raise
    
    async def _discover_models(self):
        """Discover available models from various sources"""
        
        # Discover Ollama models
        try:
            ollama_models = await self.ollama_client.list()
            for model in ollama_models.get('models', []):
                name = model['name']
                size = model.get('size', 0)
                
                # Determine model type based on name
                model_type = self._infer_model_type(name)
                
                model_info = ModelInfo(
                    name=name,
                    model_type=model_type,
                    provider="ollama",
                    path=name,
                    config={"size": size}
                )
                self.models[name] = model_info
                
            logger.info(f"📦 Discovered {len(ollama_models.get('models', []))} Ollama models")
            
        except Exception as e:
            logger.warning(f"Could not discover Ollama models: {e}")
        
        # Discover local HuggingFace models
        hf_cache_dir = Path(settings.HF_CACHE_DIR)
        if hf_cache_dir.exists():
            # Implementation for local model discovery
            pass
        
        # Discover custom trained models
        training_output_dir = Path(settings.TRAINING_OUTPUT_DIR)
        if training_output_dir.exists():
            for model_dir in training_output_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
                    model_info = ModelInfo(
                        name=model_dir.name,
                        model_type=ModelType.CUSTOM,
                        provider="local",
                        path=str(model_dir)
                    )
                    self.models[model_dir.name] = model_info
    
    def _infer_model_type(self, model_name: str) -> ModelType:
        """Infer model type from name"""
        name_lower = model_name.lower()
        
        if any(keyword in name_lower for keyword in ['code', 'programming', 'codellama']):
            return ModelType.CODE
        elif any(keyword in name_lower for keyword in ['vision', 'llava', 'multimodal']):
            return ModelType.VISION
        elif any(keyword in name_lower for keyword in ['embed', 'sentence']):
            return ModelType.EMBEDDING
        else:
            return ModelType.CHAT
    
    async def get_available_models(self) -> Dict[str, ModelInfo]:
        """Get list of available models"""
        return self.models.copy()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        gpu_info = self._get_gpu_info()
        
        return {
            "status": "operational",
            "active_model": self.active_model,
            "loaded_models": list(self.loaded_models.keys()),
            "total_models": len(self.models),
            "vram_usage": {
                "current_gb": self.current_vram_usage / (1024**3),
                "max_gb": self.max_vram / (1024**3),
                "percentage": (self.current_vram_usage / self.max_vram) * 100
            },
            "gpu_info": gpu_info,
            "performance": self._get_performance_stats()
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "utilization": gpu.load * 100,
                    "temperature": gpu.temperature
                }
        except Exception:
            pass
        
        return {"available": False}
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for model_name, times in self.response_times.items():
            if times:
                stats[model_name] = {
                    "avg_response_time": sum(times) / len(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "request_count": self.request_counts.get(model_name, 0)
                }
        return stats
    
    async def load_model(self, model_name: str, force: bool = False) -> bool:
        """Load a specific model with intelligent resource management"""
        async with self.loading_lock:
            if model_name in self.loaded_models and not force:
                self.active_model = model_name
                self.models[model_name].last_used = time.time()
                logger.info(f"✅ Model {model_name} already loaded")
                return True
            
            if model_name not in self.models:
                logger.error(f"❌ Model {model_name} not found")
                return False
            
            model_info = self.models[model_name]
            model_info.status = ModelStatus.LOADING
            
            start_time = time.time()
            
            try:
                # Check VRAM availability
                estimated_vram = await self._estimate_model_vram(model_name)
                if not await self._ensure_vram_available(estimated_vram):
                    logger.error(f"❌ Insufficient VRAM for model {model_name}")
                    model_info.status = ModelStatus.ERROR
                    return False
                
                # Load model based on provider
                if model_info.provider == "ollama":
                    success = await self._load_ollama_model(model_name, model_info)
                elif model_info.provider == "huggingface":
                    success = await self._load_huggingface_model(model_name, model_info)
                elif model_info.provider == "local":
                    success = await self._load_local_model(model_name, model_info)
                else:
                    logger.error(f"❌ Unknown provider: {model_info.provider}")
                    return False
                
                if success:
                    load_time = time.time() - start_time
                    model_info.load_time = load_time
                    model_info.status = ModelStatus.LOADED
                    model_info.last_used = time.time()
                    self.active_model = model_name
                    
                    # Cache model metadata
                    await cache_model_metadata(model_name, {
                        "load_time": load_time,
                        "vram_usage": model_info.vram_usage,
                        "status": model_info.status.value
                    })
                    
                    logger.info(f"✅ Model {model_name} loaded successfully in {load_time:.2f}s")
                    return True
                else:
                    model_info.status = ModelStatus.ERROR
                    return False
                
            except Exception as e:
                logger.error(f"❌ Failed to load model {model_name}: {e}")
                model_info.status = ModelStatus.ERROR
                return False
    
    async def _load_ollama_model(self, model_name: str, model_info: ModelInfo) -> bool:
        """Load Ollama model"""
        try:
            # Ollama models are loaded on-demand during inference
            # We just verify the model exists
            models = await self.ollama_client.list()
            model_exists = any(m['name'] == model_name for m in models.get('models', []))
            
            if model_exists:
                self.loaded_models[model_name] = self.ollama_client
                return True
            else:
                logger.error(f"Ollama model {model_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load Ollama model {model_name}: {e}")
            return False
    
    async def _load_huggingface_model(self, model_name: str, model_info: ModelInfo) -> bool:
        """Load HuggingFace model"""
        try:
            # Configure quantization for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_info.path,
                cache_dir=settings.HF_CACHE_DIR,
                trust_remote_code=True
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_info.path,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=settings.HF_CACHE_DIR,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            self.loaded_models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # Estimate VRAM usage
            model_info.vram_usage = self._estimate_model_memory(model)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {model_name}: {e}")
            return False
    
    async def _load_local_model(self, model_name: str, model_info: ModelInfo) -> bool:
        """Load local/custom trained model"""
        try:
            # Load base model and adapter
            config = PeftConfig.from_pretrained(model_info.path)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Load adapter
            model = PeftModel.from_pretrained(base_model, model_info.path)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            
            self.loaded_models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load local model {model_name}: {e}")
            return False
    
    async def _estimate_model_vram(self, model_name: str) -> float:
        """Estimate VRAM requirements for a model"""
        # Check cached metadata
        cached_metadata = await get_cached_model_metadata(model_name)
        if cached_metadata and 'vram_usage' in cached_metadata:
            return cached_metadata['vram_usage']
        
        model_info = self.models[model_name]
        
        # Rough estimation based on model size and type
        if model_info.provider == "ollama":
            size = model_info.config.get('size', 0)
            # Ollama models typically use less VRAM due to optimizations
            return size * 0.7  # 70% of model size
        else:
            # For HuggingFace models, estimate based on parameter count
            # This is a rough estimation - actual usage may vary
            return 4 * 1024 * 1024 * 1024  # Default 4GB for unknown models
    
    def _estimate_model_memory(self, model) -> float:
        """Estimate actual memory usage of loaded model"""
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return param_size + buffer_size
        except:
            return 0.0
    
    async def _ensure_vram_available(self, required_vram: float) -> bool:
        """Ensure sufficient VRAM is available, unload models if necessary"""
        available_vram = self.max_vram - self.current_vram_usage
        
        if available_vram >= required_vram:
            return True
        
        # Need to free up VRAM - unload least recently used models
        models_by_usage = sorted(
            [(name, info) for name, info in self.models.items() 
             if info.status == ModelStatus.LOADED],
            key=lambda x: x[1].last_used
        )
        
        for model_name, model_info in models_by_usage:
            if available_vram >= required_vram:
                break
            
            if model_name != self.active_model:  # Don't unload active model
                await self.unload_model(model_name)
                available_vram += model_info.vram_usage
        
        return available_vram >= required_vram
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a specific model to free VRAM"""
        if model_name not in self.loaded_models:
            return True
        
        try:
            # Remove from loaded models
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            # Update status
            if model_name in self.models:
                self.models[model_name].status = ModelStatus.UNLOADED
                self.current_vram_usage -= self.models[model_name].vram_usage
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"📤 Model {model_name} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unload model {model_name}: {e}")
            return False
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        if model_name == self.active_model:
            return True
        
        start_time = time.time()
        
        # Load the new model
        success = await self.load_model(model_name)
        
        if success:
            switch_time = time.time() - start_time
            logger.info(f"🔄 Switched to model {model_name} in {switch_time:.2f}s")
        
        return success
    
    async def get_best_model_for_task(
        self, 
        task_type: str, 
        language: str = "en",
        complexity: str = "medium"
    ) -> Optional[str]:
        """Intelligently select the best model for a specific task"""
        
        # Define task mappings
        task_mappings = {
            "code": ModelType.CODE,
            "programming": ModelType.CODE,
            "vision": ModelType.VISION,
            "image": ModelType.VISION,
            "chat": ModelType.CHAT,
            "conversation": ModelType.CHAT,
            "embedding": ModelType.EMBEDDING
        }
        
        target_type = task_mappings.get(task_type.lower(), ModelType.CHAT)
        
        # Filter models by type
        suitable_models = [
            (name, info) for name, info in self.models.items()
            if info.model_type == target_type
        ]
        
        if not suitable_models:
            # Fallback to any chat model
            suitable_models = [
                (name, info) for name, info in self.models.items()
                if info.model_type == ModelType.CHAT
            ]
        
        if not suitable_models:
            return None
        
        # Score models based on various factors
        scored_models = []
        for name, info in suitable_models:
            score = 0
            
            # Prefer loaded models
            if info.status == ModelStatus.LOADED:
                score += 100
            
            # Language support
            if language in info.languages or not info.languages:
                score += 50
            
            # Performance metrics
            avg_response_time = self.response_times.get(name)
            if avg_response_time:
                # Lower response time = higher score
                score += max(0, 50 - (sum(avg_response_time) / len(avg_response_time)))
            
            # Recency of use
            time_since_use = time.time() - info.last_used
            if time_since_use < 3600:  # Used within last hour
                score += 25
            
            scored_models.append((name, score))
        
        # Return highest scoring model
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models[0][0] if scored_models else None
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up Model Manager...")
        
        # Unload all models
        for model_name in list(self.loaded_models.keys()):
            await self.unload_model(model_name)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Model Manager cleanup complete")
    
    def record_request(self, model_name: str, response_time: float):
        """Record request for performance tracking"""
        if model_name not in self.request_counts:
            self.request_counts[model_name] = 0
            self.response_times[model_name] = []
        
        self.request_counts[model_name] += 1
        self.response_times[model_name].append(response_time)
        
        # Keep only last 100 response times
        if len(self.response_times[model_name]) > 100:
            self.response_times[model_name] = self.response_times[model_name][-100:]
    
    async def generate_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate response using specified model"""
        start_time = time.time()
        
        try:
            # Load model if not already loaded
            await self.load_model(model_name)
            
            # Get model instance
            model_info = self.models.get(model_name)
            if not model_info:
                raise ValueError(f"Model {model_name} not found")
            
            # Generate response based on provider
            if model_info.provider == "ollama":
                response = await self._generate_ollama_response(model_name, prompt, **kwargs)
            elif model_info.provider == "huggingface":
                response = await self._generate_hf_response(model_name, prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {model_info.provider}")
            
            # Update metrics
            response_time = (time.time() - start_time) * 1000
            await self._update_performance_metrics(model_name, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response with {model_name}: {e}")
            raise
    
    async def _generate_ollama_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate response using Ollama"""
        try:
            response = await self.ollama_client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    async def _generate_hf_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate response using HuggingFace model"""
        try:
            # This would be implemented based on your HF model loading logic
            # For now, return a placeholder
            return f"HuggingFace response from {model_name}: {prompt[:50]}..."
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise
