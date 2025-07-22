"""
GGUF Model Support for External Models
Supports llama.cpp compatible GGUF models from external directory
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. GGUF models will not be available.")

from .config import settings

@dataclass
class GGUFModelInfo:
    """GGUF model information"""
    name: str
    provider: str
    model_path: str
    file_size_gb: float
    quantization: str
    context_length: int = 4096
    gpu_layers: int = -1
    load_time: float = 0.0
    last_used: float = 0.0
    is_loaded: bool = False
    estimated_vram_mb: float = 0.0

class GGUFModelLoader:
    """
    Loader for GGUF models from external directory
    Integrates with CustomModelManager while maintaining separate logic
    """
    
    def __init__(self):
        self.gguf_models: Dict[str, GGUFModelInfo] = {}
        self.loaded_gguf_instances: Dict[str, Llama] = {}
        self.external_dir = Path(settings.EXTERNAL_MODELS_DIR)
        
        # Provider mapping based on common folder structure
        self.provider_types = {
            "lmstudio-community": "chat",
            "brittlewis12": "chat", 
            "MaziyarPanahi": "chat",
            "TheBloke": "chat",
            "unsloth": "instruct",
            "vilm": "vietnamese",
            "microsoft": "code"
        }
    
    async def scan_gguf_models(self) -> Dict[str, GGUFModelInfo]:
        """Scan external directory for GGUF models"""
        if not LLAMA_CPP_AVAILABLE:
            logger.warning("GGUF support disabled - llama-cpp-python not available")
            return {}
        
        if not self.external_dir.exists():
            logger.warning(f"External models directory not found: {self.external_dir}")
            return {}
        
        logger.info(f"🔍 Scanning GGUF models in {self.external_dir}")
        
        # Scan provider directories
        for provider_dir in self.external_dir.iterdir():
            if not provider_dir.is_dir():
                continue
            
            provider_name = provider_dir.name
            await self._scan_provider_models(provider_dir, provider_name)
        
        logger.info(f"📊 Found {len(self.gguf_models)} GGUF models")
        return self.gguf_models
    
    async def _scan_provider_models(self, provider_dir: Path, provider_name: str):
        """Scan models within a provider directory"""
        for model_dir in provider_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Find GGUF files
            gguf_files = list(model_dir.glob("*.gguf"))
            if not gguf_files:
                continue
            
            # Use the largest GGUF file as main model
            main_gguf = max(gguf_files, key=lambda x: x.stat().st_size)
            
            # Create model info
            model_name = f"{provider_name}/{model_dir.name}"
            file_size_gb = main_gguf.stat().st_size / (1024**3)
            quantization = self._detect_quantization(main_gguf.name)
            
            model_info = GGUFModelInfo(
                name=model_name,
                provider=provider_name,
                model_path=str(main_gguf),
                file_size_gb=file_size_gb,
                quantization=quantization,
                context_length=settings.GGUF_CONTEXT_LENGTH,
                gpu_layers=settings.GGUF_GPU_LAYERS,
                estimated_vram_mb=self._estimate_gguf_vram(file_size_gb, quantization)
            )
            
            self.gguf_models[model_name] = model_info
            logger.info(f"✅ Found GGUF: {model_name} ({file_size_gb:.1f}GB, {quantization})")
    
    def _detect_quantization(self, filename: str) -> str:
        """Detect quantization from GGUF filename"""
        filename_lower = filename.lower()
        
        if any(x in filename_lower for x in ["q4_0", "q4_1", "q4_k_m", "q4_k_s"]):
            return "Q4"
        elif any(x in filename_lower for x in ["q8_0", "q8_1"]):
            return "Q8"
        elif any(x in filename_lower for x in ["q2_k", "q3_k"]):
            return "Q2-Q3"
        elif "q5" in filename_lower:
            return "Q5"
        elif "q6" in filename_lower:
            return "Q6"
        elif "f16" in filename_lower:
            return "F16"
        elif "f32" in filename_lower:
            return "F32"
        
        return "Unknown"
    
    def _estimate_gguf_vram(self, file_size_gb: float, quantization: str) -> float:
        """Estimate VRAM usage for GGUF model"""
        # Base VRAM is roughly file size + context overhead
        base_vram_gb = file_size_gb
        
        # Add context overhead (varies by context length)
        context_overhead = (settings.GGUF_CONTEXT_LENGTH / 4096) * 0.5  # ~0.5GB per 4K context
        
        # Add model overhead
        model_overhead = 0.3  # ~300MB overhead
        
        total_vram_gb = base_vram_gb + context_overhead + model_overhead
        return total_vram_gb * 1024  # Convert to MB
    
    async def load_gguf_model(self, model_name: str) -> bool:
        """Load a GGUF model"""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("Cannot load GGUF model - llama-cpp-python not available")
            return False
        
        if model_name not in self.gguf_models:
            logger.error(f"GGUF model {model_name} not found")
            return False
        
        if model_name in self.loaded_gguf_instances:
            logger.info(f"GGUF model {model_name} already loaded")
            return True
        
        model_info = self.gguf_models[model_name]
        
        try:
            logger.info(f"🚀 Loading GGUF model: {model_name}")
            start_time = time.time()
            
            # Load with llama.cpp
            llama_instance = Llama(
                model_path=model_info.model_path,
                n_ctx=model_info.context_length,
                n_gpu_layers=model_info.gpu_layers,
                verbose=False,
                use_mlock=True,
                use_mmap=True,
                n_threads=None,  # Auto-detect
                n_batch=512
            )
            
            # Store loaded instance
            self.loaded_gguf_instances[model_name] = llama_instance
            
            # Update model info
            model_info.is_loaded = True
            model_info.load_time = time.time() - start_time
            model_info.last_used = time.time()
            
            logger.info(f"✅ GGUF model {model_name} loaded in {model_info.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load GGUF model {model_name}: {e}")
            return False
    
    async def unload_gguf_model(self, model_name: str) -> bool:
        """Unload a GGUF model"""
        if model_name not in self.loaded_gguf_instances:
            return True
        
        try:
            # Delete instance (llama.cpp will handle cleanup)
            del self.loaded_gguf_instances[model_name]
            
            # Update model info
            if model_name in self.gguf_models:
                self.gguf_models[model_name].is_loaded = False
            
            logger.info(f"🗑️ GGUF model {model_name} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unload GGUF model {model_name}: {e}")
            return False
    
    async def generate_gguf_response(
        self, 
        model_name: str, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate response using GGUF model"""
        if model_name not in self.loaded_gguf_instances:
            raise ValueError(f"GGUF model {model_name} not loaded")
        
        llama_instance = self.loaded_gguf_instances[model_name]
        
        try:
            # Generate response
            response = llama_instance(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
                stop=["</s>", "[/INST]", "<|im_end|>"],
                **kwargs
            )
            
            # Update last used time
            if model_name in self.gguf_models:
                self.gguf_models[model_name].last_used = time.time()
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"❌ GGUF generation failed for {model_name}: {e}")
            raise

# Global GGUF loader instance
_gguf_loader = None

async def get_gguf_loader() -> GGUFModelLoader:
    """Get global GGUF loader instance"""
    global _gguf_loader
    if _gguf_loader is None:
        _gguf_loader = GGUFModelLoader()
        await _gguf_loader.scan_gguf_models()
    return _gguf_loader
