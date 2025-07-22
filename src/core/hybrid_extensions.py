"""
Hybrid Model Loading Extensions for CustomModelManager
Adds GGUF support while maintaining HuggingFace compatibility
"""

from typing import Dict, Optional, Any
import time
from loguru import logger

async def load_model_hybrid(self, model_name: str, quantization: str = "4bit", force: bool = False) -> bool:
    """
    Hybrid model loading - supports both HuggingFace and GGUF models
    """
    try:
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found in registry")
            return False
        
        # Check if already loaded
        is_hf_loaded = model_name in self.loaded_models
        is_gguf_loaded = self.gguf_loader and hasattr(self.gguf_loader, 'loaded_gguf_instances') and model_name in self.gguf_loader.loaded_gguf_instances
        
        if (is_hf_loaded or is_gguf_loaded) and not force:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        model_info = self.models[model_name]
        model_info.status = self.ModelStatus.LOADING
        
        # Route to appropriate loader based on format
        if model_info.model_format == self.ModelFormat.GGUF:
            success = await self._load_gguf_model_internal(model_name)
        else:
            success = await self._load_huggingface_model_internal(model_name, quantization, force)
        
        if success:
            model_info.status = self.ModelStatus.LOADED
            self.active_model = model_name
            self._update_combined_registry()  # Sync registries
            
        return success
        
    except Exception as e:
        if model_name in self.models:
            self.models[model_name].status = self.ModelStatus.ERROR
        logger.error(f"❌ Failed to load {model_name}: {e}")
        return False

async def _load_gguf_model_internal(self, model_name: str) -> bool:
    """Load GGUF model using gguf_loader"""
    if not self.gguf_loader:
        logger.error("GGUF loader not available")
        return False
    
    try:
        success = await self.gguf_loader.load_gguf_model(model_name)
        if success:
            model_info = self.models[model_name]
            gguf_info = self.gguf_loader.gguf_models.get(model_name)
            if gguf_info:
                model_info.load_time = gguf_info.load_time
                model_info.last_used = gguf_info.last_used
                model_info.vram_usage = gguf_info.estimated_vram_mb
            
            logger.info(f"✅ GGUF model {model_name} loaded successfully")
        return success
    except Exception as e:
        logger.error(f"❌ Failed to load GGUF model {model_name}: {e}")
        return False

async def _load_huggingface_model_internal(self, model_name: str, quantization: str, force: bool) -> bool:
    """Load HuggingFace model using transformers"""
    try:
        model_info = self.models[model_name]
        logger.info(f"🚀 Loading HF model: {model_name} from {model_info.local_path}")
        
        # Configure quantization
        bnb_config = None
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            import torch
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer and model
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_info.local_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_info.local_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Store loaded model and tokenizer
        load_time = time.time()
        self.loaded_models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "info": model_info,
            "load_time": load_time,
            "generation_config": GenerationConfig.from_pretrained(model_info.local_path)
        }
        
        # Store tokenizer separately for API compatibility
        self.tokenizers[model_name] = tokenizer
        
        # Update model info
        model_info.load_time = load_time
        model_info.last_used = time.time()
        model_info.vram_usage = self._get_model_vram_usage(model)
        model_info.quantization = quantization
        
        logger.info(f"✅ HF model {model_name} loaded successfully ({model_info.vram_usage:.1f}MB VRAM)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load HF model {model_name}: {e}")
        return False

async def generate_response_hybrid(
    self, 
    model_name: str, 
    prompt: str, 
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs
) -> str:
    """
    Hybrid response generation - routes to HF or GGUF based on model format
    """
    if model_name not in self.models:
        raise ValueError(f"Model {model_name} not found")
    
    model_info = self.models[model_name]
    
    # Route based on model format
    if model_info.model_format == self.ModelFormat.GGUF:
        if not self.gguf_loader:
            raise ValueError("GGUF loader not available")
        return await self.gguf_loader.generate_gguf_response(
            model_name, prompt, max_tokens, temperature, top_p, **kwargs
        )
    else:
        # Use existing HuggingFace generation
        return await self._generate_hf_response(
            model_name, prompt, max_tokens, temperature, top_p
        )

async def _generate_hf_response(
    self, 
    model_name: str, 
    prompt: str, 
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """Generate response using HuggingFace model"""
    try:
        if model_name not in self.loaded_models:
            raise Exception(f"Model {model_name} not loaded")
        
        model_data = self.loaded_models[model_name]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        # Prepare input
        import torch
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
        self.models[model_name].last_used = time.time()
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"❌ HF generation failed for {model_name}: {e}")
        raise

async def unload_model_hybrid(self, model_name: str) -> bool:
    """
    Hybrid model unloading - handles both HF and GGUF models
    """
    try:
        model_info = self.models.get(model_name)
        if not model_info:
            return True
        
        success = True
        
        # Unload HuggingFace model if loaded
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            # Clear GPU cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Unload GGUF model if loaded
        if self.gguf_loader and hasattr(self.gguf_loader, 'loaded_gguf_instances'):
            if model_name in self.gguf_loader.loaded_gguf_instances:
                gguf_success = await self.gguf_loader.unload_gguf_model(model_name)
                success = success and gguf_success
        
        # Update status
        model_info.status = self.ModelStatus.UNLOADED
        model_info.vram_usage = 0.0
        
        if self.active_model == model_name:
            self.active_model = None
        
        logger.info(f"🗑️ Model {model_name} unloaded")
        return success
        
    except Exception as e:
        logger.error(f"❌ Failed to unload {model_name}: {e}")
        return False
