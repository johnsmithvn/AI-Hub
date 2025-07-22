#!/usr/bin/env python3
"""
Test script for the hybrid model management system
Tests both HuggingFace and GGUF model support
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import settings
from src.core.custom_model_manager import CustomModelManager

async def test_hybrid_system():
    """Test the hybrid model management system"""
    print("🧪 Testing Hybrid Model Management System")
    print("=" * 50)
    
    # Initialize the model manager
    print("🔧 Initializing CustomModelManager...")
    manager = CustomModelManager()
    
    # Give it time to initialize
    await asyncio.sleep(2)
    
    # Get available models
    print("\n📋 Available Models:")
    models = await manager.get_available_models()
    
    for name, info in models.items():
        print(f"  📦 {name}")
        print(f"     Format: {info.model_format.value}")
        print(f"     Type: {info.model_type.value}")
        print(f"     Provider: {info.provider}")
        print(f"     Status: {info.status.value}")
        print(f"     Size: {info.size_gb:.1f} GB")
        if hasattr(info, 'quantization') and info.quantization:
            print(f"     Quantization: {info.quantization}")
        print()
    
    # Test configuration
    print("\n⚙️ Configuration Status:")
    print(f"  GGUF Models Enabled: {settings.ENABLE_GGUF_MODELS}")
    print(f"  External Models Dir: {settings.EXTERNAL_MODELS_DIR}")
    print(f"  Smart Switching: {settings.ENABLE_SMART_SWITCHING}")
    print(f"  Model Keepalive: {settings.MODEL_KEEPALIVE_MINUTES} minutes")
    
    # Test hybrid extensions availability
    print(f"\n🔌 Hybrid Extensions Available: {hasattr(manager, 'load_model_hybrid')}")
    
    if hasattr(manager, 'load_model_hybrid'):
        print("  ✅ load_model_hybrid")
        print("  ✅ generate_response_hybrid") if hasattr(manager, 'generate_response_hybrid') else print("  ❌ generate_response_hybrid")
        print("  ✅ unload_model_hybrid") if hasattr(manager, 'unload_model_hybrid') else print("  ❌ unload_model_hybrid")
    
    # Test model loading if models are available
    if models:
        print("\n🚀 Testing Model Operations:")
        
        # Find a small model to test with
        test_model = None
        for name, info in models.items():
            if info.size_gb < 5.0:  # Use models smaller than 5GB for testing
                test_model = name
                break
        
        if test_model:
            print(f"  Testing with: {test_model}")
            
            try:
                # Test loading
                print("  📥 Testing model loading...")
                success = await manager.load_model(test_model)
                print(f"     Load result: {'✅ Success' if success else '❌ Failed'}")
                
                if success:
                    # Test generation with a simple prompt
                    print("  💭 Testing text generation...")
                    response = await manager.generate_response(
                        test_model, 
                        "Hello, how are you?", 
                        max_tokens=50,
                        temperature=0.7
                    )
                    print(f"     Response: {response[:100]}...")
                    
                    # Test unloading
                    print("  📤 Testing model unloading...")
                    unload_success = await manager.unload_model(test_model)
                    print(f"     Unload result: {'✅ Success' if unload_success else '❌ Failed'}")
                
            except Exception as e:
                print(f"     ❌ Error during testing: {e}")
        else:
            print("  ⚠️ No suitable small models found for testing")
    else:
        print("\n⚠️ No models found. Check configuration and model directories.")
    
    print("\n✅ Hybrid system test completed!")

if __name__ == "__main__":
    # Test if we can import the required modules
    try:
        from src.core.hybrid_extensions import load_model_hybrid
        print("✅ Hybrid extensions imported successfully")
    except ImportError as e:
        print(f"⚠️ Hybrid extensions import failed: {e}")
    
    try:
        from src.core.gguf_loader import GGUFModelLoader
        print("✅ GGUF loader imported successfully")
    except ImportError as e:
        print(f"⚠️ GGUF loader import failed: {e}")
    
    # Run the test
    asyncio.run(test_hybrid_system())
