"""
Demo script for Custom Model Management System
Tests local model loading, generation, and training
"""

import asyncio
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.custom_model_manager import get_model_manager

async def demo_model_management():
    """Demo các tính năng chính của Custom Model Manager"""
    
    print("🚀 Custom Model Management Demo")
    print("="*50)
    
    # Get model manager
    manager = await get_model_manager()
    
    # 1. Scan available models
    print("\n📦 1. Available Local Models:")
    models = await manager.get_available_models()
    if models:
        for model in models:
            print(f"   • {model['name']} ({model['type']}) - {model['size_gb']:.1f}GB")
    else:
        print("   ⚠️ No models found. Please add models to local_models/ directory")
        print("   📝 See local_models/README.md for instructions")
    
    # 2. System status
    print("\n💻 2. System Status:")
    status = await manager.get_system_status()
    if status:
        gpu = status.get('gpu', {})
        print(f"   • GPU: {gpu.get('name', 'N/A')}")
        print(f"   • VRAM: {gpu.get('memory_used', 0):.1f}/{gpu.get('memory_total', 0):.1f}MB")
        print(f"   • Loaded Models: {status.get('loaded_models', 0)}")
        print(f"   • Total Models: {status.get('total_models', 0)}")
    
    # 3. Test generation (if models available)
    if models:
        print(f"\n🧠 3. Testing Generation:")
        first_model = models[0]['name']
        
        try:
            print(f"   Loading model: {first_model}")
            success = await manager.load_model(first_model, quantization="4bit")
            
            if success:
                print(f"   ✅ Model loaded successfully")
                
                # Test Vietnamese
                prompt = "Xin chào! Bạn có thể giúp tôi học Python không?"
                print(f"   🇻🇳 Vietnamese prompt: {prompt}")
                
                response = await manager.generate_response(
                    model_name=first_model,
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.7
                )
                print(f"   🤖 Response: {response[:200]}...")
                
                # Test code generation  
                code_prompt = "Viết function Python để sắp xếp list"
                print(f"   💻 Code prompt: {code_prompt}")
                
                code_response = await manager.generate_response(
                    model_name=first_model,
                    prompt=code_prompt,
                    max_tokens=150,
                    temperature=0.3
                )
                print(f"   🔧 Code response: {code_response[:200]}...")
                
            else:
                print(f"   ❌ Failed to load model")
                
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
    
    # 4. Training demo (if dataset exists)
    dataset_path = "training_data/vietnamese_coding_dataset.json"
    if Path(dataset_path).exists():
        print(f"\n🎓 4. Training Demo:")
        print(f"   📚 Dataset: {dataset_path}")
        
        if models:
            try:
                training_config = {
                    "epochs": 1,  # Quick demo
                    "batch_size": 2,
                    "learning_rate": 2e-4,
                    "lora_r": 8,
                    "lora_alpha": 16
                }
                
                job_id = await manager.start_training(
                    model_name=first_model,
                    dataset_path=dataset_path,
                    output_dir=f"custom_models/demo_trained_{first_model}",
                    training_config=training_config
                )
                
                print(f"   🚀 Training started: {job_id}")
                print(f"   ⏳ Training will run in background...")
                
            except Exception as e:
                print(f"   ❌ Training error: {e}")
    else:
        print(f"\n🎓 4. Training Demo:")
        print(f"   📚 Dataset not found: {dataset_path}")
        print(f"   💡 Sample dataset already created!")
    
    print(f"\n🎉 Demo completed!")
    print(f"\n📋 Next steps:")
    print(f"   1. Add your models to local_models/ directories")
    print(f"   2. Start the API: python main.py")
    print(f"   3. Test via API: http://localhost:8000/docs")

async def create_sample_model_structure():
    """Tạo cấu trúc mẫu cho demo"""
    
    print("\n📁 Creating sample model structure...")
    
    # Create placeholder model directories
    model_dirs = [
        "local_models/chat_models/demo-llama-7b",
        "local_models/code_models/demo-codellama-7b", 
        "local_models/vietnamese_models/demo-vi-llama",
        "local_models/custom_models/my-custom-model"
    ]
    
    for model_dir in model_dirs:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Create sample config.json
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            sample_config = {
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "torch_dtype": "float16"
            }
            
            with open(config_path, 'w') as f:
                json.dump(sample_config, f, indent=2)
        
        print(f"   ✅ {model_dir}")
    
    print("   💡 Placeholder models created for demo")
    print("   📝 Replace with real models for actual use")

if __name__ == "__main__":
    print("🤖 AI Backend Hub - Custom Model Management Demo")
    print("🎯 Toàn quyền kiểm soát models, training, và GPU!")
    print("="*60)
    
    # Create sample structure first
    asyncio.run(create_sample_model_structure())
    
    # Run demo
    asyncio.run(demo_model_management())
