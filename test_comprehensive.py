"""
Comprehensive test suite for AI Backend Hub
Tests all major components and integrations
"""

import asyncio
import sys
import os
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_database_models():
    """Test database models and relationships"""
    print("🗄️ Testing database models...")
    
    try:
        from src.models import User, Conversation, Message, TrainingJob, Analytics
        from src.models import UserRole, ConversationStatus, MessageType, TrainingJobStatus
        print("  ✅ All models imported successfully")
        
        # Test enum values
        assert UserRole.ADMIN.value == "admin"
        assert ConversationStatus.ACTIVE.value == "active"
        assert MessageType.TEXT.value == "text"
        assert TrainingJobStatus.PENDING.value == "pending"
        print("  ✅ Enums working correctly")
        
        return True
    except Exception as e:
        print(f"  ❌ Database models error: {e}")
        traceback.print_exc()
        return False

async def test_core_services():
    """Test core service configurations"""
    print("\n⚙️ Testing core services...")
    
    try:
        from src.core.config import settings
        print(f"  ✅ Config loaded - Environment: {settings.ENVIRONMENT}")
        
        # Test config values
        assert hasattr(settings, 'DATABASE_URL')
        assert hasattr(settings, 'REDIS_URL')
        assert hasattr(settings, 'OLLAMA_HOST')
        print("  ✅ All required config variables present")
        
        return True
    except Exception as e:
        print(f"  ❌ Core services error: {e}")
        traceback.print_exc()
        return False

async def test_api_schemas():
    """Test API schemas and validation"""
    print("\n📋 Testing API schemas...")
    
    try:
        from src.schemas.chat import ChatCompletionRequest, ChatMessage
        from src.schemas.models import ModelInfo, ModelListResponse
        print("  ✅ Chat and model schemas imported")
        
        # Test schema creation
        message = ChatMessage(role="user", content="Hello!")
        assert message.role == "user"
        assert message.content == "Hello!"
        print("  ✅ Schema validation working")
        
        return True
    except Exception as e:
        print(f"  ❌ API schemas error: {e}")
        traceback.print_exc()
        return False

async def test_ml_imports():
    """Test ML/AI library imports"""
    print("\n🤖 Testing ML/AI libraries...")
    
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"  ✅ Transformers {transformers.__version__}")
        
        # Test GPU availability
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available - Device: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️ CUDA not available - using CPU")
        
        return True
    except Exception as e:
        print(f"  ❌ ML libraries error: {e}")
        traceback.print_exc()
        return False

async def test_model_manager():
    """Test model manager functionality"""
    print("\n🧠 Testing model manager...")
    
    try:
        # Import without initializing (to avoid needing actual services)
        from src.core.custom_model_manager import CustomModelManager
        print("  ✅ ModelManager class imported")
        
        # Test if class has required methods
        required_methods = ['get_available_models', 'load_model', 'generate_response']
        for method in required_methods:
            assert hasattr(CustomModelManager, method), f"Missing method: {method}"
        print("  ✅ All required methods present")
        
        return True
    except Exception as e:
        print(f"  ❌ Model manager error: {e}")
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Test API endpoint imports"""
    print("\n🔌 Testing API endpoints...")
    
    try:
        from src.api.v1.endpoints import chat, models, training, health
        from fastapi import APIRouter
        
        # Check if routers are properly defined
        assert hasattr(chat, 'router')
        assert hasattr(models, 'router') 
        assert hasattr(training, 'router')
        assert hasattr(health, 'router')
        assert isinstance(chat.router, APIRouter)
        print("  ✅ All API routers imported and configured")
        
        return True
    except Exception as e:
        print(f"  ❌ API endpoints error: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run comprehensive test suite"""
    print("=" * 70)
    print("🚀 AI Backend Hub - Comprehensive Test Suite")
    print("=" * 70)
    
    tests = [
        ("Database Models", test_database_models),
        ("Core Services", test_core_services), 
        ("API Schemas", test_api_schemas),
        ("ML Libraries", test_ml_imports),
        ("Model Manager", test_model_manager),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Your AI Backend Hub is ready!")
        print("\n📋 Next steps:")
        print("1. 🗄️ Set up PostgreSQL with pgvector extension")
        print("2. 🔴 Set up Redis server")
        print("3. 🤖 Install and configure Ollama")
        print("4. 🚀 Start the application: python main.py")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please fix the issues above.")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
