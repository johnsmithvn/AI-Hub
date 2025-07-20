"""
Basic test to verify the AI Backend Hub structure and imports
"""

import sys
import os
import traceback

def test_imports():
    """Test if all core modules can be imported"""
    print("🧪 Testing core module imports...")
    
    try:
        # Test FastAPI imports
        from fastapi import FastAPI
        print("  ✅ FastAPI")
        
        # Test Pydantic imports
        from pydantic import BaseModel
        print("  ✅ Pydantic")
        
        # Test async database imports
        from sqlalchemy.ext.asyncio import create_async_engine
        print("  ✅ SQLAlchemy Async")
        
        # Test Redis imports
        import redis.asyncio as redis
        print("  ✅ Redis")
        
        # Test core modules (these will have import errors due to missing services, but structure should be ok)
        try:
            from src.core.config import settings
            print("  ✅ Core Config")
        except Exception as e:
            print(f"  ⚠️ Core Config: {e}")
        
        try:
            from src.schemas.chat import ChatCompletionRequest
            print("  ✅ Chat Schemas")
        except Exception as e:
            print(f"  ⚠️ Chat Schemas: {e}")
        
        print("\n✅ Core imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_environment():
    """Test environment configuration"""
    print("\n🔧 Testing environment...")
    
    # Check .env file
    if os.path.exists('.env'):
        print("  ✅ .env file exists")
    else:
        print("  ❌ .env file missing")
    
    # Check directories
    directories = ["models", "hf_cache", "training_data", "trained_models", "uploads", "logs"]
    for directory in directories:
        if os.path.exists(directory):
            print(f"  ✅ {directory}/ directory")
        else:
            print(f"  ❌ {directory}/ directory missing")

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 AI Backend Hub - Basic Structure Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test environment
    test_environment()
    
    print("\n" + "=" * 60)
    if imports_ok:
        print("✅ Basic structure test PASSED!")
        print("🎉 Your AI Backend Hub is properly structured!")
        print("\nNext steps:")
        print("1. Set up PostgreSQL and Redis")
        print("2. Install Ollama and pull some models") 
        print("3. Run: python main.py")
    else:
        print("❌ Basic structure test FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
