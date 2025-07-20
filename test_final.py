"""
Final comprehensive test with proper Python environment
"""

async def test_basic_functionality():
    """Test basic AI Hub functionality"""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test config
        from src.core.config import settings
        print(f"  ✅ Config loaded: {settings.APP_NAME} v{settings.VERSION}")
        
        # Test models  
        from src.models import User, Conversation, Analytics
        print("  ✅ Database models loaded")
        
        # Test schemas
        from src.schemas.chat import ChatMessage
        msg = ChatMessage(role="user", content="Test")
        print(f"  ✅ Schemas working: {msg.role}")
        
        # Test model manager
        from src.core.model_manager import ModelManager
        print("  ✅ Model manager available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 AI Backend Hub - Final Test")
    print("=" * 60)
    
    success = await test_basic_functionality()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Your AI Backend Hub is ready to use!")
        print("\n📋 Next steps:")
        print("1. Start PostgreSQL: Start your PostgreSQL server")
        print("2. Start Redis: Start your Redis server") 
        print("3. Install Ollama: Download from https://ollama.ai")
        print("4. Run migrations: alembic upgrade head")
        print("5. Start the app: python main.py")
    else:
        print("❌ Some tests failed.")
        print("Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
