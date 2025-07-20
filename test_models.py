"""
Quick test to verify model imports
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_model_imports():
    """Test model imports directly"""
    print("Testing individual model files...")
    
    try:
        from models.user import User, UserRole
        print("✅ User model imported")
        
        from models.conversation import Conversation, Message
        print("✅ Conversation models imported")
        
        from models.training import TrainingJob
        print("✅ Training model imported")
        
        from models.analytics import Analytics
        print("✅ Analytics model imported")
        
        # Test collective import
        from models import User, Conversation, TrainingJob, Analytics
        print("✅ All models imported from __init__.py")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Quick Model Import Test")
    print("=" * 50)
    
    success = test_model_imports()
    
    if success:
        print("✅ All model imports successful!")
    else:
        print("❌ Model import test failed!")
