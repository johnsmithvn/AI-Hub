#!/usr/bin/env python3
"""
AI Backend Hub - Quick Start Script
Helps you get the system up and running quickly
"""

import sys
import subprocess
import os
import platform
from pathlib import Path

def print_banner():
    """Print the AI Backend Hub banner"""
    print("=" * 60)
    print("🚀 AI Backend Hub - Quick Start Setup")
    print("=" * 60)
    print()

def check_requirements():
    """Check if required services are available"""
    print("🔍 Checking system requirements...")
    
    requirements = {
        "python": sys.version_info >= (3, 11),
        "pip": True,
    }
    
    # Check if we can import key packages
    try:
        import fastapi
        import uvicorn
        import pydantic
        requirements["fastapi"] = True
    except ImportError:
        requirements["fastapi"] = False
    
    # Check for optional services
    optional_services = {
        "postgres": check_service("pg_isready"),
        "redis": check_service("redis-cli"),
        "ollama": check_service("ollama"),
    }
    
    print("\n📋 System Check Results:")
    for name, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {name.title()}: {'OK' if status else 'Missing'}")
    
    print("\n🔧 Optional Services:")
    for name, status in optional_services.items():
        status_icon = "✅" if status else "⚠️"
        status_text = "Available" if status else "Not available"
        print(f"  {status_icon} {name.title()}: {status_text}")
    
    return all(requirements.values())

def check_service(command):
    """Check if a service command is available"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def setup_environment():
    """Set up the environment configuration"""
    print("\n🔧 Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy example env file
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("  ✅ Created .env file from template")
        print("  ⚠️  Please review and update .env with your settings")
    elif env_file.exists():
        print("  ✅ .env file already exists")
    else:
        print("  ❌ No .env.example found")

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "models",
        "hf_cache", 
        "training_data",
        "trained_models",
        "uploads",
        "logs",
        "training_jobs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")

def print_instructions():
    """Print next steps instructions"""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete! Next Steps:")
    print("=" * 60)
    
    print("\n1. 📝 Configure your environment:")
    print("   - Review and update .env file with your settings")
    print("   - Set database connection strings")
    print("   - Configure API keys if needed")
    
    print("\n2. 🗄️ Set up databases (if not using Docker):")
    print("   - Install PostgreSQL with pgvector extension")
    print("   - Install and start Redis")
    print("   - Run: alembic upgrade head")
    
    print("\n3. 🤖 Set up Ollama (recommended):")
    print("   - Install Ollama: https://ollama.ai/download")
    print("   - Pull some models:")
    print("     ollama pull llama2:7b")
    print("     ollama pull codellama:7b")
    print("     ollama pull llava:7b")
    
    print("\n4. 🚀 Start the application:")
    print("   - Development: python main.py")
    print("   - Production: uvicorn main:app --host 0.0.0.0 --port 8000")
    print("   - Docker: docker-compose up -d")
    
    print("\n5. 🌐 Access the system:")
    print("   - API: http://localhost:8000")
    print("   - Docs: http://localhost:8000/docs")
    print("   - Health: http://localhost:8000/health")
    
    print("\n6. 📚 Learn more:")
    print("   - Read README.md for detailed setup")
    print("   - Check docs/ for API documentation")
    print("   - Visit the GitHub repository for examples")

def main():
    """Main setup function"""
    print_banner()
    
    # Check requirements
    requirements_ok = check_requirements()
    
    if not requirements_ok:
        print("\n❌ Some requirements are missing. Please install them first.")
        print("Run: pip install -r requirements.txt")
        return 1
    
    # Setup environment
    setup_environment()
    
    # Create directories
    create_directories()
    
    # Print instructions
    print_instructions()
    
    print(f"\n✨ AI Backend Hub is ready! Happy coding! ✨")
    return 0

if __name__ == "__main__":
    sys.exit(main())
