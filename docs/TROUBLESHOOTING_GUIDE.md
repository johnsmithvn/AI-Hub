# 🔧 **Troubleshooting Guide**

## 🚨 **Common Issues and Solutions**

### **🏠 Overview**

This guide covers common issues you might encounter while setting up, running, or using AI Backend Hub và their solutions.

---

## 🐛 **Installation Issues**

### **1. Docker Installation Problems**

#### **Problem: Docker not starting**
```bash
Error: Cannot connect to the Docker daemon
```

**Solutions:**
```powershell
# Windows - Start Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Verify Docker is running
docker --version
docker info
```

#### **Problem: Permission denied**
```bash
Error: permission denied while trying to connect to Docker daemon
```

**Solutions:**
```bash
# Linux - Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Windows - Run PowerShell as Administrator
```

### **2. Python Environment Issues**

#### **Problem: Python version compatibility**
```bash
Error: Python 3.11+ required, found 3.9
```

**Solutions:**
```bash
# Install Python 3.11+
# Windows
winget install Python.Python.3.11

# Linux
sudo apt update && sudo apt install python3.11

# Verify version
python --version
```

#### **Problem: Virtual environment creation fails**
```bash
Error: Failed to create virtual environment
```

**Solutions:**
```bash
# Install venv module
pip install virtualenv

# Create environment manually
python -m venv ai-hub-env

# Activate environment
# Windows
ai-hub-env\Scripts\activate
# Linux/Mac
source ai-hub-env/bin/activate
```

---

## 🤖 **Model Management Issues**

### **3. Model Loading Problems**

#### **Problem: Out of VRAM**
```json
{
  "error": "CUDA out of memory",
  "details": "Model requires 8GB VRAM, only 4GB available"
}
```

**Solutions:**
```python
# Check VRAM usage
import torch
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print(f"VRAM Used: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

# Load smaller model
curl -X POST "http://localhost:8000/api/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama-7b", "quantization": "4bit"}'
```

#### **Problem: Model not found**
```json
{
  "error": "Model not found",
  "model": "custom-model-v1"
}
```

**Solutions:**
```bash
# List available models
curl http://localhost:8000/api/v1/models

# Download model manually
wget https://huggingface.co/model-repo/model-name
```

### **4. Model Performance Issues**

#### **Problem: Slow inference**
```json
{
  "response_time": "30s",
  "expected": "2-5s"
}
```

**Solutions:**
```python
# Enable optimizations
{
  "model_config": {
    "use_cache": true,
    "torch_compile": true,
    "flash_attention": true,
    "quantization": "4bit"
  }
}

# Check hardware
nvidia-smi  # GPU utilization
htop        # CPU usage
```

---

## 🌐 **API Issues**

### **5. Connection Problems**

#### **Problem: Server not responding**
```bash
curl: (7) Failed to connect to localhost port 8000
```

**Solutions:**
```bash
# Check if server is running
netstat -tulpn | grep 8000

# Start server manually
cd ai-backend-hub
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Check logs
docker logs ai-backend-hub
```

#### **Problem: CORS errors**
```javascript
Access to fetch at 'http://localhost:8000' from origin 'http://localhost:3000' 
has been blocked by CORS policy
```

**Solutions:**
```python
# Update CORS settings in src/core/config.py
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "*"  # Allow all (development only)
]
```

### **6. Authentication Issues**

#### **Problem: Invalid API key**
```json
{
  "error": "Invalid API key",
  "status": 401
}
```

**Solutions:**
```bash
# Generate new API key
curl -X POST "http://localhost:8000/api/v1/auth/generate-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app"}'

# Set environment variable
export AI_HUB_API_KEY="your-api-key-here"
```

---

## 💾 **Database Issues**

### **7. PostgreSQL Problems**

#### **Problem: Connection refused**
```bash
psql: error: connection to server on socket failed: Connection refused
```

**Solutions:**
```bash
# Start PostgreSQL service
# Windows
net start postgresql-x64-14

# Linux
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Docker
docker run --name postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:14
```

#### **Problem: Database migration fails**
```bash
alembic.util.exc.CommandError: Target database is not up to date
```

**Solutions:**
```bash
# Reset migrations
alembic stamp head
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head

# Or reset database
docker-compose down -v
docker-compose up -d postgres
```

### **8. Redis Issues**

#### **Problem: Redis connection failed**
```bash
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solutions:**
```bash
# Start Redis service
# Windows
redis-server

# Linux
sudo systemctl start redis
sudo systemctl enable redis

# Docker
docker run --name redis -p 6379:6379 -d redis:7-alpine

# Test connection
redis-cli ping
```

---

## 🧪 **Testing Issues**

### **9. Test Failures**

#### **Problem: Tests not running**
```bash
pytest: command not found
```

**Solutions:**
```bash
# Install pytest
pip install pytest pytest-asyncio pytest-mock

# Run tests
python -m pytest tests/

# With coverage
python -m pytest --cov=src tests/
```

#### **Problem: Database tests failing**
```bash
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection failed
```

**Solutions:**
```python
# Use test database
# tests/conftest.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine

@pytest.fixture
async def test_db():
    engine = create_async_engine("postgresql+asyncpg://test:test@localhost/test_db")
    # Setup test data
    yield engine
    # Cleanup
```

---

## 🔒 **Security Issues**

### **10. SSL/TLS Problems**

#### **Problem: Certificate verification failed**
```bash
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions:**
```python
# Development only - disable SSL verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Production - use proper certificates
# nginx.conf
ssl_certificate /path/to/certificate.crt;
ssl_certificate_key /path/to/private.key;
```

---

## ⚡ **Performance Issues**

### **11. Memory Leaks**

#### **Problem: Memory usage increasing**
```bash
Memory usage: 16GB -> 32GB -> 48GB (increasing)
```

**Solutions:**
```python
# Monitor memory
import psutil
import torch

def check_memory():
    print(f"RAM: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

# Clear VRAM periodically
torch.cuda.empty_cache()

# Implement proper cleanup
@app.on_event("shutdown")
async def cleanup():
    # Cleanup resources
    pass
```

### **12. High CPU Usage**

#### **Problem: CPU at 100%**
```bash
CPU usage constantly at 100%
```

**Solutions:**
```python
# Use async operations
import asyncio
import aiofiles

async def process_file(file_path):
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
    return content

# Limit concurrent operations
semaphore = asyncio.Semaphore(4)  # Max 4 concurrent operations
```

---

## 🛠️ **Development Issues**

### **13. Hot Reload Not Working**

#### **Problem: Changes not reflecting**
```bash
Server running but changes not reflected
```

**Solutions:**
```bash
# Use reload flag
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Check file watching
# On Windows, might need to install watchfiles
pip install watchfiles
```

### **14. Import Errors**

#### **Problem: Module not found**
```python
ModuleNotFoundError: No module named 'src'
```

**Solutions:**
```python
# Add to PYTHONPATH
import sys
sys.path.append('/path/to/ai-backend-hub')

# Or use relative imports
from .core.config import settings

# Or install in development mode
pip install -e .
```

---

## 🔧 **Quick Diagnostics**

### **System Health Check**

```bash
#!/bin/bash
# health-check.sh

echo "=== AI Backend Hub Health Check ==="

# Check Docker
echo "1. Docker Status:"
docker --version && echo "✅ Docker OK" || echo "❌ Docker Error"

# Check Python
echo "2. Python Status:"
python --version && echo "✅ Python OK" || echo "❌ Python Error"

# Check API
echo "3. API Status:"
curl -s http://localhost:8000/health && echo "✅ API OK" || echo "❌ API Error"

# Check Database
echo "4. Database Status:"
psql -h localhost -U postgres -c "SELECT 1;" && echo "✅ DB OK" || echo "❌ DB Error"

# Check Redis
echo "5. Redis Status:"
redis-cli ping && echo "✅ Redis OK" || echo "❌ Redis Error"

# Check VRAM
echo "6. VRAM Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
```

### **Performance Monitoring**

```python
# monitoring.py
import psutil
import torch
import time

def system_monitor():
    """Monitor system resources"""
    while True:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory
        memory = psutil.virtual_memory()
        
        # VRAM
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            vram_used = vram_total = 0
        
        print(f"CPU: {cpu_percent}% | RAM: {memory.percent}% | VRAM: {vram_used:.1f}/{vram_total:.1f}GB")
        time.sleep(5)

if __name__ == "__main__":
    system_monitor()
```

---

## 📞 **Getting Help**

### **🔍 Before Asking for Help**

1. **Check Logs**
   ```bash
   # Application logs
   tail -f logs/app.log
   
   # Docker logs
   docker logs ai-backend-hub
   
   # System logs
   journalctl -u ai-backend-hub
   ```

2. **Verify Configuration**
   ```bash
   # Check environment variables
   env | grep AI_HUB
   
   # Validate config
   python -c "from src.core.config import settings; print(settings.dict())"
   ```

3. **Test Connectivity**
   ```bash
   # API health
   curl http://localhost:8000/health
   
   # Database connection
   psql -h localhost -U postgres -c "SELECT version();"
   
   # Model availability
   curl http://localhost:8000/api/v1/models
   ```

### **📋 Issue Report Template**

When reporting issues, include:

```markdown
**Environment:**
- OS: Windows 11 / Ubuntu 22.04 / macOS
- Python Version: 3.11.x
- Docker Version: 24.x.x
- GPU: NVIDIA RTX 4090 / None

**Issue Description:**
Clear description of the problem

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Logs:**
```
Include relevant log output
```

**Configuration:**
Include relevant config settings (remove sensitive data)
```

### **🤝 Community Support**

- **GitHub Issues**: [Report bugs and feature requests]
- **Discussions**: [Community discussions and Q&A]
- **Documentation**: [Check this documentation first]
- **Vietnamese Support**: [Vietnamese community support]

---

**Remember: Most issues have simple solutions. Check this guide first before seeking help!** 🎯
