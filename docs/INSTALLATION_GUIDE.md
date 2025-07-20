# 🛠️ **Installation & Setup Guide**

## 📋 **System Requirements**

### **Hardware Requirements**
- **GPU**: NVIDIA RTX 4060 Ti 16GB (recommended) hoặc tương đương
- **RAM**: Minimum 16GB, recommended 32GB
- **Storage**: 100GB+ SSD space for models
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5+)

### **Software Requirements**
- **OS**: Windows 11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.11+ (required)
- **CUDA**: 11.8+ (for GPU acceleration)
- **Git**: Latest version
- **Docker**: Optional, for containerized deployment

---

## 🚀 **Quick Start Installation**

### **Step 1: Clone Repository**

```bash
git clone https://github.com/johnsmithvn/AI-Hub.git
cd AI-Hub
```

### **Step 2: Setup Python Environment**

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### **Step 3: Install Dependencies**

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all requirements
pip install -r requirements.txt

# Verify GPU detection
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### **Step 4: Database Setup**

```bash
# Install PostgreSQL (Ubuntu)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database
sudo -u postgres createdb ai_hub

# Install Redis
sudo apt install redis-server
sudo systemctl start redis-server
```

### **Step 5: Environment Configuration**

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Sample .env configuration:**
```bash
# Database
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost/ai_hub
REDIS_URL=redis://localhost:6379/0

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Security
JWT_SECRET_KEY=your-super-secret-key-here
API_RATE_LIMIT=100

# AI Settings
MAX_VRAM_USAGE=0.85
DEFAULT_MODEL=llama2-7b-chat
MODEL_CACHE_SIZE=3

# Paths
LOCAL_MODELS_DIR=./local_models
TRAINING_DATA_DIR=./training_data
TRAINED_MODELS_DIR=./trained_models
```

### **Step 6: Initialize Database**

```bash
# Run database migrations
alembic upgrade head

# Verify database setup
python -c "from src.core.database import engine; print('Database connected successfully')"
```

### **Step 7: Start Application**

```bash
# Development mode
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Verify installation:** Open `http://localhost:8000/docs` để xem API documentation.

---

## 🐳 **Docker Installation** (Alternative)

### **Using Docker Compose** (Recommended)

```bash
# Clone repository
git clone https://github.com/johnsmithvn/AI-Hub.git
cd AI-Hub

# Configure environment
cp .env.example .env
# Edit .env file với your settings

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f ai-hub

# Access application
open http://localhost:8000/docs
```

### **Manual Docker Build**

```bash
# Build image
docker build -t ai-hub .

# Run container
docker run -d \
  --name ai-hub \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/local_models:/app/local_models \
  -v $(pwd)/training_data:/app/training_data \
  -e DATABASE_URL=postgresql://user:pass@host/db \
  ai-hub
```

---

## 📁 **Directory Structure Setup**

### **Create Required Directories**

```bash
# Create model directories
mkdir -p local_models/{chat_models,code_models,vietnamese_models,custom_models}
mkdir -p training_data/{datasets,logs}
mkdir -p trained_models
mkdir -p uploads
mkdir -p logs
```

### **Expected Structure**
```
AI-Hub/
├── local_models/
│   ├── chat_models/           # Llama2, Mistral, Qwen
│   ├── code_models/           # CodeLlama, Deepseek-Coder
│   ├── vietnamese_models/     # Vietnamese-specific models
│   └── custom_models/         # Your trained models
├── training_data/
│   ├── datasets/              # Training datasets
│   └── logs/                  # Training logs
├── trained_models/            # Output của custom training
├── uploads/                   # Temporary file uploads
└── logs/                      # Application logs
```

---

## 🤖 **Model Setup**

### **Download Models**

AI Backend Hub supports manual model placement. Download models từ HuggingFace:

```bash
# Using git lfs (recommended)
git lfs install

# Clone model repository
cd local_models/chat_models/
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

# Or download specific files
wget https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/resolve/main/pytorch_model.bin
```

### **Recommended Models**

**Chat Models:**
- `meta-llama/Llama-2-7b-chat-hf` - General conversation
- `mistralai/Mistral-7B-Instruct-v0.1` - Fast inference
- `Qwen/Qwen-7B-Chat` - Multilingual support

**Code Models:**
- `codellama/CodeLlama-7b-Instruct-hf` - Code generation
- `deepseek-ai/deepseek-coder-6.7b-instruct` - Code understanding

**Vietnamese Models:**
- `vilm/viet-llama2-7b-chat` - Vietnamese conversation
- `Qwen/Qwen-7B-Chat` (supports Vietnamese)

### **Model Placement**

```bash
# Place models in appropriate directories:
local_models/
├── chat_models/
│   ├── Llama-2-7b-chat-hf/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── Mistral-7B-Instruct-v0.1/
└── code_models/
    └── CodeLlama-7b-Instruct-hf/
```

---

## ⚙️ **Configuration Guide**

### **GPU Configuration**

```python
# config/gpu_settings.py
GPU_CONFIG = {
    "max_vram_usage": 0.85,        # Use 85% of 16GB = ~13.6GB
    "reserved_vram": 2048,         # Reserve 2GB for system
    "auto_offload_threshold": 0.9, # Auto-unload at 90%
    "quantization_default": "4bit", # Default quantization
    "model_parallel": False        # Single GPU setup
}
```

### **Model Loading Configuration**

```python
# Model-specific settings
MODEL_CONFIGS = {
    "llama2-7b-chat": {
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.bfloat16
    },
    "codellama-7b-instruct": {
        "max_new_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.95,
        "specialties": ["code_generation"]
    }
}
```

### **Training Configuration**

```python
# LoRA training defaults
TRAINING_CONFIG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "batch_size": 4,
    "learning_rate": 2e-4,
    "max_steps": 1000,
    "save_steps": 100
}
```

---

## 🧪 **Testing Installation**

### **Health Check Script**

```python
# test_installation.py
import asyncio
import requests
import torch
from src.core.custom_model_manager import get_model_manager

async def test_installation():
    print("🔍 Testing AI Backend Hub Installation...")
    
    # Test 1: GPU Detection
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Test 2: Model Manager
    try:
        manager = await get_model_manager()
        models = await manager.get_available_models()
        print(f"✅ Found {len(models)} local models")
    except Exception as e:
        print(f"❌ Model Manager Error: {e}")
    
    # Test 3: API Health
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ API Health: {response.status_code}")
    except Exception as e:
        print(f"❌ API Error: {e}")
    
    print("🎉 Installation test completed!")

if __name__ == "__main__":
    asyncio.run(test_installation())
```

```bash
# Run test
python test_installation.py
```

### **Demo Script**

```bash
# Run demo với sample model
python demo_custom_model_manager.py

# Expected output:
# 🔍 AI Backend Hub Demo
# 🖥️ System: Windows
# 🎮 GPU: NVIDIA GeForce RTX 4060 Ti, VRAM: 1170.0/16380.0MB
# 📦 Loaded Models: 0, Total Models: 0
# ✅ Custom Model Manager ready for use!
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

**1. CUDA Not Available**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Database Connection Error**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Reset database
sudo -u postgres dropdb ai_hub
sudo -u postgres createdb ai_hub
alembic upgrade head
```

**3. Model Loading Failed**
```bash
# Check model files
ls -la local_models/chat_models/your_model/
# Required files: config.json, pytorch_model.bin, tokenizer files

# Check VRAM availability
nvidia-smi
# Free up VRAM nếu cần
```

**4. Port Already in Use**
```bash
# Kill process using port 8000
sudo lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn main:app --port 8001
```

### **Performance Optimization**

**Memory Optimization:**
```python
# In .env file
MAX_VRAM_USAGE=0.8          # Reduce if getting OOM errors
MODEL_CACHE_SIZE=2          # Reduce number of cached models
TRAINING_BATCH_SIZE=2       # Reduce batch size for training
```

**Speed Optimization:**
```python
# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

---

## 🔄 **Updates & Maintenance**

### **Updating AI Backend Hub**

```bash
# Backup current setup
cp .env .env.backup
cp -r local_models local_models.backup

# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run migrations
alembic upgrade head

# Restart application
systemctl restart ai-hub  # If using systemd service
```

### **Monitoring Setup**

```bash
# Install monitoring tools
pip install prometheus-client grafana-api

# Setup log rotation
sudo apt install logrotate
```

**Systemd Service** (Linux):
```bash
# Create service file
sudo nano /etc/systemd/system/ai-hub.service

[Unit]
Description=AI Backend Hub
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/AI-Hub
Environment=PATH=/path/to/.venv/bin
ExecStart=/path/to/.venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable service
sudo systemctl enable ai-hub
sudo systemctl start ai-hub
```

---

## 🎉 **Deployment Ready!**

After completing this setup guide:

✅ **AI Backend Hub** running on `http://localhost:8000`  
✅ **OpenAPI Documentation** at `http://localhost:8000/docs`  
✅ **Local models** ready for loading  
✅ **Training pipeline** ready for custom datasets  
✅ **Production-ready** configuration  

**Your local AI infrastructure is now ready for integration with applications!** 🚀
