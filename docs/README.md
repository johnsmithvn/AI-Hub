# AI Backend Hub

## Comprehensive Multi-Modal AI System

A powerful, local-first AI backend that provides dynamic model management, custom training capabilities, and multi-modal processing for building advanced AI applications.

### 🚀 Key Features

- **Dynamic Model Management**: Intelligent loading, switching, and resource optimization for multiple AI models
- **Custom Training Pipeline**: LoRA/QLoRA fine-tuning with automated job management
- **Multi-Modal Processing**: Text, voice, images, and document processing capabilities
- **OpenAI-Compatible API**: Drop-in replacement for existing OpenAI integrations
- **Local Deployment**: Complete privacy with local model serving
- **Advanced Analytics**: Comprehensive monitoring and performance tracking

### 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Backend Hub                           │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Application Layer                                  │
│  ├── Chat API (OpenAI Compatible)                          │
│  ├── Model Management API                                  │
│  ├── Training Pipeline API                                 │
│  ├── Multi-Modal Processing APIs                           │
│  └── Analytics & Monitoring APIs                           │
├─────────────────────────────────────────────────────────────┤
│  Core Services Layer                                       │
│  ├── Model Manager (Dynamic Loading/Switching)             │
│  ├── Training Manager (LoRA/QLoRA)                        │
│  ├── Multi-Modal Processors                               │
│  └── Intelligent Routing System                            │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                │
│  ├── PostgreSQL (Structured Data + Vector Search)          │
│  ├── Redis (Caching + Session Management)                  │
│  └── File System (Models + Training Data)                  │
├─────────────────────────────────────────────────────────────┤
│  AI Model Providers                                        │
│  ├── Ollama (Local Model Serving)                         │
│  ├── HuggingFace Transformers                             │
│  └── Custom Trained Models                                 │
└─────────────────────────────────────────────────────────────┘
```

### 🛠️ Technology Stack

- **Backend Framework**: FastAPI with async/await support
- **Database**: PostgreSQL with pgvector for vector search
- **Caching**: Redis for high-performance caching and sessions
- **AI/ML**: HuggingFace Transformers, Ollama, PyTorch
- **Training**: Unsloth, LoRA/QLoRA, Custom pipelines
- **Multi-Modal**: Whisper (STT), Coqui TTS, LLaVA, Stable Diffusion

### 📋 Prerequisites

#### Hardware Requirements
- **GPU**: RTX 4060 Ti 16GB VRAM (or similar)
- **RAM**: 32GB+ recommended
- **Storage**: 500GB+ free space for models

#### Software Requirements
- **Python**: 3.11+
- **PostgreSQL**: 15+
- **Redis**: 7+
- **Ollama**: Latest version
- **CUDA**: 12.0+ (for GPU acceleration)

### 🚀 Quick Start

#### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-backend-hub

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Database Setup

```bash
# Install PostgreSQL with pgvector
# Ubuntu/Debian:
sudo apt-get install postgresql-15 postgresql-15-pgvector

# Windows: Download from https://www.postgresql.org/download/windows/

# Create database
createdb ai_hub

# Run migrations
alembic upgrade head
```

#### 3. Redis Setup

```bash
# Install Redis
# Ubuntu/Debian:
sudo apt-get install redis-server

# Windows: Download from https://redis.io/download

# Start Redis
redis-server
```

#### 4. Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull some models
ollama pull llama2:7b
ollama pull codellama:7b
ollama pull llava:7b
```

#### 5. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# - Database connection string
# - Redis URL
# - API keys
# - Model preferences
```

#### 6. Start the Application

```bash
# Development mode
python main.py

# Or with uvicorn directly
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at:
- **Main API**: http://127.0.0.1:8000
- **Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

### 📚 API Documentation

#### Core Endpoints

##### Chat Completions (OpenAI Compatible)
```bash
POST /api/v1/chat/completions
```
- Drop-in replacement for OpenAI Chat API
- Supports streaming responses
- Intelligent model selection
- Conversation history management

##### Model Management
```bash
GET /api/v1/models              # List available models
POST /api/v1/models/{name}/load # Load specific model
POST /api/v1/models/switch      # Switch active model
GET /api/v1/models/status/system # System status
```

##### Training Pipeline
```bash
POST /api/v1/training/jobs      # Create training job
GET /api/v1/training/jobs       # List training jobs
GET /api/v1/training/jobs/{id}  # Get job status
POST /api/v1/training/upload-dataset # Upload training data
```

##### Multi-Modal Processing
```bash
POST /api/v1/vision/analyze     # Image analysis
POST /api/v1/audio/transcribe   # Speech-to-text
POST /api/v1/audio/synthesize   # Text-to-speech
POST /api/v1/documents/process  # Document processing
```

### 🔧 Configuration

#### Model Configuration
```python
# .env file
DEFAULT_MODEL=llama2:7b
CODE_MODEL=codellama:7b
VISION_MODEL=llava:7b
MAX_VRAM_GB=14.0
CONCURRENT_MODELS=2
```

#### Training Configuration
```python
# Default LoRA settings
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
MAX_TRAINING_JOBS=2
```

### 🎯 Usage Examples

#### 1. Chat Completion
```python
import requests

response = requests.post('http://localhost:8000/api/v1/chat/completions', json={
    "model": "auto",
    "messages": [
        {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7,
    "stream": False
})

print(response.json())
```

#### 2. Model Switching
```python
# Switch to code model for programming tasks
response = requests.post('http://localhost:8000/api/v1/models/switch', json={
    "model_name": "codellama:7b"
})
```

#### 3. Training Custom Model
```python
# Create training job
response = requests.post('http://localhost:8000/api/v1/training/jobs', json={
    "name": "my-custom-model",
    "base_model": "llama2:7b",
    "epochs": 3,
    "learning_rate": 5e-4,
    "dataset_format": "alpaca"
})

job_id = response.json()["job_id"]

# Monitor progress
status = requests.get(f'http://localhost:8000/api/v1/training/jobs/{job_id}')
print(status.json())
```

### 🔍 Monitoring & Analytics

#### System Health
- Real-time VRAM usage monitoring
- Model performance metrics
- Request/response analytics
- Error tracking and alerting

#### Performance Optimization
- Automatic model unloading for memory management
- Intelligent model selection based on task type
- Response time optimization
- Resource usage analytics

### 🛡️ Security Features

- **Local-First**: No data leaves your infrastructure
- **API Authentication**: JWT tokens and API keys
- **Rate Limiting**: Configurable request limits
- **Input Validation**: Comprehensive request sanitization
- **Audit Logging**: Complete operation tracking

### 🔄 Development Workflow

#### Adding New Models
1. Add model to Ollama: `ollama pull model-name`
2. Model auto-discovery will detect it
3. Configure specialties and languages in database
4. Test with model switching API

#### Custom Training
1. Prepare dataset in supported format (Alpaca, ShareGPT)
2. Upload via training API
3. Configure training parameters
4. Monitor job progress
5. Deploy trained adapter

#### Extending APIs
1. Create new endpoint in `src/api/v1/endpoints/`
2. Add to router in `src/api/v1/__init__.py`
3. Implement business logic in `src/core/`
4. Update schemas in `src/schemas/`

### 🐛 Troubleshooting

#### Common Issues

**Model Loading Fails**
```bash
# Check VRAM usage
GET /api/v1/models/status/system

# Check Ollama connection
ollama list

# Check logs
tail -f logs/ai_hub.log
```

**Training Job Stuck**
```bash
# Check job status
GET /api/v1/training/jobs/{job_id}

# Cancel if needed
POST /api/v1/training/jobs/{job_id}/cancel

# Check system resources
htop
nvidia-smi
```

**Database Connection Issues**
```bash
# Test connection
psql -h localhost -U postgres -d ai_hub

# Check logs
tail -f logs/errors.log
```

### 📈 Performance Tuning

#### GPU Optimization
- Adjust `MAX_VRAM_GB` based on your GPU memory
- Use quantized models (Q4, Q8) for memory efficiency
- Limit `CONCURRENT_MODELS` based on VRAM capacity

#### Training Optimization
- Use gradient accumulation for larger effective batch sizes
- Adjust learning rate based on dataset size
- Monitor validation loss to prevent overfitting

### 🚀 Production Deployment

#### Docker Deployment
```bash
# Build container
docker build -t ai-backend-hub .

# Run with docker-compose
docker-compose up -d
```

#### Kubernetes Deployment
```bash
# Apply manifests
kubectl apply -f k8s/

# Monitor deployment
kubectl get pods -n ai-hub
```

### 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- **Ollama** for local model serving
- **HuggingFace** for transformer models and tools
- **Unsloth** for efficient training
- **FastAPI** for the excellent web framework
- **PostgreSQL** and **Redis** for reliable data storage

### 📞 Support

- **Documentation**: [Link to docs]
- **Issues**: [GitHub Issues]
- **Discussions**: [GitHub Discussions]
- **Email**: your-email@domain.com

---

Built with ❤️ for the AI community. Designed for privacy, performance, and extensibility.
