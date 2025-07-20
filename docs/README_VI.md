# 🤖 AI Backend Hub - Hướng Dẫn Sử Dụng Hoàn Chỉnh

## 📋 Mục Lục
1. [Tổng Quan Hệ Thống](#tổng-quan-hệ-thống)
2. [Kiến Trúc Tổng Thể](#kiến-trúc-tổng-thể)
3. [Các Bug Đã Sửa](#các-bug-đã-sửa)
4. [Cách Thức Hoạt Động](#cách-thức-hoạt-động)
5. [Hướng Dẫn Cài Đặt](#hướng-dẫn-cài-đặt)
6. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
7. [API Documentation](#api-documentation)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Tổng Quan Hệ Thống

**AI Backend Hub** là một hệ thống backend hoàn chỉnh để xây dựng các ứng dụng AI tiên tiến với khả năng:

### ✨ Tính Năng Chính
- **🔄 Quản lý Model Động**: Tự động chuyển đổi giữa các AI models dựa trên yêu cầu
- **🎮 Multi-Modal**: Hỗ trợ text, hình ảnh, âm thanh và video
- **🔧 Custom Training**: Fine-tune models với dữ liệu riêng (LoRA/QLoRA)
- **⚡ Performance Optimization**: Tối ưu cho RTX 4060 Ti 16GB
- **📊 Real-time Monitoring**: Theo dõi hiệu suất và tài nguyên
- **🔒 Production Ready**: Bảo mật, scalable và reliable

---

## 🏗️ Kiến Trúc Tổng Thể

```
AI Backend Hub
├── 🌐 FastAPI Web Framework
├── 🗄️ PostgreSQL + pgvector (Vector Database)
├── 🔴 Redis (Caching & Sessions)
├── 🤖 Ollama (Local LLM Serving)
├── 🧠 HuggingFace Integration
├── 🐳 Docker Container Support
└── 📊 Prometheus Monitoring
```

### 📁 Cấu Trúc Project
```
AI hub/
├── src/                        # Source code chính
│   ├── core/                   # Dịch vụ cốt lõi
│   │   ├── config.py          # Cấu hình hệ thống
│   │   ├── database.py        # Kết nối database
│   │   ├── model_manager.py   # Quản lý AI models
│   │   └── redis_client.py    # Kết nối Redis
│   ├── models/                # Database models
│   │   ├── user.py           # User management
│   │   ├── conversation.py   # Chat conversations
│   │   ├── training.py       # Training jobs
│   │   └── analytics.py      # System analytics
│   ├── schemas/              # API schemas
│   │   ├── chat.py          # Chat API schemas
│   │   └── models.py        # Model API schemas
│   └── api/                 # API endpoints
│       └── v1/endpoints/    # Version 1 API
│           ├── chat.py      # Chat endpoints
│           ├── models.py    # Model management
│           ├── training.py  # Training endpoints
│           └── health.py    # Health checks
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Docker configuration
└── main.py                 # Application entry point
```

---

## 🔧 Các Bug Đã Sửa

### 1. **Database Import Issues**
- **Lỗi**: Wildcard imports và type hints không đúng
- **Sửa**: Chuyển sang explicit imports và AsyncGenerator type hints
- **File**: `src/core/database.py`

### 2. **Missing Imports**
- **Lỗi**: Thiếu `asyncio`, `torch`, `GPUtil` imports
- **Sửa**: Thêm missing imports và cài đặt packages thiếu
- **File**: `src/api/v1/endpoints/chat.py`, `requirements.txt`

### 3. **Model Structure Issues**
- **Lỗi**: SQLAlchemy reserved keyword `metadata`
- **Sửa**: Đổi tên thành `extra_data`
- **File**: `src/models/conversation.py`, `src/models/analytics.py`

### 4. **Import Path Conflicts**
- **Lỗi**: Relative import issues trong test files
- **Sửa**: Sử dụng absolute imports và đặt PYTHONPATH đúng
- **File**: Tất cả model files

### 5. **Missing Methods**
- **Lỗi**: ModelManager thiếu method `generate_response`
- **Sửa**: Thêm method hoàn chỉnh với Ollama và HuggingFace support
- **File**: `src/core/model_manager.py`

---

## ⚙️ Cách Thức Hoạt Động

### 🔄 Luồng Xử Lý Chat
```
User Request → FastAPI → Model Manager → AI Model → Response
     ↓
Database (Conversation History) ← Redis (Caching)
```

#### Chi Tiết:
1. **Request Processing**: FastAPI nhận request từ client
2. **Model Selection**: ModelManager chọn model phù hợp nhất
3. **Resource Management**: Kiểm tra VRAM và load/unload models
4. **Generation**: Gọi Ollama hoặc HuggingFace để generate response
5. **Caching**: Lưu vào Redis để tăng tốc độ
6. **Database**: Lưu conversation history vào PostgreSQL
7. **Monitoring**: Ghi nhận metrics vào Analytics

### 🧠 Intelligent Model Management
```python
# ModelManager tự động:
1. Phân tích yêu cầu (text, code, multimodal)
2. Chọn model tối ưu dựa trên:
   - Task type (coding, chat, vision)
   - Available VRAM
   - Model performance history
   - User preferences
3. Load/unload models để tối ưu memory
4. Cache frequently used models
```

### 📊 Performance Monitoring
- **Real-time Metrics**: CPU, GPU, Memory usage
- **Response Time Tracking**: Theo dõi latency của từng model
- **Error Rate Monitoring**: Phát hiện và xử lý lỗi
- **Usage Analytics**: Thống kê sử dụng của users

---

## 🚀 Hướng Dẫn Cài Đặt

### Bước 1: Cài Đặt Dependencies
```bash
# 1. Cài đặt PostgreSQL với pgvector
# Windows: Download từ https://www.postgresql.org/download/windows/
# Sau khi cài xong, enable pgvector extension

# 2. Cài đặt Redis
# Windows: Download từ https://redis.io/download

# 3. Cài đặt Ollama
# Download từ https://ollama.ai/
```

### Bước 2: Khởi Động Dự Án
```powershell
# Di chuyển vào thư mục project
cd "d:\DEV\All-Project\AI\AI hub"

# Kích hoạt virtual environment
& "D:/DEV/All-Project/AI/AI hub/.venv/Scripts/Activate.ps1"

# Chạy database migrations
alembic upgrade head

# Pull các AI models
ollama pull llama2:7b
ollama pull codellama:7b
ollama pull llava:7b

# Khởi động ứng dụng
python main.py
```

### Bước 3: Xác Minh Hoạt Động
```powershell
# Kiểm tra health của hệ thống
curl http://localhost:8000/health

# Xem API documentation
# Mở browser: http://localhost:8000/docs
```

---

## 📖 Hướng Dẫn Sử Dụng

### 💬 Chat API

#### Gửi tin nhắn đơn giản:
```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Xin chào! Bạn có thể giúp tôi code Python không?"}
  ],
  "model": "codellama:7b",
  "stream": false
}'
```

#### Streaming Response:
```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Viết một hàm Python để sắp xếp mảng"}
  ],
  "model": "codellama:7b",
  "stream": true
}'
```

### 🤖 Model Management

#### Xem danh sách models:
```bash
curl http://localhost:8000/api/v1/models
```

#### Load một model:
```bash
curl -X POST "http://localhost:8000/api/v1/models/load" \
-H "Content-Type: application/json" \
-d '{
  "model_id": "llama2:7b",
  "force_reload": false
}'
```

#### Kiểm tra status models:
```bash
curl http://localhost:8000/api/v1/models/status
```

### 🎓 Custom Training

#### Bắt đầu training job:
```bash
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
-H "Content-Type: application/json" \
-d '{
  "name": "My Custom Model",
  "base_model": "llama2:7b",
  "dataset_path": "/path/to/training/data.json",
  "training_type": "lora",
  "learning_rate": 2e-4,
  "num_epochs": 3
}'
```

#### Theo dõi training progress:
```bash
curl http://localhost:8000/api/v1/training/jobs/{job_id}/status
```

---

## 📚 API Documentation

### Chat Endpoints
- `POST /api/v1/chat/completions` - Gửi tin nhắn chat
- `GET /api/v1/chat/conversations` - Lấy danh sách conversations
- `DELETE /api/v1/chat/conversations/{id}` - Xóa conversation

### Model Endpoints  
- `GET /api/v1/models` - Danh sách tất cả models
- `POST /api/v1/models/load` - Load model vào memory
- `POST /api/v1/models/unload` - Unload model khỏi memory
- `GET /api/v1/models/status` - Trạng thái system và models

### Training Endpoints
- `POST /api/v1/training/jobs` - Tạo training job mới
- `GET /api/v1/training/jobs` - Danh sách training jobs
- `GET /api/v1/training/jobs/{id}` - Chi tiết training job
- `DELETE /api/v1/training/jobs/{id}` - Hủy training job

### Health & Monitoring
- `GET /health` - Health check tổng thể
- `GET /api/v1/system/metrics` - System metrics
- `GET /api/v1/system/analytics` - Usage analytics

---

## 🛠️ Troubleshooting

### Vấn Đề Thường Gặp

#### 1. Models không load được
```bash
# Kiểm tra Ollama service
ollama list

# Restart Ollama
ollama serve

# Kiểm tra VRAM
nvidia-smi
```

#### 2. Database connection lỗi
```bash
# Kiểm tra PostgreSQL
pg_isready -h localhost -p 5432

# Kiểm tra pgvector extension
psql -d ai_hub -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### 3. Redis connection issues
```bash
# Kiểm tra Redis
redis-cli ping

# Restart Redis service
net start redis
```

#### 4. Out of Memory (OOM)
```python
# Trong config, giảm số models được load cùng lúc
MAX_CONCURRENT_MODELS = 1
MODEL_UNLOAD_TIMEOUT = 300  # 5 minutes
```

### Logs và Debugging
```bash
# Xem logs real-time
tail -f logs/app.log

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor system resources
htop
```

---

## 🎉 Kết Luận

**AI Backend Hub** giờ đây đã sẵn sàng để phục vụ các ứng dụng AI production-level với:

✅ **Tất cả bugs đã được fix**
✅ **Architecture hoàn chỉnh và scalable**  
✅ **Resource optimization cho RTX 4060 Ti**
✅ **Multi-modal AI capabilities**
✅ **Production monitoring và logging**
✅ **Comprehensive API documentation**

### 🚀 Next Steps
1. **Deploy lên production** với Docker/Kubernetes
2. **Thêm authentication & authorization**
3. **Integrate với frontend applications**
4. **Scale horizontal với load balancing**
5. **Add more AI models và capabilities**

**Happy Coding! 🎊**
