# 🌐 **AI Backend Hub - API Documentation**

## 📋 **Overview**

AI Backend Hub cung cấp OpenAI-compatible REST API cho local AI model management và inference. Tất cả endpoints đều tương thích với OpenAI API format để dễ dàng migration từ cloud services.

**Base URL**: `http://localhost:8000`
**API Version**: `v1`
**Authentication**: Bearer token (configurable)

---

## 🔐 **Authentication**

```bash
# Header format
Authorization: Bearer your-api-key

# Environment variable
export AI_HUB_API_KEY="your-api-key"
```

---

## 🤖 **Models Management**

### **List Available Models**

```http
GET /v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama2-7b-chat",
      "object": "model", 
      "created": 1642665600,
      "owned_by": "local",
      "permission": [],
      "root": "llama2-7b-chat",
      "parent": null,
      "status": "loaded",
      "size_gb": 6.7,
      "vram_usage": 4200,
      "specialties": ["general_chat", "reasoning"]
    }
  ]
}
```

### **Load Model**

```http
POST /v1/models/load
```

**Request Body:**
```json
{
  "model_name": "llama2-7b-chat",
  "quantization": "4bit",
  "max_memory": {"0": "8GB"}
}
```

**Response:**
```json
{
  "success": true,
  "model_name": "llama2-7b-chat",
  "load_time": 12.5,
  "vram_usage": 4200,
  "status": "loaded"
}
```

### **Unload Model**

```http
POST /v1/models/unload
```

**Request Body:**
```json
{
  "model_name": "llama2-7b-chat"
}
```

### **Model Status**

```http
GET /v1/models/{model_name}/status
```

**Response:**
```json
{
  "model_name": "llama2-7b-chat",
  "status": "loaded",
  "vram_usage": 4200,
  "last_used": 1642665600,
  "requests_count": 157,
  "avg_response_time": 1.2
}
```

---

## 💬 **Chat Completions** (OpenAI Compatible)

### **Create Chat Completion**

```http
POST /v1/chat/completions
```

**Request Body:**
```json
{
  "model": "llama2-7b-chat",
  "messages": [
    {
      "role": "system", 
      "content": "You are a helpful assistant."
    },
    {
      "role": "user", 
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1642665600,
  "model": "llama2-7b-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 15,
    "total_tokens": 35
  }
}
```

### **Streaming Chat Completion**

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama2-7b-chat",
  "messages": [...],
  "stream": true
}
```

**Streaming Response:**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1642665600,"model":"llama2-7b-chat","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1642665600,"model":"llama2-7b-chat","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: [DONE]
```

---

## 🎓 **Training & Fine-tuning**

### **Start Training Job**

```http
POST /v1/training/jobs
```

**Request Body:**
```json
{
  "base_model": "llama2-7b-chat",
  "dataset_path": "training_data/my_dataset.jsonl",
  "output_dir": "trained_models/my_custom_model",
  "training_type": "lora",
  "config": {
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"]
  }
}
```

**Response:**
```json
{
  "job_id": "train_123456789",
  "status": "queued",
  "base_model": "llama2-7b-chat",
  "estimated_duration": "2 hours",
  "created_at": "2024-01-01T10:00:00Z"
}
```

### **Get Training Job Status**

```http
GET /v1/training/jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "train_123456789",
  "status": "training",
  "progress": 65,
  "current_step": 650,
  "total_steps": 1000,
  "current_loss": 0.45,
  "best_loss": 0.42,
  "elapsed_time": "45 minutes",
  "estimated_remaining": "25 minutes",
  "logs": [
    "Step 650/1000: loss=0.45, lr=1.8e-4",
    "Validation loss: 0.43"
  ]
}
```

### **List Training Jobs**

```http
GET /v1/training/jobs
```

**Query Parameters:**
- `status`: filter by status (queued, training, completed, failed)
- `limit`: number of results (default: 10)
- `offset`: pagination offset

### **Cancel Training Job**

```http
DELETE /v1/training/jobs/{job_id}
```

---

## 📊 **System Monitoring**

### **System Status**

```http
GET /v1/system/status
```

**Response:**
```json
{
  "gpu": {
    "name": "NVIDIA GeForce RTX 4060 Ti",
    "memory_total": 16380,
    "memory_used": 8200,
    "memory_free": 8180,
    "utilization": 75,
    "temperature": 68
  },
  "cpu": {
    "usage": 45,
    "memory": 68,
    "cores": 16
  },
  "models": {
    "loaded_count": 2,
    "total_count": 5,
    "active_model": "llama2-7b-chat"
  },
  "training": {
    "active_jobs": 1,
    "queued_jobs": 0,
    "completed_jobs": 3
  },
  "api": {
    "requests_per_minute": 45,
    "avg_response_time": 1.2,
    "uptime": "2 days, 14 hours"
  }
}
```

### **Performance Metrics**

```http
GET /v1/system/metrics
```

**Response:**
```json
{
  "timestamp": "2024-01-01T10:00:00Z",
  "model_performance": {
    "llama2-7b-chat": {
      "requests_count": 1000,
      "avg_response_time": 1.2,
      "tokens_per_second": 45,
      "error_rate": 0.02
    }
  },
  "resource_usage": {
    "vram_usage_history": [6400, 6800, 7200, 8200],
    "cpu_usage_history": [30, 35, 40, 45],
    "request_volume_history": [20, 25, 30, 45]
  }
}
```

---

## 📁 **File Management**

### **Upload Training Data**

```http
POST /v1/files/upload
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: training dataset file (.jsonl, .csv, .txt)
- `purpose`: "training" or "validation"
- `description`: optional description

**Response:**
```json
{
  "file_id": "file_123456",
  "filename": "my_dataset.jsonl",
  "size": 1024000,
  "purpose": "training",
  "status": "uploaded",
  "created_at": "2024-01-01T10:00:00Z"
}
```

### **List Files**

```http
GET /v1/files
```

### **Delete File**

```http
DELETE /v1/files/{file_id}
```

---

## 🔧 **Configuration**

### **Get Configuration**

```http
GET /v1/config
```

**Response:**
```json
{
  "max_vram_usage": 0.85,
  "default_model": "llama2-7b-chat",
  "auto_offload_threshold": 0.9,
  "training_defaults": {
    "batch_size": 4,
    "learning_rate": 2e-4,
    "max_steps": 1000
  },
  "api_settings": {
    "rate_limit": 100,
    "max_tokens": 4096,
    "timeout": 30
  }
}
```

### **Update Configuration**

```http
PATCH /v1/config
```

**Request Body:**
```json
{
  "max_vram_usage": 0.8,
  "default_model": "codellama-7b-instruct"
}
```

---

## 💡 **Usage Examples**

### **Python Client Example**

```python
import openai

# Configure client
client = openai.AsyncOpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"
)

# Chat completion
async def chat_example():
    response = await client.chat.completions.create(
        model="llama2-7b-chat",
        messages=[
            {"role": "user", "content": "Explain quantum computing"}
        ]
    )
    return response.choices[0].message.content

# Custom training
import httpx

async def train_model():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/training/jobs",
            json={
                "base_model": "llama2-7b-chat",
                "dataset_path": "my_data.jsonl",
                "config": {"epochs": 3}
            }
        )
    return response.json()
```

### **cURL Examples**

```bash
# List models
curl -X GET "http://localhost:8000/v1/models" \
  -H "Authorization: Bearer your-api-key"

# Chat completion
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "llama2-7b-chat",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Start training
curl -X POST "http://localhost:8000/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "base_model": "llama2-7b-chat",
    "dataset_path": "training_data/my_dataset.jsonl"
  }'
```

---

## ⚠️ **Error Handling**

### **Error Response Format**

```json
{
  "error": {
    "message": "Model not found",
    "type": "model_not_found",
    "code": "MODEL_404",
    "details": {
      "model_name": "nonexistent-model",
      "available_models": ["llama2-7b-chat", "codellama-7b"]
    }
  }
}
```

### **Common Error Codes**

| Code | Description | Solution |
|------|-------------|----------|
| `MODEL_404` | Model not found | Check available models with `/v1/models` |
| `VRAM_INSUFFICIENT` | Not enough VRAM | Unload other models or use smaller model |
| `TRAINING_FAILED` | Training job failed | Check logs and dataset format |
| `RATE_LIMITED` | Too many requests | Reduce request frequency |
| `INVALID_TOKEN` | Authentication failed | Check API key |

---

## 🚀 **Rate Limits**

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/v1/chat/completions` | 100 requests | 1 minute |
| `/v1/models/load` | 10 requests | 1 minute |
| `/v1/training/jobs` | 5 requests | 1 hour |
| Other endpoints | 200 requests | 1 minute |

---

## 📝 **Changelog**

### **v1.0.0** (Current)
- OpenAI-compatible chat completions
- Model management endpoints
- LoRA/QLoRA training pipeline
- System monitoring
- File management

### **Upcoming Features**
- Function calling support
- Multi-modal endpoints (vision, audio)
- Model marketplace integration
- Advanced analytics dashboard
