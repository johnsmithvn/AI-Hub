# 📁 Local Models Directory

Thư mục này chứa tất cả AI models mà bạn đã tải và copy thủ công vào hệ thống.

## 🗂️ Cấu trúc thư mục:

```
local_models/
├── chat_models/          # Models cho chat general
│   ├── llama2-7b/       # Ví dụ: Llama2 7B
│   ├── mistral-7b/      # Ví dụ: Mistral 7B
│   └── qwen-7b/         # Ví dụ: Qwen 7B (Vietnamese support)
├── code_models/          # Models cho code generation
│   ├── codellama-7b/    # Ví dụ: CodeLlama 7B
│   ├── deepseek-coder/  # Ví dụ: DeepSeek Coder
│   └── starcoder/       # Ví dụ: StarCoder
├── vietnamese_models/    # Models tối ưu cho tiếng Việt
│   ├── viallama/        # Custom Vietnamese Llama
│   ├── phobert/         # PhoBERT Vietnamese
│   └── vimistral/       # Vietnamese Mistral
└── custom_models/        # Models custom của bạn
    ├── my_finetuned_llama/
    ├── domain_specific_model/
    └── experimental_model/
```

## 📥 Cách thêm model mới:

### 1. Copy model vào đúng thư mục:
```bash
# Ví dụ: Copy Llama2 7B vào chat_models
cp -r /path/to/llama2-7b-hf/ local_models/chat_models/llama2-7b/

# Ví dụ: Copy CodeLlama vào code_models  
cp -r /path/to/codellama-7b-hf/ local_models/code_models/codellama-7b/
```

### 2. Cấu trúc bên trong mỗi model folder:
```
model_name/
├── config.json          # Model configuration (REQUIRED)
├── tokenizer.json       # Tokenizer configuration
├── tokenizer_config.json
├── pytorch_model.bin    # Model weights (hoặc .safetensors)
├── pytorch_model.bin.index.json
├── generation_config.json
└── special_tokens_map.json
```

### 3. Verify model sau khi copy:
Hệ thống sẽ tự động scan và detect models mới khi restart.

## 🎯 Model Types và Use Cases:

### Chat Models (chat_models/)
- **Llama2**: General purpose conversation
- **Mistral**: Fast and efficient chat
- **Qwen**: Multilingual support (Vietnamese)
- **Vicuna**: Instruction-following

### Code Models (code_models/)
- **CodeLlama**: Python, JavaScript, general coding
- **DeepSeek Coder**: Advanced code generation
- **StarCoder**: Multiple programming languages
- **WizardCoder**: Code explanation and debugging

### Vietnamese Models (vietnamese_models/)
- **ViLlama**: Vietnamese fine-tuned Llama
- **PhoBERT**: Vietnamese BERT
- **mBERT**: Multilingual BERT with Vietnamese
- **Custom Vietnamese**: Your own trained models

### Custom Models (custom_models/)
- Your fine-tuned models
- Domain-specific models
- Experimental models
- LoRA adapters

## 🔧 Supported Model Formats:

- ✅ **HuggingFace Transformers** (.bin, .safetensors)
- ✅ **PyTorch** native models
- ✅ **GGML/GGUF** (with conversion)
- ✅ **ONNX** models
- ✅ **TensorFlow** (with conversion)

## 🎮 Cách sử dụng:

### 1. Load model qua API:
```bash
curl -X POST "http://localhost:8000/api/v1/models/load" \
-H "Content-Type: application/json" \
-d '{
  "model_name": "llama2-7b",
  "quantization": "4bit"
}'
```

### 2. Chat với model:
```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "llama2-7b",
  "messages": [{"role": "user", "content": "Xin chào!"}]
}'
```

### 3. Code generation:
```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "codellama-7b", 
  "messages": [{"role": "user", "content": "Write a Python function to sort array"}]
}'
```

## 🎓 Training Custom Models:

Bạn có thể fine-tune bất kỳ model nào trong thư mục này:

```bash
curl -X POST "http://localhost:8000/api/v1/training/start" \
-H "Content-Type: application/json" \
-d '{
  "base_model": "llama2-7b",
  "dataset_path": "training_data/my_vietnamese_data.json",
  "output_dir": "custom_models/my_vietnamese_llama",
  "training_config": {
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "lora_r": 16
  }
}'
```

## 💡 Tips & Best Practices:

1. **VRAM Management**: RTX 4060 Ti 16GB có thể chạy:
   - 1x 13B model (4bit quantization)
   - 2x 7B models đồng thời
   - 3-4x models nhỏ hơn

2. **Model Naming**: Sử dụng tên descriptive:
   - `llama2-7b-chat-vietnamese`
   - `codellama-7b-python-specialist`
   - `mistral-7b-instruct-v0.2`

3. **Quantization**: 
   - 4bit: Tiết kiệm VRAM nhất, quality tốt
   - 8bit: Balance giữa speed và quality
   - 16bit: Best quality, tốn VRAM nhất

4. **Storage**: Model size trung bình:
   - 7B: ~13GB (full) / ~4GB (4bit)
   - 13B: ~25GB (full) / ~7GB (4bit)
   - 30B+: Cần external storage

## 🔍 Troubleshooting:

### Model không được detect:
1. Check `config.json` có tồn tại
2. Check folder structure đúng
3. Restart application
4. Check logs: `tail -f logs/app.log`

### Out of memory:
1. Sử dụng quantization (4bit/8bit)
2. Unload models khác
3. Giảm max_length trong generation
4. Check GPU memory: `nvidia-smi`

### Slow generation:
1. Check GPU utilization
2. Tối ưu generation parameters
3. Sử dụng smaller models cho dev/test
4. Enable torch.compile() (Python 3.11+)
