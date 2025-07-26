# Sử dụng base image của NVIDIA để hỗ trợ GPU, thay vì python:3.11-slim
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Đặt các biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Cài đặt Python 3.11 và các gói hệ thống cần thiết
# Gộp các lệnh RUN để giảm số layer và tối ưu build
# ✅ Cách tối ưu hơn (không cần PPA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    curl \
    git \
    build-essential \
    libpq-dev \
    postgresql-client \
    redis-tools \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn của ứng dụng
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p \
    models \
    hf_cache \
    training_data \
    trained_models \
    uploads \
    logs \
    training_jobs

# === CẢI TIẾN BẢO MẬT: Tạo và sử dụng user không phải root ===
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app

# Chuyển sang user mới
USER appuser

# Expose port 8000
EXPOSE 8000

# Health check (giữ nguyên, đã rất tốt)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Lệnh khởi chạy ứng dụng
CMD ["python3.11", "main.py"]