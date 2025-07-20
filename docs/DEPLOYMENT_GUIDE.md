# 🔧 **Deployment Guide**

## 📋 **Overview**

Complete deployment guide cho AI Backend Hub từ development đến production environment, supporting multiple deployment strategies.

---

## 🏗️ **Deployment Architecture**

```
Production Deployment Options:

1. Single Server Deployment
   ┌─────────────────────────────────┐
   │         Single Server           │
   │  ┌─────────────────────────────┐│
   │  │     AI Backend Hub          ││
   │  │  ┌─────────┬─────────────┐  ││
   │  │  │ FastAPI │ Custom      │  ││
   │  │  │   App   │ Model Mgr   │  ││
   │  │  └─────────┴─────────────┘  ││
   │  │  ┌─────────┬─────────────┐  ││
   │  │  │ PostgreSQL │  Redis   │  ││
   │  │  └─────────┴─────────────┘  ││
   │  └─────────────────────────────┘│
   │  GPU: RTX 4060 Ti 16GB         │
   └─────────────────────────────────┘

2. Containerized Deployment
   ┌─────────────────────────────────┐
   │         Docker Host             │
   │  ┌─────────┬─────────┬───────┐  │
   │  │AI-Hub   │Database │ Redis │  │
   │  │Container│Container│ Cache │  │
   │  │         │         │       │  │
   │  └─────────┴─────────┴───────┘  │
   │  Volume Mounts: Models, Data    │
   └─────────────────────────────────┘

3. Kubernetes Deployment (Future)
   ┌─────────────────────────────────┐
   │        K8s Cluster              │
   │  ┌─────────┬─────────┬───────┐  │
   │  │AI Pods  │ DB Pod  │Cache  │  │
   │  │(Scaled) │(StatefulSet)   │  │
   │  └─────────┴─────────┴───────┘  │
   │  Persistent Volumes: Models     │
   └─────────────────────────────────┘
```

---

## 🚀 **Production Setup**

### **1. Server Requirements**

**Minimum Production Specs:**
- **CPU**: 16+ cores (Intel Xeon/AMD EPYC)
- **RAM**: 64GB+ DDR4
- **GPU**: RTX 4060 Ti 16GB+ (RTX 4090 recommended)
- **Storage**: 1TB+ NVMe SSD
- **Network**: 1Gbps+ connection

**Recommended Production Specs:**
- **CPU**: 32+ cores với hyperthreading
- **RAM**: 128GB+ DDR4/DDR5
- **GPU**: RTX 4090 24GB hoặc multiple RTX 4060 Ti
- **Storage**: 2TB+ NVMe SSD + backup storage
- **Network**: 10Gbps connection

### **2. Operating System Setup**

```bash
# Ubuntu 22.04 LTS (Recommended)
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    curl wget git vim htop \
    build-essential software-properties-common \
    apt-transport-https ca-certificates \
    gnupg lsb-release

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535
sudo reboot

# Verify GPU
nvidia-smi
```

### **3. Docker Installation** (Recommended)

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
```

---

## 🐳 **Docker Deployment**

### **1. Dockerfile**

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-pip python3.11-dev \
    postgresql-client redis-tools \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p local_models training_data trained_models logs uploads

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python3", "main.py"]
```

### **2. Docker Compose Production**

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  ai-hub:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ai-hub-prod
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://ai_user:${DB_PASSWORD}@postgres:5432/ai_hub
      - REDIS_URL=redis://redis:6379/0
      - MAX_VRAM_USAGE=0.85
      - API_HOST=0.0.0.0
      - API_PORT=8000
    volumes:
      - ./local_models:/app/local_models:ro
      - ./training_data:/app/training_data
      - ./trained_models:/app/trained_models
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - ai-hub-network

  postgres:
    image: pgvector/pgvector:pg15
    container_name: ai-hub-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=ai_hub
      - POSTGRES_USER=ai_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"
    networks:
      - ai-hub-network

  redis:
    image: redis:7-alpine
    container_name: ai-hub-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - ai-hub-network

  nginx:
    image: nginx:alpine
    container_name: ai-hub-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - ai-hub
    networks:
      - ai-hub-network

volumes:
  postgres_data:
  redis_data:

networks:
  ai-hub-network:
    driver: bridge
```

### **3. Production Environment Variables**

```bash
# .env.prod
# Database
DATABASE_URL=postgresql+asyncpg://ai_user:your_secure_password@postgres:5432/ai_hub
REDIS_URL=redis://:your_redis_password@redis:6379/0
DB_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
CORS_ORIGINS=["https://yourdomain.com","https://app.yourdomain.com"]

# Security
JWT_SECRET_KEY=your-super-secret-jwt-key-256-bit
API_RATE_LIMIT=1000
UPLOAD_MAX_SIZE=500

# AI Settings
MAX_VRAM_USAGE=0.85
DEFAULT_MODEL=llama2-7b-chat
MODEL_CACHE_SIZE=3
AUTO_OFFLOAD_THRESHOLD=0.9

# Training Settings
TRAINING_OUTPUT_DIR=/app/trained_models
TRAINING_BATCH_SIZE=4
TRAINING_LEARNING_RATE=2e-4

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
SENTRY_DSN=https://your-sentry-dsn

# File Storage
LOCAL_MODELS_DIR=/app/local_models
TRAINING_DATA_DIR=/app/training_data
UPLOAD_DIR=/app/uploads
```

### **4. Nginx Configuration**

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream ai_hub {
        server ai-hub:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

        client_max_body_size 500M;

        # API endpoints
        location /v1/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://ai_hub;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
            proxy_connect_timeout 30s;
            proxy_send_timeout 300s;
        }

        # File uploads
        location /v1/files/upload {
            limit_req zone=upload burst=5 nodelay;
            proxy_pass http://ai_hub;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_request_buffering off;
            proxy_read_timeout 600s;
        }

        # Health check
        location /health {
            proxy_pass http://ai_hub;
            access_log off;
        }

        # Documentation
        location /docs {
            proxy_pass http://ai_hub;
            proxy_set_header Host $host;
        }
    }
}
```

---

## 🔧 **Systemd Service** (Non-Docker)

### **1. Service File**

```ini
# /etc/systemd/system/ai-hub.service
[Unit]
Description=AI Backend Hub
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=ai-hub
Group=ai-hub
WorkingDirectory=/opt/ai-hub
Environment=PATH=/opt/ai-hub/.venv/bin
EnvironmentFile=/opt/ai-hub/.env
ExecStart=/opt/ai-hub/.venv/bin/python main.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ai-hub

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

[Install]
WantedBy=multi-user.target
```

### **2. Service Management**

```bash
# Install service
sudo cp ai-hub.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-hub

# Start service
sudo systemctl start ai-hub

# Check status
sudo systemctl status ai-hub

# View logs
journalctl -u ai-hub -f

# Restart service
sudo systemctl restart ai-hub
```

---

## 📊 **Monitoring & Logging**

### **1. Prometheus Configuration**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai-hub'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9445']
```

### **2. Grafana Dashboard**

```json
{
  "dashboard": {
    "title": "AI Backend Hub Monitoring",
    "panels": [
      {
        "title": "GPU Utilization",
        "type": "stat",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu"
          }
        ]
      },
      {
        "title": "VRAM Usage",
        "type": "gauge",
        "targets": [
          {
            "expr": "nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes * 100"
          }
        ]
      },
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

### **3. Log Management**

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  ai-hub:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  loki:
    image: grafana/loki:2.8.0
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:2.8.0
    volumes:
      - /var/log:/var/log:ro
      - ./promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml
```

---

## 🔒 **Security Configuration**

### **1. SSL/TLS Setup**

```bash
# Generate SSL certificate với Let's Encrypt
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### **2. Firewall Configuration**

```bash
# UFW firewall setup
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Allow specific API access (optional)
sudo ufw allow from YOUR_CLIENT_IP to any port 8000
```

### **3. Security Headers**

```nginx
# Add to nginx.conf
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline';";
```

---

## 📈 **Scaling Strategy**

### **1. Vertical Scaling**

```yaml
# Increase resources
deploy:
  resources:
    limits:
      cpus: '16.0'
      memory: 64G
    reservations:
      cpus: '8.0'
      memory: 32G
      devices:
        - driver: nvidia
          count: 2
          capabilities: [gpu]
```

### **2. Horizontal Scaling** (Future)

```yaml
# docker-compose.scale.yml
services:
  ai-hub:
    scale: 3
    
  nginx:
    depends_on:
      - ai-hub
    # Load balancing configuration
```

### **3. Model Sharding**

```python
# config/scaling.py
MODEL_SHARDING_CONFIG = {
    "gpu_0": ["llama2-7b-chat", "mistral-7b"],
    "gpu_1": ["codellama-7b", "qwen-7b"],
    "gpu_2": ["custom-models"]
}
```

---

## 🔄 **Backup & Recovery**

### **1. Database Backup**

```bash
#!/bin/bash
# backup_database.sh

DB_NAME="ai_hub"
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker exec ai-hub-postgres pg_dump -U ai_user $DB_NAME > "$BACKUP_DIR/db_backup_$DATE.sql"

# Compress backup
gzip "$BACKUP_DIR/db_backup_$DATE.sql"

# Keep only last 7 days
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete

echo "Database backup completed: db_backup_$DATE.sql.gz"
```

### **2. Model Backup**

```bash
#!/bin/bash
# backup_models.sh

MODEL_DIR="/opt/ai-hub/local_models"
BACKUP_DIR="/opt/backups/models"
DATE=$(date +%Y%m%d_%H%M%S)

# Create model backup
tar -czf "$BACKUP_DIR/models_backup_$DATE.tar.gz" -C "$MODEL_DIR" .

# Upload to cloud (optional)
# aws s3 cp "$BACKUP_DIR/models_backup_$DATE.tar.gz" s3://your-backup-bucket/

echo "Models backup completed: models_backup_$DATE.tar.gz"
```

### **3. Automated Backup**

```bash
# crontab -e
# Daily database backup at 2 AM
0 2 * * * /opt/scripts/backup_database.sh

# Weekly model backup on Sundays at 3 AM
0 3 * * 0 /opt/scripts/backup_models.sh

# Daily log cleanup
0 1 * * * find /opt/ai-hub/logs -name "*.log" -mtime +30 -delete
```

---

## 🚀 **Deployment Checklist**

### **Pre-Deployment**
- [ ] Server meets hardware requirements
- [ ] OS updated và security patches applied
- [ ] GPU drivers installed và tested
- [ ] Docker và NVIDIA Container Toolkit installed
- [ ] SSL certificates obtained
- [ ] Firewall configured
- [ ] Monitoring setup configured

### **Deployment**
- [ ] Code deployed to production server
- [ ] Environment variables configured
- [ ] Database migrations run
- [ ] Models copied to local_models directory
- [ ] Docker services started
- [ ] Health checks passing
- [ ] API endpoints responding
- [ ] SSL certificates working

### **Post-Deployment**
- [ ] Monitoring dashboards configured
- [ ] Log aggregation working
- [ ] Backup scripts scheduled
- [ ] Performance benchmarks established
- [ ] Documentation updated
- [ ] Team trained on operations

---

## 🎯 **Production Maintenance**

### **Daily Tasks**
- Monitor system health dashboards
- Check API response times
- Review error logs
- Verify backup completion

### **Weekly Tasks**
- Update security patches
- Review performance metrics
- Clean up old logs
- Test backup restoration

### **Monthly Tasks**
- Update dependencies
- Review và optimize configurations
- Capacity planning review
- Security audit

**Your AI Backend Hub is now production-ready với enterprise-grade reliability!** 🚀
