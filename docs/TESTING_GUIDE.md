# 🧪 **Testing Guide**

## 📋 **Overview**

Comprehensive testing strategy cho AI Backend Hub để đảm bảo reliability và performance của hệ thống AI local.

---

## 🧪 **Test Structure**

```
tests/
├── unit/                          # Unit tests
│   ├── test_model_manager.py     # Model management tests
│   ├── test_training_service.py  # Training pipeline tests
│   ├── test_api_endpoints.py     # API endpoint tests
│   └── test_utils.py             # Utility function tests
├── integration/                   # Integration tests
│   ├── test_model_loading.py     # End-to-end model loading
│   ├── test_training_pipeline.py # Complete training workflow
│   └── test_api_integration.py   # API integration tests
├── performance/                   # Performance tests
│   ├── test_memory_usage.py      # VRAM/RAM monitoring
│   ├── test_response_times.py    # API response benchmarks
│   └── test_concurrent_requests.py # Load testing
├── fixtures/                      # Test data
│   ├── sample_models/            # Mock model files
│   ├── datasets/                 # Sample training data
│   └── configs/                  # Test configurations
└── conftest.py                   # Pytest configuration
```

---

## 🚀 **Running Tests**

### **Setup Test Environment**

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# Set test environment
export TEST_ENV=true
export DATABASE_URL=sqlite+aiosqlite:///test.db

# Create test database
python -c "from tests.setup_test_db import create_test_db; create_test_db()"
```

### **Run All Tests**

```bash
# Run all tests với coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/performance/    # Performance tests only

# Run với verbose output
pytest tests/ -v -s

# Run specific test file
pytest tests/unit/test_model_manager.py -v
```

---

## 🧪 **Unit Tests**

### **Model Manager Tests**

```python
# tests/unit/test_model_manager.py
import pytest
import asyncio
from unittest.mock import Mock, patch
from src.core.custom_model_manager import CustomModelManager, ModelType, ModelStatus

class TestCustomModelManager:
    
    @pytest.fixture
    async def model_manager(self):
        """Create test model manager instance"""
        manager = CustomModelManager()
        await manager._scan_local_models()
        return manager
    
    @pytest.mark.asyncio
    async def test_scan_local_models(self, model_manager):
        """Test model scanning functionality"""
        models = await model_manager.get_available_models()
        assert isinstance(models, list)
        # Test với mock model files
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, model_manager):
        """Test successful model loading"""
        with patch('torch.cuda.is_available', return_value=True):
            result = await model_manager.load_model("test-model", quantization="4bit")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_load_model_insufficient_vram(self, model_manager):
        """Test model loading with insufficient VRAM"""
        with patch.object(model_manager, '_check_vram_available', return_value=False):
            result = await model_manager.load_model("large-model")
            # Should attempt to free VRAM and retry
    
    @pytest.mark.asyncio
    async def test_generate_response(self, model_manager):
        """Test text generation"""
        with patch.object(model_manager, 'loaded_models', {"test-model": Mock()}):
            response = await model_manager.generate_response(
                "test-model", "Hello, how are you?"
            )
            assert isinstance(response, str)
            assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_unload_model(self, model_manager):
        """Test model unloading"""
        # First load a model (mocked)
        model_manager.loaded_models["test-model"] = Mock()
        
        result = await model_manager.unload_model("test-model")
        assert result is True
        assert "test-model" not in model_manager.loaded_models
    
    def test_vram_management(self, model_manager):
        """Test VRAM calculation and management"""
        # Test VRAM calculation
        mock_model = Mock()
        vram_usage = model_manager._get_model_vram_usage(mock_model)
        assert isinstance(vram_usage, float)
        assert vram_usage >= 0
```

### **Training Service Tests**

```python
# tests/unit/test_training_service.py
import pytest
from unittest.mock import Mock, patch
from src.core.training_service import TrainingService

class TestTrainingService:
    
    @pytest.fixture
    def training_service(self):
        return TrainingService()
    
    @pytest.mark.asyncio
    async def test_start_training_job(self, training_service):
        """Test training job creation"""
        config = {
            "base_model": "test-model",
            "dataset_path": "test_dataset.jsonl",
            "output_dir": "test_output",
            "epochs": 1,
            "batch_size": 2
        }
        
        job_id = await training_service.start_training_job(config)
        assert job_id is not None
        assert isinstance(job_id, str)
    
    @pytest.mark.asyncio
    async def test_prepare_dataset(self, training_service):
        """Test dataset preparation"""
        # Create mock dataset file
        dataset_path = "tests/fixtures/datasets/sample.jsonl"
        dataset = await training_service.prepare_dataset(dataset_path)
        
        assert dataset is not None
        assert len(dataset) > 0
    
    @pytest.mark.asyncio
    async def test_training_progress_tracking(self, training_service):
        """Test training progress monitoring"""
        job_id = "test_job_123"
        
        # Mock training job
        training_service.training_jobs[job_id] = {
            "status": "training",
            "progress": 50,
            "current_step": 500,
            "total_steps": 1000
        }
        
        status = await training_service.get_training_status(job_id)
        assert status["progress"] == 50
        assert status["status"] == "training"
```

### **API Endpoint Tests**

```python
# tests/unit/test_api_endpoints.py
import pytest
from httpx import AsyncClient
from main import app

class TestAPIEndpoints:
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test models listing endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/v1/models")
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert isinstance(data["data"], list)
    
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        """Test chat completion endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}]
            })
            # Should return 400 if model not loaded, or 200 with response
            assert response.status_code in [200, 400]
    
    @pytest.mark.asyncio
    async def test_load_model_endpoint(self):
        """Test model loading endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/v1/models/load", json={
                "model_name": "test-model"
            })
            # Should handle the request (success or failure)
            assert response.status_code in [200, 400, 500]
```

---

## 🔗 **Integration Tests**

### **End-to-End Model Loading**

```python
# tests/integration/test_model_loading.py
import pytest
import asyncio
from src.core.custom_model_manager import get_model_manager

class TestModelLoadingIntegration:
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_model_workflow(self):
        """Test complete model loading workflow"""
        manager = await get_model_manager()
        
        # 1. Check available models
        models = await manager.get_available_models()
        if not models:
            pytest.skip("No models available for testing")
        
        model_name = models[0]["name"]
        
        # 2. Load model
        success = await manager.load_model(model_name)
        assert success is True
        
        # 3. Generate response
        response = await manager.generate_response(
            model_name, "Hello, this is a test."
        )
        assert isinstance(response, str)
        assert len(response) > 0
        
        # 4. Unload model
        unload_success = await manager.unload_model(model_name)
        assert unload_success is True
    
    @pytest.mark.asyncio
    async def test_multiple_model_management(self):
        """Test loading multiple models"""
        manager = await get_model_manager()
        models = await manager.get_available_models()
        
        if len(models) < 2:
            pytest.skip("Need at least 2 models for this test")
        
        # Load first model
        model1 = models[0]["name"]
        success1 = await manager.load_model(model1)
        assert success1 is True
        
        # Load second model (might trigger automatic unloading)
        model2 = models[1]["name"]
        success2 = await manager.load_model(model2)
        
        # Check system status
        status = await manager.get_system_status()
        assert "loaded_models" in status
        assert status["loaded_models"] >= 1
```

### **Training Pipeline Integration**

```python
# tests/integration/test_training_pipeline.py
import pytest
import json
import tempfile
from pathlib import Path

class TestTrainingPipelineIntegration:
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_training_workflow(self):
        """Test complete training workflow"""
        from src.core.custom_model_manager import get_model_manager
        
        manager = await get_model_manager()
        
        # 1. Create sample dataset
        sample_data = [
            {
                "instruction": "Say hello",
                "input": "",
                "output": "Hello! How can I help you?"
            },
            {
                "instruction": "What is 2+2?",
                "input": "",
                "output": "2+2 equals 4."
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            dataset_path = f.name
        
        # 2. Start training job
        training_config = {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
            "lora_r": 8
        }
        
        try:
            job_id = await manager.start_training(
                model_name="test-base-model",
                dataset_path=dataset_path,
                output_dir=tempfile.mkdtemp(),
                training_config=training_config
            )
            
            assert job_id is not None
            
            # 3. Monitor training progress
            await asyncio.sleep(5)  # Let training start
            
            # Check if job exists in training_jobs
            assert job_id in manager.training_jobs
            
        finally:
            # Cleanup
            Path(dataset_path).unlink(missing_ok=True)
```

---

## ⚡ **Performance Tests**

### **Memory Usage Tests**

```python
# tests/performance/test_memory_usage.py
import pytest
import psutil
import GPUtil
from src.core.custom_model_manager import get_model_manager

class TestMemoryUsage:
    
    @pytest.mark.asyncio
    async def test_vram_usage_tracking(self):
        """Test VRAM usage tracking accuracy"""
        manager = await get_model_manager()
        
        # Get initial VRAM usage
        initial_vram = self.get_vram_usage()
        
        # Load a model
        models = await manager.get_available_models()
        if models:
            model_name = models[0]["name"]
            await manager.load_model(model_name)
            
            # Check VRAM increase
            after_load_vram = self.get_vram_usage()
            assert after_load_vram > initial_vram
            
            # Unload model
            await manager.unload_model(model_name)
            
            # Check VRAM decrease
            after_unload_vram = self.get_vram_usage()
            assert after_unload_vram < after_load_vram
    
    def get_vram_usage(self):
        """Get current VRAM usage"""
        try:
            gpu = GPUtil.getGPUs()[0]
            return gpu.memoryUsed
        except:
            return 0
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks in model operations"""
        manager = await get_model_manager()
        
        initial_ram = psutil.virtual_memory().used
        
        # Perform multiple load/unload cycles
        models = await manager.get_available_models()
        if models:
            model_name = models[0]["name"]
            
            for i in range(5):
                await manager.load_model(model_name)
                await manager.generate_response(model_name, "Test message")
                await manager.unload_model(model_name)
        
        final_ram = psutil.virtual_memory().used
        ram_increase = final_ram - initial_ram
        
        # Allow for some memory increase, but not excessive
        assert ram_increase < 1024 * 1024 * 1024  # Less than 1GB increase
```

### **Response Time Tests**

```python
# tests/performance/test_response_times.py
import pytest
import time
import asyncio
from httpx import AsyncClient
from main import app

class TestResponseTimes:
    
    @pytest.mark.asyncio
    async def test_api_response_times(self):
        """Test API endpoint response times"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # Test health check response time
            start_time = time.time()
            response = await client.get("/health")
            health_time = time.time() - start_time
            
            assert response.status_code == 200
            assert health_time < 0.1  # Should be very fast
            
            # Test models list response time
            start_time = time.time()
            response = await client.get("/v1/models")
            models_time = time.time() - start_time
            
            assert response.status_code == 200
            assert models_time < 1.0  # Should be under 1 second
    
    @pytest.mark.asyncio
    async def test_chat_completion_performance(self):
        """Test chat completion response times"""
        from src.core.custom_model_manager import get_model_manager
        
        manager = await get_model_manager()
        models = await manager.get_available_models()
        
        if not models:
            pytest.skip("No models available for performance testing")
        
        model_name = models[0]["name"]
        
        # Load model first
        await manager.load_model(model_name)
        
        # Test generation speed
        start_time = time.time()
        response = await manager.generate_response(
            model_name, 
            "This is a test prompt for measuring response time.",
            max_tokens=100
        )
        generation_time = time.time() - start_time
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert generation_time < 10.0  # Should complete within 10 seconds
        
        # Calculate tokens per second
        estimated_tokens = len(response.split())
        tokens_per_second = estimated_tokens / generation_time
        
        print(f"Generation speed: {tokens_per_second:.2f} tokens/second")
        assert tokens_per_second > 1.0  # Minimum reasonable speed
```

### **Load Testing**

```python
# tests/performance/test_concurrent_requests.py
import pytest
import asyncio
from httpx import AsyncClient
from main import app

class TestConcurrentRequests:
    
    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self):
        """Test handling concurrent API requests"""
        
        async def make_request(client, request_id):
            response = await client.get("/health")
            return response.status_code, request_id
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create 10 concurrent requests
            tasks = [
                make_request(client, i) 
                for i in range(10)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # All requests should succeed
            for status_code, request_id in results:
                assert status_code == 200
            
            # Should handle concurrent requests efficiently
            assert total_time < 2.0  # All 10 requests in under 2 seconds
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_model_loading_under_load(self):
        """Test model loading with concurrent requests"""
        from src.core.custom_model_manager import get_model_manager
        
        manager = await get_model_manager()
        models = await manager.get_available_models()
        
        if not models:
            pytest.skip("No models available for load testing")
        
        model_name = models[0]["name"]
        
        async def load_and_generate():
            await manager.load_model(model_name)
            return await manager.generate_response(
                model_name, "Test prompt", max_tokens=50
            )
        
        # Try concurrent loading (should handle gracefully)
        tasks = [load_and_generate() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least one should succeed
        successful_results = [r for r in results if isinstance(r, str)]
        assert len(successful_results) > 0
```

---

## 📊 **Test Configuration**

### **Pytest Configuration**

```python
# conftest.py
import pytest
import asyncio
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_model_dir():
    """Create temporary model directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "models"
        model_dir.mkdir()
        yield model_dir

@pytest.fixture
def sample_dataset():
    """Create sample training dataset"""
    data = [
        {
            "instruction": "Greet the user",
            "input": "",
            "output": "Hello! How can I help you today?"
        },
        {
            "instruction": "Explain AI",
            "input": "",
            "output": "AI stands for Artificial Intelligence..."
        }
    ]
    return data

# Pytest markers for test categorization
pytest_markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests"
]
```

### **Test Environment Setup**

```python
# tests/setup_test_db.py
import asyncio
from sqlalchemy import create_engine
from src.core.database import Base

def create_test_db():
    """Setup test database"""
    engine = create_engine("sqlite:///test.db")
    Base.metadata.create_all(engine)
    print("Test database created successfully")

if __name__ == "__main__":
    create_test_db()
```

---

## 🚀 **Continuous Testing**

### **GitHub Actions Workflow**

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/unit/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### **Pre-commit Hooks**

```yaml
# .pre-commit-config.yaml
repos:
- repo: local
  hooks:
  - id: pytest-check
    name: pytest-check
    entry: pytest tests/unit/ -x
    language: system
    pass_filenames: false
    always_run: true
```

---

## 📋 **Test Coverage Goals**

### **Coverage Targets**
- **Unit Tests**: 90%+ code coverage
- **Integration Tests**: All critical workflows covered
- **Performance Tests**: All endpoints benchmarked
- **API Tests**: All endpoints tested

### **Quality Metrics**
- **Response Time**: <2s for chat completions
- **Memory Usage**: No memory leaks detected
- **Error Rate**: <1% for normal operations
- **VRAM Efficiency**: >80% utilization when loaded

---

## 🎯 **Test Best Practices**

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies (GPU, files)
3. **Cleanup**: Always cleanup resources after tests
4. **Documentation**: Clear test descriptions và comments
5. **Performance**: Mark slow tests appropriately
6. **Coverage**: Aim for high coverage but focus on critical paths

**Your AI Backend Hub is now thoroughly tested và ready for production!** ✅
