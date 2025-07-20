<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AI Backend Hub - Copilot Instructions

## Project Overview
This is a comprehensive AI Backend Hub built with FastAPI that provides:
- Dynamic model management with intelligent switching
- Custom training pipeline with LoRA/QLoRA support
- Multi-modal processing (text, voice, images, documents)
- OpenAI-compatible API endpoints
- Local-first deployment with advanced monitoring

## Code Style & Patterns

### Python Style
- Use Python 3.11+ features including type hints and async/await
- Follow PEP 8 and use Black for formatting
- Use Pydantic models for all API request/response schemas
- Implement proper error handling with custom exceptions
- Use dependency injection for services (FastAPI Depends)

### FastAPI Patterns
- Organize endpoints in separate modules under `src/api/v1/endpoints/`
- Use APIRouter for modular routing
- Implement proper middleware for logging, security, and CORS
- Use background tasks for long-running operations
- Implement streaming responses for real-time data

### Database Patterns
- Use SQLAlchemy 2.0 with async sessions
- Implement proper database models with relationships
- Use Alembic for database migrations
- Follow proper indexing strategies for performance
- Use pgvector for vector operations

### AI/ML Integration
- Implement proper model lifecycle management
- Use intelligent caching for model metadata
- Implement proper VRAM management and monitoring
- Support multiple model providers (Ollama, HuggingFace, Local)
- Use background tasks for training jobs

## Architecture Patterns

### Service Layer
- Keep business logic in service classes
- Use dependency injection for service instances
- Implement proper error handling and logging
- Use Redis for caching and session management

### Model Management
- Implement dynamic model loading/unloading
- Use intelligent model selection based on task type
- Implement proper resource monitoring
- Support concurrent model serving when VRAM allows

### Multi-Modal Processing
- Implement separate processors for each modality
- Use streaming for large file processing
- Implement proper file validation and security
- Support batch processing for efficiency

## Security Considerations
- Always validate input data with Pydantic
- Implement proper authentication and authorization
- Use environment variables for sensitive configuration
- Implement rate limiting for API endpoints
- Log security events for audit trails

## Performance Optimization
- Use async/await for I/O operations
- Implement proper caching strategies
- Use connection pooling for databases
- Monitor resource usage and implement limits
- Use background tasks for heavy operations

## Testing Guidelines
- Write unit tests for all business logic
- Implement integration tests for API endpoints
- Use pytest with async support
- Mock external dependencies (Ollama, etc.)
- Test error conditions and edge cases

## Documentation Standards
- Use clear docstrings for all functions and classes
- Document API endpoints with proper examples
- Include type hints for better IDE support
- Write comprehensive README files
- Document configuration options

## Common Patterns to Follow

### Error Handling
```python
from fastapi import HTTPException
from loguru import logger

async def some_operation():
    try:
        # operation logic
        pass
    except SpecificException as e:
        logger.error(f"Operation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
```

### Service Dependencies
```python
from fastapi import Depends

async def get_model_manager(request: Request):
    return request.app.state.model_manager

async def endpoint(manager: ModelManager = Depends(get_model_manager)):
    # endpoint logic
    pass
```

### Async Database Operations
```python
from sqlalchemy.ext.asyncio import AsyncSession

async def create_record(db: AsyncSession, data: dict):
    async with db.begin():
        # database operations
        pass
```

## File Organization
- Keep related functionality together
- Use clear module names
- Implement proper imports
- Follow the established directory structure
- Separate concerns into appropriate layers

Remember to always consider:
- Resource management (VRAM, CPU, memory)
- Error recovery and graceful degradation
- Logging for debugging and monitoring
- Configuration flexibility
- Security best practices
- Performance implications
