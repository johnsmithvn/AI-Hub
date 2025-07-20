# 🤝 **Contributing to AI Backend Hub**

## 🎯 **Welcome Contributors!**

We're excited that you're interested in contributing to AI Backend Hub! This document provides guidelines for contributing to make the process smooth and effective.

---

## 📋 **Ways to Contribute**

### **🐛 Bug Reports**
- **Search existing issues** first to avoid duplicates
- **Use the bug report template** when creating new issues
- **Provide detailed reproduction steps**
- **Include system information** (OS, Python version, GPU details)

### **✨ Feature Requests**
- **Check roadmap** to see if feature is planned
- **Discuss in GitHub Discussions** before implementing
- **Provide clear use cases** and benefits
- **Consider backward compatibility**

### **📝 Documentation**
- **Fix typos and improve clarity**
- **Add missing examples**
- **Translate to Vietnamese**
- **Update outdated information**

### **💻 Code Contributions**
- **Bug fixes** - Small, focused fixes
- **New features** - Major functionality additions
- **Performance improvements** - Optimization work
- **Test coverage** - Add missing tests

---

## 🚀 **Getting Started**

### **1. Fork và Clone**

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/your-username/ai-backend-hub.git
cd ai-backend-hub

# Add upstream remote
git remote add upstream https://github.com/johnsmithvn/ai-backend-hub.git
```

### **2. Development Setup**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install pytest black isort flake8 mypy pre-commit

# Setup pre-commit hooks
pre-commit install
```

### **3. Create Development Branch**

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

---

## 📋 **Development Guidelines**

### **🐍 Python Code Style**

**Follow project coding standards:**

```python
# Use type hints
async def process_model(model_name: str) -> ModelInfo:
    """Process model với proper typing."""
    pass

# Use Pydantic for validation
from pydantic import BaseModel

class ModelConfig(BaseModel):
    name: str
    max_tokens: int = 4096
    temperature: float = 0.7

# Proper error handling
try:
    result = await some_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise HTTPException(status_code=400, detail=str(e))
```

**Code formatting:**
```bash
# Format code before committing
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### **📝 Commit Message Format**

```bash
# Format: type(scope): description
# Types: feat, fix, docs, style, refactor, test, chore

# Examples:
git commit -m "feat(models): add support for Mistral 8x7B"
git commit -m "fix(api): resolve CORS issue for streaming responses"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(training): add LoRA training integration tests"
```

### **🧪 Testing Requirements**

**All code must include tests:**

```python
# tests/test_models.py
import pytest
from src.services.model_manager import ModelManager

@pytest.mark.asyncio
async def test_model_loading():
    """Test model loading functionality."""
    manager = ModelManager()
    
    # Test successful loading
    result = await manager.load_model("test-model")
    assert result.success is True
    assert result.model_name == "test-model"
    
    # Test error handling
    with pytest.raises(ModelNotFoundError):
        await manager.load_model("nonexistent-model")

# Run tests
pytest tests/ -v --cov=src
```

### **📚 Documentation Standards**

**Update documentation when needed:**

```markdown
# API changes require documentation updates
# Add examples for new features
# Update README if installation changes
# Include Vietnamese translations when possible
```

---

## 🔄 **Pull Request Process**

### **1. Before Submitting**

**Checklist:**
- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts with main

```bash
# Test everything locally
pytest tests/ -v
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy src/
```

### **2. Submit Pull Request**

**PR Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature affecting existing functionality)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### **3. Review Process**

**What to expect:**
- **Automated checks** run first (CI/CD pipeline)
- **Code review** by maintainers
- **Feedback addressed** in additional commits
- **Approval** và merge by maintainers

---

## 🎨 **Project Structure**

```
ai-backend-hub/
├── src/                    # Main application code
│   ├── api/               # API endpoints
│   ├── core/              # Core configuration
│   ├── models/            # Database models
│   └── services/          # Business logic
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── conftest.py       # Test configuration
├── docs/                  # Documentation
├── alembic/              # Database migrations
├── docker-compose.yml    # Development environment
└── requirements.txt      # Dependencies
```

---

## 🌟 **Feature Development Process**

### **1. Planning Phase**
1. **Create GitHub issue** describing the feature
2. **Discuss design** in issue comments
3. **Get approval** from maintainers
4. **Create development plan**

### **2. Implementation Phase**
1. **Create feature branch**
2. **Implement core functionality**
3. **Add comprehensive tests**
4. **Update documentation**
5. **Test integration với existing features**

### **3. Review Phase**
1. **Submit pull request**
2. **Address review feedback**
3. **Update tests if needed**
4. **Final approval và merge**

---

## 🐛 **Bug Fix Process**

### **1. Reproduce the Bug**
```python
# Create minimal reproduction case
async def test_bug_reproduction():
    """Reproduce the reported bug."""
    # Steps to reproduce
    # Expected vs actual behavior
    pass
```

### **2. Fix và Test**
```python
# Fix the issue
def fixed_function():
    """Fixed implementation."""
    pass

# Add regression test
async def test_bug_fixed():
    """Ensure bug doesn't reoccur."""
    pass
```

### **3. Verify Fix**
- **Run full test suite**
- **Test manually** if needed
- **Check for side effects**

---

## 📊 **Quality Standards**

### **Code Quality Metrics**
- **Test Coverage**: >90% for new code
- **Type Hints**: Required for all public APIs
- **Documentation**: Docstrings for all public functions
- **Performance**: No regression in benchmarks

### **Review Criteria**
- **Functionality**: Does it work as intended?
- **Tests**: Adequate test coverage?
- **Documentation**: Clear và accurate?
- **Performance**: No negative impact?
- **Security**: No security vulnerabilities?

---

## 🌍 **Community**

### **💬 Communication Channels**
- **GitHub Issues**: Bug reports và feature requests
- **GitHub Discussions**: General questions và discussions
- **Pull Request Reviews**: Code-specific discussions

### **🤝 Code of Conduct**
- **Be respectful** và professional
- **Provide constructive feedback**
- **Help newcomers** get started
- **Focus on the code**, not the person

### **🏆 Recognition**
- **Contributors list** in README
- **Release notes** mention significant contributions
- **Community recognition** for outstanding help

---

## 📞 **Getting Help**

### **🆘 Stuck? Need Help?**

1. **Check existing documentation**
2. **Search GitHub issues**
3. **Ask in GitHub Discussions**
4. **Tag relevant maintainers** if urgent

### **📧 Contact Maintainers**
- **GitHub**: @johnsmithvn
- **Email**: For security issues only
- **Vietnamese Support**: Hỗ trợ tiếng Việt available

---

## 🎉 **Thank You!**

**Every contribution makes AI Backend Hub better for everyone!**

Whether you're fixing a typo, adding a feature, or helping other users, your contribution is valuable và appreciated.

**Happy coding!** 🚀✨
