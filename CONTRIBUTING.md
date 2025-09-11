# Contributing to Yellow Rust Detection System

We welcome contributions to the Yellow Rust Detection System! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Operating system and version
   - Python version
   - PyTorch version
   - CUDA version (if applicable)
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Error messages and stack traces

### Suggesting Features

1. **Check existing feature requests** to avoid duplicates
2. **Describe the feature** clearly and concisely
3. **Explain the motivation** and use case
4. **Consider implementation** complexity and impact

### Code Contributions

#### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/yellow-rust-detection.git
   cd yellow-rust-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

#### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Run linting
   flake8 src/ tests/
   black --check src/ tests/
   isort --check-only src/ tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide

- **Follow PEP 8** for Python code style
- **Use Black** for code formatting (line length: 88 characters)
- **Use isort** for import sorting
- **Use type hints** where appropriate
- **Write docstrings** for all public functions and classes

### Code Organization

```
src/
├── models/          # Model architectures and training
├── data/           # Data loading and preprocessing
├── utils/          # Utility functions
└── __init__.py
```

### Documentation

- **Use Google-style docstrings**
- **Include examples** in docstrings when helpful
- **Update README.md** for significant changes
- **Add inline comments** for complex logic

### Example Docstring

```python
def predict_image(image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Predict rust segmentation for a single image.
    
    Args:
        image_path: Path to the input image file.
        threshold: Confidence threshold for segmentation (0.0-1.0).
        
    Returns:
        Dictionary containing:
            - 'rust_percentage': Percentage of rust coverage
            - 'severity_level': Severity classification
            - 'mask': Binary segmentation mask
            - 'probability_map': Probability heatmap
            
    Raises:
        FileNotFoundError: If image file doesn't exist.
        ValueError: If threshold is not in valid range.
        
    Example:
        >>> results = predict_image('wheat_leaf.jpg', threshold=0.3)
        >>> print(f"Rust coverage: {results['rust_percentage']:.2f}%")
    """
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_models/     # Model testing
├── test_data/       # Data processing tests
├── test_utils/      # Utility function tests
├── fixtures/        # Test data and fixtures
└── conftest.py      # Pytest configuration
```

### Writing Tests

1. **Use pytest** for testing framework
2. **Write unit tests** for individual functions
3. **Write integration tests** for workflows
4. **Use fixtures** for test data
5. **Mock external dependencies** when appropriate

### Example Test

```python
import pytest
from src.models.unet import UNet

def test_unet_forward_pass():
    """Test UNet forward pass with valid input."""
    model = UNet(encoder_name='resnet34', classes=1)
    input_tensor = torch.randn(1, 3, 256, 256)
    
    output = model(input_tensor)
    
    assert output.shape == (1, 1, 256, 256)
    assert torch.all(output >= 0) and torch.all(output <= 1)
```

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Description Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Approval** from at least one maintainer
5. **Merge** after approval

## 🏷️ Commit Message Guidelines

Use conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

### Examples
```
feat(models): add ResNet50 encoder support
fix(inference): resolve CUDA memory leak
docs(readme): update installation instructions
test(models): add unit tests for UNet architecture
```

## 🐛 Bug Reports

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. Windows 10, Ubuntu 20.04]
 - Python version: [e.g. 3.8.10]
 - PyTorch version: [e.g. 1.9.0]
 - CUDA version: [e.g. 11.1]

**Additional context**
Any other context about the problem.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## 📋 Development Workflow

### Branch Naming
- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/update-description` - Documentation updates
- `refactor/component-name` - Code refactoring

### Release Process

1. **Version bumping** following semantic versioning
2. **Changelog update** with new features and fixes
3. **Tag creation** for releases
4. **GitHub release** with release notes

## 🎯 Areas for Contribution

### High Priority
- [ ] Model architecture improvements
- [ ] Performance optimizations
- [ ] Additional data augmentation techniques
- [ ] Mobile/edge deployment support

### Medium Priority
- [ ] Additional evaluation metrics
- [ ] Visualization improvements
- [ ] Documentation enhancements
- [ ] Test coverage improvements

### Low Priority
- [ ] Code style improvements
- [ ] Minor bug fixes
- [ ] Example notebooks
- [ ] Tutorial content

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: [maintainer-email@example.com]

## 🙏 Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes
- Credited in academic publications (if applicable)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Yellow Rust Detection System! 🌾