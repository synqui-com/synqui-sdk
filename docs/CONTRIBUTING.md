# Contributing to CognitionFlow SDK

Thank you for your interest in contributing to the CognitionFlow Python SDK! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@cognitionflow.com.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda
- Basic understanding of Python development

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/cognitionflow-python.git
   cd cognitionflow-python
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/cognitionflow/cognitionflow-python.git
   ```

## Development Setup

### 1. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n cognitionflow python=3.11
conda activate cognitionflow
```

### 2. Install Development Dependencies

```bash
cd sdk
pip install -e ".[dev,monitoring]"
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

### 4. Run Tests

```bash
make test
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Report bugs and issues
2. **Feature Requests**: Suggest new features
3. **Code Contributions**: Fix bugs or implement features
4. **Documentation**: Improve documentation
5. **Examples**: Add usage examples
6. **Tests**: Add or improve tests

### Before You Start

1. Check existing issues and pull requests
2. Open an issue to discuss significant changes
3. Ensure your changes align with the project goals
4. Follow the coding standards and style guidelines

### Development Workflow

1. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make Changes**: Implement your changes following the guidelines below

3. **Test Your Changes**:
   ```bash
   make test
   make lint
   make type-check
   ```

4. **Commit Your Changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

5. **Push and Create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Process

### Before Submitting

- [ ] Code follows the style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Changelog is updated (if applicable)
- [ ] Commit messages follow conventional commits

### PR Description

Include the following in your PR description:

1. **Summary**: Brief description of changes
2. **Type**: Bug fix, feature, documentation, etc.
3. **Testing**: How you tested the changes
4. **Breaking Changes**: Any breaking changes
5. **Related Issues**: Link to related issues

### Review Process

1. Automated checks must pass
2. Code review by maintainers
3. Address feedback and make requested changes
4. Maintainer approval and merge

## Code Style

### Python Style

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black standard)
- **Import Order**: isort configuration
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all public functions

### Formatting

We use Black for code formatting and isort for import sorting:

```bash
make format  # Format code with Black and isort
```

### Type Hints

All public functions must have type hints:

```python
def process_data(data: Dict[str, Any]) -> List[str]:
    """Process data and return results."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def process_data(data: Dict[str, Any], options: Optional[Dict] = None) -> List[str]:
    """Process data according to the given options.
    
    Args:
        data: Input data to process
        options: Optional processing options
        
    Returns:
        List of processed results
        
    Raises:
        ValueError: If data is invalid
    """
    pass
```

## Testing

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test performance characteristics
- **Error Tests**: Test error handling and edge cases

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_sdk.py

# Run with coverage
make test-cov

# Run performance tests
pytest tests/test_performance.py
```

### Writing Tests

Follow these guidelines:

1. **Test Naming**: Use descriptive test names
2. **Test Structure**: Arrange, Act, Assert pattern
3. **Mocking**: Mock external dependencies
4. **Coverage**: Aim for >90% code coverage
5. **Async Tests**: Use `pytest.mark.asyncio` for async tests

Example test:

```python
import pytest
from unittest.mock import Mock, patch

class TestSDK:
    def test_trace_decorator_success(self):
        """Test successful function tracing."""
        # Arrange
        sdk = CognitionFlowSDK(test_config)
        
        # Act
        @sdk.trace("test_agent")
        def test_function():
            return "success"
        
        result = test_function()
        
        # Assert
        assert result == "success"
        assert sdk._event_queue.qsize() == 1
```

## Documentation

### Documentation Standards

1. **API Documentation**: All public APIs must be documented
2. **Examples**: Provide usage examples for new features
3. **Type Hints**: Use type hints for better documentation
4. **Docstrings**: Comprehensive docstrings for all functions

### Documentation Structure

- **README.md**: Project overview and quick start
- **docs/GETTING_STARTED.md**: Detailed getting started guide
- **docs/API_REFERENCE.md**: Complete API reference
- **docs/CONTRIBUTING.md**: This file
- **examples/**: Usage examples

### Updating Documentation

When adding new features:

1. Update API reference
2. Add usage examples
3. Update getting started guide if needed
4. Update changelog

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update Version**: Update version in `pyproject.toml`
2. **Update Changelog**: Add release notes to `CHANGELOG.md`
3. **Create Release**: Create GitHub release
4. **Publish Package**: Publish to PyPI
5. **Update Documentation**: Update online documentation

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

Examples:
```
feat(sdk): add circuit breaker pattern
fix(batch): resolve memory leak in batch processor
docs(api): update API reference for new methods
```

## Development Tools

### Makefile Commands

```bash
make help          # Show available commands
make install       # Install package
make install-dev   # Install development dependencies
make test          # Run tests
make test-cov      # Run tests with coverage
make lint          # Run linting
make format        # Format code
make type-check    # Run type checking
make clean         # Clean build artifacts
make build         # Build package
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

- Black code formatting
- isort import sorting
- flake8 linting
- mypy type checking
- pytest testing

### IDE Configuration

#### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- isort
- GitLens

#### PyCharm

Recommended plugins:
- Black
- isort
- mypy

## Getting Help

### Resources

- **Documentation**: [https://docs.cognitionflow.com](https://docs.cognitionflow.com)
- **GitHub Issues**: [https://github.com/cognitionflow/cognitionflow-python/issues](https://github.com/cognitionflow/cognitionflow-python/issues)
- **Discussions**: [https://github.com/cognitionflow/cognitionflow-python/discussions](https://github.com/cognitionflow/cognitionflow-python/discussions)
- **Email**: dev@cognitionflow.com

### Community

- **Slack**: [CognitionFlow Community](https://cognitionflow.slack.com)
- **Discord**: [CognitionFlow Discord](https://discord.gg/cognitionflow)
- **Twitter**: [@CognitionFlow](https://twitter.com/cognitionflow)

## Recognition

Contributors will be recognized in:

- **README.md**: Contributor list
- **CHANGELOG.md**: Release notes
- **GitHub**: Contributor statistics
- **Documentation**: Contributor acknowledgments

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Thank You

Thank you for contributing to the CognitionFlow SDK! Your contributions help make the project better for everyone.
