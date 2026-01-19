# Contributing to Cattle Disease Detection System

First off, thank you for considering contributing to the Cattle Disease Detection System! It's people like you that make this project better for everyone.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to support@cattlehealth.ai.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Screenshots** if applicable
- **Environment details** (OS, Python version, browser, etc.)
- **Error messages** and logs

**Bug Report Template:**
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
 - OS: [e.g. Ubuntu 20.04]
 - Python Version: [e.g. 3.9]
 - Browser: [e.g. Chrome 96]

**Additional context**
Any other context about the problem.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case** - why is this enhancement needed?
- **Proposed solution** - how should it work?
- **Alternatives considered**
- **Additional context** - mockups, examples, etc.

**Enhancement Template:**
```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions or features you've considered.

**Additional context**
Mockups, examples, or other context.
```

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `documentation` - Documentation improvements

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our style guidelines
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request**

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool

### Setup Steps

1. **Fork and clone**
```bash
git clone https://github.com/YOUR_USERNAME/cattle-disease-detector.git
cd cattle-disease-detector
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add model files**
- Place your model in `models/` directory
- Update configuration files

5. **Run application**
```bash
python app.py
```

6. **Run tests** (if available)
```bash
pytest
```

### Project Structure
```
cattle_disease_app/
├── app.py              # Main application
├── models/             # ML models
├── static/             # CSS, JS, images
├── templates/          # HTML templates
└── tests/              # Test files
```

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated
- [ ] All tests pass

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing done.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] No new warnings
- [ ] Added tests
- [ ] All tests pass

## Screenshots (if applicable)
Add screenshots here.
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in staging environment
4. **Approval** from at least one maintainer
5. **Merge** to main branch

## Style Guidelines

### Python Code Style

Follow **PEP 8** guidelines:

```python
# Good
def predict_disease(image_path):
    """
    Predict disease from cattle image.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Prediction results with confidence
    """
    # Implementation
    pass

# Bad
def predictDisease(imagePath):
    # No docstring
    pass
```

**Key points:**
- Use 4 spaces for indentation
- Max line length: 88 characters (Black formatter)
- Use descriptive variable names
- Add docstrings to functions
- Type hints where appropriate

### HTML/CSS Style

```html
<!-- Good -->
<div class="container">
    <h1 class="title">Cattle Health AI</h1>
    <p class="description">AI-powered detection</p>
</div>

<!-- Bad -->
<div class=container>
<h1>Cattle Health AI</h1>
<p>AI-powered detection</p></div>
```

**Key points:**
- Use semantic HTML
- Consistent indentation (2 spaces)
- BEM naming for CSS classes
- Mobile-first approach

### JavaScript Style

```javascript
// Good
const uploadImage = async (file) => {
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    } catch (error) {
        console.error('Upload failed:', error);
        throw error;
    }
};

// Bad
function uploadImage(file) {
    var formData = new FormData()
    formData.append('file',file)
    fetch('/upload',{method:'POST',body:formData})
}
```

**Key points:**
- Use ES6+ features
- Async/await for promises
- Proper error handling
- Descriptive names

### Commit Messages

Follow **Conventional Commits**:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(upload): add drag-and-drop support

Added drag-and-drop functionality to upload page for better UX.
Includes visual feedback and file validation.

Closes #123
```

```
fix(model): resolve prediction accuracy issue

Fixed tensor shape mismatch causing incorrect predictions.
Updated preprocessing pipeline to match training.

Fixes #456
```

## Testing

### Writing Tests

```python
import pytest
from app import app

def test_home_page():
    """Test home page loads successfully"""
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'Cattle Health AI' in response.data

def test_upload_no_file():
    """Test upload without file returns error"""
    client = app.test_client()
    response = client.post('/upload')
    assert response.status_code == 400
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_app.py::test_home_page

# Run with coverage
pytest --cov=app tests/
```

## Documentation

### Code Documentation

```python
def predict_image(image_path: str) -> dict:
    """
    Predict disease from cattle image using ViT model.
    
    This function loads an image, preprocesses it, runs inference
    using the trained Vision Transformer model, and returns
    predictions with confidence scores.
    
    Args:
        image_path (str): Absolute or relative path to image file.
                         Supported formats: JPG, PNG, GIF
    
    Returns:
        dict: Dictionary containing:
            - prediction (str): Predicted disease class
            - confidence (float): Confidence score (0-100)
            - all_probabilities (dict): All class probabilities
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is unsupported
        
    Example:
        >>> result = predict_image('cattle.jpg')
        >>> print(result['prediction'])
        'Healthy'
    """
    # Implementation
```

### README Updates

When adding features, update:
- Feature list
- Usage instructions
- API documentation
- Screenshots/GIFs

## Community

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and general discussion
- **Email**: support@cattlehealth.ai

### Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Project documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Cattle Health AI! 🐄❤️
