# Contributing to FilmQuant ML

Thank you for your interest in contributing to the FilmQuant ML project!

## Development Setup

### Prerequisites

* Python 3.10+
* Git
* Docker (optional, for containerized development)

### Setting Up the Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/filmquant-ml.git
   cd filmquant-ml
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv-filmquant-ml
   source venv-filmquant-ml/bin/activate  # On Windows: venv-filmquant-ml\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Run the tests to make sure everything is working:**
   ```bash
   pytest
   ```

6. **Start the application:**
   ```bash
   python -m app.main
   ```

## Development Workflow

1. **Create a new branch for your feature or bugfix:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and ensure they follow the project's coding standards:**
   ```bash
   # Pre-commit will run automatically on git commit
   # But you can run it manually too:
   pre-commit run --all-files
   
   # Run tests
   pytest
   ```

3. **Commit your changes with a descriptive message:**
   ```bash
   git add .
   git commit -m "Add your descriptive message here"
   ```

4. **Push your branch to GitHub:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a pull request on GitHub.**

## Continuous Integration

This project uses GitHub Actions for continuous integration. The pipeline automatically runs:

1. Pre-commit hooks:
   - Code formatting with black
   - Import sorting with isort
   - Linting with flake8
   - Type checking with mypy
   - Docstring coverage with interrogate
   - Various file checks (trailing whitespace, YAML validity, etc.)

2. Unit tests with pytest

3. Code coverage reporting

4. Docker build

The pipeline runs automatically when you push to the main branch or open a pull request. The pre-commit hooks that run in CI are the same ones that run locally when you commit code.

## Code Structure

* `app/` - Main application code
  * `api/` - API routes and endpoints
  * `data_ingestion/` - Data preprocessing and ingestion pipeline
  * `evaluation/` - Model evaluation framework
  * `ml/` - Machine learning models and prediction logic
  * `utils/` - Utility functions and helpers
* `config/` - Configuration files
* `data/` - Sample data and model files
* `tests/` - Unit and integration tests

## Best Practices

1. Write tests for new functionality
2. Keep dependencies up to date
3. Follow PEP 8 style guidelines
4. Document your code with docstrings
5. Update the README.md when necessary
