# Face Recognition Service - Unit Tests

This directory contains comprehensive unit tests for the face recognition microservice.

## Test Structure

### Test Files
- `test_app.py` - Tests for Flask API endpoints
- `test_handlers.py` - Tests for face recognition handlers
- `test_models.py` - Tests for request/response models
- `test_helpers.py` - Tests for utility functions
- `test_redis_db.py` - Tests for Redis database operations
- `test_retinaface.py` - Tests for face detection/recognition engine

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Mock Tests**: Use mocks for external dependencies (Redis, ML models)

## Running Tests

### Run All Tests
```bash
# Using the test runner script
python tests/run_tests.py

# Using pytest (if installed)
pytest

# Using unittest
python -m unittest discover -s tests -p "test_*.py"
```

### Run Specific Test Module
```bash
# Using test runner
python tests/run_tests.py test_app

# Using pytest
pytest tests/test_app.py

# Using unittest
python -m unittest tests.test_app
```

### Run Specific Test Class
```bash
pytest tests/test_app.py::TestFaceRecognitionAPI

# Or with unittest
python -m unittest tests.test_app.TestFaceRecognitionAPI
```

### Run Specific Test Method
```bash
pytest tests/test_app.py::TestFaceRecognitionAPI::test_add_identity_success

# Or with unittest  
python -m unittest tests.test_app.TestFaceRecognitionAPI.test_add_identity_success
```

## Test Coverage

### Generate Coverage Report
```bash
# Install coverage first
pip install coverage

# Run tests with coverage
coverage run -m pytest
coverage report
coverage html  # Generates HTML report in htmlcov/
```

### Coverage Targets
- Overall coverage: >80%
- Critical modules (app.py, handlers.py): >90%
- Model classes: >85%

## Test Configuration

### Environment Setup
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Or install individual packages
pip install pytest pytest-cov pytest-mock
```

### Mock Dependencies
The tests use mocks for:
- Redis connections
- ML model loading (PyTorch, etc.)
- External API calls
- File system operations

### Test Data
- Uses synthetic test images (base64 encoded)
- Mock face encodings
- Sample request/response data

## Test Guidelines

### Writing New Tests
1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies
3. **Coverage**: Aim for high test coverage
4. **Documentation**: Include docstrings for test methods
5. **Naming**: Use descriptive test method names

### Test Structure
```python
def test_method_name_scenario(self):
    """Test description explaining what is being tested"""
    # Arrange - Set up test data
    # Act - Execute the code under test  
    # Assert - Verify the results
```

### Common Patterns
```python
# Mocking external dependencies
@patch('module.external_dependency')
def test_with_mock(self, mock_dependency):
    mock_dependency.return_value = expected_value
    # Test code

# Testing exceptions
with self.assertRaises(SpecificException):
    code_that_should_raise_exception()

# Parameterized tests
@pytest.mark.parametrize("input,expected", [
    ("input1", "expected1"),
    ("input2", "expected2"),
])
def test_parameterized(self, input, expected):
    assert function(input) == expected
```

## Continuous Integration

### CI Pipeline Integration
```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install -r tests/requirements-test.txt
    python tests/run_tests.py
    
- name: Upload Coverage
  run: |
    coverage run -m pytest
    coverage xml
```

## Debugging Tests

### Common Issues
1. **Import Errors**: Check Python path and module imports
2. **Mock Issues**: Verify mock paths and return values
3. **Test Isolation**: Ensure tests don't depend on each other
4. **Resource Cleanup**: Clean up resources in tearDown methods

### Debug Commands
```bash
# Run tests with verbose output
pytest -v -s

# Run tests with debugging
pytest --pdb

# Run specific failing test
pytest tests/test_app.py::test_failing_method -v -s
```

## Performance Testing

### Benchmarking
```bash
# Install pytest-benchmark
pip install pytest-benchmark

# Run performance tests
pytest --benchmark-only
```

### Memory Testing
```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler tests/test_app.py
```

## Test Maintenance

### Regular Tasks
1. Update test data when APIs change
2. Review and update mocks
3. Check for deprecated test patterns
4. Maintain test coverage above thresholds
5. Update test documentation

### Refactoring Tests
- Keep tests simple and focused
- Extract common setup into fixtures
- Remove duplicate test code
- Update tests when refactoring application code

## Integration with Development Workflow

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Run tests before commit
pre-commit install
```

### Test-Driven Development (TDD)
1. Write failing test
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Repeat cycle

## Troubleshooting

### Common Test Failures
- **Redis Connection**: Ensure Redis mocks are properly configured
- **Model Loading**: Mock heavy ML model dependencies
- **File Paths**: Use absolute paths in tests
- **Environment Variables**: Set required env vars for tests

### Getting Help
- Check test logs for detailed error messages
- Use debugger to step through failing tests
- Review mock configurations
- Verify test data and assertions