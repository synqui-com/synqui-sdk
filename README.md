# CognitionFlow Python SDK

[![PyPI version](https://badge.fury.io/py/cognitionflow-sdk.svg)](https://badge.fury.io/py/cognitionflow-sdk)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official Python SDK for [CognitionFlow](https://cognitionflow.com), a comprehensive observability and tracing platform for AI agents and applications.

## Features

- 🔍 **Automatic Tracing**: Instrument your functions with simple decorators
- 🚀 **Async Support**: Full support for both synchronous and asynchronous code
- 📊 **Performance Monitoring**: Built-in performance profiling and memory management
- 🔄 **Batch Processing**: Efficient data transmission with configurable batching
- 💪 **Resilient**: Circuit breaker pattern for handling API failures
- 🧵 **Context Aware**: Thread-safe context management for nested traces
- 🎛️ **Configurable**: Flexible configuration with environment variable support
- 📝 **Type Safe**: Full type annotations for better development experience

## Installation

```bash
pip install cognitionflow-sdk
```

For development with all optional dependencies:

```bash
pip install cognitionflow-sdk[all]
```

## Quick Start

### 1. Configure the SDK

```python
import cognitionflow

# Configure with your API credentials
cognitionflow.configure(
    api_key="your-api-key",
    project_id="your-project-id"
)
```

### 2. Trace Your Functions

```python
@cognitionflow.trace("data_processor")
def process_data(data):
    """Process some data."""
    result = {"processed": len(data), "items": data}
    return result

# Your function is now automatically traced
result = process_data(["item1", "item2", "item3"])
```

### 3. Async Function Support

```python
@cognitionflow.trace("async_processor")
async def async_process_data(data):
    """Async data processing."""
    await asyncio.sleep(0.1)  # Simulate async work
    return {"processed": len(data)}

# Async functions work seamlessly
result = await async_process_data(["item1", "item2"])
```

## Advanced Usage

### Manual Span Creation

```python
# Synchronous span
with cognitionflow.span("custom_operation") as span:
    span.set_attribute("user_id", "12345")
    span.set_tag("environment", "production")
    # Your code here
    result = expensive_computation()
    span.set_attribute("result_size", len(result))

# Asynchronous span
async with cognitionflow.span("async_operation") as span:
    span.set_attribute("operation_type", "ml_inference")
    result = await ml_model.predict(data)
    span.set_attribute("prediction_confidence", result.confidence)
```

### Nested Tracing

```python
@cognitionflow.trace("main_processor")
def main_process(data):
    # This creates the parent span
    preprocessed = preprocess_data(data)

    # This creates a child span
    with cognitionflow.span("validation") as span:
        span.set_attribute("data_size", len(preprocessed))
        validate_data(preprocessed)

    return postprocess_data(preprocessed)

@cognitionflow.trace("preprocessor")
def preprocess_data(data):
    # This creates another child span under main_processor
    return [item.upper() for item in data]
```

### Configuration Options

```python
from cognitionflow import SDKConfig, CognitionFlowSDK

# Detailed configuration
config = SDKConfig(
    api_key="your-api-key",
    project_id="your-project-id",
    endpoint="https://api.cognitionflow.com",

    # Batching configuration
    batch_size=100,
    flush_interval=5.0,

    # Performance tuning
    max_retries=3,
    timeout=30.0,

    # Memory management
    max_memory_mb=200,
    gc_threshold_mb=100,

    # Global tags
    tags={"environment": "production", "service": "ml-pipeline"}
)

# Create SDK instance with custom config
sdk = CognitionFlowSDK(config)

@sdk.trace("custom_agent")
def my_function():
    return "traced with custom config"
```

### Environment Variables

You can also configure the SDK using environment variables:

```bash
export COGNITIONFLOW_API_KEY="your-api-key"
export COGNITIONFLOW_PROJECT_ID="your-project-id"
export COGNITIONFLOW_ENDPOINT="https://api.cognitionflow.com"
export COGNITIONFLOW_BATCH_SIZE="50"
export COGNITIONFLOW_FLUSH_INTERVAL="10.0"
export COGNITIONFLOW_ENABLED="true"
```

```python
import cognitionflow

# Configure from environment variables
cognitionflow.configure_from_env()

@cognitionflow.trace("env_configured_agent")
def my_function():
    return "configured from environment"
```

### Error Handling

The SDK automatically captures and reports errors:

```python
@cognitionflow.trace("error_prone_agent")
def risky_operation(data):
    if not data:
        raise ValueError("Data cannot be empty")
    return process(data)

try:
    result = risky_operation([])
except ValueError as e:
    # Error is automatically captured in the trace
    print(f"Operation failed: {e}")
```

### Performance Monitoring

```python
# Check SDK performance stats
stats = cognitionflow.get_default_sdk().get_stats()
print(f"Traces sent: {stats['traces_sent']}")
print(f"Queue size: {stats['queue_size']}")
print(f"Memory usage: {stats['memory_usage_mb']} MB")

# Manual flush of pending traces
cognitionflow.flush()

# Get current span (useful for manual instrumentation)
from cognitionflow import get_current_span

current_span = get_current_span()
if current_span:
    current_span.set_attribute("manual_attribute", "value")
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | Required | Your CognitionFlow API key |
| `project_id` | str | Required | Your project identifier |
| `endpoint` | str | `"https://api.cognitionflow.com"` | API endpoint URL |
| `enabled` | bool | `True` | Enable/disable tracing |
| `batch_size` | int | `50` | Number of traces per batch |
| `flush_interval` | float | `5.0` | Seconds between batch flushes |
| `max_retries` | int | `3` | Maximum API retry attempts |
| `timeout` | float | `30.0` | Request timeout in seconds |
| `max_memory_mb` | int | `100` | Memory usage warning threshold |
| `gc_threshold_mb` | int | `50` | Garbage collection trigger threshold |
| `tags` | dict | `{}` | Global tags for all traces |

## Best Practices

### 1. Use Descriptive Agent Names

```python
# Good - describes the specific functionality
@cognitionflow.trace("user_authentication_validator")
def validate_user_credentials(username, password):
    pass

# Avoid - too generic
@cognitionflow.trace("validator")
def validate_user_credentials(username, password):
    pass
```

### 2. Add Meaningful Attributes

```python
@cognitionflow.trace("recommendation_engine")
def get_recommendations(user_id, item_count=10):
    with cognitionflow.span("feature_extraction") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("requested_count", item_count)

        features = extract_user_features(user_id)
        span.set_attribute("feature_count", len(features))

        recommendations = generate_recommendations(features, item_count)
        span.set_attribute("actual_count", len(recommendations))

        return recommendations
```

### 3. Handle Sensitive Data

```python
@cognitionflow.trace("payment_processor")
def process_payment(payment_data):
    with cognitionflow.span("payment_validation") as span:
        # Don't log sensitive information
        span.set_attribute("payment_method", payment_data["method"])
        span.set_attribute("amount", payment_data["amount"])
        # span.set_attribute("card_number", payment_data["card"])  # DON'T DO THIS

        return validate_and_charge(payment_data)
```

### 4. Use Context Managers for Resources

```python
@cognitionflow.trace("database_operation")
def fetch_user_data(user_id):
    with cognitionflow.span("db_connection") as span:
        span.set_attribute("user_id", user_id)

        with get_db_connection() as conn:
            span.set_attribute("connection_pool_size", conn.pool_size)
            result = conn.execute("SELECT * FROM users WHERE id = %s", [user_id])
            span.set_attribute("rows_returned", len(result))
            return result
```

## Development

### Setting Up Development Environment

```bash
git clone https://github.com/cognitionflow/cognitionflow-sdk
cd cognitionflow-sdk
pip install -e ".[dev,monitoring]"
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_sdk.py -v
```

### Code Quality

```bash
# Format code
make format

# Check linting
make lint

# Type checking
make type-check
```

## Examples

See the [examples](examples/) directory for complete examples including:

- Basic tracing setup
- Async/await patterns
- Nested span creation
- Error handling
- Performance monitoring
- Custom configuration

## API Reference

For detailed API documentation, visit [docs.cognitionflow.com](https://docs.cognitionflow.com).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: [support@cognitionflow.com](mailto:support@cognitionflow.com)
- 📖 Documentation: [docs.cognitionflow.com](https://docs.cognitionflow.com)
- 🐛 Issue Tracker: [GitHub Issues](https://github.com/cognitionflow/cognitionflow-sdk/issues)
- 💬 Community: [Discord](https://discord.gg/cognitionflow)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.