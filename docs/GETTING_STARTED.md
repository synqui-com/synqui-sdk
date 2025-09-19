# Getting Started with CognitionFlow SDK

## Overview

The CognitionFlow Python SDK provides comprehensive observability and tracing capabilities for Python applications. This guide will help you get started with the SDK in just a few minutes.

## Installation

### From PyPI (Recommended)

```bash
pip install cognitionflow-sdk
```

### From Source

```bash
git clone https://github.com/cognitionflow/cognitionflow-python.git
cd cognitionflow-python
pip install -e .
```

## Quick Start

### 1. Configure the SDK

```python
import cognitionflow

# Basic configuration
cognitionflow.configure(
    api_key="your-api-key-here",
    project_id="your-project-id-here"
)
```

### 2. Start Tracing

```python
# Simple function tracing
@cognitionflow.trace(agent_name="my_agent")
def process_data(input_data):
    # Your function logic here
    result = transform_data(input_data)
    return result

# Async function tracing
@cognitionflow.trace(agent_name="api_client")
async def fetch_data(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### 3. Manual Span Creation

```python
# Manual span creation for complex workflows
async with cognitionflow.span("complex_operation") as span:
    span.set_attribute("operation_type", "batch_processing")
    span.set_attribute("batch_size", len(data))
    
    # Your code here
    result = await process_batch(data)
    
    span.set_attribute("result_count", len(result))
```

## Configuration Options

### Environment Variables

You can configure the SDK using environment variables:

```bash
export COGNITIONFLOW_API_KEY="your-api-key"
export COGNITIONFLOW_PROJECT_ID="your-project-id"
export COGNITIONFLOW_ENVIRONMENT="production"
```

Then configure from environment:

```python
cognitionflow.configure_from_env()
```

### Advanced Configuration

```python
from cognitionflow import SDKConfig

config = SDKConfig(
    api_key="your-api-key",
    project_id="your-project-id",
    endpoint="https://api.cognitionflow.com",
    batch_size=100,
    flush_interval=5.0,
    max_retries=3,
    capture_inputs=True,
    capture_outputs=True,
    capture_errors=True,
    environment="production",
    debug=False
)

cognitionflow.configure_from_config(config)
```

## Common Use Cases

### 1. Function Tracing

Trace individual functions to monitor their performance and behavior:

```python
@cognitionflow.trace(agent_name="data_processor")
def process_user_data(user_id, data):
    # Process user data
    result = transform_data(data)
    return result

@cognitionflow.trace(agent_name="ml_model")
async def make_prediction(features):
    # ML model prediction
    prediction = await model.predict(features)
    return prediction
```

### 2. API Endpoint Tracing

Trace API endpoints to monitor request/response patterns:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
@cognitionflow.trace(agent_name="user_service")
async def get_user(user_id: int):
    user = await database.get_user(user_id)
    return user
```

### 3. Database Operation Tracing

Trace database operations to monitor query performance:

```python
@cognitionflow.trace(agent_name="database")
async def execute_query(query, params=None):
    with cognitionflow.span("database_query") as span:
        span.set_attribute("query_type", "select")
        span.set_attribute("table", "users")
        
        result = await database.execute(query, params)
        span.set_attribute("result_count", len(result))
        
        return result
```

### 4. Error Handling and Tracing

Trace errors to understand failure patterns:

```python
@cognitionflow.trace(agent_name="risky_operation")
def risky_operation(data):
    try:
        result = process_data(data)
        return result
    except Exception as e:
        # Error is automatically captured by the SDK
        raise
```

### 5. Batch Processing

Trace batch operations to monitor processing efficiency:

```python
@cognitionflow.trace(agent_name="batch_processor")
async def process_batch(items):
    async with cognitionflow.span("batch_processing") as span:
        span.set_attribute("batch_size", len(items))
        
        results = []
        for i, item in enumerate(items):
            async with cognitionflow.span(f"process_item_{i}") as item_span:
                item_span.set_attribute("item_index", i)
                result = await process_item(item)
                results.append(result)
        
        span.set_attribute("processed_count", len(results))
        return results
```

## Integration with Popular Frameworks

### FastAPI

```python
from fastapi import FastAPI
from fastapi.middleware.base import BaseHTTPMiddleware

app = FastAPI()

# Middleware for automatic request tracing
class CognitionFlowMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        with cognitionflow.span("http_request") as span:
            span.set_attribute("method", request.method)
            span.set_attribute("url", str(request.url))
            
            response = await call_next(request)
            
            span.set_attribute("status_code", response.status_code)
            return response

app.add_middleware(CognitionFlowMiddleware)
```

### Django

```python
# In middleware.py
from django.utils.deprecation import MiddlewareMixin

class CognitionFlowMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request._cognitionflow_start_time = time.time()
        
    def process_response(self, request, response):
        if hasattr(request, '_cognitionflow_start_time'):
            duration = time.time() - request._cognitionflow_start_time
            
            with cognitionflow.span("django_request") as span:
                span.set_attribute("method", request.method)
                span.set_attribute("path", request.path)
                span.set_attribute("status_code", response.status_code)
                span.set_attribute("duration_ms", duration * 1000)
        
        return response
```

### Flask

```python
from flask import Flask, request, g

app = Flask(__name__)

@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        
        with cognitionflow.span("flask_request") as span:
            span.set_attribute("method", request.method)
            span.set_attribute("path", request.path)
            span.set_attribute("status_code", response.status_code)
            span.set_attribute("duration_ms", duration * 1000)
    
    return response
```

## Best Practices

### 1. Performance Considerations

- Use appropriate batch sizes (default: 100)
- Disable input/output capture in production if not needed
- Use circuit breakers for external API calls
- Monitor memory usage

### 2. Security

- Never log sensitive data in traces
- Use environment variables for configuration
- Disable debug mode in production
- Validate all inputs before tracing

### 3. Error Handling

- Always handle exceptions in traced functions
- Use circuit breakers for external dependencies
- Implement proper retry logic
- Monitor SDK health and performance

### 4. Configuration

```python
# Development configuration
dev_config = SDKConfig(
    api_key="dev-api-key",
    project_id="dev-project",
    batch_size=10,
    flush_interval=2.0,
    debug=True,
    capture_inputs=True,
    capture_outputs=True
)

# Production configuration
prod_config = SDKConfig(
    api_key="prod-api-key",
    project_id="prod-project",
    batch_size=100,
    flush_interval=10.0,
    debug=False,
    capture_inputs=False,  # Disable for privacy
    capture_outputs=False  # Disable for privacy
)
```

## Troubleshooting

### Common Issues

1. **Traces not appearing**: Check your API key and project ID
2. **Performance impact**: Reduce batch size or disable input/output capture
3. **Memory usage**: Use MemoryManager to monitor and control memory
4. **Network errors**: Check your endpoint URL and network connectivity

### Debug Mode

Enable debug mode to see detailed SDK logs:

```python
cognitionflow.configure(
    api_key="your-api-key",
    project_id="your-project-id",
    debug=True
)
```

### Health Checks

Monitor SDK health:

```python
# Check if SDK is enabled
if cognitionflow.is_enabled():
    print("SDK is active")

# Manually flush traces
cognitionflow.flush()

# Shutdown SDK
cognitionflow.shutdown()
```

## Next Steps

1. **Explore Examples**: Check out the `examples/` directory for comprehensive examples
2. **Read API Reference**: See `docs/API_REFERENCE.md` for detailed API documentation
3. **Framework Integration**: Look at `examples/integration_examples.py` for framework-specific patterns
4. **Advanced Features**: Explore `examples/advanced_usage.py` for advanced patterns

## Support

- **Documentation**: [https://docs.cognitionflow.com](https://docs.cognitionflow.com)
- **GitHub**: [https://github.com/cognitionflow/cognitionflow-python](https://github.com/cognitionflow/cognitionflow-python)
- **Issues**: [https://github.com/cognitionflow/cognitionflow-python/issues](https://github.com/cognitionflow/cognitionflow-python/issues)
- **Email**: support@cognitionflow.com

## License

This SDK is licensed under the MIT License. See the LICENSE file for details.
