# Getting Started with Vaquero SDK

## 🚀 5-Minute Quick Start

Get up and running with Vaquero in just 3 simple steps:

<div class="quick-start-steps">

### 1️⃣ Install
```bash
pip install vaquero-sdk
```

### 2️⃣ Initialize
```python
import vaquero

vaquero.init(api_key="your-api-key")
```

### 3️⃣ Trace
```python
@vaquero.trace("my_agent")
def my_function(data):
    # Your code here
    return processed_data

# Done! ✨ Your function is now automatically traced
```

</div>

**That's it!** Your functions are now being automatically traced and monitored.

---

## Overview

The Vaquero Python SDK provides comprehensive observability and tracing capabilities for Python applications. This guide covers everything from basic setup to advanced configuration options.

## Installation

### From PyPI (Recommended)

```bash
pip install vaquero-sdk
```

### From Source

```bash
git clone https://github.com/vaquero/vaquero-python.git
cd vaquero-python
pip install -e .
```

## Quick Start

### 1. Initialize the SDK

#### Option A: Project-Scoped API Key (Recommended)

For applications that work with a single project, use a project-scoped API key:

```python
import vaquero

# Simple initialization (recommended)
vaquero.init(api_key="cf_your-project-scoped-key-here")

# Or set environment variable
import os
os.environ["VAQUERO_API_KEY"] = "cf_your-project-scoped-key-here"
vaquero.init()  # Uses environment variable
```

#### Option B: General API Key + Project ID

For applications that need access to multiple projects:

```python
import vaquero

# Initialize SDK with production mode
vaquero.init(
    api_key="your-general-api-key-here",
    project_id="your-project-id-here",
    mode="production"  # Use production mode for optimized settings
)
```

### 2. Start Tracing

```python
# Simple function tracing
@vaquero.trace(agent_name="my_agent")
def process_data(input_data):
    # Your function logic here
    result = transform_data(input_data)
    return result

# Async function tracing
@vaquero.trace(agent_name="api_client")
async def fetch_data(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### 3. Manual Span Creation

For more detailed tracing, you can create manual spans using context managers:

```python
# Manual span creation for complex workflows
with vaquero.span("complex_operation") as span:
    span.set_attribute("operation_type", "batch_processing")
    span.set_attribute("batch_size", len(data))

    # Your code here
    result = process_batch(data)

    span.set_attribute("result_count", len(result))
```

### 3.5. Automatic LLM Instrumentation

Vaquero can automatically instrument popular LLM libraries (OpenAI, Anthropic, etc.) to capture system prompts, tokens, and performance metrics:

```python
import vaquero

# Enable auto-instrumentation (enabled by default in development mode)
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_llm=True,  # Automatically instrument LLM calls
    capture_system_prompts=True  # Capture system prompts
)

# Now any LLM calls will be automatically traced!
import openai

client = openai.OpenAI(api_key="your-openai-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
# This call is automatically instrumented with system prompt, tokens, timing, etc.
```

## Configuration Options

### API Key Types

Vaquero supports two types of API keys:

#### 1. Project-Scoped API Keys (Recommended)
- **Use case**: Applications that work with a single project
- **Benefits**: Simpler configuration, better security, automatic project scoping
- **Format**: `cf_` prefix (e.g., `cf_PfnOhuv9UPLYpU_9o1gr6s1q27JNv7lbFspUR_aoFAM`)
- **Configuration**: Only need the API key, no separate project ID required

#### 2. General API Keys
- **Use case**: Applications that need access to multiple projects
- **Benefits**: Flexible, can access multiple projects with different project IDs
- **Format**: `cf_` prefix (e.g., `cf_l5-XPrTnnqk4H42pbchNrdcR5KvGUrpMH3tG6bgw6GE`)
- **Configuration**: Requires both API key and project ID

### Environment Variables

You can configure the SDK using environment variables:

#### Project-Scoped API Key (Recommended)

```bash
export VAQUERO_API_KEY="cf_your-project-scoped-key-here"
export VAQUERO_ENDPOINT="https://api.vaquero.app"
export VAQUERO_MODE="development"  # or "production"
```

Then initialize from environment:

```python
vaquero.init()  # Uses environment variables
```

#### General API Key + Project ID

```bash
export VAQUERO_API_KEY="your-general-api-key"
export VAQUERO_PROJECT_ID="your-project-id"
export VAQUERO_ENDPOINT="https://api.vaquero.app"
export VAQUERO_MODE="development"  # or "production"
```

Then initialize from environment:

```python
vaquero.init()  # Uses environment variables
```

### Advanced Configuration

```python
from vaquero import SDKConfig

config = SDKConfig(
    api_key="your-api-key",
    project_id="your-project-id",
    endpoint="https://api.vaquero.app",
    batch_size=100,
    flush_interval=5.0,
    max_retries=3,
    capture_inputs=True,
    capture_outputs=True,
    capture_errors=True,
    capture_tokens=True,
    environment="production",
    debug=False,
    auto_instrument_llm=False,  # Disable for production
    capture_system_prompts=False,  # Disable for privacy
    mode="production"
)

vaquero.init(config=config)
```

## Common Use Cases

### 1. Function Tracing

Trace individual functions to monitor their performance and behavior:

```python
@vaquero.trace(agent_name="data_processor")
def process_user_data(user_id, data):
    # Process user data
    result = transform_data(data)
    return result

@vaquero.trace(agent_name="ml_model")
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
@vaquero.trace(agent_name="user_service")
async def get_user(user_id: int):
    user = await database.get_user(user_id)
    return user
```

### 3. Database Operation Tracing

Trace database operations to monitor query performance:

```python
@vaquero.trace(agent_name="database")
async def execute_query(query, params=None):
    with vaquero.span("database_query") as span:
        span.set_attribute("query_type", "select")
        span.set_attribute("table", "users")

        result = await database.execute(query, params)
        span.set_attribute("result_count", len(result))

        return result
```

### 4. Error Handling and Tracing

Trace errors to understand failure patterns:

```python
@vaquero.trace(agent_name="risky_operation")
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
@vaquero.trace(agent_name="batch_processor")
async def process_batch(items):
    async with vaquero.span("batch_processing") as span:
        span.set_attribute("batch_size", len(items))

        results = []
        for i, item in enumerate(items):
            async with vaquero.span(f"process_item_{i}") as item_span:
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
class VaqueroMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        with vaquero.span("http_request") as span:
            span.set_attribute("method", request.method)
            span.set_attribute("url", str(request.url))

            response = await call_next(request)

            span.set_attribute("status_code", response.status_code)
            return response

app.add_middleware(VaqueroMiddleware)
```

### Django

```python
# In middleware.py
from django.utils.deprecation import MiddlewareMixin

class VaqueroMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request._vaquero_start_time = time.time()

    def process_response(self, request, response):
        if hasattr(request, '_vaquero_start_time'):
            duration = time.time() - request._vaquero_start_time

            with vaquero.span("django_request") as span:
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

        with vaquero.span("flask_request") as span:
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
    capture_outputs=False,  # Disable for privacy
    capture_tokens=True,  # Keep token tracking for billing
    auto_instrument_llm=False,  # Disable for production
    capture_system_prompts=False,  # Disable for privacy
    mode="production"
)
```

## Troubleshooting

### Common Issues & Solutions

#### **"SDK not initialized" error**
```python
# Make sure to call init() before using the SDK
import vaquero
vaquero.init(api_key="your-key")
```

#### **Traces not appearing**
```python
# Check if SDK is enabled
if vaquero.get_default_sdk().config.enabled:
    print("SDK is active")
else:
    print("SDK is disabled")

# Manually flush pending traces
vaquero.flush()
```

#### **Performance issues**
```python
# Use development mode for lower latency, production for efficiency
vaquero.init(api_key="your-key", mode="development")  # or "production"
```

#### **High memory usage**
```python
# Monitor memory usage
stats = vaquero.get_default_sdk().get_stats()
print(f"Memory usage: {stats['memory_usage_mb']} MB")

# Reduce batch size if needed
vaquero.init(api_key="your-key", batch_size=50)
```

#### **Network connectivity problems**
```python
# Check endpoint connectivity
import requests
try:
    response = requests.get("https://api.vaquero.app/health", timeout=5)
    print("Endpoint is reachable")
except:
    print("Network connectivity issue")
```

### Debug Mode

Enable debug mode to see detailed SDK logs:

```python
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    debug=True
)
```

### Health Checks

Monitor SDK health:

```python
# Check if SDK is enabled and get stats
sdk = vaquero.get_default_sdk()
if sdk.config.enabled:
    stats = sdk.get_stats()
    print(f"Traces sent: {stats['traces_sent']}")
    print(f"Queue size: {stats['queue_size']}")

# Manually flush traces
vaquero.flush()

# Shutdown SDK
vaquero.shutdown()
```

## 🚀 What's Next?

Now that you have basic tracing working, explore these areas:

### 📖 **[Common Patterns](../patterns/)**
Learn essential patterns for different use cases:
- Function tracing
- API endpoint monitoring
- Database operation tracing
- Error handling

### 🔧 **[Advanced Features](../advanced/)**
Dive deeper into power user features:
- Automatic LLM instrumentation
- Custom performance monitoring
- Workflow orchestration
- Circuit breaker patterns

### 🛠️ **[Framework Integrations](./integrations/)**
Framework-specific guides for:
- FastAPI
- Django
- Flask
- Celery
- SQLAlchemy

### 📚 **[API Reference](../API_REFERENCE.md)**
Complete reference for all configuration options and APIs.

### 💡 **[Troubleshooting Guide](./TROUBLESHOOTING.md)**
Comprehensive guide for common issues and solutions.

### 🎯 **[Best Practices](./BEST_PRACTICES.md)**
Guidelines for consistent, high-quality SDK usage.

## Support

- **Documentation**: [https://docs.vaquero.com](https://docs.vaquero.com)
- **GitHub**: [https://github.com/vaquero/vaquero-python](https://github.com/vaquero/vaquero-python)
- **Issues**: [https://github.com/vaquero/vaquero-python/issues](https://github.com/vaquero/vaquero-python/issues)
- **Email**: support@vaquero.app

## License

This SDK is licensed under the MIT License. See the LICENSE file for details.
