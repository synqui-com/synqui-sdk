# 🔍 Auto-Instrumentation

**Automatically trace popular libraries and frameworks without manual code changes.** Vaquero's auto-instrumentation capabilities detect and instrument common patterns, reducing the need for manual tracing decorators.

## 🎯 What is Auto-Instrumentation?

Auto-instrumentation automatically detects and traces:
- **LLM API calls** (OpenAI, Anthropic, Hugging Face)
- **HTTP requests** (requests, httpx, aiohttp)
- **Database operations** (SQLAlchemy, Django ORM, psycopg2)
- **Redis operations** (redis-py)
- **Message queues** (Celery, RQ)
- **External APIs** (common REST clients)

## 🚀 Basic Auto-Instrumentation Setup

### Enable Auto-Instrumentation

```python
import vaquero

# Enable auto-instrumentation during initialization
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_llm=True,        # Auto-trace LLM calls
    auto_instrument_http=True,       # Auto-trace HTTP requests
    auto_instrument_database=True,   # Auto-trace database operations
    auto_instrument_redis=True,      # Auto-trace Redis operations
    capture_system_prompts=True      # Capture LLM system prompts
)
```

### LLM Auto-Instrumentation

Automatically trace OpenAI, Anthropic, and other LLM calls:

```python
import vaquero
import openai

# Initialize with auto-instrumentation
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_llm=True,
    capture_system_prompts=True
)

# LLM calls are automatically traced!
client = openai.OpenAI(api_key="your-openai-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

# This call is automatically instrumented with:
# - System prompt capture
# - Token usage tracking
# - Response time monitoring
# - Model and API details
```

### HTTP Request Auto-Instrumentation

```python
import vaquero
import requests

# Enable HTTP auto-instrumentation
vaquero.configure(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_http=True
)

# HTTP requests are automatically traced!
response = requests.get("https://api.example.com/users/123")
data = response.json()

# Automatically captures:
# - Request method, URL, headers
# - Response status, size, timing
# - Error handling and retries
```

## 🎨 Advanced Auto-Instrumentation Patterns

### Custom Auto-Instrumentation Rules

Define custom rules for specific libraries or patterns:

```python
import vaquero
from vaquero.instrumentation import AutoInstrumentation

# Create custom auto-instrumentation rules
class CustomInstrumentation(AutoInstrumentation):
    """Custom auto-instrumentation for specific use cases."""

    def __init__(self):
        super().__init__()

        # Define patterns to instrument
        self.add_pattern(
            module="your_custom_library",
            function="api_call",
            span_name="custom_api_call",
            capture_args=True,
            capture_return=True
        )

        self.add_pattern(
            module="your_framework",
            function="process_request",
            span_name="framework_request",
            tags={"framework": "custom"}
        )

# Enable custom instrumentation
custom_instrumentation = CustomInstrumentation()
vaquero.configure(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrumentation=custom_instrumentation
)
```

### Selective Auto-Instrumentation

Enable auto-instrumentation for specific modules only:

```python
import vaquero

# Configure selective auto-instrumentation
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_modules=[
        "openai",           # Only OpenAI calls
        "requests",         # Only requests library
        "sqlalchemy"        # Only SQLAlchemy operations
    ],
    auto_instrument_exclude=[
        "internal_module"   # Skip internal utilities
    ]
)
```

## 📊 LLM-Specific Auto-Instrumentation

### OpenAI Integration

```python
import vaquero
import openai

# Enable comprehensive OpenAI tracing
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_llm=True,
    capture_system_prompts=True,
    capture_llm_metadata=True
)

client = openai.OpenAI(api_key="your-key")

# All OpenAI calls are automatically traced
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write a Python function to sort a list"}
    ],
    max_tokens=100
)

# Automatically captured:
# - System prompt and user message
# - Model used (gpt-4)
# - Token usage (prompt_tokens, completion_tokens, total_tokens)
# - Response time
# - API endpoint and parameters
```

### Anthropic Integration

```python
import vaquero
import anthropic

# Anthropic calls are also automatically traced
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_llm=True
)

client = anthropic.Anthropic(api_key="your-anthropic-key")

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "Explain machine learning"}
    ],
    max_tokens=200
)

# Same comprehensive tracing as OpenAI
```

## 🔧 Framework Auto-Instrumentation

### FastAPI Auto-Instrumentation

```python
import vaquero
from fastapi import FastAPI

# Enable FastAPI auto-instrumentation
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_frameworks=["fastapi"]
)

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # Endpoint is automatically traced
    user = await database.get_user(user_id)
    return user

@app.post("/users")
async def create_user(user_data: dict):
    # Request/response automatically traced
    user = await database.create_user(user_data)
    return {"id": user.id}
```

### Django Auto-Instrumentation

```python
import vaquero
import os
import django

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

# Enable Django auto-instrumentation
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_frameworks=["django"]
)

django.setup()

# Django views are automatically traced
from django.http import JsonResponse

def get_user(request, user_id):
    # Request automatically traced
    user = User.objects.get(id=user_id)
    return JsonResponse({
        "id": user.id,
        "name": user.name
    })
```

## 📈 Performance Impact Management

### Sampling Configuration

Control auto-instrumentation overhead:

```python
import vaquero

# Configure sampling for high-traffic applications
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_llm=True,
    llm_sampling_rate=0.1,        # Sample 10% of LLM calls
    http_sampling_rate=0.05,      # Sample 5% of HTTP requests
    database_sampling_rate=0.2    # Sample 20% of database operations
)
```

### Performance Monitoring

Monitor the impact of auto-instrumentation:

```python
import vaquero

@vaquero.trace(agent_name="instrumentation_monitor")
class InstrumentationMonitor:
    """Monitor auto-instrumentation performance impact."""

    def __init__(self):
        self.instrumentation_overhead = []

    def measure_overhead(self, operation_func, *args, **kwargs):
        """Measure performance overhead of instrumentation."""
        # Measure without instrumentation
        start = time.time()
        result = operation_func(*args, **kwargs)
        untraced_time = time.time() - start

        # Measure with instrumentation
        vaquero.configure(auto_instrument_llm=False)  # Temporarily disable

        start = time.time()
        traced_result = operation_func(*args, **kwargs)
        traced_time = time.time() - start

        vaquero.configure(auto_instrument_llm=True)  # Re-enable

        overhead = traced_time - untraced_time
        self.instrumentation_overhead.append(overhead)

        return traced_result

    def get_performance_report(self) -> dict:
        """Generate performance impact report."""
        if not self.instrumentation_overhead:
            return {"error": "No measurements taken"}

        return {
            "average_overhead_ms": statistics.mean(self.instrumentation_overhead) * 1000,
            "max_overhead_ms": max(self.instrumentation_overhead) * 1000,
            "measurements_count": len(self.instrumentation_overhead),
            "overhead_percentage": (
                statistics.mean(self.instrumentation_overhead) / 0.001  # Assuming 1ms baseline
            ) * 100
        }
```

## 🎯 Best Practices

### 1. **Start with Minimal Instrumentation**
```python
# ✅ Good - Start simple
vaquero.configure(
    auto_instrument_llm=True,  # Focus on key operations
)

# ❌ Avoid - Too much at once
vaquero.configure(
    auto_instrument_llm=True,
    auto_instrument_http=True,
    auto_instrument_database=True,
    auto_instrument_redis=True,
    # May impact performance
)
```

### 2. **Use Sampling for High Volume**
```python
# ✅ Good - Sample high-volume operations
vaquero.configure(
    auto_instrument_http=True,
    http_sampling_rate=0.1  # 10% of HTTP requests
)

# ❌ Avoid - Trace everything
vaquero.configure(
    auto_instrument_http=True,
    http_sampling_rate=1.0  # 100% of HTTP requests
)
```

### 3. **Monitor Performance Impact**
```python
# ✅ Good - Monitor overhead
monitor = InstrumentationMonitor()
overhead = monitor.measure_overhead(api_call)
print(f"Instrumentation overhead: {overhead*1000:.2f}ms")

# ❌ Avoid - Ignore performance impact
# Just enable everything without monitoring
```

## 🚨 Common Issues

### "Auto-instrumentation not working"
```python
# Check if auto-instrumentation is enabled
import vaquero

config = vaquero.get_default_sdk().config
print(f"LLM auto-instrumentation: {config.auto_instrument_llm}")
print(f"HTTP auto-instrumentation: {config.auto_instrument_http}")

# Verify imports are available
try:
    import openai
    print("OpenAI library found")
except ImportError:
    print("OpenAI library not installed")
```

### "Performance degradation"
```python
# Reduce sampling rate
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    llm_sampling_rate=0.05,    # 5% of LLM calls
    http_sampling_rate=0.02,   # 2% of HTTP requests
    database_sampling_rate=0.1 # 10% of database operations
)

# Or disable specific instrumentations
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    auto_instrument_llm=False,  # Disable LLM tracing
    auto_instrument_http=True   # Keep HTTP tracing
)
```

### "Missing context in auto-traced spans"
```python
# Add custom context to auto-instrumented operations
with vaquero.span("business_context") as span:
    span.set_attribute("user_id", current_user.id)
    span.set_attribute("operation_type", "user_action")

    # Auto-instrumented operations will inherit this context
    response = requests.get(f"/api/users/{current_user.id}")
```

## 📚 Auto-Instrumentation Libraries

### Supported Libraries (Auto-Detected)

| Library | Auto-Instrumentation | Status |
|---------|-------------------|--------|
| **OpenAI** | ✅ LLM calls, tokens, prompts | Production |
| **Anthropic** | ✅ LLM calls, tokens, prompts | Production |
| **requests** | ✅ HTTP requests/responses | Production |
| **httpx** | ✅ HTTP requests/responses | Production |
| **aiohttp** | ✅ HTTP requests/responses | Production |
| **SQLAlchemy** | ✅ Database queries | Production |
| **psycopg2** | ✅ Database operations | Production |
| **redis-py** | ✅ Redis operations | Production |
| **Celery** | ✅ Task execution | Beta |
| **Django ORM** | ✅ Database queries | Beta |
| **FastAPI** | ✅ Request/response | Beta |

### Manual Instrumentation (When Needed)

For libraries not yet supported:

```python
import vaquero

# Manually instrument unsupported library
@vaquero.trace(agent_name="custom_library")
def custom_api_call(endpoint: str, data: dict):
    with vaquero.span("custom_api_request") as span:
        span.set_attribute("endpoint", endpoint)
        span.set_attribute("data_size", len(str(data)))

        # Your custom API logic
        response = custom_library.api_call(endpoint, data)
        span.set_attribute("response_size", len(str(response)))

        return response
```

## 🎉 You're Ready!

Auto-instrumentation dramatically reduces the amount of manual tracing code you need to write while providing comprehensive observability. Combined with manual tracing for business logic, you get complete visibility into your application's behavior.

Next, explore **[Performance Monitoring](./performance-monitoring.md)** for advanced observability features or **[Workflow Orchestration](./workflow-orchestration.md)** for complex multi-step processes.
