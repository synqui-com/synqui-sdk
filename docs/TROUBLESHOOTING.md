# Troubleshooting Guide

## Common Issues & Solutions

This guide covers the most common issues users encounter with the Vaquero SDK and provides step-by-step solutions.

## Installation Problems

### Dependency Conflicts
**Problem**: Installation fails due to conflicting dependencies.

**Solution**:
```bash
# Install in a clean virtual environment
python -m venv vaquero-env
source vaquero-env/bin/activate  # On Windows: vaquero-env\Scripts\activate
pip install vaquero-sdk

# Or use specific versions
pip install "vaquero-sdk==0.1.0"
```

**Check**: Verify Python version compatibility:
```python
python --version  # Should be 3.8+
pip --version     # Should work
```

### Platform-Specific Issues
**Problem**: Installation fails on certain platforms (Windows, macOS, Linux).

**Solution**:
```bash
# Windows - use compatible wheel
pip install --only-binary=all vaquero-sdk

# macOS - ensure development tools
xcode-select --install  # If missing

# Linux - install system dependencies
sudo apt-get install python3-dev  # Ubuntu/Debian
sudo yum install python3-devel    # CentOS/RHEL
```

## Configuration Issues

### API Key Problems
**Problem**: Invalid API key or authentication errors.

**Symptoms**:
- Traces not appearing in dashboard
- Authentication errors in logs
- HTTP 401/403 responses

**Solution**:
```python
# 1. Verify API key format
api_key = "cf_your-project-scoped-key-here"  # Should start with 'cf_'

# 2. Check key validity
import vaquero
try:
    vaquero.init(api_key="your-key", mode="development")
    print("API key is valid")
except Exception as e:
    print(f"API key error: {e}")

# 3. Use environment variables for security
import os
os.environ["VAQUERO_API_KEY"] = "cf_your-key"
vaquero.init()  # Loads from environment
```

### Project ID Configuration
**Problem**: Project ID not set or incorrect.

**Solution**:
```python
# Option 1: Project-scoped API key (recommended)
vaquero.init(api_key="cf_your-project-key")  # Project ID auto-provisioned

# Option 2: General API key + explicit project ID
vaquero.init(
    api_key="your-general-key",
    project_id="your-project-id"
)

# Check current configuration
sdk = vaquero.get_default_sdk()
print(f"Project ID: {sdk.config.project_id}")
print(f"API Key: {sdk.config.api_key[:10]}...")  # Show first 10 chars only
```

### Environment Variable Setup
**Problem**: Environment variables not being read correctly.

**Solution**:
```bash
# Set environment variables
export VAQUERO_API_KEY="cf_your-key"
export VAQUERO_PROJECT_ID="your-project-id"
export VAQUERO_MODE="development"

# Verify they're set
echo $VAQUERO_API_KEY
echo $VAQUERO_PROJECT_ID

# In Python
import os
print(os.getenv("VAQUERO_API_KEY"))  # Should show your key
```

## Performance Issues

### High Overhead
**Problem**: SDK adds significant performance overhead.

**Symptoms**:
- Slow application response times
- High CPU/memory usage
- Large batch processing delays

**Solution**:
```python
# Use production mode for optimized settings
vaquero.init(api_key="your-key", mode="production")

# Or customize specific settings
vaquero.init(
    api_key="your-key",
    batch_size=200,           # Increase batch size
    flush_interval=10.0,      # Less frequent flushing
    capture_inputs=False,     # Disable input capture
    capture_outputs=False,    # Disable output capture
    auto_instrument_llm=False # Disable LLM auto-instrumentation
)

# Monitor performance impact
import time
start = time.time()
# Your code here
duration = time.time() - start
print(f"Operation took: {duration:.3f}s")
```

### Memory Usage Problems
**Problem**: SDK consuming too much memory.

**Solution**:
```python
# Monitor memory usage
sdk = vaquero.get_default_sdk()
stats = sdk.get_stats()
print(f"Memory usage: {stats['memory_usage_mb']} MB")

# Reduce batch size to process traces more frequently
vaquero.init(api_key="your-key", batch_size=25)

# Use sampling for high-volume operations
vaquero.init(
    api_key="your-key",
    http_sampling_rate=0.1,    # Sample 10% of HTTP requests
    database_sampling_rate=0.05 # Sample 5% of database operations
)
```

### Network Connectivity
**Problem**: Traces not being sent due to network issues.

**Solution**:
```python
# Check network connectivity
import requests
try:
    response = requests.get("https://api.vaquero.app/health", timeout=5)
    print("Network connectivity OK")
except Exception as e:
    print(f"Network error: {e}")

# Increase timeout settings
vaquero.init(
    api_key="your-key",
    timeout=60.0,        # Longer timeout
    max_retries=5        # More retry attempts
)

# Use circuit breaker for external dependencies
from vaquero.error_handling import CircuitBreaker
circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
```

## Integration Issues

### Framework Compatibility
**Problem**: SDK conflicts with framework middleware or decorators.

**Solution**:
```python
# FastAPI - ensure correct middleware order
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware first
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Add Vaquero middleware after
@app.middleware("http")
async def vaquero_middleware(request, call_next):
    with vaquero.span("http_request") as span:
        # Your middleware logic
        pass

# Django - check middleware order in settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',  # Before Vaquero if using
    'your_app.middleware.VaqueroMiddleware',   # Your Vaquero middleware
    # ... other middleware
]
```

### Import Order Problems
**Problem**: SDK not properly initialized due to import order issues.

**Solution**:
```python
# Initialize SDK as early as possible
import vaquero

# Configure before importing other modules
vaquero.init(api_key="your-key")

# Then import your application modules
import your_app.models
import your_app.views
```

### Context Propagation Issues
**Problem**: Trace context not propagating between threads/async tasks.

**Solution**:
```python
# For async operations
import asyncio

@vaquero.trace("async_operation")
async def async_function():
    # Context automatically propagated in async functions
    await some_async_operation()

# For thread-based operations
import threading

@vaquero.trace("threaded_operation")
def threaded_function():
    # Context may not propagate automatically
    # Use explicit context passing if needed
    pass

# Manual context management
with vaquero.span("manual_context") as span:
    span.set_attribute("context_id", "manual")

    # Pass context to other functions
    result = process_with_context(span.context)
```

## Runtime Issues

### Circuit Breaker Patterns
**Problem**: External services failing and triggering circuit breaker.

**Solution**:
```python
# Monitor circuit breaker state
from vaquero.error_handling import CircuitBreaker

circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

# Check if circuit is open
if circuit_breaker.state == "OPEN":
    print("Circuit breaker is open - service unavailable")

# Implement graceful degradation
@vaquero.trace("resilient_operation")
async def resilient_api_call():
    try:
        return await circuit_breaker.call(external_api_call)
    except Exception:
        # Fallback logic
        return await fallback_operation()
```

### Error Handling Patterns
**Problem**: Errors not being captured properly.

**Solution**:
```python
@vaquero.trace("error_prone_operation")
def risky_operation(data):
    try:
        # Your risky logic here
        result = process_data(data)
        return result
    except ValueError as e:
        # Validation errors
        with vaquero.span("validation_error") as span:
            span.set_attribute("error_type", "validation")
            span.set_attribute("error_field", "data_format")
        raise
    except ConnectionError as e:
        # Network errors
        with vaquero.span("network_error") as span:
            span.set_attribute("error_type", "network")
            span.set_attribute("service", "external_api")
        raise
    except Exception as e:
        # Unexpected errors
        with vaquero.span("unexpected_error") as span:
            span.set_attribute("error_type", "unexpected")
        raise
```

### Logging Configuration
**Problem**: SDK logs not appearing or too verbose.

**Solution**:
```python
# Enable debug logging
vaquero.init(api_key="your-key", debug=True)

# Configure Python logging
import logging

# Reduce external library noise
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Set SDK log level
logging.getLogger("vaquero").setLevel(logging.DEBUG)

# View logs
import sys
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
```

## Debugging Tools & Techniques

### SDK Health Checks
```python
# Comprehensive health check
def check_sdk_health():
    sdk = vaquero.get_default_sdk()

    print("=== SDK Health Check ===")
    print(f"Enabled: {sdk.config.enabled}")
    print(f"API Key: {sdk.config.api_key[:10]}..." if sdk.config.api_key else "Not set")
    print(f"Project ID: {sdk.config.project_id}")
    print(f"Mode: {sdk.config.mode}")
    print(f"Batch Size: {sdk.config.batch_size}")

    # Get detailed stats
    try:
        stats = sdk.get_stats()
        print(f"Traces Sent: {stats.get('traces_sent', 0)}")
        print(f"Queue Size: {stats.get('queue_size', 0)}")
        print(f"Memory Usage: {stats.get('memory_usage_mb', 0)} MB")
    except Exception as e:
        print(f"Stats error: {e}")

    # Test connectivity
    try:
        vaquero.flush()  # Force send pending traces
        print("Connectivity: OK")
    except Exception as e:
        print(f"Connectivity error: {e}")

# Run health check
check_sdk_health()
```

### Performance Profiling
```python
import time
import cProfile
import pstats

@vaquero.trace("performance_test")
def performance_test():
    # Your code to profile
    time.sleep(0.1)

# Profile with cProfile
profiler = cProfile.Profile()
profiler.enable()

performance_test()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Trace Inspection
```python
# Inspect current trace context
current_span = vaquero.get_current_span()
if current_span:
    print(f"Current span: {current_span.operation_name}")
    print(f"Attributes: {current_span.attributes}")
else:
    print("No active span")

# List all active spans (if available)
sdk = vaquero.get_default_sdk()
# This would depend on SDK implementation
```

## Performance Optimization

### Batch Size Tuning
```python
# Find optimal batch size
def tune_batch_size():
    sizes_to_test = [10, 25, 50, 100, 200]
    results = {}

    for batch_size in sizes_to_test:
        start_time = time.time()

        # Test with this batch size
        vaquero.init(api_key="your-key", batch_size=batch_size)

        # Generate some traces
        for i in range(100):
            @vaquero.trace("test_operation")
            def test_op():
                time.sleep(0.001)  # Small operation

            test_op()

        # Force flush and measure time
        flush_start = time.time()
        vaquero.flush()
        flush_time = time.time() - flush_start

        total_time = time.time() - start_time
        results[batch_size] = {
            "total_time": total_time,
            "flush_time": flush_time,
            "efficiency": 100 / total_time  # Traces per second
        }

    # Find best batch size
    best_size = max(results.keys(), key=lambda k: results[k]["efficiency"])
    print(f"Best batch size: {best_size}")
    print(f"Results: {results}")

tune_batch_size()
```

### Sampling Strategies
```python
# Implement intelligent sampling
class SmartSampler:
    def __init__(self, sample_rate=0.1):
        self.sample_rate = sample_rate
        self.request_count = 0

    def should_sample(self, operation_type: str) -> bool:
        self.request_count += 1

        # Sample more frequently for errors
        if operation_type == "error":
            return True

        # Sample based on rate
        import random
        return random.random() < self.sample_rate

# Use smart sampling
sampler = SmartSampler(sample_rate=0.05)  # 5% sampling

@vaquero.trace("sampled_operation")
def sampled_function():
    if sampler.should_sample("normal"):
        # Your logic here
        pass
```

## Getting Help

### Community Resources
- **GitHub Issues**: [Report bugs and request features](https://github.com/vaquero/vaquero-python/issues)
- **Discussions**: [Community discussions and Q&A](https://github.com/vaquero/vaquero-python/discussions)
- **Discord**: Join our [Discord community](https://discord.gg/vaquero)

### Support Channels
- **Email**: support@vaquero.app
- **Documentation**: [Complete documentation](https://docs.vaquero.com)
- **Examples**: [Usage examples](https://github.com/vaquero/vaquero-python/tree/main/examples)

### Issue Reporting
When reporting issues, please include:

1. **SDK Version**: `pip show vaquero-sdk`
2. **Python Version**: `python --version`
3. **Error Messages**: Full stack traces
4. **Configuration**: Your `vaquero.init()` call
5. **Expected vs Actual Behavior**: What you expected vs what happened
6. **Reproduction Steps**: Minimal code to reproduce the issue

### Performance Reports
For performance issues, include:
- System specifications (CPU, RAM, OS)
- Application load characteristics
- Before/after performance comparisons
- Profiling data if available

## Common Anti-Patterns to Avoid

### ❌ Don't
- Log sensitive data in traces (passwords, API keys, PII)
- Use synchronous operations in async contexts
- Ignore error handling in traced functions
- Set batch sizes too large (causes memory issues)
- Disable all capture options (reduces observability value)

### ✅ Do
- Use descriptive agent names and operation names
- Add meaningful attributes to spans
- Handle errors gracefully in traced functions
- Monitor SDK performance and memory usage
- Use appropriate sampling for high-volume operations
- Test your integration thoroughly before production deployment

This troubleshooting guide should resolve 90%+ of common SDK issues. For complex problems, don't hesitate to reach out to our community or support team!
