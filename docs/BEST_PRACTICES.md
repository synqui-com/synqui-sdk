# Best Practices Guide

## Core Principles

These fundamental principles should guide all your Vaquero SDK usage decisions.

### 1. Performance-First Design
**Always consider performance impact** when adding tracing to your application.

```python
# ✅ Good - Minimal performance impact
@vaquero.trace("user_lookup")
def get_user(user_id: str) -> dict:
    return database.get_user(user_id)

# ❌ Avoid - High overhead for simple operations
@vaquero.trace("string_operation")
def format_name(first: str, last: str) -> str:
    return f"{first} {last}"  # Don't trace trivial string operations
```

**Guidelines**:
- Trace operations that take >10ms or have business significance
- Use sampling for high-frequency operations
- Monitor SDK performance impact regularly

### 2. Security by Default
**Never log sensitive information** in traces.

```python
# ✅ Good - Safe attribute values
with vaquero.span("user_operation") as span:
    span.set_attribute("user_id", user_id)  # Safe to log
    span.set_attribute("operation_type", "profile_update")  # Descriptive
    span.set_attribute("data_size", len(user_data))  # Quantitative

# ❌ Avoid - Sensitive data exposure
with vaquero.span("user_operation") as span:
    span.set_attribute("password", password)  # NEVER log passwords!
    span.set_attribute("api_key", api_key)    # NEVER log API keys!
    span.set_attribute("user_data", user_data)  # May contain sensitive info
```

**Security Checklist**:
- [ ] No passwords, API keys, or tokens in traces
- [ ] No personally identifiable information (PII)
- [ ] No sensitive business data
- [ ] Use data sanitization for complex objects
- [ ] Enable privacy-focused configuration in production

### 3. Observability Everywhere
**Instrument the right operations** to get meaningful insights.

```python
# ✅ Good - Meaningful observability
@vaquero.trace("payment_processing")
def process_payment(order_id: str, amount: float) -> dict:
    with vaquero.span("payment_validation") as span:
        span.set_attribute("amount_cents", int(amount * 100))
        span.set_attribute("currency", "USD")
        # Validation logic

    with vaquero.span("payment_execution") as span:
        span.set_attribute("payment_method", "credit_card")
        # Payment logic

# ❌ Avoid - Missing key operations
@vaquero.trace("api_endpoint")
def create_order(order_data: dict) -> dict:
    # Missing: payment processing, inventory checks, notifications
    return {"order_id": "123"}
```

### 4. Error Handling Patterns
**Always handle errors gracefully** in traced functions.

```python
@vaquero.trace("risky_operation")
def risky_database_operation(user_id: str) -> dict:
    try:
        with vaquero.span("database_query") as span:
            span.set_attribute("query_type", "SELECT")
            span.set_attribute("table", "users")

            user = database.get_user(user_id)
            span.set_attribute("user_found", user is not None)
            return user

    except ConnectionError as e:
        with vaquero.span("connection_error") as span:
            span.set_attribute("error_type", "connection")
            span.set_attribute("retry_attempt", attempt_number)
        raise

    except Exception as e:
        with vaquero.span("unexpected_error") as span:
            span.set_attribute("error_type", "unexpected")
            span.set_attribute("error_location", "database_query")
        raise
```

## Naming Conventions

### Agent Names
**Use clear, hierarchical naming** for agent identification.

```python
# ✅ Good - Clear hierarchy and purpose
@vaquero.trace("user_service")
def get_user_profile(user_id: str) -> dict:
    pass

@vaquero.trace("payment_service")
def process_payment(order_id: str, amount: float) -> dict:
    pass

@vaquero.trace("notification_service")
def send_welcome_email(user_id: str) -> dict:
    pass

# ❌ Avoid - Unclear or generic names
@vaquero.trace("service")
def do_something(user_id: str) -> dict:
    pass

@vaquero.trace("util")
def helper_function(data: dict) -> dict:
    pass
```

### Operation Names
**Use descriptive operation names** that indicate what the operation does.

```python
# ✅ Good - Clear operation purpose
with vaquero.span("user_profile_lookup") as span:
    span.set_attribute("lookup_method", "database")
    user = database.get_user(user_id)

with vaquero.span("payment_validation") as span:
    span.set_attribute("validation_type", "amount_check")
    validate_payment_amount(amount)

# ❌ Avoid - Generic operation names
with vaquero.span("db_op") as span:
    user = database.get_user(user_id)

with vaquero.span("validation") as span:
    validate_payment_amount(amount)
```

### Attribute Names
**Use consistent, snake_case attribute names**.

```python
# ✅ Good - Consistent naming
span.set_attribute("user_id", user_id)
span.set_attribute("operation_type", "database_query")
span.set_attribute("response_time_ms", duration * 1000)
span.set_attribute("cache_hit", True)

# ❌ Avoid - Inconsistent naming
span.set_attribute("userID", user_id)      # camelCase
span.set_attribute("operationType", "query")  # camelCase
span.set_attribute("responseTime", duration)  # camelCase
span.set_attribute("cache_hit", true)       # boolean inconsistency
```

## Performance Guidelines

### Batch Size Optimization
**Choose batch sizes appropriate for your workload**.

```python
# High-throughput applications
vaquero.init(api_key="your-key", batch_size=500, flush_interval=10.0)

# Low-latency applications
vaquero.init(api_key="your-key", batch_size=25, flush_interval=2.0)

# Memory-constrained environments
vaquero.init(api_key="your-key", batch_size=10, flush_interval=1.0)

# Monitor and adjust
sdk = vaquero.get_default_sdk()
stats = sdk.get_stats()
if stats['memory_usage_mb'] > 100:  # If using too much memory
    vaquero.init(api_key="your-key", batch_size=50)  # Reduce batch size
```

### Sampling Strategies
**Use intelligent sampling** for high-volume operations.

```python
# Sample based on operation type
class OperationSampler:
    def __init__(self):
        self.sample_rates = {
            "error": 1.0,        # Sample 100% of errors
            "payment": 0.1,      # Sample 10% of payments
            "user_action": 0.05, # Sample 5% of user actions
            "api_call": 0.01     # Sample 1% of API calls
        }

    def should_sample(self, operation_type: str) -> bool:
        rate = self.sample_rates.get(operation_type, 0.01)
        import random
        return random.random() < rate

sampler = OperationSampler()

@vaquero.trace("sampled_operation")
def high_frequency_operation(operation_type: str):
    if sampler.should_sample(operation_type):
        # Full tracing
        pass
    else:
        # Minimal tracing or no tracing
        pass
```

### Memory Management
**Monitor and control memory usage**.

```python
# Regular memory monitoring
def monitor_memory_usage():
    sdk = vaquero.get_default_sdk()
    stats = sdk.get_stats()

    memory_mb = stats.get('memory_usage_mb', 0)
    if memory_mb > 50:  # Threshold for concern
        print(f"High memory usage: {memory_mb} MB")

        # Reduce batch size if needed
        if memory_mb > 100:
            vaquero.init(api_key="your-key", batch_size=25)

# Periodic cleanup
import gc
def cleanup_memory():
    gc.collect()
    vaquero.flush()  # Flush any pending traces
```

## Configuration Management

### Environment-Based Configuration
**Use environment variables for configuration**.

```python
# .env file
VAQUERO_API_KEY=cf_your-project-key
VAQUERO_PROJECT_ID=your-project-id
VAQUERO_MODE=production
VAQUERO_BATCH_SIZE=100
VAQUERO_AUTO_INSTRUMENT_LLM=false

# Production application
import vaquero
vaquero.init()  # Loads from environment

# Development overrides
if os.getenv("ENVIRONMENT") == "development":
    vaquero.init(
        api_key="your-key",
        mode="development",
        debug=True,
        batch_size=10
    )
```

### Mode Selection
**Choose the right mode for your environment**.

```python
# Development mode - Maximum observability
vaquero.init(
    api_key="your-key",
    mode="development",  # Enables all features for debugging
    debug=True,
    batch_size=10,       # Smaller batches for faster feedback
    flush_interval=2.0   # More frequent flushing
)

# Production mode - Optimized performance
vaquero.init(
    api_key="your-key",
    mode="production",   # Optimized settings
    capture_inputs=False,    # Privacy protection
    capture_outputs=False,   # Performance optimization
    auto_instrument_llm=False,  # Reduce overhead
    batch_size=200,      # Larger batches for efficiency
    flush_interval=10.0  # Less frequent flushing
)
```

## Error Handling Patterns

### Exception Categorization
**Categorize exceptions for better analysis**.

```python
@vaquero.trace("categorized_operation")
def operation_with_categorization(data: dict) -> dict:
    try:
        result = process_data(data)
        return result

    except ValueError as e:
        with vaquero.span("validation_error") as span:
            span.set_attribute("error_category", "validation")
            span.set_attribute("error_field", "data_format")
            span.set_attribute("error_severity", "medium")
        raise

    except ConnectionError as e:
        with vaquero.span("infrastructure_error") as span:
            span.set_attribute("error_category", "infrastructure")
            span.set_attribute("error_component", "database")
            span.set_attribute("error_severity", "high")
        raise

    except Exception as e:
        with vaquero.span("unexpected_error") as span:
            span.set_attribute("error_category", "unexpected")
            span.set_attribute("error_severity", "critical")
        raise
```

### Recovery Strategies
**Implement proper recovery strategies**.

```python
@vaquero.trace("resilient_operation")
def operation_with_recovery(data: dict) -> dict:
    max_retries = 3

    for attempt in range(max_retries):
        try:
            with vaquero.span(f"attempt_{attempt + 1}") as span:
                span.set_attribute("attempt_number", attempt + 1)
                span.set_attribute("max_retries", max_retries)

                result = risky_operation(data)
                span.set_attribute("recovery_successful", True)
                return result

        except RetryableError as e:
            if attempt == max_retries - 1:
                with vaquero.span("final_failure") as span:
                    span.set_attribute("failure_reason", "max_retries_exceeded")
                    span.set_attribute("total_attempts", max_retries)
                raise

            # Wait before retry with exponential backoff
            wait_time = 2 ** attempt
            with vaquero.span("retry_wait") as span:
                span.set_attribute("wait_time_seconds", wait_time)
            time.sleep(wait_time)

        except NonRetryableError as e:
            with vaquero.span("non_retryable_error") as span:
                span.set_attribute("error_category", "business_logic")
            raise
```

## Testing Strategies

### Unit Testing Patterns
**Test your traced functions properly**.

```python
import pytest
from unittest.mock import patch

class TestTracedFunctions:
    def test_successful_operation(self):
        """Test successful operation with tracing."""
        with patch('vaquero.span') as mock_span:
            result = traced_function("test_data")

            # Verify function behavior
            assert result == "expected_result"

            # Verify tracing occurred
            mock_span.assert_called()
            call_args = mock_span.call_args
            assert "test_operation" in call_args[0][0]  # Operation name

    def test_error_handling(self):
        """Test error handling with tracing."""
        with patch('vaquero.span') as mock_span:
            with pytest.raises(ValueError):
                traced_function("invalid_data")

            # Verify error was traced
            span_context = mock_span.return_value.__enter__.return_value
            # Check that error attributes were set
            # (Implementation depends on your tracing setup)
```

### Integration Testing
**Test SDK integration with your application**.

```python
import pytest
from fastapi.testclient import TestClient

class TestAPIIntegration:
    def test_traced_endpoint(self, client):
        """Test that API endpoints are properly traced."""
        response = client.get("/users/123")

        assert response.status_code == 200

        # Verify tracing occurred (if you have access to trace data)
        # This depends on your test setup
```

### Performance Testing
**Include performance testing for traced operations**.

```python
import time
import pytest

class TestPerformance:
    def test_operation_performance(self):
        """Test that traced operations meet performance requirements."""
        start_time = time.time()

        result = traced_function("test_data")

        duration = time.time() - start_time

        # Assert performance requirement
        assert duration < 0.1  # Should complete in under 100ms

        # Verify result correctness
        assert result == "expected_result"

    def test_memory_usage(self):
        """Test that tracing doesn't cause memory leaks."""
        import psutil
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Run traced operations
        for _ in range(100):
            traced_function("test_data")

        gc.collect()
        final_memory = process.memory_info().rss

        # Memory shouldn't grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 10 * 1024 * 1024  # Less than 10MB growth
```

## Deployment Patterns

### Environment Configuration
**Configure appropriately for each environment**.

```python
# Development environment
def configure_development():
    vaquero.init(
        api_key="dev-key",
        mode="development",
        debug=True,
        batch_size=10,
        flush_interval=2.0,
        capture_inputs=True,   # Full debugging
        capture_outputs=True,  # Full debugging
        auto_instrument_llm=True  # Full observability
    )

# Staging environment
def configure_staging():
    vaquero.init(
        api_key="staging-key",
        mode="development",  # Keep debugging for staging
        batch_size=50,
        capture_inputs=False,  # Privacy protection
        capture_outputs=False, # Privacy protection
        auto_instrument_llm=True   # Keep LLM tracing
    )

# Production environment
def configure_production():
    vaquero.init(
        api_key="prod-key",
        mode="production",
        batch_size=200,
        flush_interval=10.0,
        capture_inputs=False,     # Privacy protection
        capture_outputs=False,    # Privacy protection
        auto_instrument_llm=False, # Performance optimization
        capture_system_prompts=False  # Privacy protection
    )
```

### Production Optimizations
**Optimize for production workloads**.

```python
# Production-optimized configuration
config = vaquero.SDKConfig(
    api_key="prod-key",
    project_id="prod-project",
    mode="production",

    # Performance optimizations
    batch_size=500,           # Large batches for efficiency
    flush_interval=15.0,      # Less frequent flushing

    # Privacy protection
    capture_inputs=False,     # Don't capture user inputs
    capture_outputs=False,    # Don't capture sensitive outputs
    capture_system_prompts=False,  # Privacy protection

    # Reduced overhead
    auto_instrument_llm=False,     # Disable for performance
    detect_agent_frameworks=False, # Disable for performance

    # Error tracking only
    capture_errors=True,      # Keep error tracking enabled
    capture_tokens=True,      # Keep token tracking for billing

    # Reliability
    max_retries=5,           # More retries for production
    timeout=60.0             # Longer timeout for production
)

vaquero.init(config=config)
```

## Monitoring and Maintenance

### Regular Health Checks
**Monitor SDK health regularly**.

```python
def perform_health_check():
    """Regular health check for SDK."""
    sdk = vaquero.get_default_sdk()

    # Check configuration
    if not sdk.config.enabled:
        raise RuntimeError("SDK is disabled")

    # Check connectivity
    try:
        vaquero.flush()
        connectivity_ok = True
    except Exception:
        connectivity_ok = False

    # Check memory usage
    stats = sdk.get_stats()
    memory_usage_mb = stats.get('memory_usage_mb', 0)

    # Log health status
    print(f"SDK Health: {'OK' if connectivity_ok else 'ERROR'}")
    print(f"Memory Usage: {memory_usage_mb} MB")
    print(f"Traces Sent: {stats.get('traces_sent', 0)}")

    return {
        "healthy": connectivity_ok,
        "memory_usage_mb": memory_usage_mb,
        "traces_sent": stats.get('traces_sent', 0)
    }

# Run health check periodically
import schedule

schedule.every(5).minutes.do(perform_health_check)
```

### Performance Monitoring
**Monitor SDK performance impact**.

```python
def monitor_sdk_performance():
    """Monitor SDK performance impact."""
    import time
    import psutil

    process = psutil.Process()

    # Measure baseline performance
    start_time = time.time()
    start_memory = process.memory_info().rss

    # Run operations without tracing
    for _ in range(100):
        baseline_operation()

    baseline_time = time.time() - start_time
    baseline_memory = process.memory_info().rss - start_memory

    # Reset and measure with tracing
    start_time = time.time()
    start_memory = process.memory_info().rss

    # Run same operations with tracing
    for _ in range(100):
        traced_operation()

    traced_time = time.time() - start_time
    traced_memory = process.memory_info().rss - start_memory

    # Calculate overhead
    time_overhead = ((traced_time - baseline_time) / baseline_time) * 100
    memory_overhead = traced_memory - baseline_memory

    print(f"Time overhead: {time_overhead:.1f}%")
    print(f"Memory overhead: {memory_overhead / 1024 / 1024:.1f} MB")

    # Alert if overhead is too high
    if time_overhead > 10:  # More than 10% overhead
        print("WARNING: High time overhead detected!")
```

## Common Anti-Patterns to Avoid

### ❌ Performance Anti-Patterns
- **Over-tracing**: Tracing every single function call
- **Large batch sizes**: Causing memory issues and delays
- **No sampling**: Tracing 100% of high-volume operations
- **Synchronous tracing in async contexts**: Blocking the event loop

### ✅ Performance Best Practices
- **Selective tracing**: Only trace operations with business value
- **Appropriate batch sizes**: Balance latency vs throughput
- **Intelligent sampling**: Sample based on operation importance
- **Async-aware tracing**: Use async context managers properly

### ❌ Security Anti-Patterns
- **Logging sensitive data**: Passwords, API keys, PII in traces
- **No input sanitization**: Passing raw data to tracing
- **Debug mode in production**: Exposing internal details

### ✅ Security Best Practices
- **Data sanitization**: Remove/encrypt sensitive information
- **Privacy-first configuration**: Disable unnecessary capture in production
- **Secure key management**: Use environment variables, not hardcoded keys

### ❌ Observability Anti-Patterns
- **Generic names**: Using "operation" instead of "user_lookup"
- **Missing context**: No attributes to help debugging
- **No error handling**: Not capturing error information

### ✅ Observability Best Practices
- **Descriptive naming**: Clear, hierarchical operation names
- **Rich attributes**: Meaningful context for debugging
- **Comprehensive error handling**: Proper exception categorization

## Conclusion

Following these best practices ensures that your Vaquero SDK integration provides maximum value with minimal overhead. Remember:

- **Start simple**: Begin with basic tracing and expand as needed
- **Monitor impact**: Regularly check performance and memory usage
- **Iterate**: Adjust configuration based on your application's needs
- **Security first**: Never compromise on data privacy and security

These practices will help you build observable, performant, and maintainable applications with the Vaquero SDK.
