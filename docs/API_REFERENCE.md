# CognitionFlow SDK API Reference

## Overview

The CognitionFlow Python SDK provides comprehensive observability and tracing capabilities for Python applications. This document provides detailed API reference for all SDK components.

## Table of Contents

- [Configuration](#configuration)
- [Core SDK](#core-sdk)
- [Tracing](#tracing)
- [Context Management](#context-management)
- [Batch Processing](#batch-processing)
- [Error Handling](#error-handling)
- [Memory Management](#memory-management)
- [Utilities](#utilities)

## Configuration

### SDKConfig

Configuration class for the CognitionFlow SDK.

```python
@dataclass
class SDKConfig:
    api_key: str
    project_id: str
    endpoint: str = "https://api.cognitionflow.com"
    batch_size: int = 100
    flush_interval: float = 5.0
    max_retries: int = 3
    timeout: float = 30.0
    capture_inputs: bool = True
    capture_outputs: bool = True
    capture_errors: bool = True
    environment: str = "development"
    debug: bool = False
    enabled: bool = True
```

#### Parameters

- **api_key** (str): Your CognitionFlow API key
- **project_id** (str): Your CognitionFlow project ID
- **endpoint** (str): API endpoint URL (default: "https://api.cognitionflow.com")
- **batch_size** (int): Number of traces to batch before sending (default: 100)
- **flush_interval** (float): Time in seconds between batch flushes (default: 5.0)
- **max_retries** (int): Maximum number of retry attempts for failed requests (default: 3)
- **timeout** (float): Request timeout in seconds (default: 30.0)
- **capture_inputs** (bool): Whether to capture function inputs (default: True)
- **capture_outputs** (bool): Whether to capture function outputs (default: True)
- **capture_errors** (bool): Whether to capture error information (default: True)
- **environment** (str): Environment name (default: "development")
- **debug** (bool): Enable debug logging (default: False)
- **enabled** (bool): Enable/disable the SDK (default: True)

### Configuration Functions

#### `configure(api_key, project_id, **kwargs)`

Configure the global SDK instance.

```python
cognitionflow.configure(
    api_key="your-api-key",
    project_id="your-project-id",
    endpoint="https://api.cognitionflow.com",
    batch_size=100,
    flush_interval=5.0
)
```

#### `configure_from_config(config: SDKConfig)`

Configure the SDK from an SDKConfig object.

```python
config = SDKConfig(
    api_key="your-api-key",
    project_id="your-project-id"
)
cognitionflow.configure_from_config(config)
```

#### `configure_from_env()`

Configure the SDK from environment variables.

```python
cognitionflow.configure_from_env()
```

Environment variables:
- `COGNITIONFLOW_API_KEY`
- `COGNITIONFLOW_PROJECT_ID`
- `COGNITIONFLOW_ENDPOINT`
- `COGNITIONFLOW_BATCH_SIZE`
- `COGNITIONFLOW_FLUSH_INTERVAL`
- `COGNITIONFLOW_MAX_RETRIES`
- `COGNITIONFLOW_TIMEOUT`
- `COGNITIONFLOW_CAPTURE_INPUTS`
- `COGNITIONFLOW_CAPTURE_OUTPUTS`
- `COGNITIONFLOW_CAPTURE_ERRORS`
- `COGNITIONFLOW_ENVIRONMENT`
- `COGNITIONFLOW_DEBUG`
- `COGNITIONFLOW_ENABLED`

## Core SDK

### CognitionFlowSDK

Main SDK class for CognitionFlow instrumentation.

```python
class CognitionFlowSDK:
    def __init__(self, config: SDKConfig)
    def trace(self, agent_name: str, **kwargs)
    def span(self, operation_name: str, **kwargs)
    def flush(self)
    def shutdown(self)
```

#### Methods

##### `__init__(config: SDKConfig)`

Initialize the SDK with configuration.

##### `trace(agent_name: str, **kwargs)`

Decorator for tracing function calls.

```python
@sdk.trace("my_agent")
def my_function():
    pass

@sdk.trace("my_agent", tags={"version": "1.0"})
async def my_async_function():
    pass
```

**Parameters:**
- **agent_name** (str): Name of the agent performing the operation
- **tags** (dict): Custom tags for the trace
- **metadata** (dict): Custom metadata for the trace

##### `span(operation_name: str, **kwargs)`

Context manager for manual span creation.

```python
async with sdk.span("my_operation") as span:
    span.set_attribute("key", "value")
    # Your code here
```

**Parameters:**
- **operation_name** (str): Name of the operation
- **tags** (dict): Custom tags for the span
- **metadata** (dict): Custom metadata for the span

##### `flush()`

Manually flush pending traces.

##### `shutdown()`

Shutdown the SDK and flush remaining traces.

## Tracing

### TraceData

Data structure for trace information.

```python
@dataclass
class TraceData:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    agent_name: str = ""
    function_name: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Span Methods

#### `set_attribute(key: str, value: Any)`

Set an attribute on the span.

```python
span.set_attribute("user_id", "123")
span.set_attribute("operation_type", "database_query")
```

#### `set_attributes(attributes: Dict[str, Any])`

Set multiple attributes on the span.

```python
span.set_attributes({
    "user_id": "123",
    "operation_type": "database_query",
    "table_name": "users"
})
```

## Context Management

### Context Variables

The SDK uses context variables to maintain trace context across function calls.

```python
# Get current trace context
current_trace_id = cognitionflow.get_current_trace_id()
current_span_id = cognitionflow.get_current_span_id()

# Set trace context
cognitionflow.set_trace_context(trace_id="trace_123", span_id="span_456")
```

### Context Functions

#### `get_current_trace_id() -> Optional[str]`

Get the current trace ID from context.

#### `get_current_span_id() -> Optional[str]`

Get the current span ID from context.

#### `set_trace_context(trace_id: str, span_id: str)`

Set the trace context for the current thread.

## Batch Processing

### BatchProcessor

Handles batching and sending of trace events.

```python
class BatchProcessor:
    def __init__(self, sdk: CognitionFlowSDK)
    def start(self)
    def shutdown(self)
    def flush(self)
```

#### Methods

##### `start()`

Start the batch processor thread.

##### `shutdown()`

Shutdown the batch processor and flush remaining events.

##### `flush()`

Manually flush the current batch.

## Error Handling

### CircuitBreaker

Circuit breaker implementation for API calls.

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60)
    def call(self, func: Callable, *args, **kwargs) -> Any
```

#### Parameters

- **failure_threshold** (int): Number of failures before opening circuit (default: 5)
- **recovery_timeout** (int): Time in seconds before attempting recovery (default: 60)

#### Methods

##### `call(func: Callable, *args, **kwargs) -> Any`

Execute function with circuit breaker protection.

```python
circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
result = circuit_breaker.call(my_function, arg1, arg2)
```

### Circuit States

- **CLOSED**: Normal operation, requests are allowed
- **OPEN**: Circuit is open, requests are blocked
- **HALF_OPEN**: Testing if service has recovered

## Memory Management

### MemoryManager

Manages memory usage and garbage collection.

```python
class MemoryManager:
    def __init__(self, max_memory_mb: int = 100)
    def check_memory_usage(self) -> float
    def should_force_gc(self) -> bool
    def force_gc(self)
```

#### Methods

##### `check_memory_usage() -> float`

Check current memory usage in MB.

##### `should_force_gc() -> bool`

Check if garbage collection should be forced.

##### `force_gc()`

Force garbage collection if memory usage is high.

## Utilities

### Serialization

#### `serialize_value(value: Any) -> Any`

Safely serialize values for transmission.

```python
serialized = cognitionflow.serialize_value({"key": "value"})
```

### Error Handling

#### `capture_error(error: Exception) -> Dict[str, Any]`

Capture error information for tracing.

```python
error_info = cognitionflow.capture_error(ValueError("Test error"))
```

### Trace ID Generation

#### `generate_trace_id() -> str`

Generate a unique trace ID.

```python
trace_id = cognitionflow.generate_trace_id()
```

#### `generate_span_id() -> str`

Generate a unique span ID.

```python
span_id = cognitionflow.generate_span_id()
```

## Global Functions

### `trace(agent_name: str, **kwargs)`

Global trace decorator function.

```python
@cognitionflow.trace("my_agent")
def my_function():
    pass
```

### `span(operation_name: str, **kwargs)`

Global span context manager.

```python
async with cognitionflow.span("my_operation") as span:
    # Your code here
    pass
```

### `flush()`

Global flush function.

```python
cognitionflow.flush()
```

### `shutdown()`

Global shutdown function.

```python
cognitionflow.shutdown()
```

## Error Types

### SDKError

Base exception for SDK errors.

### ConfigurationError

Raised when SDK configuration is invalid.

### NetworkError

Raised when network operations fail.

### SerializationError

Raised when data serialization fails.

## Best Practices

### Performance

1. Use appropriate batch sizes for your workload
2. Disable input/output capture in production if not needed
3. Use circuit breakers for external API calls
4. Monitor memory usage with MemoryManager

### Security

1. Never log sensitive data in traces
2. Use environment variables for configuration
3. Disable debug mode in production
4. Validate all inputs before tracing

### Reliability

1. Always handle exceptions in traced functions
2. Use circuit breakers for external dependencies
3. Implement proper retry logic
4. Monitor SDK health and performance

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.py`: Basic SDK functionality
- `advanced_usage.py`: Advanced features and patterns
- `integration_examples.py`: Framework integration examples
