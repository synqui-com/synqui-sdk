# FastAPI Integration Guide

## Overview

FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints. This guide shows how to integrate Vaquero SDK with FastAPI applications for comprehensive API observability.

## Prerequisites

- Python 3.8+
- FastAPI installed: `pip install fastapi uvicorn`
- Vaquero SDK installed: `pip install vaquero-sdk`

## Installation & Setup

### 1. Basic Setup

```python
# main.py
from fastapi import FastAPI
import vaquero

# Initialize Vaquero SDK
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    mode="development"  # Use "production" for optimized settings
)

app = FastAPI(
    title="My FastAPI Application",
    description="API with Vaquero observability"
)

@app.get("/")
@vaquero.trace(agent_name="api_service")
async def root():
    """Root endpoint with automatic tracing."""
    return {"message": "Hello World", "status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Middleware Integration

For automatic request/response tracing across all endpoints:

```python
# middleware.py
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import vaquero

class VaqueroMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic HTTP request tracing."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(id(request))  # Simple request ID

        with vaquero.span("http_request") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("method", request.method)
            span.set_attribute("url", str(request.url))
            span.set_attribute("user_agent", request.headers.get("user-agent", ""))
            span.set_attribute("content_type", request.headers.get("content-type", ""))

            # Record request size for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                span.set_attribute("request_size_bytes", len(body))

            start_time = time.time()
            response = await call_next(request)
            duration = time.time() - start_time

            # Record response details
            span.set_attribute("status_code", response.status_code)
            span.set_attribute("response_time_ms", duration * 1000)

            # Record response size (be careful with large responses)
            if hasattr(response, 'headers') and response.headers.get("content-length"):
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) < 1000000:  # < 1MB
                    span.set_attribute("response_size_bytes", int(content_length))

            return response

# main.py
from fastapi import FastAPI
from middleware import VaqueroMiddleware
import vaquero

vaquero.init(api_key="your-api-key")

app = FastAPI()
app.add_middleware(VaqueroMiddleware)  # Add middleware first

@app.get("/users/{user_id}")
@vaquero.trace(agent_name="user_service")
async def get_user(user_id: int):
    """Get user by ID with comprehensive tracing."""
    with vaquero.span("database_lookup") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("operation", "user_retrieval")

        # Your database logic here
        user = await database.get_user(user_id)

        if user:
            span.set_attribute("user_found", True)
            span.set_attribute("user_status", user.get("status", "unknown"))
        else:
            span.set_attribute("user_found", False)

        return user
```

## Advanced Integration Patterns

### Authentication Integration

```python
# auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import vaquero

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency for authenticated user with tracing."""
    with vaquero.span("token_validation") as span:
        span.set_attribute("token_prefix", credentials.credentials[:10] + "...")

        try:
            # Validate token
            user = validate_token(credentials.credentials)

            span.set_attribute("user_id", user["id"])
            span.set_attribute("token_valid", True)
            span.set_attribute("user_role", user.get("role", "user"))

            return user

        except Exception as e:
            span.set_attribute("token_valid", False)
            span.set_attribute("error_type", type(e).__name__)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

# main.py
@app.get("/protected")
@vaquero.trace(agent_name="protected_service")
async def protected_endpoint(current_user: dict = Depends(get_current_user)):
    """Protected endpoint with user context tracing."""
    with vaquero.span("user_authorization") as span:
        span.set_attribute("user_id", current_user["id"])
        span.set_attribute("user_role", current_user.get("role", "user"))
        span.set_attribute("required_permission", "read_data")

        # Check permissions
        if not has_permission(current_user, "read_data"):
            span.set_attribute("permission_granted", False)
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        else:
            span.set_attribute("permission_granted", True)

        return {"message": "Access granted", "user_id": current_user["id"]}
```

### Database Integration

```python
# database.py
import asyncpg
import vaquero

class DatabaseClient:
    """Async database client with tracing."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None

    async def connect(self):
        """Connect to database with tracing."""
        with vaquero.span("database_connection") as span:
            span.set_attribute("connection_type", "asyncpg_pool")

            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20
            )

            span.set_attribute("connection_successful", True)
            span.set_attribute("pool_size", 20)

    @vaquero.trace(agent_name="database_operation")
    async def get_user(self, user_id: int) -> dict:
        """Get user with query tracing."""
        async with self.pool.acquire() as connection:
            with vaquero.span("user_query") as span:
                span.set_attribute("query_type", "SELECT")
                span.set_attribute("table", "users")
                span.set_attribute("user_id", user_id)

                query = "SELECT * FROM users WHERE id = $1"
                start_time = time.time()

                try:
                    user = await connection.fetchrow(query, user_id)
                    duration = time.time() - start_time

                    span.set_attribute("query_duration_ms", duration * 1000)
                    span.set_attribute("rows_returned", 1 if user else 0)
                    span.set_attribute("query_successful", True)

                    return dict(user) if user else None

                except Exception as e:
                    duration = time.time() - start_time
                    span.set_attribute("query_duration_ms", duration * 1000)
                    span.set_attribute("query_successful", False)
                    span.set_attribute("error_type", type(e).__name__)
                    raise

# main.py
db_client = DatabaseClient("postgresql://user:pass@localhost/mydb")

@app.on_event("startup")
async def startup():
    await db_client.connect()

@app.get("/users/{user_id}")
@vaquero.trace(agent_name="user_api")
async def get_user(user_id: int):
    """Get user endpoint with database tracing."""
    return await db_client.get_user(user_id)
```

### Background Tasks Integration

```python
# tasks.py
import asyncio
import vaquero

@vaquero.trace(agent_name="background_tasks")
class BackgroundTaskManager:
    """Background task manager with tracing."""

    def __init__(self):
        self.tasks_running = 0

    @vaquero.trace(agent_name="email_task")
    async def send_welcome_email(self, user_id: str, email: str) -> dict:
        """Send welcome email with tracing."""
        with vaquero.span("email_processing") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("email", email)
            span.set_attribute("email_type", "welcome")

            # Simulate email sending
            await asyncio.sleep(0.1)

            span.set_attribute("email_sent", True)
            span.set_attribute("email_provider", "sendgrid")

            return {
                "status": "sent",
                "user_id": user_id,
                "sent_at": time.time()
            }

    @vaquero.trace(agent_name="batch_processor")
    async def process_user_batch(self, user_ids: list) -> dict:
        """Process multiple users with parallel tracing."""
        with vaquero.span("batch_processing") as span:
            span.set_attribute("batch_size", len(user_ids))
            span.set_attribute("processing_strategy", "parallel")

            # Process users in parallel
            tasks = []
            for user_id in user_ids:
                task = self.send_welcome_email(user_id, f"user{user_id}@example.com")
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze results
            successful = len([r for r in results if not isinstance(r, Exception)])
            failed = len(tasks) - successful

            span.set_attribute("successful_tasks", successful)
            span.set_attribute("failed_tasks", failed)
            span.set_attribute("success_rate", successful / len(tasks))

            return {
                "total_users": len(user_ids),
                "successful": successful,
                "failed": failed,
                "results": results
            }

# main.py
task_manager = BackgroundTaskManager()

@app.post("/users/batch-process")
@vaquero.trace(agent_name="batch_api")
async def batch_process_users(user_ids: list):
    """Endpoint for batch processing with tracing."""
    return await task_manager.process_user_batch(user_ids)
```

## Configuration Options

### Development Configuration
```python
# For development - maximum observability
vaquero.init(
    api_key="your-dev-key",
    mode="development",
    debug=True,
    batch_size=10,        # Smaller batches for faster feedback
    flush_interval=2.0,   # More frequent flushing
    capture_inputs=True,  # Full request/response capture
    capture_outputs=True, # Full debugging information
    auto_instrument_llm=True  # Enable LLM tracing if used
)
```

### Production Configuration
```python
# For production - optimized performance
vaquero.init(
    api_key="your-prod-key",
    mode="production",
    batch_size=100,       # Larger batches for efficiency
    flush_interval=10.0,  # Less frequent flushing
    capture_inputs=False,    # Privacy protection
    capture_outputs=False,   # Performance optimization
    auto_instrument_llm=False,  # Disable for performance
    capture_system_prompts=False  # Privacy protection
)
```

## Best Practices

### 1. Middleware Order
```python
# Add middleware in the correct order
app = FastAPI()

# 1. CORS middleware (if needed)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# 2. Authentication middleware (if needed)
app.add_middleware(AuthMiddleware)

# 3. Vaquero middleware for request tracing
app.add_middleware(VaqueroMiddleware)

# 4. Other custom middleware
```

### 2. Error Handling
```python
@app.exception_handler(Exception)
@vaquero.trace(agent_name="error_handler")
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with tracing."""
    with vaquero.span("global_error") as span:
        span.set_attribute("error_type", type(exc).__name__)
        span.set_attribute("error_message", str(exc))
        span.set_attribute("request_path", request.url.path)
        span.set_attribute("request_method", request.method)

        # Log error for monitoring
        logger.error(f"Unhandled exception: {exc}", extra={
            "request_id": getattr(request.state, "request_id", None),
            "error_type": type(exc).__name__
        })

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

### 3. Request Context
```python
# Pass request context through your application
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Add to Vaquero context
    with vaquero.span("request_context") as span:
        span.set_attribute("request_id", request_id)

        response = await call_next(request)

        span.set_attribute("response_status", response.status_code)
        return response

# Use context in your handlers
@app.get("/users/{user_id}")
@vaquero.trace(agent_name="user_service")
async def get_user(request: Request, user_id: int):
    """Handler with request context."""
    request_id = request.state.request_id

    with vaquero.span("user_lookup") as span:
        span.set_attribute("request_id", request_id)
        span.set_attribute("user_id", user_id)

        # Your logic here
        user = await database.get_user(user_id)

        span.set_attribute("user_found", user is not None)

        return user
```

## Performance Considerations

### Sampling for High-Traffic APIs
```python
# Implement sampling for high-traffic endpoints
import random

class SamplingMiddleware(BaseHTTPMiddleware):
    """Middleware with intelligent sampling."""

    async def dispatch(self, request: Request, call_next):
        # Sample based on endpoint or user
        should_sample = self._should_sample_request(request)

        if should_sample:
            with vaquero.span("sampled_request") as span:
                span.set_attribute("sampled", True)
                span.set_attribute("sample_rate", 0.1)  # 10% sampling
                response = await call_next(request)
                span.set_attribute("status_code", response.status_code)
                return response
        else:
            # No tracing for this request
            return await call_next(request)

    def _should_sample_request(self, request: Request) -> bool:
        """Determine if request should be sampled."""
        # Sample all errors
        if request.url.path.startswith("/error"):
            return True

        # Sample based on user ID (if authenticated)
        user_id = request.headers.get("x-user-id")
        if user_id:
            # Hash user ID to get consistent sampling
            import hashlib
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            return (hash_value % 100) < 10  # 10% sampling

        # Default sampling rate
        return random.random() < 0.05  # 5% sampling

# Use sampling middleware for high-traffic applications
app.add_middleware(SamplingMiddleware)
```

### Database Connection Pool Monitoring
```python
# Monitor database connection pool
@app.on_event("startup")
async def startup_monitoring():
    """Set up database monitoring."""
    # Monitor pool status every 30 seconds
    async def monitor_pool():
        while True:
            with vaquero.span("pool_monitoring") as span:
                pool_size = len(db_client.pool._holders) if db_client.pool else 0
                span.set_attribute("pool_size", pool_size)
                span.set_attribute("pool_utilization", pool_size / 20)  # Assuming max 20

                # Alert on high utilization
                if pool_size > 15:  # 75% utilization
                    span.set_attribute("high_utilization", True)

            await asyncio.sleep(30)

    # Start monitoring task
    asyncio.create_task(monitor_pool())
```

## Testing Integration

### Testing Traced Endpoints
```python
# test_main.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_traced_endpoint():
    """Test that endpoints are properly traced."""
    response = client.get("/users/123")

    assert response.status_code == 200

    # Verify response structure
    data = response.json()
    assert "user_id" in data
    assert "name" in data

def test_error_tracing():
    """Test that errors are properly traced."""
    # Test 404 error
    response = client.get("/users/99999")

    assert response.status_code == 404

    # The 404 error should be traced automatically by middleware
    # In a real test, you might verify trace data exists
```

### Performance Testing
```python
def test_endpoint_performance():
    """Test endpoint performance with tracing overhead."""
    import time

    # Measure baseline (without tracing)
    start = time.time()
    # Call endpoint logic directly
    baseline_time = time.time() - start

    # Measure with tracing
    start = time.time()
    response = client.get("/users/123")
    traced_time = time.time() - start

    # Tracing overhead should be minimal (< 10ms)
    overhead = traced_time - baseline_time
    assert overhead < 0.01  # Less than 10ms overhead
```

## Troubleshooting

### Common Issues

#### Traces not appearing in dashboard
```python
# Check SDK configuration
import vaquero

sdk = vaquero.get_default_sdk()
print(f"SDK enabled: {sdk.config.enabled}")
print(f"API key set: {bool(sdk.config.api_key)}")

# Manually flush traces
vaquero.flush()

# Check network connectivity
import requests
try:
    response = requests.get("https://api.vaquero.app/health", timeout=5)
    print("API reachable")
except Exception as e:
    print(f"Network error: {e}")
```

#### Performance degradation
```python
# Monitor middleware overhead
@app.middleware("http")
async def performance_monitoring_middleware(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    if duration > 1.0:  # More than 1 second
        print(f"Slow request: {request.method} {request.url} took {duration:.2f}s")

    return response
```

#### Memory usage issues
```python
# Monitor memory usage
import psutil
import gc

def check_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024

    if memory_mb > 100:  # More than 100MB
        print(f"High memory usage: {memory_mb:.1f} MB")
        gc.collect()  # Force garbage collection
```

## Complete Example

Here's a complete FastAPI application with comprehensive Vaquero integration:

```python
# main.py
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.security import HTTPBearer
import asyncio
import time
import vaquero

# Initialize SDK
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    mode="production"
)

# Security
security = HTTPBearer()

# Middleware
class VaqueroMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(time.time_ns())

        with vaquero.span("http_request") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("method", request.method)
            span.set_attribute("url", str(request.url))

            start_time = time.time()
            response = await call_next(request)
            duration = time.time() - start_time

            span.set_attribute("status_code", response.status_code)
            span.set_attribute("duration_ms", duration * 1000)

            return response

app = FastAPI()
app.add_middleware(VaqueroMiddleware)

# Database client
class DatabaseClient:
    @vaquero.trace(agent_name="database")
    async def get_user(self, user_id: int) -> dict:
        """Get user from database."""
        with vaquero.span("user_query") as span:
            span.set_attribute("user_id", user_id)

            # Simulate database query
            await asyncio.sleep(0.05)

            user = {"id": user_id, "name": "John Doe", "email": "john@example.com"}
            span.set_attribute("user_found", True)

            return user

db = DatabaseClient()

# Authentication
def get_current_user(token: str = Depends(security)):
    """Validate authentication token."""
    with vaquero.span("auth_validation") as span:
        span.set_attribute("token_provided", bool(token))

        # Validate token logic here
        if token.credentials == "valid-token":
            user = {"id": 123, "role": "user"}
            span.set_attribute("user_id", user["id"])
            span.set_attribute("auth_successful", True)
            return user
        else:
            span.set_attribute("auth_successful", False)
            raise HTTPException(status_code=401, detail="Invalid token")

# Endpoints
@app.get("/")
@vaquero.trace(agent_name="health_check")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/users/{user_id}")
@vaquero.trace(agent_name="user_service")
async def get_user(user_id: int, current_user: dict = Depends(get_current_user)):
    """Get user by ID."""
    with vaquero.span("user_lookup") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("requester_id", current_user["id"])

        user = await db.get_user(user_id)

        span.set_attribute("lookup_successful", user is not None)

        return user

@app.post("/users/{user_id}/action")
@vaquero.trace(agent_name="user_action")
async def user_action(user_id: int, action: str, current_user: dict = Depends(get_current_user)):
    """Perform action on user."""
    with vaquero.span("action_processing") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("action", action)
        span.set_attribute("requester_id", current_user["id"])

        # Simulate action processing
        await asyncio.sleep(0.1)

        span.set_attribute("action_successful", True)

        return {"status": "completed", "action": action}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

This integration provides comprehensive observability for your FastAPI application, including:

- **Request/response tracing**: All HTTP requests automatically traced
- **Authentication tracking**: User authentication and authorization traced
- **Database operations**: All database queries traced with performance metrics
- **Error handling**: Exceptions properly categorized and traced
- **Performance monitoring**: Response times, memory usage, and throughput tracked

## Next Steps

- Check out other framework integration guides (Django, Flask, etc.)
- Review the [Best Practices](../BEST_PRACTICES.md) guide for optimization tips
- See the [Troubleshooting Guide](../TROUBLESHOOTING.md) for common issues
