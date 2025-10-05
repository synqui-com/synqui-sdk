#!/usr/bin/env python3
"""
Integration Examples for Vaquero SDK

This file demonstrates how to integrate Vaquero SDK with popular
Python frameworks and libraries.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import vaquero

# Configure SDK for integration examples
vaquero.configure(
    api_key="integration-example-key",
    project_id="integration-example-project",
    environment="development",
    debug=True
)

# Example 1: FastAPI Integration
def fastapi_integration_example():
    """
    Example of integrating Vaquero with FastAPI.
    
    This would typically be in a separate file like main.py
    """
    
    # FastAPI integration code (commented out since we don't have FastAPI installed)
    """
    from fastapi import FastAPI, Depends, HTTPException
    from fastapi.middleware.base import BaseHTTPMiddleware
    import time
    
    app = FastAPI(title="Vaquero Integration Example")
    
    # Middleware for automatic request tracing
    class VaqueroMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            with vaquero.span("http_request") as span:
                span.set_attribute("method", request.method)
                span.set_attribute("url", str(request.url))
                span.set_attribute("user_agent", request.headers.get("user-agent", ""))
                
                start_time = time.time()
                response = await call_next(request)
                duration = time.time() - start_time
                
                span.set_attribute("status_code", response.status_code)
                span.set_attribute("duration_ms", duration * 1000)
                
                return response
    
    app.add_middleware(VaqueroMiddleware)
    
    @app.get("/users/{user_id}")
    @vaquero.trace(agent_name="user_service")
    async def get_user(user_id: int):
        # Simulate database query
        await asyncio.sleep(0.1)
        return {"user_id": user_id, "name": "John Doe", "email": "john@example.com"}
    
    @app.post("/users")
    @vaquero.trace(agent_name="user_service")
    async def create_user(user_data: dict):
        # Simulate user creation
        await asyncio.sleep(0.2)
        return {"user_id": 123, "status": "created", **user_data}
    """
    
    print("FastAPI integration example (code provided in comments)")
    print("Key features:")
    print("- Automatic HTTP request tracing with middleware")
    print("- Decorator-based endpoint tracing")
    print("- Request/response metadata capture")

# Example 2: SQLAlchemy Integration
class SQLAlchemyIntegration:
    """Example of integrating Vaquero with SQLAlchemy."""
    
    def __init__(self):
        self.queries_executed = 0
    
    @vaquero.trace(agent_name="database_orm")
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a SQLAlchemy query with tracing."""
        with vaquero.span("sqlalchemy_query") as span:
            span.set_attribute("query_type", "select")
            span.set_attribute("query_text", query[:100])  # Truncate for privacy
            span.set_attribute("param_count", len(params) if params else 0)
            
            start_time = time.time()
            
            try:
                # Simulate SQLAlchemy query execution
                await asyncio.sleep(0.1)
                
                # Simulate query results
                results = [
                    {"id": 1, "name": "John Doe", "email": "john@example.com"},
                    {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
                ]
                
                duration = time.time() - start_time
                span.set_attribute("duration_ms", duration * 1000)
                span.set_attribute("result_count", len(results))
                span.set_attribute("status", "success")
                
                self.queries_executed += 1
                return results
                
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error_type", type(e).__name__)
                raise
    
    @vaquero.trace(agent_name="database_orm")
    async def bulk_insert(self, model_class: str, records: List[Dict[str, Any]]) -> int:
        """Bulk insert with SQLAlchemy tracing."""
        with vaquero.span("sqlalchemy_bulk_insert") as span:
            span.set_attribute("model_class", model_class)
            span.set_attribute("record_count", len(records))
            
            start_time = time.time()
            
            try:
                # Simulate bulk insert
                await asyncio.sleep(0.2)
                
                duration = time.time() - start_time
                span.set_attribute("duration_ms", duration * 1000)
                span.set_attribute("status", "success")
                
                return len(records)
                
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error_type", type(e).__name__)
                raise

# Example 3: Celery Task Integration
class CeleryIntegration:
    """Example of integrating Vaquero with Celery tasks."""
    
    @staticmethod
    @vaquero.trace(agent_name="celery_worker")
    def process_data_task(data: Dict[str, Any]) -> Dict[str, Any]:
        """Celery task with tracing."""
        with vaquero.span("celery_task") as span:
            span.set_attribute("task_name", "process_data_task")
            span.set_attribute("data_size", len(str(data)))
            
            # Simulate task processing
            time.sleep(0.5)
            
            result = {
                "processed": True,
                "input_size": len(str(data)),
                "processed_at": time.time()
            }
            
            span.set_attribute("result_size", len(str(result)))
            return result
    
    @staticmethod
    @vaquero.trace(agent_name="celery_worker")
    def send_email_task(recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """Email sending task with tracing."""
        with vaquero.span("celery_email_task") as span:
            span.set_attribute("task_name", "send_email_task")
            span.set_attribute("recipient", recipient)
            span.set_attribute("subject", subject)
            span.set_attribute("body_length", len(body))
            
            # Simulate email sending
            time.sleep(0.3)
            
            return {
                "status": "sent",
                "recipient": recipient,
                "sent_at": time.time()
            }

# Example 4: Django Integration
def django_integration_example():
    """
    Example of integrating Vaquero with Django.
    
    This would typically be in Django settings and middleware files.
    """
    
    # Django integration code (commented out since we don't have Django installed)
    """
    # In settings.py
    import vaquero
    
    # Configure SDK
    vaquero.configure(
        api_key=os.getenv('VAQUERO_API_KEY'),
        project_id=os.getenv('VAQUERO_PROJECT_ID'),
        environment=os.getenv('ENVIRONMENT', 'development')
    )
    
    # In middleware.py
    from django.utils.deprecation import MiddlewareMixin
    import time
    
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
                    span.set_attribute("user_id", getattr(request.user, 'id', None))
            
            return response
    
    # In views.py
    from django.http import JsonResponse
    from django.views.decorators.http import require_http_methods
    
    @require_http_methods(["GET"])
    @vaquero.trace(agent_name="django_view")
    def user_list_view(request):
        # Simulate database query
        users = User.objects.all()[:10]
        return JsonResponse({"users": list(users.values())})
    """
    
    print("Django integration example (code provided in comments)")
    print("Key features:")
    print("- Middleware for automatic request tracing")
    print("- Decorator-based view tracing")
    print("- User context integration")

# Example 5: Flask Integration
class FlaskIntegration:
    """Example of integrating Vaquero with Flask."""
    
    def __init__(self):
        self.request_count = 0
    
    def setup_middleware(self):
        """Setup Flask middleware for tracing."""
        # Flask integration code (commented out since we don't have Flask installed)
        """
        from flask import Flask, request, g
        import time
        
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
        
        @app.route('/users/<int:user_id>')
        @vaquero.trace(agent_name="flask_view")
        def get_user(user_id):
            # Simulate database query
            time.sleep(0.1)
            return {"user_id": user_id, "name": "John Doe"}
        """
        
        print("Flask integration example (code provided in comments)")
        print("Key features:")
        print("- before_request/after_request hooks for tracing")
        print("- Decorator-based route tracing")
        print("- Request context integration")

# Example 6: AsyncIO Integration
class AsyncIOIntegration:
    """Example of integrating Vaquero with AsyncIO patterns."""
    
    @vaquero.trace(agent_name="asyncio_worker")
    async def process_concurrent_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple tasks concurrently with tracing."""
        async with vaquero.span("concurrent_processing") as span:
            span.set_attribute("task_count", len(tasks))
            
            # Create coroutines for concurrent execution
            coroutines = [self._process_single_task(task) for task in tasks]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            # Process results
            successful_results = []
            errors = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors += 1
                    span.set_attribute(f"task_{i}_error", type(result).__name__)
                else:
                    successful_results.append(result)
            
            span.set_attribute("successful_tasks", len(successful_results))
            span.set_attribute("failed_tasks", errors)
            
            return successful_results
    
    @vaquero.trace(agent_name="asyncio_worker")
    async def _process_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task with tracing."""
        with vaquero.span("single_task") as span:
            span.set_attribute("task_id", task.get("id", "unknown"))
            span.set_attribute("task_type", task.get("type", "unknown"))
            
            # Simulate task processing
            await asyncio.sleep(0.1)
            
            return {
                "task_id": task.get("id"),
                "status": "completed",
                "processed_at": time.time()
            }

# Example 7: Custom Decorator for Framework Integration
def framework_trace(agent_name: str, framework: str = "custom"):
    """Custom decorator for framework-specific tracing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with vaquero.span(f"{framework}_{func.__name__}") as span:
                span.set_attribute("agent_name", agent_name)
                span.set_attribute("framework", framework)
                span.set_attribute("function_name", func.__name__)
                span.set_attribute("arg_count", len(args))
                span.set_attribute("kwarg_count", len(kwargs))
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    span.set_attribute("duration_ms", duration * 1000)
                    span.set_attribute("status", "success")
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    span.set_attribute("duration_ms", duration * 1000)
                    span.set_attribute("status", "error")
                    span.set_attribute("error_type", type(e).__name__)
                    
                    raise
        
        return wrapper
    return decorator

# Example 8: Context Manager for Framework Integration
class FrameworkContext:
    """Context manager for framework-specific operations."""
    
    def __init__(self, operation_name: str, framework: str = "custom"):
        self.operation_name = operation_name
        self.framework = framework
        self.span = None
    
    def __enter__(self):
        self.span = vaquero.span(f"{self.framework}_{self.operation_name}")
        self.span.__enter__()
        self.span.set_attribute("framework", self.framework)
        self.span.set_attribute("operation", self.operation_name)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.span.set_attribute("status", "error")
            self.span.set_attribute("error_type", exc_type.__name__)
        else:
            self.span.set_attribute("status", "success")
        
        self.span.__exit__(exc_type, exc_val, exc_tb)

async def main():
    """Run integration examples."""
    print("🚀 Running Vaquero SDK Integration Examples")
    print("=" * 60)
    
    # FastAPI integration
    print("\n1. FastAPI Integration:")
    fastapi_integration_example()
    
    # SQLAlchemy integration
    print("\n2. SQLAlchemy Integration:")
    db_integration = SQLAlchemyIntegration()
    users = await db_integration.execute_query("SELECT * FROM users")
    print(f"   Executed query, returned {len(users)} users")
    
    # Celery integration
    print("\n3. Celery Integration:")
    celery_integration = CeleryIntegration()
    task_result = celery_integration.process_data_task({"key": "value"})
    print(f"   Task result: {task_result['processed']}")
    
    # Django integration
    print("\n4. Django Integration:")
    django_integration_example()
    
    # Flask integration
    print("\n5. Flask Integration:")
    flask_integration = FlaskIntegration()
    flask_integration.setup_middleware()
    
    # AsyncIO integration
    print("\n6. AsyncIO Integration:")
    asyncio_integration = AsyncIOIntegration()
    tasks = [
        {"id": 1, "type": "data_processing"},
        {"id": 2, "type": "email_sending"},
        {"id": 3, "type": "file_upload"}
    ]
    results = await asyncio_integration.process_concurrent_tasks(tasks)
    print(f"   Processed {len(results)} tasks concurrently")
    
    # Custom framework decorator
    print("\n7. Custom Framework Decorator:")
    @framework_trace("custom_agent", "my_framework")
    def custom_function(data: str) -> str:
        time.sleep(0.1)
        return f"processed_{data}"
    
    result = custom_function("test_data")
    print(f"   Custom function result: {result}")
    
    # Framework context manager
    print("\n8. Framework Context Manager:")
    with FrameworkContext("database_transaction", "my_framework") as span:
        span.set_attribute("transaction_id", "txn_123")
        time.sleep(0.05)  # Simulate transaction
        print("   Database transaction completed")
    
    print("\n✅ All integration examples completed!")
    print("\nNote: These examples show how to integrate Vaquero with popular frameworks.")
    print("Choose the integration pattern that best fits your application architecture.")

if __name__ == "__main__":
    asyncio.run(main())
