#!/usr/bin/env python3
"""
Advanced Usage Examples for CognitionFlow SDK

This file demonstrates advanced features and patterns for the CognitionFlow SDK,
including custom configurations, performance monitoring, and integration patterns.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import cognitionflow
from cognitionflow import SDKConfig

# Configure logging to see SDK debug information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example 1: Custom SDK Configuration
def setup_custom_sdk():
    """Set up SDK with custom configuration for different environments."""
    
    # Development configuration
    dev_config = SDKConfig(
        api_key="dev-api-key",
        project_id="dev-project",
        endpoint="http://localhost:8000/api/v1",
        batch_size=10,  # Smaller batches for development
        flush_interval=2.0,  # Faster flushing
        max_retries=1,  # Fewer retries for faster feedback
        capture_inputs=True,
        capture_outputs=True,
        capture_errors=True,
        environment="development",
        debug=True,
        enabled=True
    )
    
    # Production configuration
    prod_config = SDKConfig(
        api_key="prod-api-key",
        project_id="prod-project",
        endpoint="https://api.cognitionflow.com/api/v1",
        batch_size=100,  # Larger batches for efficiency
        flush_interval=10.0,  # Less frequent flushing
        max_retries=3,  # More retries for reliability
        capture_inputs=False,  # Disable for privacy
        capture_outputs=False,  # Disable for privacy
        capture_errors=True,  # Keep error tracking
        environment="production",
        debug=False,
        enabled=True
    )
    
    # Use development config for this example
    cognitionflow.configure_from_config(dev_config)
    return dev_config

# Example 2: Performance Monitoring Integration
class PerformanceMonitor:
    """Custom performance monitoring with CognitionFlow integration."""
    
    def __init__(self):
        self.metrics = {}
    
    @cognitionflow.trace(agent_name="performance_monitor")
    def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a performance metric with tracing."""
        with cognitionflow.span("record_metric") as span:
            span.set_attribute("metric_name", metric_name)
            span.set_attribute("metric_value", value)
            if tags:
                for key, val in tags.items():
                    span.set_attribute(f"tag_{key}", val)
            
            self.metrics[metric_name] = {
                "value": value,
                "timestamp": time.time(),
                "tags": tags or {}
            }
    
    @cognitionflow.trace(agent_name="performance_monitor")
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with cognitionflow.span("get_metrics_summary") as span:
            span.set_attribute("metric_count", len(self.metrics))
            
            summary = {
                "total_metrics": len(self.metrics),
                "metrics": self.metrics,
                "generated_at": time.time()
            }
            
            return summary

# Example 3: Database Operation Tracing
class DatabaseClient:
    """Database client with comprehensive tracing."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.performance_monitor = PerformanceMonitor()
    
    @cognitionflow.trace(agent_name="database_client")
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a database query with tracing."""
        start_time = time.time()
        
        try:
            # Simulate database query
            await asyncio.sleep(0.1)
            
            # Record performance metric
            duration = time.time() - start_time
            self.performance_monitor.record_metric(
                "query_duration",
                duration,
                {"query_type": "select", "table": "users"}
            )
            
            # Simulate query results
            results = [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
            ]
            
            return results
            
        except Exception as e:
            # Record error metric
            self.performance_monitor.record_metric(
                "query_errors",
                1,
                {"error_type": type(e).__name__}
            )
            raise
    
    @cognitionflow.trace(agent_name="database_client")
    async def batch_insert(self, table: str, records: List[Dict[str, Any]]) -> int:
        """Batch insert with tracing."""
        with cognitionflow.span("batch_insert") as span:
            span.set_attribute("table_name", table)
            span.set_attribute("record_count", len(records))
            
            start_time = time.time()
            
            try:
                # Simulate batch insert
                await asyncio.sleep(0.2)
                
                duration = time.time() - start_time
                self.performance_monitor.record_metric(
                    "batch_insert_duration",
                    duration,
                    {"table": table, "record_count": len(records)}
                )
                
                return len(records)  # Return number of inserted records
                
            except Exception as e:
                self.performance_monitor.record_metric(
                    "batch_insert_errors",
                    1,
                    {"table": table, "error_type": type(e).__name__}
                )
                raise

# Example 4: API Client with Circuit Breaker Integration
class APIClient:
    """API client with circuit breaker and tracing."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.circuit_breaker = cognitionflow.CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30
        )
    
    @cognitionflow.trace(agent_name="api_client")
    async def make_request(self, endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make an API request with circuit breaker protection."""
        with cognitionflow.span("make_request") as span:
            span.set_attribute("endpoint", endpoint)
            span.set_attribute("method", method)
            span.set_attribute("base_url", self.base_url)
            
            try:
                # Use circuit breaker to protect the actual request
                result = await self.circuit_breaker.call(
                    self._actual_request,
                    endpoint,
                    method,
                    data
                )
                
                span.set_attribute("response_status", "success")
                return result
                
            except Exception as e:
                span.set_attribute("response_status", "error")
                span.set_attribute("error_type", type(e).__name__)
                raise
    
    async def _actual_request(self, endpoint: str, method: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Actual HTTP request implementation."""
        # Simulate API call
        await asyncio.sleep(0.1)
        
        # Simulate occasional failures
        if "fail" in endpoint:
            raise ConnectionError("Simulated API failure")
        
        return {
            "status": "success",
            "data": {"endpoint": endpoint, "method": method},
            "timestamp": time.time()
        }

# Example 5: Workflow Orchestration with Tracing
class WorkflowOrchestrator:
    """Workflow orchestrator with comprehensive tracing."""
    
    def __init__(self):
        self.db_client = DatabaseClient("postgresql://localhost/db")
        self.api_client = APIClient("https://api.example.com")
        self.performance_monitor = PerformanceMonitor()
    
    @cognitionflow.trace(agent_name="workflow_orchestrator")
    async def process_user_registration(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user registration workflow with tracing."""
        workflow_id = f"registration_{int(time.time())}"
        
        async with cognitionflow.span("process_user_registration") as workflow_span:
            workflow_span.set_attribute("workflow_id", workflow_id)
            workflow_span.set_attribute("user_email", user_data.get("email", "unknown"))
            
            try:
                # Step 1: Validate user data
                async with cognitionflow.span("validate_user_data") as validate_span:
                    validate_span.set_attribute("validation_type", "registration")
                    validated_data = await self._validate_user_data(user_data)
                
                # Step 2: Check if user exists
                async with cognitionflow.span("check_user_exists") as check_span:
                    existing_users = await self.db_client.execute_query(
                        "SELECT * FROM users WHERE email = %(email)s",
                        {"email": validated_data["email"]}
                    )
                    check_span.set_attribute("existing_user_count", len(existing_users))
                
                if existing_users:
                    raise ValueError("User already exists")
                
                # Step 3: Create user in database
                async with cognitionflow.span("create_user") as create_span:
                    user_id = await self._create_user(validated_data)
                    create_span.set_attribute("user_id", user_id)
                
                # Step 4: Send welcome email
                async with cognitionflow.span("send_welcome_email") as email_span:
                    email_result = await self.api_client.make_request(
                        "/send-email",
                        "POST",
                        {
                            "to": validated_data["email"],
                            "template": "welcome",
                            "user_id": user_id
                        }
                    )
                    email_span.set_attribute("email_sent", email_result.get("status") == "success")
                
                # Step 5: Record metrics
                self.performance_monitor.record_metric(
                    "user_registrations",
                    1,
                    {"workflow_id": workflow_id}
                )
                
                return {
                    "status": "success",
                    "user_id": user_id,
                    "workflow_id": workflow_id
                }
                
            except Exception as e:
                # Record error metrics
                self.performance_monitor.record_metric(
                    "user_registration_errors",
                    1,
                    {"workflow_id": workflow_id, "error_type": type(e).__name__}
                )
                raise
    
    async def _validate_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user data."""
        await asyncio.sleep(0.05)  # Simulate validation
        
        required_fields = ["email", "name"]
        for field in required_fields:
            if field not in user_data:
                raise ValueError(f"Missing required field: {field}")
        
        return user_data
    
    async def _create_user(self, user_data: Dict[str, Any]) -> str:
        """Create user in database."""
        user_id = f"user_{int(time.time())}"
        
        await self.db_client.batch_insert("users", [{
            "id": user_id,
            "email": user_data["email"],
            "name": user_data["name"],
            "created_at": time.time()
        }])
        
        return user_id

# Example 6: Custom Context Manager for Resource Management
@asynccontextmanager
async def traced_resource_manager(resource_name: str, **attributes):
    """Custom context manager for resource management with tracing."""
    async with cognitionflow.span(f"manage_{resource_name}") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        
        # Simulate resource acquisition
        span.set_attribute("resource_status", "acquiring")
        await asyncio.sleep(0.01)
        
        try:
            span.set_attribute("resource_status", "acquired")
            yield span
        finally:
            span.set_attribute("resource_status", "releasing")
            await asyncio.sleep(0.01)
            span.set_attribute("resource_status", "released")

# Example 7: Batch Processing with Progress Tracking
@cognitionflow.trace(agent_name="batch_processor")
async def process_large_dataset(dataset: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
    """Process large dataset with progress tracking."""
    total_items = len(dataset)
    processed_items = 0
    errors = 0
    
    async with cognitionflow.span("process_large_dataset") as main_span:
        main_span.set_attribute("total_items", total_items)
        main_span.set_attribute("batch_size", batch_size)
        
        for i in range(0, total_items, batch_size):
            batch = dataset[i:i + batch_size]
            batch_number = i // batch_size + 1
            
            async with cognitionflow.span(f"process_batch_{batch_number}") as batch_span:
                batch_span.set_attribute("batch_number", batch_number)
                batch_span.set_attribute("batch_size", len(batch))
                batch_span.set_attribute("progress_percent", (i / total_items) * 100)
                
                try:
                    # Process batch
                    await asyncio.sleep(0.1)  # Simulate processing
                    processed_items += len(batch)
                    
                    batch_span.set_attribute("batch_status", "completed")
                    
                except Exception as e:
                    errors += 1
                    batch_span.set_attribute("batch_status", "failed")
                    batch_span.set_attribute("error_type", type(e).__name__)
                    logger.error(f"Batch {batch_number} failed: {e}")
        
        main_span.set_attribute("processed_items", processed_items)
        main_span.set_attribute("error_count", errors)
        main_span.set_attribute("success_rate", (processed_items - errors) / total_items * 100)
        
        return {
            "total_items": total_items,
            "processed_items": processed_items,
            "errors": errors,
            "success_rate": (processed_items - errors) / total_items * 100
        }

async def main():
    """Run advanced examples."""
    print("🚀 Running Advanced CognitionFlow SDK Examples")
    print("=" * 60)
    
    # Setup custom SDK
    config = setup_custom_sdk()
    print(f"\n1. Custom SDK Configuration:")
    print(f"   Environment: {config.environment}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Debug Mode: {config.debug}")
    
    # Performance monitoring
    print("\n2. Performance Monitoring:")
    monitor = PerformanceMonitor()
    monitor.record_metric("api_response_time", 0.15, {"endpoint": "/users"})
    monitor.record_metric("database_query_time", 0.08, {"table": "users"})
    summary = monitor.get_metrics_summary()
    print(f"   Recorded {summary['total_metrics']} metrics")
    
    # Database operations
    print("\n3. Database Operations:")
    db_client = DatabaseClient("postgresql://localhost/db")
    users = await db_client.execute_query("SELECT * FROM users")
    print(f"   Fetched {len(users)} users")
    
    # API client with circuit breaker
    print("\n4. API Client with Circuit Breaker:")
    api_client = APIClient("https://api.example.com")
    try:
        result = await api_client.make_request("/users", "GET")
        print(f"   API Response: {result['status']}")
    except Exception as e:
        print(f"   API Error: {e}")
    
    # Workflow orchestration
    print("\n5. Workflow Orchestration:")
    orchestrator = WorkflowOrchestrator()
    try:
        workflow_result = await orchestrator.process_user_registration({
            "email": "newuser@example.com",
            "name": "New User"
        })
        print(f"   Workflow Result: {workflow_result['status']}")
    except Exception as e:
        print(f"   Workflow Error: {e}")
    
    # Resource management
    print("\n6. Resource Management:")
    async with traced_resource_manager("database_connection", pool_size=10, timeout=30):
        print("   Resource acquired and released")
    
    # Batch processing
    print("\n7. Batch Processing:")
    large_dataset = [{"id": i, "data": f"item_{i}"} for i in range(50)]
    batch_result = await process_large_dataset(large_dataset, batch_size=10)
    print(f"   Processed {batch_result['processed_items']} items")
    print(f"   Success Rate: {batch_result['success_rate']:.1f}%")
    
    print("\n✅ All advanced examples completed!")
    print("\nNote: These examples demonstrate advanced patterns for production use.")
    print("Make sure to configure appropriate settings for your environment.")

if __name__ == "__main__":
    asyncio.run(main())
