# Celery Integration Guide

## Overview

Celery is a distributed task queue framework for Python. This guide shows how to integrate Vaquero SDK with Celery applications for comprehensive task observability.

## Prerequisites

- Python 3.8+
- Celery installed: `pip install celery`
- Message broker (Redis/RabbitMQ)
- Vaquero SDK installed: `pip install vaquero-sdk`

## Installation & Setup

### 1. Basic Setup

```python
# celery_app.py
from celery import Celery
import vaquero

# Initialize Vaquero SDK
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    mode="development"
)

# Celery configuration
app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@app.task(bind=True)
@vaquero.trace(agent_name="celery_worker")
def process_user_data(self, user_id: str, data: dict) -> dict:
    """Process user data with comprehensive tracing."""
    with vaquero.span("data_processing") as span:
        span.set_attribute("task_id", self.request.id)
        span.set_attribute("user_id", user_id)
        span.set_attribute("data_size", len(str(data)))
        span.set_attribute("worker_hostname", self.request.hostname)

        try:
            # Simulate data processing
            import time
            time.sleep(0.1)

            result = {
                "user_id": user_id,
                "processed": True,
                "result_size": len(str(data)) * 2
            }

            span.set_attribute("processing_successful", True)
            span.set_attribute("result_size", len(str(result)))

            return result

        except Exception as e:
            span.set_attribute("processing_successful", False)
            span.set_attribute("error_type", type(e).__name__)
            raise

@app.task(bind=True)
@vaquero.trace(agent_name="celery_worker")
def send_notification(self, user_id: str, message: str) -> dict:
    """Send notification with tracing."""
    with vaquero.span("notification_processing") as span:
        span.set_attribute("task_id", self.request.id)
        span.set_attribute("user_id", user_id)
        span.set_attribute("message_length", len(message))
        span.set_attribute("notification_type", "email")

        try:
            # Simulate notification sending
            import time
            time.sleep(0.05)

            span.set_attribute("notification_sent", True)
            span.set_attribute("delivery_time_ms", 50)

            return {
                "user_id": user_id,
                "message": message,
                "status": "sent",
                "sent_at": time.time()
            }

        except Exception as e:
            span.set_attribute("notification_sent", False)
            span.set_attribute("error_type", type(e).__name__)
            raise

if __name__ == '__main__':
    app.start()
```

### 2. Worker Process Setup

For proper tracing in worker processes:

```python
# worker.py
import os
import vaquero

# Initialize SDK in worker process
vaquero.init(
    api_key=os.getenv('VAQUERO_API_KEY'),
    project_id=os.getenv('VAQUERO_PROJECT_ID'),
    mode="production"
)

from celery_app import app

if __name__ == '__main__':
    # Start Celery worker
    app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=4',
        '--hostname=worker1@%h'
    ])
```

### 3. Task Monitoring

Monitor task execution and performance:

```python
# monitoring.py
import vaquero

@vaquero.trace(agent_name="task_monitor")
class TaskMonitor:
    """Monitor Celery task performance."""

    def __init__(self):
        self.task_metrics = {}
        self.slow_tasks = []

    @vaquero.trace(agent_name="task_completion_tracking")
    def track_task_completion(self, task_name: str, duration: float, success: bool, task_id: str):
        """Track task completion for analysis."""
        with vaquero.span("task_completion") as span:
            span.set_attribute("task_name", task_name)
            span.set_attribute("duration_seconds", duration)
            span.set_attribute("success", success)
            span.set_attribute("task_id", task_id)

            # Track metrics
            if task_name not in self.task_metrics:
                self.task_metrics[task_name] = {
                    "count": 0,
                    "total_duration": 0,
                    "success_count": 0,
                    "failure_count": 0
                }

            metrics = self.task_metrics[task_name]
            metrics["count"] += 1
            metrics["total_duration"] += duration

            if success:
                metrics["success_count"] += 1
            else:
                metrics["failure_count"] += 1

            # Track slow tasks
            if duration > 5.0:  # More than 5 seconds
                self.slow_tasks.append({
                    "task_name": task_name,
                    "duration": duration,
                    "task_id": task_id,
                    "timestamp": time.time()
                })
                span.set_attribute("slow_task_detected", True)

    @vaquero.trace(agent_name="task_analysis")
    def analyze_task_performance(self) -> dict:
        """Analyze task performance patterns."""
        with vaquero.span("performance_analysis") as span:
            if not self.task_metrics:
                return {"error": "No tasks tracked"}

            analysis = {
                "total_tasks": sum(m["count"] for m in self.task_metrics.values()),
                "task_types": {},
                "performance_insights": []
            }

            for task_name, metrics in self.task_metrics.items():
                avg_duration = metrics["total_duration"] / metrics["count"]
                success_rate = metrics["success_count"] / metrics["count"]

                task_analysis = {
                    "count": metrics["count"],
                    "avg_duration": avg_duration,
                    "success_rate": success_rate,
                    "total_duration": metrics["total_duration"]
                }

                analysis["task_types"][task_name] = task_analysis

                # Generate insights
                if avg_duration > 10.0:  # Slow average
                    analysis["performance_insights"].append({
                        "type": "slow_task",
                        "task_name": task_name,
                        "avg_duration": avg_duration,
                        "recommendation": f"Optimize {task_name} for better performance"
                    })

                if success_rate < 0.9:  # Low success rate
                    analysis["performance_insights"].append({
                        "type": "unreliable_task",
                        "task_name": task_name,
                        "success_rate": success_rate,
                        "recommendation": f"Review {task_name} for reliability issues"
                    })

            span.set_attribute("tasks_analyzed", len(self.task_metrics))
            span.set_attribute("insights_generated", len(analysis["performance_insights"]))

            return analysis

# Global task monitor
task_monitor = TaskMonitor()

# Enhanced task decorator
def monitored_task(*args, **kwargs):
    """Task decorator with monitoring."""
    def decorator(func):
        @vaquero.trace(agent_name="monitored_task")
        def wrapper(*args, **kwargs):
            task_name = func.__name__
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                task_monitor.track_task_completion(
                    task_name=task_name,
                    duration=duration,
                    success=True,
                    task_id=getattr(func, 'request', {}).get('id', 'unknown')
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                task_monitor.track_task_completion(
                    task_name=task_name,
                    duration=duration,
                    success=False,
                    task_id=getattr(func, 'request', {}).get('id', 'unknown')
                )

                raise

        return wrapper
    return decorator
```

## Advanced Integration Patterns

### Task Chain Tracing

```python
# chains.py
import vaquero

@vaquero.trace(agent_name="task_chains")
class TaskChainManager:
    """Manage complex task chains with tracing."""

    @vaquero.trace(agent_name="chain_execution")
    async def execute_user_onboarding_chain(self, user_id: str) -> dict:
        """Execute user onboarding workflow."""
        chain_id = f"chain_{user_id}_{int(time.time())}"

        with vaquero.span("onboarding_chain") as span:
            span.set_attribute("chain_id", chain_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("chain_type", "user_onboarding")

            # Step 1: Validate user
            with vaquero.span("user_validation") as validation_span:
                validation_span.set_attribute("chain_id", chain_id)
                validation_span.set_attribute("step", 1)

                user_valid = await validate_user_data(user_id)
                validation_span.set_attribute("validation_successful", user_valid)

                if not user_valid:
                    span.set_attribute("chain_successful", False)
                    span.set_attribute("failure_step", 1)
                    return {"status": "failed", "step": "validation"}

            # Step 2: Setup user account
            with vaquero.span("account_setup") as setup_span:
                setup_span.set_attribute("chain_id", chain_id)
                setup_span.set_attribute("step", 2)

                account_created = await setup_user_account(user_id)
                setup_span.set_attribute("account_creation_successful", account_created)

                if not account_created:
                    span.set_attribute("chain_successful", False)
                    span.set_attribute("failure_step", 2)
                    return {"status": "failed", "step": "account_setup"}

            # Step 3: Send welcome email
            with vaquero.span("welcome_email") as email_span:
                email_span.set_attribute("chain_id", chain_id)
                email_span.set_attribute("step", 3)

                email_sent = await send_welcome_email(user_id)
                email_span.set_attribute("email_delivery_successful", email_sent)

                if not email_sent:
                    span.set_attribute("chain_successful", False)
                    span.set_attribute("failure_step", 3)
                    return {"status": "failed", "step": "email_delivery"}

            # Chain completed successfully
            span.set_attribute("chain_successful", True)
            span.set_attribute("total_steps", 3)

            return {
                "status": "completed",
                "chain_id": chain_id,
                "user_id": user_id,
                "steps_completed": 3
            }

# Usage
@app.task(bind=True)
@vaquero.trace(agent_name="onboarding_task")
def onboard_user(self, user_id: str) -> dict:
    """Task for user onboarding."""
    chain_manager = TaskChainManager()
    return chain_manager.execute_user_onboarding_chain(user_id)
```

### Error Handling and Retry

```python
# retry_tasks.py
import vaquero

class RetryTaskManager:
    """Task manager with retry logic and tracing."""

    @vaquero.trace(agent_name="retry_task")
    def resilient_task(self, data: dict, max_retries: int = 3) -> dict:
        """Task with retry logic and comprehensive tracing."""
        task_id = f"task_{int(time.time())}"

        for attempt in range(max_retries):
            with vaquero.span(f"retry_attempt_{attempt + 1}") as span:
                span.set_attribute("task_id", task_id)
                span.set_attribute("attempt_number", attempt + 1)
                span.set_attribute("max_retries", max_retries)
                span.set_attribute("data_size", len(str(data)))

                try:
                    # Simulate task that might fail
                    import random
                    if attempt < 2 and random.random() < 0.7:  # Fail first 2 attempts 70% of time
                        raise ConnectionError(f"Simulated failure on attempt {attempt + 1}")

                    # Task succeeds
                    result = {
                        "task_id": task_id,
                        "data_processed": True,
                        "attempts_needed": attempt + 1,
                        "processed_at": time.time()
                    }

                    span.set_attribute("task_successful", True)
                    span.set_attribute("attempts_needed", attempt + 1)

                    return result

                except ConnectionError as e:
                    span.set_attribute("task_successful", False)
                    span.set_attribute("error_type", "connection_error")

                    if attempt == max_retries - 1:
                        # Final attempt failed
                        span.set_attribute("final_failure", True)
                        raise
                    else:
                        # Wait before retry
                        wait_time = 2 ** attempt  # Exponential backoff
                        span.set_attribute("retry_wait_time", wait_time)
                        time.sleep(wait_time)

                except Exception as e:
                    span.set_attribute("task_successful", False)
                    span.set_attribute("error_type", "unexpected_error")
                    raise

# Create retry-aware task
retry_manager = RetryTaskManager()

@app.task(bind=True, autoretry_for=(ConnectionError,), retry_kwargs={'max_retries': 3})
@vaquero.trace(agent_name="celery_retry_task")
def resilient_celery_task(self, data: dict) -> dict:
    """Celery task with built-in retry and tracing."""
    return retry_manager.resilient_task(data, max_retries=3)
```

### Batch Processing

```python
# batch_tasks.py
import vaquero

@vaquero.trace(agent_name="batch_processor")
class BatchTaskManager:
    """Manage batch processing tasks with tracing."""

    @vaquero.trace(agent_name="batch_execution")
    async def process_user_batch(self, user_ids: list, batch_size: int = 10) -> dict:
        """Process users in batches with parallel tracing."""
        total_users = len(user_ids)

        with vaquero.span("batch_processing") as span:
            span.set_attribute("total_users", total_users)
            span.set_attribute("batch_size", batch_size)
            span.set_attribute("batch_count", (total_users + batch_size - 1) // batch_size)

            # Process in batches
            results = []
            for i in range(0, total_users, batch_size):
                batch = user_ids[i:i + batch_size]
                batch_num = i // batch_size + 1

                with vaquero.span(f"batch_{batch_num}") as batch_span:
                    batch_span.set_attribute("batch_number", batch_num)
                    batch_span.set_attribute("batch_users", len(batch))

                    # Process individual users in parallel
                    batch_results = await self._process_user_batch(batch)

                    batch_span.set_attribute("batch_successful", True)
                    batch_span.set_attribute("users_processed", len(batch_results))

                    results.extend(batch_results)

            span.set_attribute("total_processed", len(results))
            span.set_attribute("batch_processing_successful", True)

            return {
                "total_users": total_users,
                "processed_users": len(results),
                "batches_processed": (total_users + batch_size - 1) // batch_size
            }

    @vaquero.trace(agent_name="batch_item_processor")
    async def _process_user_batch(self, user_ids: list) -> list:
        """Process individual users with tracing."""
        import asyncio

        # Create tasks for parallel processing
        tasks = []
        for user_id in user_ids:
            task = self._process_single_user(user_id)
            tasks.append(task)

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results
        successful = []
        failed = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({"user_id": user_ids[i], "error": str(result)})
            else:
                successful.append(result)

        return successful

    @vaquero.trace(agent_name="single_user_processor")
    async def _process_single_user(self, user_id: str) -> dict:
        """Process a single user with tracing."""
        with vaquero.span("user_processing") as span:
            span.set_attribute("user_id", user_id)

            # Simulate user processing
            import time
            await asyncio.sleep(0.01)

            span.set_attribute("processing_successful", True)

            return {
                "user_id": user_id,
                "processed": True,
                "processed_at": time.time()
            }

# Usage
batch_manager = BatchTaskManager()

@app.task(bind=True)
@vaquero.trace(agent_name="celery_batch_task")
def process_large_user_batch(self, user_ids: list) -> dict:
    """Celery task for large batch processing."""
    return batch_manager.process_user_batch(user_ids, batch_size=50)
```

## Configuration Options

### Development Configuration
```python
# Development - Full observability
vaquero.init(
    api_key="your-dev-key",
    mode="development",
    debug=True,
    batch_size=5,         # Small batches for quick feedback
    flush_interval=1.0,   # Frequent flushing
    capture_inputs=True,  # Full task data capture
    capture_outputs=True  # Full result capture
)
```

### Production Configuration
```python
# Production - Optimized performance
vaquero.init(
    api_key="your-prod-key",
    mode="production",
    batch_size=100,       # Larger batches for efficiency
    flush_interval=10.0,  # Less frequent flushing
    capture_inputs=False,    # Privacy protection
    capture_outputs=False,   # Performance optimization
    auto_instrument_llm=False  # Disable for performance
)
```

## Performance Considerations

### Task Sampling
```python
# Implement sampling for high-volume tasks
import random

class TaskSampler:
    """Intelligent task sampling."""

    def __init__(self, sample_rate: float = 0.1):
        self.sample_rate = sample_rate

    def should_sample_task(self, task_name: str) -> bool:
        """Determine if task should be fully traced."""
        # Always sample errors and admin tasks
        if "error" in task_name.lower() or "admin" in task_name.lower():
            return True

        # Sample based on rate
        return random.random() < self.sample_rate

# Use task sampler
sampler = TaskSampler(sample_rate=0.05)  # 5% sampling

@app.task(bind=True)
@vaquero.trace(agent_name="sampled_task")
def high_volume_task(self, data: dict) -> dict:
    """High-volume task with sampling."""
    should_sample = sampler.should_sample_task(self.name)

    with vaquero.span("task_execution") as span:
        span.set_attribute("sampled", should_sample)
        span.set_attribute("sample_rate", sampler.sample_rate)

        if should_sample:
            span.set_attribute("full_tracing", True)
            # Full tracing logic
        else:
            span.set_attribute("full_tracing", False)
            # Minimal tracing logic

        # Your task logic here
        result = process_data(data)
        span.set_attribute("task_successful", True)

        return result
```

### Worker Monitoring
```python
# Monitor worker performance
@app.task(bind=True)
@vaquero.trace(agent_name="worker_monitor")
def monitor_worker_performance(self) -> dict:
    """Monitor worker performance metrics."""
    with vaquero.span("worker_performance_check") as span:
        import psutil

        # Get worker process info
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()

        span.set_attribute("memory_usage_mb", memory_mb)
        span.set_attribute("cpu_usage_percent", cpu_percent)
        span.set_attribute("worker_pid", process.pid)

        # Check for performance issues
        if memory_mb > 500:  # More than 500MB
            span.set_attribute("high_memory_usage", True)

        if cpu_percent > 80:  # More than 80% CPU
            span.set_attribute("high_cpu_usage", True)

        return {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "timestamp": time.time()
        }

# Schedule monitoring task
@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Set up periodic monitoring tasks."""
    # Monitor every 5 minutes
    sender.add_periodic_task(300.0, monitor_worker_performance.s(), name='monitor_performance')
```

## Testing Integration

### Testing Traced Tasks
```python
# test_celery_tasks.py
import pytest
from celery_app import process_user_data, send_notification

class TestTracedTasks:
    def test_successful_task(self):
        """Test successful task execution with tracing."""
        result = process_user_data.delay("user123", {"key": "value"})

        # Wait for task completion
        task_result = result.get(timeout=10)

        assert task_result["processed"] == True
        assert task_result["user_id"] == "user123"

    def test_task_error_handling(self):
        """Test task error handling with tracing."""
        # Test with invalid data that should cause error
        result = process_user_data.delay("user123", None)

        # Should raise exception
        with pytest.raises(Exception):
            result.get(timeout=10)

    @pytest.mark.parametrize("user_id,data", [
        ("user1", {"valid": "data"}),
        ("user2", {"more": "data"}),
        ("user3", {"test": "data"}),
    ])
    def test_multiple_task_executions(self, user_id, data):
        """Test multiple task executions."""
        results = []

        for i in range(3):
            result = process_user_data.delay(user_id, data)
            task_result = result.get(timeout=10)
            results.append(task_result)

        # All tasks should succeed
        assert len(results) == 3
        assert all(r["processed"] == True for r in results)
```

### Performance Testing
```python
def test_task_performance():
    """Test task performance with tracing overhead."""
    import time

    # Measure baseline performance (without tracing)
    start = time.time()
    # Direct function call
    baseline_result = process_data_directly({"test": "data"})
    baseline_time = time.time() - start

    # Measure with tracing
    start = time.time()
    result = process_user_data.delay("user123", {"test": "data"})
    traced_result = result.get(timeout=10)
    traced_time = time.time() - start

    # Tracing overhead should be minimal
    overhead = traced_time - baseline_time
    assert overhead < 0.1  # Less than 100ms overhead

    # Results should be the same
    assert traced_result == baseline_result
```

## Complete Example

Here's a complete Celery application with comprehensive Vaquero integration:

```python
# celery_app.py
from celery import Celery
import vaquero
import time

# Initialize SDK
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    mode="production"
)

# Celery configuration
app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,  # Acknowledge after processing
    task_reject_on_worker_lost=True,  # Handle worker loss
)

# Task monitor for performance tracking
task_monitor = {}

@app.task(bind=True)
@vaquero.trace(agent_name="data_processing_task")
def process_user_data(self, user_id: str, data: dict) -> dict:
    """Process user data with comprehensive tracing."""
    task_id = self.request.id

    with vaquero.span("data_processing") as span:
        span.set_attribute("task_id", task_id)
        span.set_attribute("user_id", user_id)
        span.set_attribute("data_size", len(str(data)))
        span.set_attribute("worker_hostname", self.request.hostname)

        try:
            # Simulate data processing
            time.sleep(0.2)

            result = {
                "user_id": user_id,
                "processed": True,
                "result_size": len(str(data)) * 2,
                "processed_at": time.time()
            }

            span.set_attribute("processing_successful", True)
            span.set_attribute("result_size", len(str(result)))

            # Track task completion
            if task_id not in task_monitor:
                task_monitor[task_id] = {"start_time": time.time()}
            task_monitor[task_id]["success"] = True
            task_monitor[task_id]["end_time"] = time.time()

            return result

        except Exception as e:
            span.set_attribute("processing_successful", False)
            span.set_attribute("error_type", type(e).__name__)

            # Track task failure
            if task_id not in task_monitor:
                task_monitor[task_id] = {"start_time": time.time()}
            task_monitor[task_id]["success"] = False
            task_monitor[task_id]["end_time"] = time.time()
            task_monitor[task_id]["error"] = str(e)

            raise

@app.task(bind=True)
@vaquero.trace(agent_name="notification_task")
def send_notification(self, user_id: str, notification_type: str, message: str) -> dict:
    """Send notification with tracing."""
    task_id = self.request.id

    with vaquero.span("notification_processing") as span:
        span.set_attribute("task_id", task_id)
        span.set_attribute("user_id", user_id)
        span.set_attribute("notification_type", notification_type)
        span.set_attribute("message_length", len(message))

        try:
            # Simulate notification sending
            time.sleep(0.1)

            span.set_attribute("notification_sent", True)
            span.set_attribute("delivery_time_ms", 100)

            return {
                "user_id": user_id,
                "notification_type": notification_type,
                "message": message,
                "status": "sent",
                "sent_at": time.time()
            }

        except Exception as e:
            span.set_attribute("notification_sent", False)
            span.set_attribute("error_type", type(e).__name__)
            raise

@app.task(bind=True)
@vaquero.trace(agent_name="batch_task")
def process_user_batch(self, user_ids: list, batch_size: int = 10) -> dict:
    """Process multiple users in batches."""
    task_id = self.request.id
    total_users = len(user_ids)

    with vaquero.span("batch_processing") as span:
        span.set_attribute("task_id", task_id)
        span.set_attribute("total_users", total_users)
        span.set_attribute("batch_size", batch_size)

        results = []
        processed_count = 0

        # Process in batches
        for i in range(0, total_users, batch_size):
            batch = user_ids[i:i + batch_size]
            batch_num = i // batch_size + 1

            with vaquero.span(f"batch_{batch_num}") as batch_span:
                batch_span.set_attribute("batch_number", batch_num)
                batch_span.set_attribute("batch_users", len(batch))

                # Process users in this batch
                for user_id in batch:
                    with vaquero.span(f"user_{user_id}") as user_span:
                        user_span.set_attribute("user_id", user_id)

                        # Process individual user
                        user_result = process_user_data.delay(user_id, {"batch": batch_num})
                        results.append(user_result.get(timeout=30))

                        user_span.set_attribute("processing_successful", True)
                        processed_count += 1

                batch_span.set_attribute("batch_successful", True)

        span.set_attribute("total_processed", processed_count)
        span.set_attribute("batch_processing_successful", True)

        return {
            "task_id": task_id,
            "total_users": total_users,
            "processed_users": processed_count,
            "results": results
        }

# Worker monitoring task
@app.task(bind=True)
@vaquero.trace(agent_name="worker_monitor")
def monitor_worker_health(self) -> dict:
    """Monitor worker health and performance."""
    with vaquero.span("worker_health_check") as span:
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()

        span.set_attribute("memory_usage_mb", memory_mb)
        span.set_attribute("cpu_usage_percent", cpu_percent)
        span.set_attribute("worker_pid", process.pid)

        # Check task queue status
        from celery import current_app
        inspect = current_app.control.inspect()
        active_tasks = inspect.active()
        reserved_tasks = inspect.reserved()

        span.set_attribute("active_tasks", len(active_tasks.get(self.request.hostname, [])))
        span.set_attribute("reserved_tasks", len(reserved_tasks.get(self.request.hostname, [])))

        return {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "active_tasks": len(active_tasks.get(self.request.hostname, [])),
            "timestamp": time.time()
        }

# Schedule monitoring task
@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Set up periodic monitoring tasks."""
    # Monitor worker health every 2 minutes
    sender.add_periodic_task(120.0, monitor_worker_health.s(), name='monitor_worker')

if __name__ == '__main__':
    app.start()
```

## Best Practices

### 1. Task Naming
```python
# ✅ Good - Descriptive task names
@vaquero.trace(agent_name="user_data_processing")
def process_user_data(user_id: str, data: dict) -> dict:
    pass

@vaquero.trace(agent_name="email_notification")
def send_welcome_email(user_id: str) -> dict:
    pass

# ❌ Avoid - Generic names
@vaquero.trace(agent_name="task")
def do_something(user_id: str) -> dict:
    pass
```

### 2. Error Context
```python
# ✅ Good - Rich error context
try:
    result = risky_operation(data)
except Exception as e:
    with vaquero.span("operation_error") as span:
        span.set_attribute("error_type", type(e).__name__)
        span.set_attribute("error_location", "risky_operation")
        span.set_attribute("data_size", len(str(data)))
        span.set_attribute("retry_count", attempt_number)
    raise
```

### 3. Performance Monitoring
```python
# ✅ Good - Performance monitoring
with vaquero.span("batch_processing") as span:
    span.set_attribute("batch_size", len(items))
    span.set_attribute("processing_strategy", "parallel")

    start_time = time.time()
    results = process_batch(items)
    duration = time.time() - start_time

    span.set_attribute("processing_time_ms", duration * 1000)
    span.set_attribute("items_per_second", len(items) / duration)
```

## Troubleshooting

### Common Issues

#### Tasks not being traced
```python
# Check if @vaquero.trace decorator is applied
@vaquero.trace(agent_name="my_task")  # Must be present
@app.task
def my_task():
    pass

# Check SDK initialization in worker
# Make sure vaquero.init() is called in worker.py
```

#### Performance overhead
```python
# Monitor task execution time
@app.task(bind=True)
@vaquero.trace(agent_name="performance_test")
def performance_test_task(self):
    start_time = time.time()

    # Your task logic
    result = do_work()

    duration = time.time() - start_time
    print(f"Task took: {duration:.3f} seconds")

    return result
```

#### Memory usage issues
```python
# Monitor memory usage in tasks
@app.task(bind=True)
@vaquero.trace(agent_name="memory_monitor")
def memory_intensive_task(self, data: list):
    import psutil

    with vaquero.span("memory_check") as span:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        span.set_attribute("memory_usage_mb", memory_mb)
        span.set_attribute("data_size", len(data))

        if memory_mb > 100:  # High memory usage
            span.set_attribute("high_memory_usage", True)

        # Your memory-intensive logic
        result = process_large_dataset(data)

        return result
```

This integration provides comprehensive observability for your Celery application, including:

- **Task execution tracing**: All task executions automatically traced
- **Error handling**: Task failures properly categorized and traced
- **Performance monitoring**: Task duration, memory usage, and throughput tracked
- **Batch processing**: Complex workflows and batch operations traced
- **Worker monitoring**: Worker health and performance metrics tracked

## Next Steps

- Check out other framework integration guides (FastAPI, Django, etc.)
- Review the [Best Practices](../BEST_PRACTICES.md) guide for optimization tips
- See the [Troubleshooting Guide](../TROUBLESHOOTING.md) for common issues
