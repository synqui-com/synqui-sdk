# SQLAlchemy Integration Guide

## Overview

SQLAlchemy is the Python SQL toolkit and Object-Relational Mapping (ORM) library. This guide shows how to integrate Vaquero SDK with SQLAlchemy for comprehensive database observability.

## Prerequisites

- Python 3.8+
- SQLAlchemy installed: `pip install sqlalchemy`
- Database driver (psycopg2, pymysql, etc.)
- Vaquero SDK installed: `pip install vaquero-sdk`

## Installation & Setup

### 1. Basic Setup

```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import vaquero

# Initialize Vaquero SDK
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    mode="development"
)

Base = declarative_base()

class User(Base):
    """User model with tracing."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, server_default="CURRENT_TIMESTAMP")

    @vaquero.trace(agent_name="user_model")
    def __init__(self, name: str, email: str):
        """Initialize user with tracing."""
        with vaquero.span("user_initialization") as span:
            span.set_attribute("initialization_method", "constructor")
            span.set_attribute("name_length", len(name))
            span.set_attribute("email_domain", email.split("@")[1] if "@" in email else "invalid")

            self.name = name
            self.email = email

            span.set_attribute("initialization_successful", True)

    @vaquero.trace(agent_name="user_model")
    def save(self, session):
        """Save user to database with tracing."""
        with vaquero.span("user_save") as span:
            span.set_attribute("user_id", self.id or "new")
            span.set_attribute("save_method", "session_add")

            try:
                session.add(self)
                session.commit()

                span.set_attribute("save_successful", True)
                if not self.id:  # New user
                    span.set_attribute("user_created", True)

                return self

            except Exception as e:
                session.rollback()
                span.set_attribute("save_successful", False)
                span.set_attribute("error_type", type(e).__name__)
                raise

# Database setup
engine = create_engine("postgresql://user:pass@localhost/mydb")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database service with tracing
class DatabaseService:
    """Database service with comprehensive tracing."""

    @vaquero.trace(agent_name="database_service")
    def __init__(self):
        self.SessionLocal = SessionLocal

    @vaquero.trace(agent_name="user_operations")
    def create_user(self, name: str, email: str) -> User:
        """Create user with database tracing."""
        with vaquero.span("user_creation") as span:
            span.set_attribute("creation_method", "database_service")

            with self.SessionLocal() as session:
                user = User(name=name, email=email)
                user = user.save(session)

                span.set_attribute("created_user_id", user.id)
                span.set_attribute("creation_successful", True)

                return user

    @vaquero.trace(agent_name="user_operations")
    def get_user_by_id(self, user_id: int) -> User | None:
        """Get user by ID with query tracing."""
        with vaquero.span("user_lookup") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("lookup_method", "by_id")

            with self.SessionLocal() as session:
                user = session.query(User).filter(User.id == user_id).first()

                span.set_attribute("user_found", user is not None)
                if user:
                    span.set_attribute("user_name", user.name)
                    span.set_attribute("user_email_domain", user.email.split("@")[1])

                return user

    @vaquero.trace(agent_name="user_operations")
    def get_users_by_domain(self, domain: str) -> list[User]:
        """Get users by email domain with query tracing."""
        with vaquero.span("domain_lookup") as span:
            span.set_attribute("domain", domain)
            span.set_attribute("lookup_method", "by_domain")

            with self.SessionLocal() as session:
                users = session.query(User).filter(User.email.like(f"%@{domain}")).all()

                span.set_attribute("users_found", len(users))
                span.set_attribute("domain_search_successful", True)

                return users

# Usage
db_service = DatabaseService()

# Create user
user = db_service.create_user("John Doe", "john@example.com")

# Get user
retrieved_user = db_service.get_user_by_id(user.id)

# Get users by domain
example_users = db_service.get_users_by_domain("example.com")
```

### 2. Query Tracing

Monitor individual database queries:

```python
# query_tracing.py
import vaquero

class QueryTracer:
    """Trace SQLAlchemy queries."""

    @vaquero.trace(agent_name="query_tracer")
    def trace_query(self, query_func, *args, **kwargs):
        """Trace a database query execution."""
        with vaquero.span("sqlalchemy_query") as span:
            span.set_attribute("query_type", "custom_query")

            import time
            start_time = time.time()

            try:
                result = query_func(*args, **kwargs)
                duration = time.time() - start_time

                span.set_attribute("query_duration_ms", duration * 1000)
                span.set_attribute("result_count", len(result) if hasattr(result, '__len__') else 1)
                span.set_attribute("query_successful", True)

                return result

            except Exception as e:
                duration = time.time() - start_time
                span.set_attribute("query_duration_ms", duration * 1000)
                span.set_attribute("query_successful", False)
                span.set_attribute("error_type", type(e).__name__)
                raise

# Usage
tracer = QueryTracer()

# Trace custom queries
@vaquero.trace(agent_name="custom_query")
def get_active_users():
    """Get active users with tracing."""
    return tracer.trace_query(
        lambda: session.query(User).filter(User.created_at >= '2024-01-01').all()
    )

# Trace complex queries
@vaquero.trace(agent_name="complex_query")
def get_user_statistics():
    """Get user statistics with tracing."""
    return tracer.trace_query(
        lambda: session.query(
            User.email_domain,
            func.count(User.id).label('count')
        ).group_by(User.email_domain).all(),
        "user_statistics"
    )
```

## Advanced Integration Patterns

### Connection Pool Monitoring

```python
# pool_monitoring.py
import vaquero

class ConnectionPoolMonitor:
    """Monitor SQLAlchemy connection pool."""

    def __init__(self, engine):
        self.engine = engine

    @vaquero.trace(agent_name="pool_monitor")
    def monitor_pool_status(self) -> dict:
        """Monitor connection pool status."""
        with vaquero.span("pool_status_check") as span:
            pool = self.engine.pool

            # Get pool statistics (implementation depends on pool type)
            try:
                pool_size = pool.size()
                checked_out = pool.checkedout()
                invalid = pool.invalid()

                span.set_attribute("pool_size", pool_size)
                span.set_attribute("connections_checked_out", checked_out)
                span.set_attribute("connections_invalid", invalid)
                span.set_attribute("pool_utilization", checked_out / pool_size if pool_size > 0 else 0)

                # Alert on high utilization
                if checked_out / pool_size > 0.8:
                    span.set_attribute("high_utilization", True)

                return {
                    "pool_size": pool_size,
                    "checked_out": checked_out,
                    "invalid": invalid,
                    "utilization": checked_out / pool_size if pool_size > 0 else 0
                }

            except Exception as e:
                span.set_attribute("pool_status_error", True)
                span.set_attribute("error_type", type(e).__name__)
                raise

# Usage
pool_monitor = ConnectionPoolMonitor(engine)

# Monitor periodically
@app.before_first_request
def setup_pool_monitoring():
    """Set up periodic pool monitoring."""
    def monitor_loop():
        while True:
            try:
                status = pool_monitor.monitor_pool_status()
                print(f"Pool status: {status}")
            except Exception as e:
                print(f"Pool monitoring error: {e}")

            time.sleep(60)  # Monitor every minute

    import threading
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
```

### Transaction Tracing

```python
# transaction_tracing.py
import vaquero

class TransactionTracer:
    """Trace database transactions."""

    @vaquero.trace(agent_name="transaction_manager")
    def execute_in_transaction(self, operations: list) -> dict:
        """Execute multiple operations in a transaction."""
        transaction_id = f"txn_{int(time.time())}"

        with vaquero.span("database_transaction") as span:
            span.set_attribute("transaction_id", transaction_id)
            span.set_attribute("operation_count", len(operations))
            span.set_attribute("transaction_type", "multi_operation")

            with SessionLocal() as session:
                try:
                    # Execute operations
                    for i, operation in enumerate(operations):
                        with vaquero.span(f"operation_{i + 1}") as op_span:
                            op_span.set_attribute("transaction_id", transaction_id)
                            op_span.set_attribute("operation_index", i + 1)

                            result = operation(session)
                            op_span.set_attribute("operation_successful", True)

                    # Commit transaction
                    with vaquero.span("transaction_commit") as commit_span:
                        commit_span.set_attribute("transaction_id", transaction_id)
                        session.commit()
                        span.set_attribute("transaction_committed", True)

                    return {"status": "committed", "transaction_id": transaction_id}

                except Exception as e:
                    # Rollback transaction
                    with vaquero.span("transaction_rollback") as rollback_span:
                        rollback_span.set_attribute("transaction_id", transaction_id)
                        rollback_span.set_attribute("error_type", type(e).__name__)
                        session.rollback()
                        span.set_attribute("transaction_rolled_back", True)

                    raise

# Usage
transaction_tracer = TransactionTracer()

def complex_user_operation(user_id: int):
    """Complex operation requiring transaction."""
    operations = [
        lambda session: session.query(User).filter(User.id == user_id).first(),
        lambda session: update_user_preferences(session, user_id),
        lambda session: log_user_activity(session, user_id)
    ]

    return transaction_tracer.execute_in_transaction(operations)
```

### Bulk Operations

```python
# bulk_operations.py
import vaquero

class BulkOperationTracer:
    """Trace bulk database operations."""

    @vaquero.trace(agent_name="bulk_operations")
    def bulk_insert_users(self, users_data: list) -> dict:
        """Bulk insert users with tracing."""
        with vaquero.span("bulk_insert") as span:
            span.set_attribute("bulk_operation", "insert")
            span.set_attribute("record_count", len(users_data))
            span.set_attribute("table", "users")

            with SessionLocal() as session:
                try:
                    # Insert in batches
                    batch_size = 100
                    total_inserted = 0

                    for i in range(0, len(users_data), batch_size):
                        batch = users_data[i:i + batch_size]
                        batch_num = i // batch_size + 1

                        with vaquero.span(f"batch_{batch_num}") as batch_span:
                            batch_span.set_attribute("batch_number", batch_num)
                            batch_span.set_attribute("batch_size", len(batch))

                            # Create user objects
                            users = [User(**user_data) for user_data in batch]

                            # Bulk insert
                            session.bulk_save_objects(users)
                            session.flush()  # Flush to get IDs

                            batch_span.set_attribute("batch_successful", True)
                            batch_span.set_attribute("records_inserted", len(users))

                            total_inserted += len(users)

                    # Final commit
                    with vaquero.span("final_commit") as commit_span:
                        session.commit()
                        span.set_attribute("commit_successful", True)

                    span.set_attribute("total_inserted", total_inserted)
                    span.set_attribute("bulk_operation_successful", True)

                    return {"inserted_count": total_inserted, "status": "success"}

                except Exception as e:
                    session.rollback()
                    span.set_attribute("bulk_operation_successful", False)
                    span.set_attribute("error_type", type(e).__name__)
                    raise

    @vaquero.trace(agent_name="bulk_operations")
    def bulk_update_users(self, updates: list) -> dict:
        """Bulk update users with tracing."""
        with vaquero.span("bulk_update") as span:
            span.set_attribute("bulk_operation", "update")
            span.set_attribute("update_count", len(updates))
            span.set_attribute("table", "users")

            with SessionLocal() as session:
                try:
                    updated_count = 0

                    for update in updates:
                        with vaquero.span("single_update") as update_span:
                            update_span.set_attribute("user_id", update["user_id"])
                            update_span.set_attribute("update_fields", list(update["data"].keys()))

                            # Update user
                            user = session.query(User).filter(User.id == update["user_id"]).first()
                            if user:
                                for key, value in update["data"].items():
                                    setattr(user, key, value)

                                update_span.set_attribute("update_successful", True)
                                updated_count += 1
                            else:
                                update_span.set_attribute("update_successful", False)
                                update_span.set_attribute("user_not_found", True)

                    # Commit all updates
                    with vaquero.span("update_commit") as commit_span:
                        session.commit()
                        span.set_attribute("commit_successful", True)

                    span.set_attribute("total_updated", updated_count)
                    span.set_attribute("bulk_operation_successful", True)

                    return {"updated_count": updated_count, "status": "success"}

                except Exception as e:
                    session.rollback()
                    span.set_attribute("bulk_operation_successful", False)
                    span.set_attribute("error_type", type(e).__name__)
                    raise

# Usage
bulk_tracer = BulkOperationTracer()

# Bulk insert
users_data = [
    {"name": f"User {i}", "email": f"user{i}@example.com"}
    for i in range(1000)
]

result = bulk_tracer.bulk_insert_users(users_data)

# Bulk update
updates = [
    {"user_id": 1, "data": {"name": "Updated Name 1"}},
    {"user_id": 2, "data": {"name": "Updated Name 2"}},
    # ... more updates
]

result = bulk_tracer.bulk_update_users(updates)
```

## Configuration Options

### Development Configuration
```python
# Development - Maximum observability
vaquero.init(
    api_key="your-dev-key",
    mode="development",
    debug=True,
    batch_size=10,        # Smaller batches for faster feedback
    flush_interval=2.0,   # More frequent flushing
    capture_inputs=True,  # Full query parameter capture
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
    capture_system_prompts=False  # Privacy protection
)
```

## Performance Considerations

### Query Optimization Monitoring

```python
# query_optimization.py
import vaquero

class QueryOptimizer:
    """Monitor and optimize SQLAlchemy queries."""

    def __init__(self):
        self.slow_queries = []
        self.query_stats = {}

    @vaquero.trace(agent_name="query_optimizer")
    def monitor_query_performance(self, query: str, duration: float, row_count: int):
        """Monitor query performance."""
        with vaquero.span("query_performance_monitoring") as span:
            span.set_attribute("query_hash", hash(query) % 10000)
            span.set_attribute("duration_ms", duration * 1000)
            span.set_attribute("row_count", row_count)

            # Track slow queries
            if duration > 0.1:  # More than 100ms
                self.slow_queries.append({
                    "query": query[:100] + "...",  # Truncate for privacy
                    "duration": duration,
                    "row_count": row_count,
                    "timestamp": time.time()
                })
                span.set_attribute("slow_query_detected", True)

            # Track statistics
            query_type = "SELECT" if query.strip().upper().startswith("SELECT") else "MODIFY"
            if query_type not in self.query_stats:
                self.query_stats[query_type] = {
                    "count": 0,
                    "total_duration": 0,
                    "avg_duration": 0
                }

            stats = self.query_stats[query_type]
            stats["count"] += 1
            stats["total_duration"] += duration
            stats["avg_duration"] = stats["total_duration"] / stats["count"]

            span.set_attribute("query_type", query_type)
            span.set_attribute("query_stats_updated", True)

    @vaquero.trace(agent_name="optimization_analysis")
    def analyze_optimization_opportunities(self) -> dict:
        """Analyze queries for optimization opportunities."""
        with vaquero.span("optimization_analysis") as span:
            if not self.slow_queries:
                return {"message": "No slow queries detected"}

            analysis = {
                "slow_query_count": len(self.slow_queries),
                "query_types": self.query_stats,
                "recommendations": []
            }

            # Analyze SELECT vs MODIFY ratios
            select_count = self.query_stats.get("SELECT", {}).get("count", 0)
            modify_count = self.query_stats.get("MODIFY", {}).get("count", 0)
            total_queries = select_count + modify_count

            if total_queries > 0:
                select_ratio = select_count / total_queries
                if select_ratio > 0.8:  # More than 80% SELECT queries
                    analysis["recommendations"].append({
                        "type": "query_pattern",
                        "issue": "High SELECT query ratio",
                        "suggestion": "Consider query result caching or read replicas"
                    })

            # Analyze slow query patterns
            if self.slow_queries:
                avg_slow_duration = sum(q["duration"] for q in self.slow_queries) / len(self.slow_queries)
                if avg_slow_duration > 1.0:  # Average slow query > 1 second
                    analysis["recommendations"].append({
                        "type": "performance",
                        "issue": "Slow query performance",
                        "suggestion": "Review query patterns and consider indexing"
                    })

            span.set_attribute("recommendations_generated", len(analysis["recommendations"]))

            return analysis

# Usage
optimizer = QueryOptimizer()

# Monitor queries
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    duration = time.time() - context._query_start_time
    row_count = cursor.rowcount if hasattr(cursor, 'rowcount') else 0

    optimizer.monitor_query_performance(statement, duration, row_count)
```

### Connection Pool Optimization

```python
# pool_optimization.py
import vaquero

class PoolOptimizer:
    """Optimize database connection pool settings."""

    @vaquero.trace(agent_name="pool_optimizer")
    def optimize_pool_settings(self, current_load: dict) -> dict:
        """Optimize pool settings based on current load."""
        with vaquero.span("pool_optimization") as span:
            span.set_attribute("current_connections", current_load.get("active_connections", 0))
            span.set_attribute("current_pool_size", current_load.get("pool_size", 20))

            # Analyze current utilization
            utilization = current_load["active_connections"] / current_load["pool_size"]

            span.set_attribute("pool_utilization", utilization)

            recommendations = {}

            if utilization > 0.8:
                # High utilization - increase pool size
                new_pool_size = int(current_load["pool_size"] * 1.5)
                recommendations["pool_size"] = new_pool_size
                span.set_attribute("recommendation", "increase_pool_size")

            elif utilization < 0.3 and current_load["pool_size"] > 10:
                # Low utilization - decrease pool size
                new_pool_size = max(10, int(current_load["pool_size"] * 0.8))
                recommendations["pool_size"] = new_pool_size
                span.set_attribute("recommendation", "decrease_pool_size")

            if current_load.get("connection_wait_time", 0) > 1.0:
                # Long wait times - increase pool size
                new_pool_size = int(current_load["pool_size"] * 1.2)
                recommendations["pool_size"] = new_pool_size
                recommendations["overflow"] = int(current_load["pool_size"] * 0.2)
                span.set_attribute("recommendation", "increase_pool_size_and_overflow")

            span.set_attribute("recommendations_generated", len(recommendations))

            return recommendations

# Usage
pool_optimizer = PoolOptimizer()

# Monitor and optimize periodically
def optimize_database_performance():
    """Periodic database performance optimization."""
    current_load = get_current_database_load()  # Your monitoring logic

    recommendations = pool_optimizer.optimize_pool_settings(current_load)

    if recommendations:
        print(f"Pool optimization recommendations: {recommendations}")
        # Apply recommendations to your engine configuration
```

## Testing Integration

### Testing Traced Models

```python
# test_models.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from myapp.models import User
from myapp.database import SessionLocal

@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    User.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

class TestTracedModels:
    def test_user_creation(self, db_session):
        """Test user creation with tracing."""
        user = User(name="Test User", email="test@example.com")
        saved_user = user.save(db_session)

        assert saved_user.id is not None
        assert saved_user.name == "Test User"

    def test_user_lookup(self, db_session):
        """Test user lookup with tracing."""
        # Create test user
        user = User(name="Test User", email="test@example.com")
        saved_user = user.save(db_session)

        # Test lookup
        db_service = DatabaseService()
        found_user = db_service.get_user_by_id(saved_user.id)

        assert found_user is not None
        assert found_user.name == "Test User"

    def test_bulk_operations(self, db_session):
        """Test bulk operations with tracing."""
        bulk_tracer = BulkOperationTracer()

        # Test data
        users_data = [
            {"name": f"User {i}", "email": f"user{i}@example.com"}
            for i in range(50)
        ]

        result = bulk_tracer.bulk_insert_users(users_data)

        assert result["inserted_count"] == 50
        assert result["status"] == "success"
```

### Performance Testing

```python
def test_query_performance():
    """Test query performance with tracing overhead."""
    import time

    # Measure baseline (without tracing)
    start = time.time()
    users = session.query(User).filter(User.email.like('%@example.com')).all()
    baseline_time = time.time() - start

    # Measure with tracing
    start = time.time()
    traced_users = traced_query()  # Your traced query function
    traced_time = time.time() - start

    # Tracing overhead should be minimal
    overhead = traced_time - baseline_time
    assert overhead < 0.05  # Less than 50ms overhead

    # Results should be the same
    assert len(traced_users) == len(users)
```

## Complete Example

Here's a complete SQLAlchemy application with comprehensive Vaquero integration:

```python
# app.py
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import vaquero

# Initialize SDK
vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id",
    mode="production"
)

# Database setup
Base = declarative_base()
engine = create_engine("postgresql://user:pass@localhost/mydb")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Models with tracing
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, server_default="CURRENT_TIMESTAMP")

    @vaquero.trace(agent_name="user_model")
    def save(self, session: Session):
        """Save user with tracing."""
        with vaquero.span("user_save") as span:
            span.set_attribute("user_id", self.id or "new")

            try:
                session.add(self)
                session.commit()
                span.set_attribute("save_successful", True)
                return self
            except Exception as e:
                session.rollback()
                span.set_attribute("save_successful", False)
                span.set_attribute("error_type", type(e).__name__)
                raise

# Database service with tracing
class DatabaseService:
    @vaquero.trace(agent_name="database_service")
    def get_user_by_email(self, email: str) -> User | None:
        """Get user by email with tracing."""
        with vaquero.span("email_lookup") as span:
            span.set_attribute("email_domain", email.split("@")[1] if "@" in email else "invalid")
            span.set_attribute("lookup_method", "by_email")

            with SessionLocal() as session:
                user = session.query(User).filter(User.email == email).first()

                span.set_attribute("user_found", user is not None)
                if user:
                    span.set_attribute("user_id", user.id)

                return user

# FastAPI app
app = FastAPI()

@app.get("/users/by-email/{email}")
@vaquero.trace(agent_name="user_api")
async def get_user_by_email(email: str):
    """Get user by email with comprehensive tracing."""
    with vaquero.span("user_email_lookup") as span:
        span.set_attribute("email", email)
        span.set_attribute("email_domain", email.split("@")[1] if "@" in email else "invalid")

        db_service = DatabaseService()
        user = db_service.get_user_by_email(email)

        if not user:
            span.set_attribute("user_found", False)
            raise HTTPException(status_code=404, detail="User not found")

        span.set_attribute("user_found", True)
        span.set_attribute("user_id", user.id)

        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }

@app.post("/users")
@vaquero.trace(agent_name="user_creation_api")
async def create_user(name: str, email: str):
    """Create user with comprehensive tracing."""
    with vaquero.span("user_creation_workflow") as span:
        span.set_attribute("creation_method", "api")
        span.set_attribute("name_length", len(name))
        span.set_attribute("email_domain", email.split("@")[1] if "@" in email else "invalid")

        with SessionLocal() as session:
            # Check if user exists
            with vaquero.span("duplicate_check") as check_span:
                existing_user = session.query(User).filter(User.email == email).first()

                if existing_user:
                    check_span.set_attribute("user_exists", True)
                    check_span.set_attribute("existing_user_id", existing_user.id)
                    raise HTTPException(status_code=400, detail="User already exists")

                check_span.set_attribute("user_exists", False)

            # Create new user
            with vaquero.span("user_persistence") as persist_span:
                user = User(name=name, email=email)
                user = user.save(session)

                persist_span.set_attribute("created_user_id", user.id)
                persist_span.set_attribute("creation_successful", True)

        span.set_attribute("workflow_successful", True)

        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "status": "created"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Best Practices

### 1. Query Attribution
```python
# ✅ Good - Rich query context
with vaquero.span("user_query") as span:
    span.set_attribute("query_type", "SELECT")
    span.set_attribute("table", "users")
    span.set_attribute("filter_columns", ["id", "email"])
    span.set_attribute("expected_rows", "single")
    span.set_attribute("query_complexity", "simple")

# ❌ Avoid - Minimal context
with vaquero.span("db_op") as span:
    span.set_attribute("op", "select")
```

### 2. Transaction Tracking
```python
# ✅ Good - Complete transaction lifecycle
with vaquero.span("user_transaction") as span:
    span.set_attribute("transaction_type", "user_update")
    span.set_attribute("affected_tables", ["users", "user_preferences"])

    with vaquero.span("begin_transaction") as begin_span:
        # Begin transaction logic

    try:
        # Transaction operations
        with vaquero.span("commit_transaction") as commit_span:
            # Commit logic
    except Exception as e:
        with vaquero.span("rollback_transaction") as rollback_span:
            # Rollback logic
```

### 3. Performance Monitoring
```python
# ✅ Good - Performance monitoring
with vaquero.span("bulk_insert") as span:
    span.set_attribute("operation", "bulk_insert")
    span.set_attribute("table", "users")
    span.set_attribute("record_count", len(users))

    start_time = time.time()
    session.bulk_insert_mappings(User, user_mappings)
    session.commit()
    duration = time.time() - start_time

    span.set_attribute("insert_duration_ms", duration * 1000)
    span.set_attribute("records_per_second", len(users) / duration)
```

## Troubleshooting

### Common Issues

#### Models not being traced
```python
# Check if @vaquero.trace decorator is applied to model methods
class User(Base):
    @vaquero.trace(agent_name="user_model")  # Must be present
    def save(self, session):
        pass

    @classmethod
    @vaquero.trace(agent_name="user_model")  # Must be present
    def create_user(cls, name, email):
        pass
```

#### Queries not traced
```python
# Ensure query operations are decorated
@vaquero.trace(agent_name="query_operation")
def get_users_by_domain(domain):
    return session.query(User).filter(User.email.like(f'%@{domain}')).all()

# Or trace the entire service method
@vaquero.trace(agent_name="user_service")
def get_user_statistics():
    return session.query(User.email_domain, func.count(User.id)).group_by(User.email_domain).all()
```

#### Performance overhead
```python
# Monitor query performance
@event.listens_for(Engine, "before_cursor_execute")
def log_slow_queries(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()

@event.listens_for(Engine, "after_cursor_execute")
def check_query_performance(conn, cursor, statement, parameters, context, executemany):
    duration = time.time() - context._query_start_time

    if duration > 0.1:  # More than 100ms
        print(f"Slow query detected: {duration:.3f}s - {statement[:100]}...")
```

This integration provides comprehensive observability for your SQLAlchemy application, including:

- **Model operations**: All model saves, queries, and operations traced
- **Query performance**: Individual query execution time and row counts tracked
- **Transaction tracking**: Transaction lifecycle (begin, commit, rollback) traced
- **Bulk operations**: Batch inserts and updates properly traced
- **Connection pool monitoring**: Pool utilization and performance tracked
- **Error handling**: Database errors properly categorized and traced

## Next Steps

- Check out other framework integration guides (FastAPI, Django, etc.)
- Review the [Best Practices](../BEST_PRACTICES.md) guide for optimization tips
- See the [Troubleshooting Guide](../TROUBLESHOOTING.md) for common issues
