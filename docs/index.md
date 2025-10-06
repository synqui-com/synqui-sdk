# 🚀 Vaquero SDK Documentation

**Comprehensive observability for multi-agent AI systems.** See your agent architecture evolve in real-time, debug failures faster, and keep your AI workflows running smoothly.

## 🎯 Quick Start (5 minutes)

Get up and running with Vaquero in just 3 simple steps:

<div class="quick-start-steps">

### 1️⃣ Install
```bash
pip install vaquero-sdk
```

### 2️⃣ Initialize
```python
import vaquero

vaquero.init(
    api_key="your-api-key",
    project_id="your-project-id"
)
```

### 3️⃣ Trace
```python
@vaquero.trace("my_agent")
def my_function(data):
    # Your code here
    return processed_data
```

</div>

**That's it!** Your functions are now automatically traced and monitored.

## 📚 Documentation Sections

<div class="nav-cards">

### 🚀 [Getting Started](./GETTING_STARTED.md)
Complete guide to install, configure, and start tracing with 5-minute quick start.

### 📖 [Common Patterns](./patterns/)
Essential patterns for function tracing, API endpoints, database operations, and error handling.

### 🔧 [Advanced Features](./advanced/)
Power user features including auto-instrumentation, custom spans, and performance monitoring.

### 🛠️ [Framework Integrations](./integrations/)
Framework-specific guides for FastAPI, Django, Flask, Celery, and SQLAlchemy.

### 💡 [Troubleshooting](./TROUBLESHOOTING.md)
Common issues, solutions, and debugging strategies.

### 🎯 [Best Practices](./BEST_PRACTICES.md)
Guidelines for consistent, high-quality SDK usage.

### 📚 [API Reference](./API_REFERENCE.md)
Complete reference for configuration, tracing, spans, and utilities.

</div>

## 🌟 Key Features

<div class="feature-grid">

### ⚡ **Zero-Config Setup**
Get started with just an API key. Everything else works out of the box.

### 🔍 **Automatic LLM Instrumentation**
Automatically trace OpenAI, Anthropic, and other LLM calls with prompts, tokens, and performance metrics.

### 📊 **Real-time Monitoring**
See your agent interactions, architecture evolution, and performance metrics in real-time.

### 🛠️ **Framework Integration**
Built-in support for FastAPI, Django, Flask, Celery, and more.

### 🔒 **Enterprise Security**
Project-scoped API keys, encrypted data in transit and at rest, and comprehensive audit trails.

</div>

## 💡 Use Cases

<div class="use-cases">

### 🤖 **AI Agent Development**
Monitor agent interactions, debug complex workflows, and optimize performance.

### 🔧 **API Development**
Trace API endpoints, monitor response times, and identify bottlenecks.

### 🗄️ **Database Operations**
Monitor query performance, track data flow, and optimize database usage.

### ⚙️ **Background Jobs**
Monitor Celery tasks, Redis operations, and distributed processing.

</div>

## 🚀 Next Steps

Ready to get started? Jump to the **[Getting Started guide](./GETTING_STARTED.md)** for a complete setup guide, or check out **[common patterns](./patterns/)** for practical examples.

Need help? Check out the **[Troubleshooting guide](./TROUBLESHOOTING.md)** or **[Best Practices guide](./BEST_PRACTICES.md)** for detailed guidance.

Framework-specific help? See our **[integration guides](./integrations/)** for FastAPI, Django, Flask, Celery, and SQLAlchemy.

---

<div class="footer-note">
📖 **Need more details?** Browse the full documentation above or use the search to find specific topics.
</div>
