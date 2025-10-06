# 🚀 Vaquero Python SDK

<div align="center">
  <h3>Comprehensive observability and tracing for AI agents and applications</h3>
  <p><strong>Zero-config tracing</strong> • <strong>Auto-instrumentation</strong> • <strong>Production-ready</strong></p>

  [![PyPI version](https://badge.fury.io/py/vaquero-sdk.svg)](https://badge.fury.io/py/vaquero-sdk)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

<!-- CSS Styles for better UX -->
<style>
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
.card { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 2rem 0; }
.feature-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; }
.feature-icon { font-size: 2rem; margin-bottom: 1rem; }
.doc-tabs { display: flex; gap: 1rem; margin: 2rem 0; flex-wrap: wrap; }
.tab { background: #f1f3f4; padding: 1rem 1.5rem; border-radius: 8px; border: 2px solid transparent; cursor: pointer; }
.tab.active { background: #4285f4; color: white; border-color: #4285f4; }
.tab:hover { background: #e8f0fe; }
.btn-primary { background: #4285f4; color: white; padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; display: inline-block; }
.btn-primary:hover { background: #3367d6; }
.install-options { margin: 2rem 0; }
.option { background: #f8f9fa; border-left: 4px solid #4285f4; padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0; }
.code-block { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 6px; overflow-x: auto; font-family: 'Consolas', monospace; font-size: 0.9em; }
.code-card { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
.code-card h4 { margin-top: 0; color: #4285f4; }
.section-grid { display: grid; gap: 2rem; margin: 2rem 0; }
.env-vars { background: #e8f5e8; border: 1px solid #4caf50; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
.env-vars h4 { margin-top: 0; color: #2e7d32; }
.env-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem; margin: 1rem 0; }
.env-grid code { background: #f1f8e9; padding: 0.25rem 0.5rem; border-radius: 4px; font-family: monospace; }
.config-table { overflow-x: auto; margin: 2rem 0; }
.config-table table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }
.config-table th, .config-table td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #e9ecef; }
.config-table th { background: #f8f9fa; font-weight: 600; }
.config-table tr:hover { background: #f8f9fa; }
.best-practices { background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 1.5rem; margin: 2rem 0; }
.best-practices h4 { margin-top: 0; color: #856404; }
.dev-section { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin: 2rem 0; }
.dev-card { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 1.5rem; }
.dev-card h4 { margin-top: 0; color: #4285f4; }
.resource-links { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin: 2rem 0; }
.resource-link { display: flex; align-items: center; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 1.5rem; text-decoration: none; color: inherit; }
.resource-link:hover { background: #e9ecef; }
.resource-link .icon { font-size: 2rem; margin-right: 1rem; }
.resource-link h4 { margin: 0 0 0.5rem 0; color: #4285f4; }
.resource-link p { margin: 0; color: #6c757d; }

/* Mobile responsiveness */
@media (max-width: 768px) {
  .grid-2, .grid-3 { grid-template-columns: 1fr; }
  .feature-grid { grid-template-columns: 1fr; }
  .doc-tabs { flex-direction: column; }
  .env-grid { grid-template-columns: 1fr; }
  .dev-section { grid-template-columns: 1fr; }
  .resource-links { grid-template-columns: 1fr; }
}
</style>

---

## ⚡ Quick Start

<div class="grid-2">
  <div class="card">
    <h4>1. Install</h4>
    <pre class="code-block">pip install vaquero-sdk</pre>
  </div>

  <div class="card">
    <h4>2. Configure</h4>
    <pre class="code-block">import vaquero

vaquero.init(api_key="your-api-key")</pre>
  </div>

  <div class="card">
    <h4>3. Trace</h4>
    <pre class="code-block">@vaquero.trace("my_agent")
def process_data(data):
    return {"result": data}

# Done! ✨</pre>
  </div>
</div>

---

## 🎯 Key Features

<div class="feature-grid">
  <div class="feature-card">
    <div class="feature-icon">🔍</div>
    <h4>Automatic Tracing</h4>
    <p>One decorator instruments your entire function with comprehensive observability</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">🤖</div>
    <h4>LLM Auto-Instrumentation</h4>
    <p>Automatically captures OpenAI, Anthropic, and other LLM calls with zero code changes</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">⚡</div>
    <h4>Async-First</h4>
    <p>Full support for async/await patterns with intelligent batching</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">📊</div>
    <h4>Performance Monitoring</h4>
    <p>Built-in profiling, memory tracking, and performance insights</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">🛡️</div>
    <h4>Production Ready</h4>
    <p>Circuit breakers, retry logic, and enterprise-grade reliability</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">🎛️</div>
    <h4>Zero Configuration</h4>
    <p>Environment variables and sensible defaults get you started instantly</p>
  </div>
</div>

---

## 📚 Documentation

<div class="doc-tabs">
  <div class="tab active" data-tab="getting-started">
    <h3>🚀 Getting Started</h3>
    <p>Complete guide to install, configure, and start tracing</p>
    <a href="docs/GETTING_STARTED.md" class="btn-primary">View Guide</a>
  </div>

  <div class="tab" data-tab="api-reference">
    <h3>📖 API Reference</h3>
    <p>Detailed API documentation and configuration options</p>
    <a href="docs/API_REFERENCE.md" class="btn-primary">View Reference</a>
  </div>

  <div class="tab" data-tab="examples">
    <h3>💡 Examples</h3>
    <p>Real-world examples and integration patterns</p>
    <a href="examples/" class="btn-primary">View Examples</a>
  </div>
</div>

---

## 🔧 Installation Options

<div class="install-options">
  <div class="option">
    <h4>🛠️ From PyPI (Recommended)</h4>
    <pre class="code-block">pip install vaquero-sdk</pre>
  </div>

  <div class="option">
    <h4>🔨 From Source</h4>
    <pre class="code-block">git clone https://github.com/vaquero/vaquero-python.git
cd vaquero-python
pip install -e .</pre>
  </div>

  <div class="option">
    <h4>📦 With All Dependencies</h4>
    <pre class="code-block">pip install vaquero-sdk[all]</pre>
  </div>
</div>

---

## 💻 Code Examples

<div class="code-examples">

### Basic Function Tracing
```python
import vaquero

# Configure once
vaquero.init(api_key="your-key")

@vaquero.trace("data_processor")
def process_data(data):
    """Process some data."""
    result = {"processed": len(data), "items": data}
    return result

# Your function is now automatically traced!
result = process_data(["item1", "item2", "item3"])
```

### Async Support
```python
@vaquero.trace("api_client")
async def fetch_data(url):
    """Async data fetching."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Works seamlessly with async/await
result = await fetch_data("https://api.example.com/data")
```

### Manual Span Creation
```python
async with vaquero.span("complex_operation") as span:
    span.set_attribute("operation_type", "batch_processing")
    span.set_attribute("batch_size", len(data))

    # Your complex logic here
    result = await process_batch(data)

    span.set_attribute("result_count", len(result))
```

### Auto-Instrumentation (Zero Code Changes!)
```python
# Enable LLM auto-instrumentation
vaquero.init(api_key="your-key", auto_instrument_llm=True)

# Now any LLM calls are automatically traced!
import openai

client = openai.OpenAI(api_key="your-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# System prompts, tokens, timing all captured automatically! ✨
```

</div>

---

## 🎨 Advanced Usage

<div class="section-grid">

### Manual Span Creation
<div class="code-card">
  <h4>Sync & Async Support</h4>
  <pre class="code-block"># Same API works for both sync and async!
with vaquero.span("custom_operation") as span:
    span.set_attribute("user_id", "12345")
    result = expensive_computation()
    span.set_attribute("result_size", len(result))

async with vaquero.span("async_operation") as span:
    span.set_attribute("operation_type", "ml_inference")
    result = await ml_model.predict(data)
    span.set_attribute("confidence", result.confidence)</pre>
</div>

### Nested Tracing
<div class="code-card">
  <h4>Parent-Child Relationships</h4>
  <pre class="code-block">@vaquero.trace("main_processor")
def main_process(data):
    # Parent span automatically created
    preprocessed = preprocess_data(data)

    # Child span with context
    with vaquero.span("validation") as span:
        span.set_attribute("data_size", len(preprocessed))
        validate_data(preprocessed)

    return postprocess_data(preprocessed)</pre>
</div>

### Custom Configuration
<div class="code-card">
  <h4>Advanced Setup</h4>
  <pre class="code-block">from vaquero import SDKConfig

config = SDKConfig(
    api_key="your-api-key",
    project_id="your-project-id",  # Optional - auto-provisioned
    batch_size=100,        # Optimize for your workload
    flush_interval=5.0,    # Balance latency vs efficiency
    max_retries=3,         # Handle transient failures
    capture_inputs=True,   # Privacy vs debugging
    tags={"team": "ml", "env": "prod"}  # Global metadata
)

vaquero.init(config=config)</pre>
</div>

### Environment Variables
<div class="env-vars">
  <h4>Configuration via Environment</h4>
  <div class="env-grid">
    <code>VAQUERO_API_KEY=your-key</code>
    <code>VAQUERO_PROJECT_ID=your-project</code>
    <code>VAQUERO_ENDPOINT=https://api.vaquero.com</code>
    <code>VAQUERO_BATCH_SIZE=50</code>
    <code>VAQUERO_AUTO_INSTRUMENT_LLM=true</code>
  </div>
  <pre class="code-block">import vaquero
vaquero.init()  # Loads from env vars</pre>
</div>

### Error Handling & Resilience
<div class="code-card">
  <h4>Automatic Error Capture</h4>
  <pre class="code-block">@vaquero.trace("risky_operation")
def risky_operation(data):
    if not data:
        raise ValueError("Data cannot be empty")
    return process(data)

try:
    result = risky_operation([])
except ValueError as e:
    # Error automatically captured with full context
    print(f"Operation failed: {e}")
    # Stack trace, function args, timing all preserved</pre>
</div>

### Performance Monitoring
<div class="code-card">
  <h4>Built-in Observability</h4>
  <pre class="code-block"># Check SDK health
stats = vaquero.get_default_sdk().get_stats()
print(f"Traces: {stats['traces_sent']}")
print(f"Memory: {stats['memory_usage_mb']} MB")

# Manual control
vaquero.flush()  # Force send pending traces

# Get current context
from vaquero import get_current_span
span = get_current_span()
span.set_attribute("custom_metric", value)</pre>
</div>

</div>

---

## 🔧 Configuration Reference

<div class="config-table">
  <table>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Type</th>
        <th>Default</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>api_key</code></td>
        <td>string</td>
        <td>Required</td>
        <td>Your Vaquero API key</td>
      </tr>
      <tr>
        <td><code>project_id</code></td>
        <td>string</td>
        <td>Optional</td>
        <td>Your project identifier (auto-provisioned)</td>
      </tr>
      <tr>
        <td><code>batch_size</code></td>
        <td>int</td>
        <td>100</td>
        <td>Traces per batch</td>
      </tr>
      <tr>
        <td><code>flush_interval</code></td>
        <td>float</td>
        <td>5.0</td>
        <td>Seconds between flushes</td>
      </tr>
      <tr>
        <td><code>auto_instrument_llm</code></td>
        <td>bool</td>
        <td>true</td>
        <td>Auto-capture LLM calls</td>
      </tr>
      <tr>
        <td><code>capture_system_prompts</code></td>
        <td>bool</td>
        <td>true</td>
        <td>Capture LLM system prompts</td>
      </tr>
      <tr>
        <td><code>capture_code</code></td>
        <td>bool</td>
        <td>true</td>
        <td>Capture source code for analysis</td>
      </tr>
      <tr>
        <td><code>mode</code></td>
        <td>string</td>
        <td>"development"</td>
        <td>Operating mode ("development" or "production")</td>
      </tr>
    </tbody>
  </table>
</div>

---

## 🚨 Best Practices

<div class="best-practices">

### ✅ Do
- **Use descriptive agent names** - `@vaquero.trace("user_authentication_validator")`
- **Add meaningful attributes** - `span.set_attribute("user_id", user_id)`
- **Handle errors gracefully** - SDK captures exceptions automatically
- **Use async context managers** - `async with vaquero.span("operation"):`

### ❌ Avoid
- **Generic names** - `@vaquero.trace("validator")` (too vague)
- **Sensitive data** - Don't log passwords, keys, or PII
- **Blocking operations** - Use async patterns for I/O
- **Manual timing** - SDK handles timing automatically

</div>

---

## 🛠️ Development

<div class="dev-section">
  <div class="dev-card">
    <h4>🏗️ Setup</h4>
    <pre class="code-block">git clone https://github.com/vaquero/vaquero-python.git
cd vaquero-python
pip install -e ".[dev]"</pre>
  </div>

  <div class="dev-card">
    <h4>🧪 Testing</h4>
    <pre class="code-block">make test          # Run all tests
make test-cov      # With coverage
make lint          # Code quality</pre>
  </div>

  <div class="dev-card">
    <h4>📝 Contributing</h4>
    <p>Join our community! See <a href="docs/CONTRIBUTING.md">Contributing Guide</a></p>
  </div>
</div>

---

## 📖 Resources

<div class="resource-links">
  <a href="docs/GETTING_STARTED.md" class="resource-link">
    <span class="icon">🚀</span>
    <div>
      <h4>Getting Started</h4>
      <p>Complete installation and setup guide</p>
    </div>
  </a>

  <a href="docs/API_REFERENCE.md" class="resource-link">
    <span class="icon">📚</span>
    <div>
      <h4>API Reference</h4>
      <p>Detailed API documentation</p>
    </div>
  </a>

  <a href="examples/" class="resource-link">
    <span class="icon">💡</span>
    <div>
      <h4>Examples</h4>
      <p>Real-world usage patterns</p>
    </div>
  </a>
</div>

---

<div align="center">
  <p><strong>Need help?</strong> Join our <a href="https://discord.gg/vaquero">Discord community</a> or email <a href="mailto:support@vaquero.app">support@vaquero.app</a></p>
</div>
