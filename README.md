# 🤠 Synqui Python SDK

<div align="center">
  <h3>System intelligence for multi-agent AI</h3>
  <p><strong>Architecture extraction</strong> • <strong>Agent coordination</strong> • <strong>Performance insights</strong></p>

  [![PyPI version](https://badge.fury.io/py/synqui-sdk.svg)](https://badge.fury.io/py/synqui-sdk)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
</div>

---

## ⚡ Quick Start

### 1. Install
```bash
pip install synqui-sdk
```

> **Note:** The package is installed as `synqui-sdk`, but imported in Python as `import synqui`.

### 2. Get Your API Key
1. Go to the [Synqui Dashboard](https://www.synqui.app)
2. Create a new project (or select an existing one)
3. Navigate to the project's **Settings** page
4. Click **Create Project API Key** (keys start with `cf_`)

### 3. Initialize Synqui
```python
import synqui
import os

synqui.init(
    project_name="my-project",
    project_api_key=os.getenv("SYNQUI_PROJECT_API_KEY"),
    environment=os.getenv("SYNQUI_ENVIRONMENT", "development")
)
```

### 4. Integrate with LangChain
```python
from synqui.langchain import get_synqui_handler
from langchain_openai import ChatOpenAI

# Create handler
handler = get_synqui_handler(
    parent_context={"pipeline": "demo"}
)

# Use with LLM
llm = ChatOpenAI(callbacks=[handler])
llm.invoke("Hello, world!")

# Done! ✨ View your traces at https://www.synqui.app
```

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Automatic Failure Analysis** | AI-powered root cause analysis—know what broke and how to fix it instantly |
| 📊 **Architecture Performance Tracking** | Track architecture evolution with real data. Compare versions and know which configurations work before shipping |
| 🔧 **Development & Production Modes** | Debug fast in development. Monitor reliably in production. One platform for both |
| 🤖 **MCP Integration** | Your agents fix themselves. Connect via MCP to query insights, identify issues, and implement fixes automatically |
| 📈 **Performance Monitoring** | Track success rates, latency, and costs over time. Get alerted when patterns start degrading |
| 🏗️ **System Architecture Visualization** | Automatically extract and visualize agent relationships and coordination patterns |

---

## 📚 Documentation

- **[📖 Code Examples](docs/EXAMPLES.md)** - LangChain and LangGraph integration examples
- **[🎨 Advanced Usage](docs/ADVANCED.md)** - Session management, graph architecture, and custom configuration
- **[🔧 Configuration Reference](docs/CONFIGURATION.md)** - Complete parameter and environment variable reference
- **[🚨 Best Practices](docs/BEST_PRACTICES.md)** - Recommended patterns and common pitfalls
- **[💡 Examples Directory](examples/)** - Real-world examples and integration patterns
- **[🎯 Demo: Article Explainer](demos/article-explainer/)** - Full-featured demo application using LangGraph and Synqui
- **[🌐 Web Documentation](https://www.synqui.app/docs)** - Complete API reference and guides

---

## 🔧 Installation

### From PyPI (Recommended)
```bash
pip install synqui-sdk
```

### From Source
```bash
git clone https://github.com/nateislas/synqui-sdk.git
cd synqui-sdk
pip install -e .
```

### With All Dependencies
```bash
pip install synqui-sdk[all]
```

---

## 📝 Contributing

Join our community! See [Contributing Guide](CONTRIBUTING.md)

---

<div align="center">
  <p><strong>Need help?</strong> Join our <a href="https://discord.gg/synqui">Discord community</a> or email <a href="mailto:support@synqui.app">support@synqui.app</a></p>
</div>
