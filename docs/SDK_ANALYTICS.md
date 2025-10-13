# Vaquero SDK Analytics

## Overview

The Vaquero SDK includes privacy-respecting analytics to help us understand how developers use the SDK and which frameworks are most popular. This helps us prioritize features and improvements.

## What We Collect

The SDK collects **anonymous usage patterns only**:

### Automatically Collected
- **Framework Detection**: Which framework you're using (LangChain, CrewAI, FastAPI, etc.)
- **SDK Version**: Which version of the Vaquero SDK you're using
- **Python Version**: Your Python runtime version
- **Platform**: Operating system (e.g., linux, darwin, win32)
- **First Trace**: When you capture your first trace (milestone tracking)

### What We DON'T Collect
- ❌ Your source code
- ❌ Function names or implementations
- ❌ Trace data content
- ❌ Input/output data
- ❌ Environment variables or secrets
- ❌ Personal information
- ❌ IP addresses (beyond what's in HTTP headers)

## How It Works

### Framework Detection

The SDK automatically detects which framework you're using by checking imported modules:

```python
# If you have LangChain imported
from langchain import ...

# The SDK detects: framework = "langchain"
```

Supported frameworks:
- **LangChain**: `langchain` or `langchain_core` modules
- **CrewAI**: `crewai` module
- **FastAPI**: `fastapi` module
- **Django**: `django` module  
- **Flask**: `flask` module
- **Unknown**: If no framework is detected

### Events Tracked

#### 1. SDK Initialized
**When**: First time you create a `VaqueroSDK` instance
**Data**:
```json
{
  "event": "SDK_INITIALIZED",
  "framework": "langchain",
  "sdk_version": "0.1.0",
  "python_version": "3.11.0",
  "platform": "darwin"
}
```

#### 2. First Trace Created
**When**: Your first trace is successfully captured
**Data**:
```json
{
  "event": "FIRST_TRACE_CREATED",
  "trace_id": "abc123...",
  "framework": "langchain"
}
```

## Privacy & Opt-Out

### Opt-Out Methods

You can disable analytics in three ways:

#### 1. Environment Variable (Recommended)
```bash
export VAQUERO_ANALYTICS_DISABLED=true
```

#### 2. SDK Configuration
```python
from vaquero import VaqueroSDK, SDKConfig

config = SDKConfig(
    api_key="your-key",
    project_id="your-project",
    enabled=False  # Disables analytics
)

sdk = VaqueroSDK(config)
```

#### 3. Remove PostHog API Key
If the `VAQUERO_POSTHOG_KEY` environment variable is not set, analytics are automatically disabled.

### What Happens When You Opt Out?

- ✅ All SDK functionality works normally
- ✅ Traces are still collected and sent to Vaquero
- ✅ No analytics events are sent
- ✅ No network requests to PostHog

## Technical Implementation

### Analytics Module

The SDK uses a separate `analytics.py` module that:
1. Detects frameworks by checking `sys.modules`
2. Sends anonymous events to PostHog
3. Fails silently if PostHog is unavailable
4. Never blocks your application code

### Network Requests

Analytics events are sent to:
- **Host**: `https://us.i.posthog.com`
- **Method**: POST
- **Timeout**: 2 seconds (short to avoid blocking)
- **Retry**: No retries (fire-and-forget)

### Error Handling

The analytics module follows these principles:
- **Never crash user code**: All exceptions are caught and logged at DEBUG level
- **Non-blocking**: Short timeouts ensure analytics don't slow down your app
- **Graceful degradation**: If `requests` library is not installed, analytics are disabled

### Distinct User IDs

To track unique users without PII:
- If you provide an API key: We use SHA256 hash of the API key (first 16 chars)
- If no API key: We generate an anonymous UUID per session
- We NEVER transmit the actual API key to PostHog

```python
# Example distinct ID generation
import hashlib

api_key = "vaq_abc123..."
distinct_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]
# Result: "a1b2c3d4e5f6g7h8"
```

## Data Retention

- **PostHog**: Events stored according to PostHog's retention policy (typically 1 year)
- **Vaquero**: We don't store analytics data beyond PostHog

## Compliance

- **GDPR**: PostHog is GDPR-compliant
- **CCPA**: PostHog is CCPA-compliant
- **Data Processing Agreement**: Available from PostHog

## Questions?

### Why do you collect analytics?

To understand:
- Which frameworks developers use most (should we prioritize LangChain vs CrewAI?)
- SDK adoption and version distribution (should we maintain older versions?)
- Time-to-value metrics (how long until first trace?)

### Can I see what's being sent?

Yes! Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('vaquero.analytics')
logger.setLevel(logging.DEBUG)
```

You'll see all analytics events in your logs.

### Is this open source?

Yes! The entire analytics module is in `vaquero/analytics.py` and you can review the code.

### Does this affect performance?

No. Analytics requests:
- Use a 2-second timeout
- Don't block your application
- Are sent asynchronously where possible
- Fail silently if PostHog is down

### Can I contribute framework detection?

Absolutely! If we're missing your framework, open a PR to add it to `_detect_framework()` in `analytics.py`.

## Example: Checking What's Tracked

```python
from vaquero import VaqueroSDK, SDKConfig
from vaquero.analytics import get_analytics

# Create SDK
config = SDKConfig(api_key="your-key", project_id="your-project")
sdk = VaqueroSDK(config)

# Check what framework was detected
analytics = get_analytics()
if analytics:
    print(f"Detected framework: {analytics.detected_framework}")
    print(f"SDK version: {analytics.sdk_version}")
    print(f"Analytics enabled: {analytics.enabled}")
```

## Changelog

- **2025-01-13**: Initial analytics implementation
  - Framework detection
  - SDK initialization tracking
  - First trace tracking
  - Opt-out support

---

**Privacy First**: We take your privacy seriously. If you have concerns about analytics, please opt out using any of the methods above or contact us at support@vaquero.app.
