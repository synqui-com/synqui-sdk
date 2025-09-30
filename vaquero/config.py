"""Configuration management for Vaquero SDK."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from urllib import request as _urlreq
from urllib.error import URLError, HTTPError


@dataclass
class SDKConfig:
    """SDK configuration settings.

    This class contains all the configuration options for the Vaquero SDK.
    It supports environment variable configuration and provides sensible defaults.

    Args:
        api_key: API key for authentication with Vaquero
        project_id: Project ID to associate traces with
        endpoint: Vaquero API endpoint URL
        batch_size: Number of events to batch before sending
        flush_interval: Interval in seconds to flush pending events
        max_retries: Maximum number of retry attempts for failed requests
        timeout: Request timeout in seconds
        capture_inputs: Whether to capture function inputs
        capture_outputs: Whether to capture function outputs
        capture_errors: Whether to capture error information
        environment: Environment name (development, staging, production)
        debug: Enable debug logging
        enabled: Whether the SDK is enabled
        tags: Global tags to add to all traces
        auto_instrument_llm: Whether to automatically instrument LLM libraries
        capture_system_prompts: Whether to automatically capture system prompts
        detect_agent_frameworks: Whether to auto-detect agent frameworks
        mode: Operating mode ("development" or "production")
    """

    api_key: str
    project_id: str
    endpoint: str = "https://api.vaquero.app"
    batch_size: int = 100
    flush_interval: float = 5.0
    max_retries: int = 3
    timeout: float = 30.0
    capture_inputs: bool = True
    capture_outputs: bool = True
    capture_errors: bool = True
    capture_tokens: bool = True
    environment: str = "development"
    debug: bool = False
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    auto_instrument_llm: bool = True
    capture_system_prompts: bool = True
    detect_agent_frameworks: bool = True
    mode: str = "development"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key and self.enabled:
            raise ValueError("api_key is required when SDK is enabled")
        if not self.project_id and self.enabled:
            # Defer strict requirement; SDK may auto-provision
            pass
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.flush_interval <= 0:
            raise ValueError("flush_interval must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.mode not in ["development", "production"]:
            raise ValueError("mode must be either 'development' or 'production'")


def configure_from_env() -> SDKConfig:
    """Configure SDK from environment variables.

    Environment variables:
        VAQUERO_API_KEY: API key
        VAQUERO_PROJECT_ID: Project ID
        VAQUERO_ENDPOINT: API endpoint URL
        VAQUERO_BATCH_SIZE: Batch size for events
        VAQUERO_FLUSH_INTERVAL: Flush interval in seconds
        VAQUERO_MAX_RETRIES: Maximum retry attempts
        VAQUERO_TIMEOUT: Request timeout in seconds
        VAQUERO_CAPTURE_INPUTS: Capture function inputs (true/false)
        VAQUERO_CAPTURE_OUTPUTS: Capture function outputs (true/false)
        VAQUERO_CAPTURE_ERRORS: Capture error information (true/false)
        VAQUERO_CAPTURE_TOKENS: Capture token counts (true/false)
        VAQUERO_ENVIRONMENT: Environment name
        VAQUERO_DEBUG: Enable debug logging (true/false)
        VAQUERO_ENABLED: Enable SDK (true/false)
        VAQUERO_TAGS: Global tags as JSON string

    Returns:
        SDKConfig instance configured from environment variables
    """

    def str_to_bool(value: str) -> bool:
        """Convert string to boolean."""
        return value.lower() in ("true", "1", "yes", "on")

    def parse_tags(tags_str: str) -> Dict[str, str]:
        """Parse tags from JSON string."""
        if not tags_str:
            return {}
        try:
            import json
            return json.loads(tags_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    return SDKConfig(
        api_key=os.getenv("VAQUERO_API_KEY", ""),
        project_id=os.getenv("VAQUERO_PROJECT_ID", ""),
        endpoint=os.getenv("VAQUERO_ENDPOINT", "https://api.vaquero.com"),
        batch_size=int(os.getenv("VAQUERO_BATCH_SIZE", "100")),
        flush_interval=float(os.getenv("VAQUERO_FLUSH_INTERVAL", "5.0")),
        max_retries=int(os.getenv("VAQUERO_MAX_RETRIES", "3")),
        timeout=float(os.getenv("VAQUERO_TIMEOUT", "30.0")),
        capture_inputs=str_to_bool(os.getenv("VAQUERO_CAPTURE_INPUTS", "true")),
        capture_outputs=str_to_bool(os.getenv("VAQUERO_CAPTURE_OUTPUTS", "true")),
        capture_errors=str_to_bool(os.getenv("VAQUERO_CAPTURE_ERRORS", "true")),
        capture_tokens=str_to_bool(os.getenv("VAQUERO_CAPTURE_TOKENS", "true")),
        environment=os.getenv("VAQUERO_ENVIRONMENT", "development"),
        debug=str_to_bool(os.getenv("VAQUERO_DEBUG", "false")),
        enabled=str_to_bool(os.getenv("VAQUERO_ENABLED", "true")),
        tags=parse_tags(os.getenv("VAQUERO_TAGS", "{}")),
        auto_instrument_llm=str_to_bool(os.getenv("VAQUERO_AUTO_INSTRUMENT_LLM", "true")),
        capture_system_prompts=str_to_bool(os.getenv("VAQUERO_CAPTURE_SYSTEM_PROMPTS", "true")),
        detect_agent_frameworks=str_to_bool(os.getenv("VAQUERO_DETECT_AGENT_FRAMEWORKS", "true")),
        mode=os.getenv("VAQUERO_MODE", "development")
    )


def configure(
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    batch_size: Optional[int] = None,
    flush_interval: Optional[float] = None,
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
    capture_inputs: Optional[bool] = None,
    capture_outputs: Optional[bool] = None,
    capture_errors: Optional[bool] = None,
    capture_tokens: Optional[bool] = None,
    environment: Optional[str] = None,
    debug: Optional[bool] = None,
    enabled: Optional[bool] = None,
    tags: Optional[Dict[str, str]] = None,
    auto_instrument_llm: Optional[bool] = None,
    capture_system_prompts: Optional[bool] = None,
    detect_agent_frameworks: Optional[bool] = None,
    mode: Optional[str] = None,
    **kwargs
) -> "VaqueroSDK":
    """Configure the Vaquero SDK.

    This function creates a new SDK configuration and initializes the default
    SDK instance. It first loads configuration from environment variables,
    then overrides with any provided parameters.

    Args:
        api_key: API key for authentication
        project_id: Project ID to associate traces with
        endpoint: Vaquero API endpoint URL
        batch_size: Number of events to batch before sending
        flush_interval: Interval in seconds to flush pending events
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        capture_inputs: Whether to capture function inputs
        capture_outputs: Whether to capture function outputs
        capture_errors: Whether to capture error information
        capture_tokens: Whether to capture token counts
        environment: Environment name
        debug: Enable debug logging
        enabled: Whether the SDK is enabled
        tags: Global tags to add to all traces
        **kwargs: Additional configuration options

    Returns:
        Configured SDK instance
    """

    # Start with environment configuration
    config = configure_from_env()

    # Override with provided parameters
    if api_key is not None:
        config.api_key = api_key
    if project_id is not None:
        config.project_id = project_id
    if endpoint is not None:
        config.endpoint = endpoint
    if batch_size is not None:
        config.batch_size = batch_size
    if flush_interval is not None:
        config.flush_interval = flush_interval
    if max_retries is not None:
        config.max_retries = max_retries
    if timeout is not None:
        config.timeout = timeout
    if capture_inputs is not None:
        config.capture_inputs = capture_inputs
    if capture_outputs is not None:
        config.capture_outputs = capture_outputs
    if capture_errors is not None:
        config.capture_errors = capture_errors
    if capture_tokens is not None:
        config.capture_tokens = capture_tokens
    if environment is not None:
        config.environment = environment
    if debug is not None:
        config.debug = debug
    if enabled is not None:
        config.enabled = enabled
    if tags is not None:
        config.tags.update(tags)
    if auto_instrument_llm is not None:
        config.auto_instrument_llm = auto_instrument_llm
    if capture_system_prompts is not None:
        config.capture_system_prompts = capture_system_prompts
    if detect_agent_frameworks is not None:
        config.detect_agent_frameworks = detect_agent_frameworks
    if mode is not None:
        config.mode = mode

    # Auto-provision project if enabled and missing
    auto_provision = os.getenv("VAQUERO_AUTO_PROVISION_PROJECT", "true").lower() == "true"
    if auto_provision and config.enabled and config.api_key and not config.project_id:
        resolved = _resolve_or_create_project(config.endpoint, config.api_key)
        if resolved:
            config.project_id = resolved

    # Create and set the default SDK instance
    from . import set_default_sdk
    from .sdk import VaqueroSDK
    sdk = VaqueroSDK(config)
    set_default_sdk(sdk)
    # Also set the global instance for workflow API (use the same instance)
    import vaquero.sdk
    vaquero.sdk._sdk_instance = sdk

    return sdk


def _resolve_or_create_project(endpoint: str, api_key: str) -> Optional[str]:
    """Resolve project via whoami; if none, create-or-get default project.

    Uses stdlib urllib to avoid adding dependencies.
    Returns project_id or None.
    """
    import json

    base = endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # whoami
    try:
        req = _urlreq.Request(f"{base}/api/v1/auth/whoami", headers=headers, method="GET")
        with _urlreq.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("project_id"):
                return data["project_id"]
    except (URLError, HTTPError):
        pass

    # create-or-get default
    try:
        req = _urlreq.Request(f"{base}/api/v1/projects/default", headers=headers, method="POST")
        with _urlreq.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("id")
    except (URLError, HTTPError):
        return None