"""Configuration management for Vaquero SDK."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
from urllib import request as _urlreq
from urllib.error import URLError, HTTPError

if TYPE_CHECKING:
    from .sdk import VaqueroSDK


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
        environment: Environment name - "development", "staging", or "production" (controls batch_size and flush_interval presets)
        tags: Global tags to add to all traces
    """

    api_key: str
    project_id: str
    endpoint: str = "https://api.vaquero.app"
    batch_size: int = 100
    flush_interval: float = 5.0
    max_retries: int = 3
    timeout: float = 30.0
    environment: str = "development"
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.project_id:
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
        if self.environment not in ENVIRONMENT_PRESETS:
            raise ValueError(f"environment must be one of: {list(ENVIRONMENT_PRESETS.keys())}")


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
        VAQUERO_ENVIRONMENT: Environment name
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

    # Handle project identification from environment
    project_id = os.getenv("VAQUERO_PROJECT_ID", "")
    project_name = os.getenv("VAQUERO_PROJECT_NAME", "")
    
    # If project_name is provided, resolve it to project_id
    if project_name and not project_id:
        endpoint = os.getenv("VAQUERO_ENDPOINT", "https://api.vaquero.com")
        api_key = os.getenv("VAQUERO_PROJECT_API_KEY", "")
        if api_key:
            resolved_id = _resolve_project_by_name(endpoint, api_key, project_name)
            if resolved_id:
                project_id = resolved_id

    return SDKConfig(
        api_key=os.getenv("VAQUERO_PROJECT_API_KEY", ""),
        project_id=project_id,
        endpoint=os.getenv("VAQUERO_ENDPOINT", "https://api.vaquero.com"),
        batch_size=int(os.getenv("VAQUERO_BATCH_SIZE", "100")),
        flush_interval=float(os.getenv("VAQUERO_FLUSH_INTERVAL", "5.0")),
        max_retries=int(os.getenv("VAQUERO_MAX_RETRIES", "3")),
        timeout=float(os.getenv("VAQUERO_TIMEOUT", "30.0")),
        environment=os.getenv("VAQUERO_ENVIRONMENT", os.getenv("VAQUERO_MODE", "development")),
        tags=parse_tags(os.getenv("VAQUERO_TAGS", "{}"))
    )


def configure(
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    batch_size: Optional[int] = None,
    flush_interval: Optional[float] = None,
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
    environment: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
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
        environment: Environment name
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
    if environment is not None:
        config.environment = environment
    if tags is not None:
        config.tags.update(tags)

    # Auto-provision project if missing
    auto_provision = os.getenv("VAQUERO_AUTO_PROVISION_PROJECT", "true").lower() == "true"
    if auto_provision and config.api_key and not config.project_id:
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


# Environment presets for simplified configuration
ENVIRONMENT_PRESETS = {
    "development": {
        "batch_size": 10,
        "flush_interval": 2.0,
    },
    "staging": {
        "batch_size": 50,
        "flush_interval": 3.0,
    },
    "production": {
        "batch_size": 100,
        "flush_interval": 5.0,
    }
}


def init(
    project_api_key: str,
    project_id: Optional[str] = None,
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    environment: str = "development",
    **overrides
) -> "VaqueroSDK":
    """Initialize the Vaquero SDK with simplified configuration.

    This is the recommended way to initialize the SDK. It applies sensible
    presets based on the specified environment and allows for targeted overrides.

    Args:
        project_api_key: Project API key for authentication with Vaquero
        project_id: Project ID to associate traces with (optional)
        project_name: Project name to associate traces with (optional, takes precedence over project_id)
        endpoint: Vaquero API endpoint URL (optional)
        environment: Environment name - "development" (default), "staging", or "production"
        **overrides: Additional configuration overrides

    Returns:
        Configured SDK instance

    Example:
        # Simple development setup
        vaquero.init(project_api_key="your-key")

        # Using project name (recommended)
        vaquero.init(project_api_key="your-key", project_name="my-awesome-project")

        # Production setup
        vaquero.init(project_api_key="your-key", environment="production")

        # Custom configuration
        vaquero.init(
            project_api_key="your-key",
            project_name="my-project",
            batch_size=50
        )
    """
    # Validate environment
    if environment not in ENVIRONMENT_PRESETS:
        raise ValueError(f"Unknown environment '{environment}'. Must be one of: {list(ENVIRONMENT_PRESETS.keys())}")

    # If we have an explicit project_api_key, create config with it first to avoid validation errors
    # Otherwise, try to get from environment
    if project_api_key:
        # Create config with explicit API key and read other settings from environment manually
        # to avoid calling configure_from_env() which would fail validation
        def parse_tags(tags_str: str) -> Dict[str, str]:
            if not tags_str:
                return {}
            try:
                import json
                return json.loads(tags_str)
            except (json.JSONDecodeError, TypeError):
                return {}
        
        # Get environment values manually
        env_project_id = os.getenv("VAQUERO_PROJECT_ID", "")
        env_endpoint = os.getenv("VAQUERO_ENDPOINT", "https://api.vaquero.com")
        
        config = SDKConfig(
            api_key=project_api_key,  # Use explicit key to pass validation
            project_id=env_project_id if not project_id else None,
            endpoint=env_endpoint if not endpoint else None,
            batch_size=int(os.getenv("VAQUERO_BATCH_SIZE", "100")),
            flush_interval=float(os.getenv("VAQUERO_FLUSH_INTERVAL", "5.0")),
            max_retries=int(os.getenv("VAQUERO_MAX_RETRIES", "3")),
            timeout=float(os.getenv("VAQUERO_TIMEOUT", "30.0")),
            tags=parse_tags(os.getenv("VAQUERO_TAGS", "")),
        )
    else:
        # No explicit API key, try from environment
        config = configure_from_env()

    # Apply environment preset
    preset = ENVIRONMENT_PRESETS[environment].copy()
    for key, value in preset.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Override with provided parameters (these take precedence)
    if project_api_key is not None:
        config.api_key = project_api_key
    if endpoint is not None:
        config.endpoint = endpoint
    # Set environment after applying preset to ensure it takes precedence
    config.environment = environment
    
    # Handle project identification (project_name takes precedence over project_id)
    if project_name is not None:
        # Resolve project name to project ID
        resolved_project_id = _resolve_project_by_name(config.endpoint, config.api_key, project_name)
        if resolved_project_id:
            config.project_id = resolved_project_id
        else:
            raise ValueError(f"Project '{project_name}' not found. Please check the project name or create the project first.")
    elif project_id is not None:
        config.project_id = project_id

    # Apply additional overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration option: {key}")

    # Auto-provision project if missing
    auto_provision = os.getenv("VAQUERO_AUTO_PROVISION_PROJECT", "true").lower() == "true"
    if auto_provision and config.api_key and not config.project_id:
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


def _resolve_project_by_name(endpoint: str, api_key: str, project_name: str) -> Optional[str]:
    """Resolve project name to project ID by querying the API.

    Uses stdlib urllib to avoid adding dependencies.
    Returns project_id or None if not found.
    """
    import json

    base = endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        # First get user info to get user_id
        whoami_req = _urlreq.Request(f"{base}/api/v1/auth/whoami", headers=headers, method="GET")
        with _urlreq.urlopen(whoami_req, timeout=10) as resp:
            whoami_data = json.loads(resp.read().decode("utf-8"))
            user_id = whoami_data.get("user_id")
            
        if not user_id:
            return None
            
        # Add user ID to headers
        headers["X-User-ID"] = user_id
        
        # Get all projects for the user
        req = _urlreq.Request(f"{base}/api/v1/projects/", headers=headers, method="GET")
        with _urlreq.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
            # Look for project with matching name
            if isinstance(data, list):
                for project in data:
                    if project.get("name") == project_name:
                        return project.get("id")
            elif isinstance(data, dict) and "projects" in data:
                for project in data["projects"]:
                    if project.get("name") == project_name:
                        return project.get("id")
                        
    except (URLError, HTTPError, json.JSONDecodeError):
        pass
    
    return None