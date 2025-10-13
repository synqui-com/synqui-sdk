"""
Vaquero SDK Analytics Module

This module handles anonymous analytics collection to help us understand:
- Which frameworks are being used (LangChain, CrewAI, etc.)
- SDK adoption and usage patterns
- Common integration patterns

Privacy: Only framework detection and SDK version info is collected.
No user code, data, or sensitive information is ever transmitted.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SDKAnalytics:
    """
    Analytics collector for SDK usage patterns.
    
    Collects anonymous telemetry to understand:
    - Framework usage (LangChain, CrewAI, etc.)
    - SDK version adoption
    - Integration patterns
    
    Privacy-first: No user code or sensitive data is collected.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        enabled: bool = True,
        posthog_api_key: Optional[str] = None,
        posthog_host: str = "https://us.i.posthog.com",
    ):
        """
        Initialize SDK analytics.
        
        Args:
            api_key: Vaquero API key (for user identification)
            project_id: Project ID (for grouping)
            enabled: Whether analytics is enabled (default: True)
            posthog_api_key: PostHog API key for analytics
            posthog_host: PostHog host URL
        """
        self.enabled = enabled and REQUESTS_AVAILABLE
        self.api_key = api_key
        self.project_id = project_id
        self.posthog_api_key = posthog_api_key or os.getenv('VAQUERO_POSTHOG_KEY')
        self.posthog_host = posthog_host
        
        # Detect framework on initialization
        self.detected_framework = self._detect_framework()
        self.sdk_version = self._get_sdk_version()
        
        # Track that SDK was initialized
        if self.enabled and self.posthog_api_key:
            self._track_sdk_initialized()
    
    def _detect_framework(self) -> str:
        """
        Detect which AI framework is being used.
        
        Returns:
            Framework name ('langchain', 'crewai', 'fastapi', etc.) or 'unknown'
        """
        try:
            # Check for LangChain
            if 'langchain' in sys.modules or 'langchain_core' in sys.modules:
                return 'langchain'
            
            # Check for CrewAI
            if 'crewai' in sys.modules:
                return 'crewai'
            
            # Check for FastAPI
            if 'fastapi' in sys.modules:
                return 'fastapi'
            
            # Check for Django
            if 'django' in sys.modules:
                return 'django'
            
            # Check for Flask
            if 'flask' in sys.modules:
                return 'flask'
            
            return 'unknown'
        except Exception as e:
            logger.debug(f"Error detecting framework: {e}")
            return 'unknown'
    
    def _get_sdk_version(self) -> str:
        """Get the SDK version."""
        try:
            from vaquero import __version__
            return __version__
        except:
            return 'unknown'
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _track_event(self, event_name: str, properties: Optional[Dict[str, Any]] = None):
        """
        Track an analytics event to PostHog.
        
        Args:
            event_name: Name of the event
            properties: Event properties
        """
        if not self.enabled or not self.posthog_api_key:
            return
        
        try:
            # Create unique distinct_id from api_key if available
            distinct_id = self._get_distinct_id()
            
            # Build event payload
            payload = {
                "api_key": self.posthog_api_key,
                "event": event_name,
                "properties": {
                    "distinct_id": distinct_id,
                    "sdk_version": self.sdk_version,
                    "framework": self.detected_framework,
                    "python_version": self._get_python_version(),
                    "platform": sys.platform,
                    "project_id": self.project_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    **(properties or {})
                },
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Send to PostHog
            response = requests.post(
                f"{self.posthog_host}/capture/",
                json=payload,
                timeout=2,  # Short timeout to not block user code
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.debug(f"Analytics event failed: {response.status_code}")
        
        except Exception as e:
            # Silently fail - analytics should never break user code
            logger.debug(f"Failed to track analytics event: {e}")
    
    def _get_distinct_id(self) -> str:
        """
        Generate a distinct ID for the user.
        Uses api_key hash if available, otherwise generates anonymous ID.
        """
        if self.api_key:
            # Use hash of API key for consistent user identification
            import hashlib
            return hashlib.sha256(self.api_key.encode()).hexdigest()[:16]
        else:
            # Generate anonymous ID
            import uuid
            return f"anon_{str(uuid.uuid4())[:8]}"
    
    def _track_sdk_initialized(self):
        """Track that the SDK was initialized."""
        self._track_event("SDK_INITIALIZED", {
            "framework": self.detected_framework,
            "sdk_version": self.sdk_version,
        })
    
    def track_first_trace(self, trace_id: str):
        """
        Track when the first trace is captured.
        
        Args:
            trace_id: ID of the first trace
        """
        self._track_event("FIRST_TRACE_CREATED", {
            "trace_id": trace_id,
            "framework": self.detected_framework,
        })
    
    def track_framework_feature(self, feature: str):
        """
        Track usage of a specific framework feature.
        
        Args:
            feature: Name of the framework feature being used
        """
        self._track_event("SDK_FRAMEWORK_FEATURE_USED", {
            "feature": feature,
            "framework": self.detected_framework,
        })
    
    def track_error(self, error_type: str, error_message: str):
        """
        Track SDK errors (for improving SDK reliability).
        
        Args:
            error_type: Type of error
            error_message: Error message (sanitized, no user data)
        """
        self._track_event("SDK_ERROR", {
            "error_type": error_type,
            "error_message": error_message[:200],  # Limit length
        })


# Global analytics instance
_analytics_instance: Optional[SDKAnalytics] = None


def initialize_analytics(
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    enabled: bool = True,
) -> SDKAnalytics:
    """
    Initialize global SDK analytics.
    
    Args:
        api_key: Vaquero API key
        project_id: Project ID
        enabled: Whether to enable analytics
    
    Returns:
        SDKAnalytics instance
    """
    global _analytics_instance
    
    # Check for opt-out environment variable
    if os.getenv('VAQUERO_ANALYTICS_DISABLED', '').lower() in ('true', '1', 'yes'):
        enabled = False
    
    _analytics_instance = SDKAnalytics(
        api_key=api_key,
        project_id=project_id,
        enabled=enabled,
    )
    
    return _analytics_instance


def get_analytics() -> Optional[SDKAnalytics]:
    """Get the global analytics instance."""
    return _analytics_instance


def track_event(event_name: str, properties: Optional[Dict[str, Any]] = None):
    """
    Track an analytics event using the global instance.
    
    Args:
        event_name: Name of the event
        properties: Event properties
    """
    if _analytics_instance:
        _analytics_instance._track_event(event_name, properties)
