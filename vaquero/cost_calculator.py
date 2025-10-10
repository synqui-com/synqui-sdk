"""Cost calculation utilities for LLM providers.

This module provides cost calculation based on token usage and model pricing.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Pricing data (per 1K tokens) - updated as of October 2025
# NOTE: Pricing is subject to change. Verify with official sources before production use.
PRICING_DATA = {
    # OpenAI Models (as of October 2025)
    # Most models are priced per 1 million tokens, not 1 token.
    "gpt-5": {"input": 1.25, "output": 10.00, "unit": "per 1M tokens"},
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "unit": "per 1M tokens"},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "unit": "per 1M tokens"},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "unit": "per 1M tokens"},
    "gpt-4o": {"input": 2.50, "output": 10.00, "unit": "per 1M tokens"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "unit": "per 1M tokens"},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "unit": "per 1M tokens"},

    # Google Gemini Models (as of October 2025)
    # The prices listed in your dictionary were for a specific context size on the Vertex AI platform.
    # The corrected prices below are for the standard API based on official pricing information.
    "gemini-2.5-pro": {"input": 6.25, "output": 50.00, "unit": "per 1M tokens"},
    "gemini-2.5-pro-200k": {"input": 0.625, "output": 5.00, "unit": "per 1M tokens"},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "unit": "per 1M tokens"},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "unit": "per 1M tokens"},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30, "unit": "per 1M tokens"},
    "gemini-pro": {"input": "deprecated", "output": "deprecated", "unit": "N/A"},
    "gemini-pro-vision": {"input": "deprecated", "output": "deprecated", "unit": "N/A"},

    # Anthropic Models (as of October 2025)
    "claude-4.1-opus": {"input": 20.00, "output": 80.00, "unit": "per 1M tokens"},
    "claude-4.1-sonnet": {"input": 5.00, "output": 25.00, "unit": "per 1M tokens"},
    "claude-3.5-haiku": {"input": 0.80, "output": 4.00, "unit": "per 1M tokens"},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00, "unit": "per 1M tokens"},

    # Default fallback pricing (conservative estimate)
    "default": {"input": 0.001, "output": 0.002, "unit": "per 1k tokens"}
}


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: Optional[str] = None,
    provider: Optional[str] = None
) -> float:
    """Calculate cost based on token usage and model pricing.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_name: Name of the model used
        provider: Provider of the model (openai, google, anthropic, etc.)
        
    Returns:
        Calculated cost in USD
        
    Note:
        Pricing data may be outdated. For production use, verify current pricing
        with the official provider documentation.
    """
    if input_tokens <= 0 and output_tokens <= 0:
        return 0.0
    
    # Normalize model name for lookup
    normalized_model = _normalize_model_name(model_name, provider)
    
    # Get pricing for the model
    pricing = PRICING_DATA.get(normalized_model, PRICING_DATA["default"])
    
    # Log pricing source for transparency
    if normalized_model in PRICING_DATA:
        unit = PRICING_DATA[normalized_model].get("unit", "per 1k tokens")
        logger.debug(f"Using pricing for {normalized_model}: {unit}")
    
    # Handle deprecated models
    if pricing["input"] == "deprecated" or pricing["output"] == "deprecated":
        logger.warning(f"Model {normalized_model} is deprecated. Using default pricing.")
        pricing = PRICING_DATA["default"]
    
    # Calculate cost based on unit
    unit = pricing.get("unit", "per 1k tokens")
    if "1M" in unit or "million" in unit.lower():
        # Per 1 million tokens
        input_cost = (input_tokens / 1_000_000.0) * pricing["input"]
        output_cost = (output_tokens / 1_000_000.0) * pricing["output"]
        unit_display = "1M"
    else:
        # Per 1 thousand tokens (default)
        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]
        unit_display = "1K"
    
    total_cost = input_cost + output_cost
    
    logger.debug(f"Cost calculation for {normalized_model}: "
                f"{input_tokens} input tokens @ ${pricing['input']}/{unit_display} = ${input_cost:.6f}, "
                f"{output_tokens} output tokens @ ${pricing['output']}/{unit_display} = ${output_cost:.6f}, "
                f"total = ${total_cost:.6f}")
    
    return round(total_cost, 6)


def _normalize_model_name(model_name: Optional[str], provider: Optional[str]) -> str:
    """Normalize model name for pricing lookup.
    
    Args:
        model_name: Raw model name from the API
        provider: Provider name
        
    Returns:
        Normalized model name for pricing lookup
    """
    if not model_name:
        return "default"
    
    # Convert to lowercase for case-insensitive matching
    model_lower = model_name.lower()
    
    # Handle different naming conventions
    if "gpt-5" in model_lower:
        if "mini" in model_lower:
            return "gpt-5-mini"
        else:
            return "gpt-5"
    elif "gpt-4.1" in model_lower:
        if "mini" in model_lower:
            return "gpt-4.1-mini"
        else:
            return "gpt-4.1"
    elif "gpt-4" in model_lower:
        if "o" in model_lower and "mini" in model_lower:
            return "gpt-4o-mini"
        elif "o" in model_lower:
            return "gpt-4o"
        else:
            return "gpt-4o"  # Default to gpt-4o for legacy gpt-4
    elif "gpt-3.5" in model_lower:
        return "gpt-3.5-turbo"
    elif "gemini" in model_lower:
        if "2.5" in model_lower and "pro" in model_lower and "200k" in model_lower:
            return "gemini-2.5-pro-200k"
        elif "2.5" in model_lower and "pro" in model_lower:
            return "gemini-2.5-pro"
        elif "2.5" in model_lower and "lite" in model_lower:
            return "gemini-2.5-flash-lite"
        elif "2.5" in model_lower:
            return "gemini-2.5-flash"
        elif "2.0" in model_lower and "lite" in model_lower:
            return "gemini-2.0-flash-lite"
        elif "2.0" in model_lower:
            return "gemini-2.0-flash-lite"  # Default to lite for 2.0
        elif "vision" in model_lower:
            return "gemini-pro-vision"  # Deprecated but handle gracefully
        else:
            return "gemini-pro"  # Deprecated but handle gracefully
    elif "claude" in model_lower:
        if "4.1" in model_lower and "opus" in model_lower:
            return "claude-4.1-opus"
        elif "4.1" in model_lower and "sonnet" in model_lower:
            return "claude-4.1-sonnet"
        elif "3.5" in model_lower and "sonnet" in model_lower:
            return "claude-3.5-sonnet"
        elif "3.5" in model_lower and "haiku" in model_lower:
            return "claude-3.5-haiku"
        elif "opus" in model_lower:
            return "claude-4.1-opus"  # Default to latest version
        elif "sonnet" in model_lower:
            return "claude-3.5-sonnet"  # Default to latest version
        elif "haiku" in model_lower:
            return "claude-3.5-haiku"
    
    # If no match found, try to use provider-specific default
    if provider:
        provider_lower = provider.lower()
        if "openai" in provider_lower:
            return "gpt-3.5-turbo"  # Default OpenAI model
        elif "google" in provider_lower:
            return "gemini-2.5-flash-lite"  # Default Google model (free tier)
        elif "anthropic" in provider_lower:
            return "claude-3.5-haiku"  # Default Anthropic model (cheapest)
    
    # Fallback to generic default
    return "default"


def get_model_pricing(model_name: Optional[str], provider: Optional[str] = None) -> Dict[str, float]:
    """Get pricing information for a specific model.
    
    Args:
        model_name: Name of the model
        provider: Provider of the model
        
    Returns:
        Dictionary with input and output pricing per 1K tokens
    """
    normalized_model = _normalize_model_name(model_name, provider)
    return PRICING_DATA.get(normalized_model, PRICING_DATA["default"])


def list_supported_models() -> Dict[str, Dict[str, float]]:
    """List all supported models and their pricing.
    
    Returns:
        Dictionary mapping model names to their pricing
    """
    return PRICING_DATA.copy()
