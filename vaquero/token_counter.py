"""Token counting utilities for the CognitionFlow SDK.

This module provides automatic token counting for various LLM providers
and text inputs, enabling accurate cost estimation and performance monitoring.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class TokenCount:
    """Token count information for a text or LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    model: Optional[str] = None
    provider: Optional[str] = None
    
    def __post_init__(self):
        """Calculate total tokens if not explicitly set."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


class TokenCounter:
    """Token counter for various LLM providers and text inputs."""
    
    def __init__(self):
        """Initialize the token counter."""
        self._encoders: Dict[str, Any] = {}
        self._model_mappings = {
            # OpenAI models
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4",
            "gpt-4-32k": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k": "gpt-3.5-turbo",
            "text-davinci-003": "text-davinci-003",
            "text-davinci-002": "text-davinci-002",
            "text-davinci-001": "text-davinci-001",
            "text-curie-001": "text-curie-001",
            "text-babbage-001": "text-babbage-001",
            "text-ada-001": "text-ada-001",
            
            # Anthropic models (approximate with GPT-4)
            "claude-3-opus": "gpt-4",
            "claude-3-sonnet": "gpt-4",
            "claude-3-haiku": "gpt-3.5-turbo",
            "claude-2": "gpt-4",
            "claude-1": "gpt-4",
            
            # Other models (fallback to GPT-3.5-turbo)
            "llama-2": "gpt-3.5-turbo",
            "llama-3": "gpt-3.5-turbo",
            "mistral": "gpt-3.5-turbo",
            "mixtral": "gpt-3.5-turbo",
        }
    
    def _get_encoder(self, model: str) -> Any:
        """Get or create encoder for a model."""
        if not TIKTOKEN_AVAILABLE:
            logger.warning("tiktoken not available, using fallback token counting")
            return None
        
        # Map model to tiktoken model name
        tiktoken_model = self._model_mappings.get(model, "gpt-3.5-turbo")
        
        if tiktoken_model not in self._encoders:
            try:
                self._encoders[tiktoken_model] = tiktoken.encoding_for_model(tiktoken_model)
            except KeyError:
                # Fallback to cl100k_base encoding (used by GPT-4 and GPT-3.5-turbo)
                logger.warning(f"Unknown model {model}, using cl100k_base encoding")
                self._encoders[tiktoken_model] = tiktoken.get_encoding("cl100k_base")
        
        return self._encoders[tiktoken_model]
    
    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text using the appropriate encoder.
        
        Args:
            text: Text to count tokens for
            model: Model name to determine encoding
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        encoder = self._get_encoder(model)
        if encoder:
            try:
                return len(encoder.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed: {e}")
        
        # Fallback: rough estimation (4 characters per token)
        return len(text) // 4
    
    def count_llm_call_tokens(
        self, 
        messages: List[Dict[str, str]], 
        response: str = "",
        model: str = "gpt-3.5-turbo"
    ) -> TokenCount:
        """Count tokens for an LLM call.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            response: Response text from the LLM
            model: Model name
            
        Returns:
            TokenCount object with input, output, and total tokens
        """
        # Count input tokens
        input_text = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                input_text += message["content"] + "\n"
        
        input_tokens = self.count_tokens(input_text, model)
        output_tokens = self.count_tokens(response, model)
        
        return TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=model
        )
    
    def count_function_tokens(
        self, 
        inputs: Any, 
        outputs: Any, 
        model: str = "gpt-3.5-turbo"
    ) -> TokenCount:
        """Count tokens for function inputs and outputs.
        
        Args:
            inputs: Function inputs
            outputs: Function outputs
            model: Model name
            
        Returns:
            TokenCount object
        """
        # Convert inputs and outputs to strings for counting
        input_text = self._serialize_for_counting(inputs)
        output_text = self._serialize_for_counting(outputs)
        
        input_tokens = self.count_tokens(input_text, model)
        output_tokens = self.count_tokens(output_text, model)
        
        return TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=model
        )
    
    def _serialize_for_counting(self, obj: Any) -> str:
        """Serialize object to string for token counting.
        
        Args:
            obj: Object to serialize
            
        Returns:
            String representation
        """
        if obj is None:
            return ""
        
        if isinstance(obj, str):
            return obj
        
        if isinstance(obj, (int, float, bool)):
            return str(obj)
        
        if isinstance(obj, (list, tuple)):
            return " ".join(self._serialize_for_counting(item) for item in obj)
        
        if isinstance(obj, dict):
            return " ".join(f"{k}: {self._serialize_for_counting(v)}" for k, v in obj.items())
        
        # For complex objects, use string representation
        return str(obj)
    
    def extract_tokens_from_llm_response(self, response: Any) -> TokenCount:
        """Extract token information from LLM API response.
        
        This method attempts to extract token counts from various LLM API responses.
        
        Args:
            response: LLM API response object
            
        Returns:
            TokenCount object
        """
        # Handle OpenAI-style responses
        if hasattr(response, 'usage'):
            usage = response.usage
            return TokenCount(
                input_tokens=getattr(usage, 'prompt_tokens', 0),
                output_tokens=getattr(usage, 'completion_tokens', 0),
                total_tokens=getattr(usage, 'total_tokens', 0),
                model=getattr(response, 'model', None),
                provider="openai"
            )
        
        # Handle Anthropic-style responses
        if hasattr(response, 'usage'):
            usage = response.usage
            return TokenCount(
                input_tokens=getattr(usage, 'input_tokens', 0),
                output_tokens=getattr(usage, 'output_tokens', 0),
                total_tokens=getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0),
                model=getattr(response, 'model', None),
                provider="anthropic"
            )
        
        # Handle dictionary responses
        if isinstance(response, dict):
            # Handle standard LLM response format
            if 'input_tokens' in response and 'output_tokens' in response:
                return TokenCount(
                    input_tokens=response.get('input_tokens', 0),
                    output_tokens=response.get('output_tokens', 0),
                    total_tokens=response.get('total_tokens', 0),
                    model=response.get('model'),
                    provider=response.get('provider')
                )
            # Handle demo result format with tokens_used field
            elif 'tokens_used' in response:
                tokens_used = response.get('tokens_used', 0)
                return TokenCount(
                    input_tokens=0,  # Demo format doesn't separate input/output
                    output_tokens=tokens_used,
                    total_tokens=tokens_used,
                    model=response.get('model'),
                    provider=response.get('provider')
                )
            # Handle other dictionary formats
            else:
                return TokenCount(
                    input_tokens=response.get('input_tokens', 0),
                    output_tokens=response.get('output_tokens', 0),
                    total_tokens=response.get('total_tokens', 0),
                    model=response.get('model'),
                    provider=response.get('provider')
                )
        
        # Fallback: no token information available
        logger.warning("Could not extract token information from response")
        return TokenCount()


# Global token counter instance
_token_counter = TokenCounter()


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text.
    
    Args:
        text: Text to count tokens for
        model: Model name
        
    Returns:
        Number of tokens
    """
    return _token_counter.count_tokens(text, model)


def count_llm_call_tokens(
    messages: List[Dict[str, str]], 
    response: str = "",
    model: str = "gpt-3.5-turbo"
) -> TokenCount:
    """Count tokens for an LLM call.
    
    Args:
        messages: List of message dictionaries
        response: Response text
        model: Model name
        
    Returns:
        TokenCount object
    """
    return _token_counter.count_llm_call_tokens(messages, response, model)


def count_function_tokens(
    inputs: Any, 
    outputs: Any, 
    model: str = "gpt-3.5-turbo"
) -> TokenCount:
    """Count tokens for function inputs and outputs.
    
    Args:
        inputs: Function inputs
        outputs: Function outputs
        model: Model name
        
    Returns:
        TokenCount object
    """
    return _token_counter.count_function_tokens(inputs, outputs, model)


def extract_tokens_from_llm_response(response: Any) -> TokenCount:
    """Extract token information from LLM API response.
    
    Args:
        response: LLM API response object
        
    Returns:
        TokenCount object
    """
    return _token_counter.extract_tokens_from_llm_response(response)
