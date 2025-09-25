"""Automatic LLM instrumentation for CognitionFlow SDK.

This module provides automatic detection and instrumentation of LLM calls,
system prompts, and agent frameworks to minimize user setup.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SystemPromptDetector:
    """Detect system prompts from various sources."""
    
    def detect_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Detect system prompt from message array.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            System prompt content if found, None otherwise
        """
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if content and isinstance(content, str):
                    return content
        return None
    
    def detect_from_openai_call(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Detect system prompt from OpenAI-style messages."""
        return self.detect_from_messages(messages)
    
    def detect_from_anthropic_call(self, system: Optional[str] = None) -> Optional[str]:
        """Detect system prompt from Anthropic-style calls.
        
        Args:
            system: System prompt from Anthropic API
            
        Returns:
            System prompt content if found, None otherwise
        """
        if system and isinstance(system, str):
            return system
        return None
    
    def detect_from_langchain_prompt(self, prompt_template) -> Optional[str]:
        """Detect system prompt from LangChain prompt template.
        
        Args:
            prompt_template: LangChain prompt template
            
        Returns:
            System prompt content if found, None otherwise
        """
        try:
            if hasattr(prompt_template, 'messages'):
                for msg in prompt_template.messages:
                    if hasattr(msg, 'prompt') and 'system' in str(msg.prompt):
                        return str(msg.prompt)
            return None
        except Exception as e:
            logger.debug(f"Failed to detect LangChain system prompt: {e}")
            return None
    
    def detect_from_crewai_agent(self, agent) -> Optional[str]:
        """Detect system prompt from CrewAI agent.
        
        Args:
            agent: CrewAI agent instance
            
        Returns:
            System prompt content if found, None otherwise
        """
        try:
            parts = []
            if hasattr(agent, 'role') and agent.role:
                parts.append(f"Role: {agent.role}")
            if hasattr(agent, 'goal') and agent.goal:
                parts.append(f"Goal: {agent.goal}")
            if hasattr(agent, 'backstory') and agent.backstory:
                parts.append(f"Backstory: {agent.backstory}")
            
            if parts:
                return "\n".join(parts)
            return None
        except Exception as e:
            logger.debug(f"Failed to detect CrewAI system prompt: {e}")
            return None
    
    def detect_from_custom_agent(self, agent) -> Optional[str]:
        """Detect system prompt from custom agent implementations.
        
        Args:
            agent: Custom agent instance
            
        Returns:
            System prompt content if found, None otherwise
        """
        try:
            # Look for common patterns
            for attr_name in ['system_prompt', 'instructions', 'prompt', 'system_message']:
                if hasattr(agent, attr_name):
                    value = getattr(agent, attr_name)
                    if value and isinstance(value, str):
                        return value
            
            # Check for role + backstory pattern
            if hasattr(agent, 'role') and hasattr(agent, 'backstory'):
                role = getattr(agent, 'role', '')
                backstory = getattr(agent, 'backstory', '')
                if role or backstory:
                    parts = []
                    if role:
                        parts.append(f"Role: {role}")
                    if backstory:
                        parts.append(f"Backstory: {backstory}")
                    return "\n".join(parts)
            
            return None
        except Exception as e:
            logger.debug(f"Failed to detect custom agent system prompt: {e}")
            return None


class LLMCallTracker:
    """Track LLM calls and extract metrics."""
    
    def __init__(self, sdk_instance):
        """Initialize with SDK instance for trace creation.
        
        Args:
            sdk_instance: CognitionFlowSDK instance
        """
        self.sdk = sdk_instance
        self.system_prompt_detector = SystemPromptDetector()
    
    def track_openai_call(self, messages: List[Dict[str, Any]], model: str, 
                         result: Any, start_time: float, error: Optional[Exception] = None) -> None:
        """Track OpenAI API call.
        
        Args:
            messages: List of messages sent to API
            model: Model name used
            result: API response or None if error
            start_time: Start time of the call
            error: Exception if call failed
        """
        try:
            # Detect system prompt
            system_prompt = self.system_prompt_detector.detect_from_openai_call(messages)
            
            # Extract metrics
            duration = time.time() - start_time
            
            if error:
                self._track_failed_call(system_prompt, model, duration, error)
            else:
                # Extract token usage
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0
                
                if hasattr(result, 'usage') and result.usage:
                    input_tokens = getattr(result.usage, 'prompt_tokens', 0) or 0
                    output_tokens = getattr(result.usage, 'completion_tokens', 0) or 0
                    total_tokens = getattr(result.usage, 'total_tokens', 0) or (input_tokens + output_tokens)
                
                self._track_successful_call(
                    system_prompt=system_prompt,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    duration=duration
                )
                
        except Exception as e:
            logger.debug(f"Failed to track OpenAI call: {e}")
    
    def track_anthropic_call(self, system: Optional[str], model: str,
                           result: Any, start_time: float, error: Optional[Exception] = None) -> None:
        """Track Anthropic API call.
        
        Args:
            system: System prompt used
            model: Model name used
            result: API response or None if error
            start_time: Start time of the call
            error: Exception if call failed
        """
        try:
            # Detect system prompt
            system_prompt = self.system_prompt_detector.detect_from_anthropic_call(system)
            
            # Extract metrics
            duration = time.time() - start_time
            
            if error:
                self._track_failed_call(system_prompt, model, duration, error)
            else:
                # Extract token usage from Anthropic response
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0
                
                if hasattr(result, 'usage') and result.usage:
                    input_tokens = getattr(result.usage, 'input_tokens', 0) or 0
                    output_tokens = getattr(result.usage, 'output_tokens', 0) or 0
                    total_tokens = getattr(result.usage, 'total_tokens', 0) or (input_tokens + output_tokens)
                
                self._track_successful_call(
                    system_prompt=system_prompt,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    duration=duration
                )
                
        except Exception as e:
            logger.debug(f"Failed to track Anthropic call: {e}")
    
    def _track_successful_call(self, system_prompt: Optional[str], model: str,
                             input_tokens: int, output_tokens: int, total_tokens: int,
                             duration: float) -> None:
        """Track successful LLM call.
        
        Args:
            system_prompt: Detected system prompt
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total tokens used
            duration: Call duration in seconds
        """
        try:
            # Create a trace for the LLM call
            with self.sdk.span("llm_call") as span:
                # Set system prompt if detected
                if system_prompt:
                    span.system_prompt = system_prompt
                    span.prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
                    span.prompt_name = "auto_detected"
                    span.prompt_version = "v1"
                
                # Set model information
                span.model_name = model
                span.model_provider = self._detect_provider(model)
                
                # Set token information
                span.input_tokens = input_tokens
                span.output_tokens = output_tokens
                span.total_tokens = total_tokens
                
                # Set timing information
                span.metadata["duration"] = duration
                span.metadata["call_type"] = "llm"
                
                # Add tags
                span.tags.update({
                    "llm_call": "true",
                    "model": model,
                    "provider": span.model_provider
                })
                
        except Exception as e:
            logger.debug(f"Failed to track successful LLM call: {e}")
    
    def _track_failed_call(self, system_prompt: Optional[str], model: str,
                          duration: float, error: Exception) -> None:
        """Track failed LLM call.
        
        Args:
            system_prompt: Detected system prompt
            model: Model name
            duration: Call duration in seconds
            error: Exception that occurred
        """
        try:
            # Create a trace for the failed LLM call
            with self.sdk.span("llm_call_failed") as span:
                # Set system prompt if detected
                if system_prompt:
                    span.system_prompt = system_prompt
                    span.prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
                    span.prompt_name = "auto_detected"
                    span.prompt_version = "v1"
                
                # Set model information
                span.model_name = model
                span.model_provider = self._detect_provider(model)
                
                # Set error information
                span.set_error(error)
                
                # Set timing information
                span.metadata["duration"] = duration
                span.metadata["call_type"] = "llm_failed"
                
                # Add tags
                span.tags.update({
                    "llm_call": "true",
                    "llm_failed": "true",
                    "model": model,
                    "provider": span.model_provider
                })
                
        except Exception as e:
            logger.debug(f"Failed to track failed LLM call: {e}")
    
    def _detect_provider(self, model: str) -> str:
        """Detect provider from model name.
        
        Args:
            model: Model name
            
        Returns:
            Provider name (openai, anthropic, etc.)
        """
        model_lower = model.lower()
        if "gpt" in model_lower or "davinci" in model_lower or "curie" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "llama" in model_lower:
            return "llama"
        else:
            return "unknown"


class AutoInstrumentationEngine:
    """Core engine for automatic LLM instrumentation."""
    
    def __init__(self, sdk_instance):
        """Initialize with SDK instance.
        
        Args:
            sdk_instance: CognitionFlowSDK instance
        """
        self.sdk = sdk_instance
        self.llm_tracker = LLMCallTracker(sdk_instance)
        self.instrumented_libraries = set()
        self._original_methods = {}
    
    def instrument_openai(self) -> None:
        """Instrument OpenAI library for automatic tracking."""
        try:
            import openai
            
            if "openai" in self.instrumented_libraries:
                return
            
            # Handle different OpenAI API versions
            if hasattr(openai, 'ChatCompletion') and hasattr(openai.ChatCompletion, 'create'):
                # OpenAI < 1.0.0
                original_create = openai.ChatCompletion.create
                
                def instrumented_create(*args, **kwargs):
                    start_time = time.time()
                    messages = kwargs.get('messages', [])
                    model = kwargs.get('model', 'unknown')
                    
                    try:
                        result = original_create(*args, **kwargs)
                        self.llm_tracker.track_openai_call(messages, model, result, start_time)
                        return result
                    except Exception as e:
                        self.llm_tracker.track_openai_call(messages, model, None, start_time, e)
                        raise
                
                openai.ChatCompletion.create = instrumented_create
                self._original_methods["openai"] = original_create
                
            elif hasattr(openai, 'OpenAI'):
                # OpenAI >= 1.0.0 - instrument the client class
                original_init = openai.OpenAI.__init__
                
                def instrumented_init(self, *args, **kwargs):
                    original_init(self, *args, **kwargs)
                    
                    # Store original chat.completions.create method
                    if hasattr(self, 'chat') and hasattr(self.chat, 'completions'):
                        original_create = self.chat.completions.create
                        
                        def instrumented_create(*args, **kwargs):
                            start_time = time.time()
                            messages = kwargs.get('messages', [])
                            model = kwargs.get('model', 'unknown')
                            
                            try:
                                result = original_create(*args, **kwargs)
                                self.llm_tracker.track_openai_call(messages, model, result, start_time)
                                return result
                            except Exception as e:
                                self.llm_tracker.track_openai_call(messages, model, None, start_time, e)
                                raise
                        
                        self.chat.completions.create = instrumented_create
                
                openai.OpenAI.__init__ = instrumented_init
                self._original_methods["openai"] = original_init
            else:
                logger.debug("OpenAI library structure not recognized for instrumentation")
                return
            
            self.instrumented_libraries.add("openai")
            logger.info("OpenAI library instrumented for automatic tracking")
            
        except ImportError:
            logger.debug("OpenAI library not available for instrumentation")
        except Exception as e:
            logger.warning(f"Failed to instrument OpenAI library: {e}")
    
    def instrument_anthropic(self) -> None:
        """Instrument Anthropic library for automatic tracking."""
        try:
            import anthropic
            
            if "anthropic" in self.instrumented_libraries:
                return
            
            # Store original method
            original_create = anthropic.Anthropic.messages.create
            
            def instrumented_create(self, *args, **kwargs):
                start_time = time.time()
                system = kwargs.get('system')
                model = kwargs.get('model', 'unknown')
                
                try:
                    result = original_create(self, *args, **kwargs)
                    self.llm_tracker.track_anthropic_call(system, model, result, start_time)
                    return result
                except Exception as e:
                    self.llm_tracker.track_anthropic_call(system, model, None, start_time, e)
                    raise
            
            # Replace the method
            anthropic.Anthropic.messages.create = instrumented_create
            self.instrumented_libraries.add("anthropic")
            self._original_methods["anthropic"] = original_create
            
            logger.info("Anthropic library instrumented for automatic tracking")
            
        except ImportError:
            logger.debug("Anthropic library not available for instrumentation")
        except Exception as e:
            logger.warning(f"Failed to instrument Anthropic library: {e}")
    
    def instrument_all(self) -> None:
        """Instrument all available LLM libraries."""
        self.instrument_openai()
        self.instrument_anthropic()
    
    def restore_original_methods(self) -> None:
        """Restore original methods (for testing or cleanup)."""
        try:
            import openai
            if "openai" in self._original_methods:
                openai.ChatCompletion.create = self._original_methods["openai"]
                self.instrumented_libraries.discard("openai")
        except Exception as e:
            logger.debug(f"Failed to restore OpenAI methods: {e}")
        
        try:
            import anthropic
            if "anthropic" in self._original_methods:
                anthropic.Anthropic.messages.create = self._original_methods["anthropic"]
                self.instrumented_libraries.discard("anthropic")
        except Exception as e:
            logger.debug(f"Failed to restore Anthropic methods: {e}")
    
    def is_instrumented(self, library: str) -> bool:
        """Check if a library is instrumented.
        
        Args:
            library: Library name to check
            
        Returns:
            True if instrumented, False otherwise
        """
        return library in self.instrumented_libraries
