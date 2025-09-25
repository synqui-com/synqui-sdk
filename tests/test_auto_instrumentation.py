"""Tests for auto-instrumentation functionality."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from cognitionflow.config import SDKConfig
from cognitionflow.sdk import CognitionFlowSDK
from cognitionflow.auto_instrumentation import SystemPromptDetector, LLMCallTracker, AutoInstrumentationEngine


class TestSystemPromptDetector:
    """Test system prompt detection functionality."""
    
    def test_detect_from_messages(self):
        """Test detection from message array."""
        detector = SystemPromptDetector()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = detector.detect_from_messages(messages)
        assert result == "You are a helpful assistant"
    
    def test_detect_from_messages_no_system(self):
        """Test detection when no system message exists."""
        detector = SystemPromptDetector()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = detector.detect_from_messages(messages)
        assert result is None
    
    def test_detect_from_anthropic_call(self):
        """Test detection from Anthropic-style calls."""
        detector = SystemPromptDetector()
        
        result = detector.detect_from_anthropic_call("You are a helpful assistant")
        assert result == "You are a helpful assistant"
        
        result = detector.detect_from_anthropic_call(None)
        assert result is None
    
    def test_detect_from_crewai_agent(self):
        """Test detection from CrewAI agent."""
        detector = SystemPromptDetector()
        
        # Mock CrewAI agent
        agent = Mock()
        agent.role = "Data Analyst"
        agent.goal = "Analyze data"
        agent.backstory = "You are an expert data analyst"
        
        result = detector.detect_from_crewai_agent(agent)
        expected = "Role: Data Analyst\nGoal: Analyze data\nBackstory: You are an expert data analyst"
        assert result == expected
    
    def test_detect_from_custom_agent(self):
        """Test detection from custom agent."""
        detector = SystemPromptDetector()
        
        # Mock custom agent with system_prompt attribute
        agent = Mock()
        agent.system_prompt = "You are a helpful assistant"
        
        result = detector.detect_from_custom_agent(agent)
        assert result == "You are a helpful assistant"


class TestLLMCallTracker:
    """Test LLM call tracking functionality."""
    
    def test_track_successful_call(self):
        """Test tracking successful LLM call."""
        # Mock SDK
        sdk = Mock()
        sdk.span = Mock()
        sdk.span.return_value.__enter__ = Mock()
        sdk.span.return_value.__exit__ = Mock(return_value=None)
        
        tracker = LLMCallTracker(sdk)
        
        # Mock span
        span = Mock()
        sdk.span.return_value.__enter__.return_value = span
        
        # Track successful call
        tracker._track_successful_call(
            system_prompt="You are a helpful assistant",
            model="gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            duration=1.5
        )
        
        # Verify span was created and configured
        sdk.span.assert_called_once_with("llm_call")
        assert span.system_prompt == "You are a helpful assistant"
        assert span.model_name == "gpt-3.5-turbo"
        assert span.input_tokens == 100
        assert span.output_tokens == 50
        assert span.total_tokens == 150
    
    def test_track_failed_call(self):
        """Test tracking failed LLM call."""
        # Mock SDK
        sdk = Mock()
        sdk.span = Mock()
        sdk.span.return_value.__enter__ = Mock()
        sdk.span.return_value.__exit__ = Mock(return_value=None)
        
        tracker = LLMCallTracker(sdk)
        
        # Mock span
        span = Mock()
        sdk.span.return_value.__enter__.return_value = span
        
        # Track failed call
        error = Exception("API error")
        tracker._track_failed_call(
            system_prompt="You are a helpful assistant",
            model="gpt-3.5-turbo",
            duration=1.5,
            error=error
        )
        
        # Verify span was created and configured
        sdk.span.assert_called_once_with("llm_call_failed")
        assert span.system_prompt == "You are a helpful assistant"
        assert span.model_name == "gpt-3.5-turbo"
        span.set_error.assert_called_once_with(error)


class TestAutoInstrumentationEngine:
    """Test auto-instrumentation engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        sdk = Mock()
        engine = AutoInstrumentationEngine(sdk)
        
        assert engine.sdk == sdk
        assert engine.instrumented_libraries == set()
        assert isinstance(engine.llm_tracker, LLMCallTracker)
    
    @patch('cognitionflow.auto_instrumentation.openai')
    def test_instrument_openai(self, mock_openai):
        """Test OpenAI instrumentation."""
        sdk = Mock()
        engine = AutoInstrumentationEngine(sdk)
        
        # Mock OpenAI
        mock_openai.ChatCompletion.create = Mock()
        
        # Instrument OpenAI
        engine.instrument_openai()
        
        # Verify instrumentation
        assert "openai" in engine.instrumented_libraries
        assert "openai" in engine._original_methods
    
    @patch('cognitionflow.auto_instrumentation.anthropic')
    def test_instrument_anthropic(self, mock_anthropic):
        """Test Anthropic instrumentation."""
        sdk = Mock()
        engine = AutoInstrumentationEngine(sdk)
        
        # Mock Anthropic
        mock_anthropic.Anthropic.messages.create = Mock()
        
        # Instrument Anthropic
        engine.instrument_anthropic()
        
        # Verify instrumentation
        assert "anthropic" in engine.instrumented_libraries
        assert "anthropic" in engine._original_methods
    
    def test_is_instrumented(self):
        """Test instrumentation status check."""
        sdk = Mock()
        engine = AutoInstrumentationEngine(sdk)
        
        assert not engine.is_instrumented("openai")
        assert not engine.is_instrumented("anthropic")
        
        engine.instrumented_libraries.add("openai")
        assert engine.is_instrumented("openai")
        assert not engine.is_instrumented("anthropic")


class TestSDKIntegration:
    """Test SDK integration with auto-instrumentation."""
    
    def test_sdk_with_auto_instrumentation(self):
        """Test SDK initialization with auto-instrumentation."""
        config = SDKConfig(
            api_key="test-key",
            project_id="test-project",
            auto_instrument_llm=True,
            capture_system_prompts=True
        )
        
        with patch('cognitionflow.auto_instrumentation.AutoInstrumentationEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            
            sdk = CognitionFlowSDK(config)
            
            # Verify auto-instrumentation was initialized
            mock_engine_class.assert_called_once_with(sdk)
            mock_engine.instrument_all.assert_called_once()
    
    def test_sdk_without_auto_instrumentation(self):
        """Test SDK initialization without auto-instrumentation."""
        config = SDKConfig(
            api_key="test-key",
            project_id="test-project",
            auto_instrument_llm=False
        )
        
        with patch('cognitionflow.auto_instrumentation.AutoInstrumentationEngine') as mock_engine_class:
            sdk = CognitionFlowSDK(config)
            
            # Verify auto-instrumentation was not initialized
            mock_engine_class.assert_not_called()
    
    def test_sdk_shutdown_stops_instrumentation(self):
        """Test that SDK shutdown stops auto-instrumentation."""
        config = SDKConfig(
            api_key="test-key",
            project_id="test-project",
            auto_instrument_llm=True
        )
        
        with patch('cognitionflow.auto_instrumentation.AutoInstrumentationEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            
            sdk = CognitionFlowSDK(config)
            sdk.shutdown()
            
            # Verify auto-instrumentation was stopped
            mock_engine.restore_original_methods.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
