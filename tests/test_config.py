"""Tests for SDK configuration and initialization."""

import os
import pytest
from unittest.mock import patch

from vaquero.config import SDKConfig, init, configure, configure_from_env, MODE_PRESETS


class TestModePresets:
    """Test mode presets functionality."""

    def test_mode_presets_have_required_keys(self):
        """Test that mode presets contain all expected configuration keys."""
        expected_keys = {
            'capture_inputs', 'capture_outputs', 'capture_errors',
            'capture_code', 'capture_tokens', 'auto_instrument_llm',
            'capture_system_prompts', 'detect_agent_frameworks',
            'debug', 'batch_size', 'flush_interval'
        }

        for mode, preset in MODE_PRESETS.items():
            assert isinstance(preset, dict), f"Preset for {mode} should be a dict"
            assert expected_keys.issubset(set(preset.keys())), f"Preset for {mode} missing keys"

    def test_development_mode_values(self):
        """Test development mode has expected values."""
        dev_preset = MODE_PRESETS['development']
        assert dev_preset['capture_inputs'] is True
        assert dev_preset['capture_outputs'] is True
        assert dev_preset['capture_errors'] is True
        assert dev_preset['capture_code'] is True
        assert dev_preset['capture_tokens'] is True
        assert dev_preset['auto_instrument_llm'] is True
        assert dev_preset['capture_system_prompts'] is True
        assert dev_preset['detect_agent_frameworks'] is True
        assert dev_preset['debug'] is True
        assert dev_preset['batch_size'] == 10
        assert dev_preset['flush_interval'] == 0.5  # Updated for lower latency

    def test_production_mode_values(self):
        """Test production mode has expected values."""
        prod_preset = MODE_PRESETS['production']
        assert prod_preset['capture_inputs'] is False
        assert prod_preset['capture_outputs'] is False
        assert prod_preset['capture_errors'] is True
        assert prod_preset['capture_code'] is False
        assert prod_preset['capture_tokens'] is True
        assert prod_preset['auto_instrument_llm'] is False
        assert prod_preset['capture_system_prompts'] is False
        assert prod_preset['detect_agent_frameworks'] is False
        assert prod_preset['debug'] is False
        assert prod_preset['batch_size'] == 100
        assert prod_preset['flush_interval'] == 1.0  # Updated for better responsiveness



class TestSDKConfig:
    """Test SDKConfig class."""

    def test_sdk_config_validation(self):
        """Test SDKConfig validation logic."""
        # Valid config
        config = SDKConfig(
            api_key="test-key",
            project_id="test-project",
            mode="development"
        )
        assert config.mode == "development"

        # Invalid mode
        with pytest.raises(ValueError, match="mode must be one of"):
            SDKConfig(
                api_key="test-key",
                project_id="test-project",
                mode="invalid_mode"
            )

    def test_sdk_config_capture_code_field(self):
        """Test that SDKConfig has capture_code field."""
        config = SDKConfig(
            api_key="test-key",
            project_id="test-project"
        )
        assert hasattr(config, 'capture_code')
        assert config.capture_code is True  # Default value

    def test_sdk_config_mode_validation_with_presets(self):
        """Test that mode validation uses MODE_PRESETS."""
        # Should work with valid modes
        for mode in MODE_PRESETS.keys():
            config = SDKConfig(
                api_key="test-key",
                project_id="test-project",
                mode=mode
            )
            assert config.mode == mode


class TestInitFunction:
    """Test init() function."""

    def test_init_default_mode(self):
        """Test init() defaults to development mode."""
        with patch.dict(os.environ, {
            'VAQUERO_ENABLED': 'true',
            'VAQUERO_API_KEY': 'dummy-key-for-testing',
            'VAQUERO_PROJECT_ID': '',
            'VAQUERO_ENDPOINT': 'https://api.vaquero.com'
        }, clear=True):
            with patch('vaquero.config._resolve_or_create_project', return_value=None):
                with patch('vaquero.set_default_sdk') as mock_set_sdk:
                    sdk = init(api_key="test-key", project_id="test-project")

                    # Check that set_default_sdk was called
                    mock_set_sdk.assert_called_once()

                    # Check the SDK config has development mode defaults
                    args, kwargs = mock_set_sdk.call_args
                    sdk_instance = args[0]
                    config = sdk_instance.config

                    assert config.mode == "development"
                    assert config.capture_inputs is True
                    assert config.debug is True
                    assert config.batch_size == 10

    def test_init_explicit_development_mode(self):
        """Test init() with explicit development mode."""
        with patch.dict(os.environ, {
            'VAQUERO_ENABLED': 'true',
            'VAQUERO_API_KEY': 'dummy-key-for-testing',
            'VAQUERO_PROJECT_ID': '',
            'VAQUERO_ENDPOINT': 'https://api.vaquero.com'
        }, clear=True):
            with patch('vaquero.config._resolve_or_create_project', return_value=None):
                with patch('vaquero.set_default_sdk') as mock_set_sdk:
                    sdk = init(api_key="test-key", mode="development")

                    args, kwargs = mock_set_sdk.call_args
                    sdk_instance = args[0]
                    config = sdk_instance.config

                    assert config.mode == "development"
                    assert config.capture_inputs is True

    def test_init_production_mode(self):
        """Test init() with production mode."""
        with patch.dict(os.environ, {
            'VAQUERO_ENABLED': 'true',
            'VAQUERO_API_KEY': 'dummy-key-for-testing',
            'VAQUERO_PROJECT_ID': '',
            'VAQUERO_ENDPOINT': 'https://api.vaquero.com',
            'VAQUERO_MODE': 'development'  # Start with development in env
        }, clear=True):
            with patch('vaquero.config._resolve_or_create_project', return_value=None):
                with patch('vaquero.set_default_sdk') as mock_set_sdk:
                    sdk = init(api_key="test-key", mode="production")

                    args, kwargs = mock_set_sdk.call_args
                    sdk_instance = args[0]
                    config = sdk_instance.config

                    assert config.mode == "production"
                    assert config.capture_inputs is False
                    assert config.capture_outputs is False
                    assert config.debug is False
                    assert config.batch_size == 100


    def test_init_invalid_mode(self):
        """Test init() with invalid mode raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Unknown mode 'invalid'"):
                init(api_key="test-key", mode="invalid")

    def test_init_with_overrides(self):
        """Test init() with configuration overrides."""
        with patch.dict(os.environ, {
            'VAQUERO_ENABLED': 'true',
            'VAQUERO_API_KEY': 'dummy-key-for-testing',
            'VAQUERO_PROJECT_ID': '',
            'VAQUERO_ENDPOINT': 'https://api.vaquero.com'
        }, clear=True):
            with patch('vaquero.config._resolve_or_create_project', return_value=None):
                with patch('vaquero.set_default_sdk') as mock_set_sdk:
                    sdk = init(
                        api_key="test-key",
                        project_id="test-project",
                        capture_inputs=False,
                        batch_size=50
                    )

                    args, kwargs = mock_set_sdk.call_args
                    sdk_instance = args[0]
                    config = sdk_instance.config

                    assert config.capture_inputs is False  # Override applied
                    assert config.batch_size == 50  # Override applied
                    assert config.mode == "development"  # Default preserved

    def test_init_with_endpoint_override(self):
        """Test init() with endpoint override."""
        with patch.dict(os.environ, {
            'VAQUERO_ENABLED': 'true',
            'VAQUERO_API_KEY': 'dummy-key-for-testing',
            'VAQUERO_PROJECT_ID': '',
            'VAQUERO_ENDPOINT': 'https://api.vaquero.com'
        }, clear=True):
            with patch('vaquero.config._resolve_or_create_project', return_value=None):
                with patch('vaquero.set_default_sdk') as mock_set_sdk:
                    custom_endpoint = "https://custom.api.example.com"
                    sdk = init(
                        api_key="test-key",
                        endpoint=custom_endpoint
                    )

                    args, kwargs = mock_set_sdk.call_args
                    sdk_instance = args[0]
                    config = sdk_instance.config

                    assert config.endpoint == custom_endpoint


class TestConfigureFromEnv:
    """Test configure_from_env() function."""

    def test_configure_from_env_defaults(self):
        """Test configure_from_env() with default values."""
        with patch.dict(os.environ, {
            'VAQUERO_API_KEY': 'dummy-key-for-testing',
            'VAQUERO_PROJECT_ID': '',
            'VAQUERO_ENABLED': 'false'  # Disable SDK to avoid validation error
        }, clear=True):
            config = configure_from_env()

            # Should have default values (but we set a dummy key for testing)
            assert config.api_key == "dummy-key-for-testing"
            assert config.project_id == ""
            assert config.mode == "development"
            assert config.enabled is False

    def test_configure_from_env_with_values(self):
        """Test configure_from_env() with environment variables set."""
        with patch.dict(os.environ, {
            'VAQUERO_API_KEY': 'env-api-key',
            'VAQUERO_PROJECT_ID': 'env-project-id',
            'VAQUERO_MODE': 'production',
            'VAQUERO_CAPTURE_CODE': 'false',
            'VAQUERO_BATCH_SIZE': '200'
        }, clear=True):
            config = configure_from_env()

            assert config.api_key == 'env-api-key'
            assert config.project_id == 'env-project-id'
            assert config.mode == 'production'
            assert config.capture_code is False
            assert config.batch_size == 200


class TestConfigureFunction:
    """Test configure() function for backward compatibility."""

    def test_configure_basic(self):
        """Test basic configure() functionality."""
        with patch.dict(os.environ, {
            'VAQUERO_ENABLED': 'true',
            'VAQUERO_API_KEY': 'dummy-key-for-testing',
            'VAQUERO_PROJECT_ID': '',
            'VAQUERO_ENDPOINT': 'https://api.vaquero.com'
        }, clear=True):
            with patch('vaquero.config._resolve_or_create_project', return_value=None):
                with patch('vaquero.set_default_sdk') as mock_set_sdk:
                    sdk = configure(api_key="test-key", project_id="test-project")

                    mock_set_sdk.assert_called_once()
                    args, kwargs = mock_set_sdk.call_args
                    sdk_instance = args[0]
                    config = sdk_instance.config

                    assert config.api_key == "test-key"
                    assert config.project_id == "test-project"

    def test_configure_with_capture_code(self):
        """Test configure() with capture_code parameter."""
        with patch.dict(os.environ, {
            'VAQUERO_ENABLED': 'true',
            'VAQUERO_API_KEY': 'dummy-key-for-testing',
            'VAQUERO_PROJECT_ID': '',
            'VAQUERO_ENDPOINT': 'https://api.vaquero.com'
        }, clear=True):
            with patch('vaquero.config._resolve_or_create_project', return_value=None):
                with patch('vaquero.set_default_sdk') as mock_set_sdk:
                    sdk = configure(api_key="test-key", capture_code=False)

                    args, kwargs = mock_set_sdk.call_args
                    sdk_instance = args[0]
                    config = sdk_instance.config

                    assert config.capture_code is False

    def test_configure_with_mode(self):
        """Test configure() with mode parameter."""
        with patch.dict(os.environ, {
            'VAQUERO_ENABLED': 'true',
            'VAQUERO_API_KEY': 'dummy-key-for-testing',
            'VAQUERO_PROJECT_ID': '',
            'VAQUERO_ENDPOINT': 'https://api.vaquero.com'
        }, clear=True):
            with patch('vaquero.config._resolve_or_create_project', return_value=None):
                with patch('vaquero.set_default_sdk') as mock_set_sdk:
                    sdk = configure(api_key="test-key", mode="production")

                    args, kwargs = mock_set_sdk.call_args
                    sdk_instance = args[0]
                    config = sdk_instance.config

                    assert config.mode == "production"
