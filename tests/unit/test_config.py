"""
Unit tests for configuration management.
"""

import pytest
from langgraph_rms.config import RMSConfig, initialize, get_config


class TestRMSConfig:
    """Test RMSConfig class."""

    def test_valid_configuration(self):
        """Test that valid configuration is accepted."""
        config = RMSConfig(
            product_name="test_product",
            agent_prompts={"Agent1": "You are Agent1"},
            rms_url="https://example.com/rms",
            api_key="test_api_key",
            llm_model="gpt-4",
            compatibility_threshold=0.7,
            request_timeout=10.0,
        )
        assert config.product_name == "test_product"
        assert config.agent_prompts == {"Agent1": "You are Agent1"}
        assert config.rms_url == "https://example.com/rms"
        assert config.api_key == "test_api_key"
        assert config.llm_model == "gpt-4"
        assert config.compatibility_threshold == 0.7
        assert config.request_timeout == 10.0

    def test_empty_product_name_raises_error(self):
        """Test that empty product_name raises ValueError."""
        with pytest.raises(ValueError, match="product_name must be a non-empty string"):
            RMSConfig(
                product_name="",
                agent_prompts={"Agent1": "You are Agent1"},
                rms_url="https://example.com/rms",
                api_key="test_api_key",
            )

    def test_whitespace_product_name_raises_error(self):
        """Test that whitespace-only product_name raises ValueError."""
        with pytest.raises(ValueError, match="product_name must be a non-empty string"):
            RMSConfig(
                product_name="   ",
                agent_prompts={"Agent1": "You are Agent1"},
                rms_url="https://example.com/rms",
                api_key="test_api_key",
            )

    def test_empty_agent_prompts_raises_error(self):
        """Test that empty agent_prompts raises ValueError."""
        with pytest.raises(ValueError, match="agent_prompts must contain at least one agent"):
            RMSConfig(
                product_name="test_product",
                agent_prompts={},
                rms_url="https://example.com/rms",
                api_key="test_api_key",
            )

    def test_invalid_agent_prompts_type_raises_error(self):
        """Test that non-dict agent_prompts raises ValueError."""
        with pytest.raises(ValueError, match="agent_prompts must be a dictionary"):
            RMSConfig(
                product_name="test_product",
                agent_prompts="not a dict",
                rms_url="https://example.com/rms",
                api_key="test_api_key",
            )

    def test_empty_rms_url_raises_error(self):
        """Test that empty rms_url raises ValueError."""
        with pytest.raises(ValueError, match="rms_url must be a non-empty string"):
            RMSConfig(
                product_name="test_product",
                agent_prompts={"Agent1": "You are Agent1"},
                rms_url="",
                api_key="test_api_key",
            )

    def test_invalid_rms_url_format_raises_error(self):
        """Test that invalid URL format raises ValueError."""
        with pytest.raises(ValueError, match="rms_url must be a valid URL"):
            RMSConfig(
                product_name="test_product",
                agent_prompts={"Agent1": "You are Agent1"},
                rms_url="not-a-valid-url",
                api_key="test_api_key",
            )

    def test_empty_api_key_raises_error(self):
        """Test that empty api_key raises ValueError."""
        with pytest.raises(ValueError, match="api_key must be a non-empty string"):
            RMSConfig(
                product_name="test_product",
                agent_prompts={"Agent1": "You are Agent1"},
                rms_url="https://example.com/rms",
                api_key="",
            )

    def test_threshold_below_range_raises_error(self):
        """Test that threshold < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="compatibility_threshold must be between 0.0 and 1.0"):
            RMSConfig(
                product_name="test_product",
                agent_prompts={"Agent1": "You are Agent1"},
                rms_url="https://example.com/rms",
                api_key="test_api_key",
                compatibility_threshold=-0.1,
            )

    def test_threshold_above_range_raises_error(self):
        """Test that threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="compatibility_threshold must be between 0.0 and 1.0"):
            RMSConfig(
                product_name="test_product",
                agent_prompts={"Agent1": "You are Agent1"},
                rms_url="https://example.com/rms",
                api_key="test_api_key",
                compatibility_threshold=1.1,
            )

    def test_threshold_at_boundaries(self):
        """Test that threshold at 0.0 and 1.0 is accepted."""
        config1 = RMSConfig(
            product_name="test_product",
            agent_prompts={"Agent1": "You are Agent1"},
            rms_url="https://example.com/rms",
            api_key="test_api_key",
            compatibility_threshold=0.0,
        )
        assert config1.compatibility_threshold == 0.0

        config2 = RMSConfig(
            product_name="test_product",
            agent_prompts={"Agent1": "You are Agent1"},
            rms_url="https://example.com/rms",
            api_key="test_api_key",
            compatibility_threshold=1.0,
        )
        assert config2.compatibility_threshold == 1.0

    def test_negative_timeout_raises_error(self):
        """Test that negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="request_timeout must be positive"):
            RMSConfig(
                product_name="test_product",
                agent_prompts={"Agent1": "You are Agent1"},
                rms_url="https://example.com/rms",
                api_key="test_api_key",
                request_timeout=-1.0,
            )

    def test_zero_timeout_raises_error(self):
        """Test that zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="request_timeout must be positive"):
            RMSConfig(
                product_name="test_product",
                agent_prompts={"Agent1": "You are Agent1"},
                rms_url="https://example.com/rms",
                api_key="test_api_key",
                request_timeout=0.0,
            )


class TestConfigurationSingleton:
    """Test global configuration singleton pattern."""

    def test_initialize_and_get_config(self):
        """Test that initialize() and get_config() work correctly."""
        config = RMSConfig(
            product_name="test_product",
            agent_prompts={"Agent1": "You are Agent1"},
            rms_url="https://example.com/rms",
            api_key="test_api_key",
        )
        initialize(config)
        retrieved_config = get_config()
        assert retrieved_config is config

    def test_get_config_before_initialize_raises_error(self):
        """Test that get_config() before initialize() raises RuntimeError."""
        # Reset global config
        import langgraph_rms.config as config_module
        config_module._config = None

        with pytest.raises(RuntimeError, match="Configuration not initialized"):
            get_config()
