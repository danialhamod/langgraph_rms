"""
Configuration management for RMS integration.

This module provides centralized configuration for the langgraph-rms-integration
package using a singleton pattern.
"""

from typing import Dict, Optional
from urllib.parse import urlparse


class RMSConfig:
    """Configuration for RMS integration.

    This class holds all configuration parameters needed for the RMS integration
    package. It validates parameters on initialization to ensure correct setup.

    Example:
        >>> from langgraph_rms import RMSConfig, initialize
        >>> config = RMSConfig(
        ...     product_name="my-chatbot",
        ...     agent_prompts={
        ...         "Agent1": "You are a helpful assistant",
        ...         "Agent2": "You are a medical advisor"
        ...     },
        ...     rms_url="https://rms.example.com",
        ...     api_key="secret-key-123",
        ...     llm_model="gpt-4",
        ...     compatibility_threshold=0.7,
        ...     request_timeout=10.0
        ... )
        >>> initialize(config)
    """

    def __init__(
        self,
        product_name: str,
        agent_prompts: Dict[str, str],
        rms_url: str,
        api_key: str,
        llm_model: str,
        compatibility_threshold: float = 0.7,
        request_timeout: float = 10.0,
    ):
        """
        Initialize RMS configuration.

        Args:
            product_name: Name of the product using this integration
            agent_prompts: Dictionary mapping agent names to system prompts
            rms_url: Base URL of the RMS service
            api_key: API key for authenticating with RMS
            llm_model: LLM model name for validation
            compatibility_threshold: Minimum score for rule application (default: 0.7)
            request_timeout: HTTP request timeout in seconds (default: 10.0)
        """
        self.product_name = product_name
        self.agent_prompts = agent_prompts
        self.rms_url = rms_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.compatibility_threshold = compatibility_threshold
        self.request_timeout = request_timeout

        # Validate configuration on initialization
        self.validate()

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate product_name
        if not self.product_name or not self.product_name.strip():
            raise ValueError("product_name must be a non-empty string")

        # Validate agent_prompts
        if not isinstance(self.agent_prompts, dict):
            raise ValueError("agent_prompts must be a dictionary")
        if not self.agent_prompts:
            raise ValueError("agent_prompts must contain at least one agent")

        # Validate rms_url
        if not self.rms_url or not self.rms_url.strip():
            raise ValueError("rms_url must be a non-empty string")

        try:
            parsed_url = urlparse(self.rms_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("rms_url must be a valid URL with scheme and netloc")
        except Exception as e:
            raise ValueError(f"rms_url is not a valid URL: {e}")

        # Validate api_key
        if not self.api_key or not self.api_key.strip():
            raise ValueError("api_key must be a non-empty string")

        # Validate compatibility_threshold
        if not isinstance(self.compatibility_threshold, (int, float)):
            raise ValueError("compatibility_threshold must be a number")
        if not 0.0 <= self.compatibility_threshold <= 1.0:
            raise ValueError("compatibility_threshold must be between 0.0 and 1.0")

        # Validate request_timeout
        if not isinstance(self.request_timeout, (int, float)):
            raise ValueError("request_timeout must be a number")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")


# Global configuration instance
_config: Optional[RMSConfig] = None


def initialize(config: RMSConfig) -> None:
    """
    Initialize the package with configuration.

    Args:
        config: RMSConfig instance with validated configuration

    Raises:
        ValueError: If configuration is invalid

    Example:
        >>> from langgraph_rms import RMSConfig, initialize
        >>> config = RMSConfig(
        ...     product_name="my-product",
        ...     agent_prompts={"Agent1": "You are a helpful assistant"},
        ...     rms_url="https://rms.example.com",
        ...     api_key="secret-key"
        ... )
        >>> initialize(config)
    """
    global _config
    _config = config


def get_config() -> RMSConfig:
    """
    Get the current configuration.

    Returns:
        Current RMSConfig instance

    Raises:
        RuntimeError: If configuration has not been initialized

    Example:
        >>> from langgraph_rms import get_config
        >>> config = get_config()
        >>> print(config.product_name)
        'my-product'
    """
    if _config is None:
        raise RuntimeError(
            "Configuration not initialized. Call initialize() with RMSConfig first."
        )
    return _config
