"""
Rule caching logic for RMS integration.

This module provides thread-safe in-memory caching of active rules with
webhook-based refresh and manual fetch capabilities.
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional

import httpx

from langgraph_rms.config import get_config
from langgraph_rms.models import CachedRule


logger = logging.getLogger(__name__)


class RuleCache:
    """Thread-safe in-memory cache for active rules.

    This class provides caching functionality for rules with support for
    webhook-based refresh and manual fetching from RMS.

    Example:
        >>> from langgraph_rms.cache import _cache
        >>> # Refresh cache with new rules
        >>> await _cache.refresh_rules("my-product", rules_list)
        >>> # Get cached rules
        >>> rules = await _cache.get_active_rules("my-product")
        >>> # Check last refresh time
        >>> timestamp = _cache.get_last_refresh("my-product")
    """

    def __init__(self):
        """Initialize empty cache with lock for thread safety."""
        self._cache: Dict[str, List[CachedRule]] = {}
        self._last_refresh: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def get_active_rules(self, product_name: str) -> List[CachedRule]:
        """
        Get active rules for a product.

        Args:
            product_name: Name of the product

        Returns:
            List of cached rules for the product
        """
        async with self._lock:
            return self._cache.get(product_name, [])

    async def refresh_rules(
        self,
        product_name: str,
        rules: List[CachedRule],
    ) -> None:
        """
        Update cached rules for a product.

        Args:
            product_name: Name of the product
            rules: New list of active rules
        """
        async with self._lock:
            self._cache[product_name] = rules
            self._last_refresh[product_name] = datetime.now()
            logger.info(
                f"Cache refreshed for product '{product_name}' with {len(rules)} rules"
            )

    async def fetch_from_rms(self, product_name: str) -> None:
        """
        Manually fetch rules from RMS and update cache.

        Args:
            product_name: Name of the product

        Raises:
            No exceptions raised - errors are logged and cache is preserved
        """
        config = get_config()
        url = f"{config.rms_url}/rules/active/{product_name}"
        headers = {"Authorization": f"Bearer {config.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=config.request_timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                rules = [CachedRule(**rule) for rule in data]
                
                await self.refresh_rules(product_name, rules)
                logger.info(
                    f"Successfully fetched {len(rules)} rules from RMS for '{product_name}'"
                )
        except httpx.TimeoutException as e:
            logger.warning(
                f"RMS fetch timeout for product '{product_name}': {e}. Using cached data."
            )
        except httpx.HTTPError as e:
            logger.warning(
                f"RMS fetch HTTP error for product '{product_name}': {e}. Using cached data."
            )
        except Exception as e:
            logger.warning(
                f"RMS fetch failed for product '{product_name}': {e}. Using cached data."
            )

    def get_last_refresh(self, product_name: str) -> Optional[datetime]:
        """
        Get timestamp of last cache refresh for product.

        Args:
            product_name: Name of the product

        Returns:
            Timestamp of last refresh, or None if never refreshed
        """
        return self._last_refresh.get(product_name)


# Global cache instance
_cache = RuleCache()


async def get_rules_for_agent(
    agent_name: Optional[str] = None,
    product_name: Optional[str] = None,
) -> List[str]:
    """
    Get active rule texts for agent injection.

    Args:
        agent_name: Optional agent name to filter rules
        product_name: Optional product name (uses config default if None)

    Returns:
        List of rule texts ready for prompt injection

    Example:
        >>> from langgraph_rms import get_rules_for_agent
        >>> rules = await get_rules_for_agent(agent_name="Agent1")
        >>> print(rules)
        ['Always be polite', 'Provide concise answers']
    """
    config = get_config()
    product = product_name or config.product_name
    print('product', product)
    
    rules = await _cache.get_active_rules(product)
    
    # If no agent specified, return all rules
    if agent_name is None:
        return [
            rule.latest_validation.validation_metadata.applied_agents[0].rule_to_apply
            if rule.latest_validation and rule.latest_validation.validation_metadata.applied_agents
            else rule.rule_text
            for rule in rules
        ]
    
    # Filter rules for specific agent
    agent_rules = []
    for rule in rules:
        # Handle rules without validation metadata (backward compatibility)
        if rule.latest_validation is None:
            agent_rules.append(rule.rule_text)
            continue
        
        # Check if rule applies to this agent
        for agent_info in rule.latest_validation.validation_metadata.applied_agents:
            if agent_info.agent_name == agent_name:
                agent_rules.append(agent_info.rule_to_apply)
                break
    
    return agent_rules


def format_rules_for_prompt(
    rules: List[str],
    formatter: Optional[Callable[[List[str]], str]] = None,
) -> str:
    """
    Format rules for insertion into agent prompts.

    Args:
        rules: List of rule texts
        formatter: Optional custom formatter function that takes a list of rules
                  and returns a formatted string. If None, uses default formatting.

    Returns:
        Formatted string ready to append to agent prompt

    Example:
        >>> from langgraph_rms import format_rules_for_prompt
        >>> rules = ['Be polite', 'Be concise']
        >>> formatted = format_rules_for_prompt(rules)
        >>> print(formatted)
        
        ## Active Rules
        
        The following rules must be followed:
        
        1. Be polite
        2. Be concise
    """
    if not rules:
        return ""

    # Use custom formatter if provided
    if formatter is not None:
        return formatter(rules)

    # Default formatting
    formatted = "\n\n## Additional Rules\n\n"
    formatted += "The following rules must be followed:\n\n"

    for i, rule in enumerate(rules, 1):
        formatted += f"{i}. {rule}\n"

    return formatted


async def fetch_rules(product_name: Optional[str] = None) -> None:
    """
    Fetch active rules from RMS and populate cache.
    
    This should be called on application startup to ensure the cache
    is populated with the latest rules from RMS.
    
    Args:
        product_name: Optional product name (uses config default if None)
        
    Example:
        >>> from langgraph_rms import initialize, RMSConfig, fetch_rules
        >>> config = RMSConfig(...)
        >>> initialize(config)
        >>> await fetch_rules()  # Fetch rules on startup
    """
    config = get_config()
    product = product_name or config.product_name
    await _cache.fetch_from_rms(product)
