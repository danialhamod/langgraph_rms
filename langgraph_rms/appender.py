"""
Rules appender for automatic prompt enhancement.

This module provides functionality to automatically append active rules
to agent system prompts, with error handling and flexible integration options.
"""

import logging
from typing import Awaitable, Callable, Optional

from langgraph_rms.cache import format_rules_for_prompt, get_rules_for_agent
from langgraph_rms.config import get_config


logger = logging.getLogger(__name__)


class RulesAppender:
    """Handles automatic rule injection into agent prompts.

    This class provides methods to fetch active rules for an agent and
    append them to the agent's base system prompt automatically.

    Example:
        >>> from langgraph_rms import RulesAppender
        >>> appender = RulesAppender(product_name="my-product")
        >>> enhanced = await appender.append_rules_to_prompt(
        ...     base_prompt="You are a helpful assistant",
        ...     agent_name="Agent1"
        ... )
        >>> # Or create a wrapper for repeated use
        >>> wrapper = appender.create_prompt_wrapper("Agent1")
        >>> enhanced = await wrapper("You are a helpful assistant")
    """

    def __init__(
        self,
        product_name: Optional[str] = None,
        formatter: Optional[Callable[[list], str]] = None,
    ):
        """
        Initialize rules appender.

        Args:
            product_name: Optional product name (uses config default if None)
            formatter: Optional custom formatter function for rules
        """
        self.product_name = product_name
        self.formatter = formatter

    async def append_rules_to_prompt(
        self,
        base_prompt: str,
        agent_name: str,
    ) -> str:
        """
        Append active rules to an agent's system prompt.

        Args:
            base_prompt: The agent's base system prompt
            agent_name: Name of the agent

        Returns:
            Enhanced prompt with rules appended

        Example:
            >>> from langgraph_rms import RulesAppender
            >>> appender = RulesAppender()
            >>> enhanced = await appender.append_rules_to_prompt(
            ...     base_prompt="You are a helpful assistant",
            ...     agent_name="Agent1"
            ... )
            >>> print(enhanced)
            You are a helpful assistant
            
            ## Active Rules
            
            The following rules must be followed:
            
            1. Always be polite
        """
        try:
            # Fetch rules for the specific agent
            rules = await get_rules_for_agent(
                agent_name=agent_name,
                product_name=self.product_name,
            )

            # If no rules, return base prompt unchanged
            if not rules:
                return base_prompt

            # Format rules for prompt injection (with custom formatter if provided)
            formatted_rules = format_rules_for_prompt(rules, formatter=self.formatter)

            # Concatenate base prompt with formatted rules
            return base_prompt + formatted_rules

        except Exception as e:
            # Log error and return base prompt unchanged
            logger.warning(
                f"Failed to append rules for agent '{agent_name}': {e}. "
                f"Returning base prompt unchanged."
            )
            return base_prompt

    def create_prompt_wrapper(
        self,
        agent_name: str,
    ) -> Callable[[str], Awaitable[str]]:
        """
        Create a wrapper function for automatic rule injection.

        Args:
            agent_name: Name of the agent

        Returns:
            Async function that takes base_prompt and returns enhanced prompt
        """
        async def wrapper(base_prompt: str) -> str:
            return await self.append_rules_to_prompt(base_prompt, agent_name)

        return wrapper


async def append_rules(
    base_prompt: str,
    agent_name: str,
    product_name: Optional[str] = None,
    formatter: Optional[Callable[[list], str]] = None,
) -> str:
    """
    Convenience function to append rules to a prompt.

    Args:
        base_prompt: The agent's base system prompt
        agent_name: Name of the agent
        product_name: Optional product name (uses config default if None)
        formatter: Optional custom formatter function for rules

    Returns:
        Enhanced prompt with rules appended

    Example:
        >>> from langgraph_rms import append_rules
        >>> enhanced = await append_rules(
        ...     base_prompt="You are a helpful assistant",
        ...     agent_name="Agent1"
        ... )
        >>> print(enhanced)
        You are a helpful assistant
        
        ## Active Rules
        
        The following rules must be followed:
        
        1. Always be polite
    """
    appender = RulesAppender(product_name=product_name, formatter=formatter)
    return await appender.append_rules_to_prompt(base_prompt, agent_name)
