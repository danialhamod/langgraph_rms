"""
Unit tests for rules appender module.
"""

import pytest
from unittest.mock import AsyncMock, patch

from langgraph_rms.appender import RulesAppender, append_rules


@pytest.mark.asyncio
async def test_append_rules_to_prompt_with_rules():
    """Test that rules are appended to base prompt when rules exist."""
    appender = RulesAppender(product_name="test_product")
    base_prompt = "You are a helpful assistant."
    agent_name = "TestAgent"
    
    mock_rules = ["Rule 1: Always be polite", "Rule 2: Verify information"]
    mock_formatted = "\n\n## Active Rules\n\nThe following rules must be followed:\n\n1. Rule 1: Always be polite\n2. Rule 2: Verify information\n"
    
    with patch("langgraph_rms.appender.get_rules_for_agent", new_callable=AsyncMock) as mock_get_rules:
        with patch("langgraph_rms.appender.format_rules_for_prompt") as mock_format:
            mock_get_rules.return_value = mock_rules
            mock_format.return_value = mock_formatted
            
            result = await appender.append_rules_to_prompt(base_prompt, agent_name)
            
            # Verify rules were fetched for the correct agent and product
            mock_get_rules.assert_called_once_with(
                agent_name=agent_name,
                product_name="test_product",
            )
            
            # Verify rules were formatted
            mock_format.assert_called_once_with(mock_rules, formatter=None)
            
            # Verify result contains both base prompt and formatted rules
            assert result == base_prompt + mock_formatted


@pytest.mark.asyncio
async def test_append_rules_to_prompt_with_no_rules():
    """Test that base prompt is returned unchanged when no rules exist."""
    appender = RulesAppender(product_name="test_product")
    base_prompt = "You are a helpful assistant."
    agent_name = "TestAgent"
    
    with patch("langgraph_rms.appender.get_rules_for_agent", new_callable=AsyncMock) as mock_get_rules:
        mock_get_rules.return_value = []
        
        result = await appender.append_rules_to_prompt(base_prompt, agent_name)
        
        # Verify base prompt is returned unchanged
        assert result == base_prompt


@pytest.mark.asyncio
async def test_append_rules_to_prompt_with_error():
    """Test that base prompt is returned unchanged when error occurs."""
    appender = RulesAppender(product_name="test_product")
    base_prompt = "You are a helpful assistant."
    agent_name = "TestAgent"
    
    with patch("langgraph_rms.appender.get_rules_for_agent", new_callable=AsyncMock) as mock_get_rules:
        mock_get_rules.side_effect = Exception("Network error")
        
        result = await appender.append_rules_to_prompt(base_prompt, agent_name)
        
        # Verify base prompt is returned unchanged on error
        assert result == base_prompt


@pytest.mark.asyncio
async def test_append_rules_to_prompt_uses_default_product():
    """Test that appender uses config default product when not specified."""
    appender = RulesAppender()  # No product_name specified
    base_prompt = "You are a helpful assistant."
    agent_name = "TestAgent"
    
    with patch("langgraph_rms.appender.get_rules_for_agent", new_callable=AsyncMock) as mock_get_rules:
        mock_get_rules.return_value = []
        
        await appender.append_rules_to_prompt(base_prompt, agent_name)
        
        # Verify get_rules_for_agent was called with None for product_name
        mock_get_rules.assert_called_once_with(
            agent_name=agent_name,
            product_name=None,
        )


@pytest.mark.asyncio
async def test_create_prompt_wrapper():
    """Test that create_prompt_wrapper returns a working async callable."""
    appender = RulesAppender(product_name="test_product")
    agent_name = "TestAgent"
    base_prompt = "You are a helpful assistant."
    
    mock_rules = ["Rule 1: Test rule"]
    mock_formatted = "\n\n## Active Rules\n\nThe following rules must be followed:\n\n1. Rule 1: Test rule\n"
    
    with patch("langgraph_rms.appender.get_rules_for_agent", new_callable=AsyncMock) as mock_get_rules:
        with patch("langgraph_rms.appender.format_rules_for_prompt") as mock_format:
            mock_get_rules.return_value = mock_rules
            mock_format.return_value = mock_formatted
            
            # Create wrapper
            wrapper = appender.create_prompt_wrapper(agent_name)
            
            # Verify wrapper is callable
            assert callable(wrapper)
            
            # Call wrapper
            result = await wrapper(base_prompt)
            
            # Verify result is correct
            assert result == base_prompt + mock_formatted
            
            # Verify correct agent was used
            mock_get_rules.assert_called_once_with(
                agent_name=agent_name,
                product_name="test_product",
            )


@pytest.mark.asyncio
async def test_append_rules_convenience_function():
    """Test the module-level append_rules convenience function."""
    base_prompt = "You are a helpful assistant."
    agent_name = "TestAgent"
    product_name = "test_product"
    
    mock_rules = ["Rule 1: Test rule"]
    mock_formatted = "\n\n## Active Rules\n\nThe following rules must be followed:\n\n1. Rule 1: Test rule\n"
    
    with patch("langgraph_rms.appender.get_rules_for_agent", new_callable=AsyncMock) as mock_get_rules:
        with patch("langgraph_rms.appender.format_rules_for_prompt") as mock_format:
            mock_get_rules.return_value = mock_rules
            mock_format.return_value = mock_formatted
            
            result = await append_rules(base_prompt, agent_name, product_name)
            
            # Verify result is correct
            assert result == base_prompt + mock_formatted
            
            # Verify correct parameters were used
            mock_get_rules.assert_called_once_with(
                agent_name=agent_name,
                product_name=product_name,
            )


@pytest.mark.asyncio
async def test_append_rules_convenience_function_without_product():
    """Test append_rules convenience function without product_name."""
    base_prompt = "You are a helpful assistant."
    agent_name = "TestAgent"
    
    with patch("langgraph_rms.appender.get_rules_for_agent", new_callable=AsyncMock) as mock_get_rules:
        mock_get_rules.return_value = []
        
        result = await append_rules(base_prompt, agent_name)
        
        # Verify result is base prompt (no rules)
        assert result == base_prompt
        
        # Verify product_name was None
        mock_get_rules.assert_called_once_with(
            agent_name=agent_name,
            product_name=None,
        )


@pytest.mark.asyncio
async def test_append_rules_with_custom_formatter():
    """Test that custom formatter is used when provided."""
    def custom_formatter(rules: list) -> str:
        return "\n=== CUSTOM RULES ===\n" + "\n".join(f"* {rule}" for rule in rules)
    
    appender = RulesAppender(product_name="test_product", formatter=custom_formatter)
    base_prompt = "You are a helpful assistant."
    agent_name = "TestAgent"
    
    mock_rules = ["Rule 1: Always be polite", "Rule 2: Verify information"]
    
    with patch("langgraph_rms.appender.get_rules_for_agent", new_callable=AsyncMock) as mock_get_rules:
        mock_get_rules.return_value = mock_rules
        
        result = await appender.append_rules_to_prompt(base_prompt, agent_name)
        
        # Verify custom formatter was used
        assert "=== CUSTOM RULES ===" in result
        assert "* Rule 1: Always be polite" in result
        assert "* Rule 2: Verify information" in result
        # Verify default format is NOT used
        assert "## Active Rules" not in result


@pytest.mark.asyncio
async def test_append_rules_convenience_function_with_custom_formatter():
    """Test append_rules convenience function with custom formatter."""
    def custom_formatter(rules: list) -> str:
        return "\nCUSTOM: " + " | ".join(rules)
    
    base_prompt = "You are a helpful assistant."
    agent_name = "TestAgent"
    product_name = "test_product"
    
    mock_rules = ["Rule 1", "Rule 2"]
    
    with patch("langgraph_rms.appender.get_rules_for_agent", new_callable=AsyncMock) as mock_get_rules:
        mock_get_rules.return_value = mock_rules
        
        result = await append_rules(
            base_prompt,
            agent_name,
            product_name,
            formatter=custom_formatter
        )
        
        # Verify custom formatter was used
        assert result == base_prompt + "\nCUSTOM: Rule 1 | Rule 2"


@pytest.mark.asyncio
async def test_create_prompt_wrapper_with_custom_formatter():
    """Test that create_prompt_wrapper uses custom formatter."""
    def custom_formatter(rules: list) -> str:
        return "\n[RULES] " + ", ".join(rules)
    
    appender = RulesAppender(product_name="test_product", formatter=custom_formatter)
    agent_name = "TestAgent"
    base_prompt = "You are a helpful assistant."
    
    mock_rules = ["Rule A", "Rule B"]
    
    with patch("langgraph_rms.appender.get_rules_for_agent", new_callable=AsyncMock) as mock_get_rules:
        mock_get_rules.return_value = mock_rules
        
        wrapper = appender.create_prompt_wrapper(agent_name)
        result = await wrapper(base_prompt)
        
        # Verify custom formatter was used
        assert result == base_prompt + "\n[RULES] Rule A, Rule B"
