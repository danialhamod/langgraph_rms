"""
Unit tests for rule cache functionality.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from langgraph_rms.cache import (
    RuleCache,
    _cache,
    format_rules_for_prompt,
    get_rules_for_agent,
)
from langgraph_rms.models import (
    AgentRuleInfo,
    CachedRule,
    RuleValidation,
    ValidationMetadata,
)


@pytest.fixture
def sample_rules():
    """Sample rules with validation metadata."""
    return [
        CachedRule(
            id="rule1",
            product_name="test_product",
            rule_text="Original rule 1",
            max_length=100,
            risk_level="low",
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            latest_validation=RuleValidation(
                can_be_applied=True,
                max_compatibility_score=0.9,
                explanation="Compatible with agents",
                validation_metadata=ValidationMetadata(
                    applied_agents=[
                        AgentRuleInfo(
                            agent_name="Agent1",
                            role_consistency_score=0.9,
                            authority_expansion_score=0.9,
                            instruction_conflicts_score=0.9,
                            overall_compatibility_score=0.9,
                            analysis="Good fit",
                            concerns=[],
                            rule_to_apply="Apply rule 1 for Agent1",
                        ),
                        AgentRuleInfo(
                            agent_name="Agent2",
                            role_consistency_score=0.8,
                            authority_expansion_score=0.8,
                            instruction_conflicts_score=0.8,
                            overall_compatibility_score=0.8,
                            analysis="Good fit",
                            concerns=[],
                            rule_to_apply="Apply rule 1 for Agent2",
                        ),
                    ]
                ),
            ),
        ),
        CachedRule(
            id="rule2",
            product_name="test_product",
            rule_text="Original rule 2",
            max_length=100,
            risk_level="medium",
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            latest_validation=RuleValidation(
                can_be_applied=True,
                max_compatibility_score=0.85,
                explanation="Compatible with Agent2",
                validation_metadata=ValidationMetadata(
                    applied_agents=[
                        AgentRuleInfo(
                            agent_name="Agent2",
                            role_consistency_score=0.85,
                            authority_expansion_score=0.85,
                            instruction_conflicts_score=0.85,
                            overall_compatibility_score=0.85,
                            analysis="Good fit",
                            concerns=[],
                            rule_to_apply="Apply rule 2 for Agent2",
                        ),
                    ]
                ),
            ),
        ),
    ]


@pytest.fixture
def rule_without_validation():
    """Rule without validation metadata for backward compatibility testing."""
    return CachedRule(
        id="rule3",
        product_name="test_product",
        rule_text="Rule without validation",
        max_length=100,
        risk_level="low",
        status="active",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        latest_validation=None,
    )


class TestRuleCache:
    """Tests for RuleCache class."""

    @pytest.mark.asyncio
    async def test_get_active_rules_empty(self):
        """Test getting rules from empty cache."""
        cache = RuleCache()
        rules = await cache.get_active_rules("nonexistent_product")
        assert rules == []

    @pytest.mark.asyncio
    async def test_refresh_rules(self, sample_rules):
        """Test refreshing cache with new rules."""
        cache = RuleCache()
        
        await cache.refresh_rules("test_product", sample_rules)
        
        retrieved_rules = await cache.get_active_rules("test_product")
        assert len(retrieved_rules) == 2
        assert retrieved_rules[0].id == "rule1"
        assert retrieved_rules[1].id == "rule2"

    @pytest.mark.asyncio
    async def test_refresh_updates_timestamp(self, sample_rules):
        """Test that refresh updates last refresh timestamp."""
        cache = RuleCache()
        
        before = datetime.now()
        await cache.refresh_rules("test_product", sample_rules)
        after = datetime.now()
        
        last_refresh = cache.get_last_refresh("test_product")
        assert last_refresh is not None
        assert before <= last_refresh <= after

    @pytest.mark.asyncio
    async def test_get_last_refresh_none(self):
        """Test getting last refresh for product that was never refreshed."""
        cache = RuleCache()
        last_refresh = cache.get_last_refresh("nonexistent_product")
        assert last_refresh is None

    @pytest.mark.asyncio
    async def test_cache_product_isolation(self, sample_rules):
        """Test that cache stores rules separately by product."""
        cache = RuleCache()
        
        await cache.refresh_rules("product1", sample_rules[:1])
        await cache.refresh_rules("product2", sample_rules[1:])
        
        product1_rules = await cache.get_active_rules("product1")
        product2_rules = await cache.get_active_rules("product2")
        
        assert len(product1_rules) == 1
        assert len(product2_rules) == 1
        assert product1_rules[0].id == "rule1"
        assert product2_rules[0].id == "rule2"

    @pytest.mark.asyncio
    async def test_fetch_from_rms_success(self, sample_rules, test_config):
        """Test successful fetch from RMS."""
        cache = RuleCache()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "rules": [rule.model_dump() for rule in sample_rules]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            
            await cache.fetch_from_rms("test_product")
        
        rules = await cache.get_active_rules("test_product")
        assert len(rules) == 2

    @pytest.mark.asyncio
    async def test_fetch_from_rms_timeout_preserves_cache(self, sample_rules, test_config):
        """Test that timeout during fetch preserves existing cache."""
        cache = RuleCache()
        
        # Pre-populate cache
        await cache.refresh_rules("test_product", sample_rules)
        
        # Simulate timeout
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )
            
            await cache.fetch_from_rms("test_product")
        
        # Cache should still have original rules
        rules = await cache.get_active_rules("test_product")
        assert len(rules) == 2
        assert rules[0].id == "rule1"

    @pytest.mark.asyncio
    async def test_fetch_from_rms_http_error_preserves_cache(
        self, sample_rules, test_config
    ):
        """Test that HTTP error during fetch preserves existing cache."""
        cache = RuleCache()
        
        # Pre-populate cache
        await cache.refresh_rules("test_product", sample_rules)
        
        # Simulate HTTP error
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPError("Server error")
            )
            
            await cache.fetch_from_rms("test_product")
        
        # Cache should still have original rules
        rules = await cache.get_active_rules("test_product")
        assert len(rules) == 2

    @pytest.mark.asyncio
    async def test_thread_safety(self, sample_rules):
        """Test that cache operations are thread-safe."""
        cache = RuleCache()
        
        # Perform concurrent operations
        tasks = [
            cache.refresh_rules("test_product", sample_rules),
            cache.get_active_rules("test_product"),
            cache.get_active_rules("test_product"),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Should complete without errors
        assert results is not None


class TestGetRulesForAgent:
    """Tests for get_rules_for_agent function."""

    @pytest.mark.asyncio
    async def test_get_all_rules_no_agent_specified(self, sample_rules, test_config):
        """Test getting all rules when no agent is specified."""
        await _cache.refresh_rules("test_product", sample_rules)
        
        rules = await get_rules_for_agent(agent_name=None)
        
        assert len(rules) == 2
        assert "Apply rule 1 for Agent1" in rules[0]
        assert "Apply rule 2 for Agent2" in rules[1]

    @pytest.mark.asyncio
    async def test_get_rules_for_specific_agent(self, sample_rules, test_config):
        """Test getting rules filtered for specific agent."""
        await _cache.refresh_rules("test_product", sample_rules)
        
        rules = await get_rules_for_agent(agent_name="Agent1")
        
        assert len(rules) == 1
        assert rules[0] == "Apply rule 1 for Agent1"

    @pytest.mark.asyncio
    async def test_get_rules_for_agent_with_multiple_rules(
        self, sample_rules, test_config
    ):
        """Test getting rules for agent that has multiple rules."""
        await _cache.refresh_rules("test_product", sample_rules)
        
        rules = await get_rules_for_agent(agent_name="Agent2")
        
        assert len(rules) == 2
        assert "Apply rule 1 for Agent2" in rules
        assert "Apply rule 2 for Agent2" in rules

    @pytest.mark.asyncio
    async def test_backward_compatibility_no_validation(
        self, rule_without_validation, test_config
    ):
        """Test that rules without validation metadata are included for all agents."""
        await _cache.refresh_rules("test_product", [rule_without_validation])
        
        rules = await get_rules_for_agent(agent_name="AnyAgent")
        
        assert len(rules) == 1
        assert rules[0] == "Rule without validation"

    @pytest.mark.asyncio
    async def test_custom_product_name(self, sample_rules, test_config):
        """Test getting rules with custom product name."""
        await _cache.refresh_rules("custom_product", sample_rules)
        
        rules = await get_rules_for_agent(
            agent_name="Agent1", product_name="custom_product"
        )
        
        assert len(rules) == 1
        assert rules[0] == "Apply rule 1 for Agent1"

    @pytest.mark.asyncio
    async def test_rule_with_empty_applied_agents(self, test_config):
        """Test handling rule with validation metadata but empty applied_agents list."""
        rule_with_empty_agents = CachedRule(
            id="rule_empty",
            product_name="test_product",
            rule_text="Rule with empty agents",
            max_length=100,
            risk_level="low",
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            latest_validation=RuleValidation(
                can_be_applied=False,
                max_compatibility_score=0.0,
                explanation="Not compatible with any agent",
                validation_metadata=ValidationMetadata(applied_agents=[]),
            ),
        )
        
        await _cache.refresh_rules("test_product", [rule_with_empty_agents])
        
        # When agent_name is None, should fall back to rule_text
        rules_no_agent = await get_rules_for_agent(agent_name=None)
        assert len(rules_no_agent) == 1
        assert rules_no_agent[0] == "Rule with empty agents"
        
        # When agent_name is specified, should not include this rule
        rules_with_agent = await get_rules_for_agent(agent_name="Agent1")
        assert len(rules_with_agent) == 0


class TestFormatRulesForPrompt:
    """Tests for format_rules_for_prompt function."""

    def test_format_empty_rules(self):
        """Test formatting empty rule list."""
        formatted = format_rules_for_prompt([])
        assert formatted == ""

    def test_format_single_rule(self):
        """Test formatting single rule."""
        rules = ["Always be polite"]
        formatted = format_rules_for_prompt(rules)
        
        assert "## Active Rules" in formatted
        assert "1. Always be polite" in formatted

    def test_format_multiple_rules(self):
        """Test formatting multiple rules."""
        rules = ["Always be polite", "Never share personal data", "Verify information"]
        formatted = format_rules_for_prompt(rules)
        
        assert "## Active Rules" in formatted
        assert "1. Always be polite" in formatted
        assert "2. Never share personal data" in formatted
        assert "3. Verify information" in formatted

    def test_format_includes_header(self):
        """Test that formatted output includes proper header."""
        rules = ["Test rule"]
        formatted = format_rules_for_prompt(rules)
        
        assert "## Active Rules" in formatted
        assert "The following rules must be followed:" in formatted

    def test_format_with_custom_formatter(self):
        """Test formatting with custom formatter function."""
        rules = ["Rule 1", "Rule 2", "Rule 3"]
        
        def custom_formatter(rule_list: list) -> str:
            return "CUSTOM: " + " | ".join(rule_list)
        
        formatted = format_rules_for_prompt(rules, formatter=custom_formatter)
        
        assert formatted == "CUSTOM: Rule 1 | Rule 2 | Rule 3"
        assert "## Active Rules" not in formatted  # Should not use default format

    def test_format_with_custom_formatter_empty_rules(self):
        """Test that custom formatter is not called for empty rules."""
        def custom_formatter(rule_list: list) -> str:
            return "CUSTOM: " + " | ".join(rule_list)
        
        formatted = format_rules_for_prompt([], formatter=custom_formatter)
        
        # Should return empty string without calling custom formatter
        assert formatted == ""

    def test_format_with_custom_formatter_complex(self):
        """Test formatting with more complex custom formatter."""
        rules = ["Always be polite", "Never share personal data"]
        
        def markdown_formatter(rule_list: list) -> str:
            result = "\n### Important Rules\n\n"
            for rule in rule_list:
                result += f"- **{rule}**\n"
            return result
        
        formatted = format_rules_for_prompt(rules, formatter=markdown_formatter)
        
        assert "### Important Rules" in formatted
        assert "- **Always be polite**" in formatted
        assert "- **Never share personal data**" in formatted
