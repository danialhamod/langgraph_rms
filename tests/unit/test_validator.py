"""
Unit tests for rule validator.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from langgraph_rms.validator import RuleValidator
from langgraph_rms.models import RuleValidation, CachedRule, RuleValidation as RV
from langgraph_rms.prompts import PromptTemplate
from langgraph_rms.config import RMSConfig, initialize
from datetime import datetime


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RMSConfig(
        product_name="test_product",
        agent_prompts={
            "TestAgent": "You are a test agent.",
            "AnotherAgent": "You are another test agent.",
        },
        rms_url="https://test-rms.example.com",
        api_key="test-api-key",
        llm_model="gpt-4",
        compatibility_threshold=0.7,
        request_timeout=10.0,
    )
    initialize(config)
    return config


@pytest.fixture
def mock_llm():
    """Create mock LLM client."""
    llm = Mock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def sample_agent_prompts():
    """Sample agent prompts for testing."""
    return {
        "MedicalAdvisorAgent": "You are a medical advisor. Provide medical advice based on symptoms.",
        "RiskAssessmentAgent": "You are a risk assessment agent. Evaluate medical risks.",
    }


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for validation."""
    return Mock(
        content="""```json
{
  "applied_agents": [
    {
      "agent_name": "MedicalAdvisorAgent",
      "role_consistency_score": 0.9,
      "authority_expansion_score": 0.8,
      "instruction_conflicts_score": 0.85,
      "overall_compatibility_score": 0.85,
      "analysis": "This rule aligns well with the medical advisor role.",
      "concerns": [],
      "rule_to_apply": "Always verify patient symptoms before providing advice."
    }
  ],
  "system_summary": "Rule is compatible with MedicalAdvisorAgent."
}
```"""
    )


class TestRuleValidator:
    """Tests for RuleValidator class."""

    def test_init_with_default_template(self, mock_llm):
        """Test validator initialization with default template."""
        validator = RuleValidator(mock_llm)
        assert validator.llm_client == mock_llm
        assert validator.prompt_template is not None

    def test_init_with_custom_template(self, mock_llm):
        """Test validator initialization with custom template."""
        custom_template = PromptTemplate("Custom template: {rule_text}")
        validator = RuleValidator(mock_llm, prompt_template=custom_template)
        assert validator.prompt_template == custom_template

    @pytest.mark.asyncio
    async def test_validate_rule_success(
        self, test_config, mock_llm, sample_agent_prompts, sample_llm_response
    ):
        """Test successful rule validation."""
        mock_llm.ainvoke.return_value = sample_llm_response

        validator = RuleValidator(mock_llm)
        result = await validator.validate_rule(
            rule_text="Always verify patient symptoms.",
            agent_prompts=sample_agent_prompts,
        )

        assert isinstance(result, RuleValidation)
        assert result.can_be_applied is True
        assert abs(result.max_compatibility_score - 0.85) < 0.001
        assert len(result.validation_metadata.applied_agents) == 1
        assert (
            result.validation_metadata.applied_agents[0].agent_name
            == "MedicalAdvisorAgent"
        )

    @pytest.mark.asyncio
    async def test_validate_rule_calculates_overall_score(
        self, test_config, mock_llm, sample_agent_prompts
    ):
        """Test that overall score is calculated as average of component scores."""
        mock_llm.ainvoke.return_value = Mock(
            content="""```json
{
  "applied_agents": [
    {
      "agent_name": "MedicalAdvisorAgent",
      "role_consistency_score": 0.9,
      "authority_expansion_score": 0.8,
      "instruction_conflicts_score": 0.7,
      "analysis": "Test",
      "concerns": [],
      "rule_to_apply": "Test rule"
    }
  ],
  "system_summary": "Test"
}
```"""
        )

        validator = RuleValidator(mock_llm)
        result = await validator.validate_rule(
            rule_text="Test rule", agent_prompts=sample_agent_prompts
        )

        # Overall score should be (0.9 + 0.8 + 0.7) / 3 = 0.8
        agent = result.validation_metadata.applied_agents[0]
        expected_score = (0.9 + 0.8 + 0.7) / 3.0
        assert abs(agent.overall_compatibility_score - expected_score) < 0.001

    @pytest.mark.asyncio
    async def test_validate_rule_filters_by_threshold(
        self, test_config, mock_llm, sample_agent_prompts
    ):
        """Test that agents below threshold are filtered out."""
        mock_llm.ainvoke.return_value = Mock(
            content="""```json
{
  "applied_agents": [
    {
      "agent_name": "MedicalAdvisorAgent",
      "role_consistency_score": 0.9,
      "authority_expansion_score": 0.8,
      "instruction_conflicts_score": 0.85,
      "analysis": "High score",
      "concerns": [],
      "rule_to_apply": "High score rule"
    },
    {
      "agent_name": "RiskAssessmentAgent",
      "role_consistency_score": 0.5,
      "authority_expansion_score": 0.4,
      "instruction_conflicts_score": 0.3,
      "analysis": "Low score",
      "concerns": ["Not compatible"],
      "rule_to_apply": "Low score rule"
    }
  ],
  "system_summary": "Mixed compatibility"
}
```"""
        )

        validator = RuleValidator(mock_llm)
        result = await validator.validate_rule(
            rule_text="Test rule", agent_prompts=sample_agent_prompts
        )

        # Only MedicalAdvisorAgent should pass (avg 0.85 >= 0.7)
        # RiskAssessmentAgent should be filtered (avg 0.4 < 0.7)
        assert len(result.validation_metadata.applied_agents) == 1
        assert (
            result.validation_metadata.applied_agents[0].agent_name
            == "MedicalAdvisorAgent"
        )

    @pytest.mark.asyncio
    async def test_validate_rule_with_existing_rules(
        self, test_config, mock_llm, sample_agent_prompts, sample_llm_response
    ):
        """Test validation with existing rules provided."""
        existing_rule = CachedRule(
            id="rule-1",
            product_name="test_product",
            rule_text="Existing rule text",
            max_length=100,
            risk_level="low",
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            latest_validation=None,
        )

        mock_llm.ainvoke.return_value = sample_llm_response

        validator = RuleValidator(mock_llm)
        result = await validator.validate_rule(
            rule_text="New rule",
            agent_prompts=sample_agent_prompts,
            existing_rules=[existing_rule],
        )

        # Verify the prompt was built with existing rules
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert "EXISTING ACTIVE RULES" in call_args
        assert "rule-1" in call_args

    @pytest.mark.asyncio
    async def test_validate_rule_with_scoring_callback(
        self, test_config, mock_llm, sample_agent_prompts
    ):
        """Test validation with custom scoring callback that modifies results."""
        # Mock response with two agents that pass threshold
        mock_llm.ainvoke.return_value = Mock(
            content="""```json
{
  "applied_agents": [
    {
      "agent_name": "MedicalAdvisorAgent",
      "role_consistency_score": 0.9,
      "authority_expansion_score": 0.8,
      "instruction_conflicts_score": 0.85,
      "analysis": "High score agent",
      "concerns": [],
      "rule_to_apply": "Rule for medical advisor"
    },
    {
      "agent_name": "RiskAssessmentAgent",
      "role_consistency_score": 0.8,
      "authority_expansion_score": 0.75,
      "instruction_conflicts_score": 0.8,
      "analysis": "Medium score agent",
      "concerns": [],
      "rule_to_apply": "Rule for risk assessment"
    }
  ],
  "system_summary": "Both agents compatible"
}
```"""
        )

        # Track if callback was invoked
        callback_invoked = False
        original_agents = None

        def custom_scoring(agents):
            nonlocal callback_invoked, original_agents
            callback_invoked = True
            original_agents = agents
            # Filter to only keep the first agent
            return [agents[0]] if agents else []

        validator = RuleValidator(mock_llm)
        result = await validator.validate_rule(
            rule_text="Test rule",
            agent_prompts=sample_agent_prompts,
            scoring_callback=custom_scoring,
        )

        # Verify callback was invoked
        assert callback_invoked is True
        assert original_agents is not None
        assert len(original_agents) == 2  # Both agents passed threshold

        # Verify callback modified the result
        assert result.can_be_applied is True
        assert len(result.validation_metadata.applied_agents) == 1
        assert (
            result.validation_metadata.applied_agents[0].agent_name
            == "MedicalAdvisorAgent"
        )

    @pytest.mark.asyncio
    async def test_validate_rule_scoring_callback_can_filter_all_agents(
        self, test_config, mock_llm, sample_agent_prompts, sample_llm_response
    ):
        """Test that scoring callback can filter out all agents."""
        mock_llm.ainvoke.return_value = sample_llm_response

        # Callback that filters out all agents
        def filter_all(agents):
            return []

        validator = RuleValidator(mock_llm)
        result = await validator.validate_rule(
            rule_text="Test rule",
            agent_prompts=sample_agent_prompts,
            scoring_callback=filter_all,
        )

        # Result should indicate no agents can apply
        assert result.can_be_applied is False
        assert result.max_compatibility_score == 0.0
        assert len(result.validation_metadata.applied_agents) == 0

    @pytest.mark.asyncio
    async def test_validate_rule_llm_error_returns_safe_default(
        self, test_config, mock_llm, sample_agent_prompts
    ):
        """Test that LLM errors return safe default response."""
        mock_llm.ainvoke.side_effect = Exception("LLM API error")

        validator = RuleValidator(mock_llm)
        result = await validator.validate_rule(
            rule_text="Test rule", agent_prompts=sample_agent_prompts
        )

        # Should return safe default
        assert isinstance(result, RuleValidation)
        assert result.can_be_applied is False
        assert result.max_compatibility_score == 0.0
        assert "Validation failed" in result.explanation
        assert len(result.validation_metadata.applied_agents) == 0

    @pytest.mark.asyncio
    async def test_validate_rule_invalid_json_returns_safe_default(
        self, test_config, mock_llm, sample_agent_prompts
    ):
        """Test that invalid JSON response returns safe default."""
        mock_llm.ainvoke.return_value = Mock(content="Invalid JSON response")

        validator = RuleValidator(mock_llm)
        result = await validator.validate_rule(
            rule_text="Test rule", agent_prompts=sample_agent_prompts
        )

        assert result.can_be_applied is False
        assert result.max_compatibility_score == 0.0
        assert "Validation failed" in result.explanation

    @pytest.mark.asyncio
    async def test_validate_rule_no_compatible_agents(
        self, test_config, mock_llm, sample_agent_prompts
    ):
        """Test validation when no agents meet the threshold."""
        mock_llm.ainvoke.return_value = Mock(
            content="""```json
{
  "applied_agents": [],
  "system_summary": "No compatible agents found."
}
```"""
        )

        validator = RuleValidator(mock_llm)
        result = await validator.validate_rule(
            rule_text="Incompatible rule", agent_prompts=sample_agent_prompts
        )

        assert result.can_be_applied is False
        assert result.max_compatibility_score == 0.0
        assert len(result.validation_metadata.applied_agents) == 0

    def test_build_validation_prompt(
        self, test_config, mock_llm, sample_agent_prompts
    ):
        """Test prompt building."""
        validator = RuleValidator(mock_llm)
        prompt = validator._build_validation_prompt(
            rule_text="Test rule", agent_prompts=sample_agent_prompts
        )

        assert "Test rule" in prompt
        assert "MedicalAdvisorAgent" in prompt
        assert "RiskAssessmentAgent" in prompt
        assert "You are a medical advisor" in prompt

    def test_build_validation_prompt_with_existing_rules(
        self, test_config, mock_llm, sample_agent_prompts
    ):
        """Test prompt building with existing rules."""
        existing_rule = CachedRule(
            id="rule-1",
            product_name="test_product",
            rule_text="Existing rule",
            max_length=100,
            risk_level="low",
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            latest_validation=None,
        )

        validator = RuleValidator(mock_llm)
        prompt = validator._build_validation_prompt(
            rule_text="New rule",
            agent_prompts=sample_agent_prompts,
            existing_rules=[existing_rule],
        )

        assert "EXISTING ACTIVE RULES" in prompt
        assert "rule-1" in prompt
        assert "Existing rule" in prompt

    def test_parse_validation_response_success(self, mock_llm):
        """Test parsing valid LLM response."""
        validator = RuleValidator(mock_llm)
        response = """```json
{
  "applied_agents": [
    {
      "agent_name": "TestAgent",
      "role_consistency_score": 0.9,
      "authority_expansion_score": 0.8,
      "instruction_conflicts_score": 0.85,
      "analysis": "Good fit",
      "concerns": [],
      "rule_to_apply": "Test rule"
    }
  ],
  "system_summary": "Compatible"
}
```"""

        parsed = validator._parse_validation_response(response)
        assert "applied_agents" in parsed
        assert len(parsed["applied_agents"]) == 1
        assert parsed["applied_agents"][0]["agent_name"] == "TestAgent"

    def test_parse_validation_response_missing_field(self, mock_llm):
        """Test parsing response with missing required field."""
        validator = RuleValidator(mock_llm)
        response = """```json
{
  "applied_agents": [
    {
      "agent_name": "TestAgent",
      "role_consistency_score": 0.9
    }
  ]
}
```"""

        with pytest.raises(ValueError, match="missing required field"):
            validator._parse_validation_response(response)

    def test_parse_validation_response_invalid_score(self, mock_llm):
        """Test parsing response with invalid score value."""
        validator = RuleValidator(mock_llm)
        response = """```json
{
  "applied_agents": [
    {
      "agent_name": "TestAgent",
      "role_consistency_score": 1.5,
      "authority_expansion_score": 0.8,
      "instruction_conflicts_score": 0.85,
      "rule_to_apply": "Test"
    }
  ]
}
```"""

        with pytest.raises(ValueError, match="must be a number between 0.0 and 1.0"):
            validator._parse_validation_response(response)

    def test_parse_validation_response_missing_applied_agents(self, mock_llm):
        """Test parsing response without applied_agents field."""
        validator = RuleValidator(mock_llm)
        response = """```json
{
  "system_summary": "No agents"
}
```"""

        with pytest.raises(ValueError, match="missing 'applied_agents' field"):
            validator._parse_validation_response(response)
