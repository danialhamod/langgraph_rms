"""
Unit tests for validation prompt templates.
"""
import pytest
from langgraph_rms.prompts import (
    PromptTemplate,
    DEFAULT_VALIDATION_TEMPLATE,
    get_default_template,
    create_custom_template,
)


class TestPromptTemplate:
    """Tests for PromptTemplate class."""
    
    def test_init(self):
        """Test PromptTemplate initialization."""
        template = "Test template with {rule_text}"
        pt = PromptTemplate(template)
        assert pt.template == template
    
    def test_render_basic(self):
        """Test basic template rendering with rule text and agents."""
        template = "Rule: {rule_text}\nAgents: {agents_section}"
        pt = PromptTemplate(template)
        
        rule_text = "Always be polite"
        agent_prompts = {
            "Agent1": "You are a helpful assistant",
            "Agent2": "You are a medical advisor",
        }
        
        result = pt.render(rule_text, agent_prompts)
        
        assert "Always be polite" in result
        assert "Agent1" in result
        assert "Agent2" in result
        assert "You are a helpful assistant" in result
        assert "You are a medical advisor" in result
    
    def test_render_with_existing_rules(self):
        """Test template rendering with existing rules."""
        template = "{rule_text}\n{agents_section}\n{existing_rules_section}"
        pt = PromptTemplate(template)
        
        rule_text = "New rule"
        agent_prompts = {"Agent1": "Prompt 1"}
        existing_rules = [
            {
                "id": "rule-1",
                "rule_text": "Existing rule 1",
                "validation_metadata": {
                    "applied_agents": [
                        {"agent_name": "Agent1"}
                    ]
                }
            },
            {
                "id": "rule-2",
                "rule_text": "Existing rule 2",
                "validation_metadata": None
            }
        ]
        
        result = pt.render(rule_text, agent_prompts, existing_rules)
        
        assert "New rule" in result
        assert "EXISTING ACTIVE RULES" in result
        assert "rule-1" in result
        assert "Existing rule 1" in result
        assert "Applied to: Agent1" in result
        assert "rule-2" in result
        assert "Existing rule 2" in result
    
    def test_render_without_existing_rules(self):
        """Test template rendering without existing rules."""
        template = "{rule_text}\n{agents_section}\n{existing_rules_section}"
        pt = PromptTemplate(template)
        
        rule_text = "New rule"
        agent_prompts = {"Agent1": "Prompt 1"}
        
        result = pt.render(rule_text, agent_prompts, existing_rules=None)
        
        assert "New rule" in result
        assert "EXISTING ACTIVE RULES" not in result
    
    def test_render_with_empty_existing_rules(self):
        """Test template rendering with empty existing rules list."""
        template = "{rule_text}\n{agents_section}\n{existing_rules_section}"
        pt = PromptTemplate(template)
        
        rule_text = "New rule"
        agent_prompts = {"Agent1": "Prompt 1"}
        
        result = pt.render(rule_text, agent_prompts, existing_rules=[])
        
        assert "New rule" in result
        assert "EXISTING ACTIVE RULES" not in result
    
    def test_render_agents_section_format(self):
        """Test that agents section is formatted correctly."""
        template = "{agents_section}"
        pt = PromptTemplate(template)
        
        agent_prompts = {
            "TestAgent": "Test prompt content"
        }
        
        result = pt.render("rule", agent_prompts)
        
        # Check for proper formatting
        assert "### Agent 1: TestAgent" in result
        assert "```Test prompt content```" in result
    
    def test_render_multiple_agents_numbered(self):
        """Test that multiple agents are numbered correctly."""
        template = "{agents_section}"
        pt = PromptTemplate(template)
        
        agent_prompts = {
            "Agent1": "Prompt 1",
            "Agent2": "Prompt 2",
            "Agent3": "Prompt 3",
        }
        
        result = pt.render("rule", agent_prompts)
        
        assert "### Agent 1: Agent1" in result
        assert "### Agent 2: Agent2" in result
        assert "### Agent 3: Agent3" in result


class TestDefaultTemplate:
    """Tests for default validation template."""
    
    def test_default_template_exists(self):
        """Test that default template is defined."""
        assert DEFAULT_VALIDATION_TEMPLATE is not None
        assert len(DEFAULT_VALIDATION_TEMPLATE) > 0
    
    def test_default_template_has_placeholders(self):
        """Test that default template contains required placeholders."""
        assert "{rule_text}" in DEFAULT_VALIDATION_TEMPLATE
        assert "{agents_section}" in DEFAULT_VALIDATION_TEMPLATE
        assert "{existing_rules_section}" in DEFAULT_VALIDATION_TEMPLATE
    
    def test_default_template_has_validation_criteria(self):
        """Test that default template includes validation criteria."""
        assert "Role Consistency" in DEFAULT_VALIDATION_TEMPLATE
        assert "Authority Expansion" in DEFAULT_VALIDATION_TEMPLATE
        assert "Instruction Conflicts" in DEFAULT_VALIDATION_TEMPLATE
    
    def test_default_template_has_scoring_guidelines(self):
        """Test that default template includes scoring guidelines."""
        assert "SCORING GUIDELINES" in DEFAULT_VALIDATION_TEMPLATE
        assert "0.9-1.0" in DEFAULT_VALIDATION_TEMPLATE
        assert "0.7-0.9" in DEFAULT_VALIDATION_TEMPLATE
    
    def test_default_template_has_output_format(self):
        """Test that default template specifies JSON output format."""
        assert "OUTPUT FORMAT" in DEFAULT_VALIDATION_TEMPLATE
        assert "applied_agents" in DEFAULT_VALIDATION_TEMPLATE
        assert "agent_name" in DEFAULT_VALIDATION_TEMPLATE
        assert "overall_compatibility_score" in DEFAULT_VALIDATION_TEMPLATE
        assert "rule_to_apply" in DEFAULT_VALIDATION_TEMPLATE
    
    def test_default_template_mentions_threshold(self):
        """Test that default template mentions compatibility threshold."""
        assert "0.7" in DEFAULT_VALIDATION_TEMPLATE
        assert "threshold" in DEFAULT_VALIDATION_TEMPLATE.lower()
    
    def test_default_template_mentions_english_translation(self):
        """Test that default template requires English translation."""
        assert "English" in DEFAULT_VALIDATION_TEMPLATE
        assert "Translate" in DEFAULT_VALIDATION_TEMPLATE or "translate" in DEFAULT_VALIDATION_TEMPLATE


class TestGetDefaultTemplate:
    """Tests for get_default_template function."""
    
    def test_returns_prompt_template(self):
        """Test that function returns PromptTemplate instance."""
        result = get_default_template()
        assert isinstance(result, PromptTemplate)
    
    def test_uses_default_template(self):
        """Test that returned template uses DEFAULT_VALIDATION_TEMPLATE."""
        result = get_default_template()
        assert result.template == DEFAULT_VALIDATION_TEMPLATE
    
    def test_can_render(self):
        """Test that returned template can be rendered."""
        template = get_default_template()
        result = template.render(
            "Test rule",
            {"Agent1": "Test prompt"}
        )
        assert "Test rule" in result
        assert "Agent1" in result


class TestCreateCustomTemplate:
    """Tests for create_custom_template function."""
    
    def test_returns_prompt_template(self):
        """Test that function returns PromptTemplate instance."""
        custom = "Custom template: {rule_text}"
        result = create_custom_template(custom)
        assert isinstance(result, PromptTemplate)
    
    def test_uses_custom_template(self):
        """Test that returned template uses provided custom template."""
        custom = "Custom template: {rule_text}"
        result = create_custom_template(custom)
        assert result.template == custom
    
    def test_can_render_custom(self):
        """Test that custom template can be rendered."""
        custom = "Rule: {rule_text}, Agents: {agents_section}"
        template = create_custom_template(custom)
        result = template.render(
            "Custom rule",
            {"CustomAgent": "Custom prompt"}
        )
        assert "Custom rule" in result
        assert "CustomAgent" in result
    
    def test_different_from_default(self):
        """Test that custom template is different from default."""
        custom = "Completely different template"
        result = create_custom_template(custom)
        assert result.template != DEFAULT_VALIDATION_TEMPLATE


class TestPromptTemplateIntegration:
    """Integration tests for prompt template rendering."""
    
    def test_full_render_with_default_template(self):
        """Test full rendering with default template and realistic data."""
        template = get_default_template()
        
        rule_text = "Always verify patient identity before sharing medical information"
        agent_prompts = {
            "MedicalAdvisorAgent": "You are a medical advisor. Provide health guidance.",
            "InformationGatheringAgent": "You gather patient information.",
        }
        existing_rules = [
            {
                "id": "rule-123",
                "rule_text": "Maintain patient confidentiality",
                "validation_metadata": {
                    "applied_agents": [
                        {"agent_name": "MedicalAdvisorAgent"}
                    ]
                }
            }
        ]
        
        result = template.render(rule_text, agent_prompts, existing_rules)
        
        # Check all components are present
        assert rule_text in result
        assert "MedicalAdvisorAgent" in result
        assert "InformationGatheringAgent" in result
        assert "You are a medical advisor" in result
        assert "You gather patient information" in result
        assert "EXISTING ACTIVE RULES" in result
        assert "rule-123" in result
        assert "Maintain patient confidentiality" in result
        assert "Applied to: MedicalAdvisorAgent" in result
        
        # Check template structure is present
        assert "VALIDATION CRITERIA" in result
        assert "OUTPUT FORMAT" in result
        assert "SCORING GUIDELINES" in result
