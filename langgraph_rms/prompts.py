"""
Validation prompt templates for rule validation.

This module provides prompt templates for LLM-based rule validation
against agent system prompts. It includes a default template based on
the proven Hayatok implementation and supports custom templates.
"""
from typing import Dict, List, Optional, Any


class PromptTemplate:
    """Template for validation prompts with variable substitution.

    This class manages prompt templates for rule validation, allowing
    customization of the validation prompt structure.

    Example:
        >>> from langgraph_rms import PromptTemplate
        >>> template = PromptTemplate(
        ...     "Validate this rule: {rule_text}\\n"
        ...     "Against agents: {agents_section}"
        ... )
        >>> prompt = template.render(
        ...     rule_text="Be concise",
        ...     agent_prompts={"Agent1": "You are helpful"}
        ... )
    """
    
    def __init__(self, template: str):
        """
        Initialize with template string.
        
        Args:
            template: Template string with placeholders for {rule_text},
                     {agents_section}, and {existing_rules_section}
        """
        self.template = template
    
    def render(
        self,
        rule_text: str,
        agent_prompts: Dict[str, str],
        existing_rules: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Render template with provided data.
        
        Args:
            rule_text: The rule text to validate
            agent_prompts: Dictionary mapping agent names to system prompts
            existing_rules: Optional list of existing active rules
            
        Returns:
            Rendered prompt string ready for LLM
        """
        # Build agents section
        agents_section = ""
        for i, (agent_name, agent_prompt) in enumerate(agent_prompts.items(), 1):
            agents_section += f"""
### Agent {i}: {agent_name}
```{agent_prompt}```
"""
        
        # Build existing rules section
        existing_rules_section = ""
        if existing_rules:
            existing_rules_section = "\n**EXISTING ACTIVE RULES:**\n"
            for rule in existing_rules:
                existing_rules_section += f"- Rule ID: {rule.get('id', 'unknown')}\n"
                if rule.get('validation_metadata'):
                    applied_to = [
                        agent['agent_name']
                        for agent in rule['validation_metadata'].get('applied_agents', [])
                    ]
                    existing_rules_section += f"  Applied to: {', '.join(applied_to)}\n"
                existing_rules_section += f"  Text: {rule.get('rule_text', '')}\n\n"
        
        # Render template with substitutions
        return self.template.format(
            rule_text=rule_text,
            agents_section=agents_section,
            existing_rules_section=existing_rules_section,
        )


# Default validation template from Hayatok implementation
DEFAULT_VALIDATION_TEMPLATE = """You are a rule compatibility validator for a LangGraph-based multi-agent system. Evaluate whether a new rule is compatible with the system's agents and determine which agents it should be applied to.

**SYSTEM ARCHITECTURE:**
We have a multi-agent medical chatbot system using LangGraph. Each agent has a specific role and responsibilities. When applying rules, we need to ensure they are compatible with the intended agents' purposes.

**AGENTS IN THE SYSTEM:**
{agents_section}
{existing_rules_section}
**NEW RULE TO VALIDATE:**
```
{rule_text}
```

**VALIDATION CRITERIA FOR EACH AGENT:**

1. **Role Consistency**: Does the rule align with the agent's core role and purpose?
   - Score: 0.0 (completely inconsistent) to 1.0 (fully consistent)

2. **Authority Expansion**: Does the rule attempt to expand the agent's authority beyond its intended scope?
   - Score: 0.0 (major expansion) to 1.0 (no expansion)

3. **Instruction Conflicts**: Does the rule conflict with any explicit instructions in the agent prompt OR with existing active rules for that agent?
   - Score: 0.0 (major conflicts) to 1.0 (no conflicts)
   - IMPORTANT: Check for conflicts with both the agent's base prompt AND any existing active rules that apply to this agent

**SCORING GUIDELINES:**
- 0.9-1.0: Excellent compatibility, minor or no concerns
- 0.7-0.9: Good compatibility, some concerns but acceptable
- 0.5-0.7: Moderate compatibility, significant concerns
- 0.3-0.5: Poor compatibility, major concerns
- 0.0-0.3: Incompatible, should not be applied

**RULE TRANSFORMATION REQUIREMENT:**
For each agent that meets the compatibility threshold (>= 0.7), you MUST:
1. Translate the rule to English (if not already in English)
2. Refine it to be clear, concise, and ready to insert into an agent's system prompt
3. Format it as a direct instruction that can be appended to the agent's prompt
4. Preserve the original meaning and intent of the rule
5. Ensure it does not contradict existing rules for that agent

**OUTPUT FORMAT (JSON only):**
```json
{{
  "applied_agents": [
    {{
      "agent_name": "<exact agent name from the list above>",
      "role_consistency_score": <float 0.0-1.0>,
      "authority_expansion_score": <float 0.0-1.0>,
      "instruction_conflicts_score": <float 0.0-1.0>,
      "overall_compatibility_score": <float 0.0-1.0, average of three scores>,
      "analysis": "<brief explanation of why this agent is/isn't suitable>",
      "concerns": ["<list specific concerns if any, including conflicts with existing rules>"],
      "rule_to_apply": "<English version of the rule, formatted as a clear instruction ready to add to the agent's prompt>"
    }}
  ],
  "system_summary": "<arabic 50-75 words brief overall summary of the rule's compatibility with the system, without mentioning scores, with mention of which agents the rule can be applied to and which are not suitable>",
  "system_summary_en": "<english version of system_summary above>"
}}
```

**IMPORTANT:**
- Only include agents in `applied_agents` that have `overall_compatibility_score >= 0.7`
- If no agents meet the threshold, return an empty array for `applied_agents`
- Use the exact agent names from the list above
- `rule_to_apply` MUST be in English and formatted as a direct instruction
- `rule_to_apply` should be concise (1-3 sentences) and actionable
- CRITICAL: Check for conflicts with existing active rules when evaluating instruction_conflicts_score

Return ONLY valid JSON. No additional text."""


def get_default_template() -> PromptTemplate:
    """
    Get default validation prompt template.
    
    Returns:
        PromptTemplate instance with default template

    Example:
        >>> from langgraph_rms import get_default_template
        >>> template = get_default_template()
        >>> prompt = template.render(
        ...     rule_text="Always be polite",
        ...     agent_prompts={"Agent1": "You are helpful"}
        ... )
    """
    return PromptTemplate(DEFAULT_VALIDATION_TEMPLATE)


def create_custom_template(template: str) -> PromptTemplate:
    """
    Create custom validation prompt template.
    
    Args:
        template: Custom template string with placeholders for {rule_text},
                 {agents_section}, and {existing_rules_section}
    
    Returns:
        PromptTemplate instance with custom template

    Example:
        >>> from langgraph_rms import create_custom_template
        >>> custom = create_custom_template(
        ...     "Validate: {rule_text}\\nAgents: {agents_section}"
        ... )
        >>> prompt = custom.render(
        ...     rule_text="Be concise",
        ...     agent_prompts={"Agent1": "You are helpful"}
        ... )
    """
    return PromptTemplate(template)
