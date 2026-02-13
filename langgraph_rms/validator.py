"""
Rule validator for LangGraph RMS Integration.

This module provides LLM-based validation of rules against agent system prompts.
It evaluates rule compatibility, detects conflicts, and transforms rules into
English instructions ready for prompt injection.
"""

from typing import Any, Dict, List, Optional
import logging

from langgraph_rms.models import (
    RuleValidation,
    ValidationMetadata,
    AgentRuleInfo,
    CachedRule,
)
from langgraph_rms.prompts import get_default_template, PromptTemplate
from langgraph_rms.utils import safe_json_parse
from langgraph_rms.config import get_config


logger = logging.getLogger(__name__)


class RuleValidator:
    """Validates rules against agent system prompts using LLM.

    This class uses a language model to evaluate rule compatibility with
    agent prompts, computing compatibility scores and detecting conflicts.

    Example:
        >>> from langgraph_rms import RuleValidator, create_llm_client
        >>> llm = create_llm_client()
        >>> validator = RuleValidator(llm)
        >>> result = await validator.validate_rule(
        ...     rule_text="Always be polite and professional",
        ...     agent_prompts={"Agent1": "You are a helpful assistant"}
        ... )
        >>> print(f"Can apply: {result.can_be_applied}")
        >>> print(f"Max score: {result.max_compatibility_score}")
    """

    def __init__(
        self,
        llm_client: Any,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        """
        Initialize validator with LLM client.

        Args:
            llm_client: LangChain LLM instance for validation
            prompt_template: Optional custom prompt template (uses default if None)
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template or get_default_template()

    async def validate_rule(
        self,
        rule_text: str,
        agent_prompts: Dict[str, str],
        existing_rules: Optional[List[CachedRule]] = None,
        scoring_callback: Optional[Any] = None,
    ) -> RuleValidation:
        """
        Validate a rule against all agent prompts.

        Args:
            rule_text: The rule text to validate
            agent_prompts: Dictionary of agent names to system prompts
            existing_rules: Optional list of existing active rules
            scoring_callback: Optional callback for custom scoring logic

        Returns:
            RuleValidation with compatibility scores and metadata

        Example:
            >>> from langgraph_rms import RuleValidator, create_llm_client
            >>> llm = create_llm_client()
            >>> validator = RuleValidator(llm)
            >>> result = await validator.validate_rule(
            ...     rule_text="Always be polite",
            ...     agent_prompts={"Agent1": "You are a helpful assistant"}
            ... )
            >>> print(result.can_be_applied)
            True
        """
        try:
            # Build validation prompt
            prompt = self._build_validation_prompt(
                rule_text, agent_prompts, existing_rules
            )

            # Invoke LLM
            response = await self.llm_client.ainvoke(prompt)

            # Parse response
            parsed_data = self._parse_validation_response(response.content)

            # Build validation result
            applied_agents = []
            config = get_config()

            for agent_data in parsed_data.get("applied_agents", []):
                # Calculate overall score as average of component scores
                overall_score = (
                    agent_data["role_consistency_score"]
                    + agent_data["authority_expansion_score"]
                    + agent_data["instruction_conflicts_score"]
                ) / 3.0

                # Only include agents that meet the threshold
                if overall_score >= config.compatibility_threshold:
                    agent_info = AgentRuleInfo(
                        agent_name=agent_data["agent_name"],
                        role_consistency_score=agent_data["role_consistency_score"],
                        authority_expansion_score=agent_data[
                            "authority_expansion_score"
                        ],
                        instruction_conflicts_score=agent_data[
                            "instruction_conflicts_score"
                        ],
                        overall_compatibility_score=overall_score,
                        analysis=agent_data.get("analysis", ""),
                        concerns=agent_data.get("concerns", []),
                        rule_to_apply=agent_data["rule_to_apply"],
                    )
                    applied_agents.append(agent_info)

            # Apply custom scoring callback if provided
            if scoring_callback:
                applied_agents = scoring_callback(applied_agents)

            # Calculate max compatibility score
            max_score = (
                max(agent.overall_compatibility_score for agent in applied_agents)
                if applied_agents
                else 0.0
            )

            validation_metadata = ValidationMetadata(applied_agents=applied_agents)

            result = RuleValidation(
                can_be_applied=len(applied_agents) > 0,
                max_compatibility_score=max_score,
                explanation=parsed_data.get(
                    "system_summary", "تم التحقق من القاعدة"
                ),
                explanation_en=parsed_data.get(
                    "system_summary_en", "Rule validation completed"
                ),
                validation_metadata=validation_metadata,
            )

            return result

        except Exception as e:
            # Return safe default on any error
            logger.error(f"Rule validation failed: {e}", exc_info=True)
            return RuleValidation(
                can_be_applied=False,
                max_compatibility_score=0.0,
                explanation=f"Validation failed: {str(e)}",
                validation_metadata=ValidationMetadata(applied_agents=[]),
            )

    def _build_validation_prompt(
        self,
        rule_text: str,
        agent_prompts: Dict[str, str],
        existing_rules: Optional[List[CachedRule]] = None,
    ) -> str:
        """
        Build LLM prompt for validation.

        Args:
            rule_text: The rule text to validate
            agent_prompts: Dictionary of agent names to system prompts
            existing_rules: Optional list of existing active rules

        Returns:
            Formatted prompt string ready for LLM
        """
        # Convert existing rules to dict format for template
        existing_rules_data = None
        if existing_rules:
            existing_rules_data = []
            for rule in existing_rules:
                rule_dict = {
                    "id": rule.id,
                    "rule_text": rule.rule_text,
                }
                if rule.latest_validation:
                    rule_dict["validation_metadata"] = {
                        "applied_agents": [
                            {"agent_name": agent.agent_name}
                            for agent in rule.latest_validation.validation_metadata.applied_agents
                        ]
                    }
                existing_rules_data.append(rule_dict)

        # Render template
        return self.prompt_template.render(
            rule_text=rule_text,
            agent_prompts=agent_prompts,
            existing_rules=existing_rules_data,
        )

    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response.

        Args:
            response: Raw LLM response content

        Returns:
            Parsed validation data dictionary

        Raises:
            ValueError: If response cannot be parsed or is invalid
        """
        # Use safe_json_parse to handle markdown code blocks
        parsed = safe_json_parse(response)

        # Validate required fields
        if "applied_agents" not in parsed:
            raise ValueError("Response missing 'applied_agents' field")

        # Validate each agent entry
        for agent in parsed["applied_agents"]:
            required_fields = [
                "agent_name",
                "role_consistency_score",
                "authority_expansion_score",
                "instruction_conflicts_score",
                "rule_to_apply",
            ]
            for field in required_fields:
                if field not in agent:
                    raise ValueError(f"Agent entry missing required field: {field}")

            # Validate score ranges
            for score_field in [
                "role_consistency_score",
                "authority_expansion_score",
                "instruction_conflicts_score",
            ]:
                score = agent[score_field]
                if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"{score_field} must be a number between 0.0 and 1.0"
                    )

        return parsed
