"""
Data models for LangGraph RMS Integration.

All models use Pydantic for type safety and validation.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class AgentRuleInfo(BaseModel):
    """Information about how a rule applies to a specific agent."""
    
    agent_name: str = Field(..., description="Name of the agent")
    role_consistency_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Score indicating how well the rule aligns with the agent's role"
    )
    authority_expansion_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Score indicating if the rule expands agent authority appropriately"
    )
    instruction_conflicts_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Score indicating absence of conflicts with existing instructions"
    )
    overall_compatibility_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Overall compatibility score (average of component scores)"
    )
    analysis: str = Field(..., description="Detailed analysis of rule compatibility")
    concerns: List[str] = Field(
        default_factory=list,
        description="List of concerns or issues identified"
    )
    rule_to_apply: str = Field(
        ..., 
        description="English-formatted rule text ready for prompt injection"
    )

    class Config:
        frozen = True


class ValidationMetadata(BaseModel):
    """Validation metadata for a rule."""
    
    applied_agents: List[AgentRuleInfo] = Field(
        default_factory=list,
        description="List of agents this rule applies to with their compatibility info"
    )

    class Config:
        frozen = True


class RuleValidation(BaseModel):
    """Validation result for a rule."""
    
    can_be_applied: bool = Field(
        ..., 
        description="Whether the rule can be applied to any agent"
    )
    max_compatibility_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Maximum compatibility score across all agents"
    )
    explanation: str = Field(..., description="Arabic explanation of validation result")
    explanation_en: str = Field(..., description="English explanation of validation result")
    validation_metadata: ValidationMetadata = Field(
        ..., 
        description="Detailed validation metadata for each agent"
    )

    class Config:
        frozen = True


class CachedRule(BaseModel):
    """Cached rule representation."""
    
    id: str = Field(..., description="Unique identifier for the rule")
    product_name: str = Field(..., description="Name of the product this rule applies to")
    rule_text: str = Field(..., description="Original rule text")
    max_length: int = Field(..., description="Maximum length constraint for the rule")
    risk_level: str = Field(..., description="Risk level of the rule")
    status: str = Field(..., description="Current status of the rule")
    created_at: datetime = Field(..., description="Timestamp when rule was created")
    updated_at: datetime = Field(..., description="Timestamp when rule was last updated")
    latest_validation: Optional[RuleValidation] = Field(
        None,
        description="Most recent validation result for this rule"
    )

    class Config:
        frozen = True


class ValidationRequest(BaseModel):
    """Request model for rule validation."""
    
    rule_text: str = Field(..., description="Rule text to validate")
    product_name: str = Field(..., description="Name of the product")

    class Config:
        frozen = True


class RefreshRequest(BaseModel):
    """Request model for cache refresh."""
    
    rules: List[CachedRule] = Field(
        ..., 
        description="List of active rules to cache"
    )

    class Config:
        frozen = True
