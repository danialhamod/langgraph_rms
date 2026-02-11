"""
Shared pytest fixtures for all tests.

This module provides common fixtures used across unit, property, and integration tests.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langgraph_rms.config import RMSConfig, initialize
from langgraph_rms.models import (
    AgentRuleInfo,
    CachedRule,
    RuleValidation,
    ValidationMetadata,
)
from langgraph_rms.router import create_router


@pytest.fixture
def mock_llm():
    """
    Mock LangChain LLM for validation tests.
    
    Returns a mock LLM client with ainvoke method that can be configured
    to return specific responses for testing validation logic.
    """
    llm = Mock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def sample_agent_prompts():
    """
    Sample agent prompts for testing.
    
    Returns a dictionary mapping agent names to their system prompts,
    representing typical agents in a multi-agent system.
    """
    return {
        "MedicalAdvisorAgent": "You are a medical advisor. Provide medical advice based on symptoms and patient history.",
        "RiskAssessmentAgent": "You are a risk assessment agent. Evaluate medical risks and provide safety recommendations.",
        "InformationGatheringAgent": "You are an information gathering agent. Collect relevant patient information through questions.",
    }


@pytest.fixture
def sample_rules():
    """
    Sample rules with validation metadata for testing.
    
    Returns a list of CachedRule objects with complete validation metadata,
    representing rules that have been validated against multiple agents.
    """
    return [
        CachedRule(
            id="rule1",
            product_name="test_product",
            rule_text="Always verify patient symptoms before providing advice",
            max_length=100,
            risk_level="low",
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            latest_validation=RuleValidation(
                can_be_applied=True,
                max_compatibility_score=0.9,
                explanation="Compatible with MedicalAdvisorAgent and InformationGatheringAgent",
                validation_metadata=ValidationMetadata(
                    applied_agents=[
                        AgentRuleInfo(
                            agent_name="MedicalAdvisorAgent",
                            role_consistency_score=0.9,
                            authority_expansion_score=0.9,
                            instruction_conflicts_score=0.9,
                            overall_compatibility_score=0.9,
                            analysis="Rule aligns well with medical advisor role",
                            concerns=[],
                            rule_to_apply="Always verify patient symptoms before providing advice.",
                        ),
                        AgentRuleInfo(
                            agent_name="InformationGatheringAgent",
                            role_consistency_score=0.85,
                            authority_expansion_score=0.85,
                            instruction_conflicts_score=0.85,
                            overall_compatibility_score=0.85,
                            analysis="Rule supports information gathering process",
                            concerns=[],
                            rule_to_apply="Always verify patient symptoms before proceeding.",
                        ),
                    ]
                ),
            ),
        ),
        CachedRule(
            id="rule2",
            product_name="test_product",
            rule_text="Assess risk level for all medical recommendations",
            max_length=100,
            risk_level="medium",
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            latest_validation=RuleValidation(
                can_be_applied=True,
                max_compatibility_score=0.85,
                explanation="Compatible with RiskAssessmentAgent",
                validation_metadata=ValidationMetadata(
                    applied_agents=[
                        AgentRuleInfo(
                            agent_name="RiskAssessmentAgent",
                            role_consistency_score=0.85,
                            authority_expansion_score=0.85,
                            instruction_conflicts_score=0.85,
                            overall_compatibility_score=0.85,
                            analysis="Rule directly supports risk assessment function",
                            concerns=[],
                            rule_to_apply="Assess risk level for all medical recommendations.",
                        ),
                    ]
                ),
            ),
        ),
        CachedRule(
            id="rule3",
            product_name="test_product",
            rule_text="Document all patient interactions",
            max_length=100,
            risk_level="low",
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            latest_validation=RuleValidation(
                can_be_applied=True,
                max_compatibility_score=0.8,
                explanation="Compatible with all agents",
                validation_metadata=ValidationMetadata(
                    applied_agents=[
                        AgentRuleInfo(
                            agent_name="MedicalAdvisorAgent",
                            role_consistency_score=0.8,
                            authority_expansion_score=0.8,
                            instruction_conflicts_score=0.8,
                            overall_compatibility_score=0.8,
                            analysis="Documentation is important for medical advice",
                            concerns=[],
                            rule_to_apply="Document all patient interactions.",
                        ),
                        AgentRuleInfo(
                            agent_name="RiskAssessmentAgent",
                            role_consistency_score=0.8,
                            authority_expansion_score=0.8,
                            instruction_conflicts_score=0.8,
                            overall_compatibility_score=0.8,
                            analysis="Documentation supports risk assessment",
                            concerns=[],
                            rule_to_apply="Document all patient interactions.",
                        ),
                        AgentRuleInfo(
                            agent_name="InformationGatheringAgent",
                            role_consistency_score=0.8,
                            authority_expansion_score=0.8,
                            instruction_conflicts_score=0.8,
                            overall_compatibility_score=0.8,
                            analysis="Documentation is part of information gathering",
                            concerns=[],
                            rule_to_apply="Document all patient interactions.",
                        ),
                    ]
                ),
            ),
        ),
    ]


@pytest.fixture
def test_config():
    """
    Test configuration instance with valid RMSConfig.
    
    Returns a properly initialized RMSConfig instance suitable for testing.
    This fixture also initializes the global configuration singleton.
    """
    config = RMSConfig(
        product_name="test_product",
        agent_prompts={
            "MedicalAdvisorAgent": "You are a medical advisor. Provide medical advice based on symptoms.",
            "RiskAssessmentAgent": "You are a risk assessment agent. Evaluate medical risks.",
            "InformationGatheringAgent": "You are an information gathering agent. Collect patient information.",
        },
        rms_url="https://test-rms.example.com",
        api_key="test-api-key-12345",
        llm_model="gpt-4",
        compatibility_threshold=0.7,
        request_timeout=10.0,
    )
    initialize(config)
    return config


@pytest.fixture
def api_client(test_config):
    """
    FastAPI TestClient for testing API endpoints.
    
    Returns a configured TestClient with the RMS router mounted,
    ready for testing internal API endpoints.
    
    Args:
        test_config: Test configuration fixture (dependency)
    """
    router = create_router()
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)
