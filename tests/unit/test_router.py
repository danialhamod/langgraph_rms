"""
Unit tests for FastAPI router and authentication.
"""

import pytest
from fastapi import HTTPException
from langgraph_rms.router import create_router, verify_api_key
from langgraph_rms.config import RMSConfig, initialize


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RMSConfig(
        product_name="test_product",
        agent_prompts={"agent1": "You are agent 1"},
        rms_url="https://test-rms.example.com",
        api_key="test-api-key-12345",
        llm_model="gpt-4",
        compatibility_threshold=0.7,
        request_timeout=10.0,
    )
    initialize(config)
    return config


def test_create_router():
    """Test that create_router returns a configured APIRouter."""
    router = create_router()
    
    assert router is not None
    assert router.prefix == "/internal"


@pytest.mark.asyncio
async def test_verify_api_key_valid(test_config):
    """Test that verify_api_key accepts valid API key."""
    result = await verify_api_key(x_internal_api_key="test-api-key-12345")
    assert result is True


@pytest.mark.asyncio
async def test_verify_api_key_invalid(test_config):
    """Test that verify_api_key rejects invalid API key."""
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(x_internal_api_key="wrong-api-key")
    
    assert exc_info.value.status_code == 401
    assert "Invalid API key" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_api_key_empty(test_config):
    """Test that verify_api_key rejects empty API key."""
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(x_internal_api_key="")
    
    assert exc_info.value.status_code == 401
    assert "Invalid API key" in exc_info.value.detail


@pytest.mark.asyncio
async def test_validate_endpoint_success(test_config):
    """Test successful rule validation via API endpoint."""
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock, MagicMock, patch
    from langgraph_rms.models import RuleValidation, ValidationMetadata, AgentRuleInfo
    
    # Mock validation result
    mock_validation = RuleValidation(
        can_be_applied=True,
        max_compatibility_score=0.85,
        explanation="Rule is compatible with Agent1",
        validation_metadata=ValidationMetadata(
            applied_agents=[
                AgentRuleInfo(
                    agent_name="Agent1",
                    role_consistency_score=0.9,
                    authority_expansion_score=0.8,
                    instruction_conflicts_score=0.85,
                    overall_compatibility_score=0.85,
                    analysis="Rule aligns well with agent role",
                    concerns=[],
                    rule_to_apply="Always verify patient information before proceeding."
                )
            ]
        )
    )
    
    # Mock the validator
    with patch('langgraph_rms.router.create_llm_client') as mock_llm_factory, \
         patch('langgraph_rms.router.RuleValidator') as mock_validator_class:
        
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm
        
        mock_validator = MagicMock()
        mock_validator.validate_rule = AsyncMock(return_value=mock_validation)
        mock_validator_class.return_value = mock_validator
        
        # Create test client AFTER patching
        router = create_router()
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Make request
        response = client.post(
            "/internal/rules/validate",
            json={
                "rule_text": "Always verify patient information",
                "product_name": "test_product"
            },
            headers={"X-Internal-API-Key": "test-api-key-12345"}
        )
    
    # Verify response
    if response.status_code != 200:
        print(f"Error response: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert data["can_be_applied"] is True
    assert data["max_compatibility_score"] == 0.85
    assert len(data["validation_metadata"]["applied_agents"]) == 1


@pytest.mark.asyncio
async def test_validate_endpoint_authentication_required(test_config):
    """Test that validate endpoint requires authentication."""
    from fastapi.testclient import TestClient
    
    # Create test client
    router = create_router()
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    # Make request without API key
    response = client.post(
        "/internal/rules/validate",
        json={
            "rule_text": "Always verify patient information",
            "product_name": "test_product"
        }
    )
    
    # Verify authentication failure
    assert response.status_code == 422  # Missing required header


@pytest.mark.asyncio
async def test_validate_endpoint_invalid_api_key(test_config):
    """Test that validate endpoint rejects invalid API key."""
    from fastapi.testclient import TestClient
    
    # Create test client
    router = create_router()
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    # Make request with wrong API key
    response = client.post(
        "/internal/rules/validate",
        json={
            "rule_text": "Always verify patient information",
            "product_name": "test_product"
        },
        headers={"X-Internal-API-Key": "wrong-key"}
    )
    
    # Verify authentication failure
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_validate_endpoint_validation_error(test_config):
    """Test that validate endpoint handles validation errors."""
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock, MagicMock, patch
    
    # Mock the validator to raise an exception
    with patch('langgraph_rms.router.create_llm_client') as mock_llm_factory, \
         patch('langgraph_rms.router.RuleValidator') as mock_validator_class:
        
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm
        
        mock_validator = MagicMock()
        mock_validator.validate_rule = AsyncMock(side_effect=Exception("LLM service unavailable"))
        mock_validator_class.return_value = mock_validator
        
        # Create test client AFTER patching
        router = create_router()
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Make request
        response = client.post(
            "/internal/rules/validate",
            json={
                "rule_text": "Always verify patient information",
                "product_name": "test_product"
            },
            headers={"X-Internal-API-Key": "test-api-key-12345"}
        )
    
    # Verify error response
    assert response.status_code == 500
    assert "Validation failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_refresh_endpoint_success(test_config):
    """Test successful cache refresh via API endpoint."""
    from fastapi.testclient import TestClient
    from datetime import datetime
    from langgraph_rms.models import CachedRule, RuleValidation, ValidationMetadata, AgentRuleInfo
    
    # Create test rules
    test_rules = [
        CachedRule(
            id="rule-1",
            product_name="test_product",
            rule_text="Always verify patient information",
            max_length=500,
            risk_level="medium",
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            latest_validation=RuleValidation(
                can_be_applied=True,
                max_compatibility_score=0.85,
                explanation="Compatible",
                validation_metadata=ValidationMetadata(
                    applied_agents=[
                        AgentRuleInfo(
                            agent_name="agent1",
                            role_consistency_score=0.9,
                            authority_expansion_score=0.8,
                            instruction_conflicts_score=0.85,
                            overall_compatibility_score=0.85,
                            analysis="Good fit",
                            concerns=[],
                            rule_to_apply="Always verify patient information."
                        )
                    ]
                )
            )
        )
    ]
    
    # Create test client
    router = create_router()
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    # Make request
    response = client.post(
        "/internal/rules/refresh",
        json={
            "rules": [rule.model_dump(mode='json') for rule in test_rules]
        },
        headers={"X-Internal-API-Key": "test-api-key-12345"}
    )
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["product_name"] == "test_product"
    assert data["rule_count"] == 1
    assert "Successfully refreshed" in data["message"]


@pytest.mark.asyncio
async def test_refresh_endpoint_empty_rules(test_config):
    """Test cache refresh with empty rules list."""
    from fastapi.testclient import TestClient
    
    # Create test client
    router = create_router()
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    # Make request with empty rules
    response = client.post(
        "/internal/rules/refresh",
        json={"rules": []},
        headers={"X-Internal-API-Key": "test-api-key-12345"}
    )
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["rule_count"] == 0


@pytest.mark.asyncio
async def test_refresh_endpoint_authentication_required(test_config):
    """Test that refresh endpoint requires authentication."""
    from fastapi.testclient import TestClient
    
    # Create test client
    router = create_router()
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    # Make request without API key
    response = client.post(
        "/internal/rules/refresh",
        json={"rules": []}
    )
    
    # Verify authentication failure
    assert response.status_code == 422  # Missing required header


@pytest.mark.asyncio
async def test_refresh_endpoint_invalid_api_key(test_config):
    """Test that refresh endpoint rejects invalid API key."""
    from fastapi.testclient import TestClient
    
    # Create test client
    router = create_router()
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    # Make request with wrong API key
    response = client.post(
        "/internal/rules/refresh",
        json={"rules": []},
        headers={"X-Internal-API-Key": "wrong-key"}
    )
    
    # Verify authentication failure
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_refresh_endpoint_error_handling(test_config):
    """Test that refresh endpoint handles errors gracefully."""
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock, patch
    
    # Mock the cache to raise an exception
    with patch('langgraph_rms.router._cache.refresh_rules', new_callable=AsyncMock) as mock_refresh:
        mock_refresh.side_effect = Exception("Cache update failed")
        
        # Create test client AFTER patching
        router = create_router()
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Make request
        response = client.post(
            "/internal/rules/refresh",
            json={"rules": []},
            headers={"X-Internal-API-Key": "test-api-key-12345"}
        )
    
    # Verify error response
    assert response.status_code == 500
    assert "Refresh failed" in response.json()["detail"]
