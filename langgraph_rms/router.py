"""
FastAPI router for internal RMS endpoints.

This module provides secure API endpoints for rule validation and cache refresh
operations, authenticated via API key.
"""

from fastapi import APIRouter, Header, HTTPException, status, Depends
from langgraph_rms.config import get_config
from langgraph_rms.models import ValidationRequest, RuleValidation, RefreshRequest
from langgraph_rms.validator import RuleValidator
from langgraph_rms.utils import create_llm_client
from langgraph_rms.cache import _cache


def create_router() -> APIRouter:
    """
    Create FastAPI router with internal RMS endpoints.

    Returns:
        Configured APIRouter instance with /internal prefix

    Example:
        >>> from fastapi import FastAPI
        >>> from langgraph_rms import create_router, initialize, RMSConfig
        >>> 
        >>> # Initialize configuration
        >>> config = RMSConfig(
        ...     product_name="my-product",
        ...     agent_prompts={"Agent1": "You are helpful"},
        ...     rms_url="https://rms.example.com",
        ...     api_key="secret-key"
        ...     llm_model="gpt-4"
        ... )
        >>> initialize(config)
        >>> 
        >>> # Create FastAPI app and include router
        >>> app = FastAPI()
        >>> router = create_router()
        >>> app.include_router(router)
    """
    router = APIRouter(prefix="/internal")

    @router.post("/rules/validate", response_model=RuleValidation, dependencies=[Depends(verify_api_key)])
    async def validate_rule(request: ValidationRequest) -> RuleValidation:
        """
        Validate a rule against all configured agent prompts.

        Args:
            request: ValidationRequest with rule_text and product_name

        Returns:
            RuleValidation with compatibility scores and metadata

        Raises:
            HTTPException: If validation fails (HTTP 500)
        """
        try:
            config = get_config()

            # Create LLM client
            llm_client = create_llm_client() #TODO: back to: create_llm_client(config.llm_model)

            # Create validator
            validator = RuleValidator(llm_client)

            # Get existing rules from cache
            existing_rules = await _cache.get_active_rules(request.product_name)

            # Validate the rule
            validation_result = await validator.validate_rule(
                rule_text=request.rule_text,
                agent_prompts=config.agent_prompts,
                existing_rules=existing_rules,
            )

            return validation_result

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Validation failed: {str(e)}"
            )

    @router.post("/rules/refresh", dependencies=[Depends(verify_api_key)])
    async def refresh_rules(request: RefreshRequest) -> dict:
        """
        Refresh cached rules for a product.

        Args:
            request: RefreshRequest with list of rules

        Returns:
            Success message with count of refreshed rules

        Raises:
            HTTPException: If refresh fails (HTTP 500)
        """
        try:
            config = get_config()
            product_name = config.product_name

            # Extract product name from first rule if available
            if request.rules:
                product_name = request.rules[0].product_name

            # Refresh the cache with new rules
            await _cache.refresh_rules(product_name, request.rules)

            return {
                "success": True,
                "message": f"Successfully refreshed {len(request.rules)} rules for product '{product_name}'",
                "product_name": product_name,
                "rule_count": len(request.rules)
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Refresh failed: {str(e)}"
            )

    return router


async def verify_api_key(x_internal_api_key: str = Header(...)) -> bool:
    """
    Verify internal API key for secure communication.
    
    This dependency function checks the X-Internal-API-Key header against
    the configured API key to authenticate internal requests.
    
    Args:
        x_internal_api_key: API key from request header
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: If API key is invalid (HTTP 401 Unauthorized)
    """
    config = get_config()
    
    if x_internal_api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return True
