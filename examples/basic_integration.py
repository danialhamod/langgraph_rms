"""
Basic Integration Example for LangGraph RMS Integration Package

This example demonstrates a complete working integration showing:
1. Configuration setup
2. Rule validation
3. Cache management
4. Rule appending to agent prompts

Requirements: 9.5
"""

import asyncio
import os
from datetime import datetime

# Import the langgraph-rms-integration package
from langgraph_rms import (
    # Configuration
    RMSConfig,
    initialize,
    get_config,
    # Models
    CachedRule,
    RuleValidation,
    RefreshRequest,
    # Validation
    RuleValidator,
    # Caching
    get_rules_for_agent,
    format_rules_for_prompt,
    # Appending
    RulesAppender,
    append_rules,
    # Router
    create_router,
    # Utilities
    create_llm_client,
)


# ============================================================================
# Step 1: Define Agent System Prompts
# ============================================================================
# In a real application, these would be your actual agent prompts

AGENT_PROMPTS = {
    "MedicalAdvisorAgent": """You are a medical advisor AI assistant. Your role is to:
- Provide evidence-based medical information
- Help users understand their health conditions
- Suggest when to seek professional medical care
- Never diagnose or prescribe medication
- Always prioritize patient safety""",
    
    "InformationGatheringAgent": """You are an information gathering agent. Your role is to:
- Ask relevant questions to understand the user's situation
- Collect necessary details for accurate assistance
- Be empathetic and patient-focused
- Maintain conversation flow naturally""",
    
    "RiskAssessmentAgent": """You are a risk assessment agent. Your role is to:
- Evaluate potential health risks based on symptoms
- Identify urgent situations requiring immediate care
- Provide clear risk level assessments
- Recommend appropriate next steps""",
}


# ============================================================================
# Step 2: Initialize Configuration
# ============================================================================

def setup_configuration():
    """
    Initialize the RMS integration with configuration.
    
    This step sets up all the necessary parameters for the package to work,
    including agent prompts, RMS service URL, API credentials, and validation settings.
    """
    print("=" * 70)
    print("STEP 1: Initializing Configuration")
    print("=" * 70)
    
    # Create configuration with all required parameters
    config = RMSConfig(
        # Product identification
        product_name="example_medical_app",
        
        # Agent system prompts for validation
        agent_prompts=AGENT_PROMPTS,
        
        # RMS service connection (use environment variables in production)
        rms_url=os.getenv("RMS_URL", "https://rms.example.com"),
        api_key=os.getenv("RMS_API_KEY", "your-api-key-here"),
        
        # LLM configuration for rule validation
        llm_model=os.getenv("LLM_MODEL", "gpt-4"),
        
        # Validation threshold (rules must score >= 0.7 to be applied)
        compatibility_threshold=0.7,
        
        # HTTP request timeout in seconds
        request_timeout=10.0,
    )
    
    # Initialize the package with this configuration
    initialize(config)
    
    print(f"✓ Configuration initialized for product: {config.product_name}")
    print(f"✓ Configured {len(config.agent_prompts)} agents")
    print(f"✓ Compatibility threshold: {config.compatibility_threshold}")
    print(f"✓ LLM model: {config.llm_model}")
    print()


# ============================================================================
# Step 3: Validate a Rule
# ============================================================================

async def validate_new_rule():
    """
    Validate a new rule against all configured agent prompts.
    
    This demonstrates how the LLM-based validation evaluates rule compatibility
    with each agent, computing scores and determining which agents should receive
    the rule.
    """
    print("=" * 70)
    print("STEP 2: Validating a New Rule")
    print("=" * 70)
    
    # Get the configuration
    config = get_config()
    
    # Create an LLM client for validation
    llm_client = create_llm_client(config.llm_model)
    
    # Create a rule validator
    validator = RuleValidator(llm_client)
    
    # Example rule to validate
    rule_text = """When discussing medication, always remind users that they should 
    consult with their healthcare provider before making any changes to their 
    medication regimen."""
    
    print(f"Rule to validate: {rule_text}")
    print()
    
    # Validate the rule against all agent prompts
    print("Validating rule against all agents...")
    validation_result: RuleValidation = await validator.validate_rule(
        rule_text=rule_text,
        agent_prompts=config.agent_prompts,
        existing_rules=None,  # No existing rules for this example
    )
    
    # Display validation results
    print(f"\n✓ Validation complete!")
    print(f"  Can be applied: {validation_result.can_be_applied}")
    print(f"  Max compatibility score: {validation_result.max_compatibility_score:.2f}")
    print(f"  Explanation: {validation_result.explanation}")
    print()
    
    # Show which agents will receive this rule
    print(f"Agents that will receive this rule:")
    for agent_info in validation_result.validation_metadata.applied_agents:
        print(f"\n  Agent: {agent_info.agent_name}")
        print(f"    Role consistency: {agent_info.role_consistency_score:.2f}")
        print(f"    Authority expansion: {agent_info.authority_expansion_score:.2f}")
        print(f"    Instruction conflicts: {agent_info.instruction_conflicts_score:.2f}")
        print(f"    Overall score: {agent_info.overall_compatibility_score:.2f}")
        print(f"    Rule to apply: {agent_info.rule_to_apply}")
    
    print()
    return validation_result


# ============================================================================
# Step 4: Cache Management
# ============================================================================

async def demonstrate_caching(validation_result: RuleValidation):
    """
    Demonstrate cache refresh and rule retrieval.
    
    This shows how rules are stored in memory and can be refreshed via webhook
    or manual fetch. In production, the RMS service would call the refresh
    endpoint when rules change.
    """
    print("=" * 70)
    print("STEP 3: Cache Management")
    print("=" * 70)
    
    # Import the global cache instance
    from langgraph_rms.cache import _cache
    
    config = get_config()
    
    # Create a sample cached rule with validation metadata
    cached_rule = CachedRule(
        id="rule-001",
        product_name=config.product_name,
        rule_text="Always remind users to consult healthcare providers about medications.",
        max_length=500,
        risk_level="low",
        status="active",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        latest_validation=validation_result,
    )
    
    # Simulate a cache refresh (normally done via webhook from RMS)
    print("Refreshing cache with new rules...")
    await _cache.refresh_rules(
        product_name=config.product_name,
        rules=[cached_rule],
    )
    
    print(f"✓ Cache refreshed with 1 rule")
    
    # Check last refresh timestamp
    last_refresh = _cache.get_last_refresh(config.product_name)
    print(f"✓ Last refresh: {last_refresh}")
    print()
    
    # Retrieve all active rules
    print("Retrieving all active rules...")
    all_rules = await _cache.get_active_rules(config.product_name)
    print(f"✓ Found {len(all_rules)} active rules")
    print()


# ============================================================================
# Step 5: Agent-Specific Rule Filtering
# ============================================================================

async def demonstrate_rule_filtering():
    """
    Demonstrate retrieving rules for specific agents.
    
    This shows how the package filters rules based on validation metadata,
    returning only rules that are compatible with each agent.
    """
    print("=" * 70)
    print("STEP 4: Agent-Specific Rule Filtering")
    print("=" * 70)
    
    # Get rules for a specific agent
    agent_name = "MedicalAdvisorAgent"
    print(f"Getting rules for {agent_name}...")
    
    rules = await get_rules_for_agent(agent_name=agent_name)
    print(f"✓ Found {len(rules)} rules for {agent_name}")
    
    for i, rule in enumerate(rules, 1):
        print(f"  {i}. {rule}")
    print()
    
    # Get rules for all agents (no filtering)
    print("Getting all rules (no agent filter)...")
    all_rules = await get_rules_for_agent(agent_name=None)
    print(f"✓ Found {len(all_rules)} total rules")
    print()
    
    # Format rules for prompt injection
    print("Formatting rules for prompt injection...")
    formatted = format_rules_for_prompt(rules)
    print("✓ Formatted rules:")
    print(formatted)


# ============================================================================
# Step 6: Automatic Rule Appending
# ============================================================================

async def demonstrate_rule_appending():
    """
    Demonstrate automatic rule appending to agent prompts.
    
    This is the recommended way to use the package - it automatically fetches
    and formats rules for an agent, appending them to the base prompt.
    """
    print("=" * 70)
    print("STEP 5: Automatic Rule Appending")
    print("=" * 70)
    
    # Base system prompt for an agent
    base_prompt = AGENT_PROMPTS["MedicalAdvisorAgent"]
    
    print("Base prompt:")
    print(base_prompt)
    print()
    
    # Method 1: Using the convenience function
    print("Method 1: Using append_rules() convenience function")
    enhanced_prompt = await append_rules(
        base_prompt=base_prompt,
        agent_name="MedicalAdvisorAgent",
    )
    
    print("Enhanced prompt with rules:")
    print(enhanced_prompt)
    print()
    
    # Method 2: Using RulesAppender class
    print("Method 2: Using RulesAppender class")
    appender = RulesAppender()
    enhanced_prompt_2 = await appender.append_rules_to_prompt(
        base_prompt=base_prompt,
        agent_name="MedicalAdvisorAgent",
    )
    
    print("✓ Prompt enhanced with rules")
    print()
    
    # Method 3: Using a wrapper function (cleanest for repeated use)
    print("Method 3: Using create_prompt_wrapper() for reusable wrapper")
    prompt_wrapper = appender.create_prompt_wrapper("MedicalAdvisorAgent")
    enhanced_prompt_3 = await prompt_wrapper(base_prompt)
    
    print("✓ Prompt enhanced using wrapper function")
    print()


# ============================================================================
# Step 7: FastAPI Router Integration
# ============================================================================

def demonstrate_fastapi_integration():
    """
    Demonstrate how to integrate the FastAPI router into your application.
    
    This shows how to add the internal RMS endpoints to your FastAPI app
    for receiving validation requests and cache refresh webhooks.
    """
    print("=" * 70)
    print("STEP 6: FastAPI Router Integration")
    print("=" * 70)
    
    print("To integrate with FastAPI, add this to your application:")
    print()
    print("```python")
    print("from fastapi import FastAPI")
    print("from langgraph_rms import create_router")
    print()
    print("app = FastAPI()")
    print()
    print("# Add the RMS router to your app")
    print("app.include_router(create_router())")
    print("```")
    print()
    print("This adds the following endpoints:")
    print("  POST /internal/rules/validate - Validate a rule")
    print("  POST /internal/rules/refresh  - Refresh cached rules")
    print()
    print("Both endpoints require X-Internal-API-Key header for authentication.")
    print()


# ============================================================================
# Step 8: Custom Formatting
# ============================================================================

async def demonstrate_custom_formatting():
    """
    Demonstrate custom rule formatting.
    
    This shows how to provide a custom formatter function to change how
    rules are formatted for prompt injection.
    """
    print("=" * 70)
    print("STEP 7: Custom Rule Formatting")
    print("=" * 70)
    
    # Define a custom formatter
    def custom_formatter(rules: list) -> str:
        """Custom formatter that uses bullet points instead of numbers."""
        if not rules:
            return ""
        
        formatted = "\n\n=== IMPORTANT RULES ===\n\n"
        for rule in rules:
            formatted += f"• {rule}\n"
        formatted += "\n======================\n"
        return formatted
    
    # Get rules for an agent
    rules = await get_rules_for_agent(agent_name="MedicalAdvisorAgent")
    
    # Format with custom formatter
    print("Using custom formatter:")
    custom_formatted = format_rules_for_prompt(rules, formatter=custom_formatter)
    print(custom_formatted)
    
    # Use custom formatter with append_rules
    base_prompt = AGENT_PROMPTS["MedicalAdvisorAgent"]
    enhanced_prompt = await append_rules(
        base_prompt=base_prompt,
        agent_name="MedicalAdvisorAgent",
        formatter=custom_formatter,
    )
    
    print("✓ Custom formatting applied to prompt")
    print()


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """
    Run the complete integration example.
    
    This demonstrates all major features of the langgraph-rms-integration package
    in a single workflow.
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "LangGraph RMS Integration - Basic Example" + " " * 17 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    try:
        # Step 1: Setup configuration
        setup_configuration()
        
        # Step 2: Validate a rule
        validation_result = await validate_new_rule()
        
        # Step 3: Demonstrate caching
        await demonstrate_caching(validation_result)
        
        # Step 4: Demonstrate rule filtering
        await demonstrate_rule_filtering()
        
        # Step 5: Demonstrate rule appending
        await demonstrate_rule_appending()
        
        # Step 6: Show FastAPI integration
        demonstrate_fastapi_integration()
        
        # Step 7: Demonstrate custom formatting
        await demonstrate_custom_formatting()
        
        print("=" * 70)
        print("✓ Example completed successfully!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Replace the example configuration with your actual values")
        print("  2. Set up environment variables for RMS_URL and RMS_API_KEY")
        print("  3. Integrate the FastAPI router into your application")
        print("  4. Use append_rules() in your agent prompt generation")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
