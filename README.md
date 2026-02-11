# LangGraph RMS Integration

A standalone Python library that enables any LangGraph-based multi-agent system to integrate with a Rules Management System (RMS). The package provides LLM-based rule validation, in-memory caching with webhook refresh, agent-specific rule filtering, and secure internal API endpoints.

## Features

- **LLM-Based Rule Validation**: Automatically validates rules against agent prompts using configurable LLM models
- **Smart Rule Filtering**: Applies rules only to compatible agents based on compatibility scores
- **In-Memory Caching**: Fast rule access with webhook-based refresh mechanism
- **Agent-Specific Rules**: Filters and formats rules for specific agents automatically
- **Automatic Prompt Enhancement**: Seamlessly appends rules to agent system prompts
- **Secure API Endpoints**: FastAPI router with API key authentication for RMS communication
- **Type-Safe**: Full Pydantic models with comprehensive type hints
- **Extensible**: Support for custom validation templates, scoring logic, and formatters

## Installation

```bash
pip install langgraph-rms-integration
```

## Quick Start

### 1. Configure the Package

```python
from langgraph_rms import RMSConfig, initialize

# Define your agent system prompts
agent_prompts = {
    "MedicalAdvisor": "You are a medical advisor agent...",
    "Researcher": "You are a research agent...",
}

# Initialize configuration
config = RMSConfig(
    product_name="my-product",
    agent_prompts=agent_prompts,
    rms_url="https://rms.example.com",
    api_key="your-api-key",
    llm_model="gpt-4",
    compatibility_threshold=0.7,
    request_timeout=10.0,
)

initialize(config)
```

### 2. Append Rules to Agent Prompts

```python
from langgraph_rms import append_rules

# Automatically fetch and append rules to an agent's prompt
async def get_agent_prompt():
    base_prompt = "You are a medical advisor agent..."
    enhanced_prompt = await append_rules(base_prompt, "MedicalAdvisor")
    return enhanced_prompt
```

### 3. Integrate API Endpoints

```python
from fastapi import FastAPI
from langgraph_rms import create_router

app = FastAPI()

# Add RMS internal endpoints
rms_router = create_router()
app.include_router(rms_router)

# Your RMS service can now call:
# POST /internal/rules/validate - Validate new rules
# POST /internal/rules/refresh - Refresh cached rules
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `product_name` | `str` | *Required* | Name of your product using this integration |
| `agent_prompts` | `Dict[str, str]` | *Required* | Dictionary mapping agent names to their system prompts |
| `rms_url` | `str` | *Required* | Base URL of your RMS service |
| `api_key` | `str` | *Required* | API key for authenticating with RMS |
| `llm_model` | `str` | `"gpt-4"` | LLM model name for rule validation |
| `compatibility_threshold` | `float` | `0.7` | Minimum compatibility score (0.0-1.0) for applying rules to agents |
| `request_timeout` | `float` | `10.0` | HTTP request timeout in seconds |

## Usage Examples

### Rule Validation

The package automatically validates rules against your agent prompts when the RMS service calls the validation endpoint:

```python
# RMS service calls: POST /internal/rules/validate
# Request body:
{
    "rule_text": "Always verify patient allergies before recommendations",
    "product_name": "my-product"
}

# Response includes compatibility scores for each agent:
{
    "can_be_applied": true,
    "max_compatibility_score": 0.85,
    "explanation": "Rule is compatible with MedicalAdvisor agent",
    "validation_metadata": {
        "applied_agents": [
            {
                "agent_name": "MedicalAdvisor",
                "role_consistency_score": 0.9,
                "authority_expansion_score": 0.8,
                "instruction_conflicts_score": 0.85,
                "overall_compatibility_score": 0.85,
                "analysis": "Rule aligns well with medical advisor role...",
                "concerns": [],
                "rule_to_apply": "Always verify patient allergies before making recommendations."
            }
        ]
    }
}
```

### Cache Refresh

The RMS service can refresh cached rules via webhook:

```python
# RMS service calls: POST /internal/rules/refresh
# Request body:
{
    "rules": [
        {
            "id": "rule-123",
            "product_name": "my-product",
            "rule_text": "Always verify patient allergies",
            "max_length": 500,
            "risk_level": "high",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "latest_validation": { ... }
        }
    ]
}
```

### Manual Rule Retrieval

```python
from langgraph_rms import get_rules_for_agent, format_rules_for_prompt

# Get rules for a specific agent
rules = await get_rules_for_agent(agent_name="MedicalAdvisor")
# Returns: ["Always verify patient allergies...", "Check medication interactions..."]

# Get all rules (no filtering)
all_rules = await get_rules_for_agent()

# Format rules for prompt injection
formatted = format_rules_for_prompt(rules)
# Returns formatted string ready to append to agent prompt
```

### Using Rules Appender Class

```python
from langgraph_rms import RulesAppender

# Create appender instance
appender = RulesAppender(product_name="my-product")

# Append rules to prompt
enhanced_prompt = await appender.append_rules_to_prompt(
    base_prompt="You are a medical advisor...",
    agent_name="MedicalAdvisor"
)

# Create reusable wrapper function
prompt_wrapper = appender.create_prompt_wrapper("MedicalAdvisor")
enhanced_prompt = await prompt_wrapper("You are a medical advisor...")
```

### Custom Validation Templates

```python
from langgraph_rms import create_custom_template, RuleValidator, create_llm_client

# Create custom validation prompt template
custom_template = create_custom_template("""
Evaluate this rule: {rule_text}
Against these agents: {agent_prompts}
Existing rules: {existing_rules}
""")

# Use custom template in validator
llm = create_llm_client()
validator = RuleValidator(llm, prompt_template=custom_template)
```

### Custom Rule Formatting

```python
from langgraph_rms import format_rules_for_prompt

# Define custom formatter
def my_formatter(rules: list[str]) -> str:
    return "\n".join(f"• {rule}" for rule in rules)

# Use custom formatter
formatted = format_rules_for_prompt(rules, formatter=my_formatter)
```

## RMS API Contract

Your RMS service should implement the following contract when communicating with this package:

### Authentication

All requests to internal endpoints must include the API key header:

```
X-Internal-API-Key: your-api-key
```

### Validation Endpoint

**POST** `/internal/rules/validate`

Request:
```json
{
    "rule_text": "string",
    "product_name": "string"
}
```

Response:
```json
{
    "can_be_applied": boolean,
    "max_compatibility_score": float,
    "explanation": "string",
    "validation_metadata": {
        "applied_agents": [
            {
                "agent_name": "string",
                "role_consistency_score": float,
                "authority_expansion_score": float,
                "instruction_conflicts_score": float,
                "overall_compatibility_score": float,
                "analysis": "string",
                "concerns": ["string"],
                "rule_to_apply": "string"
            }
        ]
    }
}
```

### Refresh Endpoint

**POST** `/internal/rules/refresh`

Request:
```json
{
    "rules": [
        {
            "id": "string",
            "product_name": "string",
            "rule_text": "string",
            "max_length": integer,
            "risk_level": "string",
            "status": "string",
            "created_at": "ISO 8601 datetime",
            "updated_at": "ISO 8601 datetime",
            "latest_validation": { ... }
        }
    ]
}
```

Response:
```json
{
    "message": "Rules refreshed successfully",
    "count": integer
}
```

## Error Handling

The package handles errors gracefully:

- **Configuration Errors**: Raised at initialization with descriptive messages
- **RMS Communication Failures**: Logged as warnings, cached data continues to be used
- **LLM Validation Failures**: Returns safe default response indicating failure
- **Invalid API Keys**: Returns HTTP 401 Unauthorized
- **Malformed Requests**: Returns HTTP 400 with validation details

## Advanced Features

### Custom Scoring Logic

```python
from langgraph_rms import RuleValidator, create_llm_client

def custom_scorer(validation_result):
    # Modify scores based on custom logic
    for agent in validation_result.validation_metadata.applied_agents:
        # Custom scoring logic here
        pass
    return validation_result

llm = create_llm_client()
validator = RuleValidator(llm)
result = await validator.validate_rule(
    rule_text="...",
    agent_prompts={...},
    scoring_callback=custom_scorer
)
```

### Thread-Safe Cache Operations

The package uses asyncio locks for thread-safe cache operations:

```python
from langgraph_rms.cache import _cache

# All cache operations are automatically thread-safe
rules = await _cache.get_active_rules("my-product")
await _cache.refresh_rules("my-product", new_rules)
```

## Requirements

- Python 3.9 or higher
- pydantic >= 2.0
- fastapi >= 0.100
- httpx >= 0.24
- langchain-core >= 0.1

## Development

Install development dependencies:

```bash
pip install langgraph-rms-integration[dev]
```

Run tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=langgraph_rms --cov-report=html
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the [GitHub issue tracker](https://github.com/langgraph-rms/langgraph-rms-integration/issues).
