# LangGraph RMS Integration Examples

This directory contains example code demonstrating how to use the `langgraph-rms-integration` package.

## Examples

### basic_integration.py

A complete working example that demonstrates all major features of the package:

1. **Configuration Setup** - How to initialize the package with your agents and RMS service
2. **Rule Validation** - How to validate rules against agent prompts using LLM
3. **Cache Management** - How rules are cached and refreshed
4. **Rule Filtering** - How to retrieve rules for specific agents
5. **Rule Appending** - How to automatically append rules to agent prompts
6. **FastAPI Integration** - How to add internal RMS endpoints to your app
7. **Custom Formatting** - How to customize rule formatting

## Running the Examples

### Prerequisites

1. Install the package:
   ```bash
   pip install langgraph-rms-integration
   ```

2. Set up environment variables (optional):
   ```bash
   export RMS_URL="https://your-rms-service.com"
   export RMS_API_KEY="your-api-key"
   export LLM_MODEL="gpt-4"
   ```

### Run the Basic Integration Example

```bash
python examples/basic_integration.py
```

**Note:** The example includes mock data and will work without a real RMS service. However, the LLM validation step requires valid OpenAI API credentials.

## Example Output

When you run `basic_integration.py`, you'll see output demonstrating each step:

```
╔====================================================================╗
║          LangGraph RMS Integration - Basic Example                ║
╚====================================================================╝

======================================================================
STEP 1: Initializing Configuration
======================================================================
✓ Configuration initialized for product: example_medical_app
✓ Configured 3 agents
✓ Compatibility threshold: 0.7
✓ LLM model: gpt-4

======================================================================
STEP 2: Validating a New Rule
======================================================================
Rule to validate: When discussing medication, always remind users...
...
```

## Adapting for Your Project

To use this package in your own project:

1. **Replace agent prompts** with your actual agent system prompts
2. **Configure RMS connection** with your RMS service URL and API key
3. **Integrate the router** into your FastAPI application
4. **Use `append_rules()`** in your agent prompt generation logic

Example integration in your agent code:

```python
from langgraph_rms import append_rules

async def get_agent_prompt(agent_name: str) -> str:
    base_prompt = get_base_prompt_for_agent(agent_name)
    return await append_rules(base_prompt, agent_name)
```

## Additional Resources

- [Package README](../README.md) - Full documentation
- [API Reference](../README.md#api-reference) - Detailed API documentation
- [Configuration Guide](../README.md#configuration) - Configuration options

## Troubleshooting

### "Configuration not initialized" Error

Make sure to call `initialize(config)` before using any other package functions:

```python
from langgraph_rms import RMSConfig, initialize

config = RMSConfig(...)
initialize(config)
```

### LLM Validation Errors

Ensure you have valid OpenAI API credentials set up:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Cache Not Updating

The cache is updated via webhook from the RMS service. For testing, you can manually refresh:

```python
from langgraph_rms.cache import _cache

await _cache.fetch_from_rms("your-product-name")
```

## Contributing

If you have additional examples or improvements, please submit a pull request!
