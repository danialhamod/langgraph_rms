"""
Utility functions for LangGraph RMS Integration.

This module provides helper functions for JSON parsing, LLM client creation,
and other common operations.
"""

import json
import re
from typing import Any, Dict


def safe_json_parse(content: str) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM response.
    
    Handles common LLM response formats including:
    - Markdown code blocks (```json ... ```)
    - Plain JSON responses
    - JSON with surrounding whitespace
    
    Args:
        content: Raw LLM response content
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If JSON cannot be parsed

    Example:
        >>> from langgraph_rms import safe_json_parse
        >>> content = '```json\\n{"key": "value"}\\n```'
        >>> result = safe_json_parse(content)
        >>> print(result)
        {'key': 'value'}
    """
    if not content or not content.strip():
        raise ValueError("Content is empty or contains only whitespace")
    
    # Remove leading/trailing whitespace
    content = content.strip()
    
    # Check for markdown code blocks with json language identifier
    # Pattern: ```json ... ``` or ```JSON ... ```
    json_block_pattern = r'```(?:json|JSON)\s*\n(.*?)\n```'
    match = re.search(json_block_pattern, content, re.DOTALL)
    
    if match:
        # Extract JSON from markdown code block
        json_str = match.group(1).strip()
    else:
        # Check for generic code blocks without language identifier
        # Pattern: ``` ... ```
        generic_block_pattern = r'```\s*\n(.*?)\n```'
        match = re.search(generic_block_pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()
        else:
            # No code blocks found, treat entire content as JSON
            json_str = content
    
    # Try to parse the JSON
    try:
        parsed = json.loads(json_str)
        
        # Ensure we return a dictionary
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object (dict), got {type(parsed).__name__}")
        
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}. Content: {json_str[:200]}...")


def create_llm_client(model_name: str = 'openai/gpt-oss-120b', api_key: str = 'KEY', base_url: str = 'https://api.together.xyz/v1', temperature: float = 0.0) -> Any:
    """
    Create LangChain LLM client for validation.
    
    This function creates a LangChain ChatOpenAI instance configured
    for rule validation. It uses the OpenAI API with the specified model.
    
    Args:
        model_name: Name of the LLM model (e.g., "gpt-4", "gpt-3.5-turbo")
        api_key: Provider API Key
        base-url: Provider Base Url
        temperature: Temperature value for model
        
    Returns:
        Configured LangChain LLM instance
        
    Raises:
        ImportError: If langchain_openai is not installed
        ValueError: If model_name is empty

    Example:
        >>> from langgraph_rms import create_llm_client
        >>> llm = create_llm_client("gpt-4", ...)
        >>> # Use llm for validation
    """
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain_openai is required for LLM client creation. "
            "Install it with: pip install langchain-openai"
        )
    
    if not api_key:
        raise ValueError(f"No API key found for provider '{provider}'")

    # Create and return ChatOpenAI instance
    llm_kwargs = {
        "model": model_name,
        "api_key": api_key,
        "temperature": temperature,
    }

    if base_url:
        llm_kwargs["base_url"] = base_url

    return ChatOpenAI(**llm_kwargs)
