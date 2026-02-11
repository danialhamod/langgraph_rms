"""
Unit tests for utility functions.
"""

import pytest
from langgraph_rms.utils import safe_json_parse, create_llm_client


class TestSafeJsonParse:
    """Tests for safe_json_parse function."""
    
    def test_parse_plain_json(self):
        """Test parsing plain JSON without code blocks."""
        content = '{"key": "value", "number": 42}'
        result = safe_json_parse(content)
        assert result == {"key": "value", "number": 42}
    
    def test_parse_json_with_whitespace(self):
        """Test parsing JSON with leading/trailing whitespace."""
        content = '  \n  {"key": "value"}  \n  '
        result = safe_json_parse(content)
        assert result == {"key": "value"}
    
    def test_parse_markdown_code_block_lowercase(self):
        """Test parsing JSON from markdown code block with lowercase json."""
        content = '''```json
{
    "key": "value",
    "nested": {
        "field": 123
    }
}
```'''
        result = safe_json_parse(content)
        assert result == {"key": "value", "nested": {"field": 123}}
    
    def test_parse_markdown_code_block_uppercase(self):
        """Test parsing JSON from markdown code block with uppercase JSON."""
        content = '''```JSON
{"key": "value"}
```'''
        result = safe_json_parse(content)
        assert result == {"key": "value"}
    
    def test_parse_generic_code_block(self):
        """Test parsing JSON from generic code block without language identifier."""
        content = '''```
{"key": "value", "array": [1, 2, 3]}
```'''
        result = safe_json_parse(content)
        assert result == {"key": "value", "array": [1, 2, 3]}
    
    def test_parse_json_with_text_before_code_block(self):
        """Test parsing JSON when there's text before the code block."""
        content = '''Here is the JSON response:
```json
{"status": "success"}
```'''
        result = safe_json_parse(content)
        assert result == {"status": "success"}
    
    def test_parse_json_with_text_after_code_block(self):
        """Test parsing JSON when there's text after the code block."""
        content = '''```json
{"status": "success"}
```
This is the result.'''
        result = safe_json_parse(content)
        assert result == {"status": "success"}
    
    def test_parse_complex_nested_json(self):
        """Test parsing complex nested JSON structure."""
        content = '''{
    "validation": {
        "can_be_applied": true,
        "scores": [0.8, 0.9, 0.7],
        "metadata": {
            "agents": ["agent1", "agent2"]
        }
    }
}'''
        result = safe_json_parse(content)
        assert result["validation"]["can_be_applied"] is True
        assert result["validation"]["scores"] == [0.8, 0.9, 0.7]
        assert result["validation"]["metadata"]["agents"] == ["agent1", "agent2"]
    
    def test_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="Content is empty"):
            safe_json_parse("")
    
    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only content raises ValueError."""
        with pytest.raises(ValueError, match="Content is empty"):
            safe_json_parse("   \n  \t  ")
    
    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            safe_json_parse("{invalid json}")
    
    def test_non_dict_json_raises_error(self):
        """Test that non-dictionary JSON raises ValueError."""
        with pytest.raises(ValueError, match="Expected JSON object"):
            safe_json_parse('["array", "not", "dict"]')
    
    def test_json_with_special_characters(self):
        """Test parsing JSON with special characters."""
        content = '{"message": "Line 1\\nLine 2\\tTabbed", "quote": "He said \\"hello\\""}'
        result = safe_json_parse(content)
        assert result["message"] == "Line 1\nLine 2\tTabbed"
        assert result["quote"] == 'He said "hello"'
    
    def test_json_with_unicode(self):
        """Test parsing JSON with unicode characters."""
        content = '{"arabic": "مرحبا", "emoji": "😀", "chinese": "你好"}'
        result = safe_json_parse(content)
        assert result["arabic"] == "مرحبا"
        assert result["emoji"] == "😀"
        assert result["chinese"] == "你好"


class TestCreateLlmClient:
    """Tests for create_llm_client function."""
    
    def test_create_client_with_valid_model(self, monkeypatch):
        """Test creating LLM client with valid model name."""
        # Set a fake API key for testing
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        
        try:
            client = create_llm_client()
            assert client is not None
            assert hasattr(client, "model_name")
            assert client.model_name == "gpt-4"
            assert client.temperature == 0.0
        except ImportError:
            pytest.skip("langchain_openai not installed")
    
    def test_create_client_with_gpt35(self, monkeypatch):
        """Test creating LLM client with gpt-3.5-turbo."""
        # Set a fake API key for testing
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        
        try:
            client = create_llm_client("gpt-3.5-turbo")
            assert client is not None
            assert client.model_name == "gpt-3.5-turbo"
        except ImportError:
            pytest.skip("langchain_openai not installed")
    
    def test_empty_model_name_raises_error(self):
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            create_llm_client("")
    
    def test_whitespace_model_name_raises_error(self):
        """Test that whitespace-only model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            create_llm_client("   ")
    
    def test_missing_langchain_raises_import_error(self, monkeypatch):
        """Test that missing langchain_openai raises ImportError with helpful message."""
        # Mock the import to simulate missing package
        import sys
        import builtins
        
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == "langchain_openai":
                raise ImportError("No module named 'langchain_openai'")
            return original_import(name, *args, **kwargs)
        
        monkeypatch.setattr(builtins, "__import__", mock_import)
        
        with pytest.raises(ImportError, match="langchain_openai is required"):
            create_llm_client()
