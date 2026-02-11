# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-15

### Added

- Initial release of langgraph-rms-integration package
- **Configuration Management**: `RMSConfig` class for product-agnostic configuration with validation
- **Data Models**: Pydantic models for type-safe data structures (`CachedRule`, `RuleValidation`, `ValidationMetadata`, `AgentRuleInfo`)
- **Rule Validation**: LLM-based rule validation against agent system prompts with compatibility scoring
  - Three-dimensional scoring: role consistency, authority expansion, and instruction conflicts
  - Configurable compatibility threshold (default 0.7)
  - Automatic rule translation to English for agent application
  - Conflict detection with existing active rules
- **Rule Caching**: Thread-safe in-memory cache with webhook-based refresh
  - Product-specific rule storage
  - Timestamp tracking for cache freshness
  - Graceful error handling with cache preservation on failures
- **Agent-Specific Filtering**: Retrieve rules filtered by agent compatibility
  - Automatic filtering based on validation metadata
  - Backward compatibility for rules without metadata
  - Prompt-ready rule formatting
- **Rules Appender**: Automatic rule injection into agent system prompts
  - Async operation support
  - Wrapper function factory for LangGraph integration
  - Graceful error handling
- **Internal API Endpoints**: FastAPI router with secure endpoints
  - POST `/internal/rules/validate` for rule validation requests
  - POST `/internal/rules/refresh` for cache refresh webhooks
  - API key authentication via `X-Internal-API-Key` header
- **Extensibility Features**: Customization support for advanced use cases
  - Custom validation prompt templates
  - Custom scoring callbacks
  - Custom rule formatting functions
- **Error Handling**: Comprehensive error handling with graceful degradation
  - HTTP request timeouts
  - LLM failure resilience
  - Configuration validation
- **Documentation**: Complete README with installation, quick start, and configuration reference
- **Examples**: Working integration example demonstrating all features
- **Testing Support**: Comprehensive test fixtures and pytest configuration

### Requirements

- Python 3.9 or higher
- Dependencies: pydantic>=2.0, fastapi>=0.100, httpx>=0.24, langchain-core>=0.1

[0.1.0]: https://github.com/langgraph-rms/langgraph-rms-integration/releases/tag/v0.1.0
