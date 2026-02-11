"""
LangGraph RMS Integration Package

A standalone Python library that enables any LangGraph-based multi-agent system 
to integrate with a Rules Management System (RMS).
"""

from langgraph_rms.models import (
    AgentRuleInfo,
    ValidationMetadata,
    RuleValidation,
    CachedRule,
    ValidationRequest,
    RefreshRequest,
)
from langgraph_rms.config import (
    RMSConfig,
    initialize,
    get_config,
)
from langgraph_rms.utils import (
    safe_json_parse,
    create_llm_client,
)
from langgraph_rms.prompts import (
    PromptTemplate,
    DEFAULT_VALIDATION_TEMPLATE,
    get_default_template,
    create_custom_template,
)
from langgraph_rms.validator import (
    RuleValidator,
)
from langgraph_rms.cache import (
    get_rules_for_agent,
    format_rules_for_prompt,
    fetch_rules,
)
from langgraph_rms.appender import (
    RulesAppender,
    append_rules,
)
from langgraph_rms.router import (
    create_router,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "AgentRuleInfo",
    "ValidationMetadata",
    "RuleValidation",
    "CachedRule",
    "ValidationRequest",
    "RefreshRequest",
    # Configuration
    "RMSConfig",
    "initialize",
    "get_config",
    # Utilities
    "safe_json_parse",
    "create_llm_client",
    # Prompts
    "PromptTemplate",
    "DEFAULT_VALIDATION_TEMPLATE",
    "get_default_template",
    "create_custom_template",
    # Validator
    "RuleValidator",
    # Cache
    "get_rules_for_agent",
    "format_rules_for_prompt",
    "fetch_rules",
    # Appender
    "RulesAppender",
    "append_rules",
    # Router
    "create_router",
]
