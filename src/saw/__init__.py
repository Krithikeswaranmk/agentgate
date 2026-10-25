"""SAW: in-process security runtime for LLM agent tool calls.

Pre-release. Tool calls made through ``SAW.execute_tool`` are checked for
registration, arguments and permission before they run, and their results
are checked before they are returned. Calls that need approval wait until
the application approves or denies them. SAW does not limit how long a tool
runs. Calls that bypass SAW are not protected.
"""

from saw.approvals import ApprovalDesk, ApprovalRequest
from saw.audit import DefaultSecurityLogger, SecurityLogger
from saw.calls import CallStatus, ToolCall, ToolCallOutcome
from saw.config import AgentConfig
from saw.events import EventType, SecurityEvent
from saw.exceptions import (
    ApprovalError,
    ApprovalRequiredError,
    ArgumentValidationError,
    AuditError,
    ConfigurationError,
    PermissionDeniedError,
    ResultValidationError,
    SAWError,
    ToolExecutionError,
    ToolNotRegisteredError,
)
from saw.policies import (
    Decision,
    DefaultPermissionPolicy,
    PermissionPolicy,
    PolicyDecision,
)
from saw.registry import RegisteredTool, ToolRegistry
from saw.runtime import SAW
from saw.validation import DefaultResultValidator, ResultValidator, ValidationResult

__version__ = "0.1.0"

__all__ = [
    "SAW",
    "AgentConfig",
    "ToolCall",
    "ToolCallOutcome",
    "ApprovalRequest",
    "ApprovalDesk",
    "CallStatus",
    "ToolRegistry",
    "RegisteredTool",
    # policies
    "PermissionPolicy",
    "DefaultPermissionPolicy",
    "PolicyDecision",
    "Decision",
    # validation
    "ResultValidator",
    "DefaultResultValidator",
    "ValidationResult",
    # audit
    "SecurityLogger",
    "DefaultSecurityLogger",
    "SecurityEvent",
    "EventType",
    # exceptions
    "SAWError",
    "ConfigurationError",
    "ToolNotRegisteredError",
    "ArgumentValidationError",
    "PermissionDeniedError",
    "ApprovalRequiredError",
    "ApprovalError",
    "ResultValidationError",
    "ToolExecutionError",
    "AuditError",
    "__version__",
]
