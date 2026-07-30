"""
Common utilities for the FT-QuPAP v5.1 project.

This package contains shared components used by:

- Mobile Station modules
- Authentication Server modules
- Cryptographic modules
- Quantum simulation modules
- Machine-learning modules
- Dashboard and storage modules
"""

from .constants import (
    FEATURE_COLUMNS,
    PROTOCOL_CONFIG,
    APPLICATION_CONFIG,
)

from .enums import (
    AuthenticationDecision,
    ChannelContext,
    ProtocolStageStatus,
)

from .exceptions import (
    FTQuPAPError,
    ConfigurationError,
    CryptographicError,
    ProtocolValidationError,
    ReplayAttackError,
    ModelNotGeneratedError,
)

__all__ = [
    "FEATURE_COLUMNS",
    "PROTOCOL_CONFIG",
    "APPLICATION_CONFIG",
    "AuthenticationDecision",
    "ChannelContext",
    "ProtocolStageStatus",
    "FTQuPAPError",
    "ConfigurationError",
    "CryptographicError",
    "ProtocolValidationError",
    "ReplayAttackError",
    "ModelNotGeneratedError",
]