"""
FT-QuPAP Storage Package
========================

Persistent storage components for FT-QuPAP v5.1.

The storage package manages:

- Registered subscriber information
- Used nonce digests for replay protection
- Authentication-session lifecycle records
- Registration and trust-setup records
- Authentication and experiment results
- CSV files used by the dashboard and evaluation pipeline

Security-sensitive values must not be stored here, including:

- ML-DSA private keys
- ML-KEM private keys
- ML-KEM shared secrets
- Derived K_auth and K_ctrl session keys
- Raw authentication nonces
- Raw confidential authentication payloads
- Quantum state vectors

Only validated protocol metadata, cryptographic digests, measurements,
decisions, and non-secret public information should be persisted.
"""

from __future__ import annotations

from .csv_logger import CSVLogger
from .nonce_database import (
    NonceDatabase,
    NonceRecord,
    NonceReplayError,
)
from .registration_repository import (
    RegistrationRecord,
    RegistrationRepository,
)
from .result_repository import (
    AuthenticationResultRecord,
    ResultRepository,
)
from .session_database import (
    SessionDatabase,
    SessionRecord,
)
from .subscriber_database import (
    SubscriberDatabase,
    SubscriberRecord,
)


STORAGE_PACKAGE_VERSION = "5.1.0"


__all__ = [
    # Package metadata
    "STORAGE_PACKAGE_VERSION",

    # Subscriber storage
    "SubscriberDatabase",
    "SubscriberRecord",

    # Replay-protection storage
    "NonceDatabase",
    "NonceRecord",
    "NonceReplayError",

    # Authentication-session storage
    "SessionDatabase",
    "SessionRecord",

    # Registration and trust-setup storage
    "RegistrationRepository",
    "RegistrationRecord",

    # Authentication-result storage
    "ResultRepository",
    "AuthenticationResultRecord",

    # CSV experiment and dashboard logging
    "CSVLogger",
]