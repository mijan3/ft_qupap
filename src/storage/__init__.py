"""
FT-QuPAP Storage Package
========================

Persistent and export-oriented storage components for the
FT-QuPAP v5.1 capstone project.

The storage package supports:

- Pseudonymous subscriber registration
- Authentication-request replay protection
- Complete protocol-session tracking
- Registration and trust-setup records
- Authentication-result persistence
- CSV logging for dashboards and experiments

Storage files:

    database/subscribers.json
    database/used_nonces.json
    database/demo_sessions.json
    database/registration_records.json
    data/demo/dashboard_results.csv
    data/demo/demo_session_logs.csv

Security requirements:

The storage layer must never persist:

- Raw IMSI values
- ML-DSA private keys
- ML-KEM private keys
- ML-KEM shared secrets
- Derived K_auth or K_ctrl keys
- Raw authentication nonces
- Raw KMAC authentication tags
- Raw ML-KEM ciphertexts
- Decrypted control schedules
- Quantum state vectors
- Confidential authentication payloads

Only pseudonymous identities, public metadata, cryptographic digests,
protocol measurements, deterministic decisions, GP outputs, retry
evidence, and timing information may be stored.
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
PROTOCOL_VERSION = "FT-QuPAP-v5.1"


__all__ = [
    # Package metadata
    "STORAGE_PACKAGE_VERSION",
    "PROTOCOL_VERSION",

    # Subscriber database
    "SubscriberDatabase",
    "SubscriberRecord",

    # Nonce and replay protection
    "NonceDatabase",
    "NonceRecord",
    "NonceReplayError",

    # Protocol-session database
    "SessionDatabase",
    "SessionRecord",

    # Registration and trust setup
    "RegistrationRepository",
    "RegistrationRecord",

    # Authentication results
    "ResultRepository",
    "AuthenticationResultRecord",

    # CSV logging
    "CSVLogger",
]