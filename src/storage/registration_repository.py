"""
Registration Repository
FT-QuPAP v5.1

This module stores non-secret registration and trust-setup records for
the FT-QuPAP authentication protocol.

It represents protocol Phase 0:

1. The Authentication Server owns a long-term ML-DSA-65 credential.
2. The operator provisions the server public key as a Mobile Station
   trust anchor.
3. The operator registers a pseudonymous subscriber identity.
4. The registration record identifies permitted channel contexts and
   protocol-policy versions.

Security rules:
    - Never store the ML-DSA private key.
    - Never store an ML-KEM private key or shared secret.
    - Never store K_auth or K_ctrl.
    - Never store the subscriber's raw IMSI.
    - Store only a trust-anchor fingerprint and reference in this file.
    - The public trust-anchor key belongs in trusted_server_keys.json.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REGISTRATION_DATABASE_PATH = Path(
    "database/registration_records.json"
)

DEFAULT_TRUST_ANCHOR_DATABASE_PATH = Path(
    "database/trusted_server_keys.json"
)

DEFAULT_PROTOCOL_VERSION = "FT-QuPAP-v5.1"
DEFAULT_SERVER_ID = "AS-6G-001"
DEFAULT_TRUST_ALGORITHM = "ML-DSA-65"
DEFAULT_POLICY_VERSION = 1

SUPPORTED_SUBSCRIBER_STATUSES = {
    "active",
    "suspended",
    "revoked",
}

SUPPORTED_CONTEXTS = {
    "urban",
    "suburban",
    "rural",
}


def current_utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


@dataclass
class RegistrationRecord:
    """
    Persistent FT-QuPAP subscriber-registration record.

    Attributes:
        registration_id:
            Unique identifier for this registration record.

        pseudonym_id:
            Operator-managed pseudonymous subscriber identity.

        subscriber_status:
            Current subscriber authorization state.

        registered_contexts:
            Channel contexts in which the subscriber may authenticate.

        server_id:
            Authentication Server associated with this registration.

        trust_algorithm:
            Digital-signature algorithm used by the server trust anchor.

        trust_anchor_version:
            Version of the trusted server public key.

        trust_anchor_fingerprint:
            SHA3-256 fingerprint of the trusted server public key.

        trust_anchor_reference:
            Reference to the corresponding public-key entry in
            trusted_server_keys.json.

        protocol_version:
            FT-QuPAP protocol version assigned during registration.

        policy_version:
            Security-policy version assigned during registration.

        registered_at:
            UTC registration time.

        updated_at:
            UTC timestamp of the latest record update.

        suspended_at:
            UTC suspension time, when applicable.

        revoked_at:
            UTC revocation time, when applicable.

        metadata:
            Optional non-secret registration metadata.

        event_history:
            Structured registration and trust-rotation history.
    """

    registration_id: str
    pseudonym_id: str

    subscriber_status: str = "active"

    registered_contexts: list[str] = field(
        default_factory=lambda: [
            "urban",
            "suburban",
            "rural",
        ]
    )

    server_id: str = DEFAULT_SERVER_ID
    trust_algorithm: str = DEFAULT_TRUST_ALGORITHM
    trust_anchor_version: int = 1
    trust_anchor_fingerprint: str = ""
    trust_anchor_reference: str = ""

    protocol_version: str = DEFAULT_PROTOCOL_VERSION
    policy_version: int = DEFAULT_POLICY_VERSION

    registered_at: str = field(
        default_factory=current_utc_timestamp
    )
    updated_at: str = field(
        default_factory=current_utc_timestamp
    )

    suspended_at: str | None = None
    revoked_at: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
    event_history: list[dict[str, Any]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        """Normalize and validate registration fields."""

        self.registration_id = normalize_required_string(
            "registration_id",
            self.registration_id,
        )

        self.pseudonym_id = normalize_required_string(
            "pseudonym_id",
            self.pseudonym_id,
        )

        self.subscriber_status = validate_subscriber_status(
            self.subscriber_status
        )

        self.registered_contexts = normalize_contexts(
            self.registered_contexts
        )

        self.server_id = normalize_required_string(
            "server_id",
            self.server_id,
        )

        self.trust_algorithm = normalize_required_string(
            "trust_algorithm",
            self.trust_algorithm,
        )

        validate_positive_integer(
            "trust_anchor_version",
            self.trust_anchor_version,
        )

        self.trust_anchor_fingerprint = validate_fingerprint(
            self.trust_anchor_fingerprint
        )

        self.trust_anchor_reference = normalize_required_string(
            "trust_anchor_reference",
            self.trust_anchor_reference,
        )

        self.protocol_version = normalize_required_string(
            "protocol_version",
            self.protocol_version,
        )

        validate_positive_integer(
            "policy_version",
            self.policy_version,
        )

        self.registered_at = normalize_timestamp(
            self.registered_at
        )

        self.updated_at = normalize_timestamp(
            self.updated_at
        )

        self.suspended_at = normalize_optional_timestamp(
            self.suspended_at
        )

        self.revoked_at = normalize_optional_timestamp(
            self.revoked_at
        )

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        if not isinstance(self.event_history, list):
            raise TypeError("event_history must be a list.")

        if (
            self.subscriber_status == "suspended"
            and self.suspended_at is None
        ):
            self.suspended_at = self.updated_at

        if (
            self.subscriber_status == "revoked"
            and self.revoked_at is None
        ):
            self.revoked_at = self.updated_at

    @property
    def is_active(self) -> bool:
        """Return True when the registration is active."""

        return self.subscriber_status == "active"

    @property
    def is_suspended(self) -> bool:
        """Return True when the registration is suspended."""

        return self.subscriber_status == "suspended"

    @property
    def is_revoked(self) -> bool:
        """Return True when the registration is revoked."""

        return self.subscriber_status == "revoked"

    def permits_context(self, context: str) -> bool:
        """Check whether a channel context is registered."""

        normalized_context = validate_context(context)

        return normalized_context in self.registered_contexts

    def add_event(
        self,
        event_type: str,
        details: dict[str, Any] | None = None,
        timestamp: datetime | str | None = None,
    ) -> None:
        """Append a non-secret registration event."""

        normalized_event_type = normalize_required_string(
            "event_type",
            event_type,
        )

        event_timestamp = (
            current_utc_timestamp()
            if timestamp is None
            else normalize_timestamp(timestamp)
        )

        self.event_history.append(
            {
                "event_type": normalized_event_type,
                "timestamp": event_timestamp,
                "subscriber_status": self.subscriber_status,
                "trust_anchor_version":
                    self.trust_anchor_version,
                "policy_version": self.policy_version,
                "details": copy.deepcopy(details or {}),
            }
        )

    def to_dictionary(self) -> dict[str, Any]:
        """Convert the registration record into JSON-safe data."""

        return asdict(self)

    @classmethod
    def from_dictionary(
        cls,
        data: dict[str, Any],
    ) -> "RegistrationRecord":
        """Create a registration record from stored JSON data."""

        if not isinstance(data, dict):
            raise TypeError(
                "Registration data must be a dictionary."
            )

        required_fields = {
            "registration_id",
            "pseudonym_id",
            "trust_anchor_fingerprint",
            "trust_anchor_reference",
        }

        missing_fields = required_fields.difference(data)

        if missing_fields:
            raise ValueError(
                "Registration record is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )

        return cls(
            registration_id=str(data["registration_id"]),
            pseudonym_id=str(data["pseudonym_id"]),
            subscriber_status=str(
                data.get("subscriber_status", "active")
            ),
            registered_contexts=list(
                data.get(
                    "registered_contexts",
                    ["urban", "suburban", "rural"],
                )
            ),
            server_id=str(
                data.get("server_id", DEFAULT_SERVER_ID)
            ),
            trust_algorithm=str(
                data.get(
                    "trust_algorithm",
                    DEFAULT_TRUST_ALGORITHM,
                )
            ),
            trust_anchor_version=int(
                data.get("trust_anchor_version", 1)
            ),
            trust_anchor_fingerprint=str(
                data["trust_anchor_fingerprint"]
            ),
            trust_anchor_reference=str(
                data["trust_anchor_reference"]
            ),
            protocol_version=str(
                data.get(
                    "protocol_version",
                    DEFAULT_PROTOCOL_VERSION,
                )
            ),
            policy_version=int(
                data.get(
                    "policy_version",
                    DEFAULT_POLICY_VERSION,
                )
            ),
            registered_at=str(
                data.get(
                    "registered_at",
                    current_utc_timestamp(),
                )
            ),
            updated_at=str(
                data.get(
                    "updated_at",
                    current_utc_timestamp(),
                )
            ),
            suspended_at=data.get("suspended_at"),
            revoked_at=data.get("revoked_at"),
            metadata=dict(data.get("metadata", {})),
            event_history=list(
                data.get("event_history", [])
            ),
        )


class RegistrationRepository:
    """
    Thread-safe JSON repository for FT-QuPAP registrations.

    The repository maintains two unique indexes:

        registration_id -> RegistrationRecord
        pseudonym_id    -> registration_id

    The raw subscriber identity and private cryptographic material are
    intentionally excluded.
    """

    DATABASE_VERSION = 1

    def __init__(
        self,
        database_path: str | Path = (
            DEFAULT_REGISTRATION_DATABASE_PATH
        ),
    ) -> None:
        """
        Initialize the registration repository.

        Args:
            database_path:
                Path to registration_records.json.
        """

        self._database_path = Path(database_path)
        self._lock = threading.RLock()

        self._records_by_id: dict[
            str,
            RegistrationRecord,
        ] = {}

        self._registration_id_by_pseudonym: dict[
            str,
            str,
        ] = {}

        self._initialize_repository()

    @property
    def database_path(self) -> Path:
        """Return the repository file path."""

        return self._database_path

    def _initialize_repository(self) -> None:
        """Create the repository or load existing registrations."""

        with self._lock:
            self._database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if self._database_path.exists():
                self._load()
            else:
                self._save()

    def _load(self) -> None:
        """Load registration records from JSON storage."""

        with self._lock:
            try:
                with self._database_path.open(
                    "r",
                    encoding="utf-8",
                ) as database_file:
                    stored_data = json.load(database_file)

            except json.JSONDecodeError as error:
                raise ValueError(
                    "Registration repository contains invalid JSON."
                ) from error

            except OSError as error:
                raise OSError(
                    "Unable to read the registration repository."
                ) from error

            if not isinstance(stored_data, dict):
                raise ValueError(
                    "Registration repository root must be "
                    "a JSON object."
                )

            database_version = stored_data.get(
                "version",
                self.DATABASE_VERSION,
            )

            if database_version != self.DATABASE_VERSION:
                raise ValueError(
                    "Unsupported registration database version: "
                    f"{database_version}"
                )

            raw_records = stored_data.get(
                "registration_records",
                [],
            )

            if not isinstance(raw_records, list):
                raise ValueError(
                    "'registration_records' must be a JSON array."
                )

            records_by_id: dict[
                str,
                RegistrationRecord,
            ] = {}

            registration_id_by_pseudonym: dict[
                str,
                str,
            ] = {}

            for raw_record in raw_records:
                record = RegistrationRecord.from_dictionary(
                    raw_record
                )

                if record.registration_id in records_by_id:
                    raise ValueError(
                        "Duplicate registration_id detected: "
                        f"{record.registration_id}"
                    )

                if (
                    record.pseudonym_id
                    in registration_id_by_pseudonym
                ):
                    raise ValueError(
                        "Duplicate pseudonym_id detected: "
                        f"{record.pseudonym_id}"
                    )

                records_by_id[
                    record.registration_id
                ] = record

                registration_id_by_pseudonym[
                    record.pseudonym_id
                ] = record.registration_id

            self._records_by_id = records_by_id
            self._registration_id_by_pseudonym = (
                registration_id_by_pseudonym
            )

    def reload(self) -> None:
        """Reload all registration records from disk."""

        self._load()

    def _save(self) -> None:
        """
        Persist registration records through atomic replacement.
        """

        with self._lock:
            repository_content = {
                "version": self.DATABASE_VERSION,
                "protocol": DEFAULT_PROTOCOL_VERSION,
                "registration_records": [
                    record.to_dictionary()
                    for record in sorted(
                        self._records_by_id.values(),
                        key=lambda item: (
                            item.registered_at,
                            item.registration_id,
                        ),
                    )
                ],
            }

            temporary_file_path: Path | None = None

            try:
                file_descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f"{self._database_path.name}.",
                    suffix=".tmp",
                    dir=str(self._database_path.parent),
                )

                temporary_file_path = Path(temporary_name)

                with os.fdopen(
                    file_descriptor,
                    "w",
                    encoding="utf-8",
                ) as temporary_file:
                    json.dump(
                        repository_content,
                        temporary_file,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    )

                    temporary_file.write("\n")
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())

                os.replace(
                    temporary_file_path,
                    self._database_path,
                )

            except OSError as error:
                if (
                    temporary_file_path is not None
                    and temporary_file_path.exists()
                ):
                    temporary_file_path.unlink(
                        missing_ok=True
                    )

                raise OSError(
                    "Unable to save the registration repository."
                ) from error

    def register_subscriber(
        self,
        pseudonym_id: str,
        trust_anchor_public_key: bytes,
        registered_contexts: Iterable[str] = (
            "urban",
            "suburban",
            "rural",
        ),
        server_id: str = DEFAULT_SERVER_ID,
        trust_algorithm: str = DEFAULT_TRUST_ALGORITHM,
        trust_anchor_version: int = 1,
        trust_anchor_reference: str | None = None,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        policy_version: int = DEFAULT_POLICY_VERSION,
        registration_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RegistrationRecord:
        """
        Register a pseudonymous FT-QuPAP subscriber.

        The supplied public key is used only to calculate its SHA3-256
        fingerprint. The complete key is not stored in this repository.

        Args:
            pseudonym_id:
                Operator-assigned pseudonymous subscriber reference.

            trust_anchor_public_key:
                AS ML-DSA-65 public key provisioned into the MS.

            registered_contexts:
                Allowed urban, suburban, and rural contexts.

            server_id:
                Authentication Server identity.

            trust_algorithm:
                Expected trust-anchor signature algorithm.

            trust_anchor_version:
                Version number of the server trust anchor.

            trust_anchor_reference:
                Location of the complete trusted public-key record.

            protocol_version:
                FT-QuPAP protocol version.

            policy_version:
                Deterministic security-policy version.

            registration_id:
                Optional externally supplied registration identifier.

            metadata:
                Optional non-secret information.

        Returns:
            Newly created RegistrationRecord.
        """

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        normalized_server_id = normalize_required_string(
            "server_id",
            server_id,
        )

        normalized_algorithm = normalize_required_string(
            "trust_algorithm",
            trust_algorithm,
        )

        validate_positive_integer(
            "trust_anchor_version",
            trust_anchor_version,
        )

        validate_positive_integer(
            "policy_version",
            policy_version,
        )

        public_key_fingerprint = fingerprint_public_key(
            trust_anchor_public_key
        )

        normalized_reference = (
            normalize_required_string(
                "trust_anchor_reference",
                trust_anchor_reference,
            )
            if trust_anchor_reference is not None
            else build_trust_anchor_reference(
                server_id=normalized_server_id,
                trust_anchor_version=trust_anchor_version,
            )
        )

        generated_registration_id = (
            normalize_required_string(
                "registration_id",
                registration_id,
            )
            if registration_id is not None
            else generate_registration_id()
        )

        record = RegistrationRecord(
            registration_id=generated_registration_id,
            pseudonym_id=normalized_pseudonym,
            subscriber_status="active",
            registered_contexts=list(
                registered_contexts
            ),
            server_id=normalized_server_id,
            trust_algorithm=normalized_algorithm,
            trust_anchor_version=trust_anchor_version,
            trust_anchor_fingerprint=(
                public_key_fingerprint
            ),
            trust_anchor_reference=normalized_reference,
            protocol_version=protocol_version,
            policy_version=policy_version,
            metadata=copy.deepcopy(metadata or {}),
        )

        record.add_event(
            event_type="subscriber_registered",
            details={
                "server_id": record.server_id,
                "trust_algorithm": record.trust_algorithm,
                "trust_anchor_version":
                    record.trust_anchor_version,
                "trust_anchor_fingerprint":
                    record.trust_anchor_fingerprint,
                "registered_contexts":
                    list(record.registered_contexts),
                "protocol_version":
                    record.protocol_version,
                "policy_version":
                    record.policy_version,
            },
            timestamp=record.registered_at,
        )

        with self._lock:
            if (
                record.registration_id
                in self._records_by_id
            ):
                raise ValueError(
                    "Registration already exists: "
                    f"{record.registration_id}"
                )

            if (
                record.pseudonym_id
                in self._registration_id_by_pseudonym
            ):
                raise ValueError(
                    "Pseudonymous identity is already registered: "
                    f"{record.pseudonym_id}"
                )

            self._insert_into_memory(record)

            try:
                self._save()
            except Exception:
                self._remove_from_memory(record)
                raise

            return record

    def _insert_into_memory(
        self,
        record: RegistrationRecord,
    ) -> None:
        """Insert a record into both in-memory indexes."""

        self._records_by_id[
            record.registration_id
        ] = record

        self._registration_id_by_pseudonym[
            record.pseudonym_id
        ] = record.registration_id

    def _remove_from_memory(
        self,
        record: RegistrationRecord,
    ) -> None:
        """Remove a record from both indexes."""

        self._records_by_id.pop(
            record.registration_id,
            None,
        )

        self._registration_id_by_pseudonym.pop(
            record.pseudonym_id,
            None,
        )

    def get_by_registration_id(
        self,
        registration_id: str,
    ) -> RegistrationRecord | None:
        """Return a registration using its identifier."""

        normalized_id = normalize_required_string(
            "registration_id",
            registration_id,
        )

        with self._lock:
            return self._records_by_id.get(normalized_id)

    def get_by_pseudonym(
        self,
        pseudonym_id: str,
    ) -> RegistrationRecord | None:
        """Resolve a pseudonymous subscriber identity."""

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        with self._lock:
            registration_id = (
                self._registration_id_by_pseudonym.get(
                    normalized_pseudonym
                )
            )

            if registration_id is None:
                return None

            return self._records_by_id.get(
                registration_id
            )

    def require_by_registration_id(
        self,
        registration_id: str,
    ) -> RegistrationRecord:
        """Return a registration or raise LookupError."""

        record = self.get_by_registration_id(
            registration_id
        )

        if record is None:
            raise LookupError(
                f"Registration '{registration_id}' "
                "was not found."
            )

        return record

    def require_by_pseudonym(
        self,
        pseudonym_id: str,
    ) -> RegistrationRecord:
        """Resolve a pseudonym or raise LookupError."""

        record = self.get_by_pseudonym(pseudonym_id)

        if record is None:
            raise LookupError(
                "The pseudonymous identity is not registered."
            )

        return record

    def registration_exists(
        self,
        registration_id: str,
    ) -> bool:
        """Return True when a registration ID exists."""

        return (
            self.get_by_registration_id(registration_id)
            is not None
        )

    def pseudonym_exists(
        self,
        pseudonym_id: str,
    ) -> bool:
        """Return True when a pseudonym is registered."""

        return self.get_by_pseudonym(pseudonym_id) is not None

    def verify_registration(
        self,
        pseudonym_id: str,
        context: str,
        server_id: str = DEFAULT_SERVER_ID,
        trust_algorithm: str = DEFAULT_TRUST_ALGORITHM,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    ) -> bool:
        """
        Perform deterministic registration-policy verification.

        This check verifies that:

        - The pseudonymous identity exists
        - The subscriber is active
        - The requested context is registered
        - The expected Authentication Server matches
        - The trust-anchor algorithm matches
        - The protocol version matches

        It does not perform ML-DSA signature verification.
        """

        record = self.get_by_pseudonym(pseudonym_id)

        if record is None:
            return False

        normalized_context = validate_context(context)

        return all(
            (
                record.is_active,
                normalized_context
                in record.registered_contexts,
                record.server_id
                == normalize_required_string(
                    "server_id",
                    server_id,
                ),
                record.trust_algorithm
                == normalize_required_string(
                    "trust_algorithm",
                    trust_algorithm,
                ),
                record.protocol_version
                == normalize_required_string(
                    "protocol_version",
                    protocol_version,
                ),
            )
        )

    def verify_trust_anchor(
        self,
        pseudonym_id: str,
        public_key: bytes,
        expected_version: int | None = None,
    ) -> bool:
        """
        Compare a supplied server public key with the registered
        trust-anchor fingerprint.
        """

        record = self.get_by_pseudonym(pseudonym_id)

        if record is None:
            return False

        if expected_version is not None:
            validate_positive_integer(
                "expected_version",
                expected_version,
            )

            if (
                record.trust_anchor_version
                != expected_version
            ):
                return False

        supplied_fingerprint = fingerprint_public_key(
            public_key
        )

        return constant_time_text_compare(
            record.trust_anchor_fingerprint,
            supplied_fingerprint,
        )

    def update_status(
        self,
        registration_id: str,
        new_status: str,
        reason: str | None = None,
    ) -> RegistrationRecord:
        """Update active, suspended, or revoked state."""

        normalized_status = validate_subscriber_status(
            new_status
        )

        normalized_reason = (
            normalize_required_string("reason", reason)
            if reason is not None
            else None
        )

        def update(record: RegistrationRecord) -> None:
            previous_status = record.subscriber_status
            event_time = current_utc_timestamp()

            record.subscriber_status = normalized_status
            record.updated_at = event_time

            if normalized_status == "active":
                record.suspended_at = None

            elif normalized_status == "suspended":
                record.suspended_at = event_time

            elif normalized_status == "revoked":
                record.revoked_at = event_time

            record.add_event(
                event_type="subscriber_status_updated",
                details={
                    "previous_status": previous_status,
                    "new_status": normalized_status,
                    "reason": normalized_reason,
                },
                timestamp=event_time,
            )

        return self._update_record(
            registration_id,
            update,
        )

    def activate(
        self,
        registration_id: str,
        reason: str | None = None,
    ) -> RegistrationRecord:
        """Activate a subscriber registration."""

        return self.update_status(
            registration_id=registration_id,
            new_status="active",
            reason=reason,
        )

    def suspend(
        self,
        registration_id: str,
        reason: str | None = None,
    ) -> RegistrationRecord:
        """Temporarily suspend a registration."""

        return self.update_status(
            registration_id=registration_id,
            new_status="suspended",
            reason=reason,
        )

    def revoke(
        self,
        registration_id: str,
        reason: str | None = None,
    ) -> RegistrationRecord:
        """Permanently revoke a registration."""

        return self.update_status(
            registration_id=registration_id,
            new_status="revoked",
            reason=reason,
        )

    def update_registered_contexts(
        self,
        registration_id: str,
        registered_contexts: Iterable[str],
    ) -> RegistrationRecord:
        """Replace the subscriber's permitted contexts."""

        normalized_contexts = normalize_contexts(
            registered_contexts
        )

        def update(record: RegistrationRecord) -> None:
            previous_contexts = list(
                record.registered_contexts
            )

            record.registered_contexts = (
                normalized_contexts
            )

            record.updated_at = current_utc_timestamp()

            record.add_event(
                event_type="registered_contexts_updated",
                details={
                    "previous_contexts": previous_contexts,
                    "new_contexts": normalized_contexts,
                },
                timestamp=record.updated_at,
            )

        return self._update_record(
            registration_id,
            update,
        )

    def rotate_trust_anchor(
        self,
        registration_id: str,
        new_public_key: bytes,
        new_version: int,
        trust_anchor_reference: str | None = None,
        trust_algorithm: str = DEFAULT_TRUST_ALGORITHM,
    ) -> RegistrationRecord:
        """
        Update the registration after an operator-authorized trust-anchor
        rotation.

        The complete public key remains in trusted_server_keys.json.
        """

        validate_positive_integer(
            "new_version",
            new_version,
        )

        new_fingerprint = fingerprint_public_key(
            new_public_key
        )

        normalized_algorithm = normalize_required_string(
            "trust_algorithm",
            trust_algorithm,
        )

        def update(record: RegistrationRecord) -> None:
            if new_version <= record.trust_anchor_version:
                raise ValueError(
                    "new_version must be greater than the "
                    "current trust-anchor version."
                )

            previous_version = (
                record.trust_anchor_version
            )

            previous_fingerprint = (
                record.trust_anchor_fingerprint
            )

            record.trust_algorithm = normalized_algorithm
            record.trust_anchor_version = new_version
            record.trust_anchor_fingerprint = (
                new_fingerprint
            )

            record.trust_anchor_reference = (
                normalize_required_string(
                    "trust_anchor_reference",
                    trust_anchor_reference,
                )
                if trust_anchor_reference is not None
                else build_trust_anchor_reference(
                    server_id=record.server_id,
                    trust_anchor_version=new_version,
                )
            )

            record.updated_at = current_utc_timestamp()

            record.add_event(
                event_type="trust_anchor_rotated",
                details={
                    "previous_version":
                        previous_version,
                    "new_version": new_version,
                    "previous_fingerprint":
                        previous_fingerprint,
                    "new_fingerprint":
                        new_fingerprint,
                    "trust_algorithm":
                        normalized_algorithm,
                },
                timestamp=record.updated_at,
            )

        return self._update_record(
            registration_id,
            update,
        )

    def update_policy_version(
        self,
        registration_id: str,
        new_policy_version: int,
    ) -> RegistrationRecord:
        """Assign a new security-policy version."""

        validate_positive_integer(
            "new_policy_version",
            new_policy_version,
        )

        def update(record: RegistrationRecord) -> None:
            previous_policy_version = (
                record.policy_version
            )

            record.policy_version = new_policy_version
            record.updated_at = current_utc_timestamp()

            record.add_event(
                event_type="policy_version_updated",
                details={
                    "previous_policy_version":
                        previous_policy_version,
                    "new_policy_version":
                        new_policy_version,
                },
                timestamp=record.updated_at,
            )

        return self._update_record(
            registration_id,
            update,
        )

    def update_metadata(
        self,
        registration_id: str,
        metadata_updates: dict[str, Any],
    ) -> RegistrationRecord:
        """Merge non-secret metadata into a registration."""

        if not isinstance(metadata_updates, dict):
            raise TypeError(
                "metadata_updates must be a dictionary."
            )

        def update(record: RegistrationRecord) -> None:
            record.metadata.update(
                copy.deepcopy(metadata_updates)
            )

            record.updated_at = current_utc_timestamp()

            record.add_event(
                event_type="registration_metadata_updated",
                details={
                    "updated_keys": sorted(
                        str(key)
                        for key in metadata_updates.keys()
                    )
                },
                timestamp=record.updated_at,
            )

        return self._update_record(
            registration_id,
            update,
        )

    def list_records(
        self,
        subscriber_status: str | None = None,
        context: str | None = None,
        server_id: str | None = None,
    ) -> list[RegistrationRecord]:
        """List registration records using optional filters."""

        normalized_status = (
            validate_subscriber_status(subscriber_status)
            if subscriber_status is not None
            else None
        )

        normalized_context = (
            validate_context(context)
            if context is not None
            else None
        )

        normalized_server_id = (
            normalize_required_string(
                "server_id",
                server_id,
            )
            if server_id is not None
            else None
        )

        with self._lock:
            records: Iterable[RegistrationRecord] = (
                self._records_by_id.values()
            )

            if normalized_status is not None:
                records = (
                    record
                    for record in records
                    if (
                        record.subscriber_status
                        == normalized_status
                    )
                )

            if normalized_context is not None:
                records = (
                    record
                    for record in records
                    if (
                        normalized_context
                        in record.registered_contexts
                    )
                )

            if normalized_server_id is not None:
                records = (
                    record
                    for record in records
                    if record.server_id
                    == normalized_server_id
                )

            return sorted(
                records,
                key=lambda record: (
                    record.registered_at,
                    record.registration_id,
                ),
            )

    def count_records(
        self,
        subscriber_status: str | None = None,
    ) -> int:
        """Return the number of registration records."""

        return len(
            self.list_records(
                subscriber_status=subscriber_status
            )
        )

    def delete_registration(
        self,
        registration_id: str,
    ) -> bool:
        """
        Delete a registration record.

        Revocation is preferred because deletion removes historical
        registration and trust-rotation evidence.
        """

        normalized_id = normalize_required_string(
            "registration_id",
            registration_id,
        )

        with self._lock:
            record = self._records_by_id.get(
                normalized_id
            )

            if record is None:
                return False

            self._remove_from_memory(record)

            try:
                self._save()
            except Exception:
                self._insert_into_memory(record)
                raise

            return True

    def clear(self) -> None:
        """
        Remove all registration records.

        Intended for automated tests and controlled demo resets.
        """

        with self._lock:
            previous_records = dict(
                self._records_by_id
            )

            previous_pseudonym_index = dict(
                self._registration_id_by_pseudonym
            )

            self._records_by_id.clear()
            self._registration_id_by_pseudonym.clear()

            try:
                self._save()
            except Exception:
                self._records_by_id = previous_records
                self._registration_id_by_pseudonym = (
                    previous_pseudonym_index
                )
                raise

    def _update_record(
        self,
        registration_id: str,
        update_function: Any,
    ) -> RegistrationRecord:
        """
        Apply an update and restore the previous record on failure.
        """

        normalized_id = normalize_required_string(
            "registration_id",
            registration_id,
        )

        with self._lock:
            record = self.require_by_registration_id(
                normalized_id
            )

            previous_record = copy.deepcopy(record)

            try:
                update_function(record)
                record.__post_init__()
                self._save()

            except Exception:
                self._records_by_id[
                    normalized_id
                ] = previous_record

                self._registration_id_by_pseudonym[
                    previous_record.pseudonym_id
                ] = normalized_id

                raise

            return record


def generate_registration_id() -> str:
    """Generate a unique registration identifier."""

    return f"REG-{uuid.uuid4().hex.upper()}"


def fingerprint_public_key(
    public_key: bytes,
) -> str:
    """Return a SHA3-256 fingerprint of a public key."""

    if not isinstance(public_key, bytes):
        raise TypeError(
            "trust_anchor_public_key must be bytes."
        )

    if not public_key:
        raise ValueError(
            "trust_anchor_public_key cannot be empty."
        )

    return hashlib.sha3_256(public_key).hexdigest()


def build_trust_anchor_reference(
    server_id: str,
    trust_anchor_version: int,
) -> str:
    """Build a reference to trusted_server_keys.json."""

    normalized_server_id = normalize_required_string(
        "server_id",
        server_id,
    )

    validate_positive_integer(
        "trust_anchor_version",
        trust_anchor_version,
    )

    return (
        f"{DEFAULT_TRUST_ANCHOR_DATABASE_PATH}"
        f"#{normalized_server_id}:v{trust_anchor_version}"
    )


def constant_time_text_compare(
    first_value: str,
    second_value: str,
) -> bool:
    """Compare text values using constant-time byte comparison."""

    import hmac

    return hmac.compare_digest(
        first_value.encode("utf-8"),
        second_value.encode("utf-8"),
    )


def validate_fingerprint(
    fingerprint: str,
) -> str:
    """Validate a SHA3-256 hexadecimal fingerprint."""

    normalized_fingerprint = normalize_required_string(
        "trust_anchor_fingerprint",
        fingerprint,
    ).lower()

    if len(normalized_fingerprint) != 64:
        raise ValueError(
            "trust_anchor_fingerprint must contain "
            "64 hexadecimal characters."
        )

    try:
        bytes.fromhex(normalized_fingerprint)
    except ValueError as error:
        raise ValueError(
            "trust_anchor_fingerprint must be valid hexadecimal."
        ) from error

    return normalized_fingerprint


def normalize_contexts(
    contexts: Iterable[str],
) -> list[str]:
    """Validate and normalize registered channel contexts."""

    if isinstance(contexts, (str, bytes)):
        raise TypeError(
            "registered_contexts must be an iterable of strings."
        )

    try:
        normalized_contexts = {
            validate_context(context)
            for context in contexts
        }
    except TypeError as error:
        raise TypeError(
            "registered_contexts must be iterable."
        ) from error

    if not normalized_contexts:
        raise ValueError(
            "At least one registered context is required."
        )

    return sorted(normalized_contexts)


def validate_context(context: str) -> str:
    """Validate an urban, suburban, or rural context."""

    normalized_context = normalize_required_string(
        "context",
        context,
    ).lower()

    if normalized_context not in SUPPORTED_CONTEXTS:
        raise ValueError(
            "context must be one of: "
            + ", ".join(sorted(SUPPORTED_CONTEXTS))
        )

    return normalized_context


def validate_subscriber_status(status: str) -> str:
    """Validate subscriber registration status."""

    normalized_status = normalize_required_string(
        "subscriber_status",
        status,
    ).lower()

    if (
        normalized_status
        not in SUPPORTED_SUBSCRIBER_STATUSES
    ):
        raise ValueError(
            "subscriber_status must be one of: "
            + ", ".join(
                sorted(SUPPORTED_SUBSCRIBER_STATUSES)
            )
        )

    return normalized_status


def normalize_required_string(
    name: str,
    value: str,
) -> str:
    """Validate and normalize a required string."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(f"{name} cannot be empty.")

    return normalized_value


def validate_positive_integer(
    name: str,
    value: int,
) -> None:
    """Validate an integer greater than zero."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )


def normalize_timestamp(
    value: datetime | str,
) -> str:
    """Convert a datetime or ISO string into UTC ISO format."""

    if isinstance(value, str):
        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("Timestamp cannot be empty.")

        if normalized_value.endswith("Z"):
            normalized_value = (
                normalized_value[:-1] + "+00:00"
            )

        try:
            parsed_datetime = datetime.fromisoformat(
                normalized_value
            )
        except ValueError as error:
            raise ValueError(
                "Timestamp must use ISO 8601 format."
            ) from error

    elif isinstance(value, datetime):
        parsed_datetime = value

    else:
        raise TypeError(
            "Timestamp must be a datetime or ISO string."
        )

    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(
            tzinfo=timezone.utc
        )

    return parsed_datetime.astimezone(
        timezone.utc
    ).isoformat()


def normalize_optional_timestamp(
    value: datetime | str | None,
) -> str | None:
    """Normalize an optional timestamp."""

    if value is None:
        return None

    return normalize_timestamp(value)


def run_self_test() -> None:
    """Run registration and trust-anchor storage tests."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = (
            Path(temporary_directory)
            / "registration_records.json"
        )

        repository = RegistrationRepository(
            database_path
        )

        first_public_key = bytes.fromhex(
            "00112233445566778899aabbccddeeff"
            "102132435465768798a9bacbdcedfe0f"
        )

        registration = repository.register_subscriber(
            registration_id="REG-0001",
            pseudonym_id="PID-6G-UE-0001",
            trust_anchor_public_key=first_public_key,
            registered_contexts=[
                "urban",
                "suburban",
                "rural",
            ],
            server_id="AS-6G-001",
            trust_algorithm="ML-DSA-65",
            trust_anchor_version=1,
            protocol_version="FT-QuPAP-v5.1",
            policy_version=1,
            metadata={
                "registration_mode":
                    "operator_provisioned",
            },
        )

        assert registration.is_active
        assert repository.count_records() == 1

        assert repository.verify_registration(
            pseudonym_id="PID-6G-UE-0001",
            context="urban",
            server_id="AS-6G-001",
            trust_algorithm="ML-DSA-65",
            protocol_version="FT-QuPAP-v5.1",
        )

        assert repository.verify_trust_anchor(
            pseudonym_id="PID-6G-UE-0001",
            public_key=first_public_key,
            expected_version=1,
        )

        repository.suspend(
            registration_id="REG-0001",
            reason="Controlled suspension test.",
        )

        assert not repository.verify_registration(
            pseudonym_id="PID-6G-UE-0001",
            context="urban",
        )

        repository.activate(
            registration_id="REG-0001",
            reason="Controlled reactivation test.",
        )

        second_public_key = bytes.fromhex(
            "ffeeddccbbaa99887766554433221100"
            "0ffedcba98765432100123456789abcd"
        )

        repository.rotate_trust_anchor(
            registration_id="REG-0001",
            new_public_key=second_public_key,
            new_version=2,
        )

        assert not repository.verify_trust_anchor(
            pseudonym_id="PID-6G-UE-0001",
            public_key=first_public_key,
        )

        assert repository.verify_trust_anchor(
            pseudonym_id="PID-6G-UE-0001",
            public_key=second_public_key,
            expected_version=2,
        )

        stored_text = database_path.read_text(
            encoding="utf-8"
        )

        assert first_public_key.hex() not in stored_text
        assert second_public_key.hex() not in stored_text
        assert "trust_anchor_fingerprint" in stored_text

        reloaded_repository = RegistrationRepository(
            database_path
        )

        reloaded_record = (
            reloaded_repository.require_by_pseudonym(
                "PID-6G-UE-0001"
            )
        )

        assert reloaded_record.is_active
        assert reloaded_record.trust_anchor_version == 2
        assert len(reloaded_record.event_history) >= 4

        print("Registration repository self-test passed.")
        print(f"Database path: {database_path}")
        print(
            "Registered pseudonym: "
            f"{reloaded_record.pseudonym_id}"
        )
        print(
            "Trust-anchor version: "
            f"{reloaded_record.trust_anchor_version}"
        )


if __name__ == "__main__":
    run_self_test()