"""
Authentication Server Subscriber Database
FT-QuPAP v5.1

This module stores the pseudonymous subscriber records used by the
Authentication Server during FT-QuPAP authentication.

Notebook-aligned subscriber record:

    {
        "pseudonym_id": "PID-6G-UE-0001",
        "subscriber_status": "active",
        "registered_contexts": [
            "urban",
            "suburban",
            "rural",
        ],
    }

Security requirements:

- Never store or transport a raw IMSI.
- Never store ML-DSA or ML-KEM private keys.
- Never store ML-KEM shared secrets.
- Never store K_auth or K_ctrl.
- Never store raw nonces or KMAC tags.
- Use the operator-managed pseudonym as the subscriber lookup key.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


DEFAULT_SUBSCRIBER_DATABASE_PATH = Path(
    "database/subscribers.json"
)

PROTOCOL_VERSION = "FT-QuPAP-v5.1"
DATABASE_VERSION = 1

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

SENSITIVE_METADATA_FIELDS = {
    "imsi",
    "raw_imsi",
    "private_key",
    "secret_key",
    "ml_dsa_private_key",
    "mldsa_private_key",
    "ml_kem_private_key",
    "mlkem_private_key",
    "shared_secret",
    "session_secret",
    "k_auth",
    "k_ctrl",
    "authentication_key",
    "control_key",
    "nonce",
    "raw_nonce",
    "kmac_tag",
    "authentication_tag",
    "ciphertext",
    "quantum_state",
    "statevector",
}


def current_utc_timestamp() -> str:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass
class SubscriberRecord:
    """
    FT-QuPAP pseudonymous subscriber record.

    Protocol fields:
        pseudonym_id:
            Operator-managed pseudonymous subscriber reference.

        subscriber_status:
            Authorization state: active, suspended, or revoked.

        registered_contexts:
            Service/channel contexts in which the subscriber may
            authenticate.

    Storage metadata:
        created_at:
            UTC timestamp at which the record was created.

        updated_at:
            UTC timestamp of the most recent update.

        metadata:
            Optional non-secret operator metadata.

    The notebook-facing representation contains only:

        pseudonym_id
        subscriber_status
        registered_contexts
    """

    pseudonym_id: str
    subscriber_status: str = "active"

    registered_contexts: list[str] = field(
        default_factory=lambda: [
            "urban",
            "suburban",
            "rural",
        ]
    )

    created_at: str = field(
        default_factory=current_utc_timestamp
    )

    updated_at: str = field(
        default_factory=current_utc_timestamp
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate the subscriber record."""

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

        self.created_at = normalize_timestamp(
            self.created_at
        )

        self.updated_at = normalize_timestamp(
            self.updated_at
        )

        self.metadata = validate_metadata(
            self.metadata
        )

    @property
    def is_active(self) -> bool:
        """Return True when authentication is permitted."""

        return self.subscriber_status == "active"

    @property
    def is_suspended(self) -> bool:
        """Return True when authentication is temporarily blocked."""

        return self.subscriber_status == "suspended"

    @property
    def is_revoked(self) -> bool:
        """Return True when the registration has been revoked."""

        return self.subscriber_status == "revoked"

    def permits_context(self, context: str) -> bool:
        """Return True when the context is registered."""

        normalized_context = validate_context(context)

        return normalized_context in self.registered_contexts

    def to_notebook_record(self) -> dict[str, Any]:
        """
        Return the exact subscriber structure used by the notebook.
        """

        return {
            "pseudonym_id": self.pseudonym_id,
            "subscriber_status": self.subscriber_status,
            "registered_contexts": list(
                self.registered_contexts
            ),
        }

    def to_dictionary(self) -> dict[str, Any]:
        """Return the complete persistent representation."""

        return asdict(self)

    @classmethod
    def from_dictionary(
        cls,
        data: Mapping[str, Any],
    ) -> "SubscriberRecord":
        """Create a subscriber record from JSON-compatible data."""

        if not isinstance(data, Mapping):
            raise TypeError(
                "Subscriber data must be a mapping."
            )

        if "pseudonym_id" not in data:
            raise ValueError(
                "Subscriber record is missing pseudonym_id."
            )

        return cls(
            pseudonym_id=str(data["pseudonym_id"]),
            subscriber_status=str(
                data.get(
                    "subscriber_status",
                    "active",
                )
            ),
            registered_contexts=list(
                data.get(
                    "registered_contexts",
                    [
                        "urban",
                        "suburban",
                        "rural",
                    ],
                )
            ),
            created_at=str(
                data.get(
                    "created_at",
                    current_utc_timestamp(),
                )
            ),
            updated_at=str(
                data.get(
                    "updated_at",
                    current_utc_timestamp(),
                )
            ),
            metadata=dict(
                data.get("metadata", {})
            ),
        )


class SubscriberDatabase:
    """
    Thread-safe JSON-backed FT-QuPAP subscriber database.

    The in-memory structure follows the notebook:

        AS_SUBSCRIBER_DB[pseudonym_id] = subscriber_record

    The class also implements basic mapping behavior so it can be used
    with notebook-aligned code such as:

        if request["pseudonym_id"] not in subscriber_db:
            return False, "unknown_pseudonym"
    """

    def __init__(
        self,
        database_path: str | Path = (
            DEFAULT_SUBSCRIBER_DATABASE_PATH
        ),
    ) -> None:
        """Initialize the subscriber database."""

        self._database_path = Path(database_path)
        self._lock = threading.RLock()

        self._subscribers: dict[
            str,
            SubscriberRecord,
        ] = {}

        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the JSON database path."""

        return self._database_path

    def _initialize_database(self) -> None:
        """Create a new database or load the existing file."""

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
        """Load subscriber records from JSON storage."""

        with self._lock:
            try:
                with self._database_path.open(
                    "r",
                    encoding="utf-8",
                ) as database_file:
                    stored_data = json.load(database_file)

            except json.JSONDecodeError as error:
                raise ValueError(
                    "Subscriber database contains invalid JSON."
                ) from error

            except OSError as error:
                raise OSError(
                    "Unable to read the subscriber database."
                ) from error

            if not isinstance(stored_data, dict):
                raise ValueError(
                    "Subscriber database root must be "
                    "a JSON object."
                )

            version = stored_data.get(
                "version",
                DATABASE_VERSION,
            )

            if version != DATABASE_VERSION:
                raise ValueError(
                    "Unsupported subscriber database version: "
                    f"{version}"
                )

            raw_subscribers = stored_data.get(
                "subscribers",
                {},
            )

            if not isinstance(raw_subscribers, dict):
                raise ValueError(
                    "'subscribers' must be a JSON object "
                    "indexed by pseudonym_id."
                )

            loaded_subscribers: dict[
                str,
                SubscriberRecord,
            ] = {}

            for pseudonym_id, raw_record in (
                raw_subscribers.items()
            ):
                if not isinstance(raw_record, dict):
                    raise ValueError(
                        "Each subscriber value must be "
                        "a JSON object."
                    )

                record_data = dict(raw_record)

                record_data.setdefault(
                    "pseudonym_id",
                    pseudonym_id,
                )

                record = SubscriberRecord.from_dictionary(
                    record_data
                )

                if record.pseudonym_id != pseudonym_id:
                    raise ValueError(
                        "Subscriber dictionary key does not "
                        "match record pseudonym_id."
                    )

                if record.pseudonym_id in loaded_subscribers:
                    raise ValueError(
                        "Duplicate subscriber pseudonym: "
                        f"{record.pseudonym_id}"
                    )

                loaded_subscribers[
                    record.pseudonym_id
                ] = record

            self._subscribers = loaded_subscribers

    def reload(self) -> None:
        """Reload all subscriber records from disk."""

        self._load()

    def _save(self) -> None:
        """Persist records through atomic file replacement."""

        with self._lock:
            database_content = {
                "version": DATABASE_VERSION,
                "protocol": PROTOCOL_VERSION,
                "subscribers": {
                    pseudonym_id:
                        record.to_dictionary()
                    for pseudonym_id, record
                    in sorted(
                        self._subscribers.items(),
                        key=lambda item: item[0],
                    )
                },
            }

            temporary_path: Path | None = None

            try:
                file_descriptor, temporary_name = (
                    tempfile.mkstemp(
                        prefix=(
                            f"{self._database_path.name}."
                        ),
                        suffix=".tmp",
                        dir=str(
                            self._database_path.parent
                        ),
                    )
                )

                temporary_path = Path(temporary_name)

                with os.fdopen(
                    file_descriptor,
                    "w",
                    encoding="utf-8",
                ) as temporary_file:
                    json.dump(
                        database_content,
                        temporary_file,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    )

                    temporary_file.write("\n")
                    temporary_file.flush()
                    os.fsync(
                        temporary_file.fileno()
                    )

                os.replace(
                    temporary_path,
                    self._database_path,
                )

            except OSError as error:
                if (
                    temporary_path is not None
                    and temporary_path.exists()
                ):
                    temporary_path.unlink(
                        missing_ok=True
                    )

                raise OSError(
                    "Unable to save the subscriber database."
                ) from error

    def register_subscriber(
        self,
        pseudonym_id: str,
        subscriber_status: str = "active",
        registered_contexts: Iterable[str] = (
            "urban",
            "suburban",
            "rural",
        ),
        metadata: Mapping[str, Any] | None = None,
    ) -> SubscriberRecord:
        """
        Register a new pseudonymous subscriber.

        Raises:
            ValueError:
                When the pseudonym is already registered.
        """

        record = SubscriberRecord(
            pseudonym_id=pseudonym_id,
            subscriber_status=subscriber_status,
            registered_contexts=list(
                registered_contexts
            ),
            metadata=dict(metadata or {}),
        )

        with self._lock:
            if record.pseudonym_id in self._subscribers:
                raise ValueError(
                    "Pseudonym is already registered: "
                    f"{record.pseudonym_id}"
                )

            self._subscribers[
                record.pseudonym_id
            ] = record

            try:
                self._save()
            except Exception:
                self._subscribers.pop(
                    record.pseudonym_id,
                    None,
                )
                raise

            return copy.deepcopy(record)

    def get_subscriber(
        self,
        pseudonym_id: str,
    ) -> SubscriberRecord | None:
        """Return a subscriber by pseudonymous identity."""

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        with self._lock:
            record = self._subscribers.get(
                normalized_pseudonym
            )

            return (
                copy.deepcopy(record)
                if record is not None
                else None
            )

    def require_subscriber(
        self,
        pseudonym_id: str,
    ) -> SubscriberRecord:
        """Return a subscriber or raise LookupError."""

        record = self.get_subscriber(pseudonym_id)

        if record is None:
            raise LookupError(
                "Unknown FT-QuPAP pseudonym: "
                f"{pseudonym_id}"
            )

        return record

    def subscriber_exists(
        self,
        pseudonym_id: str,
    ) -> bool:
        """Return True when the pseudonym is registered."""

        try:
            normalized_pseudonym = (
                normalize_required_string(
                    "pseudonym_id",
                    pseudonym_id,
                )
            )
        except (TypeError, ValueError):
            return False

        with self._lock:
            return (
                normalized_pseudonym
                in self._subscribers
            )

    def verify_subscriber(
        self,
        pseudonym_id: str,
        service_context: str,
    ) -> tuple[bool, str]:
        """
        Perform deterministic subscriber authorization checks.

        Returns:
            (True, "subscriber_valid")
                The pseudonym exists, is active, and permits the
                requested service context.

            (False, "unknown_pseudonym")
                The pseudonym is not registered.

            (False, "inactive_subscriber")
                The subscriber is suspended or revoked.

            (False, "unregistered_context")
                The requested context is not permitted.
        """

        record = self.get_subscriber(pseudonym_id)

        if record is None:
            return False, "unknown_pseudonym"

        if not record.is_active:
            return False, "inactive_subscriber"

        try:
            normalized_context = validate_context(
                service_context
            )
        except (TypeError, ValueError):
            return False, "unsupported_service_context"

        if not record.permits_context(
            normalized_context
        ):
            return False, "unregistered_context"

        return True, "subscriber_valid"

    def update_status(
        self,
        pseudonym_id: str,
        subscriber_status: str,
    ) -> SubscriberRecord:
        """Update subscriber authorization status."""

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        normalized_status = validate_subscriber_status(
            subscriber_status
        )

        with self._lock:
            record = self._subscribers.get(
                normalized_pseudonym
            )

            if record is None:
                raise LookupError(
                    "Unknown FT-QuPAP pseudonym: "
                    f"{normalized_pseudonym}"
                )

            previous_record = copy.deepcopy(record)

            record.subscriber_status = normalized_status
            record.updated_at = current_utc_timestamp()

            try:
                record.__post_init__()
                self._save()
            except Exception:
                self._subscribers[
                    normalized_pseudonym
                ] = previous_record
                raise

            return copy.deepcopy(record)

    def activate_subscriber(
        self,
        pseudonym_id: str,
    ) -> SubscriberRecord:
        """Activate a subscriber."""

        return self.update_status(
            pseudonym_id,
            "active",
        )

    def suspend_subscriber(
        self,
        pseudonym_id: str,
    ) -> SubscriberRecord:
        """Temporarily suspend a subscriber."""

        return self.update_status(
            pseudonym_id,
            "suspended",
        )

    def revoke_subscriber(
        self,
        pseudonym_id: str,
    ) -> SubscriberRecord:
        """Revoke a subscriber."""

        return self.update_status(
            pseudonym_id,
            "revoked",
        )

    def update_registered_contexts(
        self,
        pseudonym_id: str,
        registered_contexts: Iterable[str],
    ) -> SubscriberRecord:
        """Replace the subscriber's permitted contexts."""

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        normalized_contexts = normalize_contexts(
            registered_contexts
        )

        with self._lock:
            record = self._subscribers.get(
                normalized_pseudonym
            )

            if record is None:
                raise LookupError(
                    "Unknown FT-QuPAP pseudonym: "
                    f"{normalized_pseudonym}"
                )

            previous_record = copy.deepcopy(record)

            record.registered_contexts = (
                normalized_contexts
            )

            record.updated_at = current_utc_timestamp()

            try:
                record.__post_init__()
                self._save()
            except Exception:
                self._subscribers[
                    normalized_pseudonym
                ] = previous_record
                raise

            return copy.deepcopy(record)

    def update_metadata(
        self,
        pseudonym_id: str,
        metadata_updates: Mapping[str, Any],
    ) -> SubscriberRecord:
        """Merge non-secret operator metadata."""

        if not isinstance(metadata_updates, Mapping):
            raise TypeError(
                "metadata_updates must be a mapping."
            )

        safe_updates = validate_metadata(
            dict(metadata_updates)
        )

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        with self._lock:
            record = self._subscribers.get(
                normalized_pseudonym
            )

            if record is None:
                raise LookupError(
                    "Unknown FT-QuPAP pseudonym: "
                    f"{normalized_pseudonym}"
                )

            previous_record = copy.deepcopy(record)

            record.metadata.update(safe_updates)
            record.updated_at = current_utc_timestamp()

            try:
                record.__post_init__()
                self._save()
            except Exception:
                self._subscribers[
                    normalized_pseudonym
                ] = previous_record
                raise

            return copy.deepcopy(record)

    def delete_subscriber(
        self,
        pseudonym_id: str,
    ) -> bool:
        """
        Delete a subscriber record.

        Revocation should normally be preferred so that the
        authorization history remains explicit.
        """

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        with self._lock:
            record = self._subscribers.pop(
                normalized_pseudonym,
                None,
            )

            if record is None:
                return False

            try:
                self._save()
            except Exception:
                self._subscribers[
                    normalized_pseudonym
                ] = record
                raise

            return True

    def list_subscribers(
        self,
        subscriber_status: str | None = None,
        context: str | None = None,
    ) -> list[SubscriberRecord]:
        """List subscribers using optional filters."""

        normalized_status = (
            validate_subscriber_status(
                subscriber_status
            )
            if subscriber_status is not None
            else None
        )

        normalized_context = (
            validate_context(context)
            if context is not None
            else None
        )

        with self._lock:
            records = list(
                self._subscribers.values()
            )

        if normalized_status is not None:
            records = [
                record
                for record in records
                if (
                    record.subscriber_status
                    == normalized_status
                )
            ]

        if normalized_context is not None:
            records = [
                record
                for record in records
                if (
                    normalized_context
                    in record.registered_contexts
                )
            ]

        return [
            copy.deepcopy(record)
            for record in sorted(
                records,
                key=lambda item: item.pseudonym_id,
            )
        ]

    def count_subscribers(
        self,
        subscriber_status: str | None = None,
    ) -> int:
        """Return the number of matching subscribers."""

        return len(
            self.list_subscribers(
                subscriber_status=subscriber_status
            )
        )

    def as_notebook_database(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Return a deep copy matching AS_SUBSCRIBER_DB in the notebook.
        """

        with self._lock:
            return {
                pseudonym_id:
                    record.to_notebook_record()
                for pseudonym_id, record
                in self._subscribers.items()
            }

    def clear(self) -> None:
        """Remove all subscribers for tests or demo resets."""

        with self._lock:
            previous_records = copy.deepcopy(
                self._subscribers
            )

            self._subscribers.clear()

            try:
                self._save()
            except Exception:
                self._subscribers = previous_records
                raise

    # ---------------------------------------------------------
    # Notebook-compatible mapping operations
    # ---------------------------------------------------------

    def __contains__(self, pseudonym_id: object) -> bool:
        """Support: pseudonym_id in subscriber_database."""

        if not isinstance(pseudonym_id, str):
            return False

        return self.subscriber_exists(pseudonym_id)

    def __len__(self) -> int:
        """Return the number of registered pseudonyms."""

        with self._lock:
            return len(self._subscribers)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered pseudonym identifiers."""

        with self._lock:
            pseudonyms = list(
                self._subscribers.keys()
            )

        return iter(pseudonyms)

    def __getitem__(
        self,
        pseudonym_id: str,
    ) -> dict[str, Any]:
        """
        Return a notebook-style subscriber dictionary.

        Raises:
            KeyError:
                When the pseudonym is unknown.
        """

        record = self.get_subscriber(pseudonym_id)

        if record is None:
            raise KeyError(pseudonym_id)

        return record.to_notebook_record()

    def get(
        self,
        pseudonym_id: str,
        default: Any = None,
    ) -> dict[str, Any] | Any:
        """Provide dictionary-style get behavior."""

        record = self.get_subscriber(pseudonym_id)

        if record is None:
            return default

        return record.to_notebook_record()

    def keys(self) -> list[str]:
        """Return registered pseudonym identifiers."""

        with self._lock:
            return sorted(self._subscribers.keys())

    def values(self) -> list[dict[str, Any]]:
        """Return notebook-style subscriber records."""

        return [
            record.to_notebook_record()
            for record in self.list_subscribers()
        ]

    def items(
        self,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return notebook-style pseudonym/record pairs."""

        return [
            (
                record.pseudonym_id,
                record.to_notebook_record(),
            )
            for record in self.list_subscribers()
        ]


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


def validate_subscriber_status(
    status: str,
) -> str:
    """Validate active, suspended, or revoked status."""

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
                sorted(
                    SUPPORTED_SUBSCRIBER_STATUSES
                )
            )
        )

    return normalized_status


def validate_context(context: str) -> str:
    """Validate an FT-QuPAP channel/service context."""

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


def normalize_contexts(
    contexts: Iterable[str],
) -> list[str]:
    """Validate and normalize registered contexts."""

    if isinstance(contexts, (str, bytes)):
        raise TypeError(
            "registered_contexts must be "
            "an iterable of context strings."
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


def normalize_timestamp(
    value: datetime | str,
) -> str:
    """Normalize a datetime or ISO timestamp to UTC."""

    if isinstance(value, datetime):
        parsed_datetime = value

    elif isinstance(value, str):
        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "Timestamp cannot be empty."
            )

        if normalized_value.endswith("Z"):
            normalized_value = (
                normalized_value[:-1]
                + "+00:00"
            )

        try:
            parsed_datetime = datetime.fromisoformat(
                normalized_value
            )
        except ValueError as error:
            raise ValueError(
                "Timestamp must use ISO 8601 format."
            ) from error

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


def validate_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Validate optional non-secret metadata.

    Metadata is checked recursively for prohibited secret fields and
    JSON serialization compatibility.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")

    copied_metadata = copy.deepcopy(dict(metadata))

    inspect_metadata_for_secrets(copied_metadata)

    try:
        json.dumps(copied_metadata)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "metadata must contain JSON-serializable values."
        ) from error

    return copied_metadata


def inspect_metadata_for_secrets(
    value: Any,
    path: str = "metadata",
) -> None:
    """Reject secret or raw-identity fields in metadata."""

    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = normalize_field_name(
                str(key)
            )

            if normalized_key in SENSITIVE_METADATA_FIELDS:
                raise ValueError(
                    "Sensitive field cannot be stored in "
                    f"subscriber metadata: {path}.{key}"
                )

            inspect_metadata_for_secrets(
                nested_value,
                f"{path}.{key}",
            )

    elif isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            inspect_metadata_for_secrets(
                nested_value,
                f"{path}[{index}]",
            )

    elif isinstance(value, (bytes, bytearray)):
        raise ValueError(
            "Binary values cannot be stored in "
            f"subscriber metadata: {path}"
        )


def normalize_field_name(value: str) -> str:
    """Normalize a field name for security-policy matching."""

    normalized_characters = [
        character.lower()
        if character.isalnum()
        else "_"
        for character in value.strip()
    ]

    normalized_value = "".join(
        normalized_characters
    )

    while "__" in normalized_value:
        normalized_value = normalized_value.replace(
            "__",
            "_",
        )

    return normalized_value.strip("_")


def run_self_test() -> None:
    """Run subscriber persistence and authorization tests."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = (
            Path(temporary_directory)
            / "subscribers.json"
        )

        database = SubscriberDatabase(
            database_path
        )

        record = database.register_subscriber(
            pseudonym_id="PID-6G-UE-0001",
            subscriber_status="active",
            registered_contexts=[
                "urban",
                "suburban",
                "rural",
            ],
            metadata={
                "registration_mode":
                    "operator_managed",
            },
        )

        assert record.pseudonym_id == (
            "PID-6G-UE-0001"
        )

        assert record.is_active
        assert len(database) == 1

        assert "PID-6G-UE-0001" in database

        notebook_record = database[
            "PID-6G-UE-0001"
        ]

        assert notebook_record == {
            "pseudonym_id":
                "PID-6G-UE-0001",
            "subscriber_status":
                "active",
            "registered_contexts": [
                "rural",
                "suburban",
                "urban",
            ],
        }

        valid, reason = database.verify_subscriber(
            pseudonym_id="PID-6G-UE-0001",
            service_context="urban",
        )

        assert valid is True
        assert reason == "subscriber_valid"

        unknown_valid, unknown_reason = (
            database.verify_subscriber(
                pseudonym_id="PID-UNKNOWN",
                service_context="urban",
            )
        )

        assert unknown_valid is False
        assert unknown_reason == "unknown_pseudonym"

        database.suspend_subscriber(
            "PID-6G-UE-0001"
        )

        suspended_valid, suspended_reason = (
            database.verify_subscriber(
                pseudonym_id="PID-6G-UE-0001",
                service_context="urban",
            )
        )

        assert suspended_valid is False
        assert suspended_reason == (
            "inactive_subscriber"
        )

        database.activate_subscriber(
            "PID-6G-UE-0001"
        )

        database.update_registered_contexts(
            pseudonym_id="PID-6G-UE-0001",
            registered_contexts=["urban"],
        )

        context_valid, context_reason = (
            database.verify_subscriber(
                pseudonym_id="PID-6G-UE-0001",
                service_context="rural",
            )
        )

        assert context_valid is False
        assert context_reason == (
            "unregistered_context"
        )

        reloaded_database = SubscriberDatabase(
            database_path
        )

        assert (
            reloaded_database.count_subscribers()
            == 1
        )

        assert (
            reloaded_database
            .require_subscriber(
                "PID-6G-UE-0001"
            )
            .registered_contexts
            == ["urban"]
        )

        stored_text = database_path.read_text(
            encoding="utf-8"
        )

        assert "IMSI" not in stored_text
        assert "shared_secret" not in stored_text
        assert "k_auth" not in stored_text
        assert "k_ctrl" not in stored_text

        print(
            "Subscriber database self-test passed."
        )
        print(f"Database path: {database_path}")
        print(
            "Registered pseudonyms: "
            f"{reloaded_database.keys()}"
        )


if __name__ == "__main__":
    run_self_test()