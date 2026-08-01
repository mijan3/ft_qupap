"""
FT-QuPAP Nonce Database
=======================

Persistent replay-protection storage for FT-QuPAP v5.1.

Notebook-aligned replay-validation logic:

1. Verify that request["pseudonym_id"] exists.
2. Verify that the request timestamp is within the configured
   freshness window.
3. Build a replay identity from:

       pseudonym_id || nonce

4. Reject the request when the same replay identity already exists.
5. Consume the nonce only when consume_nonce=True.

Notebook-compatible result reasons:

    unknown_pseudonym
    stale_timestamp
    replayed_nonce
    fresh_request

Security note:
    The notebook uses an in-memory dictionary whose key contains the
    pseudonym and Base64 nonce. This persistent implementation preserves
    the same replay behavior but stores a SHA3-256 replay identifier
    instead of the raw nonce.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_NONCE_DATABASE_PATH = Path(
    "database/used_nonces.json"
)

PROTOCOL_VERSION = "FT-QuPAP-v5.1"
DATABASE_VERSION = 1

DEFAULT_FRESHNESS_WINDOW_SECONDS = 60
EXPECTED_NONCE_LENGTH_BYTES = 16

REQUEST_TYPE = "FT-QuPAP-Authentication"

SUPPORTED_CONTEXTS = {
    "urban",
    "suburban",
    "rural",
}


class NonceReplayError(RuntimeError):
    """
    Raised when an already-consumed nonce is submitted again.
    """


@dataclass(frozen=True)
class NonceRecord:
    """
    Persistent record for one consumed authentication nonce.

    Attributes:
        replay_id:
            SHA3-256 identifier derived from the pseudonym and Base64
            nonce. It replaces the notebook's raw:

                pseudonym_id:nonce

            replay-cache key in persistent storage.

        nonce_digest:
            SHA3-256 digest of the decoded nonce bytes.

        pseudonym_id:
            Pseudonymous subscriber identity associated with the nonce.

        consumed_at:
            Authentication Server Unix timestamp at which the nonce was
            consumed.

        request_timestamp:
            Unix timestamp supplied by the Mobile Station.

        service_context:
            Urban, suburban, or rural request context.

        request_type:
            FT-QuPAP authentication request type.

        metadata:
            Optional non-secret replay-detection metadata.
    """

    replay_id: str
    nonce_digest: str
    pseudonym_id: str

    consumed_at: int
    request_timestamp: int

    service_context: str
    request_type: str = REQUEST_TYPE

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the immutable nonce record."""

        validate_hex_digest(
            "replay_id",
            self.replay_id,
        )

        validate_hex_digest(
            "nonce_digest",
            self.nonce_digest,
        )

        normalize_required_string(
            "pseudonym_id",
            self.pseudonym_id,
        )

        validate_unix_timestamp(
            "consumed_at",
            self.consumed_at,
        )

        validate_unix_timestamp(
            "request_timestamp",
            self.request_timestamp,
        )

        validate_context(self.service_context)

        normalize_required_string(
            "request_type",
            self.request_type,
        )

        validate_metadata(self.metadata)

    def to_dictionary(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)

    @classmethod
    def from_dictionary(
        cls,
        data: Mapping[str, Any],
    ) -> "NonceRecord":
        """Build a nonce record from stored JSON data."""

        if not isinstance(data, Mapping):
            raise TypeError(
                "Nonce record data must be a mapping."
            )

        required_fields = {
            "replay_id",
            "nonce_digest",
            "pseudonym_id",
            "consumed_at",
            "request_timestamp",
            "service_context",
        }

        missing_fields = required_fields.difference(
            data.keys()
        )

        if missing_fields:
            raise ValueError(
                "Nonce record is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )

        return cls(
            replay_id=str(data["replay_id"]),
            nonce_digest=str(data["nonce_digest"]),
            pseudonym_id=str(data["pseudonym_id"]),
            consumed_at=int(data["consumed_at"]),
            request_timestamp=int(
                data["request_timestamp"]
            ),
            service_context=str(
                data["service_context"]
            ),
            request_type=str(
                data.get(
                    "request_type",
                    REQUEST_TYPE,
                )
            ),
            metadata=dict(
                data.get("metadata", {})
            ),
        )


class NonceDatabase:
    """
    Thread-safe JSON-backed FT-QuPAP replay cache.

    The database performs replay check and nonce consumption inside one
    lock. This avoids a check-then-insert race between two concurrent
    authentication requests.
    """

    def __init__(
        self,
        database_path: str | Path = (
            DEFAULT_NONCE_DATABASE_PATH
        ),
        freshness_window_seconds: int = (
            DEFAULT_FRESHNESS_WINDOW_SECONDS
        ),
    ) -> None:
        """
        Initialize the replay database.

        Args:
            database_path:
                Path to database/used_nonces.json.

            freshness_window_seconds:
                Maximum allowed absolute difference between the Mobile
                Station timestamp and Authentication Server timestamp.
        """

        self._database_path = Path(database_path)

        self._freshness_window_seconds = (
            validate_positive_integer(
                "freshness_window_seconds",
                freshness_window_seconds,
            )
        )

        self._lock = threading.RLock()

        self._records: dict[str, NonceRecord] = {}

        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the persistent replay-database path."""

        return self._database_path

    @property
    def freshness_window_seconds(self) -> int:
        """Return the configured request freshness window."""

        return self._freshness_window_seconds

    def _initialize_database(self) -> None:
        """Create the database file or load existing records."""

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
        """Load consumed nonce records from JSON storage."""

        with self._lock:
            try:
                with self._database_path.open(
                    "r",
                    encoding="utf-8",
                ) as database_file:
                    stored_data = json.load(
                        database_file
                    )

            except json.JSONDecodeError as error:
                raise ValueError(
                    "Nonce database contains invalid JSON."
                ) from error

            except OSError as error:
                raise OSError(
                    "Unable to read the nonce database."
                ) from error

            if not isinstance(stored_data, dict):
                raise ValueError(
                    "Nonce database root must be "
                    "a JSON object."
                )

            database_version = stored_data.get(
                "version",
                DATABASE_VERSION,
            )

            if database_version != DATABASE_VERSION:
                raise ValueError(
                    "Unsupported nonce database version: "
                    f"{database_version}"
                )

            raw_records = stored_data.get(
                "used_nonces",
                {},
            )

            if not isinstance(raw_records, dict):
                raise ValueError(
                    "'used_nonces' must be a JSON object "
                    "indexed by replay_id."
                )

            loaded_records: dict[str, NonceRecord] = {}

            for replay_id, raw_record in (
                raw_records.items()
            ):
                if not isinstance(raw_record, dict):
                    raise ValueError(
                        "Each used nonce record must be "
                        "a JSON object."
                    )

                record_data = dict(raw_record)

                record_data.setdefault(
                    "replay_id",
                    replay_id,
                )

                record = NonceRecord.from_dictionary(
                    record_data
                )

                if record.replay_id != replay_id:
                    raise ValueError(
                        "Nonce record key does not match "
                        "its replay_id."
                    )

                if replay_id in loaded_records:
                    raise ValueError(
                        "Duplicate replay_id detected: "
                        f"{replay_id}"
                    )

                loaded_records[replay_id] = record

            self._records = loaded_records

    def reload(self) -> None:
        """Reload replay records from disk."""

        self._load()

    def _save(self) -> None:
        """Persist nonce records using atomic file replacement."""

        with self._lock:
            database_content = {
                "version": DATABASE_VERSION,
                "protocol": PROTOCOL_VERSION,
                "freshness_window_seconds":
                    self._freshness_window_seconds,
                "replay_key_algorithm":
                    "SHA3-256",
                "nonce_digest_algorithm":
                    "SHA3-256",
                "used_nonces": {
                    replay_id:
                        record.to_dictionary()
                    for replay_id, record
                    in sorted(
                        self._records.items(),
                        key=lambda item: (
                            item[1].consumed_at,
                            item[0],
                        ),
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
                    "Unable to save the nonce database."
                ) from error

    def validate_request_freshness_and_replay(
        self,
        request: Mapping[str, Any],
        subscriber_db: Any,
        now: int | None = None,
        consume_nonce: bool = True,
    ) -> tuple[bool, str]:
        """
        Validate an FT-QuPAP authentication request.

        This method follows the notebook's Cell 25 behavior.

        Args:
            request:
                Authentication request containing pseudonym_id,
                timestamp, nonce, service_context, and request_type.

            subscriber_db:
                Subscriber mapping or SubscriberDatabase instance.

            now:
                Optional Authentication Server Unix timestamp.

            consume_nonce:
                When True, a fresh nonce is atomically stored.
                When False, the method performs a non-consuming check.

        Returns:
            (False, "unknown_pseudonym")
            (False, "stale_timestamp")
            (False, "replayed_nonce")
            (True, "fresh_request")
        """

        if not isinstance(request, Mapping):
            raise TypeError(
                "request must be a mapping."
            )

        if not isinstance(consume_nonce, bool):
            raise TypeError(
                "consume_nonce must be a boolean."
            )

        current_time = (
            current_timestamp()
            if now is None
            else validate_unix_timestamp(
                "now",
                now,
            )
        )

        pseudonym_value = request.get(
            "pseudonym_id"
        )

        if not subscriber_exists(
            subscriber_db,
            pseudonym_value,
        ):
            return False, "unknown_pseudonym"

        try:
            request_timestamp = int(
                request.get("timestamp", 0)
            )
        except (TypeError, ValueError):
            return False, "stale_timestamp"

        if (
            abs(current_time - request_timestamp)
            > self._freshness_window_seconds
        ):
            return False, "stale_timestamp"

        pseudonym_id = normalize_required_string(
            "pseudonym_id",
            str(pseudonym_value),
        )

        nonce_text = normalize_nonce_text(
            request.get("nonce")
        )

        service_context = normalize_request_context(
            request.get(
                "service_context",
                "urban",
            )
        )

        request_type = normalize_required_string(
            "request_type",
            str(
                request.get(
                    "request_type",
                    REQUEST_TYPE,
                )
            ),
        )

        replay_id = generate_replay_id(
            pseudonym_id=pseudonym_id,
            nonce_text=nonce_text,
        )

        with self._lock:
            if replay_id in self._records:
                return False, "replayed_nonce"

            if consume_nonce:
                nonce_bytes = decode_nonce(
                    nonce_text
                )

                record = NonceRecord(
                    replay_id=replay_id,
                    nonce_digest=hashlib.sha3_256(
                        nonce_bytes
                    ).hexdigest(),
                    pseudonym_id=pseudonym_id,
                    consumed_at=current_time,
                    request_timestamp=request_timestamp,
                    service_context=service_context,
                    request_type=request_type,
                    metadata={},
                )

                self._records[replay_id] = record

                try:
                    self._save()
                except Exception:
                    self._records.pop(
                        replay_id,
                        None,
                    )
                    raise

        return True, "fresh_request"

    def consume_nonce(
        self,
        pseudonym_id: str,
        nonce: str | bytes,
        *,
        request_timestamp: int | None = None,
        consumed_at: int | None = None,
        service_context: str = "urban",
        request_type: str = REQUEST_TYPE,
        metadata: Mapping[str, Any] | None = None,
    ) -> NonceRecord:
        """
        Atomically consume a nonce.

        Raises:
            NonceReplayError:
                When the pseudonym/nonce pair is already present.
        """

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        nonce_text = nonce_to_base64_text(
            nonce
        )

        normalized_context = validate_context(
            service_context
        )

        normalized_request_type = (
            normalize_required_string(
                "request_type",
                request_type,
            )
        )

        current_time = (
            current_timestamp()
            if consumed_at is None
            else validate_unix_timestamp(
                "consumed_at",
                consumed_at,
            )
        )

        normalized_request_timestamp = (
            current_time
            if request_timestamp is None
            else validate_unix_timestamp(
                "request_timestamp",
                request_timestamp,
            )
        )

        safe_metadata = validate_metadata(
            metadata or {}
        )

        replay_id = generate_replay_id(
            pseudonym_id=normalized_pseudonym,
            nonce_text=nonce_text,
        )

        nonce_bytes = decode_nonce(nonce_text)

        record = NonceRecord(
            replay_id=replay_id,
            nonce_digest=hashlib.sha3_256(
                nonce_bytes
            ).hexdigest(),
            pseudonym_id=normalized_pseudonym,
            consumed_at=current_time,
            request_timestamp=(
                normalized_request_timestamp
            ),
            service_context=normalized_context,
            request_type=normalized_request_type,
            metadata=safe_metadata,
        )

        with self._lock:
            if replay_id in self._records:
                raise NonceReplayError(
                    "Replay detected for pseudonym "
                    f"'{normalized_pseudonym}'."
                )

            self._records[replay_id] = record

            try:
                self._save()
            except Exception:
                self._records.pop(
                    replay_id,
                    None,
                )
                raise

            return copy.deepcopy(record)

    def check_and_consume_nonce(
        self,
        pseudonym_id: str,
        nonce: str | bytes,
        *,
        request_timestamp: int | None = None,
        consumed_at: int | None = None,
        service_context: str = "urban",
        request_type: str = REQUEST_TYPE,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """
        Return True for a fresh nonce and False for a replay.
        """

        try:
            self.consume_nonce(
                pseudonym_id=pseudonym_id,
                nonce=nonce,
                request_timestamp=request_timestamp,
                consumed_at=consumed_at,
                service_context=service_context,
                request_type=request_type,
                metadata=metadata,
            )

        except NonceReplayError:
            return False

        return True

    def is_nonce_used(
        self,
        pseudonym_id: str,
        nonce: str | bytes,
    ) -> bool:
        """Return True when the nonce has already been consumed."""

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        nonce_text = nonce_to_base64_text(
            nonce
        )

        replay_id = generate_replay_id(
            pseudonym_id=normalized_pseudonym,
            nonce_text=nonce_text,
        )

        with self._lock:
            return replay_id in self._records

    def get_record(
        self,
        pseudonym_id: str,
        nonce: str | bytes,
    ) -> NonceRecord | None:
        """Return the matching replay record."""

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        nonce_text = nonce_to_base64_text(
            nonce
        )

        replay_id = generate_replay_id(
            pseudonym_id=normalized_pseudonym,
            nonce_text=nonce_text,
        )

        return self.get_record_by_id(
            replay_id
        )

    def get_record_by_id(
        self,
        replay_id: str,
    ) -> NonceRecord | None:
        """Return a replay record using its digest identifier."""

        normalized_replay_id = validate_hex_digest(
            "replay_id",
            replay_id,
        )

        with self._lock:
            record = self._records.get(
                normalized_replay_id
            )

            return (
                copy.deepcopy(record)
                if record is not None
                else None
            )

    def list_records(
        self,
        pseudonym_id: str | None = None,
        service_context: str | None = None,
    ) -> list[NonceRecord]:
        """List replay records using optional filters."""

        normalized_pseudonym = (
            normalize_required_string(
                "pseudonym_id",
                pseudonym_id,
            )
            if pseudonym_id is not None
            else None
        )

        normalized_context = (
            validate_context(service_context)
            if service_context is not None
            else None
        )

        with self._lock:
            records = list(
                self._records.values()
            )

        if normalized_pseudonym is not None:
            records = [
                record
                for record in records
                if (
                    record.pseudonym_id
                    == normalized_pseudonym
                )
            ]

        if normalized_context is not None:
            records = [
                record
                for record in records
                if (
                    record.service_context
                    == normalized_context
                )
            ]

        return [
            copy.deepcopy(record)
            for record in sorted(
                records,
                key=lambda item: (
                    item.consumed_at,
                    item.replay_id,
                ),
            )
        ]

    def count_records(
        self,
        pseudonym_id: str | None = None,
    ) -> int:
        """Return the number of stored replay records."""

        return len(
            self.list_records(
                pseudonym_id=pseudonym_id
            )
        )

    def remove_nonce(
        self,
        pseudonym_id: str,
        nonce: str | bytes,
    ) -> bool:
        """
        Remove one consumed nonce.

        This operation is intended for tests and controlled demo reset
        operations. Authentication code should not normally remove a
        consumed nonce.
        """

        normalized_pseudonym = normalize_required_string(
            "pseudonym_id",
            pseudonym_id,
        )

        nonce_text = nonce_to_base64_text(
            nonce
        )

        replay_id = generate_replay_id(
            pseudonym_id=normalized_pseudonym,
            nonce_text=nonce_text,
        )

        with self._lock:
            record = self._records.pop(
                replay_id,
                None,
            )

            if record is None:
                return False

            try:
                self._save()
            except Exception:
                self._records[
                    replay_id
                ] = record
                raise

            return True

    def purge_before(
        self,
        unix_timestamp: int,
    ) -> int:
        """
        Remove replay records consumed before a supplied timestamp.

        No automatic expiry is applied because the notebook replay cache
        retains consumed nonces until the experiment or scenario resets
        the cache.
        """

        cutoff = validate_unix_timestamp(
            "unix_timestamp",
            unix_timestamp,
        )

        with self._lock:
            removable_ids = [
                replay_id
                for replay_id, record
                in self._records.items()
                if record.consumed_at < cutoff
            ]

            if not removable_ids:
                return 0

            previous_records = copy.deepcopy(
                self._records
            )

            for replay_id in removable_ids:
                self._records.pop(
                    replay_id,
                    None,
                )

            try:
                self._save()
            except Exception:
                self._records = previous_records
                raise

            return len(removable_ids)

    def clear(self) -> None:
        """
        Remove all replay records.

        This corresponds to the notebook's:

            REPLAY_CACHE.clear()

        and is intended for a new controlled scenario or test run.
        """

        with self._lock:
            previous_records = copy.deepcopy(
                self._records
            )

            self._records.clear()

            try:
                self._save()
            except Exception:
                self._records = previous_records
                raise

    def __len__(self) -> int:
        """Return the number of consumed nonces."""

        with self._lock:
            return len(self._records)


def validate_request_freshness_and_replay(
    request: Mapping[str, Any],
    subscriber_db: Any,
    replay_cache: (
        NonceDatabase
        | MutableMapping[str, float]
    ),
    now: int | None = None,
    consume_nonce: bool = True,
) -> tuple[bool, str]:
    """
    Notebook-compatible module-level request validator.

    When replay_cache is a NonceDatabase, raw nonces are not persisted.
    When replay_cache is a normal dictionary, this function follows the
    notebook's in-memory behavior exactly.
    """

    if isinstance(replay_cache, NonceDatabase):
        return (
            replay_cache
            .validate_request_freshness_and_replay(
                request=request,
                subscriber_db=subscriber_db,
                now=now,
                consume_nonce=consume_nonce,
            )
        )

    if not isinstance(replay_cache, MutableMapping):
        raise TypeError(
            "replay_cache must be a NonceDatabase "
            "or mutable mapping."
        )

    current_time = (
        current_timestamp()
        if now is None
        else int(now)
    )

    pseudonym_id = request.get(
        "pseudonym_id"
    )

    if not subscriber_exists(
        subscriber_db,
        pseudonym_id,
    ):
        return False, "unknown_pseudonym"

    try:
        request_timestamp = int(
            request.get("timestamp", 0)
        )
    except (TypeError, ValueError):
        return False, "stale_timestamp"

    if (
        abs(current_time - request_timestamp)
        > DEFAULT_FRESHNESS_WINDOW_SECONDS
    ):
        return False, "stale_timestamp"

    nonce_key = (
        f"{request.get('pseudonym_id')}:"
        f"{request.get('nonce')}"
    )

    if nonce_key in replay_cache:
        return False, "replayed_nonce"

    if consume_nonce:
        replay_cache[nonce_key] = float(
            current_time
        )

    return True, "fresh_request"


def subscriber_exists(
    subscriber_db: Any,
    pseudonym_id: Any,
) -> bool:
    """
    Check pseudonym membership in a mapping or SubscriberDatabase.
    """

    if not isinstance(pseudonym_id, str):
        return False

    normalized_pseudonym = pseudonym_id.strip()

    if not normalized_pseudonym:
        return False

    try:
        return normalized_pseudonym in subscriber_db
    except TypeError:
        return False


def generate_replay_id(
    pseudonym_id: str,
    nonce_text: str,
) -> str:
    """
    Generate the persistent equivalent of:

        f"{pseudonym_id}:{nonce}"

    Length prefixes prevent ambiguous concatenation.
    """

    normalized_pseudonym = normalize_required_string(
        "pseudonym_id",
        pseudonym_id,
    )

    normalized_nonce = normalize_nonce_text(
        nonce_text
    )

    pseudonym_bytes = normalized_pseudonym.encode(
        "utf-8"
    )

    nonce_bytes = normalized_nonce.encode(
        "ascii"
    )

    replay_material = (
        length_prefix(pseudonym_bytes)
        + length_prefix(nonce_bytes)
    )

    return hashlib.sha3_256(
        replay_material
    ).hexdigest()


def nonce_to_base64_text(
    nonce: str | bytes,
) -> str:
    """Convert nonce bytes or Base64 text into canonical text."""

    if isinstance(nonce, bytes):
        if not nonce:
            raise ValueError(
                "nonce cannot be empty."
            )

        return base64.b64encode(
            nonce
        ).decode("ascii")

    return normalize_nonce_text(nonce)


def normalize_nonce_text(
    nonce: Any,
) -> str:
    """Validate and normalize a Base64-encoded nonce."""

    if not isinstance(nonce, str):
        raise TypeError(
            "nonce must be a Base64 string."
        )

    normalized_nonce = nonce.strip()

    if not normalized_nonce:
        raise ValueError(
            "nonce cannot be empty."
        )

    decode_nonce(normalized_nonce)

    return normalized_nonce


def decode_nonce(
    nonce_text: str,
) -> bytes:
    """Decode and validate a Base64 nonce."""

    if not isinstance(nonce_text, str):
        raise TypeError(
            "nonce_text must be a string."
        )

    try:
        nonce_bytes = base64.b64decode(
            nonce_text.encode("ascii"),
            validate=True,
        )
    except (
        ValueError,
        UnicodeEncodeError,
        base64.binascii.Error,
    ) as error:
        raise ValueError(
            "nonce must be valid Base64 data."
        ) from error

    if len(nonce_bytes) != EXPECTED_NONCE_LENGTH_BYTES:
        raise ValueError(
            "FT-QuPAP authentication nonce must contain "
            f"{EXPECTED_NONCE_LENGTH_BYTES} bytes."
        )

    return nonce_bytes


def length_prefix(value: bytes) -> bytes:
    """Prefix bytes with an eight-byte unsigned length."""

    if not isinstance(value, bytes):
        raise TypeError(
            "value must be bytes."
        )

    return (
        len(value).to_bytes(
            8,
            byteorder="big",
            signed=False,
        )
        + value
    )


def current_timestamp() -> int:
    """Return the current Unix timestamp."""

    return int(time.time())


def validate_unix_timestamp(
    name: str,
    value: int,
) -> int:
    """Validate and return a nonnegative Unix timestamp."""

    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be an integer."
        )

    try:
        normalized_value = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be an integer."
        ) from error

    if normalized_value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return normalized_value


def validate_positive_integer(
    name: str,
    value: int,
) -> int:
    """Validate and return an integer greater than zero."""

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return value


def normalize_required_string(
    name: str,
    value: str,
) -> str:
    """Validate and normalize a required string."""

    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return normalized_value


def validate_context(
    context: str,
) -> str:
    """Validate an FT-QuPAP service context."""

    normalized_context = (
        normalize_required_string(
            "service_context",
            context,
        ).lower()
    )

    if normalized_context not in SUPPORTED_CONTEXTS:
        raise ValueError(
            "service_context must be one of: "
            + ", ".join(
                sorted(SUPPORTED_CONTEXTS)
            )
        )

    return normalized_context


def normalize_request_context(
    context: Any,
) -> str:
    """
    Normalize request context without changing replay result reasons.

    Request syntax validation belongs to the authentication-request
    validator. For replay storage, an unsupported value is retained as
    "unknown".
    """

    if not isinstance(context, str):
        return "unknown"

    normalized_context = context.strip().lower()

    if normalized_context in SUPPORTED_CONTEXTS:
        return normalized_context

    return "unknown"


def validate_hex_digest(
    name: str,
    value: str,
) -> str:
    """Validate a 256-bit hexadecimal digest."""

    normalized_value = normalize_required_string(
        name,
        value,
    ).lower()

    if len(normalized_value) != 64:
        raise ValueError(
            f"{name} must contain 64 hexadecimal characters."
        )

    try:
        bytes.fromhex(normalized_value)
    except ValueError as error:
        raise ValueError(
            f"{name} must contain valid hexadecimal data."
        ) from error

    return normalized_value


def validate_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate non-secret JSON-compatible metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError(
            "metadata must be a mapping."
        )

    safe_metadata = copy.deepcopy(
        dict(metadata)
    )

    prohibited_fields = {
        "nonce",
        "raw_nonce",
        "shared_secret",
        "session_secret",
        "k_auth",
        "k_ctrl",
        "private_key",
        "secret_key",
        "ciphertext",
        "kmac_tag",
        "authentication_tag",
        "quantum_state",
        "statevector",
    }

    inspect_metadata_fields(
        safe_metadata,
        prohibited_fields,
    )

    try:
        json.dumps(safe_metadata)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "metadata must be JSON serializable."
        ) from error

    return safe_metadata


def inspect_metadata_fields(
    value: Any,
    prohibited_fields: set[str],
    path: str = "metadata",
) -> None:
    """Reject secret fields inside nested metadata."""

    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = normalize_field_name(
                str(key)
            )

            if normalized_key in prohibited_fields:
                raise ValueError(
                    "Sensitive field cannot be stored: "
                    f"{path}.{key}"
                )

            inspect_metadata_fields(
                nested_value,
                prohibited_fields,
                f"{path}.{key}",
            )

    elif isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            inspect_metadata_fields(
                nested_value,
                prohibited_fields,
                f"{path}[{index}]",
            )

    elif isinstance(value, (bytes, bytearray)):
        raise ValueError(
            "Binary values cannot be stored in "
            f"{path}."
        )


def normalize_field_name(
    value: str,
) -> str:
    """Normalize a field name for security checking."""

    characters = [
        character.lower()
        if character.isalnum()
        else "_"
        for character in value.strip()
    ]

    normalized_value = "".join(characters)

    while "__" in normalized_value:
        normalized_value = (
            normalized_value.replace(
                "__",
                "_",
            )
        )

    return normalized_value.strip("_")


def run_self_test() -> None:
    """Run freshness, replay, and persistence tests."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = (
            Path(temporary_directory)
            / "used_nonces.json"
        )

        nonce_database = NonceDatabase(
            database_path=database_path,
            freshness_window_seconds=60,
        )

        subscriber_database = {
            "PID-6G-UE-0001": {
                "pseudonym_id":
                    "PID-6G-UE-0001",
                "subscriber_status":
                    "active",
                "registered_contexts": [
                    "urban",
                    "suburban",
                    "rural",
                ],
            }
        }

        fixed_now = 1_800_000_000

        nonce_bytes = bytes.fromhex(
            "00112233445566778899aabbccddeeff"
        )

        nonce_text = base64.b64encode(
            nonce_bytes
        ).decode("ascii")

        request = {
            "pseudonym_id":
                "PID-6G-UE-0001",
            "timestamp":
                fixed_now,
            "nonce":
                nonce_text,
            "service_context":
                "urban",
            "request_type":
                REQUEST_TYPE,
        }

        check_valid, check_reason = (
            nonce_database
            .validate_request_freshness_and_replay(
                request=request,
                subscriber_db=subscriber_database,
                now=fixed_now,
                consume_nonce=False,
            )
        )

        assert check_valid is True
        assert check_reason == "fresh_request"
        assert len(nonce_database) == 0

        first_valid, first_reason = (
            nonce_database
            .validate_request_freshness_and_replay(
                request=request,
                subscriber_db=subscriber_database,
                now=fixed_now,
                consume_nonce=True,
            )
        )

        assert first_valid is True
        assert first_reason == "fresh_request"
        assert len(nonce_database) == 1

        replay_valid, replay_reason = (
            nonce_database
            .validate_request_freshness_and_replay(
                request=request,
                subscriber_db=subscriber_database,
                now=fixed_now,
                consume_nonce=True,
            )
        )

        assert replay_valid is False
        assert replay_reason == "replayed_nonce"

        stale_request = dict(request)
        stale_request["nonce"] = base64.b64encode(
            bytes.fromhex(
                "102132435465768798a9bacbdcedfe0f"
            )
        ).decode("ascii")

        stale_request["timestamp"] = (
            fixed_now - 61
        )

        stale_valid, stale_reason = (
            nonce_database
            .validate_request_freshness_and_replay(
                request=stale_request,
                subscriber_db=subscriber_database,
                now=fixed_now,
            )
        )

        assert stale_valid is False
        assert stale_reason == "stale_timestamp"

        unknown_request = dict(request)
        unknown_request["pseudonym_id"] = (
            "PID-UNKNOWN"
        )

        unknown_request["nonce"] = (
            base64.b64encode(
                bytes.fromhex(
                    "ffeeddccbbaa99887766554433221100"
                )
            ).decode("ascii")
        )

        unknown_valid, unknown_reason = (
            nonce_database
            .validate_request_freshness_and_replay(
                request=unknown_request,
                subscriber_db=subscriber_database,
                now=fixed_now,
            )
        )

        assert unknown_valid is False
        assert unknown_reason == "unknown_pseudonym"

        assert nonce_database.is_nonce_used(
            pseudonym_id="PID-6G-UE-0001",
            nonce=nonce_text,
        )

        record = nonce_database.get_record(
            pseudonym_id="PID-6G-UE-0001",
            nonce=nonce_text,
        )

        assert record is not None
        assert record.pseudonym_id == (
            "PID-6G-UE-0001"
        )
        assert record.service_context == "urban"

        reloaded_database = NonceDatabase(
            database_path=database_path,
            freshness_window_seconds=60,
        )

        assert len(reloaded_database) == 1

        replay_after_reload, reload_reason = (
            reloaded_database
            .validate_request_freshness_and_replay(
                request=request,
                subscriber_db=subscriber_database,
                now=fixed_now,
            )
        )

        assert replay_after_reload is False
        assert reload_reason == "replayed_nonce"

        stored_text = database_path.read_text(
            encoding="utf-8"
        )

        assert nonce_text not in stored_text
        assert nonce_bytes.hex() not in stored_text
        assert "nonce_digest" in stored_text
        assert "replay_id" in stored_text

        dictionary_cache: dict[str, float] = {}

        dictionary_request = dict(request)

        dictionary_result = (
            validate_request_freshness_and_replay(
                request=dictionary_request,
                subscriber_db=subscriber_database,
                replay_cache=dictionary_cache,
                now=fixed_now,
                consume_nonce=True,
            )
        )

        assert dictionary_result == (
            True,
            "fresh_request",
        )

        dictionary_replay = (
            validate_request_freshness_and_replay(
                request=dictionary_request,
                subscriber_db=subscriber_database,
                replay_cache=dictionary_cache,
                now=fixed_now,
                consume_nonce=True,
            )
        )

        assert dictionary_replay == (
            False,
            "replayed_nonce",
        )

        print("Nonce database self-test passed.")
        print(f"Database path: {database_path}")
        print(
            "Stored replay records: "
            f"{len(reloaded_database)}"
        )


if __name__ == "__main__":
    run_self_test()