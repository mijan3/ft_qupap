"""
Nonce generation and replay-attack protection for FT-QuPAP v5.1.

The Mobile Station generates a fresh cryptographic nonce for every
authentication request.

The Authentication Server stores accepted nonces temporarily. Reusing
the same nonce within the freshness period is treated as a replay attack.

This module provides:

- Secure nonce generation
- Nonce registration
- Replay detection
- Expired-nonce cleanup
- Optional JSON persistence
- Thread-safe nonce management
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.constants import (
    FRESHNESS_WINDOW_SECONDS,
    NONCE_SIZE_BYTES,
    USED_NONCE_DATABASE_PATH,
)

from src.common.exceptions import (
    ProtocolValidationError,
    ReplayAttackError,
    StorageError,
)

from src.common.random_manager import (
    generate_nonce_hex,
)

from src.common.serialization import (
    load_json_file,
    save_json_file,
)

from src.common.time_utils import (
    current_timestamp,
)

from src.common.validators import (
    validate_integer,
    validate_nonce_hex,
    validate_pseudonym_id,
)


@dataclass(frozen=True)
class NonceRecord:
    """
    Information stored for one accepted authentication nonce.
    """

    nonce: str
    pseudonym_id: str
    received_at: int
    expires_at: int

    def __post_init__(self) -> None:
        validate_nonce_hex(
            self.nonce,
            expected_bytes=NONCE_SIZE_BYTES,
        )

        validate_pseudonym_id(
            self.pseudonym_id
        )

        validate_integer(
            self.received_at,
            field_name="received_at",
            minimum=0,
        )

        validate_integer(
            self.expires_at,
            field_name="expires_at",
            minimum=0,
        )

        if self.expires_at <= self.received_at:
            raise ProtocolValidationError(
                "Nonce expiration must be later than its receive time.",
                details={
                    "received_at": self.received_at,
                    "expires_at": self.expires_at,
                },
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the record into JSON-compatible data.
        """

        return {
            "nonce": self.nonce,
            "pseudonym_id": self.pseudonym_id,
            "received_at": self.received_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "NonceRecord":
        """
        Restore a nonce record from dictionary data.
        """

        required_fields = (
            "nonce",
            "pseudonym_id",
            "received_at",
            "expires_at",
        )

        missing_fields = [
            field_name
            for field_name in required_fields
            if field_name not in data
        ]

        if missing_fields:
            raise ProtocolValidationError(
                "Stored nonce record is incomplete.",
                details={
                    "missing_fields": missing_fields,
                },
            )

        return cls(
            nonce=data["nonce"],
            pseudonym_id=data["pseudonym_id"],
            received_at=data["received_at"],
            expires_at=data["expires_at"],
        )


def create_authentication_nonce(
    byte_length: int = NONCE_SIZE_BYTES,
) -> str:
    """
    Generate a cryptographically secure authentication nonce.

    The default 16-byte nonce is represented by 32 hexadecimal
    characters.
    """

    validated_length = validate_integer(
        byte_length,
        field_name="byte_length",
        minimum=16,
        maximum=64,
    )

    nonce = generate_nonce_hex(
        validated_length
    )

    return validate_nonce_hex(
        nonce,
        expected_bytes=validated_length,
    )


class NonceManager:
    """
    Thread-safe replay-protection manager.

    The manager can operate in either:

    - Memory-only mode
    - JSON-persistent mode

    Persistent mode is used by the capstone Authentication Server so
    replay protection survives an application restart.
    """

    def __init__(
        self,
        *,
        storage_path: Path | str | None = USED_NONCE_DATABASE_PATH,
        freshness_window_seconds: int = FRESHNESS_WINDOW_SECONDS,
        persist: bool = True,
    ) -> None:
        self.freshness_window_seconds = validate_integer(
            freshness_window_seconds,
            field_name="freshness_window_seconds",
            minimum=1,
            maximum=86_400,
        )

        self.persist = bool(persist)

        self.storage_path = (
            Path(storage_path)
            if storage_path is not None
            else None
        )

        if self.persist and self.storage_path is None:
            raise ProtocolValidationError(
                "Persistent nonce management requires a storage path."
            )

        self._records: dict[str, NonceRecord] = {}
        self._lock = threading.RLock()

        if self.persist:
            self.load()

    def __len__(self) -> int:
        """
        Return the number of currently stored nonce records.
        """

        with self._lock:
            self.cleanup_expired()
            return len(self._records)

    def load(self) -> None:
        """
        Load nonce records from the configured JSON file.

        A missing file is treated as an empty nonce database.
        """

        if not self.persist or self.storage_path is None:
            return

        with self._lock:
            if not self.storage_path.exists():
                self._records = {}
                return

            try:
                stored_data = load_json_file(
                    self.storage_path,
                    restore_special_types=True,
                )
            except Exception as exc:
                raise StorageError(
                    "Unable to load the used-nonce database.",
                    path=str(self.storage_path),
                ) from exc

            if not isinstance(stored_data, dict):
                raise StorageError(
                    "The used-nonce database must contain a JSON object.",
                    path=str(self.storage_path),
                )

            raw_records = stored_data.get(
                "records",
                [],
            )

            if not isinstance(raw_records, list):
                raise StorageError(
                    "The nonce database records field must be a list.",
                    path=str(self.storage_path),
                )

            loaded_records: dict[str, NonceRecord] = {}

            for item in raw_records:
                if not isinstance(item, dict):
                    continue

                try:
                    record = NonceRecord.from_dict(
                        item
                    )
                except ProtocolValidationError:
                    continue

                loaded_records[record.nonce] = record

            self._records = loaded_records
            self.cleanup_expired(save_after_cleanup=False)

    def save(self) -> None:
        """
        Save all active nonce records to the configured JSON file.
        """

        if not self.persist or self.storage_path is None:
            return

        with self._lock:
            data = {
                "version": "5.1",
                "updated_at": current_timestamp(),
                "freshness_window_seconds": (
                    self.freshness_window_seconds
                ),
                "record_count": len(self._records),
                "records": [
                    record.to_dict()
                    for record in sorted(
                        self._records.values(),
                        key=lambda item: (
                            item.received_at,
                            item.nonce,
                        ),
                    )
                ],
            }

            try:
                save_json_file(
                    self.storage_path,
                    data,
                )
            except Exception as exc:
                raise StorageError(
                    "Unable to save the used-nonce database.",
                    path=str(self.storage_path),
                ) from exc

    def cleanup_expired(
        self,
        *,
        reference_time: int | None = None,
        save_after_cleanup: bool = True,
    ) -> int:
        """
        Remove expired nonce records.

        Returns the number of deleted records.
        """

        current = (
            current_timestamp()
            if reference_time is None
            else validate_integer(
                reference_time,
                field_name="reference_time",
                minimum=0,
            )
        )

        with self._lock:
            expired_nonces = [
                nonce
                for nonce, record
                in self._records.items()
                if record.expires_at < current
            ]

            for nonce in expired_nonces:
                del self._records[nonce]

            if (
                expired_nonces
                and save_after_cleanup
                and self.persist
            ):
                self.save()

            return len(expired_nonces)

    def has_nonce(
        self,
        nonce: str,
        *,
        reference_time: int | None = None,
    ) -> bool:
        """
        Return True when an active nonce record already exists.
        """

        validated_nonce = validate_nonce_hex(
            nonce,
            expected_bytes=NONCE_SIZE_BYTES,
        )

        with self._lock:
            self.cleanup_expired(
                reference_time=reference_time,
            )

            return validated_nonce in self._records

    def require_unused_nonce(
        self,
        nonce: str,
        *,
        reference_time: int | None = None,
    ) -> str:
        """
        Raise ReplayAttackError when a nonce has already been used.
        """

        validated_nonce = validate_nonce_hex(
            nonce,
            expected_bytes=NONCE_SIZE_BYTES,
        )

        if self.has_nonce(
            validated_nonce,
            reference_time=reference_time,
        ):
            raise ReplayAttackError(
                validated_nonce
            )

        return validated_nonce

    def register_nonce(
        self,
        nonce: str,
        pseudonym_id: str,
        *,
        received_at: int | None = None,
    ) -> NonceRecord:
        """
        Register a fresh nonce after request validation.

        Replay verification and registration are performed while holding
        the same lock to prevent two simultaneous requests from accepting
        the same nonce.
        """

        validated_nonce = validate_nonce_hex(
            nonce,
            expected_bytes=NONCE_SIZE_BYTES,
        )

        validated_pseudonym = validate_pseudonym_id(
            pseudonym_id
        )

        timestamp = (
            current_timestamp()
            if received_at is None
            else validate_integer(
                received_at,
                field_name="received_at",
                minimum=0,
            )
        )

        with self._lock:
            self.cleanup_expired(
                reference_time=timestamp,
                save_after_cleanup=False,
            )

            if validated_nonce in self._records:
                raise ReplayAttackError(
                    validated_nonce
                )

            record = NonceRecord(
                nonce=validated_nonce,
                pseudonym_id=validated_pseudonym,
                received_at=timestamp,
                expires_at=(
                    timestamp
                    + self.freshness_window_seconds
                ),
            )

            self._records[
                validated_nonce
            ] = record

            if self.persist:
                self.save()

            return record

    def check_and_register(
        self,
        nonce: str,
        pseudonym_id: str,
        *,
        received_at: int | None = None,
    ) -> NonceRecord:
        """
        Atomically verify that a nonce is unused and then register it.
        """

        return self.register_nonce(
            nonce=nonce,
            pseudonym_id=pseudonym_id,
            received_at=received_at,
        )

    def remove_nonce(
        self,
        nonce: str,
    ) -> bool:
        """
        Remove one nonce record.

        Returns True when a record existed.
        """

        validated_nonce = validate_nonce_hex(
            nonce,
            expected_bytes=NONCE_SIZE_BYTES,
        )

        with self._lock:
            existed = (
                validated_nonce
                in self._records
            )

            if existed:
                del self._records[
                    validated_nonce
                ]

                if self.persist:
                    self.save()

            return existed

    def clear(self) -> int:
        """
        Remove all nonce records.

        This method is intended for tests and demo reset operations.
        """

        with self._lock:
            deleted_count = len(
                self._records
            )

            self._records.clear()

            if self.persist:
                self.save()

            return deleted_count

    def get_record(
        self,
        nonce: str,
    ) -> NonceRecord | None:
        """
        Return one active nonce record.
        """

        validated_nonce = validate_nonce_hex(
            nonce,
            expected_bytes=NONCE_SIZE_BYTES,
        )

        with self._lock:
            self.cleanup_expired()

            return self._records.get(
                validated_nonce
            )

    def list_records(self) -> list[NonceRecord]:
        """
        Return all active nonce records ordered by receive time.
        """

        with self._lock:
            self.cleanup_expired()

            return sorted(
                self._records.values(),
                key=lambda record: (
                    record.received_at,
                    record.nonce,
                ),
            )

    def statistics(self) -> dict[str, Any]:
        """
        Return non-sensitive replay-protection statistics.
        """

        with self._lock:
            removed_count = self.cleanup_expired()

            records = self.list_records()

            pseudonyms = {
                record.pseudonym_id
                for record in records
            }

            return {
                "active_nonce_count": len(records),
                "unique_pseudonym_count": len(
                    pseudonyms
                ),
                "expired_records_removed": (
                    removed_count
                ),
                "freshness_window_seconds": (
                    self.freshness_window_seconds
                ),
                "persistent": self.persist,
                "storage_path": (
                    str(self.storage_path)
                    if self.storage_path is not None
                    else None
                ),
            }


def run_nonce_manager_self_test() -> dict[str, Any]:
    """
    Run an in-memory replay-protection self-test.
    """

    manager = NonceManager(
        storage_path=None,
        freshness_window_seconds=60,
        persist=False,
    )

    test_time = 1_700_000_000

    nonce = create_authentication_nonce()

    first_registration = manager.check_and_register(
        nonce=nonce,
        pseudonym_id="PID-SELF-TEST-001",
        received_at=test_time,
    )

    replay_detected = False

    try:
        manager.check_and_register(
            nonce=nonce,
            pseudonym_id="PID-SELF-TEST-001",
            received_at=test_time + 1,
        )
    except ReplayAttackError:
        replay_detected = True

    active_before_expiration = manager.has_nonce(
        nonce,
        reference_time=test_time + 30,
    )

    removed_count = manager.cleanup_expired(
        reference_time=test_time + 61,
    )

    absent_after_expiration = not manager.has_nonce(
        nonce,
        reference_time=test_time + 61,
    )

    fresh_nonce = create_authentication_nonce()

    unique_nonce_generated = (
        fresh_nonce != nonce
    )

    success = all(
        (
            first_registration.nonce == nonce,
            replay_detected,
            active_before_expiration,
            removed_count == 1,
            absent_after_expiration,
            unique_nonce_generated,
        )
    )

    return {
        "success": success,
        "nonce_bytes": NONCE_SIZE_BYTES,
        "nonce_hex_characters": len(nonce),
        "replay_detected": replay_detected,
        "active_before_expiration": (
            active_before_expiration
        ),
        "expired_record_removed": (
            removed_count == 1
        ),
        "absent_after_expiration": (
            absent_after_expiration
        ),
        "unique_nonce_generated": (
            unique_nonce_generated
        ),
    }


__all__ = [
    "NonceRecord",
    "NonceManager",
    "create_authentication_nonce",
    "run_nonce_manager_self_test",
]