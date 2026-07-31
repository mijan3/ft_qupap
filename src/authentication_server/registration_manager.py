"""
Subscriber registration management for FT-QuPAP v5.1.

During secure offline registration, the Authentication Server stores a
subscriber record containing:

- Internal registration identifier
- Hashed permanent subscriber identity
- Current pseudonymous identity
- Subscriber identity key
- Pseudonym rotation epoch
- Registration status and timestamps

The permanent identity is not transmitted during normal FT-QuPAP
authentication. Authentication requests use only the current
pseudonymous identity.

This module provides:

- Subscriber registration
- Duplicate-registration protection
- Pseudonym lookup
- Permanent-identity hash lookup
- Subscriber activation and deactivation
- Pseudonym rotation
- Identity-key verification
- Atomic JSON persistence
- Thread-safe registry access

The identity key is sensitive and is excluded from public
representations and normal object output.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from src.common.constants import (
    PROTOCOL_DOMAIN_LABEL,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.serialization import (
    canonical_json_bytes,
    decode_base64,
    encode_base64,
)

from src.common.time_utils import (
    current_timestamp,
)

from src.common.validators import (
    validate_bytes,
    validate_integer,
    validate_non_empty_string,
    validate_pseudonym_id,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REGISTRATION_DATABASE_VERSION = "FT-QuPAP-Registration-v1"

DEFAULT_REGISTRATION_DATABASE_PATH = Path(
    "data/registration/subscribers.json"
)

MINIMUM_IDENTITY_KEY_BYTES = 32

MAXIMUM_IDENTITY_KEY_BYTES = 4096

REGISTRATION_FILE_PERMISSION = 0o600

PERMANENT_IDENTITY_HASH_ALGORITHM = "SHA3-256"


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class RegistrationManagerError(RuntimeError):
    """Raised when subscriber registration management fails."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.details = (
            {}
            if details is None
            else dict(details)
        )


# ---------------------------------------------------------------------
# Subscriber record
# ---------------------------------------------------------------------

@dataclass
class SubscriberRegistrationRecord:
    """
    Authentication Server subscriber-registration record.

    Attributes
    ----------
    registration_id:
        Internal non-secret registration identifier.

    permanent_identity_hash:
        SHA3-256 hash of the permanent subscriber identity.

    pseudonym_id:
        Current public pseudonymous identity used in M1.

    identity_key:
        Subscriber-specific secret provisioned during registration.

    pseudonym_epoch:
        Number of successful pseudonym rotations.

    previous_pseudonyms:
        Historical pseudonyms retained for replay and migration checks.

    active:
        Whether authentication is currently permitted.
    """

    registration_id: str
    permanent_identity_hash: str

    pseudonym_id: str
    identity_key: bytes

    pseudonym_epoch: int = 0

    active: bool = True

    registered_at: int = 0
    updated_at: int = 0

    previous_pseudonyms: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.registration_id = (
            validate_non_empty_string(
                self.registration_id,
                field_name="registration_id",
                minimum_length=8,
                maximum_length=128,
            )
        )

        self.permanent_identity_hash = (
            validate_non_empty_string(
                self.permanent_identity_hash,
                field_name=(
                    "permanent_identity_hash"
                ),
                minimum_length=64,
                maximum_length=64,
            )
            .lower()
        )

        try:
            bytes.fromhex(
                self.permanent_identity_hash
            )

        except ValueError as exc:
            raise ProtocolValidationError(
                (
                    "permanent_identity_hash must be "
                    "valid hexadecimal text."
                )
            ) from exc

        self.pseudonym_id = (
            validate_pseudonym_id(
                self.pseudonym_id
            )
        )

        self.identity_key = validate_bytes(
            self.identity_key,
            field_name="identity_key",
            minimum_length=(
                MINIMUM_IDENTITY_KEY_BYTES
            ),
            maximum_length=(
                MAXIMUM_IDENTITY_KEY_BYTES
            ),
        )

        self.pseudonym_epoch = (
            validate_integer(
                self.pseudonym_epoch,
                field_name="pseudonym_epoch",
                minimum=0,
            )
        )

        if not isinstance(
            self.active,
            bool,
        ):
            raise ProtocolValidationError(
                "active must be Boolean."
            )

        self.registered_at = (
            validate_integer(
                self.registered_at,
                field_name="registered_at",
                minimum=0,
            )
        )

        self.updated_at = validate_integer(
            self.updated_at,
            field_name="updated_at",
            minimum=0,
        )

        if self.updated_at < self.registered_at:
            raise ProtocolValidationError(
                (
                    "updated_at cannot be earlier "
                    "than registered_at."
                )
            )

        if not isinstance(
            self.previous_pseudonyms,
            list,
        ):
            raise ProtocolValidationError(
                (
                    "previous_pseudonyms must "
                    "be a list."
                )
            )

        normalized_previous: list[str] = []

        for index, pseudonym in enumerate(
            self.previous_pseudonyms
        ):
            normalized = validate_pseudonym_id(
                pseudonym
            )

            if normalized == self.pseudonym_id:
                raise ProtocolValidationError(
                    (
                        "The current pseudonym cannot "
                        "also appear in previous_pseudonyms."
                    ),
                    details={
                        "index": index,
                        "pseudonym_id": normalized,
                    },
                )

            normalized_previous.append(
                normalized
            )

        if (
            len(set(normalized_previous))
            != len(normalized_previous)
        ):
            raise ProtocolValidationError(
                (
                    "previous_pseudonyms contains "
                    "duplicate values."
                )
            )

        self.previous_pseudonyms = (
            normalized_previous
        )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise ProtocolValidationError(
                "metadata must be a dictionary."
            )

    def public_dict(self) -> dict[str, Any]:
        """
        Return non-sensitive subscriber information.
        """

        return {
            "registration_id": (
                self.registration_id
            ),
            "pseudonym_id": self.pseudonym_id,
            "pseudonym_epoch": (
                self.pseudonym_epoch
            ),
            "active": self.active,
            "registered_at": (
                self.registered_at
            ),
            "updated_at": self.updated_at,
            "metadata": dict(
                self.metadata
            ),
        }

    def protected_dict(self) -> dict[str, Any]:
        """
        Return the complete record for protected local storage.

        The returned dictionary contains secret identity-key material
        and must never be written to ordinary logs.
        """

        return {
            "registration_id": (
                self.registration_id
            ),
            "permanent_identity_hash": (
                self.permanent_identity_hash
            ),
            "pseudonym_id": self.pseudonym_id,
            "identity_key": encode_base64(
                self.identity_key
            ),
            "pseudonym_epoch": (
                self.pseudonym_epoch
            ),
            "active": self.active,
            "registered_at": (
                self.registered_at
            ),
            "updated_at": self.updated_at,
            "previous_pseudonyms": list(
                self.previous_pseudonyms
            ),
            "metadata": dict(
                self.metadata
            ),
        }

    @classmethod
    def from_protected_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "SubscriberRegistrationRecord":
        """
        Restore a protected subscriber record.
        """

        if not isinstance(
            data,
            Mapping,
        ):
            raise RegistrationManagerError(
                (
                    "Stored subscriber record must "
                    "be a mapping."
                )
            )

        required_fields = (
            "registration_id",
            "permanent_identity_hash",
            "pseudonym_id",
            "identity_key",
            "registered_at",
            "updated_at",
        )

        missing_fields = [
            field_name
            for field_name in required_fields
            if field_name not in data
        ]

        if missing_fields:
            raise RegistrationManagerError(
                (
                    "Stored subscriber record "
                    "is incomplete."
                ),
                details={
                    "missing_fields": (
                        missing_fields
                    ),
                },
            )

        try:
            identity_key = decode_base64(
                data["identity_key"]
            )

        except Exception as exc:
            raise RegistrationManagerError(
                (
                    "Unable to decode the stored "
                    "subscriber identity key."
                ),
                details={
                    "reason": str(exc),
                },
            ) from exc

        return cls(
            registration_id=data[
                "registration_id"
            ],
            permanent_identity_hash=data[
                "permanent_identity_hash"
            ],
            pseudonym_id=data[
                "pseudonym_id"
            ],
            identity_key=identity_key,
            pseudonym_epoch=data.get(
                "pseudonym_epoch",
                0,
            ),
            active=data.get(
                "active",
                True,
            ),
            registered_at=data[
                "registered_at"
            ],
            updated_at=data[
                "updated_at"
            ],
            previous_pseudonyms=list(
                data.get(
                    "previous_pseudonyms",
                    [],
                )
            ),
            metadata=dict(
                data.get(
                    "metadata",
                    {},
                )
            ),
        )

    def verify_identity_key(
        self,
        candidate_key: bytes,
    ) -> bool:
        """
        Verify a subscriber identity key in constant time.
        """

        validated_candidate = (
            validate_bytes(
                candidate_key,
                field_name=(
                    "candidate_identity_key"
                ),
                minimum_length=(
                    MINIMUM_IDENTITY_KEY_BYTES
                ),
                maximum_length=(
                    MAXIMUM_IDENTITY_KEY_BYTES
                ),
            )
        )

        return hmac.compare_digest(
            self.identity_key,
            validated_candidate,
        )

    def matches_permanent_identity(
        self,
        permanent_identity: str,
    ) -> bool:
        """
        Compare a permanent identity using its SHA3-256 hash.
        """

        candidate_hash = (
            hash_permanent_identity(
                permanent_identity
            )
        )

        return hmac.compare_digest(
            self.permanent_identity_hash,
            candidate_hash,
        )

    def __repr__(self) -> str:
        return (
            "SubscriberRegistrationRecord("
            f"registration_id={self.registration_id!r}, "
            f"pseudonym_id={self.pseudonym_id!r}, "
            f"pseudonym_epoch={self.pseudonym_epoch}, "
            f"active={self.active}, "
            f"registered_at={self.registered_at}, "
            f"updated_at={self.updated_at}, "
            "permanent_identity_hash=<hidden>, "
            "identity_key=<hidden>"
            ")"
        )


# ---------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------

def normalize_permanent_identity(
    permanent_identity: str,
) -> str:
    """
    Normalize a permanent subscriber identity.

    The exact identity is used only during secure registration and
    protected administrative lookup.
    """

    return validate_non_empty_string(
        permanent_identity,
        field_name="permanent_identity",
        minimum_length=3,
        maximum_length=256,
    ).strip()


def hash_permanent_identity(
    permanent_identity: str,
) -> str:
    """
    Calculate a domain-separated SHA3-256 permanent-identity hash.
    """

    normalized_identity = (
        normalize_permanent_identity(
            permanent_identity
        )
    )

    digest = hashlib.sha3_256()

    digest.update(
        PROTOCOL_DOMAIN_LABEL
    )

    digest.update(
        b"\x00subscriber-registration\x00"
    )

    digest.update(
        normalized_identity.encode(
            "utf-8"
        )
    )

    return digest.hexdigest()


def create_registration_id(
    *,
    permanent_identity_hash: str,
    registered_at: int,
    random_value: bytes | None = None,
) -> str:
    """
    Create a unique internal registration identifier.
    """

    validated_hash = (
        validate_non_empty_string(
            permanent_identity_hash,
            field_name=(
                "permanent_identity_hash"
            ),
            minimum_length=64,
            maximum_length=64,
        )
        .lower()
    )

    validated_timestamp = (
        validate_integer(
            registered_at,
            field_name="registered_at",
            minimum=0,
        )
    )

    selected_random_value = (
        secrets.token_bytes(16)
        if random_value is None
        else validate_bytes(
            random_value,
            field_name=(
                "registration_random_value"
            ),
            minimum_length=16,
            maximum_length=64,
        )
    )

    payload = {
        "domain": PROTOCOL_DOMAIN_LABEL.decode(
            "utf-8",
            errors="strict",
        ),
        "purpose": (
            "subscriber-registration-id"
        ),
        "permanent_identity_hash": (
            validated_hash
        ),
        "registered_at": (
            validated_timestamp
        ),
        "random_value": encode_base64(
            selected_random_value
        ),
    }

    digest = hashlib.sha3_256(
        canonical_json_bytes(
            payload
        )
    ).hexdigest()

    return (
        "REG-"
        + digest[:32].upper()
    )


# ---------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------

def _read_database_file(
    path: Path,
) -> dict[str, Any]:
    """
    Read the protected registration database.
    """

    if not path.exists():
        return {
            "version": (
                REGISTRATION_DATABASE_VERSION
            ),
            "updated_at": 0,
            "records": [],
        }

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

    except Exception as exc:
        raise RegistrationManagerError(
            (
                "Unable to read the subscriber "
                "registration database."
            ),
            details={
                "path": str(path),
                "reason": str(exc),
            },
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise RegistrationManagerError(
            (
                "Registration database must "
                "contain a JSON object."
            )
        )

    return data


def _write_database_file(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    """
    Atomically write the protected registration database.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    encoded = canonical_json_bytes(
        dict(data)
    )

    try:
        with temporary_path.open(
            "wb",
        ) as file:
            file.write(
                encoded
            )

            file.flush()

            try:
                os.fsync(
                    file.fileno()
                )

            except OSError:
                pass

        try:
            os.chmod(
                temporary_path,
                REGISTRATION_FILE_PERMISSION,
            )

        except OSError:
            pass

        os.replace(
            temporary_path,
            path,
        )

        try:
            os.chmod(
                path,
                REGISTRATION_FILE_PERMISSION,
            )

        except OSError:
            pass

    except Exception as exc:
        try:
            temporary_path.unlink(
                missing_ok=True
            )

        except OSError:
            pass

        raise RegistrationManagerError(
            (
                "Unable to save the subscriber "
                "registration database."
            ),
            details={
                "path": str(path),
                "reason": str(exc),
            },
        ) from exc


# ---------------------------------------------------------------------
# Registration manager
# ---------------------------------------------------------------------

class RegistrationManager:
    """
    Thread-safe FT-QuPAP subscriber registration manager.
    """

    def __init__(
        self,
        *,
        storage_path: Path | str = (
            DEFAULT_REGISTRATION_DATABASE_PATH
        ),
        auto_load: bool = True,
    ) -> None:
        self.storage_path = Path(
            storage_path
        )

        if not isinstance(
            auto_load,
            bool,
        ):
            raise ProtocolValidationError(
                "auto_load must be Boolean."
            )

        self._records: dict[
            str,
            SubscriberRegistrationRecord,
        ] = {}

        self._pseudonym_index: dict[
            str,
            str,
        ] = {}

        self._identity_hash_index: dict[
            str,
            str,
        ] = {}

        self._historical_pseudonym_index: dict[
            str,
            str,
        ] = {}

        self._lock = threading.RLock()

        if auto_load and self.storage_path.exists():
            self.load()

    def _rebuild_indexes(self) -> None:
        """
        Rebuild and validate all lookup indexes.
        """

        pseudonym_index: dict[
            str,
            str,
        ] = {}

        identity_hash_index: dict[
            str,
            str,
        ] = {}

        historical_index: dict[
            str,
            str,
        ] = {}

        for (
            registration_id,
            record,
        ) in self._records.items():
            if (
                record.pseudonym_id
                in pseudonym_index
            ):
                raise RegistrationManagerError(
                    (
                        "Duplicate active pseudonym "
                        "found in registration database."
                    ),
                    details={
                        "pseudonym_id": (
                            record.pseudonym_id
                        ),
                    },
                )

            if (
                record.permanent_identity_hash
                in identity_hash_index
            ):
                raise RegistrationManagerError(
                    (
                        "Duplicate permanent identity "
                        "found in registration database."
                    )
                )

            pseudonym_index[
                record.pseudonym_id
            ] = registration_id

            identity_hash_index[
                record.permanent_identity_hash
            ] = registration_id

            for previous_pseudonym in (
                record.previous_pseudonyms
            ):
                if (
                    previous_pseudonym
                    in pseudonym_index
                    or previous_pseudonym
                    in historical_index
                ):
                    raise RegistrationManagerError(
                        (
                            "Duplicate historical pseudonym "
                            "found in registration database."
                        ),
                        details={
                            "pseudonym_id": (
                                previous_pseudonym
                            ),
                        },
                    )

                historical_index[
                    previous_pseudonym
                ] = registration_id

        self._pseudonym_index = (
            pseudonym_index
        )

        self._identity_hash_index = (
            identity_hash_index
        )

        self._historical_pseudonym_index = (
            historical_index
        )

    def load(self) -> int:
        """
        Load and validate all subscriber records.

        Returns the number of loaded subscribers.
        """

        with self._lock:
            database = _read_database_file(
                self.storage_path
            )

            version = database.get(
                "version"
            )

            if (
                version
                != REGISTRATION_DATABASE_VERSION
            ):
                raise RegistrationManagerError(
                    (
                        "Unsupported registration "
                        "database version."
                    ),
                    details={
                        "received_version": version,
                        "expected_version": (
                            REGISTRATION_DATABASE_VERSION
                        ),
                    },
                )

            raw_records = database.get(
                "records"
            )

            if not isinstance(
                raw_records,
                list,
            ):
                raise RegistrationManagerError(
                    (
                        "Registration database records "
                        "must be a list."
                    )
                )

            loaded_records: dict[
                str,
                SubscriberRegistrationRecord,
            ] = {}

            for index, raw_record in enumerate(
                raw_records
            ):
                try:
                    record = (
                        SubscriberRegistrationRecord
                        .from_protected_dict(
                            raw_record
                        )
                    )

                except Exception as exc:
                    raise RegistrationManagerError(
                        (
                            "Unable to load subscriber "
                            "registration record."
                        ),
                        details={
                            "record_index": index,
                            "reason": str(exc),
                        },
                    ) from exc

                if (
                    record.registration_id
                    in loaded_records
                ):
                    raise RegistrationManagerError(
                        (
                            "Duplicate registration ID "
                            "found in database."
                        ),
                        details={
                            "registration_id": (
                                record.registration_id
                            ),
                        },
                    )

                loaded_records[
                    record.registration_id
                ] = record

            self._records = (
                loaded_records
            )

            self._rebuild_indexes()

            return len(
                self._records
            )

    def save(
        self,
        *,
        timestamp: int | None = None,
    ) -> None:
        """
        Persist all subscriber records atomically.
        """

        selected_timestamp = (
            current_timestamp()
            if timestamp is None
            else validate_integer(
                timestamp,
                field_name="timestamp",
                minimum=0,
            )
        )

        with self._lock:
            database = {
                "version": (
                    REGISTRATION_DATABASE_VERSION
                ),
                "updated_at": (
                    selected_timestamp
                ),
                "record_count": len(
                    self._records
                ),
                "records": [
                    record.protected_dict()
                    for record in sorted(
                        self._records.values(),
                        key=lambda item: (
                            item.registration_id
                        ),
                    )
                ],
            }

            _write_database_file(
                self.storage_path,
                database,
            )

    def register_subscriber(
        self,
        *,
        permanent_identity: str,
        pseudonym_id: str,
        identity_key: bytes,
        metadata: Mapping[str, Any] | None = None,
        registered_at: int | None = None,
        registration_random_value: bytes | None = None,
        persist: bool = True,
    ) -> SubscriberRegistrationRecord:
        """
        Register a new FT-QuPAP subscriber.

        Duplicate permanent identities and duplicate pseudonyms are
        rejected.
        """

        permanent_identity_hash = (
            hash_permanent_identity(
                permanent_identity
            )
        )

        validated_pseudonym = (
            validate_pseudonym_id(
                pseudonym_id
            )
        )

        validated_identity_key = (
            validate_bytes(
                identity_key,
                field_name="identity_key",
                minimum_length=(
                    MINIMUM_IDENTITY_KEY_BYTES
                ),
                maximum_length=(
                    MAXIMUM_IDENTITY_KEY_BYTES
                ),
            )
        )

        selected_timestamp = (
            current_timestamp()
            if registered_at is None
            else validate_integer(
                registered_at,
                field_name="registered_at",
                minimum=0,
            )
        )

        normalized_metadata = (
            {}
            if metadata is None
            else dict(metadata)
        )

        if not isinstance(
            persist,
            bool,
        ):
            raise ProtocolValidationError(
                "persist must be Boolean."
            )

        with self._lock:
            if (
                permanent_identity_hash
                in self._identity_hash_index
            ):
                raise RegistrationManagerError(
                    (
                        "This permanent subscriber "
                        "identity is already registered."
                    ),
                    details={
                        "registration_id": (
                            self._identity_hash_index[
                                permanent_identity_hash
                            ]
                        ),
                    },
                )

            if (
                validated_pseudonym
                in self._pseudonym_index
                or validated_pseudonym
                in self._historical_pseudonym_index
            ):
                raise RegistrationManagerError(
                    (
                        "The pseudonymous identity is "
                        "already registered or was "
                        "previously used."
                    ),
                    details={
                        "pseudonym_id": (
                            validated_pseudonym
                        ),
                    },
                )

            registration_id = (
                create_registration_id(
                    permanent_identity_hash=(
                        permanent_identity_hash
                    ),
                    registered_at=(
                        selected_timestamp
                    ),
                    random_value=(
                        registration_random_value
                    ),
                )
            )

            while (
                registration_id
                in self._records
            ):
                registration_id = (
                    create_registration_id(
                        permanent_identity_hash=(
                            permanent_identity_hash
                        ),
                        registered_at=(
                            selected_timestamp
                        ),
                    )
                )

            record = (
                SubscriberRegistrationRecord(
                    registration_id=(
                        registration_id
                    ),
                    permanent_identity_hash=(
                        permanent_identity_hash
                    ),
                    pseudonym_id=(
                        validated_pseudonym
                    ),
                    identity_key=(
                        validated_identity_key
                    ),
                    pseudonym_epoch=0,
                    active=True,
                    registered_at=(
                        selected_timestamp
                    ),
                    updated_at=(
                        selected_timestamp
                    ),
                    metadata=(
                        normalized_metadata
                    ),
                )
            )

            self._records[
                registration_id
            ] = record

            self._rebuild_indexes()

            if persist:
                self.save(
                    timestamp=selected_timestamp
                )

            return record

    def get_by_registration_id(
        self,
        registration_id: str,
    ) -> SubscriberRegistrationRecord:
        """
        Retrieve a subscriber using the internal registration ID.
        """

        validated_id = (
            validate_non_empty_string(
                registration_id,
                field_name="registration_id",
                minimum_length=8,
                maximum_length=128,
            )
        )

        with self._lock:
            record = self._records.get(
                validated_id
            )

            if record is None:
                raise RegistrationManagerError(
                    "Subscriber registration was not found.",
                    details={
                        "registration_id": (
                            validated_id
                        ),
                    },
                )

            return record

    def get_by_pseudonym(
        self,
        pseudonym_id: str,
        *,
        require_active: bool = False,
    ) -> SubscriberRegistrationRecord:
        """
        Retrieve a subscriber using the current pseudonym.
        """

        validated_pseudonym = (
            validate_pseudonym_id(
                pseudonym_id
            )
        )

        if not isinstance(
            require_active,
            bool,
        ):
            raise ProtocolValidationError(
                "require_active must be Boolean."
            )

        with self._lock:
            registration_id = (
                self._pseudonym_index.get(
                    validated_pseudonym
                )
            )

            if registration_id is None:
                if (
                    validated_pseudonym
                    in self._historical_pseudonym_index
                ):
                    raise RegistrationManagerError(
                        (
                            "The supplied pseudonym is "
                            "an expired historical identity."
                        ),
                        details={
                            "pseudonym_id": (
                                validated_pseudonym
                            ),
                        },
                    )

                raise RegistrationManagerError(
                    (
                        "No subscriber is registered "
                        "with this pseudonym."
                    ),
                    details={
                        "pseudonym_id": (
                            validated_pseudonym
                        ),
                    },
                )

            record = self._records[
                registration_id
            ]

            if (
                require_active
                and not record.active
            ):
                raise RegistrationManagerError(
                    (
                        "The subscriber registration "
                        "is inactive."
                    ),
                    details={
                        "registration_id": (
                            record.registration_id
                        ),
                        "pseudonym_id": (
                            record.pseudonym_id
                        ),
                    },
                )

            return record

    def get_by_permanent_identity(
        self,
        permanent_identity: str,
    ) -> SubscriberRegistrationRecord:
        """
        Retrieve a registration using the permanent identity hash.
        """

        identity_hash = (
            hash_permanent_identity(
                permanent_identity
            )
        )

        with self._lock:
            registration_id = (
                self._identity_hash_index.get(
                    identity_hash
                )
            )

            if registration_id is None:
                raise RegistrationManagerError(
                    (
                        "No subscriber is registered "
                        "with this permanent identity."
                    )
                )

            return self._records[
                registration_id
            ]

    def activate_subscriber(
        self,
        *,
        registration_id: str,
        timestamp: int | None = None,
        persist: bool = True,
    ) -> SubscriberRegistrationRecord:
        """
        Permit authentication for a subscriber.
        """

        return self._set_active_status(
            registration_id=registration_id,
            active=True,
            timestamp=timestamp,
            persist=persist,
        )

    def deactivate_subscriber(
        self,
        *,
        registration_id: str,
        timestamp: int | None = None,
        persist: bool = True,
    ) -> SubscriberRegistrationRecord:
        """
        Block authentication for a subscriber.
        """

        return self._set_active_status(
            registration_id=registration_id,
            active=False,
            timestamp=timestamp,
            persist=persist,
        )

    def _set_active_status(
        self,
        *,
        registration_id: str,
        active: bool,
        timestamp: int | None,
        persist: bool,
    ) -> SubscriberRegistrationRecord:
        """
        Update one subscriber's active status.
        """

        if not isinstance(
            active,
            bool,
        ):
            raise ProtocolValidationError(
                "active must be Boolean."
            )

        if not isinstance(
            persist,
            bool,
        ):
            raise ProtocolValidationError(
                "persist must be Boolean."
            )

        selected_timestamp = (
            current_timestamp()
            if timestamp is None
            else validate_integer(
                timestamp,
                field_name="timestamp",
                minimum=0,
            )
        )

        with self._lock:
            record = self.get_by_registration_id(
                registration_id
            )

            record.active = active
            record.updated_at = (
                selected_timestamp
            )

            if persist:
                self.save(
                    timestamp=selected_timestamp
                )

            return record

    def rotate_pseudonym(
        self,
        *,
        registration_id: str,
        new_pseudonym_id: str,
        new_identity_key: bytes | None = None,
        timestamp: int | None = None,
        persist: bool = True,
    ) -> SubscriberRegistrationRecord:
        """
        Replace the subscriber's public pseudonym.

        The old pseudonym is retained as historical and cannot be reused
        by another subscriber.
        """

        validated_new_pseudonym = (
            validate_pseudonym_id(
                new_pseudonym_id
            )
        )

        selected_timestamp = (
            current_timestamp()
            if timestamp is None
            else validate_integer(
                timestamp,
                field_name="timestamp",
                minimum=0,
            )
        )

        if not isinstance(
            persist,
            bool,
        ):
            raise ProtocolValidationError(
                "persist must be Boolean."
            )

        with self._lock:
            record = self.get_by_registration_id(
                registration_id
            )

            if (
                validated_new_pseudonym
                == record.pseudonym_id
            ):
                raise RegistrationManagerError(
                    (
                        "The new pseudonym must differ "
                        "from the current pseudonym."
                    )
                )

            if (
                validated_new_pseudonym
                in self._pseudonym_index
                or validated_new_pseudonym
                in self._historical_pseudonym_index
            ):
                raise RegistrationManagerError(
                    (
                        "The new pseudonym is already "
                        "registered or was previously used."
                    ),
                    details={
                        "pseudonym_id": (
                            validated_new_pseudonym
                        ),
                    },
                )

            old_pseudonym = (
                record.pseudonym_id
            )

            record.previous_pseudonyms.append(
                old_pseudonym
            )

            record.pseudonym_id = (
                validated_new_pseudonym
            )

            record.pseudonym_epoch += 1

            if new_identity_key is not None:
                record.identity_key = (
                    validate_bytes(
                        new_identity_key,
                        field_name=(
                            "new_identity_key"
                        ),
                        minimum_length=(
                            MINIMUM_IDENTITY_KEY_BYTES
                        ),
                        maximum_length=(
                            MAXIMUM_IDENTITY_KEY_BYTES
                        ),
                    )
                )

            record.updated_at = (
                selected_timestamp
            )

            self._rebuild_indexes()

            if persist:
                self.save(
                    timestamp=selected_timestamp
                )

            return record

    def verify_subscriber_identity_key(
        self,
        *,
        pseudonym_id: str,
        candidate_identity_key: bytes,
        require_active: bool = True,
    ) -> bool:
        """
        Verify a subscriber-specific identity key.
        """

        record = self.get_by_pseudonym(
            pseudonym_id,
            require_active=require_active,
        )

        return record.verify_identity_key(
            candidate_identity_key
        )

    def is_historical_pseudonym(
        self,
        pseudonym_id: str,
    ) -> bool:
        """
        Return True when a pseudonym was previously used.
        """

        validated_pseudonym = (
            validate_pseudonym_id(
                pseudonym_id
            )
        )

        with self._lock:
            return (
                validated_pseudonym
                in self._historical_pseudonym_index
            )

    def subscriber_count(
        self,
        *,
        active_only: bool = False,
    ) -> int:
        """
        Return the number of registered subscribers.
        """

        if not isinstance(
            active_only,
            bool,
        ):
            raise ProtocolValidationError(
                "active_only must be Boolean."
            )

        with self._lock:
            if not active_only:
                return len(
                    self._records
                )

            return sum(
                1
                for record
                in self._records.values()
                if record.active
            )

    def list_public_records(
        self,
        *,
        active_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Return non-sensitive registration information.
        """

        if not isinstance(
            active_only,
            bool,
        ):
            raise ProtocolValidationError(
                "active_only must be Boolean."
            )

        with self._lock:
            records = [
                record
                for record
                in self._records.values()
                if (
                    not active_only
                    or record.active
                )
            ]

            records.sort(
                key=lambda item: (
                    item.registration_id
                )
            )

            return [
                record.public_dict()
                for record in records
            ]


# ---------------------------------------------------------------------
# Default manager
# ---------------------------------------------------------------------

_DEFAULT_REGISTRATION_MANAGER: (
    RegistrationManager | None
) = None

_DEFAULT_MANAGER_LOCK = threading.RLock()


def get_default_registration_manager() -> RegistrationManager:
    """
    Return the process-wide registration manager.
    """

    global _DEFAULT_REGISTRATION_MANAGER

    with _DEFAULT_MANAGER_LOCK:
        if (
            _DEFAULT_REGISTRATION_MANAGER
            is None
        ):
            _DEFAULT_REGISTRATION_MANAGER = (
                RegistrationManager()
            )

        return _DEFAULT_REGISTRATION_MANAGER


def reset_default_registration_manager() -> None:
    """
    Reset the process-wide registration manager.
    """

    global _DEFAULT_REGISTRATION_MANAGER

    with _DEFAULT_MANAGER_LOCK:
        _DEFAULT_REGISTRATION_MANAGER = None


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_registration_manager_self_test() -> dict[str, Any]:
    """
    Test registration, lookup, activation, rotation, and persistence.
    """

    try:
        with TemporaryDirectory() as directory:
            storage_path = (
                Path(directory)
                / "subscribers.json"
            )

            manager = RegistrationManager(
                storage_path=storage_path,
                auto_load=False,
            )

            identity_key = bytes(
                range(32)
            )

            record = manager.register_subscriber(
                permanent_identity=(
                    "IMSI-470010000000001"
                ),
                pseudonym_id=(
                    "PID-FTQ-SELF-TEST-001"
                ),
                identity_key=identity_key,
                metadata={
                    "context": "urban",
                },
                registered_at=(
                    1_700_000_000
                ),
                registration_random_value=(
                    b"\x01" * 16
                ),
            )

            pseudonym_lookup_pass = (
                manager.get_by_pseudonym(
                    "PID-FTQ-SELF-TEST-001",
                    require_active=True,
                ).registration_id
                == record.registration_id
            )

            permanent_lookup_pass = (
                manager.get_by_permanent_identity(
                    "IMSI-470010000000001"
                ).registration_id
                == record.registration_id
            )

            identity_key_pass = (
                manager.verify_subscriber_identity_key(
                    pseudonym_id=(
                        "PID-FTQ-SELF-TEST-001"
                    ),
                    candidate_identity_key=(
                        identity_key
                    ),
                )
            )

            wrong_identity_key_rejected = (
                not manager
                .verify_subscriber_identity_key(
                    pseudonym_id=(
                        "PID-FTQ-SELF-TEST-001"
                    ),
                    candidate_identity_key=(
                        b"\xFF" * 32
                    ),
                )
            )

            duplicate_identity_rejected = False

            try:
                manager.register_subscriber(
                    permanent_identity=(
                        "IMSI-470010000000001"
                    ),
                    pseudonym_id=(
                        "PID-FTQ-SELF-TEST-002"
                    ),
                    identity_key=(
                        b"\x02" * 32
                    ),
                    registered_at=(
                        1_700_000_001
                    ),
                )

            except RegistrationManagerError:
                duplicate_identity_rejected = True

            manager.deactivate_subscriber(
                registration_id=(
                    record.registration_id
                ),
                timestamp=1_700_000_010,
            )

            inactive_rejected = False

            try:
                manager.get_by_pseudonym(
                    "PID-FTQ-SELF-TEST-001",
                    require_active=True,
                )

            except RegistrationManagerError:
                inactive_rejected = True

            manager.activate_subscriber(
                registration_id=(
                    record.registration_id
                ),
                timestamp=1_700_000_020,
            )

            rotated = manager.rotate_pseudonym(
                registration_id=(
                    record.registration_id
                ),
                new_pseudonym_id=(
                    "PID-FTQ-SELF-TEST-NEW"
                ),
                timestamp=1_700_000_030,
            )

            rotation_pass = all(
                (
                    rotated.pseudonym_id
                    == "PID-FTQ-SELF-TEST-NEW",

                    rotated.pseudonym_epoch == 1,

                    manager.is_historical_pseudonym(
                        "PID-FTQ-SELF-TEST-001"
                    ),
                )
            )

            manager.save(
                timestamp=1_700_000_040
            )

            reloaded_manager = (
                RegistrationManager(
                    storage_path=storage_path,
                    auto_load=True,
                )
            )

            reloaded_record = (
                reloaded_manager
                .get_by_pseudonym(
                    "PID-FTQ-SELF-TEST-NEW",
                    require_active=True,
                )
            )

            persistence_pass = all(
                (
                    reloaded_record.registration_id
                    == record.registration_id,

                    reloaded_record.pseudonym_epoch
                    == 1,

                    reloaded_record
                    .verify_identity_key(
                        identity_key
                    ),

                    reloaded_manager
                    .is_historical_pseudonym(
                        "PID-FTQ-SELF-TEST-001"
                    ),
                )
            )

            public_record = (
                reloaded_record.public_dict()
            )

            no_secret_in_public_output = all(
                field_name
                not in public_record
                for field_name in (
                    "identity_key",
                    "permanent_identity_hash",
                    "previous_pseudonyms",
                )
            )

            success = all(
                (
                    pseudonym_lookup_pass,
                    permanent_lookup_pass,
                    identity_key_pass,
                    wrong_identity_key_rejected,
                    duplicate_identity_rejected,
                    inactive_rejected,
                    rotation_pass,
                    persistence_pass,
                    no_secret_in_public_output,
                    reloaded_manager
                    .subscriber_count()
                    == 1,
                )
            )

            return {
                "success": success,
                "registration_id": (
                    record.registration_id
                ),
                "current_pseudonym": (
                    reloaded_record.pseudonym_id
                ),
                "pseudonym_epoch": (
                    reloaded_record
                    .pseudonym_epoch
                ),
                "pseudonym_lookup_pass": (
                    pseudonym_lookup_pass
                ),
                "permanent_lookup_pass": (
                    permanent_lookup_pass
                ),
                "identity_key_pass": (
                    identity_key_pass
                ),
                "wrong_identity_key_rejected": (
                    wrong_identity_key_rejected
                ),
                "duplicate_identity_rejected": (
                    duplicate_identity_rejected
                ),
                "inactive_rejected": (
                    inactive_rejected
                ),
                "rotation_pass": (
                    rotation_pass
                ),
                "persistence_pass": (
                    persistence_pass
                ),
                "no_secret_in_public_output": (
                    no_secret_in_public_output
                ),
                "subscriber_count": (
                    reloaded_manager
                    .subscriber_count()
                ),
            }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": type(
                exc
            ).__name__,
        }


__all__ = [
    "REGISTRATION_DATABASE_VERSION",
    "DEFAULT_REGISTRATION_DATABASE_PATH",
    "MINIMUM_IDENTITY_KEY_BYTES",
    "PERMANENT_IDENTITY_HASH_ALGORITHM",
    "RegistrationManagerError",
    "SubscriberRegistrationRecord",
    "normalize_permanent_identity",
    "hash_permanent_identity",
    "create_registration_id",
    "RegistrationManager",
    "get_default_registration_manager",
    "reset_default_registration_manager",
    "run_registration_manager_self_test",
]