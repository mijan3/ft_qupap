"""
Registration Client Module
FT-QuPAP Mobile Station

This module implements the offline registration and trust-provisioning
stage of FT-QuPAP.

Registration responsibilities:

1. Install the Authentication Server's ML-DSA-65 public key as the
   Mobile Station trust anchor.
2. Register an operator-managed pseudonymous subscriber identity.
3. Store the subscriber's permitted service contexts.
4. Validate that no private keys or raw subscriber identity are placed
   inside the Mobile Station registration package.
5. Save and load the public registration configuration safely.

The online FT-QuPAP protocol uses only the pseudonym ID. It does not
place a raw IMSI or other permanent subscriber identity inside the
authentication request.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from pqcrypto.sign import ml_dsa_65
except ImportError as error:
    raise ImportError(
        "The 'pqcrypto' package is required. Install it using: "
        "python -m pip install pqcrypto"
    ) from error


ML_DSA_ALGORITHM = "ML-DSA-65"

DEFAULT_SERVER_ID = "AS-6G-001"
DEFAULT_PSEUDONYM_ID = "PID-6G-UE-0001"
DEFAULT_TRUST_ANCHOR_VERSION = 1
DEFAULT_REGISTRATION_VERSION = 1

SUPPORTED_CONTEXTS = (
    "urban",
    "suburban",
    "rural",
)

SUPPORTED_SUBSCRIBER_STATUSES = (
    "active",
    "suspended",
    "revoked",
)

REQUIRED_REGISTRATION_FIELDS = frozenset(
    {
        "registration_version",
        "pseudonym_id",
        "subscriber_status",
        "registered_contexts",
        "server_id",
        "ml_dsa_algorithm",
        "ml_dsa_public_key",
        "trust_anchor_version",
    }
)

# These values must never be included in a Mobile Station
# registration package.
FORBIDDEN_REGISTRATION_FIELDS = frozenset(
    {
        "imsi",
        "raw_imsi",
        "subscriber_identity",
        "raw_subscriber_identity",
        "private_key",
        "secret_key",
        "ml_dsa_secret_key",
        "ml_kem_secret_key",
        "shared_secret",
        "k_auth",
        "k_ctrl",
    }
)

PSEUDONYM_PATTERN = re.compile(
    r"^PID-[A-Za-z0-9][A-Za-z0-9._-]{2,63}$"
)

SERVER_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$"
)


class RegistrationClientError(Exception):
    """Base exception for Mobile Station registration."""


class RegistrationPackageError(RegistrationClientError):
    """Raised when an operator registration package is invalid."""


class RegistrationStorageError(RegistrationClientError):
    """Raised when registration storage processing fails."""


class SubscriberNotActiveError(RegistrationClientError):
    """Raised when an inactive subscriber attempts authentication."""


@dataclass(frozen=True)
class MobileTrustAnchor:
    """
    Authentication Server trust anchor provisioned into the MS.

    Attributes:
        server_id:
            Expected Authentication Server identity.

        algorithm:
            ML-DSA signature algorithm.

        public_key:
            Authentication Server's long-term ML-DSA public key.

        trust_anchor_version:
            Operator-managed trust-anchor version.
    """

    server_id: str
    algorithm: str
    public_key: bytes
    trust_anchor_version: int = (
        DEFAULT_TRUST_ANCHOR_VERSION
    )

    def __post_init__(self) -> None:
        validate_server_id(self.server_id)

        if self.algorithm != ML_DSA_ALGORITHM:
            raise ValueError(
                "FT-QuPAP requires the trust-anchor algorithm "
                f"{ML_DSA_ALGORITHM}."
            )

        if not isinstance(self.public_key, bytes):
            raise TypeError(
                "public_key must be bytes."
            )

        if len(self.public_key) != (
            ml_dsa_65.PUBLIC_KEY_SIZE
        ):
            raise ValueError(
                "Invalid ML-DSA-65 public-key length. "
                f"Expected {ml_dsa_65.PUBLIC_KEY_SIZE} bytes, "
                f"received {len(self.public_key)}."
            )

        if not isinstance(
            self.trust_anchor_version,
            int,
        ):
            raise TypeError(
                "trust_anchor_version must be an integer."
            )

        if self.trust_anchor_version <= 0:
            raise ValueError(
                "trust_anchor_version must be positive."
            )

    @property
    def public_key_fingerprint(self) -> str:
        """
        Return the SHA3-256 trust-anchor fingerprint.
        """

        return hashlib.sha3_256(
            self.public_key
        ).hexdigest()

    def to_dictionary(self) -> dict[str, Any]:
        """
        Return the format expected by server_package_verifier.py.
        """

        return {
            "server_id": self.server_id,
            "algorithm": self.algorithm,
            "public_key": self.public_key,
            "trust_anchor_version":
                self.trust_anchor_version,
        }

    def to_storage_dictionary(self) -> dict[str, Any]:
        """
        Return a JSON-safe trust-anchor representation.
        """

        return {
            "server_id": self.server_id,
            "algorithm": self.algorithm,
            "public_key": encode_base64(
                self.public_key
            ),
            "trust_anchor_version":
                self.trust_anchor_version,
            "public_key_fingerprint":
                self.public_key_fingerprint,
        }

    def safe_summary(self) -> dict[str, Any]:
        """
        Return trust-anchor information without the full key.
        """

        return {
            "server_id": self.server_id,
            "algorithm": self.algorithm,
            "trust_anchor_version":
                self.trust_anchor_version,
            "public_key_bytes":
                len(self.public_key),
            "public_key_fingerprint":
                self.public_key_fingerprint[:16],
        }


@dataclass(frozen=True)
class SubscriberRecord:
    """
    Pseudonymous FT-QuPAP subscriber record.

    No raw IMSI or permanent subscriber identity is included.
    """

    pseudonym_id: str
    subscriber_status: str
    registered_contexts: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_pseudonym_id(
            self.pseudonym_id
        )

        if self.subscriber_status not in (
            SUPPORTED_SUBSCRIBER_STATUSES
        ):
            raise ValueError(
                "subscriber_status must be one of "
                f"{SUPPORTED_SUBSCRIBER_STATUSES}."
            )

        validate_registered_contexts(
            self.registered_contexts
        )

    @property
    def is_active(self) -> bool:
        """Return whether the subscriber may authenticate."""

        return self.subscriber_status == "active"

    def require_active(self) -> None:
        """
        Raise an error when the subscriber is not active.
        """

        if not self.is_active:
            raise SubscriberNotActiveError(
                "Subscriber cannot authenticate because "
                f"status is {self.subscriber_status!r}."
            )

    def supports_context(
        self,
        context: str,
    ) -> bool:
        """
        Check whether a service context is registered.
        """

        return context in self.registered_contexts

    def require_context(
        self,
        context: str,
    ) -> None:
        """
        Ensure that the subscriber may use a context.
        """

        if context not in SUPPORTED_CONTEXTS:
            raise ValueError(
                f"Unsupported context: {context!r}."
            )

        if not self.supports_context(context):
            raise RegistrationClientError(
                f"Subscriber {self.pseudonym_id!r} "
                f"is not registered for {context!r}."
            )

    def to_dictionary(self) -> dict[str, Any]:
        """
        Return the notebook-aligned subscriber record.
        """

        return {
            "pseudonym_id": self.pseudonym_id,
            "subscriber_status":
                self.subscriber_status,
            "registered_contexts":
                list(self.registered_contexts),
        }

    def safe_summary(self) -> dict[str, Any]:
        """Return non-sensitive subscriber information."""

        return self.to_dictionary()


@dataclass(frozen=True)
class RegistrationBundle:
    """
    Complete Mobile Station registration configuration.
    """

    subscriber: SubscriberRecord
    trust_anchor: MobileTrustAnchor
    registration_version: int = (
        DEFAULT_REGISTRATION_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.registration_version,
            int,
        ):
            raise TypeError(
                "registration_version must be an integer."
            )

        if self.registration_version <= 0:
            raise ValueError(
                "registration_version must be positive."
            )

    def mobile_station_arguments(
        self,
    ) -> dict[str, Any]:
        """
        Return arguments accepted by MobileStation(...).

        Example:

            config = bundle.mobile_station_arguments()

            mobile = MobileStation(
                pseudonym_id=config["pseudonym_id"],
                trust_anchor=config["trust_anchor"],
            )
        """

        self.subscriber.require_active()

        return {
            "pseudonym_id":
                self.subscriber.pseudonym_id,
            "trust_anchor":
                self.trust_anchor.to_dictionary(),
        }

    def server_subscriber_record(
        self,
    ) -> dict[str, Any]:
        """
        Return the record installed in the AS subscriber database.

        In deployment, the operator sends this record to the AS over
        a trusted registration channel. It is not transmitted during
        every online authentication.
        """

        return self.subscriber.to_dictionary()

    def to_storage_dictionary(
        self,
    ) -> dict[str, Any]:
        """Return a JSON-safe registration bundle."""

        return {
            "registration_version":
                self.registration_version,
            "subscriber":
                self.subscriber.to_dictionary(),
            "trust_anchor":
                self.trust_anchor
                .to_storage_dictionary(),
        }

    def safe_summary(self) -> dict[str, Any]:
        """Return a non-secret registration summary."""

        return {
            "registration_version":
                self.registration_version,
            "subscriber":
                self.subscriber.safe_summary(),
            "trust_anchor":
                self.trust_anchor.safe_summary(),
        }


def encode_base64(data: bytes) -> str:
    """Encode non-empty bytes as Base64 text."""

    if not isinstance(data, bytes):
        raise TypeError(
            "data must be bytes."
        )

    if not data:
        raise ValueError(
            "data cannot be empty."
        )

    return base64.b64encode(
        data
    ).decode("ascii")


def decode_base64(
    encoded_value: str,
    field_name: str,
) -> bytes:
    """Decode a validated Base64 field."""

    if not isinstance(encoded_value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    if not encoded_value:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    try:
        return base64.b64decode(
            encoded_value.encode("ascii"),
            validate=True,
        )

    except Exception as error:
        raise ValueError(
            f"{field_name} is not valid Base64."
        ) from error


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """Serialize a mapping deterministically."""

    if not isinstance(value, Mapping):
        raise TypeError(
            "value must be a mapping."
        )

    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def validate_pseudonym_id(
    pseudonym_id: str,
) -> None:
    """Validate an operator-managed pseudonym."""

    if not isinstance(pseudonym_id, str):
        raise TypeError(
            "pseudonym_id must be a string."
        )

    if not PSEUDONYM_PATTERN.fullmatch(
        pseudonym_id
    ):
        raise ValueError(
            "pseudonym_id must begin with 'PID-' and "
            "contain only letters, numbers, '.', '_' or '-'."
        )


def validate_server_id(
    server_id: str,
) -> None:
    """Validate an Authentication Server identifier."""

    if not isinstance(server_id, str):
        raise TypeError(
            "server_id must be a string."
        )

    if not SERVER_ID_PATTERN.fullmatch(
        server_id
    ):
        raise ValueError(
            "server_id contains unsupported characters."
        )


def validate_registered_contexts(
    registered_contexts: Sequence[str],
) -> None:
    """Validate the subscriber's permitted contexts."""

    if not isinstance(
        registered_contexts,
        Sequence,
    ):
        raise TypeError(
            "registered_contexts must be a sequence."
        )

    if isinstance(
        registered_contexts,
        (str, bytes),
    ):
        raise TypeError(
            "registered_contexts must be a sequence "
            "of context strings."
        )

    if len(registered_contexts) == 0:
        raise ValueError(
            "At least one registered context is required."
        )

    if len(registered_contexts) != len(
        set(registered_contexts)
    ):
        raise ValueError(
            "registered_contexts contains duplicates."
        )

    for context in registered_contexts:
        if context not in SUPPORTED_CONTEXTS:
            raise ValueError(
                f"Unsupported registered context: "
                f"{context!r}."
            )


def reject_forbidden_fields(
    registration_package: Mapping[str, Any],
) -> None:
    """
    Reject private keys and raw subscriber identities.

    The check recursively examines all mapping keys.
    """

    def inspect_mapping(
        value: Mapping[str, Any],
        path: str,
    ) -> None:
        for key, item in value.items():
            normalized_key = str(
                key
            ).strip().lower()

            if normalized_key in (
                FORBIDDEN_REGISTRATION_FIELDS
            ):
                raise RegistrationPackageError(
                    "Registration package contains "
                    f"forbidden field {path + normalized_key!r}."
                )

            if isinstance(item, Mapping):
                inspect_mapping(
                    item,
                    f"{path}{normalized_key}.",
                )

    inspect_mapping(
        registration_package,
        "",
    )


def validate_registration_package(
    registration_package: Mapping[str, Any],
) -> None:
    """
    Validate an operator-provided registration package.

    Expected format:

        {
            "registration_version": 1,
            "pseudonym_id": "PID-6G-UE-0001",
            "subscriber_status": "active",
            "registered_contexts": [
                "urban",
                "suburban",
                "rural"
            ],
            "server_id": "AS-6G-001",
            "ml_dsa_algorithm": "ML-DSA-65",
            "ml_dsa_public_key": "<Base64>",
            "trust_anchor_version": 1
        }
    """

    if not isinstance(
        registration_package,
        Mapping,
    ):
        raise TypeError(
            "registration_package must be a mapping."
        )

    reject_forbidden_fields(
        registration_package
    )

    missing_fields = (
        REQUIRED_REGISTRATION_FIELDS
        .difference(
            registration_package.keys()
        )
    )

    if missing_fields:
        raise RegistrationPackageError(
            "Registration package is missing fields: "
            f"{sorted(missing_fields)}"
        )

    unexpected_fields = set(
        registration_package.keys()
    ).difference(
        REQUIRED_REGISTRATION_FIELDS
    )

    if unexpected_fields:
        raise RegistrationPackageError(
            "Registration package contains unexpected fields: "
            f"{sorted(unexpected_fields)}"
        )

    version = registration_package[
        "registration_version"
    ]

    if not isinstance(version, int):
        raise RegistrationPackageError(
            "registration_version must be an integer."
        )

    if version <= 0:
        raise RegistrationPackageError(
            "registration_version must be positive."
        )

    validate_pseudonym_id(
        registration_package[
            "pseudonym_id"
        ]
    )

    status = registration_package[
        "subscriber_status"
    ]

    if status not in (
        SUPPORTED_SUBSCRIBER_STATUSES
    ):
        raise RegistrationPackageError(
            "Unsupported subscriber status."
        )

    contexts = registration_package[
        "registered_contexts"
    ]

    validate_registered_contexts(
        contexts
    )

    validate_server_id(
        registration_package[
            "server_id"
        ]
    )

    if registration_package[
        "ml_dsa_algorithm"
    ] != ML_DSA_ALGORITHM:
        raise RegistrationPackageError(
            "Registration package must use "
            f"{ML_DSA_ALGORITHM}."
        )

    public_key = decode_base64(
        registration_package[
            "ml_dsa_public_key"
        ],
        "ml_dsa_public_key",
    )

    if len(public_key) != (
        ml_dsa_65.PUBLIC_KEY_SIZE
    ):
        raise RegistrationPackageError(
            "Invalid ML-DSA public-key length."
        )

    trust_anchor_version = (
        registration_package[
            "trust_anchor_version"
        ]
    )

    if not isinstance(
        trust_anchor_version,
        int,
    ):
        raise RegistrationPackageError(
            "trust_anchor_version must be an integer."
        )

    if trust_anchor_version <= 0:
        raise RegistrationPackageError(
            "trust_anchor_version must be positive."
        )


def install_registration_package(
    registration_package: Mapping[str, Any],
) -> RegistrationBundle:
    """
    Validate and install one trusted operator package.

    This operation models secure operator provisioning. It is not an
    unauthenticated Internet registration endpoint.
    """

    validate_registration_package(
        registration_package
    )

    public_key = decode_base64(
        registration_package[
            "ml_dsa_public_key"
        ],
        "ml_dsa_public_key",
    )

    subscriber = SubscriberRecord(
        pseudonym_id=registration_package[
            "pseudonym_id"
        ],
        subscriber_status=registration_package[
            "subscriber_status"
        ],
        registered_contexts=tuple(
            registration_package[
                "registered_contexts"
            ]
        ),
    )

    trust_anchor = MobileTrustAnchor(
        server_id=registration_package[
            "server_id"
        ],
        algorithm=registration_package[
            "ml_dsa_algorithm"
        ],
        public_key=public_key,
        trust_anchor_version=(
            registration_package[
                "trust_anchor_version"
            ]
        ),
    )

    return RegistrationBundle(
        subscriber=subscriber,
        trust_anchor=trust_anchor,
        registration_version=(
            registration_package[
                "registration_version"
            ]
        ),
    )


def save_registration_bundle(
    bundle: RegistrationBundle,
    output_path: str | Path,
) -> Path:
    """
    Save a registration bundle using an atomic file replacement.

    The file contains only:
        - pseudonym information
        - registered contexts
        - the public ML-DSA trust anchor

    No private or session keys are stored.
    """

    if not isinstance(
        bundle,
        RegistrationBundle,
    ):
        raise TypeError(
            "bundle must be a RegistrationBundle."
        )

    destination = Path(
        output_path
    ).expanduser().resolve()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    storage_dictionary = (
        bundle.to_storage_dictionary()
    )

    serialized = json.dumps(
        storage_dictionary,
        indent=4,
        sort_keys=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(
                serialized
            )

            temporary_file.write("\n")

            temporary_file.flush()

            os.fsync(
                temporary_file.fileno()
            )

            temporary_path = Path(
                temporary_file.name
            )

        temporary_path.replace(
            destination
        )

        try:
            destination.chmod(0o600)
        except OSError:
            # Some platforms, including Windows, do not apply
            # POSIX file permissions in the same way.
            pass

    except Exception as error:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink(
                missing_ok=True
            )

        raise RegistrationStorageError(
            "Unable to save registration bundle."
        ) from error

    return destination


def load_registration_bundle(
    input_path: str | Path,
) -> RegistrationBundle:
    """
    Load and validate a stored Mobile Station bundle.
    """

    source = Path(
        input_path
    ).expanduser().resolve()

    if not source.exists():
        raise RegistrationStorageError(
            f"Registration file does not exist: {source}"
        )

    if not source.is_file():
        raise RegistrationStorageError(
            f"Registration path is not a file: {source}"
        )

    try:
        stored_value = json.loads(
            source.read_text(
                encoding="utf-8"
            )
        )

    except Exception as error:
        raise RegistrationStorageError(
            "Unable to read registration bundle."
        ) from error

    if not isinstance(stored_value, Mapping):
        raise RegistrationStorageError(
            "Stored registration bundle is not a mapping."
        )

    required_fields = {
        "registration_version",
        "subscriber",
        "trust_anchor",
    }

    if not required_fields.issubset(
        stored_value.keys()
    ):
        raise RegistrationStorageError(
            "Stored registration bundle is incomplete."
        )

    subscriber_value = stored_value[
        "subscriber"
    ]

    trust_anchor_value = stored_value[
        "trust_anchor"
    ]

    if not isinstance(
        subscriber_value,
        Mapping,
    ):
        raise RegistrationStorageError(
            "Stored subscriber record is invalid."
        )

    if not isinstance(
        trust_anchor_value,
        Mapping,
    ):
        raise RegistrationStorageError(
            "Stored trust anchor is invalid."
        )

    try:
        public_key = decode_base64(
            trust_anchor_value[
                "public_key"
            ],
            "public_key",
        )

        stored_fingerprint = (
            trust_anchor_value.get(
                "public_key_fingerprint"
            )
        )

        calculated_fingerprint = (
            hashlib.sha3_256(
                public_key
            ).hexdigest()
        )

        if (
            stored_fingerprint is not None
            and stored_fingerprint
            != calculated_fingerprint
        ):
            raise RegistrationStorageError(
                "Stored trust-anchor fingerprint mismatch."
            )

        bundle = RegistrationBundle(
            subscriber=SubscriberRecord(
                pseudonym_id=subscriber_value[
                    "pseudonym_id"
                ],
                subscriber_status=subscriber_value[
                    "subscriber_status"
                ],
                registered_contexts=tuple(
                    subscriber_value[
                        "registered_contexts"
                    ]
                ),
            ),
            trust_anchor=MobileTrustAnchor(
                server_id=trust_anchor_value[
                    "server_id"
                ],
                algorithm=trust_anchor_value[
                    "algorithm"
                ],
                public_key=public_key,
                trust_anchor_version=(
                    trust_anchor_value[
                        "trust_anchor_version"
                    ]
                ),
            ),
            registration_version=stored_value[
                "registration_version"
            ],
        )

    except RegistrationClientError:
        raise

    except Exception as error:
        raise RegistrationStorageError(
            "Stored registration bundle failed validation."
        ) from error

    return bundle


def build_registration_package(
    pseudonym_id: str,
    server_id: str,
    server_ml_dsa_public_key: bytes,
    registered_contexts: Sequence[str] = (
        SUPPORTED_CONTEXTS
    ),
    subscriber_status: str = "active",
    registration_version: int = (
        DEFAULT_REGISTRATION_VERSION
    ),
    trust_anchor_version: int = (
        DEFAULT_TRUST_ANCHOR_VERSION
    ),
) -> dict[str, Any]:
    """
    Build a trusted operator registration package.

    This helper represents the operator/provisioning system. It does
    not generate or store the Authentication Server private key.
    """

    package = {
        "registration_version":
            registration_version,
        "pseudonym_id":
            pseudonym_id,
        "subscriber_status":
            subscriber_status,
        "registered_contexts":
            list(registered_contexts),
        "server_id":
            server_id,
        "ml_dsa_algorithm":
            ML_DSA_ALGORITHM,
        "ml_dsa_public_key":
            encode_base64(
                server_ml_dsa_public_key
            ),
        "trust_anchor_version":
            trust_anchor_version,
    }

    validate_registration_package(
        package
    )

    return package


class RegistrationClient:
    """
    Stateful Mobile Station registration client.
    """

    def __init__(self) -> None:
        self._bundle: RegistrationBundle | None = None

    @property
    def is_registered(self) -> bool:
        """Return whether registration is installed."""

        return self._bundle is not None

    @property
    def bundle(self) -> RegistrationBundle:
        """Return the currently installed registration bundle."""

        if self._bundle is None:
            raise RegistrationClientError(
                "Mobile Station is not registered."
            )

        return self._bundle

    def install(
        self,
        registration_package: Mapping[str, Any],
    ) -> RegistrationBundle:
        """Install an operator registration package."""

        installed_bundle = (
            install_registration_package(
                registration_package
            )
        )

        self._bundle = installed_bundle

        return installed_bundle

    def save(
        self,
        output_path: str | Path,
    ) -> Path:
        """Save the installed registration bundle."""

        return save_registration_bundle(
            self.bundle,
            output_path,
        )

    def load(
        self,
        input_path: str | Path,
    ) -> RegistrationBundle:
        """Load and install a stored registration bundle."""

        loaded_bundle = (
            load_registration_bundle(
                input_path
            )
        )

        self._bundle = loaded_bundle

        return loaded_bundle

    def mobile_station_arguments(
        self,
    ) -> dict[str, Any]:
        """
        Return arguments for the MobileStation constructor.
        """

        return self.bundle.mobile_station_arguments()

    def validate_authentication_context(
        self,
        context: str,
    ) -> None:
        """
        Validate subscriber status and selected service context.
        """

        self.bundle.subscriber.require_active()

        self.bundle.subscriber.require_context(
            context
        )

    def clear(self) -> None:
        """
        Remove the installed registration reference.
        """

        self._bundle = None


def run_self_test() -> None:
    """
    Test trust-anchor provisioning, pseudonymous registration,
    storage, loading, and forbidden-field rejection.
    """

    print("=" * 70)
    print("FT-QuPAP Registration Client Self-Test")
    print("=" * 70)

    server_public_key, _server_secret_key = (
        ml_dsa_65.generate_keypair()
    )

    registration_package = (
        build_registration_package(
            pseudonym_id=
                DEFAULT_PSEUDONYM_ID,
            server_id=
                DEFAULT_SERVER_ID,
            server_ml_dsa_public_key=
                server_public_key,
            registered_contexts=
                SUPPORTED_CONTEXTS,
            subscriber_status=
                "active",
        )
    )

    client = RegistrationClient()

    installed_bundle = client.install(
        registration_package
    )

    client.validate_authentication_context(
        "urban"
    )

    configuration = (
        client.mobile_station_arguments()
    )

    if configuration["pseudonym_id"] != (
        DEFAULT_PSEUDONYM_ID
    ):
        raise RegistrationClientError(
            "Installed pseudonym does not match."
        )

    if configuration[
        "trust_anchor"
    ]["public_key"] != server_public_key:
        raise RegistrationClientError(
            "Installed trust anchor does not match."
        )

    with tempfile.TemporaryDirectory() as directory:
        registration_path = (
            Path(directory)
            / "mobile_registration.json"
        )

        saved_path = client.save(
            registration_path
        )

        second_client = RegistrationClient()

        loaded_bundle = second_client.load(
            saved_path
        )

        if (
            loaded_bundle
            .trust_anchor
            .public_key_fingerprint
            != installed_bundle
            .trust_anchor
            .public_key_fingerprint
        ):
            raise RegistrationClientError(
                "Stored trust anchor failed round-trip."
            )

        if (
            loaded_bundle.subscriber
            != installed_bundle.subscriber
        ):
            raise RegistrationClientError(
                "Stored subscriber failed round-trip."
            )

    forbidden_package = dict(
        registration_package
    )

    forbidden_package["imsi"] = (
        "001010123456789"
    )

    forbidden_rejected = False

    try:
        install_registration_package(
            forbidden_package
        )

    except RegistrationPackageError:
        forbidden_rejected = True

    if not forbidden_rejected:
        raise RegistrationClientError(
            "Registration package containing a raw IMSI "
            "was not rejected."
        )

    print(
        f"Pseudonym ID              : "
        f"{installed_bundle.subscriber.pseudonym_id}"
    )
    print(
        f"Subscriber status         : "
        f"{installed_bundle.subscriber.subscriber_status}"
    )
    print(
        f"Registered contexts       : "
        f"{list(installed_bundle.subscriber.registered_contexts)}"
    )
    print(
        f"Trust-anchor server       : "
        f"{installed_bundle.trust_anchor.server_id}"
    )
    print(
        f"Trust-anchor algorithm    : "
        f"{installed_bundle.trust_anchor.algorithm}"
    )
    print(
        f"ML-DSA public-key bytes   : "
        f"{len(installed_bundle.trust_anchor.public_key)}"
    )
    print(
        f"Trust-anchor fingerprint  : "
        f"{installed_bundle.trust_anchor.public_key_fingerprint[:16]}"
    )
    print(
        f"Raw IMSI package rejected : "
        f"{forbidden_rejected}"
    )

    print(
        "\nRegistration client self-test "
        "completed successfully."
    )


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        RegistrationClientError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        print(
            f"\n[REGISTRATION CLIENT ERROR] "
            f"{error}"
        )