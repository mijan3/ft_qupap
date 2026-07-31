"""
Shared cryptographic data models for FT-QuPAP v5.1.

These models store:

- ML-DSA key pairs and signatures
- ML-KEM key pairs and encapsulation results
- Transcript-bound session keys
- KMAC authentication tags
- Signed Authentication Server packages

Sensitive values such as secret keys and shared secrets are not included
in normal string representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.constants import (
    KMAC_TAG_BYTES,
    ML_DSA_ALGORITHM,
    ML_KEM_ALGORITHM,
)

from src.common.exceptions import (
    CryptographicError,
    ProtocolValidationError,
)

from src.common.serialization import (
    decode_base64,
    encode_base64,
)

from src.common.validators import (
    validate_bytes,
    validate_integer,
    validate_non_empty_string,
)


# ---------------------------------------------------------------------
# ML-DSA models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MLDSAKeyPair:
    """
    ML-DSA public and secret key pair.

    The Authentication Server uses the secret key for signing.
    The Mobile Station uses the public key for signature verification.
    """

    public_key: bytes
    secret_key: bytes
    algorithm: str = ML_DSA_ALGORITHM

    def __post_init__(self) -> None:
        validate_bytes(
            self.public_key,
            field_name="mldsa_public_key",
            minimum_length=1,
        )

        validate_bytes(
            self.secret_key,
            field_name="mldsa_secret_key",
            minimum_length=1,
        )

        validate_non_empty_string(
            self.algorithm,
            field_name="mldsa_algorithm",
            maximum_length=64,
        )

    def public_dict(self) -> dict[str, Any]:
        """
        Return only the shareable public-key information.
        """

        return {
            "algorithm": self.algorithm,
            "public_key": encode_base64(
                self.public_key
            ),
        }

    def private_dict(self) -> dict[str, Any]:
        """
        Return the complete key pair.

        This representation must only be stored securely.
        """

        return {
            "algorithm": self.algorithm,
            "public_key": encode_base64(
                self.public_key
            ),
            "secret_key": encode_base64(
                self.secret_key
            ),
        }

    @classmethod
    def from_private_dict(
        cls,
        data: dict[str, Any],
    ) -> "MLDSAKeyPair":
        """
        Restore an ML-DSA key pair from serialized data.
        """

        try:
            return cls(
                public_key=decode_base64(
                    data["public_key"]
                ),
                secret_key=decode_base64(
                    data["secret_key"]
                ),
                algorithm=data.get(
                    "algorithm",
                    ML_DSA_ALGORITHM,
                ),
            )
        except KeyError as exc:
            raise ProtocolValidationError(
                "Incomplete ML-DSA key-pair data.",
                details={
                    "missing_field": str(exc),
                },
            ) from exc

    def __repr__(self) -> str:
        return (
            "MLDSAKeyPair("
            f"algorithm={self.algorithm!r}, "
            f"public_key_bytes={len(self.public_key)}, "
            "secret_key=<hidden>"
            ")"
        )


@dataclass(frozen=True)
class MLDSASignature:
    """
    Detached ML-DSA signature and associated algorithm.
    """

    signature: bytes
    algorithm: str = ML_DSA_ALGORITHM

    def __post_init__(self) -> None:
        validate_bytes(
            self.signature,
            field_name="mldsa_signature",
            minimum_length=1,
        )

        validate_non_empty_string(
            self.algorithm,
            field_name="mldsa_algorithm",
            maximum_length=64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "signature": encode_base64(
                self.signature
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "MLDSASignature":
        try:
            return cls(
                signature=decode_base64(
                    data["signature"]
                ),
                algorithm=data.get(
                    "algorithm",
                    ML_DSA_ALGORITHM,
                ),
            )
        except KeyError as exc:
            raise ProtocolValidationError(
                "Incomplete ML-DSA signature data.",
                details={
                    "missing_field": str(exc),
                },
            ) from exc


# ---------------------------------------------------------------------
# ML-KEM models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MLKEMKeyPair:
    """
    ML-KEM public and secret key pair.

    In FT-QuPAP, the Authentication Server creates this key pair and
    signs the public-key package using ML-DSA.
    """

    public_key: bytes
    secret_key: bytes
    algorithm: str = ML_KEM_ALGORITHM

    def __post_init__(self) -> None:
        validate_bytes(
            self.public_key,
            field_name="mlkem_public_key",
            minimum_length=1,
        )

        validate_bytes(
            self.secret_key,
            field_name="mlkem_secret_key",
            minimum_length=1,
        )

        validate_non_empty_string(
            self.algorithm,
            field_name="mlkem_algorithm",
            maximum_length=64,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "public_key": encode_base64(
                self.public_key
            ),
        }

    def private_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "public_key": encode_base64(
                self.public_key
            ),
            "secret_key": encode_base64(
                self.secret_key
            ),
        }

    @classmethod
    def from_private_dict(
        cls,
        data: dict[str, Any],
    ) -> "MLKEMKeyPair":
        try:
            return cls(
                public_key=decode_base64(
                    data["public_key"]
                ),
                secret_key=decode_base64(
                    data["secret_key"]
                ),
                algorithm=data.get(
                    "algorithm",
                    ML_KEM_ALGORITHM,
                ),
            )
        except KeyError as exc:
            raise ProtocolValidationError(
                "Incomplete ML-KEM key-pair data.",
                details={
                    "missing_field": str(exc),
                },
            ) from exc

    def __repr__(self) -> str:
        return (
            "MLKEMKeyPair("
            f"algorithm={self.algorithm!r}, "
            f"public_key_bytes={len(self.public_key)}, "
            "secret_key=<hidden>"
            ")"
        )


@dataclass(frozen=True)
class MLKEMEncapsulationResult:
    """
    Result produced by ML-KEM encapsulation at the Mobile Station.

    The ciphertext is sent to the Authentication Server.

    The shared secret remains local and is used for transcript-bound
    session-key derivation.
    """

    ciphertext: bytes
    shared_secret: bytes
    algorithm: str = ML_KEM_ALGORITHM

    def __post_init__(self) -> None:
        validate_bytes(
            self.ciphertext,
            field_name="mlkem_ciphertext",
            minimum_length=1,
        )

        validate_bytes(
            self.shared_secret,
            field_name="mlkem_shared_secret",
            minimum_length=1,
        )

        validate_non_empty_string(
            self.algorithm,
            field_name="mlkem_algorithm",
            maximum_length=64,
        )

    def public_dict(self) -> dict[str, Any]:
        """
        Return only the ciphertext that may be transmitted.
        """

        return {
            "algorithm": self.algorithm,
            "ciphertext": encode_base64(
                self.ciphertext
            ),
        }

    def __repr__(self) -> str:
        return (
            "MLKEMEncapsulationResult("
            f"algorithm={self.algorithm!r}, "
            f"ciphertext_bytes={len(self.ciphertext)}, "
            "shared_secret=<hidden>"
            ")"
        )


@dataclass(frozen=True)
class MLKEMDecapsulationResult:
    """
    Shared secret recovered by ML-KEM decapsulation at the server.
    """

    shared_secret: bytes
    algorithm: str = ML_KEM_ALGORITHM
    success: bool = True

    def __post_init__(self) -> None:
        validate_bytes(
            self.shared_secret,
            field_name="decapsulated_shared_secret",
            minimum_length=1,
        )

        validate_non_empty_string(
            self.algorithm,
            field_name="mlkem_algorithm",
            maximum_length=64,
        )

    def __repr__(self) -> str:
        return (
            "MLKEMDecapsulationResult("
            f"algorithm={self.algorithm!r}, "
            f"success={self.success}, "
            "shared_secret=<hidden>"
            ")"
        )


# ---------------------------------------------------------------------
# Transcript and session-key models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TranscriptDigest:
    """
    Hash digest representing the ordered FT-QuPAP classical transcript.
    """

    digest: bytes
    algorithm: str = "SHA3-256"

    def __post_init__(self) -> None:
        validate_bytes(
            self.digest,
            field_name="transcript_digest",
            minimum_length=1,
        )

        validate_non_empty_string(
            self.algorithm,
            field_name="transcript_hash_algorithm",
            maximum_length=64,
        )

    def hex_digest(self) -> str:
        return self.digest.hex()

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "digest": encode_base64(
                self.digest
            ),
            "hex_digest": self.hex_digest(),
        }


@dataclass(frozen=True)
class SessionKeys:
    """
    Transcript-bound keys derived after ML-KEM key establishment.

    Fields:

    master_key:
        Intermediate session master key.

    authentication_key:
        Key used by KMAC to produce the 128-bit authentication tag.

    control_key:
        Key used to protect the secret quantum control schedule.

    session_id:
        Unique identifier for the current authentication session.

    transcript_hash:
        Hash binding these keys to the current protocol messages.
    """

    master_key: bytes
    authentication_key: bytes
    control_key: bytes
    session_id: str
    transcript_hash: bytes

    def __post_init__(self) -> None:
        validate_bytes(
            self.master_key,
            field_name="master_key",
            minimum_length=16,
        )

        validate_bytes(
            self.authentication_key,
            field_name="authentication_key",
            minimum_length=16,
        )

        validate_bytes(
            self.control_key,
            field_name="control_key",
            minimum_length=16,
        )

        validate_non_empty_string(
            self.session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=128,
        )

        validate_bytes(
            self.transcript_hash,
            field_name="transcript_hash",
            minimum_length=1,
        )

    def metadata_dict(self) -> dict[str, Any]:
        """
        Return non-secret session metadata for logs and dashboards.
        """

        return {
            "session_id": self.session_id,
            "transcript_hash": (
                self.transcript_hash.hex()
            ),
            "master_key_bytes": len(
                self.master_key
            ),
            "authentication_key_bytes": len(
                self.authentication_key
            ),
            "control_key_bytes": len(
                self.control_key
            ),
        }

    def __repr__(self) -> str:
        return (
            "SessionKeys("
            f"session_id={self.session_id!r}, "
            f"transcript_hash={self.transcript_hash.hex()!r}, "
            "master_key=<hidden>, "
            "authentication_key=<hidden>, "
            "control_key=<hidden>"
            ")"
        )


# ---------------------------------------------------------------------
# KMAC model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class KMACTag:
    """
    Fixed-size KMAC authentication tag.

    FT-QuPAP uses a 128-bit tag, equal to 16 bytes.
    """

    tag: bytes
    customization: str = "FT-QuPAP-v5.1"

    def __post_init__(self) -> None:
        validate_bytes(
            self.tag,
            field_name="kmac_tag",
            exact_length=KMAC_TAG_BYTES,
        )

        validate_non_empty_string(
            self.customization,
            field_name="kmac_customization",
            maximum_length=128,
        )

    def to_bits(self) -> list[int]:
        """
        Convert the 128-bit tag into classical bits.

        These classical bits are later mapped to logical qubit states.
        """

        bits: list[int] = []

        for byte_value in self.tag:
            for shift in range(7, -1, -1):
                bits.append(
                    (byte_value >> shift) & 1
                )

        return bits

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": encode_base64(
                self.tag
            ),
            "tag_hex": self.tag.hex(),
            "tag_bytes": len(self.tag),
            "tag_bits": len(self.tag) * 8,
            "customization": self.customization,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "KMACTag":
        try:
            return cls(
                tag=decode_base64(
                    data["tag"]
                ),
                customization=data.get(
                    "customization",
                    "FT-QuPAP-v5.1",
                ),
            )
        except KeyError as exc:
            raise ProtocolValidationError(
                "Incomplete KMAC-tag data.",
                details={
                    "missing_field": str(exc),
                },
            ) from exc


# ---------------------------------------------------------------------
# Signed server package model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SignedServerPackage:
    """
    Authentication Server package signed with ML-DSA.

    This package delivers an authenticated ML-KEM public key to the
    Mobile Station.
    """

    server_id: str
    session_id: str
    request_nonce: str

    issued_at: int
    expires_at: int

    mlkem_algorithm: str
    mlkem_public_key: bytes

    mldsa_algorithm: str
    signature: bytes

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_non_empty_string(
            self.server_id,
            field_name="server_id",
            minimum_length=2,
            maximum_length=128,
        )

        validate_non_empty_string(
            self.session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=128,
        )

        validate_non_empty_string(
            self.request_nonce,
            field_name="request_nonce",
            minimum_length=2,
            maximum_length=512,
        )

        validate_integer(
            self.issued_at,
            field_name="issued_at",
            minimum=0,
        )

        validate_integer(
            self.expires_at,
            field_name="expires_at",
            minimum=0,
        )

        if self.expires_at <= self.issued_at:
            raise ProtocolValidationError(
                "Server package expiration must be later than issue time.",
                details={
                    "issued_at": self.issued_at,
                    "expires_at": self.expires_at,
                },
            )

        validate_non_empty_string(
            self.mlkem_algorithm,
            field_name="mlkem_algorithm",
            maximum_length=64,
        )

        validate_bytes(
            self.mlkem_public_key,
            field_name="mlkem_public_key",
            minimum_length=1,
        )

        validate_non_empty_string(
            self.mldsa_algorithm,
            field_name="mldsa_algorithm",
            maximum_length=64,
        )

        validate_bytes(
            self.signature,
            field_name="server_signature",
            minimum_length=1,
        )

        if not isinstance(self.metadata, dict):
            raise ProtocolValidationError(
                "Server package metadata must be a dictionary."
            )

    def unsigned_payload(self) -> dict[str, Any]:
        """
        Return the exact fields covered by the ML-DSA signature.

        The signature itself is intentionally excluded.
        """

        return {
            "server_id": self.server_id,
            "session_id": self.session_id,
            "request_nonce": self.request_nonce,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "mlkem_algorithm": self.mlkem_algorithm,
            "mlkem_public_key": encode_base64(
                self.mlkem_public_key
            ),
            "mldsa_algorithm": self.mldsa_algorithm,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Return the complete signed package.
        """

        payload = self.unsigned_payload()

        payload["signature"] = encode_base64(
            self.signature
        )

        return payload

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "SignedServerPackage":
        """
        Restore a signed server package from serialized data.
        """

        required_fields = (
            "server_id",
            "session_id",
            "request_nonce",
            "issued_at",
            "expires_at",
            "mlkem_algorithm",
            "mlkem_public_key",
            "mldsa_algorithm",
            "signature",
        )

        missing_fields = [
            field_name
            for field_name in required_fields
            if field_name not in data
        ]

        if missing_fields:
            raise ProtocolValidationError(
                "Incomplete signed server package.",
                details={
                    "missing_fields": missing_fields,
                },
            )

        try:
            return cls(
                server_id=data["server_id"],
                session_id=data["session_id"],
                request_nonce=data["request_nonce"],
                issued_at=data["issued_at"],
                expires_at=data["expires_at"],
                mlkem_algorithm=data[
                    "mlkem_algorithm"
                ],
                mlkem_public_key=decode_base64(
                    data["mlkem_public_key"]
                ),
                mldsa_algorithm=data[
                    "mldsa_algorithm"
                ],
                signature=decode_base64(
                    data["signature"]
                ),
                metadata=data.get(
                    "metadata",
                    {},
                ),
            )
        except Exception as exc:
            if isinstance(
                exc,
                ProtocolValidationError,
            ):
                raise

            raise CryptographicError(
                "Unable to restore signed server package.",
                code="SERVER_PACKAGE_DESERIALIZATION_ERROR",
                details={
                    "reason": str(exc),
                },
            ) from exc


__all__ = [
    "MLDSAKeyPair",
    "MLDSASignature",
    "MLKEMKeyPair",
    "MLKEMEncapsulationResult",
    "MLKEMDecapsulationResult",
    "TranscriptDigest",
    "SessionKeys",
    "KMACTag",
    "SignedServerPackage",
]