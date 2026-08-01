"""
FT-QuPAP Protocol Transcript

This module creates the canonical FT-QuPAP session transcript and
computes its SHA3-256 digest.

The transcript binds:

    1. Mobile-station authentication request
    2. Unsigned authentication-server package
    3. Protocol name
    4. Protocol version

Notebook-compatible transcript:

    transcript = {
        "request": request,
        "server_info": unsigned_server_info,
        "protocol": protocol_name,
        "version": protocol_version,
    }

The ML-DSA signature is intentionally excluded from server_info before
the transcript hash is calculated.

The resulting transcript hash is used for:

    - transcript-bound HKDF session-key derivation
    - K_auth and K_ctrl separation
    - KMAC authentication-tag generation
    - AES-GCM control-schedule binding
    - request and server-package integrity binding
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_PROTOCOL_NAME = "FT-QuPAP"

DEFAULT_PROTOCOL_VERSION = (
    "research-simulator-v5-1-large-ml-operational-threshold"
)

SERVER_SIGNATURE_FIELD = "signature"

SHA3_256_DIGEST_SIZE = 32


class TranscriptError(Exception):
    """Base exception for FT-QuPAP transcript failures."""


class InvalidTranscriptMessageError(TranscriptError):
    """Raised when a transcript message is not a valid mapping."""


class TranscriptSerializationError(TranscriptError):
    """Raised when transcript data cannot be canonically serialized."""


class InvalidTranscriptHashError(TranscriptError):
    """Raised when a supplied transcript hash is malformed."""


def validate_nonempty_string(
    value: Any,
    field_name: str,
) -> str:
    """
    Validate and normalize a required string.
    """

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return normalized


def normalize_message_mapping(
    message: Any,
    message_name: str,
) -> dict[str, Any]:
    """
    Convert a protocol message into a detached dictionary.

    Supported inputs:

        - dictionary or Mapping
        - object implementing as_dict()
        - object implementing to_dictionary()

    A deep copy is returned so later mutation of the original request or
    server package cannot silently change an existing transcript object.
    """

    if isinstance(message, Mapping):
        normalized = dict(message)

    elif hasattr(message, "as_dict"):
        normalized = message.as_dict()

    elif hasattr(message, "to_dictionary"):
        normalized = message.to_dictionary()

    else:
        raise InvalidTranscriptMessageError(
            f"{message_name} must be a mapping or provide "
            "as_dict()/to_dictionary()."
        )

    if not isinstance(normalized, Mapping):
        raise InvalidTranscriptMessageError(
            f"{message_name} conversion did not return a mapping."
        )

    normalized = copy.deepcopy(
        dict(normalized)
    )

    if not normalized:
        raise InvalidTranscriptMessageError(
            f"{message_name} cannot be empty."
        )

    for key in normalized:
        if not isinstance(key, str):
            raise InvalidTranscriptMessageError(
                f"Every {message_name} key must be a string."
            )

    return normalized


def remove_server_signature(
    server_package: Any,
    signature_field: str = SERVER_SIGNATURE_FIELD,
) -> dict[str, Any]:
    """
    Return a detached server package without its ML-DSA signature.

    The notebook explicitly creates:

        unsigned_server_info = {
            key: value
            for key, value in server_info.items()
            if key != "signature"
        }

    This function reproduces that operation.
    """

    signature_field = validate_nonempty_string(
        signature_field,
        "signature_field",
    )

    normalized_package = normalize_message_mapping(
        server_package,
        "server_package",
    )

    return {
        key: value
        for key, value in normalized_package.items()
        if key != signature_field
    }


def canonical_json_bytes(
    data: Mapping[str, Any],
) -> bytes:
    """
    Serialize a mapping into deterministic compact UTF-8 JSON.

    Notebook-compatible settings:

        sort_keys=True
        separators=(",", ":")
        encoding="utf-8"

    Both the mobile station and authentication server must serialize the
    same values in exactly the same way.
    """

    if not isinstance(data, Mapping):
        raise TypeError(
            "data must be a mapping."
        )

    try:
        serialized = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TranscriptSerializationError(
            "Transcript data is not valid canonical JSON."
        ) from error

    return serialized.encode(
        "utf-8"
    )


def hash_bytes(
    data: bytes,
) -> bytes:
    """
    Return the SHA3-256 digest of a byte sequence.
    """

    if not isinstance(
        data,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        raise TypeError(
            "data must be bytes-like."
        )

    return hashlib.sha3_256(
        bytes(data)
    ).digest()


def hash_hex(
    data: bytes,
) -> str:
    """
    Return the hexadecimal SHA3-256 digest of a byte sequence.
    """

    if not isinstance(
        data,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        raise TypeError(
            "data must be bytes-like."
        )

    return hashlib.sha3_256(
        bytes(data)
    ).hexdigest()


def build_transcript_record(
    request: Any,
    server_info_unsigned: Any,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
) -> dict[str, Any]:
    """
    Build the notebook-compatible FT-QuPAP transcript record.

    The function removes a signature field defensively even when the
    caller accidentally supplies a signed server package.
    """

    normalized_request = normalize_message_mapping(
        request,
        "request",
    )

    normalized_server_info = remove_server_signature(
        server_info_unsigned
    )

    if not normalized_server_info:
        raise InvalidTranscriptMessageError(
            "server_info_unsigned cannot be empty."
        )

    normalized_protocol_name = validate_nonempty_string(
        protocol_name,
        "protocol_name",
    )

    normalized_protocol_version = validate_nonempty_string(
        protocol_version,
        "protocol_version",
    )

    return {
        "request":
            normalized_request,
        "server_info":
            normalized_server_info,
        "protocol":
            normalized_protocol_name,
        "version":
            normalized_protocol_version,
    }


def build_transcript_bytes(
    request: Any,
    server_info_unsigned: Any,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
) -> bytes:
    """
    Return the canonical serialized FT-QuPAP transcript.
    """

    transcript = build_transcript_record(
        request=request,
        server_info_unsigned=server_info_unsigned,
        protocol_name=protocol_name,
        protocol_version=protocol_version,
    )

    return canonical_json_bytes(
        transcript
    )


def build_transcript_hash(
    request: Any,
    server_info_unsigned: Any,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
) -> bytes:
    """
    Calculate the SHA3-256 FT-QuPAP transcript hash.

    This is the main function used by the protocol engine.
    """

    transcript_bytes = build_transcript_bytes(
        request=request,
        server_info_unsigned=server_info_unsigned,
        protocol_name=protocol_name,
        protocol_version=protocol_version,
    )

    return hash_bytes(
        transcript_bytes
    )


def build_transcript_hash_hex(
    request: Any,
    server_info_unsigned: Any,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
) -> str:
    """
    Calculate the hexadecimal FT-QuPAP transcript hash.
    """

    transcript_bytes = build_transcript_bytes(
        request=request,
        server_info_unsigned=server_info_unsigned,
        protocol_name=protocol_name,
        protocol_version=protocol_version,
    )

    return hash_hex(
        transcript_bytes
    )


def validate_transcript_hash(
    transcript_hash: Any,
) -> bytes:
    """
    Validate a SHA3-256 transcript digest.

    SHA3-256 always produces 32 bytes.
    """

    if not isinstance(
        transcript_hash,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        raise InvalidTranscriptHashError(
            "transcript_hash must be bytes-like."
        )

    normalized_hash = bytes(
        transcript_hash
    )

    if len(normalized_hash) != SHA3_256_DIGEST_SIZE:
        raise InvalidTranscriptHashError(
            "A SHA3-256 transcript hash must contain "
            f"{SHA3_256_DIGEST_SIZE} bytes."
        )

    return normalized_hash


def verify_transcript_hash(
    expected_hash: bytes,
    request: Any,
    server_info_unsigned: Any,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
) -> bool:
    """
    Recalculate and securely compare an FT-QuPAP transcript hash.
    """

    normalized_expected_hash = validate_transcript_hash(
        expected_hash
    )

    calculated_hash = build_transcript_hash(
        request=request,
        server_info_unsigned=server_info_unsigned,
        protocol_name=protocol_name,
        protocol_version=protocol_version,
    )

    return hmac.compare_digest(
        normalized_expected_hash,
        calculated_hash,
    )


@dataclass
class ProtocolTranscript:
    """
    Structured FT-QuPAP request/server transcript.

    The constructor stores detached copies of the supplied messages.
    """

    request: dict[str, Any]
    server_info_unsigned: dict[str, Any]

    protocol_name: str = DEFAULT_PROTOCOL_NAME
    protocol_version: str = DEFAULT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        self.request = normalize_message_mapping(
            self.request,
            "request",
        )

        self.server_info_unsigned = remove_server_signature(
            self.server_info_unsigned
        )

        if not self.server_info_unsigned:
            raise InvalidTranscriptMessageError(
                "server_info_unsigned cannot be empty."
            )

        self.protocol_name = validate_nonempty_string(
            self.protocol_name,
            "protocol_name",
        )

        self.protocol_version = validate_nonempty_string(
            self.protocol_version,
            "protocol_version",
        )

    @classmethod
    def from_server_package(
        cls,
        request: Any,
        server_package: Any,
        protocol_name: str = DEFAULT_PROTOCOL_NAME,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    ) -> "ProtocolTranscript":
        """
        Create a transcript directly from a signed server package.

        The signature is removed automatically.
        """

        return cls(
            request=normalize_message_mapping(
                request,
                "request",
            ),
            server_info_unsigned=remove_server_signature(
                server_package
            ),
            protocol_name=protocol_name,
            protocol_version=protocol_version,
        )

    def to_dictionary(
        self,
    ) -> dict[str, Any]:
        """
        Return a detached notebook-compatible transcript dictionary.
        """

        return build_transcript_record(
            request=self.request,
            server_info_unsigned=self.server_info_unsigned,
            protocol_name=self.protocol_name,
            protocol_version=self.protocol_version,
        )

    def to_bytes(
        self,
    ) -> bytes:
        """
        Return canonical transcript bytes.
        """

        return canonical_json_bytes(
            self.to_dictionary()
        )

    def digest(
        self,
    ) -> bytes:
        """
        Return the binary SHA3-256 transcript hash.
        """

        return hash_bytes(
            self.to_bytes()
        )

    def hexdigest(
        self,
    ) -> str:
        """
        Return the hexadecimal SHA3-256 transcript hash.
        """

        return hash_hex(
            self.to_bytes()
        )

    def matches(
        self,
        expected_hash: bytes,
    ) -> bool:
        """
        Securely compare this transcript against an expected digest.
        """

        normalized_expected_hash = validate_transcript_hash(
            expected_hash
        )

        return hmac.compare_digest(
            normalized_expected_hash,
            self.digest(),
        )


def run_self_test() -> None:
    """
    Verify canonical ordering, signature exclusion, and tamper binding.
    """

    request_a = {
        "pseudonym_id": "PID-6G-UE-0001",
        "timestamp": 1785578400,
        "nonce": "MDEyMzQ1Njc4OWFiY2RlZg==",
        "service_context": "urban",
        "request_type": "FT-QuPAP-Authentication",
    }

    # Same values in a different insertion order.
    request_b = {
        "request_type": "FT-QuPAP-Authentication",
        "service_context": "urban",
        "nonce": "MDEyMzQ1Njc4OWFiY2RlZg==",
        "timestamp": 1785578400,
        "pseudonym_id": "PID-6G-UE-0001",
    }

    signed_server_package = {
        "server_id": "AS-6G-001",
        "timestamp": 1785578401,
        "request_nonce": request_a["nonce"],
        "ml_kem_algorithm": "ML-KEM-768",
        "ml_kem_public_key": "DEMO-PUBLIC-KEY",
        "ml_dsa_algorithm": "ML-DSA-65",
        "service_context": "urban",
        "signature": "DEMO-SIGNATURE",
    }

    unsigned_server_package = {
        key: value
        for key, value in signed_server_package.items()
        if key != "signature"
    }

    hash_from_signed_package = build_transcript_hash(
        request_a,
        signed_server_package,
    )

    hash_from_unsigned_package = build_transcript_hash(
        request_a,
        unsigned_server_package,
    )

    reordered_request_hash = build_transcript_hash(
        request_b,
        unsigned_server_package,
    )

    if hash_from_signed_package != hash_from_unsigned_package:
        raise TranscriptError(
            "Server signature exclusion failed."
        )

    if reordered_request_hash != hash_from_unsigned_package:
        raise TranscriptError(
            "Canonical JSON ordering failed."
        )

    transcript = ProtocolTranscript.from_server_package(
        request=request_a,
        server_package=signed_server_package,
    )

    if transcript.digest() != hash_from_unsigned_package:
        raise TranscriptError(
            "ProtocolTranscript digest is inconsistent."
        )

    if not transcript.matches(
        hash_from_unsigned_package
    ):
        raise TranscriptError(
            "Transcript comparison failed."
        )

    tampered_request = copy.deepcopy(
        request_a
    )

    tampered_request[
        "service_context"
    ] = "rural"

    tampered_hash = build_transcript_hash(
        tampered_request,
        unsigned_server_package,
    )

    if tampered_hash == hash_from_unsigned_package:
        raise TranscriptError(
            "Request modification did not alter the transcript hash."
        )

    tampered_server_package = copy.deepcopy(
        unsigned_server_package
    )

    tampered_server_package[
        "ml_kem_public_key"
    ] = "MODIFIED-PUBLIC-KEY"

    tampered_server_hash = build_transcript_hash(
        request_a,
        tampered_server_package,
    )

    if tampered_server_hash == hash_from_unsigned_package:
        raise TranscriptError(
            "Server-package modification did not alter "
            "the transcript hash."
        )

    if len(hash_from_unsigned_package) != SHA3_256_DIGEST_SIZE:
        raise TranscriptError(
            "SHA3-256 digest length is incorrect."
        )

    print(
        "Protocol transcript self-test completed successfully."
    )

    print(
        "Transcript SHA3-256:",
        hash_from_unsigned_package.hex(),
    )


__all__ = [
    "DEFAULT_PROTOCOL_NAME",
    "DEFAULT_PROTOCOL_VERSION",
    "SERVER_SIGNATURE_FIELD",
    "SHA3_256_DIGEST_SIZE",
    "TranscriptError",
    "InvalidTranscriptMessageError",
    "TranscriptSerializationError",
    "InvalidTranscriptHashError",
    "ProtocolTranscript",
    "validate_nonempty_string",
    "normalize_message_mapping",
    "remove_server_signature",
    "canonical_json_bytes",
    "hash_bytes",
    "hash_hex",
    "build_transcript_record",
    "build_transcript_bytes",
    "build_transcript_hash",
    "build_transcript_hash_hex",
    "validate_transcript_hash",
    "verify_transcript_hash",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        TranscriptError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[TRANSCRIPT ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error