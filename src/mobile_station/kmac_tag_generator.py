"""
KMAC Authentication Tag Generator
FT-QuPAP Mobile Station

This module implements the transcript-bound KMAC authentication tag
used by FT-QuPAP.

Notebook-aligned definition:

    tau = KMAC256(
        key=K_auth,
        data=Encode(
            pseudonym_id,
            timestamp,
            nonce,
            H(Transcript),
        ),
        output_length=128 bits,
        customization="FT-QuPAP/KMAC256/v1",
    )

The authentication tag proves possession of K_auth without placing
a raw IMSI or permanent subscriber identity inside the quantum
payload.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from Crypto.Hash import KMAC256
except ImportError as error:
    raise ImportError(
        "The 'pycryptodome' package is required. "
        "Install it using: "
        "python -m pip install pycryptodome"
    ) from error


KMAC_CUSTOMIZATION = b"FT-QuPAP/KMAC256/v1"

KMAC_KEY_LENGTH_BYTES = 32
KMAC_TAG_LENGTH_BYTES = 16
KMAC_TAG_LENGTH_BITS = 128

TRANSCRIPT_HASH_LENGTH_BYTES = 32

REQUIRED_REQUEST_FIELDS = frozenset(
    {
        "pseudonym_id",
        "timestamp",
        "nonce",
    }
)


class KMACTagGenerationError(Exception):
    """Raised when FT-QuPAP KMAC processing fails."""


@dataclass(frozen=True)
class KMACTagResult:
    """
    Result of FT-QuPAP authentication-tag generation.

    Attributes:
        tag:
            The 16-byte KMAC256 authentication tag.

        tag_input:
            Canonical non-secret data authenticated by KMAC.

        customization:
            KMAC domain-separation string.
    """

    tag: bytes
    tag_input: dict[str, Any]
    customization: bytes = KMAC_CUSTOMIZATION

    def __post_init__(self) -> None:
        validate_authentication_tag(self.tag)

        if not isinstance(self.tag_input, dict):
            raise TypeError(
                "tag_input must be a dictionary."
            )

        if not isinstance(self.customization, bytes):
            raise TypeError(
                "customization must be bytes."
            )

        if not self.customization:
            raise ValueError(
                "customization cannot be empty."
            )

    @property
    def tag_length_bits(self) -> int:
        """Return the authentication-tag length in bits."""

        return len(self.tag) * 8

    @property
    def tag_fingerprint(self) -> str:
        """
        Return a short SHA3-256 fingerprint.

        This avoids displaying the complete authentication tag in
        normal diagnostic output.
        """

        return hashlib.sha3_256(
            self.tag
        ).hexdigest()[:16]

    def safe_summary(self) -> dict[str, Any]:
        """Return non-secret KMAC diagnostic information."""

        return {
            "algorithm": "KMAC256",
            "tag_length_bytes": len(self.tag),
            "tag_length_bits": self.tag_length_bits,
            "customization":
                self.customization.decode(
                    "utf-8"
                ),
            "tag_fingerprint":
                self.tag_fingerprint,
            "pseudonym_id":
                self.tag_input.get(
                    "pseudonym_id"
                ),
            "timestamp":
                self.tag_input.get(
                    "timestamp"
                ),
        }


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """
    Serialize a mapping into deterministic UTF-8 JSON.

    Both the Mobile Station and Authentication Server must use exactly
    the same serialization configuration:

        sort_keys=True
        separators=(",", ":")
    """

    if not isinstance(value, Mapping):
        raise TypeError(
            "value must be a mapping."
        )

    try:
        serialized = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise KMACTagGenerationError(
            "Unable to serialize KMAC input."
        ) from error

    return serialized.encode("utf-8")


def encode_base64(data: bytes) -> str:
    """Encode bytes as Base64 ASCII text."""

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


def validate_k_auth(
    k_auth: bytes,
) -> None:
    """
    Validate the FT-QuPAP KMAC authentication key.

    The notebook derives K_auth as the first 32 bytes of the
    transcript-bound HKDF output.
    """

    if not isinstance(k_auth, bytes):
        raise TypeError(
            "k_auth must be bytes."
        )

    if len(k_auth) != KMAC_KEY_LENGTH_BYTES:
        raise ValueError(
            "FT-QuPAP K_auth must contain exactly "
            f"{KMAC_KEY_LENGTH_BYTES} bytes. "
            f"Received {len(k_auth)} bytes."
        )


def validate_transcript_hash(
    transcript_hash: bytes,
) -> None:
    """
    Validate SHA3-256 H(Transcript).
    """

    if not isinstance(
        transcript_hash,
        bytes,
    ):
        raise TypeError(
            "transcript_hash must be bytes."
        )

    if len(transcript_hash) != (
        TRANSCRIPT_HASH_LENGTH_BYTES
    ):
        raise ValueError(
            "transcript_hash must contain exactly "
            f"{TRANSCRIPT_HASH_LENGTH_BYTES} bytes."
        )


def validate_authentication_tag(
    tag: bytes,
    expected_length_bytes: int = (
        KMAC_TAG_LENGTH_BYTES
    ),
) -> None:
    """Validate a generated or recovered KMAC tag."""

    if not isinstance(tag, bytes):
        raise TypeError(
            "tag must be bytes."
        )

    if not isinstance(
        expected_length_bytes,
        int,
    ):
        raise TypeError(
            "expected_length_bytes must be an integer."
        )

    if expected_length_bytes <= 0:
        raise ValueError(
            "expected_length_bytes must be positive."
        )

    if len(tag) != expected_length_bytes:
        raise ValueError(
            "Authentication tag must contain exactly "
            f"{expected_length_bytes} bytes. "
            f"Received {len(tag)} bytes."
        )


def validate_authentication_request(
    request: Mapping[str, Any],
) -> None:
    """
    Validate the request fields authenticated by KMAC.

    Required fields:

        pseudonym_id
        timestamp
        nonce
    """

    if not isinstance(request, Mapping):
        raise TypeError(
            "request must be a mapping."
        )

    missing_fields = (
        REQUIRED_REQUEST_FIELDS.difference(
            request.keys()
        )
    )

    if missing_fields:
        raise KMACTagGenerationError(
            "Authentication request is missing fields: "
            f"{sorted(missing_fields)}"
        )

    pseudonym_id = request[
        "pseudonym_id"
    ]

    if not isinstance(
        pseudonym_id,
        str,
    ):
        raise TypeError(
            "request['pseudonym_id'] must be a string."
        )

    if not pseudonym_id.strip():
        raise ValueError(
            "request['pseudonym_id'] cannot be empty."
        )

    timestamp = request[
        "timestamp"
    ]

    if isinstance(timestamp, bool):
        raise TypeError(
            "request['timestamp'] must be an integer."
        )

    if not isinstance(timestamp, int):
        raise TypeError(
            "request['timestamp'] must be an integer."
        )

    if timestamp < 0:
        raise ValueError(
            "request['timestamp'] cannot be negative."
        )

    nonce = request[
        "nonce"
    ]

    if not isinstance(nonce, str):
        raise TypeError(
            "request['nonce'] must be a string."
        )

    if not nonce:
        raise ValueError(
            "request['nonce'] cannot be empty."
        )


def build_authentication_tag_input(
    request: Mapping[str, Any],
    transcript_hash: bytes,
) -> dict[str, Any]:
    """
    Construct the exact KMAC input used by the notebook.

    Tag input:

        {
            "pseudonym_id": request["pseudonym_id"],
            "timestamp": request["timestamp"],
            "nonce": request["nonce"],
            "transcript_hash": Base64(H(Transcript))
        }
    """

    validate_authentication_request(
        request
    )

    validate_transcript_hash(
        transcript_hash
    )

    return {
        "pseudonym_id":
            request["pseudonym_id"],
        "timestamp":
            request["timestamp"],
        "nonce":
            request["nonce"],
        "transcript_hash":
            encode_base64(
                transcript_hash
            ),
    }


def compute_authentication_tag(
    k_auth: bytes,
    request: Mapping[str, Any],
    transcript_hash: bytes,
    tag_length_bytes: int = (
        KMAC_TAG_LENGTH_BYTES
    ),
) -> bytes:
    """
    Generate the transcript-bound FT-QuPAP KMAC tag.

    Args:
        k_auth:
            The 32-byte authentication key derived from ML-KEM and
            H(Transcript).

        request:
            Authentication request containing the pseudonym ID,
            timestamp, and fresh nonce.

        transcript_hash:
            The 32-byte SHA3-256 protocol-transcript hash.

        tag_length_bytes:
            KMAC output size. Standard FT-QuPAP uses 16 bytes.

    Returns:
        The generated KMAC256 tag.
    """

    validate_k_auth(k_auth)

    validate_transcript_hash(
        transcript_hash
    )

    if not isinstance(
        tag_length_bytes,
        int,
    ):
        raise TypeError(
            "tag_length_bytes must be an integer."
        )

    if tag_length_bytes != (
        KMAC_TAG_LENGTH_BYTES
    ):
        raise ValueError(
            "The standard FT-QuPAP configuration requires "
            f"a {KMAC_TAG_LENGTH_BITS}-bit tag."
        )

    tag_input = (
        build_authentication_tag_input(
            request=request,
            transcript_hash=transcript_hash,
        )
    )

    encoded_tag_input = (
        canonical_json_bytes(
            tag_input
        )
    )

    try:
        tag = KMAC256.new(
            key=k_auth,
            data=encoded_tag_input,
            mac_len=tag_length_bytes,
            custom=KMAC_CUSTOMIZATION,
        ).digest()

    except Exception as error:
        raise KMACTagGenerationError(
            "KMAC256 authentication-tag generation failed."
        ) from error

    validate_authentication_tag(
        tag,
        expected_length_bytes=
            tag_length_bytes,
    )

    return tag


def generate_authentication_tag_result(
    k_auth: bytes,
    request: Mapping[str, Any],
    transcript_hash: bytes,
) -> KMACTagResult:
    """
    Generate a KMAC tag and return its structured result.
    """

    tag_input = (
        build_authentication_tag_input(
            request=request,
            transcript_hash=transcript_hash,
        )
    )

    tag = compute_authentication_tag(
        k_auth=k_auth,
        request=request,
        transcript_hash=transcript_hash,
    )

    return KMACTagResult(
        tag=tag,
        tag_input=tag_input,
    )


def recompute_expected_tag(
    k_auth: bytes,
    request: Mapping[str, Any],
    transcript_hash: bytes,
    tag_length_bytes: int = (
        KMAC_TAG_LENGTH_BYTES
    ),
) -> bytes:
    """
    Recompute the expected KMAC tag.

    This function is intended for Authentication Server deterministic
    verification. It intentionally uses the same implementation as
    the Mobile Station to prevent serialization differences.
    """

    return compute_authentication_tag(
        k_auth=k_auth,
        request=request,
        transcript_hash=transcript_hash,
        tag_length_bytes=
            tag_length_bytes,
    )


def verify_authentication_tag(
    recovered_tag: bytes,
    k_auth: bytes,
    request: Mapping[str, Any],
    transcript_hash: bytes,
) -> bool:
    """
    Compare a recovered quantum-payload tag with the expected KMAC tag.

    Constant-time comparison is used.
    """

    validate_authentication_tag(
        recovered_tag
    )

    expected_tag = (
        recompute_expected_tag(
            k_auth=k_auth,
            request=request,
            transcript_hash=
                transcript_hash,
        )
    )

    return hmac.compare_digest(
        recovered_tag,
        expected_tag,
    )


def run_self_test() -> None:
    """
    Test deterministic KMAC generation, transcript binding,
    nonce binding, and tag verification.
    """

    print("=" * 70)
    print("FT-QuPAP KMAC Authentication Tag Self-Test")
    print("=" * 70)

    test_k_auth = hashlib.sha3_256(
        b"FT-QuPAP K_auth self-test"
    ).digest()

    test_transcript_hash = (
        hashlib.sha3_256(
            b"FT-QuPAP transcript self-test"
        ).digest()
    )

    test_request = {
        "pseudonym_id":
            "PID-6G-UE-0001",
        "timestamp":
            1785520000,
        "nonce":
            "MDEyMzQ1Njc4OUFCQ0RFRg==",
        "service_context":
            "urban",
        "request_type":
            "FT-QuPAP-Authentication",
    }

    first_result = (
        generate_authentication_tag_result(
            k_auth=test_k_auth,
            request=test_request,
            transcript_hash=
                test_transcript_hash,
        )
    )

    second_tag = (
        compute_authentication_tag(
            k_auth=test_k_auth,
            request=test_request,
            transcript_hash=
                test_transcript_hash,
        )
    )

    deterministic_match = (
        hmac.compare_digest(
            first_result.tag,
            second_tag,
        )
    )

    verification_passed = (
        verify_authentication_tag(
            recovered_tag=
                first_result.tag,
            k_auth=test_k_auth,
            request=test_request,
            transcript_hash=
                test_transcript_hash,
        )
    )

    changed_nonce_request = dict(
        test_request
    )

    changed_nonce_request["nonce"] = (
        "RkVEQ0JBOTg3NjU0MzIxMA=="
    )

    changed_nonce_tag = (
        compute_authentication_tag(
            k_auth=test_k_auth,
            request=
                changed_nonce_request,
            transcript_hash=
                test_transcript_hash,
        )
    )

    nonce_binding_valid = not (
        hmac.compare_digest(
            first_result.tag,
            changed_nonce_tag,
        )
    )

    changed_transcript_hash = (
        hashlib.sha3_256(
            b"Different FT-QuPAP transcript"
        ).digest()
    )

    changed_transcript_tag = (
        compute_authentication_tag(
            k_auth=test_k_auth,
            request=test_request,
            transcript_hash=
                changed_transcript_hash,
        )
    )

    transcript_binding_valid = not (
        hmac.compare_digest(
            first_result.tag,
            changed_transcript_tag,
        )
    )

    tampered_tag = bytearray(
        first_result.tag
    )

    tampered_tag[0] ^= 0x01

    tampered_rejected = not (
        verify_authentication_tag(
            recovered_tag=
                bytes(tampered_tag),
            k_auth=test_k_auth,
            request=test_request,
            transcript_hash=
                test_transcript_hash,
        )
    )

    print(
        f"KMAC algorithm            : "
        f"KMAC256"
    )
    print(
        f"K_auth bytes              : "
        f"{len(test_k_auth)}"
    )
    print(
        f"Transcript-hash bytes     : "
        f"{len(test_transcript_hash)}"
    )
    print(
        f"Authentication-tag bytes : "
        f"{len(first_result.tag)}"
    )
    print(
        f"Authentication-tag bits  : "
        f"{first_result.tag_length_bits}"
    )
    print(
        f"Customization string      : "
        f"{KMAC_CUSTOMIZATION.decode('utf-8')}"
    )
    print(
        f"Tag fingerprint           : "
        f"{first_result.tag_fingerprint}"
    )
    print(
        f"Deterministic result      : "
        f"{deterministic_match}"
    )
    print(
        f"Valid tag accepted        : "
        f"{verification_passed}"
    )
    print(
        f"Nonce binding valid       : "
        f"{nonce_binding_valid}"
    )
    print(
        f"Transcript binding valid  : "
        f"{transcript_binding_valid}"
    )
    print(
        f"Tampered tag rejected     : "
        f"{tampered_rejected}"
    )

    if not deterministic_match:
        raise KMACTagGenerationError(
            "Repeated KMAC generation produced "
            "different tags."
        )

    if not verification_passed:
        raise KMACTagGenerationError(
            "Valid authentication tag was rejected."
        )

    if not nonce_binding_valid:
        raise KMACTagGenerationError(
            "Changing the nonce did not change the tag."
        )

    if not transcript_binding_valid:
        raise KMACTagGenerationError(
            "Changing the transcript did not change the tag."
        )

    if not tampered_rejected:
        raise KMACTagGenerationError(
            "Tampered authentication tag was accepted."
        )

    print(
        "\nSafe KMAC summary:"
    )

    print(
        json.dumps(
            first_result.safe_summary(),
            indent=4,
        )
    )

    print(
        "\nKMAC authentication-tag self-test "
        "completed successfully."
    )


__all__ = [
    "KMACTagResult",
    "KMACTagGenerationError",
    "build_authentication_tag_input",
    "compute_authentication_tag",
    "generate_authentication_tag_result",
    "recompute_expected_tag",
    "verify_authentication_tag",
    "validate_authentication_tag",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        KMACTagGenerationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"\n[KMAC TAG ERROR] {error}"
        )