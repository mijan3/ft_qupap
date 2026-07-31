"""
Protocol transcript hashing for FT-QuPAP v5.1.

The transcript hash binds all important classical protocol messages into
one fixed 256-bit digest.

The transcript normally contains:

1. M1 authentication request
2. M2 signed Authentication Server package
3. M3 ML-KEM ciphertext message
4. Session and quantum-frame metadata

Both the Mobile Station and Authentication Server must construct the
messages in the same order. Any modification to a field, message order,
sender, receiver, or metadata produces a different transcript hash.

The transcript hash is later used for:

- Session-key derivation
- KMAC authentication-tag generation
- Message-integrity verification
- Mobile/server consistency checking
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.common.constants import (
    AUTHENTICATION_REQUEST_LABEL,
    MESSAGE_M1,
    MESSAGE_M2,
    MESSAGE_M3,
    MESSAGE_QUANTUM_FRAME,
    PROTOCOL_DOMAIN_LABEL,
    TRANSCRIPT_HASH_ALGORITHM,
)

from src.common.exceptions import (
    ProtocolValidationError,
    TranscriptMismatchError,
)

from src.common.serialization import (
    canonical_json_bytes,
    to_json_safe,
)

from src.common.validators import (
    validate_bytes,
    validate_integer,
    validate_non_empty_string,
)

from src.cryptography.crypto_models import (
    TranscriptDigest,
)


# ---------------------------------------------------------------------
# Transcript configuration
# ---------------------------------------------------------------------

DEFAULT_TRANSCRIPT_HASH_ALGORITHM = TRANSCRIPT_HASH_ALGORITHM

TRANSCRIPT_DIGEST_BYTES = 32

TRANSCRIPT_VERSION = "FT-QuPAP-Transcript-v1"

TRANSCRIPT_ENTRY_DOMAIN = b"FT-QuPAP-Transcript-Entry"

TRANSCRIPT_FINAL_DOMAIN = b"FT-QuPAP-Transcript-Final"


SUPPORTED_TRANSCRIPT_MESSAGE_TYPES = (
    MESSAGE_M1,
    MESSAGE_M2,
    MESSAGE_M3,
    MESSAGE_QUANTUM_FRAME,
)


# ---------------------------------------------------------------------
# Algorithm handling
# ---------------------------------------------------------------------

def normalize_transcript_hash_algorithm(
    algorithm: str = DEFAULT_TRANSCRIPT_HASH_ALGORITHM,
) -> str:
    """
    Normalize the configured transcript-hash algorithm.

    Accepted forms:

        SHA3-256
        SHA3_256
        sha3-256
        SHA3256
    """

    normalized_input = validate_non_empty_string(
        algorithm,
        field_name="transcript_hash_algorithm",
        minimum_length=1,
        maximum_length=64,
    )

    compact = (
        normalized_input
        .strip()
        .upper()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    aliases = {
        "SHA3256": "SHA3-256",
    }

    selected = aliases.get(compact)

    if selected is None:
        raise ProtocolValidationError(
            f"Unsupported transcript hash algorithm: {algorithm}",
            code="UNSUPPORTED_TRANSCRIPT_HASH",
            details={
                "received_algorithm": algorithm,
                "supported_algorithms": [
                    "SHA3-256",
                ],
            },
        )

    return selected


def create_transcript_hasher(
    algorithm: str = DEFAULT_TRANSCRIPT_HASH_ALGORITHM,
):
    """
    Create the configured transcript hash object.
    """

    normalized_algorithm = (
        normalize_transcript_hash_algorithm(
            algorithm
        )
    )

    if normalized_algorithm == "SHA3-256":
        return hashlib.sha3_256()

    raise ProtocolValidationError(
        "Unable to create transcript hash object.",
        code="TRANSCRIPT_HASH_INITIALIZATION_ERROR",
        details={
            "algorithm": normalized_algorithm,
        },
    )


# ---------------------------------------------------------------------
# Length-prefixed hashing
# ---------------------------------------------------------------------

def _length_prefix(
    value: bytes,
) -> bytes:
    """
    Prefix a byte value with its eight-byte big-endian length.

    Length-prefixing prevents ambiguity between concatenated values.

    For example, these must not hash identically:

        b"ab" + b"c"
        b"a"  + b"bc"
    """

    validated = validate_bytes(
        value,
        field_name="length_prefixed_value",
        minimum_length=0,
    )

    return (
        len(validated).to_bytes(
            8,
            byteorder="big",
            signed=False,
        )
        + validated
    )


def hash_transcript_bytes(
    data: bytes,
    *,
    domain: bytes = TRANSCRIPT_FINAL_DOMAIN,
    algorithm: str = DEFAULT_TRANSCRIPT_HASH_ALGORITHM,
) -> bytes:
    """
    Hash raw transcript bytes using domain separation.
    """

    validated_data = validate_bytes(
        data,
        field_name="transcript_data",
        minimum_length=0,
        maximum_length=100_000_000,
    )

    validated_domain = validate_bytes(
        domain,
        field_name="transcript_domain",
        minimum_length=1,
        maximum_length=256,
    )

    hasher = create_transcript_hasher(
        algorithm
    )

    hasher.update(
        _length_prefix(
            PROTOCOL_DOMAIN_LABEL
        )
    )

    hasher.update(
        _length_prefix(
            validated_domain
        )
    )

    hasher.update(
        _length_prefix(
            validated_data
        )
    )

    digest = hasher.digest()

    return validate_bytes(
        digest,
        field_name="transcript_digest",
        exact_length=TRANSCRIPT_DIGEST_BYTES,
    )


# ---------------------------------------------------------------------
# Transcript entry model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TranscriptEntry:
    """
    One ordered protocol-transcript entry.

    Attributes
    ----------
    sequence_number:
        Starting from 1 and increasing without gaps.

    message_type:
        Protocol message name such as M1, M2, or M3.

    sender:
        Logical message sender.

    receiver:
        Logical message receiver.

    payload:
        Message fields covered by the transcript hash.

    metadata:
        Non-message details that must also be transcript-bound.
    """

    sequence_number: int
    message_type: str
    sender: str
    receiver: str
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_integer(
            self.sequence_number,
            field_name="sequence_number",
            minimum=1,
        )

        validate_non_empty_string(
            self.message_type,
            field_name="message_type",
            minimum_length=1,
            maximum_length=128,
        )

        validate_non_empty_string(
            self.sender,
            field_name="sender",
            minimum_length=1,
            maximum_length=128,
        )

        validate_non_empty_string(
            self.receiver,
            field_name="receiver",
            minimum_length=1,
            maximum_length=128,
        )

        if not isinstance(
            self.payload,
            Mapping,
        ):
            raise ProtocolValidationError(
                "Transcript entry payload must be a mapping.",
                details={
                    "received_type": type(
                        self.payload
                    ).__name__,
                },
            )

        if not isinstance(
            self.metadata,
            Mapping,
        ):
            raise ProtocolValidationError(
                "Transcript entry metadata must be a mapping.",
                details={
                    "received_type": type(
                        self.metadata
                    ).__name__,
                },
            )

    def canonical_dict(self) -> dict[str, Any]:
        """
        Return the exact canonical entry structure.
        """

        return {
            "sequence_number": self.sequence_number,
            "message_type": self.message_type,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }

    def canonical_bytes(self) -> bytes:
        """
        Return deterministic bytes for this entry.
        """

        return canonical_json_bytes(
            self.canonical_dict()
        )

    def digest(
        self,
        *,
        algorithm: str = DEFAULT_TRANSCRIPT_HASH_ALGORITHM,
    ) -> bytes:
        """
        Hash this individual transcript entry.
        """

        return hash_transcript_bytes(
            self.canonical_bytes(),
            domain=TRANSCRIPT_ENTRY_DOMAIN,
            algorithm=algorithm,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-safe representation.
        """

        result = self.canonical_dict()

        result["entry_digest"] = (
            self.digest().hex()
        )

        return to_json_safe(result)


# ---------------------------------------------------------------------
# Ordered transcript builder
# ---------------------------------------------------------------------

class ProtocolTranscript:
    """
    Ordered FT-QuPAP protocol transcript.

    Entries are appended in protocol order. The final digest includes
    both the complete canonical entries and their sequence numbers.
    """

    def __init__(
        self,
        *,
        session_id: str,
        version: str = TRANSCRIPT_VERSION,
        algorithm: str = DEFAULT_TRANSCRIPT_HASH_ALGORITHM,
    ) -> None:
        self.session_id = validate_non_empty_string(
            session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=128,
        )

        self.version = validate_non_empty_string(
            version,
            field_name="transcript_version",
            minimum_length=1,
            maximum_length=64,
        )

        self.algorithm = (
            normalize_transcript_hash_algorithm(
                algorithm
            )
        )

        self._entries: list[TranscriptEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[TranscriptEntry, ...]:
        """
        Return an immutable view of transcript entries.
        """

        return tuple(self._entries)

    def add_entry(
        self,
        *,
        message_type: str,
        sender: str,
        receiver: str,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> TranscriptEntry:
        """
        Append one message to the transcript.

        The sequence number is assigned automatically.
        """

        entry = TranscriptEntry(
            sequence_number=len(
                self._entries
            ) + 1,
            message_type=message_type,
            sender=sender,
            receiver=receiver,
            payload=dict(payload),
            metadata=(
                {}
                if metadata is None
                else dict(metadata)
            ),
        )

        self._entries.append(entry)

        return entry

    def add_m1_request(
        self,
        request: Mapping[str, Any],
        *,
        mobile_station_id: str = "MOBILE_STATION",
        authentication_server_id: str = "AUTHENTICATION_SERVER",
    ) -> TranscriptEntry:
        """
        Add the M1 authentication request.
        """

        return self.add_entry(
            message_type=MESSAGE_M1,
            sender=mobile_station_id,
            receiver=authentication_server_id,
            payload=request,
            metadata={
                "message_label": (
                    AUTHENTICATION_REQUEST_LABEL.decode(
                        "utf-8",
                        errors="strict",
                    )
                ),
            },
        )

    def add_m2_server_package(
        self,
        package: Mapping[str, Any],
        *,
        authentication_server_id: str = "AUTHENTICATION_SERVER",
        mobile_station_id: str = "MOBILE_STATION",
    ) -> TranscriptEntry:
        """
        Add the signed server package.
        """

        return self.add_entry(
            message_type=MESSAGE_M2,
            sender=authentication_server_id,
            receiver=mobile_station_id,
            payload=package,
        )

    def add_m3_ciphertext(
        self,
        ciphertext_message: Mapping[str, Any],
        *,
        mobile_station_id: str = "MOBILE_STATION",
        authentication_server_id: str = "AUTHENTICATION_SERVER",
    ) -> TranscriptEntry:
        """
        Add the ML-KEM ciphertext message.
        """

        return self.add_entry(
            message_type=MESSAGE_M3,
            sender=mobile_station_id,
            receiver=authentication_server_id,
            payload=ciphertext_message,
        )

    def add_quantum_frame_metadata(
        self,
        frame_metadata: Mapping[str, Any],
        *,
        mobile_station_id: str = "MOBILE_STATION",
        authentication_server_id: str = "AUTHENTICATION_SERVER",
    ) -> TranscriptEntry:
        """
        Add public metadata associated with the quantum frame.

        Secret check positions, preparation bases, and secret control
        schedules must not be exposed in public metadata.
        """

        return self.add_entry(
            message_type=MESSAGE_QUANTUM_FRAME,
            sender=mobile_station_id,
            receiver=authentication_server_id,
            payload=frame_metadata,
        )

    def canonical_dict(self) -> dict[str, Any]:
        """
        Return the complete canonical transcript.
        """

        return {
            "domain": PROTOCOL_DOMAIN_LABEL.decode(
                "utf-8",
                errors="strict",
            ),
            "version": self.version,
            "session_id": self.session_id,
            "hash_algorithm": self.algorithm,
            "entry_count": len(
                self._entries
            ),
            "entries": [
                entry.canonical_dict()
                for entry in self._entries
            ],
        }

    def canonical_bytes(self) -> bytes:
        """
        Return deterministic bytes for the full transcript.
        """

        return canonical_json_bytes(
            self.canonical_dict()
        )

    def digest(self) -> TranscriptDigest:
        """
        Compute the final 256-bit transcript digest.
        """

        digest_bytes = hash_transcript_bytes(
            self.canonical_bytes(),
            domain=TRANSCRIPT_FINAL_DOMAIN,
            algorithm=self.algorithm,
        )

        return TranscriptDigest(
            digest=digest_bytes,
            algorithm=self.algorithm,
        )

    def hex_digest(self) -> str:
        """
        Return the final digest as hexadecimal text.
        """

        return self.digest().digest.hex()

    def to_dict(
        self,
        *,
        include_entries: bool = True,
    ) -> dict[str, Any]:
        """
        Return transcript information for logs or the dashboard.
        """

        result: dict[str, Any] = {
            "version": self.version,
            "session_id": self.session_id,
            "hash_algorithm": self.algorithm,
            "entry_count": len(
                self._entries
            ),
            "digest": self.hex_digest(),
        }

        if include_entries:
            result["entries"] = [
                entry.to_dict()
                for entry in self._entries
            ]

        return result

    def copy(self) -> "ProtocolTranscript":
        """
        Create an independent copy of the transcript.
        """

        copied = ProtocolTranscript(
            session_id=self.session_id,
            version=self.version,
            algorithm=self.algorithm,
        )

        for entry in self._entries:
            copied.add_entry(
                message_type=entry.message_type,
                sender=entry.sender,
                receiver=entry.receiver,
                payload=dict(
                    entry.payload
                ),
                metadata=dict(
                    entry.metadata
                ),
            )

        return copied


# ---------------------------------------------------------------------
# Direct transcript helpers
# ---------------------------------------------------------------------

def create_protocol_transcript(
    *,
    session_id: str,
    m1_request: Mapping[str, Any],
    m2_server_package: Mapping[str, Any],
    m3_ciphertext_message: Mapping[str, Any],
    quantum_frame_metadata: Mapping[str, Any] | None = None,
    algorithm: str = DEFAULT_TRANSCRIPT_HASH_ALGORITHM,
) -> ProtocolTranscript:
    """
    Build the standard ordered FT-QuPAP transcript.
    """

    transcript = ProtocolTranscript(
        session_id=session_id,
        algorithm=algorithm,
    )

    transcript.add_m1_request(
        m1_request
    )

    transcript.add_m2_server_package(
        m2_server_package
    )

    transcript.add_m3_ciphertext(
        m3_ciphertext_message
    )

    if quantum_frame_metadata is not None:
        transcript.add_quantum_frame_metadata(
            quantum_frame_metadata
        )

    return transcript


def create_protocol_transcript_digest(
    *,
    session_id: str,
    m1_request: Mapping[str, Any],
    m2_server_package: Mapping[str, Any],
    m3_ciphertext_message: Mapping[str, Any],
    quantum_frame_metadata: Mapping[str, Any] | None = None,
    algorithm: str = DEFAULT_TRANSCRIPT_HASH_ALGORITHM,
) -> TranscriptDigest:
    """
    Build and hash the standard FT-QuPAP transcript.
    """

    transcript = create_protocol_transcript(
        session_id=session_id,
        m1_request=m1_request,
        m2_server_package=m2_server_package,
        m3_ciphertext_message=m3_ciphertext_message,
        quantum_frame_metadata=quantum_frame_metadata,
        algorithm=algorithm,
    )

    return transcript.digest()


# ---------------------------------------------------------------------
# Transcript verification
# ---------------------------------------------------------------------

def verify_transcript_digest(
    expected_digest: bytes | TranscriptDigest,
    received_digest: bytes | TranscriptDigest,
) -> bool:
    """
    Compare two transcript digests in constant time.
    """

    expected_bytes = (
        expected_digest.digest
        if isinstance(
            expected_digest,
            TranscriptDigest,
        )
        else expected_digest
    )

    received_bytes = (
        received_digest.digest
        if isinstance(
            received_digest,
            TranscriptDigest,
        )
        else received_digest
    )

    validated_expected = validate_bytes(
        expected_bytes,
        field_name="expected_transcript_digest",
        exact_length=TRANSCRIPT_DIGEST_BYTES,
    )

    validated_received = validate_bytes(
        received_bytes,
        field_name="received_transcript_digest",
        exact_length=TRANSCRIPT_DIGEST_BYTES,
    )

    return hmac.compare_digest(
        validated_expected,
        validated_received,
    )


def require_matching_transcript_digest(
    expected_digest: bytes | TranscriptDigest,
    received_digest: bytes | TranscriptDigest,
) -> None:
    """
    Require matching Mobile Station and server transcript digests.
    """

    if not verify_transcript_digest(
        expected_digest,
        received_digest,
    ):
        raise TranscriptMismatchError(
            details={
                "expected_digest": (
                    expected_digest.digest.hex()
                    if isinstance(
                        expected_digest,
                        TranscriptDigest,
                    )
                    else expected_digest.hex()
                ),
                "received_digest": (
                    received_digest.digest.hex()
                    if isinstance(
                        received_digest,
                        TranscriptDigest,
                    )
                    else received_digest.hex()
                ),
            },
        )


def verify_transcript_against_digest(
    transcript: ProtocolTranscript,
    expected_digest: bytes | TranscriptDigest,
) -> bool:
    """
    Recompute a transcript and compare it with a supplied digest.
    """

    if not isinstance(
        transcript,
        ProtocolTranscript,
    ):
        raise ProtocolValidationError(
            "transcript must be a ProtocolTranscript object.",
            details={
                "received_type": type(
                    transcript
                ).__name__,
            },
        )

    return verify_transcript_digest(
        transcript.digest(),
        expected_digest,
    )


# ---------------------------------------------------------------------
# Generic ordered component hash
# ---------------------------------------------------------------------

def hash_ordered_components(
    components: Sequence[Any],
    *,
    context: str = "generic-transcript-components",
    algorithm: str = DEFAULT_TRANSCRIPT_HASH_ALGORITHM,
) -> TranscriptDigest:
    """
    Hash an arbitrary ordered sequence of canonical components.

    Component order is security-relevant.
    """

    if isinstance(
        components,
        (str, bytes, bytearray),
    ) or not isinstance(
        components,
        Sequence,
    ):
        raise ProtocolValidationError(
            "components must be a non-string sequence."
        )

    validated_context = validate_non_empty_string(
        context,
        field_name="context",
        minimum_length=1,
        maximum_length=128,
    )

    payload = {
        "domain": PROTOCOL_DOMAIN_LABEL.decode(
            "utf-8",
            errors="strict",
        ),
        "context": validated_context,
        "component_count": len(
            components
        ),
        "components": list(
            components
        ),
    }

    digest_bytes = hash_transcript_bytes(
        canonical_json_bytes(
            payload
        ),
        domain=TRANSCRIPT_FINAL_DOMAIN,
        algorithm=algorithm,
    )

    return TranscriptDigest(
        digest=digest_bytes,
        algorithm=(
            normalize_transcript_hash_algorithm(
                algorithm
            )
        ),
    )


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_transcript_hash_self_test() -> dict[str, Any]:
    """
    Run deterministic transcript-hashing tests.

    The self-test confirms:

    - Identical transcripts produce identical hashes
    - Message modification changes the hash
    - Message reordering changes the hash
    - Session-ID changes alter the hash
    - Digest length is 32 bytes
    """

    m1 = {
        "pseudonym_id": "PID-SELF-TEST-001",
        "timestamp": 1_700_000_000,
        "nonce": "00" * 16,
        "context": "urban",
        "request_type": "FT_QUPAP_AUTHENTICATION",
    }

    m2 = {
        "server_id": "AS-6G-001",
        "session_id": "FTQ-TRANSCRIPT-SELF-TEST",
        "request_nonce": "00" * 16,
        "mlkem_algorithm": "ML-KEM-768",
        "signature": "SELF-TEST-SIGNATURE",
    }

    m3 = {
        "session_id": "FTQ-TRANSCRIPT-SELF-TEST",
        "mlkem_algorithm": "ML-KEM-768",
        "ciphertext": "SELF-TEST-CIPHERTEXT",
    }

    first = create_protocol_transcript(
        session_id="FTQ-TRANSCRIPT-SELF-TEST",
        m1_request=m1,
        m2_server_package=m2,
        m3_ciphertext_message=m3,
    )

    second = create_protocol_transcript(
        session_id="FTQ-TRANSCRIPT-SELF-TEST",
        m1_request=m1,
        m2_server_package=m2,
        m3_ciphertext_message=m3,
    )

    identical_transcript_pass = (
        verify_transcript_digest(
            first.digest(),
            second.digest(),
        )
    )

    modified_m1 = dict(m1)

    modified_m1["context"] = "rural"

    modified = create_protocol_transcript(
        session_id="FTQ-TRANSCRIPT-SELF-TEST",
        m1_request=modified_m1,
        m2_server_package=m2,
        m3_ciphertext_message=m3,
    )

    modification_detected = (
        not verify_transcript_digest(
            first.digest(),
            modified.digest(),
        )
    )

    reordered = ProtocolTranscript(
        session_id="FTQ-TRANSCRIPT-SELF-TEST"
    )

    reordered.add_m2_server_package(m2)
    reordered.add_m1_request(m1)
    reordered.add_m3_ciphertext(m3)

    reordering_detected = (
        not verify_transcript_digest(
            first.digest(),
            reordered.digest(),
        )
    )

    changed_session = create_protocol_transcript(
        session_id="FTQ-DIFFERENT-SESSION",
        m1_request=m1,
        m2_server_package=m2,
        m3_ciphertext_message=m3,
    )

    session_binding_pass = (
        not verify_transcript_digest(
            first.digest(),
            changed_session.digest(),
        )
    )

    digest_length_pass = (
        len(first.digest().digest)
        == TRANSCRIPT_DIGEST_BYTES
    )

    success = all(
        (
            identical_transcript_pass,
            modification_detected,
            reordering_detected,
            session_binding_pass,
            digest_length_pass,
        )
    )

    return {
        "success": success,
        "algorithm": first.algorithm,
        "digest_bytes": len(
            first.digest().digest
        ),
        "digest_hex": first.hex_digest(),
        "entry_count": len(first),
        "identical_transcript_pass": (
            identical_transcript_pass
        ),
        "modification_detected": (
            modification_detected
        ),
        "reordering_detected": (
            reordering_detected
        ),
        "session_binding_pass": (
            session_binding_pass
        ),
        "digest_length_pass": (
            digest_length_pass
        ),
    }


__all__ = [
    "DEFAULT_TRANSCRIPT_HASH_ALGORITHM",
    "TRANSCRIPT_DIGEST_BYTES",
    "TRANSCRIPT_VERSION",
    "TRANSCRIPT_ENTRY_DOMAIN",
    "TRANSCRIPT_FINAL_DOMAIN",
    "SUPPORTED_TRANSCRIPT_MESSAGE_TYPES",
    "TranscriptEntry",
    "ProtocolTranscript",
    "normalize_transcript_hash_algorithm",
    "create_transcript_hasher",
    "hash_transcript_bytes",
    "create_protocol_transcript",
    "create_protocol_transcript_digest",
    "verify_transcript_digest",
    "require_matching_transcript_digest",
    "verify_transcript_against_digest",
    "hash_ordered_components",
    "run_transcript_hash_self_test",
]