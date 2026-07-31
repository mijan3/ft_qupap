"""
Authenticated control-schedule decryption for FT-QuPAP v5.1.

The Mobile Station interleaves:

- 128 logical KMAC payload blocks
- 32 independent logical check blocks

The Authentication Server needs the secret control schedule to identify:

- Payload-block positions
- Check-block positions
- Check-block identifiers
- Check preparation and measurement bases
- Expected physical reference patterns

The control schedule is encrypted using AES-256-GCM with the
transcript-bound control key derived after ML-KEM decapsulation.

Authenticated associated data binds the encrypted schedule to:

- FT-QuPAP protocol domain
- Session identifier
- Transcript hash
- Authentication-attempt number

Any modification to the ciphertext, nonce, session ID, transcript hash,
or attempt number causes authenticated decryption to fail.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.common.constants import (
    BASIS_X,
    BASIS_Z,
    CHECK_LOGICAL_QUBITS,
    PAYLOAD_LOGICAL_QUBITS,
    PROTOCOL_DOMAIN_LABEL,
    STEANE_PHYSICAL_QUBITS_PER_LOGICAL,
    TOTAL_LOGICAL_QUBITS,
)

from src.common.exceptions import (
    ControlScheduleError,
    ProtocolValidationError,
)

from src.common.serialization import (
    canonical_json_bytes,
    decode_base64,
    encode_base64,
    parse_json_bytes,
)

from src.common.validators import (
    validate_bytes,
    validate_integer,
    validate_non_empty_string,
)


CONTROL_SCHEDULE_ALGORITHM = "AES-256-GCM"

CONTROL_SCHEDULE_VERSION = "FT-QuPAP-Control-Schedule-v1"

AES_GCM_KEY_BYTES = 32

AES_GCM_NONCE_BYTES = 12

AES_GCM_TAG_BYTES = 16

TRANSCRIPT_HASH_BYTES = 32


@dataclass(frozen=True)
class ControlScheduleDecryptionResult:
    """
    Result of authenticated control-schedule decryption.

    Attributes
    ----------
    schedule:
        Validated decrypted control schedule.

    authenticated:
        True only when AES-GCM authentication succeeds.

    algorithm:
        Authenticated-encryption algorithm.

    nonce:
        Public AES-GCM nonce.

    associated_data_hash:
        SHA3-256 fingerprint of the associated data.

    ciphertext_bytes:
        Number of encrypted bytes, including the GCM tag.
    """

    schedule: dict[str, Any]
    authenticated: bool
    algorithm: str
    nonce: bytes
    associated_data_hash: str
    ciphertext_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(
            self.schedule,
            dict,
        ):
            raise ProtocolValidationError(
                "Decrypted control schedule must be a dictionary."
            )

        if not isinstance(
            self.authenticated,
            bool,
        ):
            raise ProtocolValidationError(
                "authenticated must be Boolean."
            )

        validate_non_empty_string(
            self.algorithm,
            field_name="algorithm",
            minimum_length=1,
            maximum_length=64,
        )

        validate_bytes(
            self.nonce,
            field_name="control_schedule_nonce",
            exact_length=AES_GCM_NONCE_BYTES,
        )

        validate_non_empty_string(
            self.associated_data_hash,
            field_name="associated_data_hash",
            minimum_length=64,
            maximum_length=64,
        )

        validate_integer(
            self.ciphertext_bytes,
            field_name="ciphertext_bytes",
            minimum=AES_GCM_TAG_BYTES,
        )

    def to_dict(
        self,
        *,
        include_schedule: bool = True,
    ) -> dict[str, Any]:
        """
        Return a JSON-compatible result dictionary.
        """

        result = asdict(self)

        result["nonce"] = encode_base64(
            self.nonce
        )

        if not include_schedule:
            result.pop(
                "schedule",
                None,
            )

        return result


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a dictionary-like value."""

    if not isinstance(
        value,
        Mapping,
    ):
        raise ProtocolValidationError(
            f"{field_name} must be a mapping.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    return value


def _require_sequence(
    value: Any,
    *,
    field_name: str,
) -> Sequence[Any]:
    """Require a non-string sequence."""

    if isinstance(
        value,
        (str, bytes, bytearray),
    ) or not isinstance(
        value,
        Sequence,
    ):
        raise ProtocolValidationError(
            f"{field_name} must be a sequence.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    return value


def _validate_bit_sequence(
    value: Any,
    *,
    field_name: str,
    exact_length: int,
) -> list[int]:
    """
    Validate a fixed-length classical bit sequence.
    """

    sequence = _require_sequence(
        value,
        field_name=field_name,
    )

    if len(sequence) != exact_length:
        raise ControlScheduleError(
            (
                f"{field_name} must contain exactly "
                f"{exact_length} bits."
            ),
            details={
                "field_name": field_name,
                "expected_length": exact_length,
                "actual_length": len(sequence),
            },
        )

    normalized: list[int] = []

    for index, bit in enumerate(
        sequence
    ):
        if (
            isinstance(bit, bool)
            or bit not in (0, 1)
        ):
            raise ControlScheduleError(
                f"{field_name} contains an invalid bit.",
                details={
                    "field_name": field_name,
                    "index": index,
                    "received_value": bit,
                },
            )

        normalized.append(
            int(bit)
        )

    return normalized


def normalize_control_schedule_algorithm(
    algorithm: str,
) -> str:
    """
    Normalize and validate the control-schedule encryption algorithm.
    """

    normalized = (
        validate_non_empty_string(
            algorithm,
            field_name="control_schedule_algorithm",
            minimum_length=1,
            maximum_length=64,
        )
        .strip()
        .upper()
        .replace("_", "-")
    )

    aliases = {
        "AES-256-GCM": CONTROL_SCHEDULE_ALGORITHM,
        "AES256-GCM": CONTROL_SCHEDULE_ALGORITHM,
        "AES256GCM": CONTROL_SCHEDULE_ALGORITHM,
    }

    compact = normalized.replace(
        "-",
        "",
    )

    selected = aliases.get(
        normalized,
        aliases.get(
            compact,
        ),
    )

    if selected is None:
        raise ControlScheduleError(
            (
                "Unsupported control-schedule "
                f"encryption algorithm: {algorithm}"
            ),
            details={
                "supported_algorithm": (
                    CONTROL_SCHEDULE_ALGORITHM
                ),
            },
        )

    return selected


def normalize_transcript_hash(
    transcript_hash: bytes,
) -> bytes:
    """
    Validate the 256-bit protocol transcript hash.
    """

    return validate_bytes(
        transcript_hash,
        field_name="transcript_hash",
        exact_length=TRANSCRIPT_HASH_BYTES,
    )


def build_control_schedule_associated_data(
    *,
    session_id: str,
    transcript_hash: bytes,
    attempt_number: int = 1,
) -> bytes:
    """
    Build deterministic AES-GCM associated data.

    Associated data is authenticated but not encrypted.
    Both protocol participants must construct identical bytes.
    """

    validated_session_id = validate_non_empty_string(
        session_id,
        field_name="session_id",
        minimum_length=3,
        maximum_length=128,
    )

    validated_transcript_hash = (
        normalize_transcript_hash(
            transcript_hash
        )
    )

    validated_attempt = validate_integer(
        attempt_number,
        field_name="attempt_number",
        minimum=1,
        maximum=100,
    )

    payload = {
        "domain": PROTOCOL_DOMAIN_LABEL.decode(
            "utf-8",
            errors="strict",
        ),
        "purpose": "control-schedule-encryption",
        "schedule_version": (
            CONTROL_SCHEDULE_VERSION
        ),
        "algorithm": (
            CONTROL_SCHEDULE_ALGORITHM
        ),
        "session_id": validated_session_id,
        "transcript_hash": (
            validated_transcript_hash.hex()
        ),
        "attempt_number": validated_attempt,
    }

    return canonical_json_bytes(
        payload
    )


def control_schedule_aad_hash(
    associated_data: bytes,
) -> str:
    """
    Return the SHA3-256 hash of AES-GCM associated data.
    """

    validated_data = validate_bytes(
        associated_data,
        field_name="associated_data",
        minimum_length=1,
    )

    return hashlib.sha3_256(
        validated_data
    ).hexdigest()


def validate_control_schedule(
    schedule: Mapping[str, Any],
    *,
    expected_session_id: str,
    expected_attempt_number: int,
) -> dict[str, Any]:
    """
    Validate the decrypted FT-QuPAP control schedule.

    Required structure:

        {
            "version": "FT-QuPAP-Control-Schedule-v1",
            "session_id": "...",
            "attempt_number": 1,
            "total_logical_blocks": 160,
            "payload_positions": [128 unique positions],
            "check_blocks": [
                {
                    "position": 128,
                    "block_id": "CHECK-00",
                    "basis": "Z",
                    "expected_reference_bits": [1,0,1,0,1,0,1]
                },
                ...
            ]
        }

    All 160 logical positions must be covered exactly once.
    """

    validated_schedule = _require_mapping(
        schedule,
        field_name="control_schedule",
    )

    required_fields = (
        "version",
        "session_id",
        "attempt_number",
        "total_logical_blocks",
        "payload_positions",
        "check_blocks",
    )

    missing_fields = [
        field_name
        for field_name in required_fields
        if field_name not in validated_schedule
    ]

    if missing_fields:
        raise ControlScheduleError(
            "Decrypted control schedule is incomplete.",
            details={
                "missing_fields": missing_fields,
            },
        )

    expected_session = validate_non_empty_string(
        expected_session_id,
        field_name="expected_session_id",
        minimum_length=3,
        maximum_length=128,
    )

    expected_attempt = validate_integer(
        expected_attempt_number,
        field_name="expected_attempt_number",
        minimum=1,
        maximum=100,
    )

    version = validate_non_empty_string(
        validated_schedule["version"],
        field_name="schedule.version",
        minimum_length=1,
        maximum_length=128,
    )

    if version != CONTROL_SCHEDULE_VERSION:
        raise ControlScheduleError(
            "Unsupported control-schedule version.",
            details={
                "received_version": version,
                "expected_version": (
                    CONTROL_SCHEDULE_VERSION
                ),
            },
        )

    session_id = validate_non_empty_string(
        validated_schedule["session_id"],
        field_name="schedule.session_id",
        minimum_length=3,
        maximum_length=128,
    )

    if session_id != expected_session:
        raise ControlScheduleError(
            (
                "Control schedule belongs to a "
                "different authentication session."
            ),
            details={
                "received_session_id": session_id,
                "expected_session_id": expected_session,
            },
        )

    attempt_number = validate_integer(
        validated_schedule["attempt_number"],
        field_name="schedule.attempt_number",
        minimum=1,
        maximum=100,
    )

    if attempt_number != expected_attempt:
        raise ControlScheduleError(
            (
                "Control-schedule attempt number "
                "does not match the current attempt."
            ),
            details={
                "received_attempt_number": (
                    attempt_number
                ),
                "expected_attempt_number": (
                    expected_attempt
                ),
            },
        )

    total_logical_blocks = validate_integer(
        validated_schedule[
            "total_logical_blocks"
        ],
        field_name=(
            "schedule.total_logical_blocks"
        ),
        minimum=1,
    )

    if (
        total_logical_blocks
        != TOTAL_LOGICAL_QUBITS
    ):
        raise ControlScheduleError(
            "Invalid total logical-block count.",
            details={
                "received_total": (
                    total_logical_blocks
                ),
                "expected_total": (
                    TOTAL_LOGICAL_QUBITS
                ),
            },
        )

    raw_payload_positions = _require_sequence(
        validated_schedule[
            "payload_positions"
        ],
        field_name=(
            "schedule.payload_positions"
        ),
    )

    if (
        len(raw_payload_positions)
        != PAYLOAD_LOGICAL_QUBITS
    ):
        raise ControlScheduleError(
            "Invalid number of payload positions.",
            details={
                "received_count": len(
                    raw_payload_positions
                ),
                "expected_count": (
                    PAYLOAD_LOGICAL_QUBITS
                ),
            },
        )

    payload_positions: list[int] = []

    for index, value in enumerate(
        raw_payload_positions
    ):
        position = validate_integer(
            value,
            field_name=(
                f"payload_positions[{index}]"
            ),
            minimum=0,
            maximum=(
                TOTAL_LOGICAL_QUBITS - 1
            ),
        )

        payload_positions.append(
            position
        )

    if (
        len(set(payload_positions))
        != len(payload_positions)
    ):
        raise ControlScheduleError(
            "Payload positions contain duplicates."
        )

    raw_check_blocks = _require_sequence(
        validated_schedule[
            "check_blocks"
        ],
        field_name="schedule.check_blocks",
    )

    if (
        len(raw_check_blocks)
        != CHECK_LOGICAL_QUBITS
    ):
        raise ControlScheduleError(
            "Invalid number of check blocks.",
            details={
                "received_count": len(
                    raw_check_blocks
                ),
                "expected_count": (
                    CHECK_LOGICAL_QUBITS
                ),
            },
        )

    check_blocks: list[
        dict[str, Any]
    ] = []

    check_positions: list[int] = []
    check_block_ids: list[str] = []

    for check_index, raw_check in enumerate(
        raw_check_blocks
    ):
        check = _require_mapping(
            raw_check,
            field_name=(
                f"check_blocks[{check_index}]"
            ),
        )

        required_check_fields = (
            "position",
            "block_id",
            "basis",
            "expected_reference_bits",
        )

        missing_check_fields = [
            field_name
            for field_name
            in required_check_fields
            if field_name not in check
        ]

        if missing_check_fields:
            raise ControlScheduleError(
                "Check-block schedule entry is incomplete.",
                details={
                    "check_index": check_index,
                    "missing_fields": (
                        missing_check_fields
                    ),
                },
            )

        position = validate_integer(
            check["position"],
            field_name=(
                f"check_blocks[{check_index}].position"
            ),
            minimum=0,
            maximum=(
                TOTAL_LOGICAL_QUBITS - 1
            ),
        )

        block_id = validate_non_empty_string(
            check["block_id"],
            field_name=(
                f"check_blocks[{check_index}].block_id"
            ),
            minimum_length=1,
            maximum_length=128,
        )

        basis = validate_non_empty_string(
            check["basis"],
            field_name=(
                f"check_blocks[{check_index}].basis"
            ),
            minimum_length=1,
            maximum_length=1,
        ).upper()

        if basis not in (
            BASIS_X,
            BASIS_Z,
        ):
            raise ControlScheduleError(
                "Check-block basis must be X or Z.",
                details={
                    "check_index": check_index,
                    "received_basis": basis,
                },
            )

        expected_reference_bits = (
            _validate_bit_sequence(
                check[
                    "expected_reference_bits"
                ],
                field_name=(
                    "check_blocks"
                    f"[{check_index}]"
                    ".expected_reference_bits"
                ),
                exact_length=(
                    STEANE_PHYSICAL_QUBITS_PER_LOGICAL
                ),
            )
        )

        check_positions.append(
            position
        )

        check_block_ids.append(
            block_id
        )

        normalized_check = {
            "position": position,
            "block_id": block_id,
            "basis": basis,
            "expected_reference_bits": (
                expected_reference_bits
            ),
        }

        for optional_field in (
            "logical_bit",
            "preparation_basis",
            "metadata",
        ):
            if optional_field in check:
                normalized_check[
                    optional_field
                ] = check[
                    optional_field
                ]

        check_blocks.append(
            normalized_check
        )

    if (
        len(set(check_positions))
        != len(check_positions)
    ):
        raise ControlScheduleError(
            "Check-block positions contain duplicates."
        )

    if (
        len(set(check_block_ids))
        != len(check_block_ids)
    ):
        raise ControlScheduleError(
            "Check-block identifiers contain duplicates."
        )

    payload_position_set = set(
        payload_positions
    )

    check_position_set = set(
        check_positions
    )

    overlap = (
        payload_position_set
        & check_position_set
    )

    if overlap:
        raise ControlScheduleError(
            (
                "Payload and check positions "
                "must be disjoint."
            ),
            details={
                "overlapping_positions": sorted(
                    overlap
                ),
            },
        )

    all_positions = (
        payload_position_set
        | check_position_set
    )

    expected_positions = set(
        range(
            TOTAL_LOGICAL_QUBITS
        )
    )

    if all_positions != expected_positions:
        raise ControlScheduleError(
            (
                "Control schedule does not cover "
                "all logical-frame positions exactly once."
            ),
            details={
                "missing_positions": sorted(
                    expected_positions
                    - all_positions
                ),
                "unexpected_positions": sorted(
                    all_positions
                    - expected_positions
                ),
            },
        )

    normalized_schedule: dict[
        str,
        Any
    ] = {
        "version": version,
        "session_id": session_id,
        "attempt_number": attempt_number,
        "total_logical_blocks": (
            total_logical_blocks
        ),
        "payload_positions": (
            payload_positions
        ),
        "check_blocks": check_blocks,
    }

    for optional_field in (
        "context",
        "created_at",
        "interleaving_seed_id",
        "metadata",
    ):
        if optional_field in validated_schedule:
            normalized_schedule[
                optional_field
            ] = validated_schedule[
                optional_field
            ]

    return normalized_schedule


def decrypt_control_schedule_result(
    *,
    control_key: bytes,
    encrypted_schedule: Mapping[str, Any],
    session_id: str,
    transcript_hash: bytes,
    attempt_number: int = 1,
) -> ControlScheduleDecryptionResult:
    """
    Authenticate, decrypt, parse, and validate a control schedule.

    Expected encrypted envelope:

        {
            "algorithm": "AES-256-GCM",
            "nonce": "<Base64>",
            "ciphertext": "<Base64>",
            "associated_data_hash": "<optional SHA3-256 hex>"
        }
    """

    validated_control_key = validate_bytes(
        control_key,
        field_name="control_key",
        exact_length=AES_GCM_KEY_BYTES,
    )

    envelope = _require_mapping(
        encrypted_schedule,
        field_name="encrypted_schedule",
    )

    required_fields = (
        "algorithm",
        "nonce",
        "ciphertext",
    )

    missing_fields = [
        field_name
        for field_name in required_fields
        if field_name not in envelope
    ]

    if missing_fields:
        raise ControlScheduleError(
            "Encrypted control-schedule envelope is incomplete.",
            details={
                "missing_fields": missing_fields,
            },
        )

    algorithm = normalize_control_schedule_algorithm(
        envelope["algorithm"]
    )

    try:
        nonce = decode_base64(
            envelope["nonce"]
        )

        ciphertext = decode_base64(
            envelope["ciphertext"]
        )

    except Exception as exc:
        raise ControlScheduleError(
            (
                "Unable to decode the encrypted "
                "control-schedule envelope."
            ),
            details={
                "reason": str(exc),
            },
        ) from exc

    validated_nonce = validate_bytes(
        nonce,
        field_name="control_schedule_nonce",
        exact_length=AES_GCM_NONCE_BYTES,
    )

    validated_ciphertext = validate_bytes(
        ciphertext,
        field_name="control_schedule_ciphertext",
        minimum_length=AES_GCM_TAG_BYTES,
        maximum_length=10_000_000,
    )

    associated_data = (
        build_control_schedule_associated_data(
            session_id=session_id,
            transcript_hash=transcript_hash,
            attempt_number=attempt_number,
        )
    )

    expected_aad_hash = (
        control_schedule_aad_hash(
            associated_data
        )
    )

    supplied_aad_hash = envelope.get(
        "associated_data_hash"
    )

    if supplied_aad_hash is not None:
        validated_supplied_hash = (
            validate_non_empty_string(
                supplied_aad_hash,
                field_name=(
                    "associated_data_hash"
                ),
                minimum_length=64,
                maximum_length=64,
            )
            .lower()
        )

        if not hashlib.compare_digest(
            expected_aad_hash,
            validated_supplied_hash,
        ):
            raise ControlScheduleError(
                (
                    "Control-schedule associated-data "
                    "hash does not match the current session."
                ),
                details={
                    "expected_hash": (
                        expected_aad_hash
                    ),
                    "received_hash": (
                        validated_supplied_hash
                    ),
                },
            )

    try:
        aes_gcm = AESGCM(
            validated_control_key
        )

        plaintext = aes_gcm.decrypt(
            validated_nonce,
            validated_ciphertext,
            associated_data,
        )

    except InvalidTag as exc:
        raise ControlScheduleError(
            (
                "Control-schedule authentication failed. "
                "The ciphertext, key, nonce, transcript, "
                "session ID, or attempt number is invalid."
            ),
            details={
                "algorithm": algorithm,
                "session_id": session_id,
                "attempt_number": (
                    attempt_number
                ),
            },
        ) from exc

    except Exception as exc:
        raise ControlScheduleError(
            "Control-schedule decryption failed.",
            details={
                "algorithm": algorithm,
                "reason": str(exc),
            },
        ) from exc

    try:
        decoded_schedule = parse_json_bytes(
            plaintext,
            restore_special_types=True,
        )

    except Exception as exc:
        raise ControlScheduleError(
            (
                "Decrypted control schedule "
                "does not contain valid JSON."
            ),
            details={
                "reason": str(exc),
            },
        ) from exc

    if not isinstance(
        decoded_schedule,
        Mapping,
    ):
        raise ControlScheduleError(
            (
                "Decrypted control schedule "
                "must contain a JSON object."
            )
        )

    validated_schedule = validate_control_schedule(
        decoded_schedule,
        expected_session_id=session_id,
        expected_attempt_number=attempt_number,
    )

    return ControlScheduleDecryptionResult(
        schedule=validated_schedule,
        authenticated=True,
        algorithm=algorithm,
        nonce=validated_nonce,
        associated_data_hash=(
            expected_aad_hash
        ),
        ciphertext_bytes=len(
            validated_ciphertext
        ),
    )


def decrypt_control_schedule(
    *,
    control_key: bytes,
    encrypted_schedule: Mapping[str, Any],
    session_id: str,
    transcript_hash: bytes,
    attempt_number: int = 1,
) -> dict[str, Any]:
    """
    Decrypt and return only the validated control schedule.
    """

    result = decrypt_control_schedule_result(
        control_key=control_key,
        encrypted_schedule=encrypted_schedule,
        session_id=session_id,
        transcript_hash=transcript_hash,
        attempt_number=attempt_number,
    )

    return result.schedule


def run_control_schedule_decryptor_self_test() -> dict[str, Any]:
    """
    Run deterministic authenticated-decryption tests.

    The self-test confirms:

    - A valid schedule decrypts successfully
    - All 160 logical positions are recovered
    - 128 payload positions are present
    - 32 check blocks are present
    - Modified ciphertext is rejected
    - Incorrect session binding is rejected
    """

    control_key = bytes(
        range(32)
    )

    transcript_hash = bytes(
        reversed(
            range(32)
        )
    )

    session_id = (
        "FTQ-CONTROL-SCHEDULE-SELF-TEST"
    )

    attempt_number = 1

    payload_positions = list(
        range(
            PAYLOAD_LOGICAL_QUBITS
        )
    )

    check_blocks: list[
        dict[str, Any]
    ] = []

    for index in range(
        CHECK_LOGICAL_QUBITS
    ):
        check_blocks.append(
            {
                "position": (
                    PAYLOAD_LOGICAL_QUBITS
                    + index
                ),
                "block_id": (
                    f"CHECK-{index:02d}"
                ),
                "basis": (
                    BASIS_Z
                    if index % 2 == 0
                    else BASIS_X
                ),
                "expected_reference_bits": [
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ],
            }
        )

    schedule = {
        "version": (
            CONTROL_SCHEDULE_VERSION
        ),
        "session_id": session_id,
        "attempt_number": attempt_number,
        "total_logical_blocks": (
            TOTAL_LOGICAL_QUBITS
        ),
        "payload_positions": (
            payload_positions
        ),
        "check_blocks": check_blocks,
        "context": "urban",
        "metadata": {
            "self_test": True,
        },
    }

    associated_data = (
        build_control_schedule_associated_data(
            session_id=session_id,
            transcript_hash=transcript_hash,
            attempt_number=attempt_number,
        )
    )

    nonce = bytes(
        range(
            AES_GCM_NONCE_BYTES
        )
    )

    aes_gcm = AESGCM(
        control_key
    )

    plaintext = canonical_json_bytes(
        schedule
    )

    ciphertext = aes_gcm.encrypt(
        nonce,
        plaintext,
        associated_data,
    )

    envelope = {
        "algorithm": (
            CONTROL_SCHEDULE_ALGORITHM
        ),
        "nonce": encode_base64(
            nonce
        ),
        "ciphertext": encode_base64(
            ciphertext
        ),
        "associated_data_hash": (
            control_schedule_aad_hash(
                associated_data
            )
        ),
    }

    result = decrypt_control_schedule_result(
        control_key=control_key,
        encrypted_schedule=envelope,
        session_id=session_id,
        transcript_hash=transcript_hash,
        attempt_number=attempt_number,
    )

    tampered_ciphertext = bytearray(
        ciphertext
    )

    tampered_ciphertext[0] ^= 0x01

    tampered_envelope = dict(
        envelope
    )

    tampered_envelope["ciphertext"] = (
        encode_base64(
            bytes(
                tampered_ciphertext
            )
        )
    )

    tampering_rejected = False

    try:
        decrypt_control_schedule(
            control_key=control_key,
            encrypted_schedule=(
                tampered_envelope
            ),
            session_id=session_id,
            transcript_hash=transcript_hash,
            attempt_number=attempt_number,
        )

    except ControlScheduleError:
        tampering_rejected = True

    wrong_session_rejected = False

    try:
        decrypt_control_schedule(
            control_key=control_key,
            encrypted_schedule=envelope,
            session_id="FTQ-WRONG-SESSION",
            transcript_hash=transcript_hash,
            attempt_number=attempt_number,
        )

    except ControlScheduleError:
        wrong_session_rejected = True

    success = all(
        (
            result.authenticated,
            len(
                result.schedule[
                    "payload_positions"
                ]
            )
            == PAYLOAD_LOGICAL_QUBITS,
            len(
                result.schedule[
                    "check_blocks"
                ]
            )
            == CHECK_LOGICAL_QUBITS,
            result.schedule[
                "total_logical_blocks"
            ]
            == TOTAL_LOGICAL_QUBITS,
            tampering_rejected,
            wrong_session_rejected,
        )
    )

    return {
        "success": success,
        "authenticated": (
            result.authenticated
        ),
        "algorithm": result.algorithm,
        "payload_positions": len(
            result.schedule[
                "payload_positions"
            ]
        ),
        "check_blocks": len(
            result.schedule[
                "check_blocks"
            ]
        ),
        "total_logical_blocks": (
            result.schedule[
                "total_logical_blocks"
            ]
        ),
        "ciphertext_bytes": (
            result.ciphertext_bytes
        ),
        "tampering_rejected": (
            tampering_rejected
        ),
        "wrong_session_rejected": (
            wrong_session_rejected
        ),
    }


__all__ = [
    "CONTROL_SCHEDULE_ALGORITHM",
    "CONTROL_SCHEDULE_VERSION",
    "AES_GCM_KEY_BYTES",
    "AES_GCM_NONCE_BYTES",
    "ControlScheduleDecryptionResult",
    "normalize_control_schedule_algorithm",
    "normalize_transcript_hash",
    "build_control_schedule_associated_data",
    "control_schedule_aad_hash",
    "validate_control_schedule",
    "decrypt_control_schedule_result",
    "decrypt_control_schedule",
    "run_control_schedule_decryptor_self_test",
]