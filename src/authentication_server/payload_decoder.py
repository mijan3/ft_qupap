"""
Recovered authentication-payload decoding for FT-QuPAP v5.1.

After Steane [[7,1,3]] syndrome processing, the Authentication Server
obtains one corrected logical bit from each payload block.

The encrypted control schedule contains the ordered frame positions of
the 128 payload blocks. This module:

1. Collects corrected logical bits from those positions.
2. Preserves the original payload-bit order.
3. Rejects missing or unsuccessful payload blocks.
4. Converts the 128 recovered bits into a 16-byte KMAC tag.
5. Produces safe diagnostic information without exposing the tag by
   default.

Check blocks are never included in payload reconstruction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.common.constants import (
    KMAC_TAG_BITS,
    PAYLOAD_LOGICAL_QUBITS,
    TOTAL_LOGICAL_QUBITS,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.serialization import (
    encode_base64,
)

from src.common.validators import (
    validate_bytes,
    validate_integer,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EXPECTED_PAYLOAD_BITS = PAYLOAD_LOGICAL_QUBITS

EXPECTED_PAYLOAD_BYTES = KMAC_TAG_BITS // 8

PAYLOAD_FINGERPRINT_ALGORITHM = "SHA3-256"


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class PayloadDecodingError(RuntimeError):
    """Raised when the logical authentication payload cannot be decoded."""

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
# Result model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PayloadDecodingResult:
    """
    Result of logical payload reconstruction.

    Attributes
    ----------
    logical_bits:
        Corrected payload bits in their authenticated schedule order.

    payload:
        Reconstructed 16-byte KMAC authentication tag.

    payload_fingerprint:
        SHA3-256 fingerprint used for safe diagnostics.

    recovered_payload_bits:
        Number of successfully recovered logical bits.

    expected_payload_bits:
        Required number of logical payload bits.

    complete:
        True only when the full payload was recovered.
    """

    logical_bits: tuple[int, ...]
    payload: bytes

    payload_fingerprint: str

    recovered_payload_bits: int
    expected_payload_bits: int

    complete: bool

    def __post_init__(self) -> None:
        normalized_bits = tuple(
            normalize_logical_bit(
                bit,
                field_name=(
                    f"logical_bits[{index}]"
                ),
            )
            for index, bit in enumerate(
                self.logical_bits
            )
        )

        object.__setattr__(
            self,
            "logical_bits",
            normalized_bits,
        )

        validated_payload = validate_bytes(
            self.payload,
            field_name="decoded_payload",
            exact_length=EXPECTED_PAYLOAD_BYTES,
        )

        object.__setattr__(
            self,
            "payload",
            validated_payload,
        )

        validate_integer(
            self.recovered_payload_bits,
            field_name="recovered_payload_bits",
            minimum=0,
        )

        validate_integer(
            self.expected_payload_bits,
            field_name="expected_payload_bits",
            minimum=1,
        )

        if not isinstance(
            self.complete,
            bool,
        ):
            raise ProtocolValidationError(
                "complete must be Boolean."
            )

        if (
            self.recovered_payload_bits
            != len(self.logical_bits)
        ):
            raise ProtocolValidationError(
                (
                    "recovered_payload_bits does not match "
                    "the logical-bit sequence length."
                )
            )

        expected_complete = (
            self.recovered_payload_bits
            == self.expected_payload_bits
        )

        if self.complete != expected_complete:
            raise ProtocolValidationError(
                (
                    "complete does not match the recovered "
                    "payload-bit count."
                )
            )

        expected_fingerprint = (
            calculate_payload_fingerprint(
                self.payload
            )
        )

        if (
            self.payload_fingerprint.lower()
            != expected_fingerprint
        ):
            raise ProtocolValidationError(
                "Decoded payload fingerprint is invalid."
            )

    @property
    def payload_hex(self) -> str:
        """Return the reconstructed payload as hexadecimal text."""

        return self.payload.hex()

    def public_dict(self) -> dict[str, Any]:
        """
        Return non-secret payload diagnostics.

        The recovered KMAC tag is intentionally omitted.
        """

        return {
            "complete": self.complete,
            "recovered_payload_bits": (
                self.recovered_payload_bits
            ),
            "expected_payload_bits": (
                self.expected_payload_bits
            ),
            "payload_bytes": len(
                self.payload
            ),
            "payload_fingerprint": (
                self.payload_fingerprint
            ),
        }

    def protected_dict(self) -> dict[str, Any]:
        """
        Return the full result for protected internal processing.

        Do not write this representation to ordinary logs.
        """

        result = self.public_dict()

        result.update(
            {
                "logical_bits": list(
                    self.logical_bits
                ),
                "payload": encode_base64(
                    self.payload
                ),
                "payload_hex": self.payload_hex,
            }
        )

        return result

    def __repr__(self) -> str:
        return (
            "PayloadDecodingResult("
            f"complete={self.complete}, "
            f"recovered_payload_bits="
            f"{self.recovered_payload_bits}, "
            f"expected_payload_bits="
            f"{self.expected_payload_bits}, "
            f"payload_fingerprint="
            f"{self.payload_fingerprint!r}, "
            "payload=<hidden>"
            ")"
        )


# ---------------------------------------------------------------------
# Bit validation and conversion
# ---------------------------------------------------------------------

def normalize_logical_bit(
    value: Any,
    *,
    field_name: str = "logical_bit",
) -> int:
    """
    Normalize one corrected logical bit.

    Accepted values:

        0
        1
        False
        True

    Booleans are converted to their corresponding integer bits.
    """

    if isinstance(
        value,
        bool,
    ):
        return int(value)

    if isinstance(
        value,
        int,
    ) and value in (0, 1):
        return value

    raise PayloadDecodingError(
        f"{field_name} must be a binary logical bit.",
        details={
            "field_name": field_name,
            "received_value": value,
            "received_type": type(
                value
            ).__name__,
        },
    )


def normalize_bit_sequence(
    bits: Sequence[Any],
    *,
    field_name: str = "bits",
    exact_length: int | None = None,
) -> tuple[int, ...]:
    """
    Validate and normalize a classical bit sequence.
    """

    if isinstance(
        bits,
        (str, bytes, bytearray),
    ) or not isinstance(
        bits,
        Sequence,
    ):
        raise PayloadDecodingError(
            f"{field_name} must be a bit sequence.",
            details={
                "received_type": type(
                    bits
                ).__name__,
            },
        )

    normalized = tuple(
        normalize_logical_bit(
            bit,
            field_name=(
                f"{field_name}[{index}]"
            ),
        )
        for index, bit in enumerate(
            bits
        )
    )

    if (
        exact_length is not None
        and len(normalized) != exact_length
    ):
        raise PayloadDecodingError(
            f"{field_name} has an invalid length.",
            details={
                "expected_length": exact_length,
                "actual_length": len(
                    normalized
                ),
            },
        )

    return normalized


def bits_to_bytes(
    bits: Sequence[Any],
) -> bytes:
    """
    Convert big-endian classical bits into bytes.

    Example:

        [1, 0, 1, 00001] -> b"\\xA1"

    The bit count must be divisible by eight.
    """

    normalized_bits = normalize_bit_sequence(
        bits,
        field_name="bits",
    )

    if len(normalized_bits) == 0:
        raise PayloadDecodingError(
            "Cannot convert an empty bit sequence."
        )

    if len(normalized_bits) % 8 != 0:
        raise PayloadDecodingError(
            (
                "The bit-sequence length must be "
                "divisible by eight."
            ),
            details={
                "bit_count": len(
                    normalized_bits
                ),
            },
        )

    output = bytearray()

    for offset in range(
        0,
        len(normalized_bits),
        8,
    ):
        byte_value = 0

        for bit in normalized_bits[
            offset:offset + 8
        ]:
            byte_value = (
                byte_value << 1
            ) | bit

        output.append(
            byte_value
        )

    return bytes(
        output
    )


def bytes_to_bits(
    data: bytes,
) -> tuple[int, ...]:
    """
    Convert bytes into a big-endian classical bit sequence.
    """

    validated_data = validate_bytes(
        data,
        field_name="data",
        minimum_length=1,
        maximum_length=1_000_000,
    )

    bits: list[int] = []

    for byte_value in validated_data:
        for shift in range(
            7,
            -1,
            -1,
        ):
            bits.append(
                (
                    byte_value
                    >> shift
                )
                & 1
            )

    return tuple(
        bits
    )


def calculate_payload_fingerprint(
    payload: bytes,
) -> str:
    """
    Calculate a SHA3-256 fingerprint of the recovered payload.
    """

    validated_payload = validate_bytes(
        payload,
        field_name="payload",
        minimum_length=1,
        maximum_length=1_000_000,
    )

    return hashlib.sha3_256(
        validated_payload
    ).hexdigest()


# ---------------------------------------------------------------------
# Corrected-block field extraction
# ---------------------------------------------------------------------

def _read_field(
    value: Any,
    field_names: tuple[str, ...],
) -> Any:
    """
    Read the first available field from an object or mapping.
    """

    if isinstance(
        value,
        Mapping,
    ):
        for field_name in field_names:
            if field_name in value:
                return value[
                    field_name
                ]

    else:
        for field_name in field_names:
            if hasattr(
                value,
                field_name,
            ):
                return getattr(
                    value,
                    field_name,
                )

    return None


def _extract_block_success(
    block: Any,
) -> bool:
    """
    Determine whether a corrected block is usable.

    Supported fields include:

        success
        correction_success
        decoded
        valid

    Missing status fields are treated as successful when a logical bit
    is present.
    """

    status = _read_field(
        block,
        (
            "success",
            "correction_success",
            "decoded",
            "valid",
        ),
    )

    if status is None:
        return True

    if not isinstance(
        status,
        bool,
    ):
        raise PayloadDecodingError(
            "Corrected-block success status must be Boolean."
        )

    return status


def _extract_logical_bit(
    block: Any,
    *,
    field_name: str,
) -> int:
    """
    Extract one corrected logical bit from a block representation.

    Supported fields:

        logical_bit
        corrected_logical_bit
        decoded_bit
        bit

    A direct integer or Boolean value is also accepted.
    """

    if isinstance(
        block,
        (bool, int),
    ):
        return normalize_logical_bit(
            block,
            field_name=field_name,
        )

    value = _read_field(
        block,
        (
            "logical_bit",
            "corrected_logical_bit",
            "decoded_bit",
            "bit",
        ),
    )

    if value is None:
        raise PayloadDecodingError(
            "Corrected payload block contains no logical bit.",
            details={
                "field_name": field_name,
                "received_type": type(
                    block
                ).__name__,
            },
        )

    return normalize_logical_bit(
        value,
        field_name=field_name,
    )


def _extract_frame_position(
    block: Any,
) -> int | None:
    """
    Extract an optional frame position from a corrected block.
    """

    position = _read_field(
        block,
        (
            "position",
            "frame_position",
            "block_position",
        ),
    )

    if position is None:
        return None

    return validate_integer(
        position,
        field_name="frame_position",
        minimum=0,
        maximum=(
            TOTAL_LOGICAL_QUBITS - 1
        ),
    )


# ---------------------------------------------------------------------
# Payload-position validation
# ---------------------------------------------------------------------

def validate_payload_positions(
    payload_positions: Sequence[Any],
    *,
    expected_payload_bits: int = EXPECTED_PAYLOAD_BITS,
) -> tuple[int, ...]:
    """
    Validate ordered payload positions from the control schedule.

    Position order represents the original KMAC payload-bit order.
    """

    if isinstance(
        payload_positions,
        (str, bytes, bytearray),
    ) or not isinstance(
        payload_positions,
        Sequence,
    ):
        raise PayloadDecodingError(
            "payload_positions must be a sequence."
        )

    expected_count = validate_integer(
        expected_payload_bits,
        field_name="expected_payload_bits",
        minimum=1,
        maximum=TOTAL_LOGICAL_QUBITS,
    )

    normalized_positions = tuple(
        validate_integer(
            value,
            field_name=(
                f"payload_positions[{index}]"
            ),
            minimum=0,
            maximum=(
                TOTAL_LOGICAL_QUBITS - 1
            ),
        )
        for index, value in enumerate(
            payload_positions
        )
    )

    if (
        len(normalized_positions)
        != expected_count
    ):
        raise PayloadDecodingError(
            "Invalid number of payload positions.",
            details={
                "expected_count": expected_count,
                "actual_count": len(
                    normalized_positions
                ),
            },
        )

    if (
        len(set(normalized_positions))
        != len(normalized_positions)
    ):
        raise PayloadDecodingError(
            "Payload positions contain duplicates."
        )

    return normalized_positions


# ---------------------------------------------------------------------
# Payload-block collection
# ---------------------------------------------------------------------

def _build_position_map(
    corrected_blocks: Mapping[Any, Any]
    | Sequence[Any],
) -> dict[int, Any]:
    """
    Convert corrected block data into a frame-position mapping.

    Supported forms:

    1. Mapping:
       {
           12: corrected_block,
           44: corrected_block
       }

    2. Full frame sequence:
       sequence index is treated as frame position.

    3. Sequence of objects containing an explicit position field.
    """

    if isinstance(
        corrected_blocks,
        Mapping,
    ):
        position_map: dict[int, Any] = {}

        for raw_position, block in (
            corrected_blocks.items()
        ):
            position = validate_integer(
                raw_position,
                field_name=(
                    "corrected_block_position"
                ),
                minimum=0,
                maximum=(
                    TOTAL_LOGICAL_QUBITS - 1
                ),
            )

            if position in position_map:
                raise PayloadDecodingError(
                    "Duplicate corrected-block position.",
                    details={
                        "position": position,
                    },
                )

            position_map[position] = block

        return position_map

    if isinstance(
        corrected_blocks,
        (str, bytes, bytearray),
    ) or not isinstance(
        corrected_blocks,
        Sequence,
    ):
        raise PayloadDecodingError(
            (
                "corrected_blocks must be a mapping "
                "or sequence."
            )
        )

    explicit_positions = [
        _extract_frame_position(
            block
        )
        for block in corrected_blocks
    ]

    has_explicit_positions = any(
        position is not None
        for position in explicit_positions
    )

    if has_explicit_positions:
        if any(
            position is None
            for position in explicit_positions
        ):
            raise PayloadDecodingError(
                (
                    "Corrected blocks must either all include "
                    "positions or none include positions."
                )
            )

        position_map = {}

        for position, block in zip(
            explicit_positions,
            corrected_blocks,
            strict=True,
        ):
            assert position is not None

            if position in position_map:
                raise PayloadDecodingError(
                    "Duplicate corrected-block position.",
                    details={
                        "position": position,
                    },
                )

            position_map[position] = block

        return position_map

    return {
        index: block
        for index, block in enumerate(
            corrected_blocks
        )
    }


def collect_ordered_payload_bits(
    *,
    corrected_blocks: Mapping[Any, Any]
    | Sequence[Any],
    payload_positions: Sequence[Any],
    expected_payload_bits: int = EXPECTED_PAYLOAD_BITS,
) -> tuple[int, ...]:
    """
    Collect corrected logical bits in control-schedule order.

    A missing, failed, or malformed block causes fail-closed rejection.
    """

    positions = validate_payload_positions(
        payload_positions,
        expected_payload_bits=(
            expected_payload_bits
        ),
    )

    position_map = _build_position_map(
        corrected_blocks
    )

    recovered_bits: list[int] = []

    missing_positions: list[int] = []
    failed_positions: list[int] = []

    for payload_index, position in enumerate(
        positions
    ):
        if position not in position_map:
            missing_positions.append(
                position
            )

            continue

        block = position_map[
            position
        ]

        if not _extract_block_success(
            block
        ):
            failed_positions.append(
                position
            )

            continue

        bit = _extract_logical_bit(
            block,
            field_name=(
                f"payload_bit[{payload_index}]"
            ),
        )

        recovered_bits.append(
            bit
        )

    if (
        missing_positions
        or failed_positions
    ):
        raise PayloadDecodingError(
            (
                "The complete logical authentication payload "
                "could not be recovered."
            ),
            details={
                "expected_payload_bits": (
                    expected_payload_bits
                ),
                "recovered_payload_bits": len(
                    recovered_bits
                ),
                "missing_positions": (
                    missing_positions
                ),
                "failed_positions": (
                    failed_positions
                ),
            },
        )

    return normalize_bit_sequence(
        recovered_bits,
        field_name="recovered_payload_bits",
        exact_length=expected_payload_bits,
    )


# ---------------------------------------------------------------------
# Main decoding functions
# ---------------------------------------------------------------------

def decode_payload_from_bits(
    logical_bits: Sequence[Any],
    *,
    expected_payload_bits: int = EXPECTED_PAYLOAD_BITS,
) -> PayloadDecodingResult:
    """
    Decode an already ordered sequence of corrected logical bits.
    """

    expected_bits = validate_integer(
        expected_payload_bits,
        field_name="expected_payload_bits",
        minimum=8,
    )

    if expected_bits % 8 != 0:
        raise PayloadDecodingError(
            (
                "expected_payload_bits must be "
                "divisible by eight."
            )
        )

    normalized_bits = normalize_bit_sequence(
        logical_bits,
        field_name="logical_bits",
        exact_length=expected_bits,
    )

    payload = bits_to_bytes(
        normalized_bits
    )

    expected_bytes = (
        expected_bits // 8
    )

    if len(payload) != expected_bytes:
        raise PayloadDecodingError(
            "Decoded payload has an invalid byte length.",
            details={
                "expected_bytes": expected_bytes,
                "actual_bytes": len(
                    payload
                ),
            },
        )

    return PayloadDecodingResult(
        logical_bits=normalized_bits,
        payload=payload,
        payload_fingerprint=(
            calculate_payload_fingerprint(
                payload
            )
        ),
        recovered_payload_bits=len(
            normalized_bits
        ),
        expected_payload_bits=(
            expected_bits
        ),
        complete=True,
    )


def decode_payload_from_blocks(
    *,
    corrected_blocks: Mapping[Any, Any]
    | Sequence[Any],
    payload_positions: Sequence[Any],
    expected_payload_bits: int = EXPECTED_PAYLOAD_BITS,
) -> PayloadDecodingResult:
    """
    Reconstruct the authentication payload from corrected frame blocks.
    """

    ordered_bits = collect_ordered_payload_bits(
        corrected_blocks=corrected_blocks,
        payload_positions=payload_positions,
        expected_payload_bits=(
            expected_payload_bits
        ),
    )

    return decode_payload_from_bits(
        ordered_bits,
        expected_payload_bits=(
            expected_payload_bits
        ),
    )


def decode_kmac_tag(
    *,
    corrected_blocks: Mapping[Any, Any]
    | Sequence[Any],
    payload_positions: Sequence[Any],
) -> bytes:
    """
    Recover and return only the 128-bit KMAC authentication tag.
    """

    result = decode_payload_from_blocks(
        corrected_blocks=corrected_blocks,
        payload_positions=payload_positions,
        expected_payload_bits=KMAC_TAG_BITS,
    )

    return bytes(
        result.payload
    )


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _SelfTestCorrectedBlock:
    position: int
    corrected_logical_bit: int
    success: bool = True


def run_payload_decoder_self_test() -> dict[str, Any]:
    """
    Test payload ordering, byte conversion, and failure handling.
    """

    expected_payload = bytes(
        range(
            EXPECTED_PAYLOAD_BYTES
        )
    )

    expected_bits = bytes_to_bits(
        expected_payload
    )

    payload_positions = tuple(
        reversed(
            range(
                EXPECTED_PAYLOAD_BITS
            )
        )
    )

    corrected_blocks = [
        _SelfTestCorrectedBlock(
            position=position,
            corrected_logical_bit=(
                expected_bits[
                    payload_index
                ]
            ),
        )
        for payload_index, position
        in enumerate(
            payload_positions
        )
    ]

    result = decode_payload_from_blocks(
        corrected_blocks=corrected_blocks,
        payload_positions=payload_positions,
    )

    round_trip_pass = (
        result.payload
        == expected_payload
    )

    bits_round_trip_pass = (
        bytes_to_bits(
            bits_to_bytes(
                expected_bits
            )
        )
        == expected_bits
    )

    missing_block_rejected = False

    try:
        decode_payload_from_blocks(
            corrected_blocks=(
                corrected_blocks[:-1]
            ),
            payload_positions=(
                payload_positions
            ),
        )

    except PayloadDecodingError:
        missing_block_rejected = True

    failed_blocks = list(
        corrected_blocks
    )

    failed_blocks[0] = (
        _SelfTestCorrectedBlock(
            position=(
                corrected_blocks[0]
                .position
            ),
            corrected_logical_bit=(
                corrected_blocks[0]
                .corrected_logical_bit
            ),
            success=False,
        )
    )

    failed_block_rejected = False

    try:
        decode_payload_from_blocks(
            corrected_blocks=failed_blocks,
            payload_positions=(
                payload_positions
            ),
        )

    except PayloadDecodingError:
        failed_block_rejected = True

    success = all(
        (
            result.complete,
            round_trip_pass,
            bits_round_trip_pass,
            result.recovered_payload_bits
            == EXPECTED_PAYLOAD_BITS,
            len(result.payload)
            == EXPECTED_PAYLOAD_BYTES,
            missing_block_rejected,
            failed_block_rejected,
        )
    )

    return {
        "success": success,
        "complete": result.complete,
        "recovered_payload_bits": (
            result.recovered_payload_bits
        ),
        "expected_payload_bits": (
            result.expected_payload_bits
        ),
        "payload_bytes": len(
            result.payload
        ),
        "payload_fingerprint": (
            result.payload_fingerprint
        ),
        "round_trip_pass": (
            round_trip_pass
        ),
        "bits_round_trip_pass": (
            bits_round_trip_pass
        ),
        "missing_block_rejected": (
            missing_block_rejected
        ),
        "failed_block_rejected": (
            failed_block_rejected
        ),
    }


__all__ = [
    "EXPECTED_PAYLOAD_BITS",
    "EXPECTED_PAYLOAD_BYTES",
    "PAYLOAD_FINGERPRINT_ALGORITHM",
    "PayloadDecodingError",
    "PayloadDecodingResult",
    "normalize_logical_bit",
    "normalize_bit_sequence",
    "bits_to_bytes",
    "bytes_to_bits",
    "calculate_payload_fingerprint",
    "validate_payload_positions",
    "collect_ordered_payload_bits",
    "decode_payload_from_bits",
    "decode_payload_from_blocks",
    "decode_kmac_tag",
    "run_payload_decoder_self_test",
]