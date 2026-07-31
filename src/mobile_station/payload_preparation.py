"""
Payload Preparation Module
FT-QuPAP Mobile Station

This module implements Cell 36 of the FT-QuPAP notebook:

    Convert the 128-bit KMAC authentication tag into
    128 logical payload-qubit specifications.

Each tag bit becomes one logical payload block:

    block_id      : P0000 ... P0127
    role          : payload
    logical_index : 0 ... 127
    logical_bit   : 0 or 1
    basis         : Z

The Steane encoder later converts each logical payload block into
seven physical qubits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from .control_schedule import LogicalSpec
except ImportError:
    from control_schedule import LogicalSpec


KMAC_TAG_LENGTH_BYTES = 16
KMAC_TAG_LENGTH_BITS = 128

PAYLOAD_BLOCK_PREFIX = "P"
PAYLOAD_PREPARATION_BASIS = "Z"


class PayloadPreparationError(Exception):
    """Raised when FT-QuPAP payload preparation fails."""


def bits_from_bytes(data: bytes) -> list[int]:
    """
    Convert bytes into a most-significant-bit-first bit list.

    Example:

        b"\\xA0" -> [1, 0, 1, 0, 0, 0, 0, 0]

    This matches the bit-conversion method used by the FT-QuPAP
    simulation notebook.
    """

    if not isinstance(data, bytes):
        raise TypeError(
            "data must be bytes."
        )

    if not data:
        raise ValueError(
            "data cannot be empty."
        )

    return [
        int(bit)
        for byte in data
        for bit in f"{byte:08b}"
    ]


def bytes_from_bits(
    bits: Sequence[int],
) -> bytes:
    """
    Convert a most-significant-bit-first bit sequence into bytes.

    This function is the inverse of bits_from_bytes().
    """

    if isinstance(bits, (str, bytes, bytearray)):
        raise TypeError(
            "bits must be a sequence of integers."
        )

    if not isinstance(bits, Sequence):
        raise TypeError(
            "bits must be a sequence."
        )

    if len(bits) == 0:
        raise ValueError(
            "bits cannot be empty."
        )

    if len(bits) % 8 != 0:
        raise ValueError(
            "Bit length must be a multiple of 8."
        )

    normalized_bits: list[int] = []

    for index, bit in enumerate(bits):
        if isinstance(bit, bool):
            normalized_bit = int(bit)

        elif isinstance(bit, int):
            normalized_bit = bit

        else:
            raise TypeError(
                f"Bit at index {index} must be an integer."
            )

        if normalized_bit not in (0, 1):
            raise ValueError(
                f"Bit at index {index} must be 0 or 1."
            )

        normalized_bits.append(
            normalized_bit
        )

    return bytes(
        int(
            "".join(
                str(bit)
                for bit in normalized_bits[
                    index:index + 8
                ]
            ),
            2,
        )
        for index in range(
            0,
            len(normalized_bits),
            8,
        )
    )


def validate_authentication_tag(
    tag: bytes,
    required_length_bytes: int = (
        KMAC_TAG_LENGTH_BYTES
    ),
) -> None:
    """
    Validate the FT-QuPAP KMAC authentication tag.

    The standard protocol configuration requires a 128-bit tag:

        16 bytes × 8 = 128 logical payload blocks
    """

    if not isinstance(tag, bytes):
        raise TypeError(
            "tag must be bytes."
        )

    if not isinstance(
        required_length_bytes,
        int,
    ):
        raise TypeError(
            "required_length_bytes must be an integer."
        )

    if required_length_bytes <= 0:
        raise ValueError(
            "required_length_bytes must be positive."
        )

    if len(tag) != required_length_bytes:
        raise PayloadPreparationError(
            "FT-QuPAP requires a "
            f"{required_length_bytes * 8}-bit "
            "authentication tag. "
            f"Received {len(tag) * 8} bits."
        )


def map_tag_to_logical_specs(
    tag: bytes,
) -> list[LogicalSpec]:
    """
    Convert a 128-bit KMAC tag into 128 logical payload blocks.

    The bits are read in most-significant-bit-first order.

    Args:
        tag:
            The 16-byte KMAC256 authentication tag.

    Returns:
        A list of 128 LogicalSpec objects.
    """

    validate_authentication_tag(tag)

    tag_bits = bits_from_bytes(tag)

    if len(tag_bits) != KMAC_TAG_LENGTH_BITS:
        raise PayloadPreparationError(
            "Payload conversion did not produce "
            f"{KMAC_TAG_LENGTH_BITS} bits."
        )

    payload_specs = [
        LogicalSpec(
            block_id=(
                f"{PAYLOAD_BLOCK_PREFIX}"
                f"{index:04d}"
            ),
            role="payload",
            logical_index=index,
            logical_bit=bit,
            basis=PAYLOAD_PREPARATION_BASIS,
        )
        for index, bit in enumerate(
            tag_bits
        )
    ]

    validate_payload_specs(
        payload_specs
    )

    return payload_specs


def validate_payload_specs(
    payload_specs: Sequence[Any],
) -> None:
    """
    Validate the complete logical payload specification list.
    """

    if isinstance(
        payload_specs,
        (str, bytes, bytearray),
    ):
        raise TypeError(
            "payload_specs must be a sequence of "
            "LogicalSpec objects."
        )

    if not isinstance(
        payload_specs,
        Sequence,
    ):
        raise TypeError(
            "payload_specs must be a sequence."
        )

    if len(payload_specs) != KMAC_TAG_LENGTH_BITS:
        raise PayloadPreparationError(
            "FT-QuPAP requires exactly "
            f"{KMAC_TAG_LENGTH_BITS} payload blocks."
        )

    seen_block_ids: set[str] = set()
    seen_indices: set[int] = set()

    for expected_index, spec in enumerate(
        payload_specs
    ):
        required_attributes = (
            "block_id",
            "role",
            "logical_index",
            "logical_bit",
            "basis",
        )

        for attribute in required_attributes:
            if not hasattr(spec, attribute):
                raise TypeError(
                    "Payload specification is missing "
                    f"{attribute!r}."
                )

        expected_block_id = (
            f"{PAYLOAD_BLOCK_PREFIX}"
            f"{expected_index:04d}"
        )

        if spec.block_id != expected_block_id:
            raise PayloadPreparationError(
                f"Expected block ID "
                f"{expected_block_id!r}, "
                f"received {spec.block_id!r}."
            )

        if spec.role != "payload":
            raise PayloadPreparationError(
                "Every tag block must use "
                "role='payload'."
            )

        if spec.logical_index != expected_index:
            raise PayloadPreparationError(
                "Payload logical indices must be "
                "continuous from 0 to 127."
            )

        if spec.logical_bit not in (0, 1):
            raise PayloadPreparationError(
                "Payload logical bits must be 0 or 1."
            )

        if spec.basis != (
            PAYLOAD_PREPARATION_BASIS
        ):
            raise PayloadPreparationError(
                "FT-QuPAP payload blocks must use "
                "the Z preparation basis."
            )

        if spec.block_id in seen_block_ids:
            raise PayloadPreparationError(
                f"Duplicate block ID: "
                f"{spec.block_id!r}."
            )

        if spec.logical_index in seen_indices:
            raise PayloadPreparationError(
                "Duplicate payload logical index: "
                f"{spec.logical_index}."
            )

        seen_block_ids.add(
            spec.block_id
        )

        seen_indices.add(
            spec.logical_index
        )


def logical_specs_to_tag(
    payload_specs: Sequence[Any],
) -> bytes:
    """
    Reconstruct a KMAC tag from ordered logical payload specs.

    This helper is mainly used for testing and server-side payload
    recovery validation.
    """

    validate_payload_specs(
        payload_specs
    )

    ordered_specs = sorted(
        payload_specs,
        key=lambda spec: spec.logical_index,
    )

    recovered_bits = [
        int(spec.logical_bit)
        for spec in ordered_specs
    ]

    recovered_tag = bytes_from_bits(
        recovered_bits
    )

    validate_authentication_tag(
        recovered_tag
    )

    return recovered_tag


def payload_public_summary(
    payload_specs: Sequence[Any],
) -> dict[str, Any]:
    """
    Return a summary without exposing the raw authentication tag.
    """

    validate_payload_specs(
        payload_specs
    )

    zero_count = sum(
        int(spec.logical_bit == 0)
        for spec in payload_specs
    )

    one_count = sum(
        int(spec.logical_bit == 1)
        for spec in payload_specs
    )

    return {
        "payload_block_count":
            len(payload_specs),
        "logical_zero_count":
            zero_count,
        "logical_one_count":
            one_count,
        "preparation_basis":
            PAYLOAD_PREPARATION_BASIS,
        "first_block_id":
            payload_specs[0].block_id,
        "last_block_id":
            payload_specs[-1].block_id,
    }


def run_self_test() -> None:
    """
    Test tag-to-logical-payload conversion and reconstruction.
    """

    print("=" * 68)
    print("FT-QuPAP Payload Preparation Self-Test")
    print("=" * 68)

    test_tag = bytes.fromhex(
        "a55ac33c96696996f00f0ff05aa55aa5"
    )

    payload_specs = (
        map_tag_to_logical_specs(
            test_tag
        )
    )

    recovered_tag = (
        logical_specs_to_tag(
            payload_specs
        )
    )

    round_trip_matches = (
        recovered_tag == test_tag
    )

    print(
        f"KMAC tag bytes          : "
        f"{len(test_tag)}"
    )
    print(
        f"KMAC tag bits           : "
        f"{len(bits_from_bytes(test_tag))}"
    )
    print(
        f"Payload logical blocks  : "
        f"{len(payload_specs)}"
    )
    print(
        f"First payload block     : "
        f"{payload_specs[0]}"
    )
    print(
        f"Last payload block      : "
        f"{payload_specs[-1]}"
    )
    print(
        f"All blocks use Z basis  : "
        f"{all(spec.basis == 'Z' for spec in payload_specs)}"
    )
    print(
        f"Tag reconstruction match: "
        f"{round_trip_matches}"
    )

    if not round_trip_matches:
        raise PayloadPreparationError(
            "Tag reconstruction self-test failed."
        )

    print("\nSafe payload summary:")

    print(
        payload_public_summary(
            payload_specs
        )
    )

    print(
        "\nPayload preparation self-test "
        "completed successfully."
    )


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        PayloadPreparationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"\n[PAYLOAD PREPARATION ERROR] "
            f"{error}"
        )