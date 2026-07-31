"""
Classical-to-Qubit Conversion
FT-QuPAP Quantum Simulation Package

This module converts classical binary values into logical quantum-state
descriptions used by the FT-QuPAP syndrome-level simulator.

FT-QuPAP payload conversion:

    16-byte KMAC tag
            |
            v
    128 classical bits
            |
            v
    128 logical payload states in the Z basis

Bit ordering:

    Every byte is converted most-significant bit first.

Example:

    0xA2 = 10100010

The functions in this module do not create physical quantum hardware
states. They produce deterministic classical descriptions that are later
used by logical-qubit and Steane CSS simulation modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


SUPPORTED_BASES = (
    "Z",
    "X",
)

KMAC_TAG_LENGTH_BYTES = 16
KMAC_TAG_LENGTH_BITS = 128

BIT_ORDER = "MSB-first"


class ClassicalToQubitError(Exception):
    """Base exception for classical-to-qubit conversion failures."""


class InvalidClassicalBitError(ClassicalToQubitError):
    """Raised when a value is not a valid classical bit."""


class InvalidQubitBasisError(ClassicalToQubitError):
    """Raised when an unsupported qubit basis is requested."""


class InvalidBitSequenceError(ClassicalToQubitError):
    """Raised when a binary sequence is malformed."""


@dataclass(frozen=True)
class QubitStateDescription:
    """
    Classical description of one intended logical qubit state.

    Attributes:
        classical_bit:
            Source classical value, either 0 or 1.

        basis:
            Preparation basis, either Z or X.

        ket:
            Human-readable state label:

                Z basis:
                    0 -> |0>
                    1 -> |1>

                X basis:
                    0 -> |+>
                    1 -> |->

        bit_index:
            Original position in the classical bit sequence.
    """

    classical_bit: int
    basis: str
    ket: str
    bit_index: int

    def __post_init__(self) -> None:
        validate_classical_bit(
            self.classical_bit
        )

        validate_basis(
            self.basis
        )

        if not isinstance(
            self.ket,
            str,
        ):
            raise TypeError(
                "ket must be a string."
            )

        expected_ket = state_label(
            self.classical_bit,
            self.basis,
        )

        if self.ket != expected_ket:
            raise ValueError(
                "ket does not match the classical bit "
                "and preparation basis."
            )

        if isinstance(
            self.bit_index,
            bool,
        ) or not isinstance(
            self.bit_index,
            int,
        ):
            raise TypeError(
                "bit_index must be an integer."
            )

        if self.bit_index < 0:
            raise ValueError(
                "bit_index cannot be negative."
            )

    def to_dictionary(self) -> dict[str, Any]:
        """Return a serializable state description."""

        return {
            "classical_bit":
                self.classical_bit,
            "basis":
                self.basis,
            "ket":
                self.ket,
            "bit_index":
                self.bit_index,
        }


def validate_classical_bit(
    bit: Any,
) -> int:
    """
    Validate and normalize one classical bit.

    Boolean values are rejected to avoid silently treating True and
    False as protocol bit values.
    """

    if isinstance(bit, bool):
        raise InvalidClassicalBitError(
            "Boolean values are not accepted as classical bits."
        )

    if not isinstance(bit, int):
        raise InvalidClassicalBitError(
            "A classical bit must be an integer."
        )

    if bit not in (
        0,
        1,
    ):
        raise InvalidClassicalBitError(
            "A classical bit must be either 0 or 1."
        )

    return bit


def validate_basis(
    basis: Any,
) -> str:
    """
    Validate and normalize a logical preparation basis.
    """

    if not isinstance(
        basis,
        str,
    ):
        raise InvalidQubitBasisError(
            "basis must be a string."
        )

    normalized_basis = (
        basis.strip().upper()
    )

    if normalized_basis not in (
        SUPPORTED_BASES
    ):
        raise InvalidQubitBasisError(
            "basis must be either 'Z' or 'X'."
        )

    return normalized_basis


def normalize_bits(
    bits: Iterable[int],
    require_non_empty: bool = False,
) -> list[int]:
    """
    Validate and normalize an iterable of classical bits.
    """

    if isinstance(
        bits,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise InvalidBitSequenceError(
            "bits must be an iterable of integers, "
            "not text or raw bytes."
        )

    try:
        bit_list = list(bits)

    except TypeError as error:
        raise InvalidBitSequenceError(
            "bits must be iterable."
        ) from error

    if (
        require_non_empty
        and not bit_list
    ):
        raise InvalidBitSequenceError(
            "bits cannot be empty."
        )

    normalized: list[int] = []

    for index, bit in enumerate(
        bit_list
    ):
        try:
            normalized_bit = (
                validate_classical_bit(
                    bit
                )
            )

        except InvalidClassicalBitError as error:
            raise InvalidBitSequenceError(
                f"Invalid bit at index {index}: {error}"
            ) from error

        normalized.append(
            normalized_bit
        )

    return normalized


def bits_from_bytes(
    data: bytes,
) -> list[int]:
    """
    Convert bytes into classical bits using MSB-first ordering.

    Notebook-aligned definition:

        [int(bit) for byte in data for bit in f"{byte:08b}"]

    Example:

        bits_from_bytes(bytes([0xA2]))

        returns:

        [1, 0, 1, 0, 0, 0, 1, 0]
    """

    if not isinstance(
        data,
        bytes,
    ):
        raise TypeError(
            "data must be bytes."
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
    Convert an MSB-first classical-bit sequence into bytes.

    The number of bits must be a multiple of eight.
    """

    normalized_bits = normalize_bits(
        bits
    )

    if len(normalized_bits) % 8 != 0:
        raise InvalidBitSequenceError(
            "Bit length must be a multiple of 8."
        )

    return bytes(
        int(
            "".join(
                str(bit)
                for bit in normalized_bits[
                    start:start + 8
                ]
            ),
            2,
        )
        for start in range(
            0,
            len(normalized_bits),
            8,
        )
    )


def state_label(
    classical_bit: int,
    basis: str = "Z",
) -> str:
    """
    Map one classical bit to a logical qubit-state label.

    Z-basis mapping:

        0 -> |0>
        1 -> |1>

    X-basis mapping:

        0 -> |+>
        1 -> |->
    """

    normalized_bit = (
        validate_classical_bit(
            classical_bit
        )
    )

    normalized_basis = (
        validate_basis(
            basis
        )
    )

    if normalized_basis == "Z":
        return (
            "|0>"
            if normalized_bit == 0
            else "|1>"
        )

    return (
        "|+>"
        if normalized_bit == 0
        else "|->"
    )


def classical_bit_to_qubit_state(
    classical_bit: int,
    basis: str = "Z",
    bit_index: int = 0,
) -> QubitStateDescription:
    """
    Convert one classical bit into a logical state description.
    """

    normalized_bit = (
        validate_classical_bit(
            classical_bit
        )
    )

    normalized_basis = (
        validate_basis(
            basis
        )
    )

    if isinstance(
        bit_index,
        bool,
    ) or not isinstance(
        bit_index,
        int,
    ):
        raise TypeError(
            "bit_index must be an integer."
        )

    if bit_index < 0:
        raise ValueError(
            "bit_index cannot be negative."
        )

    return QubitStateDescription(
        classical_bit=
            normalized_bit,
        basis=
            normalized_basis,
        ket=state_label(
            normalized_bit,
            normalized_basis,
        ),
        bit_index=
            bit_index,
    )


def classical_bits_to_qubit_states(
    bits: Sequence[int],
    basis: str = "Z",
) -> list[QubitStateDescription]:
    """
    Convert a binary sequence into logical qubit-state descriptions.

    FT-QuPAP payload bits normally use the Z basis. Independent check
    blocks may later use either the Z or X basis.
    """

    normalized_bits = normalize_bits(
        bits
    )

    normalized_basis = (
        validate_basis(
            basis
        )
    )

    return [
        classical_bit_to_qubit_state(
            classical_bit=bit,
            basis=normalized_basis,
            bit_index=index,
        )
        for index, bit in enumerate(
            normalized_bits
        )
    ]


def tag_to_payload_bits(
    authentication_tag: bytes,
    require_standard_length: bool = True,
) -> list[int]:
    """
    Convert a KMAC authentication tag into payload bits.

    Under the standard FT-QuPAP configuration:

        16 bytes -> 128 bits
    """

    if not isinstance(
        authentication_tag,
        bytes,
    ):
        raise TypeError(
            "authentication_tag must be bytes."
        )

    if (
        require_standard_length
        and len(authentication_tag)
        != KMAC_TAG_LENGTH_BYTES
    ):
        raise ValueError(
            "The FT-QuPAP authentication tag must contain "
            f"exactly {KMAC_TAG_LENGTH_BYTES} bytes."
        )

    payload_bits = bits_from_bytes(
        authentication_tag
    )

    if (
        require_standard_length
        and len(payload_bits)
        != KMAC_TAG_LENGTH_BITS
    ):
        raise ClassicalToQubitError(
            "Authentication-tag conversion did not produce "
            f"{KMAC_TAG_LENGTH_BITS} bits."
        )

    return payload_bits


def payload_bits_to_tag(
    payload_bits: Sequence[int],
    require_standard_length: bool = True,
) -> bytes:
    """
    Reconstruct the KMAC tag from decoded payload bits.
    """

    normalized_bits = normalize_bits(
        payload_bits
    )

    if (
        require_standard_length
        and len(normalized_bits)
        != KMAC_TAG_LENGTH_BITS
    ):
        raise ValueError(
            "The FT-QuPAP payload must contain exactly "
            f"{KMAC_TAG_LENGTH_BITS} bits."
        )

    authentication_tag = bytes_from_bits(
        normalized_bits
    )

    if (
        require_standard_length
        and len(authentication_tag)
        != KMAC_TAG_LENGTH_BYTES
    ):
        raise ClassicalToQubitError(
            "Payload reconstruction did not produce "
            f"{KMAC_TAG_LENGTH_BYTES} bytes."
        )

    return authentication_tag


def tag_to_qubit_states(
    authentication_tag: bytes,
) -> list[QubitStateDescription]:
    """
    Convert the standard 128-bit KMAC tag into 128 Z-basis states.
    """

    payload_bits = tag_to_payload_bits(
        authentication_tag,
        require_standard_length=True,
    )

    return classical_bits_to_qubit_states(
        payload_bits,
        basis="Z",
    )


# More descriptive aliases for external modules.
bytes_to_bits = bits_from_bytes
bits_to_bytes = bytes_from_bits
classical_to_qubit = classical_bit_to_qubit_state


def run_self_test() -> None:
    """
    Verify byte/bit conversion and logical-state mapping.
    """

    print("=" * 70)
    print("FT-QuPAP Classical-to-Qubit Self-Test")
    print("=" * 70)

    sample_byte = bytes(
        [0xA2]
    )

    expected_bits = [
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
    ]

    converted_bits = bits_from_bytes(
        sample_byte
    )

    byte_round_trip = (
        bytes_from_bits(
            converted_bits
        )
        == sample_byte
    )

    sample_tag = bytes(
        range(
            KMAC_TAG_LENGTH_BYTES
        )
    )

    tag_bits = tag_to_payload_bits(
        sample_tag
    )

    payload_states = (
        tag_to_qubit_states(
            sample_tag
        )
    )

    tag_round_trip = (
        payload_bits_to_tag(
            tag_bits
        )
        == sample_tag
    )

    all_payload_states_z_basis = all(
        state.basis == "Z"
        for state in payload_states
    )

    state_bits_preserved = [
        state.classical_bit
        for state in payload_states
    ] == tag_bits

    x_zero = state_label(
        0,
        "X",
    )

    x_one = state_label(
        1,
        "X",
    )

    print(
        f"Bit order                 : {BIT_ORDER}"
    )

    print(
        f"0xA2 conversion           : {converted_bits}"
    )

    print(
        f"Expected conversion       : {expected_bits}"
    )

    print(
        f"Single-byte round-trip    : {byte_round_trip}"
    )

    print(
        f"KMAC tag bytes            : {len(sample_tag)}"
    )

    print(
        f"Payload bits              : {len(tag_bits)}"
    )

    print(
        f"Logical state descriptions: {len(payload_states)}"
    )

    print(
        f"All payload states Z basis: {all_payload_states_z_basis}"
    )

    print(
        f"Payload bits preserved    : {state_bits_preserved}"
    )

    print(
        f"Tag round-trip            : {tag_round_trip}"
    )

    print(
        f"X-basis mapping           : 0 -> {x_zero}, 1 -> {x_one}"
    )

    if converted_bits != expected_bits:
        raise ClassicalToQubitError(
            "MSB-first conversion failed."
        )

    if not byte_round_trip:
        raise ClassicalToQubitError(
            "Single-byte round-trip failed."
        )

    if len(tag_bits) != (
        KMAC_TAG_LENGTH_BITS
    ):
        raise ClassicalToQubitError(
            "Incorrect FT-QuPAP payload-bit count."
        )

    if len(payload_states) != (
        KMAC_TAG_LENGTH_BITS
    ):
        raise ClassicalToQubitError(
            "Incorrect logical-state count."
        )

    if not all_payload_states_z_basis:
        raise ClassicalToQubitError(
            "A payload state was not assigned to the Z basis."
        )

    if not state_bits_preserved:
        raise ClassicalToQubitError(
            "Logical-state conversion changed a bit."
        )

    if not tag_round_trip:
        raise ClassicalToQubitError(
            "Authentication-tag round-trip failed."
        )

    print(
        "\nClassical-to-qubit self-test "
        "completed successfully."
    )


__all__ = [
    "SUPPORTED_BASES",
    "KMAC_TAG_LENGTH_BYTES",
    "KMAC_TAG_LENGTH_BITS",
    "BIT_ORDER",
    "ClassicalToQubitError",
    "InvalidClassicalBitError",
    "InvalidQubitBasisError",
    "InvalidBitSequenceError",
    "QubitStateDescription",
    "validate_classical_bit",
    "validate_basis",
    "normalize_bits",
    "bits_from_bytes",
    "bytes_from_bits",
    "bytes_to_bits",
    "bits_to_bytes",
    "state_label",
    "classical_bit_to_qubit_state",
    "classical_bits_to_qubit_states",
    "classical_to_qubit",
    "tag_to_payload_bits",
    "payload_bits_to_tag",
    "tag_to_qubit_states",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        ClassicalToQubitError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[CLASSICAL-TO-QUBIT ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error