"""
Quantum Payload Generator
FT-QuPAP Quantum Simulation Package

This module converts the 128-bit KMAC authentication tag into the
logical payload used by the FT-QuPAP quantum plane.

Protocol mapping:

    16-byte KMAC tag
            |
            v
    128 classical bits, MSB first
            |
            v
    128 logical payload blocks
            |
            v
    P0000, P0001, ..., P0127

Every payload block uses the Z basis:

    bit 0 -> |0_L>
    bit 1 -> |1_L>

The payload blocks are not yet Steane encoded and do not yet have
interleaved frame positions. Check blocks and random interleaving are
performed by separate modules.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .classical_to_qubit import (
    KMAC_TAG_LENGTH_BITS,
    KMAC_TAG_LENGTH_BYTES,
    payload_bits_to_tag,
    tag_to_payload_bits,
)
from .logical_qubit import (
    LogicalQubit,
    PAYLOAD_ROLE,
    create_payload_logical_qubit,
    validate_logical_qubit_collection,
)


PAYLOAD_LOGICAL_BLOCK_COUNT = 128
PAYLOAD_BASIS = "Z"
PAYLOAD_BLOCK_PREFIX = "P"


class PayloadGenerationError(Exception):
    """Base exception for quantum-payload generation."""


class InvalidAuthenticationTagError(
    PayloadGenerationError
):
    """Raised when the KMAC authentication tag is invalid."""


class InvalidPayloadCollectionError(
    PayloadGenerationError
):
    """Raised when logical payload blocks are inconsistent."""


@dataclass(frozen=True)
class PayloadGenerationResult:
    """
    Result of mapping a KMAC tag into logical payload blocks.

    Attributes:
        authentication_tag:
            Original 16-byte KMAC authentication tag.

        payload_bits:
            The 128 MSB-first classical tag bits.

        payload_blocks:
            The 128 Z-basis logical payload blocks.
    """

    authentication_tag: bytes
    payload_bits: tuple[int, ...]
    payload_blocks: tuple[LogicalQubit, ...]

    def __post_init__(self) -> None:
        validate_authentication_tag(
            self.authentication_tag
        )

        validate_payload_bits(
            self.payload_bits,
            require_standard_count=True,
        )

        validate_payload_blocks(
            self.payload_blocks,
            require_standard_count=True,
            require_unpositioned=True,
        )

        block_bits = tuple(
            block.logical_bit
            for block in self.payload_blocks
        )

        if block_bits != self.payload_bits:
            raise InvalidPayloadCollectionError(
                "Payload block values do not match "
                "the original authentication-tag bits."
            )

        reconstructed_tag = (
            payload_bits_to_tag(
                self.payload_bits,
                require_standard_length=True,
            )
        )

        if (
            reconstructed_tag
            != self.authentication_tag
        ):
            raise InvalidPayloadCollectionError(
                "Payload bits cannot reconstruct "
                "the original authentication tag."
            )

    @property
    def tag_length_bytes(self) -> int:
        """Return authentication-tag length."""

        return len(
            self.authentication_tag
        )

    @property
    def tag_length_bits(self) -> int:
        """Return authentication-tag bit length."""

        return len(
            self.payload_bits
        )

    @property
    def logical_block_count(self) -> int:
        """Return payload logical-block count."""

        return len(
            self.payload_blocks
        )

    def safe_summary(self) -> dict[str, Any]:
        """
        Return metadata without exposing the raw tag or its bits.
        """

        return {
            "tag_algorithm":
                "KMAC256",
            "tag_length_bytes":
                self.tag_length_bytes,
            "tag_length_bits":
                self.tag_length_bits,
            "logical_payload_blocks":
                self.logical_block_count,
            "payload_basis":
                PAYLOAD_BASIS,
            "first_block_id":
                self.payload_blocks[0].block_id,
            "last_block_id":
                self.payload_blocks[-1].block_id,
            "tag_fingerprint":
                hashlib.sha3_256(
                    self.authentication_tag
                ).hexdigest()[:16],
        }

    def to_block_dictionaries(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return serializable logical-block descriptions.

        This method includes payload bits and should therefore be used
        only inside the controlled protocol simulator.
        """

        return [
            block.to_dictionary()
            for block in self.payload_blocks
        ]


def validate_authentication_tag(
    authentication_tag: Any,
    require_standard_length: bool = True,
) -> bytes:
    """
    Validate a KMAC authentication tag.

    Standard FT-QuPAP requires exactly 16 bytes, or 128 bits.
    """

    if not isinstance(
        authentication_tag,
        bytes,
    ):
        raise InvalidAuthenticationTagError(
            "authentication_tag must be bytes."
        )

    if len(authentication_tag) == 0:
        raise InvalidAuthenticationTagError(
            "authentication_tag cannot be empty."
        )

    if (
        require_standard_length
        and len(authentication_tag)
        != KMAC_TAG_LENGTH_BYTES
    ):
        raise InvalidAuthenticationTagError(
            "FT-QuPAP requires a "
            f"{KMAC_TAG_LENGTH_BYTES}-byte "
            "KMAC authentication tag."
        )

    return authentication_tag


def validate_payload_bits(
    payload_bits: Sequence[int],
    require_standard_count: bool = True,
) -> list[int]:
    """
    Validate a logical payload-bit sequence.
    """

    if isinstance(
        payload_bits,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise InvalidPayloadCollectionError(
            "payload_bits must be a sequence "
            "of integer bits."
        )

    if not isinstance(
        payload_bits,
        Sequence,
    ):
        raise InvalidPayloadCollectionError(
            "payload_bits must be a sequence."
        )

    normalized_bits: list[int] = []

    for index, bit in enumerate(
        payload_bits
    ):
        if isinstance(bit, bool):
            raise InvalidPayloadCollectionError(
                f"Payload bit at index {index} "
                "cannot be boolean."
            )

        if not isinstance(bit, int):
            raise InvalidPayloadCollectionError(
                f"Payload bit at index {index} "
                "must be an integer."
            )

        if bit not in (
            0,
            1,
        ):
            raise InvalidPayloadCollectionError(
                f"Payload bit at index {index} "
                "must be 0 or 1."
            )

        normalized_bits.append(
            bit
        )

    if (
        require_standard_count
        and len(normalized_bits)
        != PAYLOAD_LOGICAL_BLOCK_COUNT
    ):
        raise InvalidPayloadCollectionError(
            "FT-QuPAP requires exactly "
            f"{PAYLOAD_LOGICAL_BLOCK_COUNT} "
            "payload bits."
        )

    return normalized_bits


def validate_payload_blocks(
    payload_blocks: Sequence[LogicalQubit],
    require_standard_count: bool = True,
    require_unpositioned: bool = False,
) -> None:
    """
    Validate logical payload blocks.

    Required properties:

    - role is payload
    - basis is Z
    - IDs are P0000 through P0127
    - logical indices are sequential
    - positions are normally None before interleaving
    """

    expected_count = (
        PAYLOAD_LOGICAL_BLOCK_COUNT
        if require_standard_count
        else None
    )

    validate_logical_qubit_collection(
        logical_qubits=payload_blocks,
        expected_role=PAYLOAD_ROLE,
        expected_count=expected_count,
        require_positions=False,
        require_unique_ids=True,
        require_unique_positions=True,
    )

    for expected_index, block in enumerate(
        payload_blocks
    ):
        if block.basis != PAYLOAD_BASIS:
            raise InvalidPayloadCollectionError(
                f"Payload block "
                f"{block.block_id!r} must use "
                "the Z basis."
            )

        if (
            block.logical_index
            != expected_index
        ):
            raise InvalidPayloadCollectionError(
                f"Payload block at collection index "
                f"{expected_index} declares logical "
                f"index {block.logical_index}."
            )

        expected_block_id = (
            f"{PAYLOAD_BLOCK_PREFIX}"
            f"{expected_index:04d}"
        )

        if (
            block.block_id
            != expected_block_id
        ):
            raise InvalidPayloadCollectionError(
                f"Expected payload block ID "
                f"{expected_block_id!r}, received "
                f"{block.block_id!r}."
            )

        if (
            require_unpositioned
            and block.position is not None
        ):
            raise InvalidPayloadCollectionError(
                f"Payload block "
                f"{block.block_id!r} already has "
                "an interleaved position."
            )


def generate_payload_bits(
    authentication_tag: bytes,
) -> list[int]:
    """
    Convert the standard KMAC tag into 128 MSB-first bits.
    """

    validate_authentication_tag(
        authentication_tag,
        require_standard_length=True,
    )

    payload_bits = tag_to_payload_bits(
        authentication_tag,
        require_standard_length=True,
    )

    validate_payload_bits(
        payload_bits,
        require_standard_count=True,
    )

    return payload_bits


def create_payload_blocks_from_bits(
    payload_bits: Sequence[int],
    require_standard_count: bool = True,
) -> list[LogicalQubit]:
    """
    Create ordered Z-basis logical payload blocks from bits.

    No random values are used. The same bit sequence always creates
    the same logical payload description.
    """

    normalized_bits = validate_payload_bits(
        payload_bits,
        require_standard_count=(
            require_standard_count
        ),
    )

    payload_blocks = [
        create_payload_logical_qubit(
            logical_bit=bit,
            logical_index=index,
            position=None,
        )
        for index, bit in enumerate(
            normalized_bits
        )
    ]

    validate_payload_blocks(
        payload_blocks,
        require_standard_count=(
            require_standard_count
        ),
        require_unpositioned=True,
    )

    return payload_blocks


def generate_payload_blocks(
    authentication_tag: bytes,
) -> list[LogicalQubit]:
    """
    Generate the 128 logical FT-QuPAP payload blocks.

    This is the primary function used by the quantum payload layer.
    """

    payload_bits = generate_payload_bits(
        authentication_tag
    )

    return create_payload_blocks_from_bits(
        payload_bits,
        require_standard_count=True,
    )


def generate_quantum_payload(
    authentication_tag: bytes,
) -> PayloadGenerationResult:
    """
    Generate a complete logical-payload result object.
    """

    validated_tag = (
        validate_authentication_tag(
            authentication_tag,
            require_standard_length=True,
        )
    )

    payload_bits = generate_payload_bits(
        validated_tag
    )

    payload_blocks = (
        create_payload_blocks_from_bits(
            payload_bits,
            require_standard_count=True,
        )
    )

    return PayloadGenerationResult(
        authentication_tag=(
            bytes(validated_tag)
        ),
        payload_bits=tuple(
            payload_bits
        ),
        payload_blocks=tuple(
            payload_blocks
        ),
    )


def recover_payload_bits(
    payload_blocks: Sequence[LogicalQubit],
) -> list[int]:
    """
    Recover ordered payload bits from logical blocks.

    The blocks may have been randomly interleaved earlier. Recovery
    therefore sorts them by logical_index rather than frame position.
    """

    validate_payload_blocks(
        payload_blocks,
        require_standard_count=True,
        require_unpositioned=False,
    )

    ordered_blocks = sorted(
        payload_blocks,
        key=lambda block:
            block.logical_index,
    )

    return [
        block.logical_bit
        for block in ordered_blocks
    ]


def recover_authentication_tag(
    payload_blocks: Sequence[LogicalQubit],
) -> bytes:
    """
    Reconstruct the original 16-byte KMAC tag from logical blocks.
    """

    payload_bits = recover_payload_bits(
        payload_blocks
    )

    return payload_bits_to_tag(
        payload_bits,
        require_standard_length=True,
    )


def payload_block_lookup(
    payload_blocks: Sequence[LogicalQubit],
) -> dict[str, LogicalQubit]:
    """
    Build a payload-block lookup table keyed by block ID.
    """

    validate_payload_blocks(
        payload_blocks,
        require_standard_count=True,
        require_unpositioned=False,
    )

    return {
        block.block_id: block
        for block in payload_blocks
    }


def payload_index_lookup(
    payload_blocks: Sequence[LogicalQubit],
) -> dict[int, LogicalQubit]:
    """
    Build a payload-block lookup table keyed by logical index.
    """

    validate_payload_blocks(
        payload_blocks,
        require_standard_count=True,
        require_unpositioned=False,
    )

    return {
        block.logical_index: block
        for block in payload_blocks
    }


# Compatibility aliases used by other FT-QuPAP modules.
map_tag_to_payload_blocks = (
    generate_payload_blocks
)

map_tag_to_logical_payload = (
    generate_payload_blocks
)

prepare_payload = generate_quantum_payload


def run_self_test() -> None:
    """
    Verify deterministic payload creation and tag recovery.
    """

    print("=" * 72)
    print("FT-QuPAP Quantum Payload Generator Self-Test")
    print("=" * 72)

    sample_tag = bytes.fromhex(
        "00112233445566778899aabbccddeeff"
    )

    first_result = (
        generate_quantum_payload(
            sample_tag
        )
    )

    second_result = (
        generate_quantum_payload(
            sample_tag
        )
    )

    payload_blocks = list(
        first_result.payload_blocks
    )

    recovered_tag = (
        recover_authentication_tag(
            payload_blocks
        )
    )

    generated_deterministically = (
        first_result.payload_bits
        == second_result.payload_bits
        and [
            block.to_dictionary()
            for block
            in first_result.payload_blocks
        ]
        == [
            block.to_dictionary()
            for block
            in second_result.payload_blocks
        ]
    )

    all_payload_roles = all(
        block.role == PAYLOAD_ROLE
        for block in payload_blocks
    )

    all_z_basis = all(
        block.basis == PAYLOAD_BASIS
        for block in payload_blocks
    )

    all_unpositioned = all(
        block.position is None
        for block in payload_blocks
    )

    sequential_indices = all(
        block.logical_index == index
        for index, block in enumerate(
            payload_blocks
        )
    )

    sequential_ids = all(
        block.block_id
        == f"P{index:04d}"
        for index, block in enumerate(
            payload_blocks
        )
    )

    first_byte_bits = list(
        first_result.payload_bits[:8]
    )

    expected_first_byte_bits = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    second_byte_bits = list(
        first_result.payload_bits[8:16]
    )

    expected_second_byte_bits = [
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
    ]

    print(
        f"KMAC tag bytes            : "
        f"{first_result.tag_length_bytes}"
    )

    print(
        f"KMAC tag bits             : "
        f"{first_result.tag_length_bits}"
    )

    print(
        f"Logical payload blocks    : "
        f"{first_result.logical_block_count}"
    )

    print(
        f"First payload block       : "
        f"{payload_blocks[0].to_dictionary()}"
    )

    print(
        f"Last payload block        : "
        f"{payload_blocks[-1].to_dictionary()}"
    )

    print(
        f"First tag byte bits       : "
        f"{first_byte_bits}"
    )

    print(
        f"Second tag byte bits      : "
        f"{second_byte_bits}"
    )

    print(
        f"All roles are payload     : "
        f"{all_payload_roles}"
    )

    print(
        f"All blocks use Z basis    : "
        f"{all_z_basis}"
    )

    print(
        f"All blocks unpositioned   : "
        f"{all_unpositioned}"
    )

    print(
        f"Logical indices sequential: "
        f"{sequential_indices}"
    )

    print(
        f"Block IDs sequential      : "
        f"{sequential_ids}"
    )

    print(
        f"Generation deterministic  : "
        f"{generated_deterministically}"
    )

    print(
        f"Recovered tag matches     : "
        f"{recovered_tag == sample_tag}"
    )

    print(
        f"Safe result summary       : "
        f"{first_result.safe_summary()}"
    )

    if (
        first_result.tag_length_bytes
        != KMAC_TAG_LENGTH_BYTES
    ):
        raise PayloadGenerationError(
            "Incorrect authentication-tag length."
        )

    if (
        first_result.tag_length_bits
        != KMAC_TAG_LENGTH_BITS
    ):
        raise PayloadGenerationError(
            "Incorrect authentication-tag bit count."
        )

    if (
        first_result.logical_block_count
        != PAYLOAD_LOGICAL_BLOCK_COUNT
    ):
        raise PayloadGenerationError(
            "Incorrect payload logical-block count."
        )

    if (
        first_byte_bits
        != expected_first_byte_bits
    ):
        raise PayloadGenerationError(
            "First byte was not converted MSB first."
        )

    if (
        second_byte_bits
        != expected_second_byte_bits
    ):
        raise PayloadGenerationError(
            "Second byte was not converted MSB first."
        )

    if not all_payload_roles:
        raise PayloadGenerationError(
            "A generated block has the wrong role."
        )

    if not all_z_basis:
        raise PayloadGenerationError(
            "A payload block does not use the Z basis."
        )

    if not all_unpositioned:
        raise PayloadGenerationError(
            "A generated payload block already has "
            "a frame position."
        )

    if not sequential_indices:
        raise PayloadGenerationError(
            "Payload logical indices are not sequential."
        )

    if not sequential_ids:
        raise PayloadGenerationError(
            "Payload block IDs are not sequential."
        )

    if not generated_deterministically:
        raise PayloadGenerationError(
            "Identical tags generated different payloads."
        )

    if recovered_tag != sample_tag:
        raise PayloadGenerationError(
            "Authentication-tag recovery failed."
        )

    print(
        "\nQuantum payload generator self-test "
        "completed successfully."
    )


__all__ = [
    "PAYLOAD_LOGICAL_BLOCK_COUNT",
    "PAYLOAD_BASIS",
    "PAYLOAD_BLOCK_PREFIX",
    "PayloadGenerationError",
    "InvalidAuthenticationTagError",
    "InvalidPayloadCollectionError",
    "PayloadGenerationResult",
    "validate_authentication_tag",
    "validate_payload_bits",
    "validate_payload_blocks",
    "generate_payload_bits",
    "create_payload_blocks_from_bits",
    "generate_payload_blocks",
    "generate_quantum_payload",
    "recover_payload_bits",
    "recover_authentication_tag",
    "payload_block_lookup",
    "payload_index_lookup",
    "map_tag_to_payload_blocks",
    "map_tag_to_logical_payload",
    "prepare_payload",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        PayloadGenerationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[PAYLOAD GENERATION ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error