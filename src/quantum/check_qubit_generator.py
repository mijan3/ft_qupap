"""
FT-QuPAP Check-Qubit Generator

This module generates the independent logical check states used for
raw-QBER estimation in the FT-QuPAP quantum authentication protocol.

Standard FT-QuPAP configuration:

    32 independent logical check blocks

For each check block:

    logical bit  -> randomly selected from {0, 1}
    basis        -> randomly selected from {Z, X}
    block ID     -> C0000, C0001, ..., C0031
    position     -> None before random interleaving

The check states are independent of the KMAC authentication payload.
They are later randomly interleaved with the 128 payload blocks and
protected inside the encrypted control schedule.

This module creates syndrome-level logical descriptions. It does not
create physical quantum hardware states.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .logical_qubit import (
    CHECK_ROLE,
    LogicalQubit,
    create_check_logical_qubit,
    validate_logical_qubit_collection,
)


CHECK_LOGICAL_BLOCK_COUNT = 32

CHECK_BLOCK_PREFIX = "C"

SUPPORTED_CHECK_BASES = (
    "Z",
    "X",
)


class CheckQubitGenerationError(Exception):
    """Base exception for check-qubit generation failures."""


class InvalidCheckCountError(
    CheckQubitGenerationError
):
    """Raised when the requested check-block count is invalid."""


class InvalidCheckQubitCollectionError(
    CheckQubitGenerationError
):
    """Raised when generated check blocks are inconsistent."""


@dataclass(frozen=True)
class CheckQubitGenerationResult:
    """
    Result of generating independent FT-QuPAP check blocks.

    Attributes:
        check_blocks:
            Generated logical check blocks.

        check_count:
            Number of generated blocks.

        z_basis_count:
            Number of blocks prepared in the Z basis.

        x_basis_count:
            Number of blocks prepared in the X basis.

        zero_bit_count:
            Number of blocks representing logical bit zero.

        one_bit_count:
            Number of blocks representing logical bit one.
    """

    check_blocks: tuple[LogicalQubit, ...]
    check_count: int
    z_basis_count: int
    x_basis_count: int
    zero_bit_count: int
    one_bit_count: int

    def __post_init__(self) -> None:
        validate_check_blocks(
            self.check_blocks,
            expected_count=self.check_count,
            require_unpositioned=True,
        )

        if self.check_count != len(
            self.check_blocks
        ):
            raise InvalidCheckQubitCollectionError(
                "check_count does not match the number "
                "of generated blocks."
            )

        actual_z_count = sum(
            block.basis == "Z"
            for block in self.check_blocks
        )

        actual_x_count = sum(
            block.basis == "X"
            for block in self.check_blocks
        )

        actual_zero_count = sum(
            block.logical_bit == 0
            for block in self.check_blocks
        )

        actual_one_count = sum(
            block.logical_bit == 1
            for block in self.check_blocks
        )

        if self.z_basis_count != actual_z_count:
            raise InvalidCheckQubitCollectionError(
                "z_basis_count is inconsistent."
            )

        if self.x_basis_count != actual_x_count:
            raise InvalidCheckQubitCollectionError(
                "x_basis_count is inconsistent."
            )

        if self.zero_bit_count != actual_zero_count:
            raise InvalidCheckQubitCollectionError(
                "zero_bit_count is inconsistent."
            )

        if self.one_bit_count != actual_one_count:
            raise InvalidCheckQubitCollectionError(
                "one_bit_count is inconsistent."
            )

        if (
            self.z_basis_count
            + self.x_basis_count
            != self.check_count
        ):
            raise InvalidCheckQubitCollectionError(
                "Basis counts do not equal the total "
                "check-block count."
            )

        if (
            self.zero_bit_count
            + self.one_bit_count
            != self.check_count
        ):
            raise InvalidCheckQubitCollectionError(
                "Logical-bit counts do not equal the total "
                "check-block count."
            )

    def safe_summary(self) -> dict[str, Any]:
        """
        Return non-secret generation metadata.

        Individual check bits and bases are not included because they
        belong inside the protected control schedule.
        """

        return {
            "check_block_count":
                self.check_count,
            "supported_bases":
                list(SUPPORTED_CHECK_BASES),
            "all_blocks_independent":
                True,
            "positions_assigned":
                False,
        }

    def internal_summary(self) -> dict[str, Any]:
        """
        Return simulator-only generation statistics.

        This method does not reveal the complete check schedule but
        should still be kept out of public protocol messages.
        """

        return {
            "check_block_count":
                self.check_count,
            "z_basis_count":
                self.z_basis_count,
            "x_basis_count":
                self.x_basis_count,
            "zero_bit_count":
                self.zero_bit_count,
            "one_bit_count":
                self.one_bit_count,
        }

    def to_block_dictionaries(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return full logical check descriptions.

        These descriptions contain expected check bits and bases.
        They must only be used inside the protected simulator flow.
        """

        return [
            block.to_dictionary()
            for block in self.check_blocks
        ]


def validate_check_count(
    check_count: Any,
    require_standard_count: bool = False,
) -> int:
    """
    Validate a requested check-block count.
    """

    if isinstance(
        check_count,
        bool,
    ):
        raise InvalidCheckCountError(
            "check_count cannot be boolean."
        )

    if not isinstance(
        check_count,
        int,
    ):
        raise InvalidCheckCountError(
            "check_count must be an integer."
        )

    if check_count <= 0:
        raise InvalidCheckCountError(
            "check_count must be greater than zero."
        )

    if (
        require_standard_count
        and check_count
        != CHECK_LOGICAL_BLOCK_COUNT
    ):
        raise InvalidCheckCountError(
            "Standard FT-QuPAP requires exactly "
            f"{CHECK_LOGICAL_BLOCK_COUNT} check blocks."
        )

    return check_count


def validate_rng(
    rng: Any,
) -> np.random.Generator:
    """
    Validate a NumPy random-number generator.
    """

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    return rng


def validate_check_blocks(
    check_blocks: Sequence[LogicalQubit],
    expected_count: int | None = None,
    require_unpositioned: bool = False,
) -> None:
    """
    Validate a collection of logical check blocks.

    Required properties:

    - role is check
    - basis is Z or X
    - logical bit is 0 or 1
    - IDs follow C0000, C0001, ...
    - logical indices are sequential
    - positions are normally None before interleaving
    """

    if expected_count is not None:
        normalized_count = validate_check_count(
            expected_count,
            require_standard_count=False,
        )
    else:
        normalized_count = None

    validate_logical_qubit_collection(
        logical_qubits=check_blocks,
        expected_role=CHECK_ROLE,
        expected_count=normalized_count,
        require_positions=False,
        require_unique_ids=True,
        require_unique_positions=True,
    )

    for expected_index, block in enumerate(
        check_blocks
    ):
        if block.basis not in (
            SUPPORTED_CHECK_BASES
        ):
            raise InvalidCheckQubitCollectionError(
                f"Check block {block.block_id!r} "
                "has an unsupported basis."
            )

        if block.logical_bit not in (
            0,
            1,
        ):
            raise InvalidCheckQubitCollectionError(
                f"Check block {block.block_id!r} "
                "has an invalid logical bit."
            )

        if (
            block.logical_index
            != expected_index
        ):
            raise InvalidCheckQubitCollectionError(
                f"Check block at collection index "
                f"{expected_index} declares logical index "
                f"{block.logical_index}."
            )

        expected_block_id = (
            f"{CHECK_BLOCK_PREFIX}"
            f"{expected_index:04d}"
        )

        if block.block_id != expected_block_id:
            raise InvalidCheckQubitCollectionError(
                f"Expected check block ID "
                f"{expected_block_id!r}, received "
                f"{block.block_id!r}."
            )

        if (
            require_unpositioned
            and block.position is not None
        ):
            raise InvalidCheckQubitCollectionError(
                f"Check block {block.block_id!r} "
                "already has an interleaved position."
            )


def generate_check_bit(
    rng: np.random.Generator,
) -> int:
    """
    Generate one independent logical check bit.
    """

    validate_rng(rng)

    return int(
        rng.integers(
            low=0,
            high=2,
        )
    )


def generate_check_basis(
    rng: np.random.Generator,
) -> str:
    """
    Generate one independent Z- or X-basis choice.
    """

    validate_rng(rng)

    return str(
        rng.choice(
            SUPPORTED_CHECK_BASES
        )
    )


def generate_one_check_qubit(
    logical_index: int,
    rng: np.random.Generator,
) -> LogicalQubit:
    """
    Generate one independent logical check block.
    """

    validate_rng(rng)

    if isinstance(
        logical_index,
        bool,
    ) or not isinstance(
        logical_index,
        int,
    ):
        raise TypeError(
            "logical_index must be an integer."
        )

    if logical_index < 0:
        raise ValueError(
            "logical_index cannot be negative."
        )

    logical_bit = generate_check_bit(
        rng
    )

    basis = generate_check_basis(
        rng
    )

    return create_check_logical_qubit(
        logical_bit=logical_bit,
        basis=basis,
        logical_index=logical_index,
        position=None,
    )


def generate_check_blocks(
    check_count: int = CHECK_LOGICAL_BLOCK_COUNT,
    rng: np.random.Generator | None = None,
    require_standard_count: bool = True,
) -> list[LogicalQubit]:
    """
    Generate independent logical FT-QuPAP check blocks.

    Notebook-aligned operation:

        for index in range(check_count):
            logical_bit = rng.integers(0, 2)
            basis = rng.choice(["Z", "X"])

    The supplied generator controls reproducibility. When no generator
    is supplied, a fresh NumPy generator is created.
    """

    normalized_count = validate_check_count(
        check_count,
        require_standard_count=require_standard_count,
    )

    if rng is None:
        rng = np.random.default_rng()

    validate_rng(rng)

    check_blocks = [
        generate_one_check_qubit(
            logical_index=index,
            rng=rng,
        )
        for index in range(
            normalized_count
        )
    ]

    validate_check_blocks(
        check_blocks,
        expected_count=normalized_count,
        require_unpositioned=True,
    )

    return check_blocks


def generate_check_qubits(
    check_count: int = CHECK_LOGICAL_BLOCK_COUNT,
    rng: np.random.Generator | None = None,
    require_standard_count: bool = True,
) -> list[LogicalQubit]:
    """
    Compatibility alias for generate_check_blocks().
    """

    return generate_check_blocks(
        check_count=check_count,
        rng=rng,
        require_standard_count=require_standard_count,
    )


def generate_check_specs(
    check_count: int = CHECK_LOGICAL_BLOCK_COUNT,
    rng: np.random.Generator | None = None,
    require_standard_count: bool = True,
) -> list[LogicalQubit]:
    """
    Notebook-compatible check-specification function.
    """

    return generate_check_blocks(
        check_count=check_count,
        rng=rng,
        require_standard_count=require_standard_count,
    )


def generate_check_qubit_result(
    check_count: int = CHECK_LOGICAL_BLOCK_COUNT,
    rng: np.random.Generator | None = None,
    require_standard_count: bool = True,
) -> CheckQubitGenerationResult:
    """
    Generate check blocks and return a detailed result object.
    """

    check_blocks = generate_check_blocks(
        check_count=check_count,
        rng=rng,
        require_standard_count=require_standard_count,
    )

    basis_counter = Counter(
        block.basis
        for block in check_blocks
    )

    bit_counter = Counter(
        block.logical_bit
        for block in check_blocks
    )

    return CheckQubitGenerationResult(
        check_blocks=tuple(
            check_blocks
        ),
        check_count=len(
            check_blocks
        ),
        z_basis_count=int(
            basis_counter.get(
                "Z",
                0,
            )
        ),
        x_basis_count=int(
            basis_counter.get(
                "X",
                0,
            )
        ),
        zero_bit_count=int(
            bit_counter.get(
                0,
                0,
            )
        ),
        one_bit_count=int(
            bit_counter.get(
                1,
                0,
            )
        ),
    )


def copy_check_blocks(
    check_blocks: Sequence[LogicalQubit],
) -> list[LogicalQubit]:
    """
    Return independent copies of logical check blocks.
    """

    validate_check_blocks(
        check_blocks,
        expected_count=len(
            check_blocks
        ),
        require_unpositioned=False,
    )

    return [
        copy.deepcopy(block)
        for block in check_blocks
    ]


def check_block_lookup(
    check_blocks: Sequence[LogicalQubit],
) -> dict[str, LogicalQubit]:
    """
    Build a lookup table keyed by check-block ID.
    """

    validate_check_blocks(
        check_blocks,
        expected_count=len(
            check_blocks
        ),
        require_unpositioned=False,
    )

    return {
        block.block_id: block
        for block in check_blocks
    }


def check_index_lookup(
    check_blocks: Sequence[LogicalQubit],
) -> dict[int, LogicalQubit]:
    """
    Build a lookup table keyed by logical check index.
    """

    validate_check_blocks(
        check_blocks,
        expected_count=len(
            check_blocks
        ),
        require_unpositioned=False,
    )

    return {
        block.logical_index: block
        for block in check_blocks
    }


def run_self_test() -> None:
    """
    Verify deterministic seeded generation and protocol structure.
    """

    print("=" * 72)
    print("FT-QuPAP Check-Qubit Generator Self-Test")
    print("=" * 72)

    first_result = (
        generate_check_qubit_result(
            check_count=(
                CHECK_LOGICAL_BLOCK_COUNT
            ),
            rng=np.random.default_rng(
                20260701
            ),
            require_standard_count=True,
        )
    )

    second_result = (
        generate_check_qubit_result(
            check_count=(
                CHECK_LOGICAL_BLOCK_COUNT
            ),
            rng=np.random.default_rng(
                20260701
            ),
            require_standard_count=True,
        )
    )

    first_blocks = list(
        first_result.check_blocks
    )

    second_blocks = list(
        second_result.check_blocks
    )

    reproducible = (
        [
            block.to_dictionary()
            for block in first_blocks
        ]
        == [
            block.to_dictionary()
            for block in second_blocks
        ]
    )

    all_check_role = all(
        block.role == CHECK_ROLE
        for block in first_blocks
    )

    all_supported_bases = all(
        block.basis
        in SUPPORTED_CHECK_BASES
        for block in first_blocks
    )

    all_valid_bits = all(
        block.logical_bit in (
            0,
            1,
        )
        for block in first_blocks
    )

    all_unpositioned = all(
        block.position is None
        for block in first_blocks
    )

    sequential_ids = all(
        block.block_id
        == f"C{index:04d}"
        for index, block in enumerate(
            first_blocks
        )
    )

    sequential_indices = all(
        block.logical_index == index
        for index, block in enumerate(
            first_blocks
        )
    )

    first_five = [
        {
            "block_id":
                block.block_id,
            "logical_bit":
                block.logical_bit,
            "basis":
                block.basis,
        }
        for block in first_blocks[:5]
    ]

    print(
        "Check blocks generated    : "
        f"{first_result.check_count}"
    )

    print(
        "Z-basis check blocks      : "
        f"{first_result.z_basis_count}"
    )

    print(
        "X-basis check blocks      : "
        f"{first_result.x_basis_count}"
    )

    print(
        "Logical-zero check blocks : "
        f"{first_result.zero_bit_count}"
    )

    print(
        "Logical-one check blocks  : "
        f"{first_result.one_bit_count}"
    )

    print(
        "All roles are check       : "
        f"{all_check_role}"
    )

    print(
        "All bases valid           : "
        f"{all_supported_bases}"
    )

    print(
        "All logical bits valid    : "
        f"{all_valid_bits}"
    )

    print(
        "All blocks unpositioned   : "
        f"{all_unpositioned}"
    )

    print(
        "Block IDs sequential      : "
        f"{sequential_ids}"
    )

    print(
        "Logical indices sequential: "
        f"{sequential_indices}"
    )

    print(
        "Fixed-seed reproducible   : "
        f"{reproducible}"
    )

    print(
        "First five check blocks   : "
        f"{first_five}"
    )

    if (
        first_result.check_count
        != CHECK_LOGICAL_BLOCK_COUNT
    ):
        raise CheckQubitGenerationError(
            "Incorrect check-block count."
        )

    if not all_check_role:
        raise CheckQubitGenerationError(
            "A generated block has the wrong role."
        )

    if not all_supported_bases:
        raise CheckQubitGenerationError(
            "A generated check block has an "
            "unsupported basis."
        )

    if not all_valid_bits:
        raise CheckQubitGenerationError(
            "A generated check block has an "
            "invalid logical bit."
        )

    if not all_unpositioned:
        raise CheckQubitGenerationError(
            "A generated check block already has "
            "a frame position."
        )

    if not sequential_ids:
        raise CheckQubitGenerationError(
            "Check-block IDs are not sequential."
        )

    if not sequential_indices:
        raise CheckQubitGenerationError(
            "Check logical indices are not sequential."
        )

    if not reproducible:
        raise CheckQubitGenerationError(
            "Identical seeds generated different "
            "check schedules."
        )

    print(
        "\nCheck-qubit generator self-test "
        "completed successfully."
    )


__all__ = [
    "CHECK_LOGICAL_BLOCK_COUNT",
    "CHECK_BLOCK_PREFIX",
    "SUPPORTED_CHECK_BASES",
    "CheckQubitGenerationError",
    "InvalidCheckCountError",
    "InvalidCheckQubitCollectionError",
    "CheckQubitGenerationResult",
    "validate_check_count",
    "validate_rng",
    "validate_check_blocks",
    "generate_check_bit",
    "generate_check_basis",
    "generate_one_check_qubit",
    "generate_check_blocks",
    "generate_check_qubits",
    "generate_check_specs",
    "generate_check_qubit_result",
    "copy_check_blocks",
    "check_block_lookup",
    "check_index_lookup",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        CheckQubitGenerationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[CHECK-QUBIT GENERATION ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error