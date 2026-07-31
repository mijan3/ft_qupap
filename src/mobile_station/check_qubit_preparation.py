"""
Check-Qubit Preparation Module
FT-QuPAP Mobile Station

This module generates independent logical check states for the
FT-QuPAP authentication protocol.

For each check block, the Mobile Station randomly selects:

    logical bit : 0 or 1
    basis       : Z or X

Standard FT-QuPAP configuration:

    Payload logical blocks : 128
    Check logical blocks   : 32
    Total logical blocks   : 160

Check blocks are later:

1. Randomly interleaved with KMAC payload blocks.
2. Encoded using the Steane [[7,1,3]] CSS code.
3. Described inside the encrypted K_ctrl control schedule.
4. Measured by the Authentication Server in the declared basis.
5. Used to calculate raw QBER before payload acceptance.

The check states must be generated independently of the KMAC tag.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .control_schedule import LogicalSpec
except ImportError:
    from control_schedule import LogicalSpec


STANDARD_CHECK_COUNT = 32
CHECK_BLOCK_PREFIX = "C"

SUPPORTED_BASES = (
    "Z",
    "X",
)

DEFAULT_RANDOM_SEED = 20260701


class CheckQubitPreparationError(Exception):
    """Raised when FT-QuPAP check preparation fails."""


@dataclass(frozen=True)
class CheckPreparationSummary:
    """
    Non-secret summary of prepared check states.
    """

    check_count: int
    z_basis_count: int
    x_basis_count: int
    logical_zero_count: int
    logical_one_count: int
    first_block_id: str
    last_block_id: str

    def to_dictionary(self) -> dict[str, Any]:
        """Return the summary as a dictionary."""

        return {
            "check_count": self.check_count,
            "z_basis_count": self.z_basis_count,
            "x_basis_count": self.x_basis_count,
            "logical_zero_count":
                self.logical_zero_count,
            "logical_one_count":
                self.logical_one_count,
            "first_block_id":
                self.first_block_id,
            "last_block_id":
                self.last_block_id,
        }


def validate_check_count(
    check_count: int,
    require_standard_count: bool = True,
) -> None:
    """
    Validate the requested number of check blocks.

    Args:
        check_count:
            Number of logical check blocks.

        require_standard_count:
            When True, require the standard FT-QuPAP value of 32.
    """

    if isinstance(check_count, bool):
        raise TypeError(
            "check_count must be an integer."
        )

    if not isinstance(check_count, int):
        raise TypeError(
            "check_count must be an integer."
        )

    if check_count <= 0:
        raise ValueError(
            "check_count must be greater than zero."
        )

    if (
        require_standard_count
        and check_count != STANDARD_CHECK_COUNT
    ):
        raise CheckQubitPreparationError(
            "The standard FT-QuPAP configuration requires "
            f"exactly {STANDARD_CHECK_COUNT} check blocks. "
            f"Received {check_count}."
        )


def create_random_generator(
    seed: int | None = None,
) -> np.random.Generator:
    """
    Create a NumPy random generator.

    Supplying a seed enables reproducible simulation runs.
    Without a seed, NumPy obtains fresh system entropy.
    """

    if seed is not None:
        if isinstance(seed, bool):
            raise TypeError(
                "seed must be an integer or None."
            )

        if not isinstance(seed, int):
            raise TypeError(
                "seed must be an integer or None."
            )

        if seed < 0:
            raise ValueError(
                "seed cannot be negative."
            )

    return np.random.default_rng(seed)


def generate_random_logical_bit(
    rng: np.random.Generator,
) -> int:
    """
    Generate one random logical check bit.

    Returns:
        Either 0 or 1.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    logical_bit = int(
        rng.integers(0, 2)
    )

    if logical_bit not in (0, 1):
        raise CheckQubitPreparationError(
            "Random generator produced an invalid "
            "logical bit."
        )

    return logical_bit


def generate_random_basis(
    rng: np.random.Generator,
) -> str:
    """
    Randomly select the Z or X preparation basis.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    basis = str(
        rng.choice(
            SUPPORTED_BASES
        )
    )

    if basis not in SUPPORTED_BASES:
        raise CheckQubitPreparationError(
            "Random generator produced an invalid basis."
        )

    return basis


def generate_check_spec(
    logical_index: int,
    rng: np.random.Generator,
) -> LogicalSpec:
    """
    Generate one independent logical check specification.

    Args:
        logical_index:
            Check-state index.

        rng:
            Random generator used for bit and basis selection.

    Returns:
        LogicalSpec describing one check state.
    """

    if isinstance(logical_index, bool):
        raise TypeError(
            "logical_index must be an integer."
        )

    if not isinstance(logical_index, int):
        raise TypeError(
            "logical_index must be an integer."
        )

    if logical_index < 0:
        raise ValueError(
            "logical_index cannot be negative."
        )

    logical_bit = generate_random_logical_bit(
        rng
    )

    basis = generate_random_basis(
        rng
    )

    return LogicalSpec(
        block_id=(
            f"{CHECK_BLOCK_PREFIX}"
            f"{logical_index:04d}"
        ),
        role="check",
        logical_index=logical_index,
        logical_bit=logical_bit,
        basis=basis,
    )


def generate_check_specs(
    check_count: int = STANDARD_CHECK_COUNT,
    rng: np.random.Generator | None = None,
    require_standard_count: bool = True,
) -> list[LogicalSpec]:
    """
    Generate independent logical FT-QuPAP check states.

    This function matches the notebook interface used by
    mobile_station.py:

        check_specs = generate_check_specs(
            check_count=32,
            rng=rng,
        )

    Args:
        check_count:
            Number of logical check blocks.

        rng:
            Optional NumPy random generator.

        require_standard_count:
            Require exactly 32 blocks when True.

    Returns:
        Ordered list of logical check specifications.
    """

    validate_check_count(
        check_count=check_count,
        require_standard_count=
            require_standard_count,
    )

    if rng is None:
        rng = create_random_generator()

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    check_specs = [
        generate_check_spec(
            logical_index=index,
            rng=rng,
        )
        for index in range(check_count)
    ]

    validate_check_specs(
        check_specs=check_specs,
        expected_count=check_count,
    )

    return check_specs


def validate_check_specs(
    check_specs: Sequence[Any],
    expected_count: int = STANDARD_CHECK_COUNT,
) -> None:
    """
    Validate a complete logical check-state collection.
    """

    if isinstance(
        check_specs,
        (str, bytes, bytearray),
    ):
        raise TypeError(
            "check_specs must be a sequence of "
            "logical specifications."
        )

    if not isinstance(check_specs, Sequence):
        raise TypeError(
            "check_specs must be a sequence."
        )

    if isinstance(expected_count, bool):
        raise TypeError(
            "expected_count must be an integer."
        )

    if not isinstance(expected_count, int):
        raise TypeError(
            "expected_count must be an integer."
        )

    if expected_count <= 0:
        raise ValueError(
            "expected_count must be positive."
        )

    if len(check_specs) != expected_count:
        raise CheckQubitPreparationError(
            f"Expected {expected_count} check blocks, "
            f"received {len(check_specs)}."
        )

    seen_block_ids: set[str] = set()
    seen_indices: set[int] = set()

    for expected_index, spec in enumerate(
        check_specs
    ):
        required_attributes = (
            "block_id",
            "role",
            "logical_index",
            "logical_bit",
            "basis",
            "position",
        )

        for attribute in required_attributes:
            if not hasattr(spec, attribute):
                raise TypeError(
                    "Check specification is missing "
                    f"{attribute!r}."
                )

        expected_block_id = (
            f"{CHECK_BLOCK_PREFIX}"
            f"{expected_index:04d}"
        )

        if spec.block_id != expected_block_id:
            raise CheckQubitPreparationError(
                f"Expected block ID "
                f"{expected_block_id!r}, "
                f"received {spec.block_id!r}."
            )

        if spec.role != "check":
            raise CheckQubitPreparationError(
                "Every check specification must use "
                "role='check'."
            )

        if spec.logical_index != expected_index:
            raise CheckQubitPreparationError(
                "Check logical indices must be continuous "
                f"from 0 to {expected_count - 1}."
            )

        if spec.logical_bit not in (0, 1):
            raise CheckQubitPreparationError(
                "Check logical bits must be 0 or 1."
            )

        if spec.basis not in SUPPORTED_BASES:
            raise CheckQubitPreparationError(
                "Check preparation basis must be Z or X."
            )

        if spec.position is not None:
            raise CheckQubitPreparationError(
                "A newly generated check block must not "
                "have a frame position before interleaving."
            )

        if spec.block_id in seen_block_ids:
            raise CheckQubitPreparationError(
                f"Duplicate check block ID: "
                f"{spec.block_id!r}."
            )

        if spec.logical_index in seen_indices:
            raise CheckQubitPreparationError(
                "Duplicate check logical index: "
                f"{spec.logical_index}."
            )

        seen_block_ids.add(
            spec.block_id
        )

        seen_indices.add(
            spec.logical_index
        )


def summarize_check_specs(
    check_specs: Sequence[Any],
) -> CheckPreparationSummary:
    """
    Build a non-secret summary of generated check states.

    The complete check pattern should not be logged during an active
    protocol session because the encrypted schedule protects this
    information from an eavesdropper.
    """

    validate_check_specs(
        check_specs=check_specs,
        expected_count=len(check_specs),
    )

    z_basis_count = sum(
        int(spec.basis == "Z")
        for spec in check_specs
    )

    x_basis_count = sum(
        int(spec.basis == "X")
        for spec in check_specs
    )

    logical_zero_count = sum(
        int(spec.logical_bit == 0)
        for spec in check_specs
    )

    logical_one_count = sum(
        int(spec.logical_bit == 1)
        for spec in check_specs
    )

    return CheckPreparationSummary(
        check_count=len(check_specs),
        z_basis_count=z_basis_count,
        x_basis_count=x_basis_count,
        logical_zero_count=
            logical_zero_count,
        logical_one_count=
            logical_one_count,
        first_block_id=
            check_specs[0].block_id,
        last_block_id=
            check_specs[-1].block_id,
    )


def verify_reproducible_generation(
    check_count: int,
    seed: int,
) -> bool:
    """
    Verify that a fixed simulation seed reproduces the same checks.
    """

    first_specs = generate_check_specs(
        check_count=check_count,
        rng=create_random_generator(seed),
        require_standard_count=(
            check_count == STANDARD_CHECK_COUNT
        ),
    )

    second_specs = generate_check_specs(
        check_count=check_count,
        rng=create_random_generator(seed),
        require_standard_count=(
            check_count == STANDARD_CHECK_COUNT
        ),
    )

    first_values = [
        (
            spec.block_id,
            spec.logical_index,
            spec.logical_bit,
            spec.basis,
        )
        for spec in first_specs
    ]

    second_values = [
        (
            spec.block_id,
            spec.logical_index,
            spec.logical_bit,
            spec.basis,
        )
        for spec in second_specs
    ]

    return first_values == second_values


def run_self_test() -> None:
    """
    Test standard FT-QuPAP check-state preparation.
    """

    print("=" * 70)
    print("FT-QuPAP Check-Qubit Preparation Self-Test")
    print("=" * 70)

    first_rng = create_random_generator(
        DEFAULT_RANDOM_SEED
    )

    check_specs = generate_check_specs(
        check_count=STANDARD_CHECK_COUNT,
        rng=first_rng,
        require_standard_count=True,
    )

    summary = summarize_check_specs(
        check_specs
    )

    reproducible = (
        verify_reproducible_generation(
            check_count=STANDARD_CHECK_COUNT,
            seed=DEFAULT_RANDOM_SEED,
        )
    )

    all_ids_unique = (
        len(
            {
                spec.block_id
                for spec in check_specs
            }
        )
        == len(check_specs)
    )

    all_bits_valid = all(
        spec.logical_bit in (0, 1)
        for spec in check_specs
    )

    all_bases_valid = all(
        spec.basis in SUPPORTED_BASES
        for spec in check_specs
    )

    all_positions_empty = all(
        spec.position is None
        for spec in check_specs
    )

    print(
        f"Configured check blocks    : "
        f"{STANDARD_CHECK_COUNT}"
    )
    print(
        f"Generated check blocks     : "
        f"{len(check_specs)}"
    )
    print(
        f"First check block          : "
        f"{check_specs[0]}"
    )
    print(
        f"Last check block           : "
        f"{check_specs[-1]}"
    )
    print(
        f"Z-basis blocks             : "
        f"{summary.z_basis_count}"
    )
    print(
        f"X-basis blocks             : "
        f"{summary.x_basis_count}"
    )
    print(
        f"Logical-zero blocks        : "
        f"{summary.logical_zero_count}"
    )
    print(
        f"Logical-one blocks         : "
        f"{summary.logical_one_count}"
    )
    print(
        f"Unique block IDs           : "
        f"{all_ids_unique}"
    )
    print(
        f"All logical bits valid     : "
        f"{all_bits_valid}"
    )
    print(
        f"All bases valid            : "
        f"{all_bases_valid}"
    )
    print(
        f"Positions unset initially  : "
        f"{all_positions_empty}"
    )
    print(
        f"Fixed-seed reproducibility : "
        f"{reproducible}"
    )

    if len(check_specs) != STANDARD_CHECK_COUNT:
        raise CheckQubitPreparationError(
            "Incorrect number of check blocks."
        )

    if not all_ids_unique:
        raise CheckQubitPreparationError(
            "Check block IDs are not unique."
        )

    if not all_bits_valid:
        raise CheckQubitPreparationError(
            "An invalid logical bit was generated."
        )

    if not all_bases_valid:
        raise CheckQubitPreparationError(
            "An invalid preparation basis was generated."
        )

    if not all_positions_empty:
        raise CheckQubitPreparationError(
            "Check positions were assigned before "
            "schedule interleaving."
        )

    if not reproducible:
        raise CheckQubitPreparationError(
            "Fixed-seed check generation is not reproducible."
        )

    print(
        "\nCheck-qubit preparation self-test "
        "completed successfully."
    )


__all__ = [
    "STANDARD_CHECK_COUNT",
    "CheckPreparationSummary",
    "CheckQubitPreparationError",
    "create_random_generator",
    "generate_random_logical_bit",
    "generate_random_basis",
    "generate_check_spec",
    "generate_check_specs",
    "validate_check_specs",
    "summarize_check_specs",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        CheckQubitPreparationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"\n[CHECK-QUBIT PREPARATION ERROR] "
            f"{error}"
        )