"""
Steane [[7,1,3]] syndrome processing for FT-QuPAP v5.1.

Each logical authentication bit is encoded into seven physical qubits
using the Steane CSS code. After measurement, the Authentication Server
processes each seven-bit block to:

1. Detect a single physical-bit error.
2. Correct the detected error.
3. Recover up to two missing measurement results as erasures.
4. Decode the corrected logical bit.
5. Produce syndrome statistics for the GP attack detector.

The Steane code is derived from the classical [7,4,3] Hamming code.

Parity-check matrix:

    H = [
        1 0 1 0 1 0 1
        0 1 1 0 0 1 1
        0 0 0 1 1 1 1
    ]

The three-bit syndrome identifies one erroneous physical position.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from src.common.constants import (
    BASIS_X,
    BASIS_Z,
    STEANE_PHYSICAL_QUBITS_PER_LOGICAL,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.validators import (
    validate_integer,
    validate_non_empty_string,
    validate_probability,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

STEANE_BLOCK_SIZE = (
    STEANE_PHYSICAL_QUBITS_PER_LOGICAL
)

STEANE_MAXIMUM_CORRECTABLE_ERRORS = 1

STEANE_MAXIMUM_CORRECTABLE_ERASURES = 2


# Each column represents one physical-qubit position.
#
# Column values:
#
# position 0 -> 001
# position 1 -> 010
# position 2 -> 011
# position 3 -> 100
# position 4 -> 101
# position 5 -> 110
# position 6 -> 111

STEANE_PARITY_CHECK_MATRIX: tuple[
    tuple[int, ...],
    ...,
] = (
    (
        1, 0, 1, 0, 1, 0, 1,
    ),
    (
        0, 1, 1, 0, 0, 1, 1,
    ),
    (
        0, 0, 0, 1, 1, 1, 1,
    ),
)


REASON_NO_ERROR = "no_error"

REASON_SINGLE_ERROR_CORRECTED = (
    "single_error_corrected"
)

REASON_ERASURE_RECOVERED = (
    "erasure_recovered"
)

REASON_TOO_MANY_ERASURES = (
    "too_many_erasures"
)

REASON_AMBIGUOUS_ERASURE = (
    "ambiguous_erasure_recovery"
)

REASON_ERASURE_RECOVERY_FAILED = (
    "erasure_recovery_failed"
)

REASON_RESIDUAL_SYNDROME = (
    "residual_syndrome"
)


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class SyndromeProcessingError(RuntimeError):
    """Raised when a Steane block cannot be processed safely."""

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
# Block result
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SyndromeBlockResult:
    """
    Result of processing one seven-qubit Steane block.

    `received_bits` may contain None values representing lost physical
    measurements.

    `corrected_bits` is None when the block cannot be decoded.
    """

    position: int
    basis: str

    received_bits: tuple[
        int | None,
        ...,
    ]

    corrected_bits: tuple[int, ...] | None

    syndrome: tuple[int, int, int] | None
    residual_syndrome: tuple[int, int, int] | None

    syndrome_value: int
    syndrome_weight: int

    logical_bit: int | None

    error_position: int | None

    erasure_count: int
    recovered_erasure_count: int

    correction_applied: bool
    correction_success: bool
    erasure_recovery_used: bool

    uncorrectable: bool
    reason: str

    def __post_init__(self) -> None:
        validate_integer(
            self.position,
            field_name="position",
            minimum=0,
        )

        normalize_measurement_basis(
            self.basis
        )

        if (
            len(self.received_bits)
            != STEANE_BLOCK_SIZE
        ):
            raise ProtocolValidationError(
                (
                    "received_bits must contain exactly "
                    f"{STEANE_BLOCK_SIZE} values."
                )
            )

        for index, bit in enumerate(
            self.received_bits
        ):
            normalize_physical_bit(
                bit,
                field_name=(
                    f"received_bits[{index}]"
                ),
                allow_erasure=True,
            )

        if self.corrected_bits is not None:
            if (
                len(self.corrected_bits)
                != STEANE_BLOCK_SIZE
            ):
                raise ProtocolValidationError(
                    (
                        "corrected_bits must contain "
                        f"{STEANE_BLOCK_SIZE} values."
                    )
                )

            for index, bit in enumerate(
                self.corrected_bits
            ):
                normalize_physical_bit(
                    bit,
                    field_name=(
                        f"corrected_bits[{index}]"
                    ),
                    allow_erasure=False,
                )

        for field_name in (
            "syndrome",
            "residual_syndrome",
        ):
            syndrome = getattr(
                self,
                field_name,
            )

            if syndrome is not None:
                normalize_syndrome(
                    syndrome,
                    field_name=field_name,
                )

        validate_integer(
            self.syndrome_value,
            field_name="syndrome_value",
            minimum=0,
            maximum=7,
        )

        validate_integer(
            self.syndrome_weight,
            field_name="syndrome_weight",
            minimum=0,
            maximum=3,
        )

        if self.logical_bit is not None:
            normalize_physical_bit(
                self.logical_bit,
                field_name="logical_bit",
                allow_erasure=False,
            )

        if self.error_position is not None:
            validate_integer(
                self.error_position,
                field_name="error_position",
                minimum=0,
                maximum=(
                    STEANE_BLOCK_SIZE - 1
                ),
            )

        validate_integer(
            self.erasure_count,
            field_name="erasure_count",
            minimum=0,
            maximum=STEANE_BLOCK_SIZE,
        )

        validate_integer(
            self.recovered_erasure_count,
            field_name=(
                "recovered_erasure_count"
            ),
            minimum=0,
            maximum=STEANE_BLOCK_SIZE,
        )

        for field_name in (
            "correction_applied",
            "correction_success",
            "erasure_recovery_used",
            "uncorrectable",
        ):
            if not isinstance(
                getattr(
                    self,
                    field_name,
                ),
                bool,
            ):
                raise ProtocolValidationError(
                    f"{field_name} must be Boolean."
                )

        validate_non_empty_string(
            self.reason,
            field_name="reason",
            minimum_length=1,
            maximum_length=256,
        )

        if self.correction_success:
            if self.uncorrectable:
                raise ProtocolValidationError(
                    (
                        "A successfully corrected block "
                        "cannot be uncorrectable."
                    )
                )

            if self.corrected_bits is None:
                raise ProtocolValidationError(
                    (
                        "A successfully processed block "
                        "must contain corrected_bits."
                    )
                )

            if self.logical_bit is None:
                raise ProtocolValidationError(
                    (
                        "A successfully processed block "
                        "must contain a logical_bit."
                    )
                )

        if self.uncorrectable:
            if self.correction_success:
                raise ProtocolValidationError(
                    (
                        "An uncorrectable block cannot "
                        "report correction success."
                    )
                )

            if self.logical_bit is not None:
                raise ProtocolValidationError(
                    (
                        "An uncorrectable block cannot "
                        "contain a logical bit."
                    )
                )

    @property
    def decoded(self) -> bool:
        """Return True when a logical bit was recovered."""

        return (
            self.correction_success
            and self.logical_bit is not None
        )

    @property
    def valid(self) -> bool:
        """Compatibility alias for successful decoding."""

        return self.decoded

    @property
    def corrected_logical_bit(self) -> int | None:
        """Compatibility alias used by the payload decoder."""

        return self.logical_bit

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result dictionary."""

        result = asdict(
            self
        )

        result["received_bits"] = list(
            self.received_bits
        )

        result["corrected_bits"] = (
            None
            if self.corrected_bits is None
            else list(
                self.corrected_bits
            )
        )

        result["syndrome"] = (
            None
            if self.syndrome is None
            else list(
                self.syndrome
            )
        )

        result["residual_syndrome"] = (
            None
            if self.residual_syndrome is None
            else list(
                self.residual_syndrome
            )
        )

        result["decoded"] = self.decoded

        return result


# ---------------------------------------------------------------------
# Batch summary
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SyndromeProcessingSummary:
    """
    Aggregate syndrome statistics for multiple logical blocks.

    The GP feature extractor uses:

    - mean_syndrome_weight
    - max_syndrome_weight
    - correction_failure_rate
    """

    total_blocks: int
    decoded_blocks: int

    corrected_blocks: int
    erasure_recovered_blocks: int
    failed_blocks: int

    total_erasures: int

    mean_syndrome_weight: float
    max_syndrome_weight: int

    correction_failure_rate: float

    results: tuple[
        SyndromeBlockResult,
        ...,
    ]

    def __post_init__(self) -> None:
        integer_fields = (
            "total_blocks",
            "decoded_blocks",
            "corrected_blocks",
            "erasure_recovered_blocks",
            "failed_blocks",
            "total_erasures",
        )

        for field_name in integer_fields:
            validate_integer(
                getattr(
                    self,
                    field_name,
                ),
                field_name=field_name,
                minimum=0,
            )

        validate_integer(
            self.max_syndrome_weight,
            field_name="max_syndrome_weight",
            minimum=0,
            maximum=3,
        )

        validate_probability(
            self.correction_failure_rate,
            field_name=(
                "correction_failure_rate"
            ),
        )

        if not isinstance(
            self.mean_syndrome_weight,
            (int, float),
        ):
            raise ProtocolValidationError(
                (
                    "mean_syndrome_weight must "
                    "be numeric."
                )
            )

        if (
            self.mean_syndrome_weight < 0
            or self.mean_syndrome_weight > 3
        ):
            raise ProtocolValidationError(
                (
                    "mean_syndrome_weight must "
                    "be between 0 and 3."
                )
            )

        if (
            len(self.results)
            != self.total_blocks
        ):
            raise ProtocolValidationError(
                (
                    "total_blocks does not match "
                    "the number of results."
                )
            )

        if (
            self.decoded_blocks
            + self.failed_blocks
            != self.total_blocks
        ):
            raise ProtocolValidationError(
                (
                    "Decoded and failed block counts "
                    "do not equal total_blocks."
                )
            )

    def to_feature_dict(self) -> dict[str, float]:
        """
        Return the syndrome-related GP feature values.
        """

        return {
            "mean_syndrome_weight": float(
                self.mean_syndrome_weight
            ),
            "max_syndrome_weight": float(
                self.max_syndrome_weight
            ),
            "correction_failure_rate": float(
                self.correction_failure_rate
            ),
        }

    def to_dict(
        self,
        *,
        include_results: bool = True,
    ) -> dict[str, Any]:
        """Return a JSON-compatible summary."""

        if not isinstance(
            include_results,
            bool,
        ):
            raise ProtocolValidationError(
                "include_results must be Boolean."
            )

        result: dict[str, Any] = {
            "total_blocks": self.total_blocks,
            "decoded_blocks": (
                self.decoded_blocks
            ),
            "corrected_blocks": (
                self.corrected_blocks
            ),
            "erasure_recovered_blocks": (
                self.erasure_recovered_blocks
            ),
            "failed_blocks": (
                self.failed_blocks
            ),
            "total_erasures": (
                self.total_erasures
            ),
            "mean_syndrome_weight": (
                self.mean_syndrome_weight
            ),
            "max_syndrome_weight": (
                self.max_syndrome_weight
            ),
            "correction_failure_rate": (
                self.correction_failure_rate
            ),
        }

        if include_results:
            result["results"] = [
                block_result.to_dict()
                for block_result in self.results
            ]

        return result


# ---------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------

def normalize_measurement_basis(
    basis: str,
) -> str:
    """
    Normalize a Steane block measurement basis.
    """

    validated = validate_non_empty_string(
        basis,
        field_name="basis",
        minimum_length=1,
        maximum_length=16,
    ).strip().upper()

    valid_bases = {
        str(BASIS_X).upper(): str(
            BASIS_X
        ).upper(),
        str(BASIS_Z).upper(): str(
            BASIS_Z
        ).upper(),
        "X": "X",
        "Z": "Z",
    }

    normalized = valid_bases.get(
        validated
    )

    if normalized is None:
        raise ProtocolValidationError(
            (
                "Steane measurement basis must "
                "be X or Z."
            ),
            details={
                "received_basis": basis,
            },
        )

    return normalized


def normalize_physical_bit(
    value: Any,
    *,
    field_name: str = "physical_bit",
    allow_erasure: bool = True,
) -> int | None:
    """
    Normalize one measured physical qubit.

    Accepted binary values:

    - 0
    - 1
    - False
    - True

    Erasure representations:

    - None
    - "lost"
    - "erasure"
    - "missing"
    """

    if value is None:
        if allow_erasure:
            return None

        raise SyndromeProcessingError(
            f"{field_name} cannot be missing."
        )

    if isinstance(
        value,
        bool,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        int,
    ) and value in (0, 1):
        return value

    if (
        allow_erasure
        and isinstance(
            value,
            str,
        )
        and value.strip().lower()
        in {
            "lost",
            "erasure",
            "missing",
            "none",
        }
    ):
        return None

    raise SyndromeProcessingError(
        (
            f"{field_name} must be 0, 1, "
            "or an erasure value."
        ),
        details={
            "field_name": field_name,
            "received_value": value,
            "received_type": type(
                value
            ).__name__,
        },
    )


def normalize_steane_block(
    measured_bits: Sequence[Any],
) -> tuple[
    int | None,
    ...,
]:
    """
    Validate one seven-value Steane measurement block.
    """

    if isinstance(
        measured_bits,
        (
            str,
            bytes,
            bytearray,
        ),
    ) or not isinstance(
        measured_bits,
        Sequence,
    ):
        raise SyndromeProcessingError(
            (
                "measured_bits must be a "
                "seven-value sequence."
            )
        )

    if (
        len(measured_bits)
        != STEANE_BLOCK_SIZE
    ):
        raise SyndromeProcessingError(
            (
                "A Steane logical block must contain "
                f"exactly {STEANE_BLOCK_SIZE} "
                "physical measurement values."
            ),
            details={
                "expected_length": (
                    STEANE_BLOCK_SIZE
                ),
                "actual_length": len(
                    measured_bits
                ),
            },
        )

    return tuple(
        normalize_physical_bit(
            value,
            field_name=(
                f"measured_bits[{index}]"
            ),
            allow_erasure=True,
        )
        for index, value in enumerate(
            measured_bits
        )
    )


def normalize_syndrome(
    syndrome: Sequence[Any],
    *,
    field_name: str = "syndrome",
) -> tuple[int, int, int]:
    """
    Normalize a three-bit Steane syndrome.
    """

    if (
        not isinstance(
            syndrome,
            Sequence,
        )
        or isinstance(
            syndrome,
            (
                str,
                bytes,
                bytearray,
            ),
        )
        or len(syndrome) != 3
    ):
        raise SyndromeProcessingError(
            (
                f"{field_name} must contain "
                "exactly three binary values."
            )
        )

    normalized = tuple(
        int(
            normalize_physical_bit(
                value,
                field_name=(
                    f"{field_name}[{index}]"
                ),
                allow_erasure=False,
            )
        )
        for index, value in enumerate(
            syndrome
        )
    )

    return (
        normalized[0],
        normalized[1],
        normalized[2],
    )


# ---------------------------------------------------------------------
# Syndrome mathematics
# ---------------------------------------------------------------------

def calculate_steane_syndrome(
    bits: Sequence[Any],
) -> tuple[int, int, int]:
    """
    Calculate the Steane syndrome:

        s = H × rᵀ mod 2

    The input must contain seven non-erased physical bits.
    """

    normalized = normalize_steane_block(
        bits
    )

    if any(
        bit is None
        for bit in normalized
    ):
        raise SyndromeProcessingError(
            (
                "A syndrome cannot be calculated "
                "while erasures remain."
            )
        )

    concrete_bits = tuple(
        int(bit)
        for bit in normalized
        if bit is not None
    )

    syndrome_values: list[int] = []

    for row in STEANE_PARITY_CHECK_MATRIX:
        parity = sum(
            matrix_value * bit
            for matrix_value, bit
            in zip(
                row,
                concrete_bits,
                strict=True,
            )
        ) % 2

        syndrome_values.append(
            parity
        )

    return (
        syndrome_values[0],
        syndrome_values[1],
        syndrome_values[2],
    )


def syndrome_to_value(
    syndrome: Sequence[Any],
) -> int:
    """
    Convert a syndrome into its physical-position value.

    The syndrome rows have binary weights 1, 2, and 4:

        value = s0 + 2*s1 + 4*s2

    A value from 1 to 7 identifies the erroneous physical qubit.
    """

    normalized = normalize_syndrome(
        syndrome
    )

    return (
        normalized[0]
        + 2 * normalized[1]
        + 4 * normalized[2]
    )


def syndrome_to_error_position(
    syndrome: Sequence[Any],
) -> int | None:
    """
    Convert a syndrome into a zero-based physical-qubit position.
    """

    syndrome_value = syndrome_to_value(
        syndrome
    )

    if syndrome_value == 0:
        return None

    return (
        syndrome_value - 1
    )


def syndrome_weight(
    syndrome: Sequence[Any],
) -> int:
    """
    Return the Hamming weight of a three-bit syndrome.
    """

    normalized = normalize_syndrome(
        syndrome
    )

    return sum(
        normalized
    )


def flip_physical_bit(
    bits: Sequence[Any],
    position: int,
) -> tuple[int, ...]:
    """
    Flip one physical bit at a zero-based position.
    """

    normalized = normalize_steane_block(
        bits
    )

    if any(
        bit is None
        for bit in normalized
    ):
        raise SyndromeProcessingError(
            (
                "A physical bit cannot be corrected "
                "while erasures remain."
            )
        )

    validated_position = validate_integer(
        position,
        field_name="position",
        minimum=0,
        maximum=(
            STEANE_BLOCK_SIZE - 1
        ),
    )

    corrected = [
        int(bit)
        for bit in normalized
        if bit is not None
    ]

    corrected[
        validated_position
    ] ^= 1

    return tuple(
        corrected
    )


def decode_steane_logical_bit(
    corrected_bits: Sequence[Any],
) -> int:
    """
    Decode the Steane logical bit using transversal parity.

    For valid Steane computational or Hadamard-basis codewords:

        logical_bit = b0 XOR b1 XOR ... XOR b6
    """

    normalized = normalize_steane_block(
        corrected_bits
    )

    if any(
        bit is None
        for bit in normalized
    ):
        raise SyndromeProcessingError(
            (
                "The logical bit cannot be decoded "
                "while erasures remain."
            )
        )

    return (
        sum(
            int(bit)
            for bit in normalized
            if bit is not None
        )
        % 2
    )


# ---------------------------------------------------------------------
# Erasure recovery
# ---------------------------------------------------------------------

def recover_steane_erasures(
    measured_bits: Sequence[Any],
) -> tuple[int, ...]:
    """
    Recover up to two missing physical measurements.

    Every possible binary assignment is tested. A candidate is accepted
    only when its Steane syndrome is zero.

    The [7,4,3] Hamming code uniquely recovers up to two erasures when
    the remaining measurements are internally consistent.
    """

    normalized = normalize_steane_block(
        measured_bits
    )

    erasure_positions = [
        index
        for index, bit in enumerate(
            normalized
        )
        if bit is None
    ]

    erasure_count = len(
        erasure_positions
    )

    if erasure_count == 0:
        return tuple(
            int(bit)
            for bit in normalized
            if bit is not None
        )

    if (
        erasure_count
        > STEANE_MAXIMUM_CORRECTABLE_ERASURES
    ):
        raise SyndromeProcessingError(
            (
                "The Steane block contains more than "
                f"{STEANE_MAXIMUM_CORRECTABLE_ERASURES} "
                "erasures."
            ),
            details={
                "erasure_count": erasure_count,
                "erasure_positions": (
                    erasure_positions
                ),
            },
        )

    candidates: list[
        tuple[int, ...]
    ] = []

    for assignment in itertools.product(
        (0, 1),
        repeat=erasure_count,
    ):
        candidate = list(
            normalized
        )

        for position, bit in zip(
            erasure_positions,
            assignment,
            strict=True,
        ):
            candidate[position] = bit

        concrete_candidate = tuple(
            int(bit)
            for bit in candidate
            if bit is not None
        )

        candidate_syndrome = (
            calculate_steane_syndrome(
                concrete_candidate
            )
        )

        if candidate_syndrome == (
            0,
            0,
            0,
        ):
            candidates.append(
                concrete_candidate
            )

    if len(candidates) == 0:
        raise SyndromeProcessingError(
            (
                "No valid Steane codeword matches "
                "the observed measurements."
            ),
            details={
                "erasure_positions": (
                    erasure_positions
                ),
            },
        )

    if len(candidates) > 1:
        raise SyndromeProcessingError(
            (
                "Erasure recovery produced multiple "
                "valid Steane codewords."
            ),
            details={
                "candidate_count": len(
                    candidates
                ),
                "erasure_positions": (
                    erasure_positions
                ),
            },
        )

    return candidates[0]


# ---------------------------------------------------------------------
# Single-block processing
# ---------------------------------------------------------------------

def process_steane_block(
    measured_bits: Sequence[Any],
    *,
    position: int = 0,
    basis: str = BASIS_Z,
) -> SyndromeBlockResult:
    """
    Process one measured Steane [[7,1,3]] logical block.

    Fail-closed behavior is used when too many erasures or inconsistent
    measurements prevent reliable decoding.
    """

    validated_position = validate_integer(
        position,
        field_name="position",
        minimum=0,
    )

    normalized_basis = (
        normalize_measurement_basis(
            basis
        )
    )

    received_bits = normalize_steane_block(
        measured_bits
    )

    erasure_count = sum(
        bit is None
        for bit in received_bits
    )

    # -------------------------------------------------------------
    # Erasure-recovery path
    # -------------------------------------------------------------

    if erasure_count > 0:
        if (
            erasure_count
            > STEANE_MAXIMUM_CORRECTABLE_ERASURES
        ):
            return SyndromeBlockResult(
                position=validated_position,
                basis=normalized_basis,
                received_bits=received_bits,
                corrected_bits=None,
                syndrome=None,
                residual_syndrome=None,
                syndrome_value=0,
                syndrome_weight=0,
                logical_bit=None,
                error_position=None,
                erasure_count=erasure_count,
                recovered_erasure_count=0,
                correction_applied=False,
                correction_success=False,
                erasure_recovery_used=True,
                uncorrectable=True,
                reason=(
                    REASON_TOO_MANY_ERASURES
                ),
            )

        try:
            recovered_bits = (
                recover_steane_erasures(
                    received_bits
                )
            )

        except SyndromeProcessingError as exc:
            reason = (
                REASON_AMBIGUOUS_ERASURE
                if "multiple" in str(
                    exc
                ).lower()
                else (
                    REASON_ERASURE_RECOVERY_FAILED
                )
            )

            return SyndromeBlockResult(
                position=validated_position,
                basis=normalized_basis,
                received_bits=received_bits,
                corrected_bits=None,
                syndrome=None,
                residual_syndrome=None,
                syndrome_value=0,
                syndrome_weight=0,
                logical_bit=None,
                error_position=None,
                erasure_count=erasure_count,
                recovered_erasure_count=0,
                correction_applied=False,
                correction_success=False,
                erasure_recovery_used=True,
                uncorrectable=True,
                reason=reason,
            )

        recovered_syndrome = (
            calculate_steane_syndrome(
                recovered_bits
            )
        )

        logical_bit = (
            decode_steane_logical_bit(
                recovered_bits
            )
        )

        return SyndromeBlockResult(
            position=validated_position,
            basis=normalized_basis,
            received_bits=received_bits,
            corrected_bits=recovered_bits,
            syndrome=recovered_syndrome,
            residual_syndrome=(
                recovered_syndrome
            ),
            syndrome_value=(
                syndrome_to_value(
                    recovered_syndrome
                )
            ),
            syndrome_weight=(
                syndrome_weight(
                    recovered_syndrome
                )
            ),
            logical_bit=logical_bit,
            error_position=None,
            erasure_count=erasure_count,
            recovered_erasure_count=(
                erasure_count
            ),
            correction_applied=True,
            correction_success=True,
            erasure_recovery_used=True,
            uncorrectable=False,
            reason=REASON_ERASURE_RECOVERED,
        )

    # -------------------------------------------------------------
    # Standard syndrome-correction path
    # -------------------------------------------------------------

    syndrome = calculate_steane_syndrome(
        received_bits
    )

    syndrome_value = (
        syndrome_to_value(
            syndrome
        )
    )

    weight = syndrome_weight(
        syndrome
    )

    error_position = (
        syndrome_to_error_position(
            syndrome
        )
    )

    if error_position is None:
        corrected_bits = tuple(
            int(bit)
            for bit in received_bits
            if bit is not None
        )

        logical_bit = (
            decode_steane_logical_bit(
                corrected_bits
            )
        )

        return SyndromeBlockResult(
            position=validated_position,
            basis=normalized_basis,
            received_bits=received_bits,
            corrected_bits=corrected_bits,
            syndrome=syndrome,
            residual_syndrome=syndrome,
            syndrome_value=syndrome_value,
            syndrome_weight=weight,
            logical_bit=logical_bit,
            error_position=None,
            erasure_count=0,
            recovered_erasure_count=0,
            correction_applied=False,
            correction_success=True,
            erasure_recovery_used=False,
            uncorrectable=False,
            reason=REASON_NO_ERROR,
        )

    corrected_bits = flip_physical_bit(
        received_bits,
        error_position,
    )

    residual_syndrome = (
        calculate_steane_syndrome(
            corrected_bits
        )
    )

    if residual_syndrome != (
        0,
        0,
        0,
    ):
        return SyndromeBlockResult(
            position=validated_position,
            basis=normalized_basis,
            received_bits=received_bits,
            corrected_bits=None,
            syndrome=syndrome,
            residual_syndrome=(
                residual_syndrome
            ),
            syndrome_value=syndrome_value,
            syndrome_weight=weight,
            logical_bit=None,
            error_position=error_position,
            erasure_count=0,
            recovered_erasure_count=0,
            correction_applied=True,
            correction_success=False,
            erasure_recovery_used=False,
            uncorrectable=True,
            reason=REASON_RESIDUAL_SYNDROME,
        )

    logical_bit = decode_steane_logical_bit(
        corrected_bits
    )

    return SyndromeBlockResult(
        position=validated_position,
        basis=normalized_basis,
        received_bits=received_bits,
        corrected_bits=corrected_bits,
        syndrome=syndrome,
        residual_syndrome=(
            residual_syndrome
        ),
        syndrome_value=syndrome_value,
        syndrome_weight=weight,
        logical_bit=logical_bit,
        error_position=error_position,
        erasure_count=0,
        recovered_erasure_count=0,
        correction_applied=True,
        correction_success=True,
        erasure_recovery_used=False,
        uncorrectable=False,
        reason=(
            REASON_SINGLE_ERROR_CORRECTED
        ),
    )


# ---------------------------------------------------------------------
# Generic block extraction
# ---------------------------------------------------------------------

def _read_field(
    value: Any,
    field_names: tuple[str, ...],
) -> Any:
    """Read the first matching mapping key or object attribute."""

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


def _extract_block_bits(
    block: Any,
) -> Sequence[Any]:
    """
    Extract physical measurements from a block representation.
    """

    if (
        isinstance(
            block,
            Sequence,
        )
        and not isinstance(
            block,
            (
                str,
                bytes,
                bytearray,
            ),
        )
    ):
        return block

    bits = _read_field(
        block,
        (
            "measured_bits",
            "measurement_bits",
            "physical_bits",
            "received_bits",
            "bits",
        ),
    )

    if bits is None:
        raise SyndromeProcessingError(
            (
                "A Steane block does not contain "
                "physical measurement bits."
            ),
            details={
                "received_type": type(
                    block
                ).__name__,
            },
        )

    return bits


def _extract_block_position(
    block: Any,
    *,
    default_position: int,
) -> int:
    """Extract an optional logical-frame position."""

    position = _read_field(
        block,
        (
            "position",
            "frame_position",
            "block_position",
        ),
    )

    if position is None:
        return default_position

    return validate_integer(
        position,
        field_name="position",
        minimum=0,
    )


def _extract_block_basis(
    block: Any,
    *,
    default_basis: str,
) -> str:
    """Extract an optional block measurement basis."""

    basis = _read_field(
        block,
        (
            "basis",
            "measurement_basis",
            "declared_basis",
        ),
    )

    if basis is None:
        return normalize_measurement_basis(
            default_basis
        )

    return normalize_measurement_basis(
        basis
    )


# ---------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------

def summarize_syndrome_results(
    results: Sequence[
        SyndromeBlockResult
    ],
) -> SyndromeProcessingSummary:
    """
    Calculate aggregate correction and GP feature statistics.
    """

    if isinstance(
        results,
        (
            str,
            bytes,
            bytearray,
        ),
    ) or not isinstance(
        results,
        Sequence,
    ):
        raise SyndromeProcessingError(
            (
                "results must be a sequence of "
                "SyndromeBlockResult objects."
            )
        )

    normalized_results = tuple(
        results
    )

    for index, result in enumerate(
        normalized_results
    ):
        if not isinstance(
            result,
            SyndromeBlockResult,
        ):
            raise ProtocolValidationError(
                (
                    f"results[{index}] must be a "
                    "SyndromeBlockResult object."
                )
            )

    total_blocks = len(
        normalized_results
    )

    decoded_blocks = sum(
        result.decoded
        for result in normalized_results
    )

    corrected_blocks = sum(
        (
            result.correction_applied
            and result.correction_success
            and not result.erasure_recovery_used
        )
        for result in normalized_results
    )

    erasure_recovered_blocks = sum(
        (
            result.erasure_recovery_used
            and result.correction_success
        )
        for result in normalized_results
    )

    failed_blocks = sum(
        not result.correction_success
        for result in normalized_results
    )

    total_erasures = sum(
        result.erasure_count
        for result in normalized_results
    )

    weights = [
        result.syndrome_weight
        for result in normalized_results
    ]

    mean_weight = (
        float(
            mean(
                weights
            )
        )
        if weights
        else 0.0
    )

    max_weight = (
        max(
            weights
        )
        if weights
        else 0
    )

    failure_rate = (
        failed_blocks
        / total_blocks
        if total_blocks > 0
        else 0.0
    )

    return SyndromeProcessingSummary(
        total_blocks=total_blocks,
        decoded_blocks=decoded_blocks,
        corrected_blocks=corrected_blocks,
        erasure_recovered_blocks=(
            erasure_recovered_blocks
        ),
        failed_blocks=failed_blocks,
        total_erasures=total_erasures,
        mean_syndrome_weight=(
            mean_weight
        ),
        max_syndrome_weight=(
            max_weight
        ),
        correction_failure_rate=(
            failure_rate
        ),
        results=normalized_results,
    )


def process_steane_blocks(
    blocks: Sequence[Any],
    *,
    default_basis: str = BASIS_Z,
) -> SyndromeProcessingSummary:
    """
    Process multiple Steane logical blocks.

    Each element may be:

    - A direct seven-value sequence
    - A mapping containing measured_bits
    - An object containing measured_bits

    Optional position and basis fields are supported.
    """

    if isinstance(
        blocks,
        (
            str,
            bytes,
            bytearray,
        ),
    ) or not isinstance(
        blocks,
        Sequence,
    ):
        raise SyndromeProcessingError(
            "blocks must be a sequence."
        )

    normalized_default_basis = (
        normalize_measurement_basis(
            default_basis
        )
    )

    results: list[
        SyndromeBlockResult
    ] = []

    for index, block in enumerate(
        blocks
    ):
        measured_bits = (
            _extract_block_bits(
                block
            )
        )

        position = (
            _extract_block_position(
                block,
                default_position=index,
            )
        )

        basis = _extract_block_basis(
            block,
            default_basis=(
                normalized_default_basis
            ),
        )

        result = process_steane_block(
            measured_bits,
            position=position,
            basis=basis,
        )

        results.append(
            result
        )

    return summarize_syndrome_results(
        results
    )


def require_decodable_block(
    result: SyndromeBlockResult,
) -> int:
    """
    Require successful Steane decoding and return the logical bit.
    """

    if not isinstance(
        result,
        SyndromeBlockResult,
    ):
        raise ProtocolValidationError(
            (
                "result must be a "
                "SyndromeBlockResult object."
            )
        )

    if not result.decoded:
        raise SyndromeProcessingError(
            (
                "The Steane logical block could "
                "not be decoded."
            ),
            details=result.to_dict(),
        )

    assert result.logical_bit is not None

    return result.logical_bit


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_syndrome_processor_self_test() -> dict[str, Any]:
    """
    Test valid decoding, single-error correction, erasure recovery,
    excessive-loss rejection, and GP feature generation.
    """

    logical_zero_codeword = (
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    )

    logical_one_codeword = (
        0,
        1,
        0,
        1,
        0,
        1,
        0,
    )

    no_error_result = (
        process_steane_block(
            logical_zero_codeword,
            position=0,
            basis=BASIS_Z,
        )
    )

    single_error_bits = list(
        logical_one_codeword
    )

    single_error_bits[4] ^= 1

    corrected_result = (
        process_steane_block(
            single_error_bits,
            position=1,
            basis=BASIS_Z,
        )
    )

    one_erasure_bits = list(
        logical_zero_codeword
    )

    one_erasure_bits[2] = None

    one_erasure_result = (
        process_steane_block(
            one_erasure_bits,
            position=2,
            basis=BASIS_X,
        )
    )

    two_erasure_bits = list(
        logical_one_codeword
    )

    two_erasure_bits[0] = None
    two_erasure_bits[6] = None

    two_erasure_result = (
        process_steane_block(
            two_erasure_bits,
            position=3,
            basis=BASIS_Z,
        )
    )

    excessive_erasure_bits = list(
        logical_zero_codeword
    )

    excessive_erasure_bits[0] = None
    excessive_erasure_bits[1] = None
    excessive_erasure_bits[2] = None

    failed_result = process_steane_block(
        excessive_erasure_bits,
        position=4,
        basis=BASIS_Z,
    )

    summary = summarize_syndrome_results(
        (
            no_error_result,
            corrected_result,
            one_erasure_result,
            two_erasure_result,
            failed_result,
        )
    )

    no_error_pass = all(
        (
            no_error_result.decoded,
            no_error_result.logical_bit == 0,
            no_error_result.syndrome
            == (0, 0, 0),
            not no_error_result
            .correction_applied,
        )
    )

    single_error_pass = all(
        (
            corrected_result.decoded,
            corrected_result.logical_bit == 1,
            corrected_result
            .correction_applied,
            corrected_result.error_position
            == 4,
            corrected_result
            .residual_syndrome
            == (0, 0, 0),
        )
    )

    erasure_recovery_pass = all(
        (
            one_erasure_result.decoded,
            one_erasure_result.logical_bit
            == 0,
            one_erasure_result
            .recovered_erasure_count
            == 1,

            two_erasure_result.decoded,
            two_erasure_result.logical_bit
            == 1,
            two_erasure_result
            .recovered_erasure_count
            == 2,
        )
    )

    excessive_erasure_rejected = all(
        (
            not failed_result.decoded,
            failed_result.uncorrectable,
            failed_result.reason
            == REASON_TOO_MANY_ERASURES,
        )
    )

    feature_dict = (
        summary.to_feature_dict()
    )

    feature_generation_pass = all(
        feature_name in feature_dict
        for feature_name in (
            "mean_syndrome_weight",
            "max_syndrome_weight",
            "correction_failure_rate",
        )
    )

    success = all(
        (
            no_error_pass,
            single_error_pass,
            erasure_recovery_pass,
            excessive_erasure_rejected,
            feature_generation_pass,
            summary.total_blocks == 5,
            summary.decoded_blocks == 4,
            summary.failed_blocks == 1,
            summary.correction_failure_rate
            == 0.2,
        )
    )

    return {
        "success": success,

        "no_error_decode_pass": (
            no_error_pass
        ),

        "single_error_corrected": (
            single_error_pass
        ),

        "corrected_error_position": (
            corrected_result.error_position
        ),

        "erasure_recovery_pass": (
            erasure_recovery_pass
        ),

        "one_erasure_recovered": (
            one_erasure_result.decoded
        ),

        "two_erasures_recovered": (
            two_erasure_result.decoded
        ),

        "excessive_erasure_rejected": (
            excessive_erasure_rejected
        ),

        "total_blocks": (
            summary.total_blocks
        ),

        "decoded_blocks": (
            summary.decoded_blocks
        ),

        "failed_blocks": (
            summary.failed_blocks
        ),

        "mean_syndrome_weight": (
            summary.mean_syndrome_weight
        ),

        "max_syndrome_weight": (
            summary.max_syndrome_weight
        ),

        "correction_failure_rate": (
            summary.correction_failure_rate
        ),

        "gp_features": feature_dict,
    }


__all__ = [
    "STEANE_BLOCK_SIZE",
    "STEANE_MAXIMUM_CORRECTABLE_ERRORS",
    "STEANE_MAXIMUM_CORRECTABLE_ERASURES",
    "STEANE_PARITY_CHECK_MATRIX",
    "REASON_NO_ERROR",
    "REASON_SINGLE_ERROR_CORRECTED",
    "REASON_ERASURE_RECOVERED",
    "REASON_TOO_MANY_ERASURES",
    "REASON_AMBIGUOUS_ERASURE",
    "REASON_ERASURE_RECOVERY_FAILED",
    "REASON_RESIDUAL_SYNDROME",
    "SyndromeProcessingError",
    "SyndromeBlockResult",
    "SyndromeProcessingSummary",
    "normalize_measurement_basis",
    "normalize_physical_bit",
    "normalize_steane_block",
    "normalize_syndrome",
    "calculate_steane_syndrome",
    "syndrome_to_value",
    "syndrome_to_error_position",
    "syndrome_weight",
    "flip_physical_bit",
    "decode_steane_logical_bit",
    "recover_steane_erasures",
    "process_steane_block",
    "summarize_syndrome_results",
    "process_steane_blocks",
    "require_decodable_block",
    "run_syndrome_processor_self_test",
]