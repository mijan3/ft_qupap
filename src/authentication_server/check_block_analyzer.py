"""
Check-block measurement and raw-QBER analysis for FT-QuPAP v5.1.

The Authentication Server receives a noisy quantum frame containing:

- 128 logical KMAC payload blocks
- 32 independent logical check blocks

The encrypted control schedule identifies the check-block positions,
expected physical reference patterns, and declared measurement bases.

This module:

1. Locates declared check blocks.
2. Measures each available block in its declared X or Z basis.
3. Compares measured physical bits with expected reference bits.
4. Calculates raw QBER only from observed check evidence.
5. Calculates check-block loss and evidence availability.
6. Enforces the minimum requirement of 24 observed check blocks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.common.constants import (
    BASIS_X,
    BASIS_Z,
    CHECK_LOGICAL_QUBITS,
    FIXED_QBER_THRESHOLD,
    MINIMUM_OBSERVED_CHECK_BLOCKS,
)

from src.common.exceptions import (
    InsufficientCheckBlocksError,
    ProtocolValidationError,
)

from src.common.validators import (
    validate_integer,
    validate_probability,
)


# ---------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CheckBlockAnalysisResult:
    """
    Complete check-block analysis result.

    qber_raw
        Number of mismatched observed physical check bits divided by the
        number of observed physical check bits.

    observed_check_blocks
        Number of check blocks that produced usable measurements.

    unavailable_check_blocks
        Declared check blocks that were erased, missing, malformed, or
        outside the received-frame range.

    structural_mismatch_blocks
        Blocks whose received identity or physical length did not match
        the authenticated control schedule.
    """

    qber_raw: float

    mismatched_bits: int
    observed_bits: int

    declared_check_blocks: int
    observed_check_blocks: int
    unavailable_check_blocks: int
    structural_mismatch_blocks: int

    check_block_loss_rate: float

    minimum_evidence_pass: bool
    fixed_qber_pass: bool

    minimum_required_check_blocks: int
    fixed_qber_threshold: float

    details: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        validate_probability(
            self.qber_raw,
            field_name="qber_raw",
        )

        for field_name in (
            "mismatched_bits",
            "observed_bits",
            "declared_check_blocks",
            "observed_check_blocks",
            "unavailable_check_blocks",
            "structural_mismatch_blocks",
            "minimum_required_check_blocks",
        ):
            validate_integer(
                getattr(self, field_name),
                field_name=field_name,
                minimum=0,
            )

        validate_probability(
            self.check_block_loss_rate,
            field_name="check_block_loss_rate",
        )

        validate_probability(
            self.fixed_qber_threshold,
            field_name="fixed_qber_threshold",
        )

        if self.mismatched_bits > self.observed_bits:
            raise ProtocolValidationError(
                "Mismatched bits cannot exceed observed bits."
            )

        if (
            self.observed_check_blocks
            + self.unavailable_check_blocks
            != self.declared_check_blocks
        ):
            raise ProtocolValidationError(
                "Observed and unavailable check-block counts are inconsistent."
            )

    @property
    def deterministic_check_pass(self) -> bool:
        """
        Return True when both evidence and fixed-QBER checks pass.
        """

        return (
            self.minimum_evidence_pass
            and self.fixed_qber_pass
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result dictionary."""

        result = asdict(self)

        result["details"] = [
            dict(item)
            for item in self.details
        ]

        result["deterministic_check_pass"] = (
            self.deterministic_check_pass
        )

        return result


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a dictionary-like value."""

    if not isinstance(value, Mapping):
        raise ProtocolValidationError(
            f"{field_name} must be a mapping.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    return value


def _require_non_string_sequence(
    value: Any,
    *,
    field_name: str,
) -> Sequence[Any]:
    """Require a sequence that is not text or raw bytes."""

    if isinstance(
        value,
        (str, bytes, bytearray),
    ) or not isinstance(value, Sequence):
        raise ProtocolValidationError(
            f"{field_name} must be a sequence.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    return value


def _to_binary_array(
    value: Any,
    *,
    field_name: str,
) -> np.ndarray:
    """
    Convert a sequence into a one-dimensional NumPy bit array.
    """

    if isinstance(value, np.ndarray):
        array = np.asarray(
            value,
            dtype=np.int8,
        ).reshape(-1)

    else:
        sequence = _require_non_string_sequence(
            value,
            field_name=field_name,
        )

        array = np.asarray(
            list(sequence),
            dtype=np.int8,
        ).reshape(-1)

    if array.size == 0:
        raise ProtocolValidationError(
            f"{field_name} cannot be empty."
        )

    if not np.all(
        np.isin(
            array,
            (0, 1),
        )
    ):
        raise ProtocolValidationError(
            f"{field_name} must contain only 0 and 1."
        )

    return array


def _to_boolean_array(
    value: Any,
    *,
    field_name: str,
    expected_length: int | None = None,
) -> np.ndarray:
    """Convert a value into a one-dimensional Boolean array."""

    array = np.asarray(
        value,
        dtype=bool,
    ).reshape(-1)

    if (
        expected_length is not None
        and len(array) != expected_length
    ):
        raise ProtocolValidationError(
            f"{field_name} has an invalid length.",
            details={
                "expected_length": expected_length,
                "actual_length": len(array),
            },
        )

    return array


def _get_block_attribute(
    block: Any,
    attribute_name: str,
) -> Any:
    """
    Read a block value from either an object or a dictionary.
    """

    if isinstance(block, Mapping):
        if attribute_name not in block:
            raise ProtocolValidationError(
                f"Received block is missing '{attribute_name}'."
            )

        return block[attribute_name]

    if not hasattr(block, attribute_name):
        raise ProtocolValidationError(
            f"Received block is missing '{attribute_name}'.",
            details={
                "block_type": type(block).__name__,
            },
        )

    return getattr(
        block,
        attribute_name,
    )


def _get_block_id(
    block: Any,
) -> str | None:
    """
    Read the logical block ID from common block representations.

    Supported forms:

        block.spec.block_id
        block["spec"]["block_id"]
        block.block_id
        block["block_id"]
    """

    if isinstance(block, Mapping):
        direct_id = block.get("block_id")

        if direct_id is not None:
            return str(direct_id)

        spec = block.get("spec")

        if isinstance(spec, Mapping):
            block_id = spec.get("block_id")

            if block_id is not None:
                return str(block_id)

        return None

    direct_id = getattr(
        block,
        "block_id",
        None,
    )

    if direct_id is not None:
        return str(direct_id)

    spec = getattr(
        block,
        "spec",
        None,
    )

    block_id = getattr(
        spec,
        "block_id",
        None,
    )

    return (
        str(block_id)
        if block_id is not None
        else None
    )


# ---------------------------------------------------------------------
# Declared-basis measurement
# ---------------------------------------------------------------------

def measure_block_in_declared_basis(
    block: Any,
    declared_basis: str,
) -> np.ndarray | None:
    """
    Measure one physical check block in the declared basis.

    Simulator interpretation:

    Z-basis measurement:
        Observes reference bits affected by X errors.

    X-basis measurement:
        Observes reference bits affected by Z errors.

    A block containing any erasure returns None and contributes no
    measured evidence.
    """

    if not isinstance(declared_basis, str):
        raise ProtocolValidationError(
            "Declared measurement basis must be a string."
        )

    normalized_basis = (
        declared_basis
        .strip()
        .upper()
    )

    if normalized_basis not in {
        BASIS_Z,
        BASIS_X,
    }:
        raise ProtocolValidationError(
            "Declared basis must be Z or X.",
            details={
                "received_basis": declared_basis,
            },
        )

    reference_bits = _to_binary_array(
        _get_block_attribute(
            block,
            "reference_bits",
        ),
        field_name="reference_bits",
    )

    erasures = _to_boolean_array(
        _get_block_attribute(
            block,
            "erasures",
        ),
        field_name="erasures",
        expected_length=len(
            reference_bits
        ),
    )

    if np.any(erasures):
        return None

    error_attribute = (
        "x_errors"
        if normalized_basis == BASIS_Z
        else "z_errors"
    )

    relevant_errors = _to_binary_array(
        _get_block_attribute(
            block,
            error_attribute,
        ),
        field_name=error_attribute,
    )

    if len(relevant_errors) != len(
        reference_bits
    ):
        raise ProtocolValidationError(
            "Physical error and reference arrays have different lengths.",
            details={
                "reference_length": len(
                    reference_bits
                ),
                "error_length": len(
                    relevant_errors
                ),
                "basis": normalized_basis,
            },
        )

    return np.bitwise_xor(
        reference_bits,
        relevant_errors,
    ).astype(
        np.int8,
        copy=False,
    )


# ---------------------------------------------------------------------
# Raw-QBER calculation
# ---------------------------------------------------------------------

def calculate_raw_qber(
    received_frame: Sequence[Any],
    schedule: Mapping[str, Any],
) -> tuple[float, int, int]:
    """
    Calculate raw QBER only from declared check blocks.

    Returns:

        qber_raw,
        mismatched_physical_bits,
        observed_physical_bits

    Formula:

        QBER_raw = mismatched observed check bits
                   --------------------------------
                       total observed check bits

    When no usable check evidence exists, QBER is conservatively 1.0.
    """

    analysis = analyze_check_blocks(
        received_frame=received_frame,
        schedule=schedule,
        enforce_minimum=False,
    )

    return (
        analysis.qber_raw,
        analysis.mismatched_bits,
        analysis.observed_bits,
    )


def analyze_check_blocks(
    received_frame: Sequence[Any],
    schedule: Mapping[str, Any],
    *,
    minimum_observed_check_blocks: int = (
        MINIMUM_OBSERVED_CHECK_BLOCKS
    ),
    fixed_qber_threshold: float = (
        FIXED_QBER_THRESHOLD
    ),
    enforce_minimum: bool = False,
) -> CheckBlockAnalysisResult:
    """
    Perform complete declared check-block analysis.

    Structural mismatches are treated as strong disturbance evidence:

    - Wrong logical block ID
    - Received pattern length different from expected pattern length

    For these cases, the expected comparison length is counted as fully
    mismatched evidence rather than silently ignoring the block.
    """

    frame = _require_non_string_sequence(
        received_frame,
        field_name="received_frame",
    )

    validated_schedule = _require_mapping(
        schedule,
        field_name="schedule",
    )

    raw_check_entries = validated_schedule.get(
        "check_blocks",
    )

    if raw_check_entries is None:
        raise ProtocolValidationError(
            "Control schedule does not contain 'check_blocks'."
        )

    check_entries = _require_non_string_sequence(
        raw_check_entries,
        field_name="schedule.check_blocks",
    )

    minimum_required = validate_integer(
        minimum_observed_check_blocks,
        field_name="minimum_observed_check_blocks",
        minimum=1,
        maximum=CHECK_LOGICAL_QUBITS,
    )

    qber_threshold = validate_probability(
        fixed_qber_threshold,
        field_name="fixed_qber_threshold",
    )

    if len(check_entries) == 0:
        raise ProtocolValidationError(
            "The control schedule contains no check blocks."
        )

    if len(check_entries) > CHECK_LOGICAL_QUBITS:
        raise ProtocolValidationError(
            "The control schedule declares too many check blocks.",
            details={
                "maximum_check_blocks": (
                    CHECK_LOGICAL_QUBITS
                ),
                "declared_check_blocks": len(
                    check_entries
                ),
            },
        )

    mismatched_bits = 0
    observed_bits = 0

    observed_check_blocks = 0
    unavailable_check_blocks = 0
    structural_mismatch_blocks = 0

    details: list[dict[str, Any]] = []

    seen_positions: set[int] = set()
    seen_block_ids: set[str] = set()

    for check_index, raw_entry in enumerate(
        check_entries
    ):
        entry = _require_mapping(
            raw_entry,
            field_name=(
                f"schedule.check_blocks[{check_index}]"
            ),
        )

        required_fields = (
            "position",
            "block_id",
            "basis",
            "expected_reference_bits",
        )

        missing_fields = [
            field_name
            for field_name in required_fields
            if field_name not in entry
        ]

        if missing_fields:
            raise ProtocolValidationError(
                "Check-block schedule entry is incomplete.",
                details={
                    "check_index": check_index,
                    "missing_fields": missing_fields,
                },
            )

        position = validate_integer(
            entry["position"],
            field_name=(
                f"check_blocks[{check_index}].position"
            ),
            minimum=0,
        )

        expected_block_id = str(
            entry["block_id"]
        )

        declared_basis = str(
            entry["basis"]
        ).strip().upper()

        expected_bits = _to_binary_array(
            entry["expected_reference_bits"],
            field_name=(
                "expected_reference_bits"
            ),
        )

        if position in seen_positions:
            raise ProtocolValidationError(
                "Duplicate check-block position in control schedule.",
                details={
                    "position": position,
                },
            )

        if expected_block_id in seen_block_ids:
            raise ProtocolValidationError(
                "Duplicate check-block ID in control schedule.",
                details={
                    "block_id": expected_block_id,
                },
            )

        seen_positions.add(position)
        seen_block_ids.add(expected_block_id)

        block_detail: dict[str, Any] = {
            "check_index": check_index,
            "position": position,
            "expected_block_id": expected_block_id,
            "basis": declared_basis,
            "expected_bits": len(expected_bits),
            "status": "pending",
            "mismatched_bits": 0,
            "observed_bits": 0,
        }

        if position >= len(frame):
            unavailable_check_blocks += 1

            block_detail["status"] = (
                "position_out_of_range"
            )

            details.append(block_detail)
            continue

        block = frame[position]

        received_block_id = _get_block_id(
            block
        )

        block_detail["received_block_id"] = (
            received_block_id
        )

        if received_block_id != expected_block_id:
            mismatch_count = len(
                expected_bits
            )

            mismatched_bits += mismatch_count
            observed_bits += mismatch_count

            observed_check_blocks += 1
            structural_mismatch_blocks += 1

            block_detail.update(
                {
                    "status": "block_id_mismatch",
                    "mismatched_bits": (
                        mismatch_count
                    ),
                    "observed_bits": (
                        mismatch_count
                    ),
                }
            )

            details.append(block_detail)
            continue

        try:
            measurement = (
                measure_block_in_declared_basis(
                    block,
                    declared_basis,
                )
            )

        except ProtocolValidationError as exc:
            unavailable_check_blocks += 1

            block_detail.update(
                {
                    "status": (
                        "measurement_error"
                    ),
                    "error": str(exc),
                }
            )

            details.append(block_detail)
            continue

        if measurement is None:
            unavailable_check_blocks += 1

            block_detail["status"] = (
                "erased"
            )

            details.append(block_detail)
            continue

        if len(measurement) != len(
            expected_bits
        ):
            comparison_length = max(
                len(measurement),
                len(expected_bits),
            )

            mismatched_bits += comparison_length
            observed_bits += comparison_length

            observed_check_blocks += 1
            structural_mismatch_blocks += 1

            block_detail.update(
                {
                    "status": "length_mismatch",
                    "received_bits": len(
                        measurement
                    ),
                    "mismatched_bits": (
                        comparison_length
                    ),
                    "observed_bits": (
                        comparison_length
                    ),
                }
            )

            details.append(block_detail)
            continue

        block_mismatches = int(
            np.count_nonzero(
                measurement
                != expected_bits
            )
        )

        block_observed_bits = len(
            expected_bits
        )

        mismatched_bits += block_mismatches
        observed_bits += block_observed_bits
        observed_check_blocks += 1

        block_detail.update(
            {
                "status": "observed",
                "mismatched_bits": (
                    block_mismatches
                ),
                "observed_bits": (
                    block_observed_bits
                ),
                "measurement": (
                    measurement.tolist()
                ),
            }
        )

        details.append(block_detail)

    declared_check_blocks = len(
        check_entries
    )

    qber_raw = (
        mismatched_bits / observed_bits
        if observed_bits > 0
        else 1.0
    )

    unavailable_check_blocks = (
        declared_check_blocks
        - observed_check_blocks
    )

    check_block_loss_rate = (
        unavailable_check_blocks
        / declared_check_blocks
    )

    minimum_evidence_pass = (
        observed_check_blocks
        >= minimum_required
    )

    fixed_qber_pass = (
        qber_raw
        <= qber_threshold
    )

    result = CheckBlockAnalysisResult(
        qber_raw=float(qber_raw),

        mismatched_bits=mismatched_bits,
        observed_bits=observed_bits,

        declared_check_blocks=(
            declared_check_blocks
        ),

        observed_check_blocks=(
            observed_check_blocks
        ),

        unavailable_check_blocks=(
            unavailable_check_blocks
        ),

        structural_mismatch_blocks=(
            structural_mismatch_blocks
        ),

        check_block_loss_rate=float(
            check_block_loss_rate
        ),

        minimum_evidence_pass=(
            minimum_evidence_pass
        ),

        fixed_qber_pass=(
            fixed_qber_pass
        ),

        minimum_required_check_blocks=(
            minimum_required
        ),

        fixed_qber_threshold=(
            qber_threshold
        ),

        details=tuple(details),
    )

    if (
        enforce_minimum
        and not result.minimum_evidence_pass
    ):
        raise InsufficientCheckBlocksError(
            observed=result.observed_check_blocks,
            required=minimum_required,
        )

    return result


def require_sufficient_check_evidence(
    result: CheckBlockAnalysisResult,
) -> None:
    """
    Require that enough check blocks were successfully observed.
    """

    if not isinstance(
        result,
        CheckBlockAnalysisResult,
    ):
        raise ProtocolValidationError(
            "result must be a CheckBlockAnalysisResult object."
        )

    if not result.minimum_evidence_pass:
        raise InsufficientCheckBlocksError(
            observed=result.observed_check_blocks,
            required=(
                result.minimum_required_check_blocks
            ),
        )


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

@dataclass
class _SelfTestBlockSpec:
    block_id: str


@dataclass
class _SelfTestPhysicalBlock:
    spec: _SelfTestBlockSpec
    reference_bits: np.ndarray
    x_errors: np.ndarray
    z_errors: np.ndarray
    erasures: np.ndarray


def run_check_block_analyzer_self_test() -> dict[str, Any]:
    """
    Run deterministic ideal, noisy, and erased check-block tests.
    """

    frame: list[_SelfTestPhysicalBlock] = []
    schedule_entries: list[dict[str, Any]] = []

    reference = np.asarray(
        [1, 0, 1, 0, 1, 0, 1],
        dtype=np.int8,
    )

    for index in range(
        CHECK_LOGICAL_QUBITS
    ):
        block_id = (
            f"CHECK-{index:02d}"
        )

        x_errors = np.zeros(
            7,
            dtype=np.int8,
        )

        z_errors = np.zeros(
            7,
            dtype=np.int8,
        )

        erasures = np.zeros(
            7,
            dtype=bool,
        )

        # Add two single-bit mismatches.
        if index in (0, 1):
            x_errors[0] = 1

        # Erase two blocks, leaving 30 observed.
        if index in (30, 31):
            erasures[0] = True

        frame.append(
            _SelfTestPhysicalBlock(
                spec=_SelfTestBlockSpec(
                    block_id=block_id
                ),
                reference_bits=reference.copy(),
                x_errors=x_errors,
                z_errors=z_errors,
                erasures=erasures,
            )
        )

        schedule_entries.append(
            {
                "position": index,
                "block_id": block_id,
                "basis": BASIS_Z,
                "expected_reference_bits": (
                    reference.tolist()
                ),
            }
        )

    result = analyze_check_blocks(
        received_frame=frame,
        schedule={
            "check_blocks": (
                schedule_entries
            )
        },
    )

    expected_observed_bits = (
        30 * 7
    )

    expected_qber = (
        2 / expected_observed_bits
    )

    success = all(
        (
            result.observed_check_blocks == 30,
            result.unavailable_check_blocks == 2,
            result.mismatched_bits == 2,
            result.observed_bits
            == expected_observed_bits,
            np.isclose(
                result.qber_raw,
                expected_qber,
            ),
            result.minimum_evidence_pass,
            result.fixed_qber_pass,
            result.deterministic_check_pass,
        )
    )

    return {
        "success": success,
        "qber_raw": result.qber_raw,
        "expected_qber": expected_qber,
        "mismatched_bits": (
            result.mismatched_bits
        ),
        "observed_bits": result.observed_bits,
        "declared_check_blocks": (
            result.declared_check_blocks
        ),
        "observed_check_blocks": (
            result.observed_check_blocks
        ),
        "unavailable_check_blocks": (
            result.unavailable_check_blocks
        ),
        "minimum_evidence_pass": (
            result.minimum_evidence_pass
        ),
        "fixed_qber_pass": (
            result.fixed_qber_pass
        ),
        "deterministic_check_pass": (
            result.deterministic_check_pass
        ),
    }


__all__ = [
    "CheckBlockAnalysisResult",
    "measure_block_in_declared_basis",
    "calculate_raw_qber",
    "analyze_check_blocks",
    "require_sufficient_check_evidence",
    "run_check_block_analyzer_self_test",
]