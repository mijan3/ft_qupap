"""
FT-QuPAP Raw QBER Calculator

This module calculates receiver-observable raw Quantum Bit Error Rate
from the independent FT-QuPAP check blocks.

Protocol order:

    1. Authentication Server decrypts and validates the control schedule.
    2. Declared check blocks are measured in their protected bases.
    3. Raw QBER is calculated before CSS correction.
    4. Syndrome extraction and error correction are performed later.
    5. Raw QBER is supplied to deterministic checks and GP features.

Notebook-compatible equation:

    QBER_raw = mismatched observed physical check bits
               ---------------------------------------
               total observed physical check bits

Notebook-compatible behavior:

- Only schedule["check_blocks"] is used.
- Payload blocks are never used to calculate raw QBER.
- Invalid check positions are skipped.
- Erased check blocks are excluded from observed-bit counts.
- A block-ID mismatch counts as a complete expected-pattern mismatch.
- A measurement-length mismatch counts max(actual, expected) errors.
- When no check bits are observed, QBER is 1.0.
- At least 24 of 32 check blocks are normally required.

Security boundary:

The expected check patterns are obtained from the authenticated and
AES-GCM-protected control schedule. Hidden Eve settings, attack labels,
and attacked-mask values are never used in QBER calculation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .measurement import (
    measure_block_in_declared_basis,
)
from .steane_css import (
    PhysicalBlock,
    STEANE_BLOCK_SIZE,
    SteaneEncodedFrame,
    validate_physical_block,
)


DEFAULT_CHECK_BLOCK_COUNT = 32
DEFAULT_MIN_OBSERVED_CHECK_BLOCKS = 24
DEFAULT_FIXED_QBER_THRESHOLD = 0.11

QBER_POLICY_ACCEPTED = "qber_policy_accepted"
QBER_POLICY_INSUFFICIENT_EVIDENCE = (
    "insufficient_observed_check_blocks"
)
QBER_POLICY_THRESHOLD_EXCEEDED = (
    "fixed_qber_threshold_exceeded"
)


class QBERCalculationError(Exception):
    """Base exception for raw-QBER calculation failures."""


class InvalidQBERFrameError(
    QBERCalculationError
):
    """Raised when a received quantum frame is invalid."""


class InvalidQBERScheduleError(
    QBERCalculationError
):
    """Raised when a check-block schedule is malformed."""


class InvalidExpectedCheckPatternError(
    QBERCalculationError
):
    """Raised when an expected check pattern is invalid."""


class InvalidQBERPolicyError(
    QBERCalculationError
):
    """Raised when a QBER policy parameter is invalid."""


@dataclass(frozen=True)
class RawQBERResult:
    """
    Detailed raw-QBER calculation result.

    Attributes:
        qber_raw:
            Raw physical check-bit error rate.

        mismatches:
            Number of mismatching observed physical check bits.

        observed:
            Number of physical check bits included in the QBER
            denominator.

        scheduled_check_blocks:
            Number of entries declared in schedule["check_blocks"].

        observed_check_blocks:
            Number of check blocks that contributed physical bits.

        erased_check_blocks:
            Number of declared check blocks excluded due to erasure.

        skipped_invalid_positions:
            Number of entries whose positions were outside the frame.

        block_id_mismatches:
            Number of scheduled entries whose block ID did not match
            the received block at the declared position.

        length_mismatches:
            Number of measurements whose length differed from the
            protected expected physical pattern.
    """

    qber_raw: float
    mismatches: int
    observed: int

    scheduled_check_blocks: int
    observed_check_blocks: int
    erased_check_blocks: int
    skipped_invalid_positions: int
    block_id_mismatches: int
    length_mismatches: int

    def __post_init__(self) -> None:
        validate_qber_value(
            self.qber_raw,
            field_name="qber_raw",
        )

        integer_fields = {
            "mismatches":
                self.mismatches,
            "observed":
                self.observed,
            "scheduled_check_blocks":
                self.scheduled_check_blocks,
            "observed_check_blocks":
                self.observed_check_blocks,
            "erased_check_blocks":
                self.erased_check_blocks,
            "skipped_invalid_positions":
                self.skipped_invalid_positions,
            "block_id_mismatches":
                self.block_id_mismatches,
            "length_mismatches":
                self.length_mismatches,
        }

        for field_name, value in integer_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise TypeError(
                    f"{field_name} must be an integer."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if self.mismatches > self.observed:
            raise ValueError(
                "mismatches cannot exceed observed bits."
            )

        if (
            self.observed_check_blocks
            > self.scheduled_check_blocks
        ):
            raise ValueError(
                "observed_check_blocks cannot exceed "
                "scheduled_check_blocks."
            )

        expected_qber = (
            self.mismatches / self.observed
            if self.observed > 0
            else 1.0
        )

        if not np.isclose(
            self.qber_raw,
            expected_qber,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "qber_raw is inconsistent with mismatch "
                "and observation counts."
            )

    @property
    def has_observations(self) -> bool:
        """Return whether any physical check bits were observed."""

        return self.observed > 0

    @property
    def perfect_match(self) -> bool:
        """Return whether every observed check bit matched."""

        return (
            self.has_observations
            and self.mismatches == 0
        )

    @property
    def missing_check_blocks(self) -> int:
        """
        Return declared check blocks that did not contribute bits.
        """

        return (
            self.scheduled_check_blocks
            - self.observed_check_blocks
        )

    def has_sufficient_check_blocks(
        self,
        minimum_observed_check_blocks: int = (
            DEFAULT_MIN_OBSERVED_CHECK_BLOCKS
        ),
    ) -> bool:
        """
        Return whether enough declared check blocks were observed.
        """

        minimum = validate_nonnegative_integer(
            minimum_observed_check_blocks,
            field_name=(
                "minimum_observed_check_blocks"
            ),
        )

        return (
            self.observed_check_blocks
            >= minimum
        )

    def to_tuple(
        self,
    ) -> tuple[float, int, int]:
        """
        Return notebook-compatible output.

        Returns:
            qber_raw, mismatches, observed
        """

        return (
            self.qber_raw,
            self.mismatches,
            self.observed,
        )

    def to_dictionary(self) -> dict[str, Any]:
        """
        Return receiver-observable QBER evidence.
        """

        return {
            "qber_raw":
                self.qber_raw,
            "qber_mismatches":
                self.mismatches,
            "qber_observed":
                self.observed,
            "scheduled_check_blocks":
                self.scheduled_check_blocks,
            "observed_check_blocks":
                self.observed_check_blocks,
            "missing_check_blocks":
                self.missing_check_blocks,
            "erased_check_blocks":
                self.erased_check_blocks,
            "skipped_invalid_positions":
                self.skipped_invalid_positions,
            "block_id_mismatches":
                self.block_id_mismatches,
            "length_mismatches":
                self.length_mismatches,
            "has_observations":
                self.has_observations,
        }


@dataclass(frozen=True)
class QBERPolicyDecision:
    """
    Fixed-QBER baseline policy decision.

    FT-QuPAP's full GP mode uses raw QBER as an observable feature.
    This fixed threshold decision is primarily used by comparison
    baselines.
    """

    accepted: bool
    reason: str

    qber_raw: float
    fixed_qber_threshold: float

    observed_check_blocks: int
    minimum_observed_check_blocks: int

    def __post_init__(self) -> None:
        if not isinstance(
            self.accepted,
            bool,
        ):
            raise TypeError(
                "accepted must be boolean."
            )

        valid_reasons = {
            QBER_POLICY_ACCEPTED,
            QBER_POLICY_INSUFFICIENT_EVIDENCE,
            QBER_POLICY_THRESHOLD_EXCEEDED,
        }

        if self.reason not in valid_reasons:
            raise ValueError(
                "Unsupported QBER policy reason."
            )

        validate_qber_value(
            self.qber_raw,
            field_name="qber_raw",
        )

        validate_qber_value(
            self.fixed_qber_threshold,
            field_name="fixed_qber_threshold",
        )

        validate_nonnegative_integer(
            self.observed_check_blocks,
            field_name="observed_check_blocks",
        )

        validate_nonnegative_integer(
            self.minimum_observed_check_blocks,
            field_name=(
                "minimum_observed_check_blocks"
            ),
        )

        sufficient_observations = (
            self.observed_check_blocks
            >= self.minimum_observed_check_blocks
        )

        within_threshold = (
            self.qber_raw
            <= self.fixed_qber_threshold
        )

        expected_accepted = (
            sufficient_observations
            and within_threshold
        )

        if self.accepted != expected_accepted:
            raise ValueError(
                "QBER policy decision is inconsistent."
            )

    @property
    def sufficient_observations(self) -> bool:
        """Return whether the minimum check-block count was met."""

        return (
            self.observed_check_blocks
            >= self.minimum_observed_check_blocks
        )

    @property
    def within_threshold(self) -> bool:
        """Return whether raw QBER is within the fixed threshold."""

        return (
            self.qber_raw
            <= self.fixed_qber_threshold
        )

    def to_dictionary(self) -> dict[str, Any]:
        """Return a serializable policy decision."""

        return {
            "accepted":
                self.accepted,
            "reason":
                self.reason,
            "qber_raw":
                self.qber_raw,
            "fixed_qber_threshold":
                self.fixed_qber_threshold,
            "observed_check_blocks":
                self.observed_check_blocks,
            "minimum_observed_check_blocks":
                self.minimum_observed_check_blocks,
            "sufficient_observations":
                self.sufficient_observations,
            "within_threshold":
                self.within_threshold,
        }


def validate_qber_value(
    value: Any,
    field_name: str = "qber",
) -> float:
    """
    Validate a finite QBER value in the closed interval [0, 1].
    """

    if isinstance(
        value,
        bool,
    ):
        raise InvalidQBERPolicyError(
            f"{field_name} cannot be boolean."
        )

    if not isinstance(
        value,
        (int, float),
    ):
        raise InvalidQBERPolicyError(
            f"{field_name} must be numeric."
        )

    normalized_value = float(
        value
    )

    if not np.isfinite(
        normalized_value
    ):
        raise InvalidQBERPolicyError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= normalized_value <= 1.0:
        raise InvalidQBERPolicyError(
            f"{field_name} must be between zero and one."
        )

    return normalized_value


def validate_nonnegative_integer(
    value: Any,
    field_name: str,
) -> int:
    """
    Validate a nonnegative integer.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return value


def validate_expected_pattern(
    expected_pattern: Any,
) -> np.ndarray:
    """
    Validate an expected physical check pattern.
    """

    try:
        normalized = np.asarray(
            expected_pattern,
            dtype=np.int8,
        )

    except Exception as error:
        raise InvalidExpectedCheckPatternError(
            "expected_reference_bits cannot be "
            "converted to a NumPy array."
        ) from error

    if normalized.ndim != 1:
        raise InvalidExpectedCheckPatternError(
            "expected_reference_bits must be "
            "one-dimensional."
        )

    if len(normalized) == 0:
        raise InvalidExpectedCheckPatternError(
            "expected_reference_bits cannot be empty."
        )

    if not np.all(
        np.isin(
            normalized,
            [0, 1],
        )
    ):
        raise InvalidExpectedCheckPatternError(
            "expected_reference_bits must contain "
            "only 0 or 1."
        )

    return normalized.copy()


def normalize_received_frame(
    received_frame: (
        SteaneEncodedFrame
        | Sequence[PhysicalBlock]
    ),
) -> list[PhysicalBlock]:
    """
    Normalize and validate a received physical frame.
    """

    if isinstance(
        received_frame,
        SteaneEncodedFrame,
    ):
        blocks = list(
            received_frame.frame
        )

    else:
        if isinstance(
            received_frame,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            raise TypeError(
                "received_frame must be a physical-block sequence."
            )

        if not isinstance(
            received_frame,
            Sequence,
        ):
            raise TypeError(
                "received_frame must be a sequence."
            )

        blocks = list(
            received_frame
        )

    if not blocks:
        raise InvalidQBERFrameError(
            "received_frame cannot be empty."
        )

    seen_block_ids: set[str] = set()

    for frame_position, block in enumerate(
        blocks
    ):
        if not isinstance(
            block,
            PhysicalBlock,
        ):
            raise TypeError(
                "Every frame item must be a PhysicalBlock."
            )

        validate_physical_block(
            block
        )

        if block.block_id in seen_block_ids:
            raise InvalidQBERFrameError(
                "Duplicate received block ID: "
                f"{block.block_id!r}."
            )

        seen_block_ids.add(
            block.block_id
        )

        if (
            block.spec.position is not None
            and block.spec.position != frame_position
        ):
            raise InvalidQBERFrameError(
                f"Block {block.block_id!r} is not at "
                "its declared frame position."
            )

    return blocks


def get_check_schedule_entries(
    schedule: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """
    Extract and validate the check-block schedule list.
    """

    if not isinstance(
        schedule,
        Mapping,
    ):
        raise InvalidQBERScheduleError(
            "schedule must be a mapping."
        )

    check_entries = schedule.get(
        "check_blocks",
        [],
    )

    if not isinstance(
        check_entries,
        list,
    ):
        raise InvalidQBERScheduleError(
            "schedule['check_blocks'] must be a list."
        )

    for index, entry in enumerate(
        check_entries
    ):
        if not isinstance(
            entry,
            Mapping,
        ):
            raise InvalidQBERScheduleError(
                f"Check entry {index} must be a mapping."
            )

        required_fields = {
            "position",
            "block_id",
            "basis",
            "expected_reference_bits",
        }

        missing_fields = (
            required_fields.difference(
                entry.keys()
            )
        )

        if missing_fields:
            raise InvalidQBERScheduleError(
                f"Check entry {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

    return check_entries


def normalize_check_position(
    raw_position: Any,
    entry_index: int,
) -> int:
    """
    Convert one check position to an integer.

    The notebook calls int(position), so numeric strings remain
    compatible.
    """

    if isinstance(
        raw_position,
        bool,
    ):
        raise InvalidQBERScheduleError(
            f"Check entry {entry_index} position "
            "cannot be boolean."
        )

    try:
        position = int(
            raw_position
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise InvalidQBERScheduleError(
            f"Check entry {entry_index} contains "
            "an invalid position."
        ) from error

    return position


def calculate_raw_qber_result(
    received_frame: (
        SteaneEncodedFrame
        | Sequence[PhysicalBlock]
    ),
    schedule: Mapping[str, Any],
) -> RawQBERResult:
    """
    Calculate detailed raw QBER from declared check blocks.

    This function preserves the exact notebook counting behavior.
    """

    blocks = normalize_received_frame(
        received_frame
    )

    check_entries = (
        get_check_schedule_entries(
            schedule
        )
    )

    mismatches = 0
    observed = 0

    observed_check_blocks = 0
    erased_check_blocks = 0
    skipped_invalid_positions = 0
    block_id_mismatches = 0
    length_mismatches = 0

    for entry_index, check_entry in enumerate(
        check_entries
    ):
        position = normalize_check_position(
            check_entry["position"],
            entry_index=entry_index,
        )

        # Notebook behavior: positions outside the frame are skipped.
        if (
            position < 0
            or position >= len(blocks)
        ):
            skipped_invalid_positions += 1
            continue

        expected = validate_expected_pattern(
            check_entry[
                "expected_reference_bits"
            ]
        )

        expected_block_id = check_entry[
            "block_id"
        ]

        if not isinstance(
            expected_block_id,
            str,
        ) or not expected_block_id:
            raise InvalidQBERScheduleError(
                f"Check entry {entry_index} contains "
                "an invalid block_id."
            )

        block = blocks[
            position
        ]

        # Notebook behavior: wrong block at a declared position
        # contributes a complete expected-pattern mismatch.
        if block.block_id != expected_block_id:
            expected_length = len(
                expected
            )

            mismatches += expected_length
            observed += expected_length

            observed_check_blocks += 1
            block_id_mismatches += 1
            continue

        measurement = (
            measure_block_in_declared_basis(
                block=block,
                declared_basis=(
                    check_entry["basis"]
                ),
            )
        )

        # Notebook behavior: erased blocks are excluded.
        if measurement is None:
            erased_check_blocks += 1
            continue

        measurement = np.asarray(
            measurement,
            dtype=np.int8,
        )

        if measurement.ndim != 1:
            raise QBERCalculationError(
                f"Measurement for block "
                f"{block.block_id!r} is not one-dimensional."
            )

        if len(measurement) != len(
            expected
        ):
            comparison_length = max(
                len(measurement),
                len(expected),
            )

            mismatches += comparison_length
            observed += comparison_length

            observed_check_blocks += 1
            length_mismatches += 1
            continue

        block_mismatches = int(
            np.sum(
                measurement
                != expected
            )
        )

        mismatches += block_mismatches
        observed += len(
            expected
        )

        observed_check_blocks += 1

    qber_raw = (
        mismatches / observed
        if observed > 0
        else 1.0
    )

    return RawQBERResult(
        qber_raw=float(
            qber_raw
        ),
        mismatches=int(
            mismatches
        ),
        observed=int(
            observed
        ),
        scheduled_check_blocks=len(
            check_entries
        ),
        observed_check_blocks=(
            observed_check_blocks
        ),
        erased_check_blocks=(
            erased_check_blocks
        ),
        skipped_invalid_positions=(
            skipped_invalid_positions
        ),
        block_id_mismatches=(
            block_id_mismatches
        ),
        length_mismatches=(
            length_mismatches
        ),
    )


def calculate_raw_qber(
    received_frame: (
        SteaneEncodedFrame
        | Sequence[PhysicalBlock]
    ),
    schedule: Mapping[str, Any],
) -> tuple[float, int, int]:
    """
    Calculate notebook-compatible raw QBER.

    Returns:
        qber_raw:
            Mismatches divided by observed physical check bits.

        mismatches:
            Number of mismatching observed physical bits.

        observed:
            Number of physical check bits in the denominator.
    """

    return calculate_raw_qber_result(
        received_frame=received_frame,
        schedule=schedule,
    ).to_tuple()


def observed_check_blocks_from_bits(
    observed_physical_check_bits: int,
    physical_qubits_per_check_block: int,
) -> int:
    """
    Reproduce the notebook's observed-check-block calculation.

    Notebook operation:

        observed_check_blocks =
            qber_observed //
            physical_qubits_per_check_block
    """

    observed_bits = validate_nonnegative_integer(
        observed_physical_check_bits,
        field_name=(
            "observed_physical_check_bits"
        ),
    )

    qubits_per_block = (
        validate_nonnegative_integer(
            physical_qubits_per_check_block,
            field_name=(
                "physical_qubits_per_check_block"
            ),
        )
    )

    if qubits_per_block == 0:
        raise ValueError(
            "physical_qubits_per_check_block "
            "must be greater than zero."
        )

    return int(
        observed_bits
        // qubits_per_block
    )


def required_observed_physical_bits(
    minimum_observed_check_blocks: int = (
        DEFAULT_MIN_OBSERVED_CHECK_BLOCKS
    ),
    physical_qubits_per_check_block: int = (
        STEANE_BLOCK_SIZE
    ),
) -> int:
    """
    Return the minimum physical check-bit observation count.

    Standard CSS session:

        24 blocks × 7 physical bits = 168 observed bits
    """

    minimum_blocks = validate_nonnegative_integer(
        minimum_observed_check_blocks,
        field_name=(
            "minimum_observed_check_blocks"
        ),
    )

    qubits_per_block = (
        validate_nonnegative_integer(
            physical_qubits_per_check_block,
            field_name=(
                "physical_qubits_per_check_block"
            ),
        )
    )

    if qubits_per_block == 0:
        raise ValueError(
            "physical_qubits_per_check_block "
            "must be greater than zero."
        )

    return (
        minimum_blocks
        * qubits_per_block
    )


def has_sufficient_qber_evidence(
    qber_result: RawQBERResult,
    minimum_observed_check_blocks: int = (
        DEFAULT_MIN_OBSERVED_CHECK_BLOCKS
    ),
) -> bool:
    """
    Return whether enough check blocks contributed QBER evidence.
    """

    if not isinstance(
        qber_result,
        RawQBERResult,
    ):
        raise TypeError(
            "qber_result must be RawQBERResult."
        )

    return qber_result.has_sufficient_check_blocks(
        minimum_observed_check_blocks
    )


def evaluate_fixed_qber_policy(
    qber: RawQBERResult | float,
    observed_check_blocks: int | None = None,
    fixed_qber_threshold: float = (
        DEFAULT_FIXED_QBER_THRESHOLD
    ),
    minimum_observed_check_blocks: int = (
        DEFAULT_MIN_OBSERVED_CHECK_BLOCKS
    ),
) -> QBERPolicyDecision:
    """
    Apply the fixed-QBER comparison policy.

    Full FT-QuPAP GP mode does not rely on this threshold alone.
    """

    threshold = validate_qber_value(
        fixed_qber_threshold,
        field_name=(
            "fixed_qber_threshold"
        ),
    )

    minimum_blocks = validate_nonnegative_integer(
        minimum_observed_check_blocks,
        field_name=(
            "minimum_observed_check_blocks"
        ),
    )

    if isinstance(
        qber,
        RawQBERResult,
    ):
        qber_raw = qber.qber_raw
        actual_observed_blocks = (
            qber.observed_check_blocks
        )

    else:
        qber_raw = validate_qber_value(
            qber,
            field_name="qber",
        )

        if observed_check_blocks is None:
            raise InvalidQBERPolicyError(
                "observed_check_blocks is required "
                "when qber is supplied as a number."
            )

        actual_observed_blocks = (
            validate_nonnegative_integer(
                observed_check_blocks,
                field_name=(
                    "observed_check_blocks"
                ),
            )
        )

    sufficient_observations = (
        actual_observed_blocks
        >= minimum_blocks
    )

    within_threshold = (
        qber_raw <= threshold
    )

    accepted = (
        sufficient_observations
        and within_threshold
    )

    if not sufficient_observations:
        reason = (
            QBER_POLICY_INSUFFICIENT_EVIDENCE
        )

    elif not within_threshold:
        reason = (
            QBER_POLICY_THRESHOLD_EXCEEDED
        )

    else:
        reason = (
            QBER_POLICY_ACCEPTED
        )

    return QBERPolicyDecision(
        accepted=accepted,
        reason=reason,
        qber_raw=qber_raw,
        fixed_qber_threshold=threshold,
        observed_check_blocks=(
            actual_observed_blocks
        ),
        minimum_observed_check_blocks=(
            minimum_blocks
        ),
    )


def calculate_mismatch_rate(
    measured_bits: Sequence[int],
    expected_bits: Sequence[int],
) -> tuple[float, int, int]:
    """
    Calculate mismatch rate between two physical bit sequences.

    A length mismatch follows the notebook rule and counts the larger
    sequence length as both mismatches and observations.
    """

    measured = validate_expected_pattern(
        measured_bits
    )

    expected = validate_expected_pattern(
        expected_bits
    )

    if len(measured) != len(
        expected
    ):
        observed = max(
            len(measured),
            len(expected),
        )

        return (
            1.0,
            observed,
            observed,
        )

    mismatches = int(
        np.sum(
            measured != expected
        )
    )

    observed = len(
        expected
    )

    qber = (
        mismatches / observed
        if observed > 0
        else 1.0
    )

    return (
        float(qber),
        mismatches,
        observed,
    )


def run_self_test() -> None:
    """
    Verify ideal, noisy, erased, mismatch, and empty-schedule cases.
    """

    import copy

    from .logical_qubit import (
        create_check_logical_qubit,
    )
    from .steane_css import (
        encode_one_logical_qubit,
    )

    print("=" * 72)
    print("FT-QuPAP Raw QBER Calculator Self-Test")
    print("=" * 72)

    check_specs = [
        create_check_logical_qubit(
            logical_bit=0,
            basis="Z",
            logical_index=0,
            position=0,
        ),
        create_check_logical_qubit(
            logical_bit=1,
            basis="X",
            logical_index=1,
            position=1,
        ),
        create_check_logical_qubit(
            logical_bit=1,
            basis="Z",
            logical_index=2,
            position=2,
        ),
        create_check_logical_qubit(
            logical_bit=0,
            basis="X",
            logical_index=3,
            position=3,
        ),
    ]

    ideal_frame = [
        encode_one_logical_qubit(
            spec=spec,
            use_css=True,
            rng=np.random.default_rng(
                10000 + index
            ),
        )
        for index, spec in enumerate(
            check_specs
        )
    ]

    schedule = {
        "check_blocks": [
            {
                "block_id":
                    block.block_id,
                "position":
                    position,
                "basis":
                    block.basis,
                "expected_reference_bits":
                    block.reference_bits.tolist(),
            }
            for position, block in enumerate(
                ideal_frame
            )
        ],
        "payload_blocks": [],
    }

    ideal_result = (
        calculate_raw_qber_result(
            received_frame=ideal_frame,
            schedule=schedule,
        )
    )

    noisy_frame = copy.deepcopy(
        ideal_frame
    )

    # Z-basis check: physical X error is observable.
    noisy_frame[0].x_errors[2] = 1

    # X-basis check: physical Z error is observable.
    noisy_frame[1].z_errors[5] = 1

    noisy_result = (
        calculate_raw_qber_result(
            received_frame=noisy_frame,
            schedule=schedule,
        )
    )

    erased_frame = copy.deepcopy(
        ideal_frame
    )

    erased_frame[2].erasures[4] = True

    erased_result = (
        calculate_raw_qber_result(
            received_frame=erased_frame,
            schedule=schedule,
        )
    )

    mismatch_schedule = copy.deepcopy(
        schedule
    )

    mismatch_schedule[
        "check_blocks"
    ][0]["block_id"] = "C9999"

    block_id_result = (
        calculate_raw_qber_result(
            received_frame=ideal_frame,
            schedule=mismatch_schedule,
        )
    )

    invalid_position_schedule = (
        copy.deepcopy(
            schedule
        )
    )

    invalid_position_schedule[
        "check_blocks"
    ][0]["position"] = 999

    invalid_position_result = (
        calculate_raw_qber_result(
            received_frame=ideal_frame,
            schedule=(
                invalid_position_schedule
            ),
        )
    )

    empty_result = (
        calculate_raw_qber_result(
            received_frame=ideal_frame,
            schedule={
                "check_blocks": [],
            },
        )
    )

    noisy_policy = (
        evaluate_fixed_qber_policy(
            noisy_result,
            fixed_qber_threshold=0.11,
            minimum_observed_check_blocks=4,
        )
    )

    erased_policy = (
        evaluate_fixed_qber_policy(
            erased_result,
            fixed_qber_threshold=0.11,
            minimum_observed_check_blocks=4,
        )
    )

    notebook_tuple = calculate_raw_qber(
        received_frame=noisy_frame,
        schedule=schedule,
    )

    ideal_correct = (
        ideal_result.qber_raw == 0.0
        and ideal_result.mismatches == 0
        and ideal_result.observed == 28
        and ideal_result.observed_check_blocks == 4
    )

    noisy_correct = (
        noisy_result.mismatches == 2
        and noisy_result.observed == 28
        and np.isclose(
            noisy_result.qber_raw,
            2 / 28,
            rtol=0.0,
            atol=1e-12,
        )
    )

    erased_excluded = (
        erased_result.observed == 21
        and erased_result.observed_check_blocks == 3
        and erased_result.erased_check_blocks == 1
    )

    block_id_full_mismatch = (
        block_id_result.mismatches == 7
        and block_id_result.observed == 28
        and block_id_result.block_id_mismatches == 1
    )

    invalid_position_skipped = (
        invalid_position_result.observed == 21
        and invalid_position_result
        .skipped_invalid_positions
        == 1
    )

    empty_defaults_to_one = (
        empty_result.qber_raw == 1.0
        and empty_result.observed == 0
    )

    notebook_tuple_correct = (
        notebook_tuple
        == noisy_result.to_tuple()
    )

    print(
        "Ideal QBER                : "
        f"{ideal_result.qber_raw:.6f}"
    )

    print(
        "Noisy mismatches          : "
        f"{noisy_result.mismatches}"
    )

    print(
        "Noisy observed bits       : "
        f"{noisy_result.observed}"
    )

    print(
        "Noisy raw QBER            : "
        f"{noisy_result.qber_raw:.6f}"
    )

    print(
        "Erased block excluded     : "
        f"{erased_excluded}"
    )

    print(
        "ID mismatch counted fully : "
        f"{block_id_full_mismatch}"
    )

    print(
        "Invalid position skipped  : "
        f"{invalid_position_skipped}"
    )

    print(
        "Empty schedule gives 1.0  : "
        f"{empty_defaults_to_one}"
    )

    print(
        "Noisy fixed policy accepts: "
        f"{noisy_policy.accepted}"
    )

    print(
        "Erased evidence rejected  : "
        f"{not erased_policy.accepted}"
    )

    print(
        "Notebook tuple compatible : "
        f"{notebook_tuple_correct}"
    )

    if not ideal_correct:
        raise QBERCalculationError(
            "Ideal QBER calculation failed."
        )

    if not noisy_correct:
        raise QBERCalculationError(
            "Noisy QBER calculation failed."
        )

    if not erased_excluded:
        raise QBERCalculationError(
            "Erased check block was not excluded correctly."
        )

    if not block_id_full_mismatch:
        raise QBERCalculationError(
            "Block-ID mismatch counting failed."
        )

    if not invalid_position_skipped:
        raise QBERCalculationError(
            "Invalid check position was not skipped."
        )

    if not empty_defaults_to_one:
        raise QBERCalculationError(
            "Empty observation case did not return QBER 1.0."
        )

    if not noisy_policy.accepted:
        raise QBERCalculationError(
            "Valid noisy case failed the fixed-QBER policy."
        )

    if erased_policy.accepted:
        raise QBERCalculationError(
            "Insufficient check evidence was accepted."
        )

    if erased_policy.reason != (
        QBER_POLICY_INSUFFICIENT_EVIDENCE
    ):
        raise QBERCalculationError(
            "Incorrect insufficient-evidence reason."
        )

    if not notebook_tuple_correct:
        raise QBERCalculationError(
            "Notebook-compatible tuple output failed."
        )

    print(
        "\nRaw QBER calculator self-test "
        "completed successfully."
    )


__all__ = [
    "DEFAULT_CHECK_BLOCK_COUNT",
    "DEFAULT_MIN_OBSERVED_CHECK_BLOCKS",
    "DEFAULT_FIXED_QBER_THRESHOLD",
    "QBER_POLICY_ACCEPTED",
    "QBER_POLICY_INSUFFICIENT_EVIDENCE",
    "QBER_POLICY_THRESHOLD_EXCEEDED",
    "QBERCalculationError",
    "InvalidQBERFrameError",
    "InvalidQBERScheduleError",
    "InvalidExpectedCheckPatternError",
    "InvalidQBERPolicyError",
    "RawQBERResult",
    "QBERPolicyDecision",
    "validate_qber_value",
    "validate_nonnegative_integer",
    "validate_expected_pattern",
    "normalize_received_frame",
    "get_check_schedule_entries",
    "calculate_raw_qber_result",
    "calculate_raw_qber",
    "observed_check_blocks_from_bits",
    "required_observed_physical_bits",
    "has_sufficient_qber_evidence",
    "evaluate_fixed_qber_policy",
    "calculate_mismatch_rate",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        QBERCalculationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[QBER CALCULATION ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error