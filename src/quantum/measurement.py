"""
FT-QuPAP Quantum Measurement Module

This module models Authentication Server measurements of received
FT-QuPAP physical blocks.

Protocol behavior:

1. The Authentication Server decrypts and validates the control
   schedule.
2. Check blocks are measured in the basis declared by that protected
   schedule.
3. Payload blocks use the Z basis.
4. A block containing one or more erasures produces no valid
   measurement.
5. Z-basis measurements are affected by physical X errors.
6. X-basis measurements are affected by physical Z errors.
7. Raw measurements are produced before CSS correction.
8. Raw check measurements are later used by qber_calculator.py.

The notebook-compatible core operation is:

    measurement =
        reference_bits XOR x_errors     for Z-basis measurement

    measurement =
        reference_bits XOR z_errors     for X-basis measurement

Security boundary:

Measurement functions use only receiver-visible block state and the
authenticated control schedule. Hidden Eve settings and attacked-mask
values are never returned as Authentication Server observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .steane_css import (
    PhysicalBlock,
    SteaneEncodedFrame,
    validate_physical_block,
)


SUPPORTED_MEASUREMENT_BASES = (
    "Z",
    "X",
)

MEASUREMENT_STATUS_OBSERVED = "observed"
MEASUREMENT_STATUS_ERASED = "erased"
MEASUREMENT_STATUS_INVALID_POSITION = "invalid_position"
MEASUREMENT_STATUS_BLOCK_ID_MISMATCH = "block_id_mismatch"
MEASUREMENT_STATUS_INVALID_EXPECTED_PATTERN = (
    "invalid_expected_pattern"
)

PAYLOAD_ROLE = "payload"
CHECK_ROLE = "check"


class QuantumMeasurementError(Exception):
    """Base exception for FT-QuPAP measurement failures."""


class InvalidMeasurementBasisError(
    QuantumMeasurementError
):
    """Raised when a measurement basis is invalid."""


class InvalidMeasurementFrameError(
    QuantumMeasurementError
):
    """Raised when a received physical frame is invalid."""


class InvalidMeasurementScheduleError(
    QuantumMeasurementError
):
    """Raised when the protected measurement schedule is malformed."""


class InvalidExpectedPatternError(
    QuantumMeasurementError
):
    """Raised when an expected physical pattern is invalid."""


@dataclass(frozen=True)
class PhysicalMeasurementResult:
    """
    Raw measurement result for one physical block.

    Attributes:
        block_id:
            Logical block identifier.

        role:
            Either payload or check.

        frame_position:
            Position of the block in the received frame.

        measurement_basis:
            Basis selected through the authenticated schedule.

        status:
            observed or erased.

        measured_bits:
            Raw physical measurement bits. None for erased blocks.

        physical_position_count:
            Number of physical positions in the block.

        erasure_count:
            Number of erased physical positions.

        use_css:
            True for Steane encoding and False for no-CSS baseline.
    """

    block_id: str
    role: str
    frame_position: int
    measurement_basis: str
    status: str
    measured_bits: tuple[int, ...] | None
    physical_position_count: int
    erasure_count: int
    use_css: bool

    def __post_init__(self) -> None:
        if not isinstance(
            self.block_id,
            str,
        ):
            raise TypeError(
                "block_id must be a string."
            )

        if not self.block_id:
            raise ValueError(
                "block_id cannot be empty."
            )

        if self.role not in (
            PAYLOAD_ROLE,
            CHECK_ROLE,
        ):
            raise ValueError(
                "role must be payload or check."
            )

        if (
            isinstance(
                self.frame_position,
                bool,
            )
            or not isinstance(
                self.frame_position,
                int,
            )
        ):
            raise TypeError(
                "frame_position must be an integer."
            )

        if self.frame_position < 0:
            raise ValueError(
                "frame_position cannot be negative."
            )

        validate_measurement_basis(
            self.measurement_basis
        )

        if self.status not in (
            MEASUREMENT_STATUS_OBSERVED,
            MEASUREMENT_STATUS_ERASED,
        ):
            raise ValueError(
                "Unsupported measurement status."
            )

        if (
            isinstance(
                self.physical_position_count,
                bool,
            )
            or not isinstance(
                self.physical_position_count,
                int,
            )
        ):
            raise TypeError(
                "physical_position_count must be an integer."
            )

        if self.physical_position_count <= 0:
            raise ValueError(
                "physical_position_count must be positive."
            )

        if (
            isinstance(
                self.erasure_count,
                bool,
            )
            or not isinstance(
                self.erasure_count,
                int,
            )
        ):
            raise TypeError(
                "erasure_count must be an integer."
            )

        if not 0 <= self.erasure_count <= (
            self.physical_position_count
        ):
            raise ValueError(
                "erasure_count is outside the block."
            )

        if not isinstance(
            self.use_css,
            bool,
        ):
            raise TypeError(
                "use_css must be boolean."
            )

        if self.status == MEASUREMENT_STATUS_ERASED:
            if self.erasure_count == 0:
                raise ValueError(
                    "Erased result must contain an erasure."
                )

            if self.measured_bits is not None:
                raise ValueError(
                    "Erased result cannot contain measured bits."
                )

        if self.status == MEASUREMENT_STATUS_OBSERVED:
            if self.erasure_count != 0:
                raise ValueError(
                    "Observed result cannot contain erasures."
                )

            if self.measured_bits is None:
                raise ValueError(
                    "Observed result must contain measured bits."
                )

            validate_binary_vector(
                vector=self.measured_bits,
                field_name="measured_bits",
                expected_length=(
                    self.physical_position_count
                ),
            )

    @property
    def observed(self) -> bool:
        """Return whether valid measurement bits are available."""

        return (
            self.status
            == MEASUREMENT_STATUS_OBSERVED
        )

    @property
    def erased(self) -> bool:
        """Return whether measurement failed due to erasure."""

        return (
            self.status
            == MEASUREMENT_STATUS_ERASED
        )

    def to_dictionary(self) -> dict[str, Any]:
        """
        Return a serializable receiver-visible measurement record.
        """

        return {
            "block_id":
                self.block_id,
            "role":
                self.role,
            "frame_position":
                self.frame_position,
            "measurement_basis":
                self.measurement_basis,
            "status":
                self.status,
            "observed":
                self.observed,
            "measured_bits": (
                list(self.measured_bits)
                if self.measured_bits is not None
                else None
            ),
            "physical_position_count":
                self.physical_position_count,
            "erasure_count":
                self.erasure_count,
            "use_css":
                self.use_css,
        }


@dataclass(frozen=True)
class ScheduledCheckMeasurement:
    """
    Measurement result for one declared check schedule entry.

    Attributes:
        schedule_index:
            Original index inside schedule["check_blocks"].

        expected_block_id:
            Block ID declared by the protected schedule.

        frame_position:
            Declared frame position.

        measurement_basis:
            Declared Z or X basis.

        expected_reference_bits:
            Expected physical check pattern protected by K_ctrl.

        status:
            observed, erased, invalid_position, block_id_mismatch,
            or invalid_expected_pattern.

        received_block_id:
            Actual block found at the declared frame position.

        measured_bits:
            Raw measurement bits when observation succeeds.
    """

    schedule_index: int
    expected_block_id: str
    frame_position: int
    measurement_basis: str
    expected_reference_bits: tuple[int, ...] | None
    status: str
    received_block_id: str | None
    measured_bits: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.schedule_index,
                bool,
            )
            or not isinstance(
                self.schedule_index,
                int,
            )
        ):
            raise TypeError(
                "schedule_index must be an integer."
            )

        if self.schedule_index < 0:
            raise ValueError(
                "schedule_index cannot be negative."
            )

        if not isinstance(
            self.expected_block_id,
            str,
        ):
            raise TypeError(
                "expected_block_id must be a string."
            )

        if not self.expected_block_id:
            raise ValueError(
                "expected_block_id cannot be empty."
            )

        if (
            isinstance(
                self.frame_position,
                bool,
            )
            or not isinstance(
                self.frame_position,
                int,
            )
        ):
            raise TypeError(
                "frame_position must be an integer."
            )

        validate_measurement_basis(
            self.measurement_basis
        )

        valid_statuses = {
            MEASUREMENT_STATUS_OBSERVED,
            MEASUREMENT_STATUS_ERASED,
            MEASUREMENT_STATUS_INVALID_POSITION,
            MEASUREMENT_STATUS_BLOCK_ID_MISMATCH,
            MEASUREMENT_STATUS_INVALID_EXPECTED_PATTERN,
        }

        if self.status not in valid_statuses:
            raise ValueError(
                "Unsupported scheduled-measurement status."
            )

        if self.expected_reference_bits is not None:
            validate_binary_vector(
                vector=self.expected_reference_bits,
                field_name="expected_reference_bits",
                expected_length=len(
                    self.expected_reference_bits
                ),
            )

        if self.measured_bits is not None:
            validate_binary_vector(
                vector=self.measured_bits,
                field_name="measured_bits",
                expected_length=len(
                    self.measured_bits
                ),
            )

    @property
    def observed(self) -> bool:
        """Return whether a valid raw measurement is available."""

        return (
            self.status
            == MEASUREMENT_STATUS_OBSERVED
            and self.measured_bits is not None
        )

    @property
    def expected_physical_count(self) -> int:
        """Return expected physical-pattern length."""

        if self.expected_reference_bits is None:
            return 0

        return len(
            self.expected_reference_bits
        )

    def to_dictionary(self) -> dict[str, Any]:
        """Return a serializable scheduled measurement."""

        return {
            "schedule_index":
                self.schedule_index,
            "expected_block_id":
                self.expected_block_id,
            "received_block_id":
                self.received_block_id,
            "frame_position":
                self.frame_position,
            "measurement_basis":
                self.measurement_basis,
            "expected_reference_bits": (
                list(
                    self.expected_reference_bits
                )
                if self.expected_reference_bits
                is not None
                else None
            ),
            "status":
                self.status,
            "observed":
                self.observed,
            "measured_bits": (
                list(self.measured_bits)
                if self.measured_bits is not None
                else None
            ),
        }


@dataclass(frozen=True)
class MeasurementSummary:
    """
    Aggregate declared-check measurement statistics.
    """

    scheduled_check_blocks: int
    observed_check_blocks: int
    erased_check_blocks: int
    invalid_position_entries: int
    block_id_mismatches: int
    invalid_expected_patterns: int
    observed_physical_positions: int
    expected_physical_positions: int

    def __post_init__(self) -> None:
        fields = {
            "scheduled_check_blocks":
                self.scheduled_check_blocks,
            "observed_check_blocks":
                self.observed_check_blocks,
            "erased_check_blocks":
                self.erased_check_blocks,
            "invalid_position_entries":
                self.invalid_position_entries,
            "block_id_mismatches":
                self.block_id_mismatches,
            "invalid_expected_patterns":
                self.invalid_expected_patterns,
            "observed_physical_positions":
                self.observed_physical_positions,
            "expected_physical_positions":
                self.expected_physical_positions,
        }

        for field_name, value in fields.items():
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

    @property
    def observation_rate(self) -> float:
        """Return observed declared-check block rate."""

        if self.scheduled_check_blocks == 0:
            return 0.0

        return float(
            self.observed_check_blocks
            / self.scheduled_check_blocks
        )

    def to_dictionary(self) -> dict[str, Any]:
        """Return aggregate measurement statistics."""

        return {
            "scheduled_check_blocks":
                self.scheduled_check_blocks,
            "observed_check_blocks":
                self.observed_check_blocks,
            "erased_check_blocks":
                self.erased_check_blocks,
            "invalid_position_entries":
                self.invalid_position_entries,
            "block_id_mismatches":
                self.block_id_mismatches,
            "invalid_expected_patterns":
                self.invalid_expected_patterns,
            "observed_physical_positions":
                self.observed_physical_positions,
            "expected_physical_positions":
                self.expected_physical_positions,
            "observation_rate":
                self.observation_rate,
        }


def validate_measurement_basis(
    measurement_basis: Any,
) -> str:
    """
    Validate and normalize a Z- or X-basis label.
    """

    if not isinstance(
        measurement_basis,
        str,
    ):
        raise InvalidMeasurementBasisError(
            "measurement_basis must be a string."
        )

    normalized_basis = (
        measurement_basis.strip().upper()
    )

    if normalized_basis not in (
        SUPPORTED_MEASUREMENT_BASES
    ):
        raise InvalidMeasurementBasisError(
            "measurement_basis must be Z or X."
        )

    return normalized_basis


def validate_binary_vector(
    vector: Any,
    field_name: str,
    expected_length: int,
) -> np.ndarray:
    """
    Validate and normalize a one-dimensional binary vector.
    """

    if (
        isinstance(
            expected_length,
            bool,
        )
        or not isinstance(
            expected_length,
            int,
        )
    ):
        raise TypeError(
            "expected_length must be an integer."
        )

    if expected_length <= 0:
        raise ValueError(
            "expected_length must be positive."
        )

    try:
        normalized = np.asarray(
            vector,
            dtype=np.int8,
        )

    except Exception as error:
        raise InvalidExpectedPatternError(
            f"{field_name} cannot be converted "
            "to a binary array."
        ) from error

    if normalized.ndim != 1:
        raise InvalidExpectedPatternError(
            f"{field_name} must be one-dimensional."
        )

    if len(normalized) != expected_length:
        raise InvalidExpectedPatternError(
            f"{field_name} must contain exactly "
            f"{expected_length} bits."
        )

    if not np.all(
        np.isin(
            normalized,
            [0, 1],
        )
    ):
        raise InvalidExpectedPatternError(
            f"{field_name} must contain only 0 or 1."
        )

    return normalized.copy()


def normalize_received_frame(
    received_frame: (
        SteaneEncodedFrame
        | Sequence[PhysicalBlock]
    ),
) -> list[PhysicalBlock]:
    """
    Normalize a Steane frame or physical-block sequence.
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
        raise InvalidMeasurementFrameError(
            "received_frame cannot be empty."
        )

    seen_ids: set[str] = set()

    for expected_position, block in enumerate(
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

        if block.block_id in seen_ids:
            raise InvalidMeasurementFrameError(
                "Duplicate physical block ID: "
                f"{block.block_id!r}."
            )

        seen_ids.add(
            block.block_id
        )

        if (
            block.spec.position is not None
            and block.spec.position
            != expected_position
        ):
            raise InvalidMeasurementFrameError(
                f"Block {block.block_id!r} "
                "is not at its declared frame position."
            )

    return blocks


def measure_block_in_declared_basis(
    block: PhysicalBlock,
    declared_basis: str,
) -> np.ndarray | None:
    """
    Measure one physical block in the declared basis.

    Notebook-compatible behavior:

        Z basis:
            reference_bits XOR x_errors

        X basis:
            reference_bits XOR z_errors

    Any erasure causes the function to return None.
    """

    if not isinstance(
        block,
        PhysicalBlock,
    ):
        raise TypeError(
            "block must be a PhysicalBlock."
        )

    validate_physical_block(
        block
    )

    normalized_basis = (
        validate_measurement_basis(
            declared_basis
        )
    )

    if np.any(
        block.erasures
    ):
        return None

    relevant_errors = (
        block.x_errors
        if normalized_basis == "Z"
        else block.z_errors
    )

    measurement = (
        block.reference_bits
        ^ relevant_errors
    ).astype(
        np.int8
    )

    return measurement


def measure_physical_block(
    block: PhysicalBlock,
    measurement_basis: str,
    frame_position: int | None = None,
) -> PhysicalMeasurementResult:
    """
    Measure one block and return a structured result.
    """

    if not isinstance(
        block,
        PhysicalBlock,
    ):
        raise TypeError(
            "block must be a PhysicalBlock."
        )

    validate_physical_block(
        block
    )

    normalized_basis = (
        validate_measurement_basis(
            measurement_basis
        )
    )

    selected_position = (
        frame_position
        if frame_position is not None
        else block.spec.position
    )

    if selected_position is None:
        raise InvalidMeasurementFrameError(
            "Block has no frame position."
        )

    if (
        isinstance(
            selected_position,
            bool,
        )
        or not isinstance(
            selected_position,
            int,
        )
    ):
        raise TypeError(
            "frame_position must be an integer."
        )

    if selected_position < 0:
        raise ValueError(
            "frame_position cannot be negative."
        )

    erasure_count = int(
        np.count_nonzero(
            block.erasures
        )
    )

    measured_bits = (
        measure_block_in_declared_basis(
            block=block,
            declared_basis=(
                normalized_basis
            ),
        )
    )

    if measured_bits is None:
        return PhysicalMeasurementResult(
            block_id=block.block_id,
            role=block.role,
            frame_position=selected_position,
            measurement_basis=(
                normalized_basis
            ),
            status=(
                MEASUREMENT_STATUS_ERASED
            ),
            measured_bits=None,
            physical_position_count=(
                block.physical_qubit_count
            ),
            erasure_count=(
                erasure_count
            ),
            use_css=block.use_css,
        )

    return PhysicalMeasurementResult(
        block_id=block.block_id,
        role=block.role,
        frame_position=selected_position,
        measurement_basis=(
            normalized_basis
        ),
        status=(
            MEASUREMENT_STATUS_OBSERVED
        ),
        measured_bits=tuple(
            int(bit)
            for bit in measured_bits.tolist()
        ),
        physical_position_count=(
            block.physical_qubit_count
        ),
        erasure_count=0,
        use_css=block.use_css,
    )


def validate_check_schedule_entry(
    check_entry: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Validate one protected check-block schedule entry.

    Required fields:

        block_id
        position
        basis
        expected_reference_bits
    """

    if not isinstance(
        check_entry,
        Mapping,
    ):
        raise InvalidMeasurementScheduleError(
            "Check schedule entry must be a mapping."
        )

    required_fields = {
        "block_id",
        "position",
        "basis",
        "expected_reference_bits",
    }

    missing_fields = (
        required_fields.difference(
            check_entry.keys()
        )
    )

    if missing_fields:
        raise InvalidMeasurementScheduleError(
            "Check schedule entry is missing fields: "
            f"{sorted(missing_fields)}"
        )

    block_id = check_entry[
        "block_id"
    ]

    position = check_entry[
        "position"
    ]

    basis = check_entry[
        "basis"
    ]

    expected_reference_bits = (
        check_entry[
            "expected_reference_bits"
        ]
    )

    if not isinstance(
        block_id,
        str,
    ) or not block_id:
        raise InvalidMeasurementScheduleError(
            "Check block_id must be a non-empty string."
        )

    if (
        isinstance(
            position,
            bool,
        )
        or not isinstance(
            position,
            int,
        )
    ):
        raise InvalidMeasurementScheduleError(
            "Check position must be an integer."
        )

    normalized_basis = (
        validate_measurement_basis(
            basis
        )
    )

    try:
        expected_array = np.asarray(
            expected_reference_bits,
            dtype=np.int8,
        )

    except Exception as error:
        raise InvalidExpectedPatternError(
            "expected_reference_bits cannot "
            "be converted to an array."
        ) from error

    if expected_array.ndim != 1:
        raise InvalidExpectedPatternError(
            "expected_reference_bits must "
            "be one-dimensional."
        )

    if len(expected_array) == 0:
        raise InvalidExpectedPatternError(
            "expected_reference_bits cannot be empty."
        )

    expected_array = validate_binary_vector(
        vector=expected_array,
        field_name="expected_reference_bits",
        expected_length=len(
            expected_array
        ),
    )

    return {
        **dict(
            check_entry
        ),
        "block_id":
            block_id,
        "position":
            position,
        "basis":
            normalized_basis,
        "expected_reference_bits":
            expected_array,
    }


def measure_scheduled_check_block(
    received_frame: (
        SteaneEncodedFrame
        | Sequence[PhysicalBlock]
    ),
    check_entry: Mapping[str, Any],
    schedule_index: int = 0,
) -> ScheduledCheckMeasurement:
    """
    Measure one declared check-block schedule entry.

    Invalid positions and block-ID mismatches are retained as explicit
    records. The QBER calculator later treats structural mismatches as
    full physical-pattern errors.
    """

    blocks = normalize_received_frame(
        received_frame
    )

    if (
        isinstance(
            schedule_index,
            bool,
        )
        or not isinstance(
            schedule_index,
            int,
        )
    ):
        raise TypeError(
            "schedule_index must be an integer."
        )

    if schedule_index < 0:
        raise ValueError(
            "schedule_index cannot be negative."
        )

    try:
        normalized_entry = (
            validate_check_schedule_entry(
                check_entry
            )
        )

    except InvalidExpectedPatternError:
        block_id = str(
            check_entry.get(
                "block_id",
                "",
            )
        )

        raw_position = check_entry.get(
            "position",
            -1,
        )

        position = (
            int(raw_position)
            if isinstance(
                raw_position,
                int,
            )
            and not isinstance(
                raw_position,
                bool,
            )
            else -1
        )

        basis = str(
            check_entry.get(
                "basis",
                "Z",
            )
        ).upper()

        if basis not in (
            SUPPORTED_MEASUREMENT_BASES
        ):
            basis = "Z"

        return ScheduledCheckMeasurement(
            schedule_index=schedule_index,
            expected_block_id=(
                block_id or "UNKNOWN"
            ),
            frame_position=position,
            measurement_basis=basis,
            expected_reference_bits=None,
            status=(
                MEASUREMENT_STATUS_INVALID_EXPECTED_PATTERN
            ),
            received_block_id=None,
            measured_bits=None,
        )

    position = int(
        normalized_entry[
            "position"
        ]
    )

    expected_block_id = str(
        normalized_entry[
            "block_id"
        ]
    )

    basis = str(
        normalized_entry[
            "basis"
        ]
    )

    expected_bits_array = np.asarray(
        normalized_entry[
            "expected_reference_bits"
        ],
        dtype=np.int8,
    )

    expected_bits = tuple(
        int(bit)
        for bit in expected_bits_array.tolist()
    )

    if not 0 <= position < len(
        blocks
    ):
        return ScheduledCheckMeasurement(
            schedule_index=schedule_index,
            expected_block_id=(
                expected_block_id
            ),
            frame_position=position,
            measurement_basis=basis,
            expected_reference_bits=(
                expected_bits
            ),
            status=(
                MEASUREMENT_STATUS_INVALID_POSITION
            ),
            received_block_id=None,
            measured_bits=None,
        )

    block = blocks[
        position
    ]

    if block.block_id != (
        expected_block_id
    ):
        return ScheduledCheckMeasurement(
            schedule_index=schedule_index,
            expected_block_id=(
                expected_block_id
            ),
            frame_position=position,
            measurement_basis=basis,
            expected_reference_bits=(
                expected_bits
            ),
            status=(
                MEASUREMENT_STATUS_BLOCK_ID_MISMATCH
            ),
            received_block_id=(
                block.block_id
            ),
            measured_bits=None,
        )

    measurement_result = (
        measure_physical_block(
            block=block,
            measurement_basis=basis,
            frame_position=position,
        )
    )

    return ScheduledCheckMeasurement(
        schedule_index=schedule_index,
        expected_block_id=(
            expected_block_id
        ),
        frame_position=position,
        measurement_basis=basis,
        expected_reference_bits=(
            expected_bits
        ),
        status=(
            measurement_result.status
        ),
        received_block_id=(
            block.block_id
        ),
        measured_bits=(
            measurement_result.measured_bits
        ),
    )


def measure_declared_check_blocks(
    received_frame: (
        SteaneEncodedFrame
        | Sequence[PhysicalBlock]
    ),
    schedule: Mapping[str, Any],
) -> list[ScheduledCheckMeasurement]:
    """
    Measure every declared check block in the protected schedule.
    """

    blocks = normalize_received_frame(
        received_frame
    )

    if not isinstance(
        schedule,
        Mapping,
    ):
        raise InvalidMeasurementScheduleError(
            "schedule must be a mapping."
        )

    check_entries = schedule.get(
        "check_blocks"
    )

    if not isinstance(
        check_entries,
        list,
    ):
        raise InvalidMeasurementScheduleError(
            "schedule must contain a check_blocks list."
        )

    return [
        measure_scheduled_check_block(
            received_frame=blocks,
            check_entry=entry,
            schedule_index=index,
        )
        for index, entry in enumerate(
            check_entries
        )
    ]


def summarize_check_measurements(
    measurements: Sequence[
        ScheduledCheckMeasurement
    ],
) -> MeasurementSummary:
    """
    Summarize declared check-block measurement availability.
    """

    if isinstance(
        measurements,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "measurements must be a sequence."
        )

    if not isinstance(
        measurements,
        Sequence,
    ):
        raise TypeError(
            "measurements must be a sequence."
        )

    for measurement in measurements:
        if not isinstance(
            measurement,
            ScheduledCheckMeasurement,
        ):
            raise TypeError(
                "Every item must be "
                "ScheduledCheckMeasurement."
            )

    observed_measurements = [
        measurement
        for measurement in measurements
        if measurement.observed
    ]

    return MeasurementSummary(
        scheduled_check_blocks=len(
            measurements
        ),
        observed_check_blocks=len(
            observed_measurements
        ),
        erased_check_blocks=sum(
            measurement.status
            == MEASUREMENT_STATUS_ERASED
            for measurement in measurements
        ),
        invalid_position_entries=sum(
            measurement.status
            == MEASUREMENT_STATUS_INVALID_POSITION
            for measurement in measurements
        ),
        block_id_mismatches=sum(
            measurement.status
            == MEASUREMENT_STATUS_BLOCK_ID_MISMATCH
            for measurement in measurements
        ),
        invalid_expected_patterns=sum(
            measurement.status
            == MEASUREMENT_STATUS_INVALID_EXPECTED_PATTERN
            for measurement in measurements
        ),
        observed_physical_positions=sum(
            len(
                measurement.measured_bits
            )
            for measurement in observed_measurements
            if measurement.measured_bits
            is not None
        ),
        expected_physical_positions=sum(
            measurement.expected_physical_count
            for measurement in measurements
        ),
    )


def measurement_matches_expected(
    measurement: ScheduledCheckMeasurement,
) -> bool:
    """
    Return whether an observed check pattern exactly matches expected.
    """

    if not isinstance(
        measurement,
        ScheduledCheckMeasurement,
    ):
        raise TypeError(
            "measurement must be ScheduledCheckMeasurement."
        )

    if not measurement.observed:
        return False

    if (
        measurement.measured_bits is None
        or measurement.expected_reference_bits
        is None
    ):
        return False

    if len(
        measurement.measured_bits
    ) != len(
        measurement.expected_reference_bits
    ):
        return False

    return bool(
        np.array_equal(
            np.asarray(
                measurement.measured_bits,
                dtype=np.int8,
            ),
            np.asarray(
                measurement.expected_reference_bits,
                dtype=np.int8,
            ),
        )
    )


def mismatch_count(
    measurement: ScheduledCheckMeasurement,
) -> int:
    """
    Count physical mismatches in one scheduled check measurement.

    Structural schedule mismatches are treated as full expected-pattern
    errors. Erased blocks contribute no observed physical bits and are
    therefore excluded from raw QBER.
    """

    if not isinstance(
        measurement,
        ScheduledCheckMeasurement,
    ):
        raise TypeError(
            "measurement must be ScheduledCheckMeasurement."
        )

    expected = (
        measurement.expected_reference_bits
    )

    if expected is None:
        return 0

    expected_count = len(
        expected
    )

    if measurement.status in {
        MEASUREMENT_STATUS_INVALID_POSITION,
        MEASUREMENT_STATUS_BLOCK_ID_MISMATCH,
    }:
        return expected_count

    if measurement.status in {
        MEASUREMENT_STATUS_ERASED,
        MEASUREMENT_STATUS_INVALID_EXPECTED_PATTERN,
    }:
        return 0

    measured = measurement.measured_bits

    if measured is None:
        return 0

    if len(measured) != expected_count:
        return max(
            len(measured),
            expected_count,
        )

    return int(
        np.sum(
            np.asarray(
                measured,
                dtype=np.int8,
            )
            != np.asarray(
                expected,
                dtype=np.int8,
            )
        )
    )


def run_self_test() -> None:
    """
    Verify Z/X measurement, erasure handling, and schedule processing.
    """

    from .logical_qubit import (
        create_check_logical_qubit,
    )
    from .steane_css import (
        encode_one_logical_qubit,
    )

    print("=" * 72)
    print("FT-QuPAP Quantum Measurement Self-Test")
    print("=" * 72)

    z_spec = create_check_logical_qubit(
        logical_bit=0,
        basis="Z",
        logical_index=0,
        position=0,
    )

    x_spec = create_check_logical_qubit(
        logical_bit=1,
        basis="X",
        logical_index=1,
        position=1,
    )

    z_block = encode_one_logical_qubit(
        spec=z_spec,
        use_css=True,
        rng=np.random.default_rng(
            9001
        ),
    )

    x_block = encode_one_logical_qubit(
        spec=x_spec,
        use_css=True,
        rng=np.random.default_rng(
            9002
        ),
    )

    z_block.x_errors[3] = 1
    x_block.z_errors[5] = 1

    frame = [
        z_block,
        x_block,
    ]

    schedule = {
        "check_blocks": [
            {
                "block_id":
                    z_block.block_id,
                "position":
                    0,
                "basis":
                    "Z",
                "expected_reference_bits":
                    z_block.reference_bits.tolist(),
            },
            {
                "block_id":
                    x_block.block_id,
                "position":
                    1,
                "basis":
                    "X",
                "expected_reference_bits":
                    x_block.reference_bits.tolist(),
            },
        ]
    }

    measurements = (
        measure_declared_check_blocks(
            received_frame=frame,
            schedule=schedule,
        )
    )

    summary = (
        summarize_check_measurements(
            measurements
        )
    )

    z_measurement_correct = (
        measurements[0].measured_bits
        == tuple(
            int(bit)
            for bit in (
                z_block.reference_bits
                ^ z_block.x_errors
            ).tolist()
        )
    )

    x_measurement_correct = (
        measurements[1].measured_bits
        == tuple(
            int(bit)
            for bit in (
                x_block.reference_bits
                ^ x_block.z_errors
            ).tolist()
        )
    )

    erased_block = z_block.copy()
    erased_block.spec.position = 0
    erased_block.erasures[2] = True

    erased_measurement = (
        measure_physical_block(
            block=erased_block,
            measurement_basis="Z",
            frame_position=0,
        )
    )

    z_mismatches = mismatch_count(
        measurements[0]
    )

    x_mismatches = mismatch_count(
        measurements[1]
    )

    print(
        "Scheduled check blocks    : "
        f"{summary.scheduled_check_blocks}"
    )

    print(
        "Observed check blocks     : "
        f"{summary.observed_check_blocks}"
    )

    print(
        "Z-basis measurement valid : "
        f"{z_measurement_correct}"
    )

    print(
        "X-basis measurement valid : "
        f"{x_measurement_correct}"
    )

    print(
        "Z-basis mismatch count    : "
        f"{z_mismatches}"
    )

    print(
        "X-basis mismatch count    : "
        f"{x_mismatches}"
    )

    print(
        "Erased block unobserved   : "
        f"{not erased_measurement.observed}"
    )

    if not z_measurement_correct:
        raise QuantumMeasurementError(
            "Z-basis measurement used the wrong error vector."
        )

    if not x_measurement_correct:
        raise QuantumMeasurementError(
            "X-basis measurement used the wrong error vector."
        )

    if z_mismatches != 1:
        raise QuantumMeasurementError(
            "Z-basis mismatch counting failed."
        )

    if x_mismatches != 1:
        raise QuantumMeasurementError(
            "X-basis mismatch counting failed."
        )

    if erased_measurement.measured_bits is not None:
        raise QuantumMeasurementError(
            "Erased block produced measurement bits."
        )

    if summary.observed_check_blocks != 2:
        raise QuantumMeasurementError(
            "Observed check-block count is incorrect."
        )

    print(
        "\nQuantum measurement self-test "
        "completed successfully."
    )


__all__ = [
    "SUPPORTED_MEASUREMENT_BASES",
    "MEASUREMENT_STATUS_OBSERVED",
    "MEASUREMENT_STATUS_ERASED",
    "MEASUREMENT_STATUS_INVALID_POSITION",
    "MEASUREMENT_STATUS_BLOCK_ID_MISMATCH",
    "MEASUREMENT_STATUS_INVALID_EXPECTED_PATTERN",
    "QuantumMeasurementError",
    "InvalidMeasurementBasisError",
    "InvalidMeasurementFrameError",
    "InvalidMeasurementScheduleError",
    "InvalidExpectedPatternError",
    "PhysicalMeasurementResult",
    "ScheduledCheckMeasurement",
    "MeasurementSummary",
    "validate_measurement_basis",
    "validate_binary_vector",
    "normalize_received_frame",
    "measure_block_in_declared_basis",
    "measure_physical_block",
    "validate_check_schedule_entry",
    "measure_scheduled_check_block",
    "measure_declared_check_blocks",
    "summarize_check_measurements",
    "measurement_matches_expected",
    "mismatch_count",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        QuantumMeasurementError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[QUANTUM MEASUREMENT ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error