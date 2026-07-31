"""
FT-QuPAP Steane Error Correction

This module applies syndrome-level Steane [[7,1,3]] correction to
received FT-QuPAP physical blocks.

Bounded-error model used by the project notebook:

1. A clean Steane block is accepted.
2. One physical position containing an X, Z, or Y-type error can be
   corrected.
3. Any block containing an erasure is rejected.
4. Errors affecting more than one physical position are rejected.
5. Z-basis measurement uses the X-error syndrome.
6. X-basis measurement uses the Z-error syndrome.
7. No-CSS baseline blocks are accepted only when no physical error
   is present.

Important protocol rule:

Raw check-block QBER must be calculated before applying correction.
Error correction improves payload recovery but must not erase the
channel-disturbance evidence used by the authentication policy.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .steane_css import (
    PhysicalBlock,
    STEANE_BLOCK_SIZE,
    STEANE_SYNDROME_TO_POSITION,
    SteaneEncodedFrame,
    syndrome_tuple,
    validate_physical_block,
)
from .syndrome_extraction import (
    steane_syndrome,
)


SUPPORTED_MEASUREMENT_BASES = (
    "Z",
    "X",
)

CORRECTION_STATUS_CLEAN = "clean"
CORRECTION_STATUS_CORRECTED = "corrected"
CORRECTION_STATUS_ERASED = "erased"
CORRECTION_STATUS_UNCORRECTABLE = "uncorrectable"
CORRECTION_STATUS_RAW_OK = "raw_ok"
CORRECTION_STATUS_RAW_ERROR = "raw_error"
CORRECTION_STATUS_INVALID_BASIS = "invalid_measurement_basis"

SUCCESSFUL_CORRECTION_STATUSES = {
    CORRECTION_STATUS_CLEAN,
    CORRECTION_STATUS_CORRECTED,
    CORRECTION_STATUS_RAW_OK,
}

FAILED_CORRECTION_STATUSES = {
    CORRECTION_STATUS_ERASED,
    CORRECTION_STATUS_UNCORRECTABLE,
    CORRECTION_STATUS_RAW_ERROR,
    CORRECTION_STATUS_INVALID_BASIS,
}


class ErrorCorrectionError(Exception):
    """Base exception for FT-QuPAP error-correction failures."""


class InvalidCorrectionInputError(ErrorCorrectionError):
    """Raised when correction input is malformed."""


class InvalidCorrectionScheduleError(ErrorCorrectionError):
    """Raised when the decoder schedule is inconsistent."""


class CorrectionFrameError(ErrorCorrectionError):
    """Raised when a complete frame cannot be corrected."""


@dataclass(frozen=True)
class BlockCorrectionResult:
    """
    Correction result for one physical block.

    Attributes:
        block_id:
            Logical block identifier.

        role:
            Either payload or check.

        declared_basis:
            Measurement basis supplied by the protected schedule.

        status:
            clean, corrected, erased, uncorrectable, raw_ok, or
            raw_error.

        correctable:
            Whether a valid corrected measurement is available.

        corrected_measurement:
            Corrected physical measurement string or None.

        relevant_syndrome:
            X syndrome for Z-basis measurement or Z syndrome for
            X-basis measurement.

        correction_vector:
            Lookup-based physical correction vector.

        physical_error_weight:
            Number of physical positions containing any Pauli error.

        erasure_count:
            Number of erased physical positions.

        use_css:
            Whether Steane CSS encoding was used.
    """

    block_id: str
    role: str
    declared_basis: str
    status: str
    correctable: bool
    corrected_measurement: tuple[int, ...] | None
    relevant_syndrome: tuple[int, int, int]
    correction_vector: tuple[int, ...]
    physical_error_weight: int
    erasure_count: int
    use_css: bool

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str):
            raise TypeError(
                "block_id must be a string."
            )

        if not self.block_id:
            raise ValueError(
                "block_id cannot be empty."
            )

        if self.role not in (
            "payload",
            "check",
        ):
            raise ValueError(
                "role must be 'payload' or 'check'."
            )

        if self.declared_basis not in (
            SUPPORTED_MEASUREMENT_BASES
        ):
            raise ValueError(
                "declared_basis must be Z or X."
            )

        valid_statuses = (
            SUCCESSFUL_CORRECTION_STATUSES
            | FAILED_CORRECTION_STATUSES
        )

        if self.status not in valid_statuses:
            raise ValueError(
                f"Unsupported correction status: {self.status!r}."
            )

        if not isinstance(
            self.correctable,
            bool,
        ):
            raise TypeError(
                "correctable must be boolean."
            )

        if len(self.relevant_syndrome) != 3:
            raise ValueError(
                "relevant_syndrome must contain three bits."
            )

        if any(
            value not in (0, 1)
            for value in self.relevant_syndrome
        ):
            raise ValueError(
                "relevant_syndrome must be binary."
            )

        expected_vector_length = (
            STEANE_BLOCK_SIZE
            if self.use_css
            else 1
        )

        if len(self.correction_vector) != expected_vector_length:
            raise ValueError(
                "correction_vector has an invalid length."
            )

        if any(
            value not in (0, 1)
            for value in self.correction_vector
        ):
            raise ValueError(
                "correction_vector must be binary."
            )

        if self.corrected_measurement is not None:
            if len(self.corrected_measurement) != expected_vector_length:
                raise ValueError(
                    "corrected_measurement has an invalid length."
                )

            if any(
                value not in (0, 1)
                for value in self.corrected_measurement
            ):
                raise ValueError(
                    "corrected_measurement must be binary."
                )

        if isinstance(
            self.physical_error_weight,
            bool,
        ) or not isinstance(
            self.physical_error_weight,
            int,
        ):
            raise TypeError(
                "physical_error_weight must be an integer."
            )

        if self.physical_error_weight < 0:
            raise ValueError(
                "physical_error_weight cannot be negative."
            )

        if isinstance(
            self.erasure_count,
            bool,
        ) or not isinstance(
            self.erasure_count,
            int,
        ):
            raise TypeError(
                "erasure_count must be an integer."
            )

        if self.erasure_count < 0:
            raise ValueError(
                "erasure_count cannot be negative."
            )

        if not isinstance(
            self.use_css,
            bool,
        ):
            raise TypeError(
                "use_css must be boolean."
            )

        if self.correctable:
            if self.corrected_measurement is None:
                raise ValueError(
                    "A correctable result must contain "
                    "corrected_measurement."
                )

            if self.status not in SUCCESSFUL_CORRECTION_STATUSES:
                raise ValueError(
                    "A correctable result has a failure status."
                )

        else:
            if self.status in SUCCESSFUL_CORRECTION_STATUSES:
                raise ValueError(
                    "A failed result has a successful status."
                )

    @property
    def correction_applied(self) -> bool:
        """Return whether a nonzero correction vector was applied."""

        return any(
            self.correction_vector
        )

    @property
    def failed(self) -> bool:
        """Return whether block recovery failed."""

        return not self.correctable

    def to_dictionary(self) -> dict[str, Any]:
        """Return a serializable correction record."""

        return {
            "block_id":
                self.block_id,
            "role":
                self.role,
            "declared_basis":
                self.declared_basis,
            "status":
                self.status,
            "correctable":
                self.correctable,
            "corrected_measurement": (
                list(self.corrected_measurement)
                if self.corrected_measurement is not None
                else None
            ),
            "relevant_syndrome":
                list(self.relevant_syndrome),
            "correction_vector":
                list(self.correction_vector),
            "correction_applied":
                self.correction_applied,
            "physical_error_weight":
                self.physical_error_weight,
            "erasure_count":
                self.erasure_count,
            "use_css":
                self.use_css,
        }

    def to_notebook_dictionary(self) -> dict[str, Any]:
        """
        Return the notebook-compatible result structure.
        """

        return {
            "status":
                self.status,
            "correctable":
                self.correctable,
            "corrected_measurement": (
                np.asarray(
                    self.corrected_measurement,
                    dtype=np.int8,
                )
                if self.corrected_measurement is not None
                else None
            ),
        }


@dataclass(frozen=True)
class CorrectionSummary:
    """
    Aggregate correction information for one received frame.
    """

    total_blocks: int
    clean_blocks: int
    corrected_blocks: int
    raw_ok_blocks: int
    raw_error_blocks: int
    erased_blocks: int
    uncorrectable_blocks: int
    failed_blocks: int
    correction_applied_blocks: int
    correction_failure_rate: float
    payload_blocks: int
    failed_payload_blocks: int
    payload_recovery_success: bool

    def to_dictionary(self) -> dict[str, Any]:
        """Return aggregate correction information."""

        return {
            "total_blocks":
                self.total_blocks,
            "clean_blocks":
                self.clean_blocks,
            "corrected_blocks":
                self.corrected_blocks,
            "raw_ok_blocks":
                self.raw_ok_blocks,
            "raw_error_blocks":
                self.raw_error_blocks,
            "erased_blocks":
                self.erased_blocks,
            "uncorrectable_blocks":
                self.uncorrectable_blocks,
            "failed_blocks":
                self.failed_blocks,
            "correction_applied_blocks":
                self.correction_applied_blocks,
            "correction_failure_rate":
                self.correction_failure_rate,
            "payload_blocks":
                self.payload_blocks,
            "failed_payload_blocks":
                self.failed_payload_blocks,
            "payload_recovery_success":
                self.payload_recovery_success,
        }


def validate_measurement_basis(
    measurement_basis: Any,
) -> str:
    """Validate a Z- or X-basis label."""

    if not isinstance(
        measurement_basis,
        str,
    ):
        raise TypeError(
            "measurement_basis must be a string."
        )

    normalized_basis = (
        measurement_basis.strip().upper()
    )

    if normalized_basis not in (
        SUPPORTED_MEASUREMENT_BASES
    ):
        raise ValueError(
            "measurement_basis must be 'Z' or 'X'."
        )

    return normalized_basis


def normalize_binary_vector(
    vector: Any,
    field_name: str,
    expected_length: int,
) -> np.ndarray:
    """Validate and normalize a binary vector."""

    try:
        normalized = np.asarray(
            vector,
            dtype=np.int8,
        )

    except Exception as error:
        raise InvalidCorrectionInputError(
            f"{field_name} cannot be converted "
            "to a NumPy array."
        ) from error

    if normalized.ndim != 1:
        raise InvalidCorrectionInputError(
            f"{field_name} must be one-dimensional."
        )

    if len(normalized) != expected_length:
        raise InvalidCorrectionInputError(
            f"{field_name} must contain exactly "
            f"{expected_length} bits."
        )

    if not np.all(
        np.isin(
            normalized,
            [0, 1],
        )
    ):
        raise InvalidCorrectionInputError(
            f"{field_name} must contain only 0 or 1."
        )

    return normalized.copy()


def physical_error_mask(
    block: PhysicalBlock,
) -> np.ndarray:
    """
    Return physical positions containing any X or Z component.

    A Y error has both X and Z components but counts as one affected
    physical position.
    """

    validate_physical_block(
        block
    )

    return np.logical_or(
        block.x_errors.astype(bool),
        block.z_errors.astype(bool),
    )


def physical_error_weight(
    block: PhysicalBlock,
) -> int:
    """Return the number of affected physical positions."""

    return int(
        np.count_nonzero(
            physical_error_mask(
                block
            )
        )
    )


def measure_block_in_declared_basis(
    block: PhysicalBlock,
    declared_basis: str,
) -> np.ndarray | None:
    """
    Produce the receiver's raw physical measurement string.

    Z-basis measurement is affected by X-error components.

    X-basis measurement is affected by Z-error components.

    Erased blocks return None.
    """

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

    relevant_error_vector = (
        block.x_errors
        if normalized_basis == "Z"
        else block.z_errors
    )

    return (
        block.reference_bits
        ^ relevant_error_vector
    ).astype(
        np.int8
    )


def correction_vector_from_syndrome(
    syndrome: np.ndarray | Sequence[int],
    use_css: bool = True,
) -> np.ndarray:
    """
    Convert a syndrome into a single-position correction vector.

    Zero and unknown syndromes produce an all-zero vector.
    """

    if not isinstance(
        use_css,
        bool,
    ):
        raise TypeError(
            "use_css must be boolean."
        )

    if not use_css:
        return np.zeros(
            1,
            dtype=np.int8,
        )

    normalized_syndrome = (
        normalize_binary_vector(
            vector=syndrome,
            field_name="syndrome",
            expected_length=3,
        )
    )

    correction = np.zeros(
        STEANE_BLOCK_SIZE,
        dtype=np.int8,
    )

    position = (
        STEANE_SYNDROME_TO_POSITION
        .get(
            syndrome_tuple(
                normalized_syndrome
            ),
            None,
        )
    )

    if position is None or position < 0:
        return correction

    correction[position] = 1

    return correction


def relevant_error_vector(
    block: PhysicalBlock,
    declared_basis: str,
) -> np.ndarray:
    """
    Return the error vector relevant to a declared basis.
    """

    normalized_basis = (
        validate_measurement_basis(
            declared_basis
        )
    )

    return (
        block.x_errors
        if normalized_basis == "Z"
        else block.z_errors
    ).copy()


def correct_block(
    block: PhysicalBlock,
    declared_basis: str,
) -> BlockCorrectionResult:
    """
    Correct one syndrome-level physical block.

    The simulator accepts at most one physical position containing
    any Pauli error. Erasures and multi-position errors fail.
    """

    validate_physical_block(
        block
    )

    normalized_basis = (
        validate_measurement_basis(
            declared_basis
        )
    )

    block_error_weight = (
        physical_error_weight(
            block
        )
    )

    erasure_count = int(
        np.count_nonzero(
            block.erasures
        )
    )

    physical_count = (
        block.physical_qubit_count
    )

    zero_correction = tuple(
        0
        for _ in range(
            physical_count
        )
    )

    zero_syndrome = (
        0,
        0,
        0,
    )

    if erasure_count > 0:
        return BlockCorrectionResult(
            block_id=block.block_id,
            role=block.role,
            declared_basis=normalized_basis,
            status=CORRECTION_STATUS_ERASED,
            correctable=False,
            corrected_measurement=None,
            relevant_syndrome=zero_syndrome,
            correction_vector=zero_correction,
            physical_error_weight=(
                block_error_weight
            ),
            erasure_count=erasure_count,
            use_css=block.use_css,
        )

    measurement = (
        measure_block_in_declared_basis(
            block,
            normalized_basis,
        )
    )

    if measurement is None:
        raise ErrorCorrectionError(
            "Non-erased block produced no measurement."
        )

    if not block.use_css:
        status = (
            CORRECTION_STATUS_RAW_OK
            if block_error_weight == 0
            else CORRECTION_STATUS_RAW_ERROR
        )

        correctable = (
            block_error_weight == 0
        )

        return BlockCorrectionResult(
            block_id=block.block_id,
            role=block.role,
            declared_basis=normalized_basis,
            status=status,
            correctable=correctable,
            corrected_measurement=(
                tuple(
                    int(value)
                    for value in measurement.tolist()
                )
                if correctable
                else None
            ),
            relevant_syndrome=zero_syndrome,
            correction_vector=zero_correction,
            physical_error_weight=(
                block_error_weight
            ),
            erasure_count=0,
            use_css=False,
        )

    if block_error_weight > 1:
        relevant_vector = (
            relevant_error_vector(
                block,
                normalized_basis,
            )
        )

        syndrome = steane_syndrome(
            error_vector=relevant_vector,
            use_css=True,
        )

        return BlockCorrectionResult(
            block_id=block.block_id,
            role=block.role,
            declared_basis=normalized_basis,
            status=(
                CORRECTION_STATUS_UNCORRECTABLE
            ),
            correctable=False,
            corrected_measurement=None,
            relevant_syndrome=(
                syndrome_tuple(
                    syndrome
                )
            ),
            correction_vector=zero_correction,
            physical_error_weight=(
                block_error_weight
            ),
            erasure_count=0,
            use_css=True,
        )

    relevant_vector = (
        relevant_error_vector(
            block,
            normalized_basis,
        )
    )

    syndrome = steane_syndrome(
        error_vector=relevant_vector,
        use_css=True,
    )

    correction = (
        correction_vector_from_syndrome(
            syndrome=syndrome,
            use_css=True,
        )
    )

    corrected_measurement = (
        measurement
        ^ correction
    ).astype(
        np.int8
    )

    status = (
        CORRECTION_STATUS_CLEAN
        if block_error_weight == 0
        else CORRECTION_STATUS_CORRECTED
    )

    return BlockCorrectionResult(
        block_id=block.block_id,
        role=block.role,
        declared_basis=normalized_basis,
        status=status,
        correctable=True,
        corrected_measurement=tuple(
            int(value)
            for value in corrected_measurement.tolist()
        ),
        relevant_syndrome=(
            syndrome_tuple(
                syndrome
            )
        ),
        correction_vector=tuple(
            int(value)
            for value in correction.tolist()
        ),
        physical_error_weight=(
            block_error_weight
        ),
        erasure_count=0,
        use_css=True,
    )


def correct_css_block(
    block: PhysicalBlock,
    declared_basis: str,
) -> dict[str, Any]:
    """
    Notebook-compatible block-correction function.

    Invalid basis values return a failure dictionary rather than
    raising an exception.
    """

    try:
        result = correct_block(
            block=block,
            declared_basis=(
                declared_basis
            ),
        )

    except (
        TypeError,
        ValueError,
    ):
        return {
            "status":
                CORRECTION_STATUS_INVALID_BASIS,
            "correctable":
                False,
            "corrected_measurement":
                None,
        }

    return result.to_notebook_dictionary()


def correct_qiskit_receiver_measurement(
    syndrome_record: Mapping[str, Any],
    measurement_basis: str,
) -> dict[str, Any]:
    """
    Apply syndrome correction to a representative Qiskit result.

    Required syndrome_record fields:

        receiver_bits
        x_syndrome
        z_syndrome
    """

    if not isinstance(
        syndrome_record,
        Mapping,
    ):
        raise TypeError(
            "syndrome_record must be a mapping."
        )

    normalized_basis = (
        validate_measurement_basis(
            measurement_basis
        )
    )

    required_fields = {
        "receiver_bits",
        "x_syndrome",
        "z_syndrome",
    }

    missing_fields = (
        required_fields.difference(
            syndrome_record.keys()
        )
    )

    if missing_fields:
        raise InvalidCorrectionInputError(
            "syndrome_record is missing fields: "
            f"{sorted(missing_fields)}"
        )

    receiver_bits = (
        normalize_binary_vector(
            vector=syndrome_record[
                "receiver_bits"
            ],
            field_name="receiver_bits",
            expected_length=(
                STEANE_BLOCK_SIZE
            ),
        )
    )

    relevant_syndrome = (
        syndrome_record[
            "x_syndrome"
        ]
        if normalized_basis == "Z"
        else syndrome_record[
            "z_syndrome"
        ]
    )

    normalized_syndrome = (
        normalize_binary_vector(
            vector=relevant_syndrome,
            field_name=(
                "relevant_syndrome"
            ),
            expected_length=3,
        )
    )

    correction = (
        correction_vector_from_syndrome(
            syndrome=(
                normalized_syndrome
            ),
            use_css=True,
        )
    )

    corrected_bits = (
        receiver_bits
        ^ correction
    ).astype(
        np.int8
    )

    return {
        **dict(
            syndrome_record
        ),
        "measurement_basis":
            normalized_basis,
        "relevant_syndrome":
            normalized_syndrome,
        "correction_vector":
            correction,
        "correction_applied":
            bool(
                np.any(
                    correction
                )
            ),
        "corrected_receiver_bits":
            corrected_bits,
    }


def normalize_received_frame(
    received_frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> list[PhysicalBlock]:
    """
    Normalize a received frame into a physical-block list.
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
        raise CorrectionFrameError(
            "received_frame cannot be empty."
        )

    seen_ids: set[str] = set()

    for block in blocks:
        if not isinstance(
            block,
            PhysicalBlock,
        ):
            raise TypeError(
                "Every received-frame item must "
                "be a PhysicalBlock."
            )

        validate_physical_block(
            block
        )

        if block.block_id in seen_ids:
            raise CorrectionFrameError(
                "Duplicate received block ID: "
                f"{block.block_id!r}."
            )

        seen_ids.add(
            block.block_id
        )

    return blocks


def build_schedule_by_position(
    schedule: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """
    Convert the decrypted control schedule into a position lookup.

    Payload blocks always use the Z basis. Check-block bases are read
    from the protected schedule.
    """

    if not isinstance(
        schedule,
        Mapping,
    ):
        raise TypeError(
            "schedule must be a mapping."
        )

    required_fields = {
        "ordered_block_ids",
        "payload_blocks",
        "check_blocks",
    }

    missing_fields = (
        required_fields.difference(
            schedule.keys()
        )
    )

    if missing_fields:
        raise InvalidCorrectionScheduleError(
            "Schedule is missing fields: "
            f"{sorted(missing_fields)}"
        )

    ordered_ids = schedule[
        "ordered_block_ids"
    ]

    payload_entries = schedule[
        "payload_blocks"
    ]

    check_entries = schedule[
        "check_blocks"
    ]

    if not isinstance(
        ordered_ids,
        list,
    ):
        raise TypeError(
            "ordered_block_ids must be a list."
        )

    if not isinstance(
        payload_entries,
        list,
    ):
        raise TypeError(
            "payload_blocks must be a list."
        )

    if not isinstance(
        check_entries,
        list,
    ):
        raise TypeError(
            "check_blocks must be a list."
        )

    lookup: dict[
        int,
        dict[str, Any],
    ] = {}

    for entry in payload_entries:
        if not isinstance(
            entry,
            Mapping,
        ):
            raise TypeError(
                "Payload schedule entries must be mappings."
            )

        required_payload_fields = {
            "block_id",
            "position",
            "logical_index",
        }

        if not required_payload_fields.issubset(
            entry.keys()
        ):
            raise InvalidCorrectionScheduleError(
                "Payload schedule entry is incomplete."
            )

        normalized_entry = {
            **dict(entry),
            "role":
                "payload",
            "basis":
                "Z",
        }

        _register_schedule_position(
            lookup=lookup,
            ordered_ids=ordered_ids,
            entry=normalized_entry,
        )

    for entry in check_entries:
        if not isinstance(
            entry,
            Mapping,
        ):
            raise TypeError(
                "Check schedule entries must be mappings."
            )

        required_check_fields = {
            "block_id",
            "position",
            "basis",
            "expected_logical_bit",
        }

        if not required_check_fields.issubset(
            entry.keys()
        ):
            raise InvalidCorrectionScheduleError(
                "Check schedule entry is incomplete."
            )

        normalized_entry = {
            **dict(entry),
            "role":
                "check",
            "basis":
                validate_measurement_basis(
                    entry["basis"]
                ),
        }

        _register_schedule_position(
            lookup=lookup,
            ordered_ids=ordered_ids,
            entry=normalized_entry,
        )

    if set(lookup) != set(
        range(
            len(
                ordered_ids
            )
        )
    ):
        raise InvalidCorrectionScheduleError(
            "Schedule does not cover every frame position."
        )

    return lookup


def _register_schedule_position(
    lookup: dict[int, dict[str, Any]],
    ordered_ids: Sequence[str],
    entry: Mapping[str, Any],
) -> None:
    """Validate and register one schedule position."""

    position = entry[
        "position"
    ]

    block_id = entry[
        "block_id"
    ]

    if isinstance(
        position,
        bool,
    ) or not isinstance(
        position,
        int,
    ):
        raise InvalidCorrectionScheduleError(
            "Schedule position must be an integer."
        )

    if not 0 <= position < len(
        ordered_ids
    ):
        raise InvalidCorrectionScheduleError(
            "Schedule position is outside the frame."
        )

    if not isinstance(
        block_id,
        str,
    ):
        raise InvalidCorrectionScheduleError(
            "Schedule block_id must be a string."
        )

    if ordered_ids[position] != block_id:
        raise InvalidCorrectionScheduleError(
            "Schedule block ID does not match "
            "ordered_block_ids."
        )

    if position in lookup:
        raise InvalidCorrectionScheduleError(
            f"Duplicate schedule position: {position}."
        )

    lookup[position] = dict(
        entry
    )


def build_decoder_records(
    received_frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    schedule: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Build Authentication Server decoder records.

    The declared basis and logical role come from the decrypted and
    validated control schedule.
    """

    blocks = normalize_received_frame(
        received_frame
    )

    schedule_by_position = (
        build_schedule_by_position(
            schedule
        )
    )

    if len(blocks) != len(
        schedule_by_position
    ):
        raise CorrectionFrameError(
            "Received frame length does not match "
            "the control schedule."
        )

    records: list[
        dict[str, Any]
    ] = []

    for position, block in enumerate(
        blocks
    ):
        schedule_entry = (
            schedule_by_position[
                position
            ]
        )

        if (
            block.block_id
            != schedule_entry[
                "block_id"
            ]
        ):
            raise CorrectionFrameError(
                f"Received block {block.block_id!r} "
                f"does not match scheduled block "
                f"{schedule_entry['block_id']!r} "
                f"at position {position}."
            )

        declared_basis = (
            schedule_entry[
                "basis"
            ]
        )

        correction_result = (
            correct_block(
                block=block,
                declared_basis=(
                    declared_basis
                ),
            )
        )

        record = {
            "position":
                position,
            "block_id":
                block.block_id,
            "role":
                schedule_entry[
                    "role"
                ],
            "declared_basis":
                declared_basis,
            "status":
                correction_result.status,
            "correctable":
                correction_result.correctable,
            "corrected_measurement": (
                np.asarray(
                    correction_result
                    .corrected_measurement,
                    dtype=np.int8,
                )
                if correction_result
                .corrected_measurement
                is not None
                else None
            ),
            "relevant_syndrome":
                np.asarray(
                    correction_result
                    .relevant_syndrome,
                    dtype=np.int8,
                ),
            "correction_vector":
                np.asarray(
                    correction_result
                    .correction_vector,
                    dtype=np.int8,
                ),
            "correction_applied":
                correction_result
                .correction_applied,
            "physical_error_weight":
                correction_result
                .physical_error_weight,
            "erasure_count":
                correction_result
                .erasure_count,
        }

        if (
            schedule_entry[
                "role"
            ] == "payload"
        ):
            record[
                "logical_index"
            ] = schedule_entry[
                "logical_index"
            ]

        else:
            record[
                "expected_logical_bit"
            ] = schedule_entry[
                "expected_logical_bit"
            ]

        records.append(
            record
        )

    return records


def correct_received_frame(
    received_frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    schedule: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Compatibility alias for build_decoder_records().
    """

    return build_decoder_records(
        received_frame=received_frame,
        schedule=schedule,
    )


def summarize_correction_results(
    correction_results: Sequence[
        BlockCorrectionResult | Mapping[str, Any]
    ],
) -> CorrectionSummary:
    """
    Summarize correction success and failure rates.
    """

    if isinstance(
        correction_results,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "correction_results must be a sequence."
        )

    if not isinstance(
        correction_results,
        Sequence,
    ):
        raise TypeError(
            "correction_results must be a sequence."
        )

    if not correction_results:
        raise ValueError(
            "correction_results cannot be empty."
        )

    normalized_records: list[
        dict[str, Any]
    ] = []

    for result in correction_results:
        if isinstance(
            result,
            BlockCorrectionResult,
        ):
            normalized_records.append(
                result.to_dictionary()
            )

        elif isinstance(
            result,
            Mapping,
        ):
            normalized_records.append(
                dict(result)
            )

        else:
            raise TypeError(
                "Each correction result must be "
                "BlockCorrectionResult or a mapping."
            )

    statuses = Counter(
        record.get(
            "status",
            "unknown",
        )
        for record in normalized_records
    )

    failed_records = [
        record
        for record in normalized_records
        if not bool(
            record.get(
                "correctable",
                False,
            )
        )
    ]

    payload_records = [
        record
        for record in normalized_records
        if record.get(
            "role"
        ) == "payload"
    ]

    failed_payload_records = [
        record
        for record in payload_records
        if not bool(
            record.get(
                "correctable",
                False,
            )
        )
    ]

    correction_applied_count = sum(
        bool(
            record.get(
                "correction_applied",
                False,
            )
        )
        for record in normalized_records
    )

    total_blocks = len(
        normalized_records
    )

    failed_blocks = len(
        failed_records
    )

    return CorrectionSummary(
        total_blocks=total_blocks,
        clean_blocks=int(
            statuses.get(
                CORRECTION_STATUS_CLEAN,
                0,
            )
        ),
        corrected_blocks=int(
            statuses.get(
                CORRECTION_STATUS_CORRECTED,
                0,
            )
        ),
        raw_ok_blocks=int(
            statuses.get(
                CORRECTION_STATUS_RAW_OK,
                0,
            )
        ),
        raw_error_blocks=int(
            statuses.get(
                CORRECTION_STATUS_RAW_ERROR,
                0,
            )
        ),
        erased_blocks=int(
            statuses.get(
                CORRECTION_STATUS_ERASED,
                0,
            )
        ),
        uncorrectable_blocks=int(
            statuses.get(
                CORRECTION_STATUS_UNCORRECTABLE,
                0,
            )
        ),
        failed_blocks=failed_blocks,
        correction_applied_blocks=(
            correction_applied_count
        ),
        correction_failure_rate=float(
            failed_blocks
            / total_blocks
        ),
        payload_blocks=len(
            payload_records
        ),
        failed_payload_blocks=len(
            failed_payload_records
        ),
        payload_recovery_success=(
            len(
                failed_payload_records
            ) == 0
        ),
    )


def corrected_measurement_matches_reference(
    block: PhysicalBlock,
    correction_result: BlockCorrectionResult,
) -> bool:
    """
    Verify that a successful correction restores the reference word.
    """

    if not isinstance(
        correction_result,
        BlockCorrectionResult,
    ):
        raise TypeError(
            "correction_result must be "
            "a BlockCorrectionResult."
        )

    if (
        correction_result.block_id
        != block.block_id
    ):
        raise ValueError(
            "Correction result belongs to a different block."
        )

    if not correction_result.correctable:
        return False

    if correction_result.corrected_measurement is None:
        return False

    return np.array_equal(
        np.asarray(
            correction_result
            .corrected_measurement,
            dtype=np.int8,
        ),
        block.reference_bits,
    )


def run_self_test() -> None:
    """
    Verify clean, correctable, erased, and uncorrectable cases.
    """

    from .logical_qubit import (
        create_check_logical_qubit,
        create_payload_logical_qubit,
    )
    from .steane_css import (
        encode_one_logical_qubit,
    )

    print("=" * 72)
    print("FT-QuPAP Error Correction Self-Test")
    print("=" * 72)

    clean_spec = (
        create_payload_logical_qubit(
            logical_bit=0,
            logical_index=0,
            position=0,
        )
    )

    z_basis_spec = (
        create_check_logical_qubit(
            logical_bit=1,
            basis="Z",
            logical_index=0,
            position=1,
        )
    )

    x_basis_spec = (
        create_check_logical_qubit(
            logical_bit=0,
            basis="X",
            logical_index=1,
            position=2,
        )
    )

    clean_block = (
        encode_one_logical_qubit(
            spec=clean_spec,
            use_css=True,
            rng=np.random.default_rng(
                1001
            ),
        )
    )

    x_error_block = (
        encode_one_logical_qubit(
            spec=z_basis_spec,
            use_css=True,
            rng=np.random.default_rng(
                1002
            ),
        )
    )

    z_error_block = (
        encode_one_logical_qubit(
            spec=x_basis_spec,
            use_css=True,
            rng=np.random.default_rng(
                1003
            ),
        )
    )

    multi_error_block = (
        encode_one_logical_qubit(
            spec=z_basis_spec.copy(),
            use_css=True,
            rng=np.random.default_rng(
                1004
            ),
        )
    )

    erased_block = (
        encode_one_logical_qubit(
            spec=x_basis_spec.copy(),
            use_css=True,
            rng=np.random.default_rng(
                1005
            ),
        )
    )

    x_error_block.x_errors[4] = 1
    z_error_block.z_errors[2] = 1

    multi_error_block.x_errors[1] = 1
    multi_error_block.x_errors[5] = 1

    erased_block.erasures[6] = True

    clean_result = correct_block(
        clean_block,
        "Z",
    )

    x_result = correct_block(
        x_error_block,
        "Z",
    )

    z_result = correct_block(
        z_error_block,
        "X",
    )

    multi_result = correct_block(
        multi_error_block,
        "Z",
    )

    erased_result = correct_block(
        erased_block,
        "X",
    )

    clean_restored = (
        corrected_measurement_matches_reference(
            clean_block,
            clean_result,
        )
    )

    x_error_restored = (
        corrected_measurement_matches_reference(
            x_error_block,
            x_result,
        )
    )

    z_error_restored = (
        corrected_measurement_matches_reference(
            z_error_block,
            z_result,
        )
    )

    qiskit_result = (
        correct_qiskit_receiver_measurement(
            syndrome_record={
                "receiver_bits":
                    [0, 0, 0, 0, 1, 0, 0],
                "x_syndrome":
                    [
                        int(value)
                        for value in (
                            steane_syndrome(
                                np.array(
                                    [0, 0, 0, 0, 1, 0, 0],
                                    dtype=np.int8,
                                ),
                                use_css=True,
                            )
                        )
                    ],
                "z_syndrome":
                    [0, 0, 0],
            },
            measurement_basis="Z",
        )
    )

    qiskit_correction_applied = bool(
        qiskit_result[
            "correction_applied"
        ]
    )

    summary = (
        summarize_correction_results(
            [
                clean_result,
                x_result,
                z_result,
                multi_result,
                erased_result,
            ]
        )
    )

    print(
        "Clean block status        : "
        f"{clean_result.status}"
    )

    print(
        "Single X-error status     : "
        f"{x_result.status}"
    )

    print(
        "Single X-error position   : "
        f"{list(x_result.correction_vector).index(1)}"
    )

    print(
        "Single Z-error status     : "
        f"{z_result.status}"
    )

    print(
        "Single Z-error position   : "
        f"{list(z_result.correction_vector).index(1)}"
    )

    print(
        "Multi-error status        : "
        f"{multi_result.status}"
    )

    print(
        "Erasure status            : "
        f"{erased_result.status}"
    )

    print(
        "Clean measurement restored: "
        f"{clean_restored}"
    )

    print(
        "X-error measurement fixed : "
        f"{x_error_restored}"
    )

    print(
        "Z-error measurement fixed : "
        f"{z_error_restored}"
    )

    print(
        "Qiskit correction applied : "
        f"{qiskit_correction_applied}"
    )

    print(
        "Correction failure rate   : "
        f"{summary.correction_failure_rate:.6f}"
    )

    if clean_result.status != (
        CORRECTION_STATUS_CLEAN
    ):
        raise ErrorCorrectionError(
            "Clean block received an incorrect status."
        )

    if x_result.status != (
        CORRECTION_STATUS_CORRECTED
    ):
        raise ErrorCorrectionError(
            "Single X error was not corrected."
        )

    if z_result.status != (
        CORRECTION_STATUS_CORRECTED
    ):
        raise ErrorCorrectionError(
            "Single Z error was not corrected."
        )

    if multi_result.status != (
        CORRECTION_STATUS_UNCORRECTABLE
    ):
        raise ErrorCorrectionError(
            "Multi-position error was not rejected."
        )

    if erased_result.status != (
        CORRECTION_STATUS_ERASED
    ):
        raise ErrorCorrectionError(
            "Erased block was not rejected."
        )

    if not clean_restored:
        raise ErrorCorrectionError(
            "Clean measurement does not match its reference."
        )

    if not x_error_restored:
        raise ErrorCorrectionError(
            "X-error correction did not restore the reference."
        )

    if not z_error_restored:
        raise ErrorCorrectionError(
            "Z-error correction did not restore the reference."
        )

    if not qiskit_correction_applied:
        raise ErrorCorrectionError(
            "Qiskit correction lookup was not applied."
        )

    if summary.failed_blocks != 2:
        raise ErrorCorrectionError(
            "Correction failure count is incorrect."
        )

    print(
        "\nError correction self-test "
        "completed successfully."
    )


__all__ = [
    "SUPPORTED_MEASUREMENT_BASES",
    "CORRECTION_STATUS_CLEAN",
    "CORRECTION_STATUS_CORRECTED",
    "CORRECTION_STATUS_ERASED",
    "CORRECTION_STATUS_UNCORRECTABLE",
    "CORRECTION_STATUS_RAW_OK",
    "CORRECTION_STATUS_RAW_ERROR",
    "CORRECTION_STATUS_INVALID_BASIS",
    "SUCCESSFUL_CORRECTION_STATUSES",
    "FAILED_CORRECTION_STATUSES",
    "ErrorCorrectionError",
    "InvalidCorrectionInputError",
    "InvalidCorrectionScheduleError",
    "CorrectionFrameError",
    "BlockCorrectionResult",
    "CorrectionSummary",
    "validate_measurement_basis",
    "physical_error_mask",
    "physical_error_weight",
    "measure_block_in_declared_basis",
    "correction_vector_from_syndrome",
    "relevant_error_vector",
    "correct_block",
    "correct_css_block",
    "correct_qiskit_receiver_measurement",
    "normalize_received_frame",
    "build_schedule_by_position",
    "build_decoder_records",
    "correct_received_frame",
    "summarize_correction_results",
    "corrected_measurement_matches_reference",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        ErrorCorrectionError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[ERROR CORRECTION ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error