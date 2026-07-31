"""
FT-QuPAP Steane Syndrome Extraction

This module extracts Steane [[7,1,3]] syndrome information from
received FT-QuPAP physical blocks.

For every seven-qubit CSS block, the simulator extracts:

    X syndrome = H @ x_errors mod 2
    Z syndrome = H @ z_errors mod 2

Basis-specific interpretation:

    Z-basis measurement:
        X/bit syndrome is used for correction.

    X-basis measurement:
        Z/phase syndrome is used for correction.

The complete protocol performs syndrome extraction after raw check-block
QBER calculation and before logical payload decoding.

Research boundary:

This module supports the scalable syndrome-level simulator. The optional
Qiskit helper only converts already-produced circuit measurement results
into syndrome records.

Security boundary:

Eve's configured attack fraction, attack mode, and attacked-mask values
must never be included in GP feature extraction.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .steane_css import (
    PhysicalBlock,
    STEANE_BLOCK_SIZE,
    STEANE_H,
    SteaneEncodedFrame,
    correction_position,
    validate_physical_block,
)


SYNDROME_LENGTH = 3

SUPPORTED_BASES = (
    "Z",
    "X",
)


class SyndromeExtractionError(Exception):
    """Base exception for syndrome-extraction failures."""


class InvalidSyndromeError(SyndromeExtractionError):
    """Raised when a syndrome is malformed."""


class InvalidReceivedFrameError(SyndromeExtractionError):
    """Raised when a received physical frame is invalid."""


class InvalidCircuitResultError(SyndromeExtractionError):
    """Raised when a Qiskit result dictionary is malformed."""


def validate_binary_vector(
    vector: Any,
    field_name: str,
    expected_length: int,
) -> np.ndarray:
    """
    Validate and normalize a one-dimensional binary vector.
    """

    if isinstance(expected_length, bool) or not isinstance(
        expected_length,
        int,
    ):
        raise TypeError(
            "expected_length must be an integer."
        )

    if expected_length <= 0:
        raise ValueError(
            "expected_length must be greater than zero."
        )

    try:
        normalized = np.asarray(
            vector,
            dtype=np.int8,
        )

    except Exception as error:
        raise InvalidSyndromeError(
            f"{field_name} cannot be converted to a binary array."
        ) from error

    if normalized.ndim != 1:
        raise InvalidSyndromeError(
            f"{field_name} must be one-dimensional."
        )

    if len(normalized) != expected_length:
        raise InvalidSyndromeError(
            f"{field_name} must contain exactly "
            f"{expected_length} bits."
        )

    if not np.all(
        np.isin(
            normalized,
            [0, 1],
        )
    ):
        raise InvalidSyndromeError(
            f"{field_name} must contain only 0 or 1."
        )

    return normalized.copy()


def validate_basis(
    basis: Any,
) -> str:
    """
    Validate a Z- or X-basis label.
    """

    if not isinstance(basis, str):
        raise TypeError(
            "basis must be a string."
        )

    normalized_basis = basis.strip().upper()

    if normalized_basis not in SUPPORTED_BASES:
        raise ValueError(
            "basis must be either 'Z' or 'X'."
        )

    return normalized_basis


def syndrome_as_tuple(
    syndrome: Any,
) -> tuple[int, int, int]:
    """
    Convert a three-bit syndrome into a tuple.
    """

    normalized = validate_binary_vector(
        vector=syndrome,
        field_name="syndrome",
        expected_length=SYNDROME_LENGTH,
    )

    return (
        int(normalized[0]),
        int(normalized[1]),
        int(normalized[2]),
    )


def syndrome_weight(
    syndrome: Any,
) -> int:
    """
    Return the Hamming weight of one syndrome.
    """

    normalized = validate_binary_vector(
        vector=syndrome,
        field_name="syndrome",
        expected_length=SYNDROME_LENGTH,
    )

    return int(
        np.sum(normalized)
    )


def steane_syndrome(
    error_vector: Any,
    use_css: bool = True,
) -> np.ndarray:
    """
    Calculate a three-bit Steane syndrome.

    For a CSS block:

        syndrome = H @ error_vector mod 2

    For the no-CSS baseline, a zero syndrome is returned because
    no seven-qubit stabilizer code is available.
    """

    if not isinstance(use_css, bool):
        raise TypeError(
            "use_css must be boolean."
        )

    if not use_css:
        return np.zeros(
            SYNDROME_LENGTH,
            dtype=np.int8,
        )

    normalized_error = validate_binary_vector(
        vector=error_vector,
        field_name="error_vector",
        expected_length=STEANE_BLOCK_SIZE,
    )

    return (
        (
            STEANE_H
            @ normalized_error
        )
        % 2
    ).astype(
        np.int8
    )


@dataclass(frozen=True)
class SyndromeRecord:
    """
    Syndrome information for one received physical block.

    Attributes:
        block_id:
            Logical payload/check block identifier.

        role:
            Either payload or check.

        basis:
            Declared preparation or measurement basis.

        use_css:
            Whether the block uses Steane CSS encoding.

        x_syndrome:
            Syndrome generated from the X-error vector.

        z_syndrome:
            Syndrome generated from the Z-error vector.

        x_error_weight:
            Simulator-side X-error-vector weight.

        z_error_weight:
            Simulator-side Z-error-vector weight.

        physical_error_weight:
            Number of physical positions containing any X or Z error.

        erased:
            Whether at least one physical position was erased.

        erasure_count:
            Number of erased physical positions.
    """

    block_id: str
    role: str
    basis: str
    use_css: bool
    x_syndrome: tuple[int, int, int]
    z_syndrome: tuple[int, int, int]
    x_error_weight: int
    z_error_weight: int
    physical_error_weight: int
    erased: bool
    erasure_count: int

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

        validate_basis(
            self.basis
        )

        if not isinstance(self.use_css, bool):
            raise TypeError(
                "use_css must be boolean."
            )

        syndrome_as_tuple(
            self.x_syndrome
        )

        syndrome_as_tuple(
            self.z_syndrome
        )

        for field_name, value in (
            (
                "x_error_weight",
                self.x_error_weight,
            ),
            (
                "z_error_weight",
                self.z_error_weight,
            ),
            (
                "physical_error_weight",
                self.physical_error_weight,
            ),
            (
                "erasure_count",
                self.erasure_count,
            ),
        ):
            if isinstance(value, bool) or not isinstance(
                value,
                int,
            ):
                raise TypeError(
                    f"{field_name} must be an integer."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if not isinstance(self.erased, bool):
            raise TypeError(
                "erased must be boolean."
            )

        if self.erased != (
            self.erasure_count > 0
        ):
            raise ValueError(
                "erased is inconsistent with erasure_count."
            )

    @property
    def x_syndrome_weight(self) -> int:
        """Return the X-syndrome Hamming weight."""

        return sum(
            self.x_syndrome
        )

    @property
    def z_syndrome_weight(self) -> int:
        """Return the Z-syndrome Hamming weight."""

        return sum(
            self.z_syndrome
        )

    @property
    def combined_syndrome_weight(self) -> int:
        """Return combined X- and Z-syndrome weight."""

        return (
            self.x_syndrome_weight
            + self.z_syndrome_weight
        )

    @property
    def has_nonzero_syndrome(self) -> bool:
        """Return whether either syndrome is nonzero."""

        return self.combined_syndrome_weight > 0

    @property
    def x_correction_position(self) -> int | None:
        """
        Return the position indicated by the X syndrome.
        """

        return correction_position(
            self.x_syndrome
        )

    @property
    def z_correction_position(self) -> int | None:
        """
        Return the position indicated by the Z syndrome.
        """

        return correction_position(
            self.z_syndrome
        )

    def relevant_syndrome(
        self,
        measurement_basis: str | None = None,
    ) -> tuple[int, int, int]:
        """
        Return the syndrome relevant to a measurement basis.

        Z-basis measurement:
            X syndrome corrects bit errors.

        X-basis measurement:
            Z syndrome corrects phase errors.
        """

        selected_basis = validate_basis(
            measurement_basis
            if measurement_basis is not None
            else self.basis
        )

        if selected_basis == "Z":
            return self.x_syndrome

        return self.z_syndrome

    def relevant_correction_position(
        self,
        measurement_basis: str | None = None,
    ) -> int | None:
        """
        Return the correction position for the relevant syndrome.
        """

        return correction_position(
            self.relevant_syndrome(
                measurement_basis
            )
        )

    def to_dictionary(
        self,
        include_simulator_error_weights: bool = True,
    ) -> dict[str, Any]:
        """
        Return a serializable syndrome record.

        Error-vector weights are internal simulator diagnostics and
        can be excluded from receiver-visible records.
        """

        result: dict[str, Any] = {
            "block_id":
                self.block_id,
            "role":
                self.role,
            "basis":
                self.basis,
            "use_css":
                self.use_css,
            "x_syndrome":
                list(self.x_syndrome),
            "z_syndrome":
                list(self.z_syndrome),
            "x_syndrome_weight":
                self.x_syndrome_weight,
            "z_syndrome_weight":
                self.z_syndrome_weight,
            "syndrome_weight":
                self.combined_syndrome_weight,
            "erased":
                self.erased,
            "erasure_count":
                self.erasure_count,
        }

        if include_simulator_error_weights:
            result.update(
                {
                    "x_error_weight":
                        self.x_error_weight,
                    "z_error_weight":
                        self.z_error_weight,
                    "physical_error_weight":
                        self.physical_error_weight,
                }
            )

        return result


@dataclass(frozen=True)
class SyndromeSummary:
    """
    Aggregate syndrome statistics for one received frame.
    """

    total_blocks: int
    css_blocks: int
    nonzero_syndrome_blocks: int
    zero_syndrome_blocks: int
    erased_blocks: int
    erased_physical_positions: int
    total_x_syndrome_weight: int
    total_z_syndrome_weight: int
    total_syndrome_weight: int
    mean_syndrome_weight: float
    nonzero_syndrome_rate: float
    erasure_block_rate: float

    def to_dictionary(self) -> dict[str, Any]:
        """Return aggregate syndrome statistics."""

        return {
            "total_blocks":
                self.total_blocks,
            "css_blocks":
                self.css_blocks,
            "nonzero_syndrome_blocks":
                self.nonzero_syndrome_blocks,
            "zero_syndrome_blocks":
                self.zero_syndrome_blocks,
            "erased_blocks":
                self.erased_blocks,
            "erased_physical_positions":
                self.erased_physical_positions,
            "total_x_syndrome_weight":
                self.total_x_syndrome_weight,
            "total_z_syndrome_weight":
                self.total_z_syndrome_weight,
            "total_syndrome_weight":
                self.total_syndrome_weight,
            "mean_syndrome_weight":
                self.mean_syndrome_weight,
            "nonzero_syndrome_rate":
                self.nonzero_syndrome_rate,
            "erasure_block_rate":
                self.erasure_block_rate,
        }


@dataclass(frozen=True)
class CircuitSyndromeRecord:
    """
    Syndrome information from one representative Qiskit shot.
    """

    shot_index: int
    receiver_bits: tuple[int, ...]
    x_syndrome: tuple[int, int, int]
    z_syndrome: tuple[int, int, int]
    eve_bits: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.shot_index, bool) or not isinstance(
            self.shot_index,
            int,
        ):
            raise TypeError(
                "shot_index must be an integer."
            )

        if self.shot_index < 0:
            raise ValueError(
                "shot_index cannot be negative."
            )

        validate_binary_vector(
            self.receiver_bits,
            "receiver_bits",
            STEANE_BLOCK_SIZE,
        )

        syndrome_as_tuple(
            self.x_syndrome
        )

        syndrome_as_tuple(
            self.z_syndrome
        )

        if self.eve_bits is not None:
            validate_binary_vector(
                self.eve_bits,
                "eve_bits",
                STEANE_BLOCK_SIZE,
            )

    @property
    def x_syndrome_weight(self) -> int:
        """Return X-syndrome weight."""

        return sum(
            self.x_syndrome
        )

    @property
    def z_syndrome_weight(self) -> int:
        """Return Z-syndrome weight."""

        return sum(
            self.z_syndrome
        )

    @property
    def combined_syndrome_weight(self) -> int:
        """Return combined syndrome weight."""

        return (
            self.x_syndrome_weight
            + self.z_syndrome_weight
        )

    def to_dictionary(
        self,
        include_eve_bits: bool = False,
    ) -> dict[str, Any]:
        """
        Return a serializable circuit record.

        Eve measurement bits are hidden by default.
        """

        result: dict[str, Any] = {
            "shot_index":
                self.shot_index,
            "receiver_bits":
                list(self.receiver_bits),
            "x_syndrome":
                list(self.x_syndrome),
            "z_syndrome":
                list(self.z_syndrome),
            "x_syndrome_weight":
                self.x_syndrome_weight,
            "z_syndrome_weight":
                self.z_syndrome_weight,
            "syndrome_weight":
                self.combined_syndrome_weight,
        }

        if (
            include_eve_bits
            and self.eve_bits is not None
        ):
            result["eve_bits"] = list(
                self.eve_bits
            )

        return result


def extract_block_syndrome(
    block: PhysicalBlock,
) -> SyndromeRecord:
    """
    Extract X and Z syndromes from one received block.
    """

    validate_physical_block(
        block
    )

    x_syndrome_array = steane_syndrome(
        error_vector=block.x_errors,
        use_css=block.use_css,
    )

    z_syndrome_array = steane_syndrome(
        error_vector=block.z_errors,
        use_css=block.use_css,
    )

    physical_error_mask = np.logical_or(
        block.x_errors.astype(bool),
        block.z_errors.astype(bool),
    )

    erasure_count = int(
        np.count_nonzero(
            block.erasures
        )
    )

    return SyndromeRecord(
        block_id=block.block_id,
        role=block.role,
        basis=block.basis,
        use_css=block.use_css,
        x_syndrome=syndrome_as_tuple(
            x_syndrome_array
        ),
        z_syndrome=syndrome_as_tuple(
            z_syndrome_array
        ),
        x_error_weight=int(
            np.count_nonzero(
                block.x_errors
            )
        ),
        z_error_weight=int(
            np.count_nonzero(
                block.z_errors
            )
        ),
        physical_error_weight=int(
            np.count_nonzero(
                physical_error_mask
            )
        ),
        erased=erasure_count > 0,
        erasure_count=erasure_count,
    )


def normalize_received_frame(
    received_frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> list[PhysicalBlock]:
    """
    Normalize a Steane frame or a physical-block sequence.
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
        raise InvalidReceivedFrameError(
            "received_frame cannot be empty."
        )

    seen_ids: set[str] = set()

    for block in blocks:
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
            raise InvalidReceivedFrameError(
                "Duplicate physical block ID: "
                f"{block.block_id!r}."
            )

        seen_ids.add(
            block.block_id
        )

    return blocks


def extract_syndrome_records(
    received_frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> list[SyndromeRecord]:
    """
    Extract structured syndrome records for a received frame.
    """

    blocks = normalize_received_frame(
        received_frame
    )

    return [
        extract_block_syndrome(
            block
        )
        for block in blocks
    ]


def extract_syndrome_statistics(
    received_frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> list[dict[str, Any]]:
    """
    Notebook-compatible syndrome-statistics function.

    Each result contains:

        block_id
        x_syndrome
        z_syndrome
        syndrome_weight
        x_error_weight
        z_error_weight
        erased
    """

    records = extract_syndrome_records(
        received_frame
    )

    return [
        {
            "block_id":
                record.block_id,
            "x_syndrome":
                np.asarray(
                    record.x_syndrome,
                    dtype=np.int8,
                ),
            "z_syndrome":
                np.asarray(
                    record.z_syndrome,
                    dtype=np.int8,
                ),
            "syndrome_weight":
                record.combined_syndrome_weight,
            "x_error_weight":
                record.x_error_weight,
            "z_error_weight":
                record.z_error_weight,
            "erased":
                record.erased,
        }
        for record in records
    ]


def summarize_syndrome_records(
    records: Sequence[SyndromeRecord],
) -> SyndromeSummary:
    """
    Calculate aggregate observable syndrome statistics.
    """

    if isinstance(
        records,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "records must be a sequence."
        )

    if not isinstance(
        records,
        Sequence,
    ):
        raise TypeError(
            "records must be a sequence."
        )

    if not records:
        raise ValueError(
            "records cannot be empty."
        )

    for record in records:
        if not isinstance(
            record,
            SyndromeRecord,
        ):
            raise TypeError(
                "Every item must be a SyndromeRecord."
            )

    total_blocks = len(
        records
    )

    css_blocks = sum(
        record.use_css
        for record in records
    )

    nonzero_blocks = sum(
        record.has_nonzero_syndrome
        for record in records
    )

    erased_blocks = sum(
        record.erased
        for record in records
    )

    erased_positions = sum(
        record.erasure_count
        for record in records
    )

    total_x_weight = sum(
        record.x_syndrome_weight
        for record in records
    )

    total_z_weight = sum(
        record.z_syndrome_weight
        for record in records
    )

    total_weight = (
        total_x_weight
        + total_z_weight
    )

    return SyndromeSummary(
        total_blocks=total_blocks,
        css_blocks=css_blocks,
        nonzero_syndrome_blocks=nonzero_blocks,
        zero_syndrome_blocks=(
            total_blocks
            - nonzero_blocks
        ),
        erased_blocks=erased_blocks,
        erased_physical_positions=(
            erased_positions
        ),
        total_x_syndrome_weight=(
            total_x_weight
        ),
        total_z_syndrome_weight=(
            total_z_weight
        ),
        total_syndrome_weight=(
            total_weight
        ),
        mean_syndrome_weight=float(
            total_weight
            / total_blocks
        ),
        nonzero_syndrome_rate=float(
            nonzero_blocks
            / total_blocks
        ),
        erasure_block_rate=float(
            erased_blocks
            / total_blocks
        ),
    )


def syndrome_position_histogram(
    records: Sequence[SyndromeRecord],
    syndrome_type: str,
) -> dict[str, int]:
    """
    Count syndrome-indicated physical positions.

    syndrome_type must be either "X" or "Z".
    """

    normalized_type = validate_basis(
        syndrome_type
    )

    if normalized_type == "Z":
        selected_positions = [
            record.z_correction_position
            for record in records
        ]
    else:
        selected_positions = [
            record.x_correction_position
            for record in records
        ]

    counter = Counter(
        "zero"
        if position == -1
        else "unknown"
        if position is None
        else str(position)
        for position in selected_positions
    )

    return dict(
        sorted(
            counter.items()
        )
    )


def relevant_syndrome_for_basis(
    record: SyndromeRecord,
    measurement_basis: str,
) -> tuple[int, int, int]:
    """
    Return the error syndrome relevant to the declared basis.
    """

    if not isinstance(
        record,
        SyndromeRecord,
    ):
        raise TypeError(
            "record must be a SyndromeRecord."
        )

    return record.relevant_syndrome(
        measurement_basis
    )


def relevant_correction_position_for_basis(
    record: SyndromeRecord,
    measurement_basis: str,
) -> int | None:
    """
    Return the correction position relevant to the basis.
    """

    if not isinstance(
        record,
        SyndromeRecord,
    ):
        raise TypeError(
            "record must be a SyndromeRecord."
        )

    return record.relevant_correction_position(
        measurement_basis
    )


def normalize_circuit_bits(
    value: Any,
    field_name: str,
    expected_length: int,
) -> tuple[int, ...]:
    """
    Normalize Qiskit measurement bits.
    """

    normalized = validate_binary_vector(
        vector=value,
        field_name=field_name,
        expected_length=expected_length,
    )

    return tuple(
        int(bit)
        for bit in normalized.tolist()
    )


def extract_qiskit_syndrome_record_objects(
    execution_result: Mapping[str, Any],
    include_eve_bits: bool = False,
) -> list[CircuitSyndromeRecord]:
    """
    Convert representative Qiskit shots into structured records.

    Expected input:

        {
            "shots": [
                {
                    "receiver_bits": [...7 bits...],
                    "x_syndrome": [...3 bits...],
                    "z_syndrome": [...3 bits...],
                    "eve_bits": [...7 bits...]
                }
            ]
        }

    Eve bits are ignored unless include_eve_bits=True.
    """

    if not isinstance(
        execution_result,
        Mapping,
    ):
        raise TypeError(
            "execution_result must be a mapping."
        )

    shots = execution_result.get(
        "shots"
    )

    if not isinstance(
        shots,
        list,
    ):
        raise InvalidCircuitResultError(
            "execution_result must contain a shots list."
        )

    records: list[
        CircuitSyndromeRecord
    ] = []

    for shot_index, shot in enumerate(
        shots
    ):
        if not isinstance(
            shot,
            Mapping,
        ):
            raise InvalidCircuitResultError(
                f"Shot {shot_index} must be a mapping."
            )

        required_fields = {
            "receiver_bits",
            "x_syndrome",
            "z_syndrome",
        }

        missing_fields = required_fields.difference(
            shot.keys()
        )

        if missing_fields:
            raise InvalidCircuitResultError(
                f"Shot {shot_index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        eve_bits: tuple[int, ...] | None = None

        if include_eve_bits:
            raw_eve_bits = shot.get(
                "eve_bits"
            )

            if raw_eve_bits is not None:
                eve_bits = normalize_circuit_bits(
                    raw_eve_bits,
                    "eve_bits",
                    STEANE_BLOCK_SIZE,
                )

        records.append(
            CircuitSyndromeRecord(
                shot_index=shot_index,
                receiver_bits=(
                    normalize_circuit_bits(
                        shot["receiver_bits"],
                        "receiver_bits",
                        STEANE_BLOCK_SIZE,
                    )
                ),
                x_syndrome=(
                    syndrome_as_tuple(
                        shot["x_syndrome"]
                    )
                ),
                z_syndrome=(
                    syndrome_as_tuple(
                        shot["z_syndrome"]
                    )
                ),
                eve_bits=eve_bits,
            )
        )

    return records


def extract_qiskit_syndrome_records(
    execution_result: Mapping[str, Any],
    include_eve_bits: bool = False,
) -> list[dict[str, Any]]:
    """
    Notebook-compatible Qiskit syndrome-record function.

    Hidden Eve measurement bits are excluded by default.
    """

    records = extract_qiskit_syndrome_record_objects(
        execution_result=execution_result,
        include_eve_bits=include_eve_bits,
    )

    return [
        record.to_dictionary(
            include_eve_bits=include_eve_bits
        )
        for record in records
    ]


def run_self_test() -> None:
    """
    Verify syndrome extraction and basis-specific interpretation.
    """

    from .logical_qubit import (
        create_check_logical_qubit,
        create_payload_logical_qubit,
    )
    from .steane_css import (
        encode_one_logical_qubit,
    )

    print("=" * 72)
    print("FT-QuPAP Syndrome Extraction Self-Test")
    print("=" * 72)

    payload_spec = (
        create_payload_logical_qubit(
            logical_bit=0,
            logical_index=0,
            position=0,
        )
    )

    z_check_spec = (
        create_check_logical_qubit(
            logical_bit=1,
            basis="Z",
            logical_index=0,
            position=1,
        )
    )

    x_check_spec = (
        create_check_logical_qubit(
            logical_bit=0,
            basis="X",
            logical_index=1,
            position=2,
        )
    )

    clean_block = (
        encode_one_logical_qubit(
            spec=payload_spec,
            use_css=True,
            rng=np.random.default_rng(
                1001
            ),
        )
    )

    x_error_block = (
        encode_one_logical_qubit(
            spec=z_check_spec,
            use_css=True,
            rng=np.random.default_rng(
                1002
            ),
        )
    )

    z_error_block = (
        encode_one_logical_qubit(
            spec=x_check_spec,
            use_css=True,
            rng=np.random.default_rng(
                1003
            ),
        )
    )

    x_error_block.x_errors[4] = 1
    z_error_block.z_errors[2] = 1
    z_error_block.erasures[6] = True

    records = extract_syndrome_records(
        [
            clean_block,
            x_error_block,
            z_error_block,
        ]
    )

    summary = summarize_syndrome_records(
        records
    )

    clean_record = records[0]
    x_record = records[1]
    z_record = records[2]

    x_position_correct = (
        x_record.x_correction_position
        == 4
    )

    z_position_correct = (
        z_record.z_correction_position
        == 2
    )

    z_basis_uses_x_syndrome = (
        x_record.relevant_syndrome("Z")
        == x_record.x_syndrome
    )

    x_basis_uses_z_syndrome = (
        z_record.relevant_syndrome("X")
        == z_record.z_syndrome
    )

    notebook_records = (
        extract_syndrome_statistics(
            [
                clean_block,
                x_error_block,
                z_error_block,
            ]
        )
    )

    circuit_result = {
        "shots": [
            {
                "receiver_bits":
                    [0, 1, 0, 1, 0, 1, 0],
                "x_syndrome":
                    [1, 0, 1],
                "z_syndrome":
                    [0, 0, 0],
                "eve_bits":
                    [1, 1, 1, 1, 1, 1, 1],
            }
        ]
    }

    public_circuit_records = (
        extract_qiskit_syndrome_records(
            circuit_result,
            include_eve_bits=False,
        )
    )

    eve_hidden = (
        "eve_bits"
        not in public_circuit_records[0]
    )

    print(
        "Syndrome records extracted: "
        f"{len(records)}"
    )

    print(
        "Clean block syndrome      : "
        f"{clean_record.combined_syndrome_weight}"
    )

    print(
        "X-error syndrome          : "
        f"{list(x_record.x_syndrome)}"
    )

    print(
        "X-error located position  : "
        f"{x_record.x_correction_position}"
    )

    print(
        "Z-error syndrome          : "
        f"{list(z_record.z_syndrome)}"
    )

    print(
        "Z-error located position  : "
        f"{z_record.z_correction_position}"
    )

    print(
        "Z basis uses X syndrome   : "
        f"{z_basis_uses_x_syndrome}"
    )

    print(
        "X basis uses Z syndrome   : "
        f"{x_basis_uses_z_syndrome}"
    )

    print(
        "Erased blocks             : "
        f"{summary.erased_blocks}"
    )

    print(
        "Nonzero-syndrome blocks   : "
        f"{summary.nonzero_syndrome_blocks}"
    )

    print(
        "Notebook record count     : "
        f"{len(notebook_records)}"
    )

    print(
        "Public circuit hides Eve  : "
        f"{eve_hidden}"
    )

    if len(records) != 3:
        raise SyndromeExtractionError(
            "Incorrect syndrome-record count."
        )

    if clean_record.has_nonzero_syndrome:
        raise SyndromeExtractionError(
            "Clean block produced a nonzero syndrome."
        )

    if not x_position_correct:
        raise SyndromeExtractionError(
            "X syndrome located the wrong position."
        )

    if not z_position_correct:
        raise SyndromeExtractionError(
            "Z syndrome located the wrong position."
        )

    if not z_basis_uses_x_syndrome:
        raise SyndromeExtractionError(
            "Z-basis correction selected the wrong syndrome."
        )

    if not x_basis_uses_z_syndrome:
        raise SyndromeExtractionError(
            "X-basis correction selected the wrong syndrome."
        )

    if summary.erased_blocks != 1:
        raise SyndromeExtractionError(
            "Erasure counting failed."
        )

    if summary.nonzero_syndrome_blocks != 2:
        raise SyndromeExtractionError(
            "Nonzero-syndrome counting failed."
        )

    if not eve_hidden:
        raise SyndromeExtractionError(
            "Public Qiskit record exposed Eve data."
        )

    print(
        "\nSyndrome extraction self-test "
        "completed successfully."
    )


__all__ = [
    "SYNDROME_LENGTH",
    "SUPPORTED_BASES",
    "SyndromeExtractionError",
    "InvalidSyndromeError",
    "InvalidReceivedFrameError",
    "InvalidCircuitResultError",
    "SyndromeRecord",
    "SyndromeSummary",
    "CircuitSyndromeRecord",
    "validate_binary_vector",
    "validate_basis",
    "syndrome_as_tuple",
    "syndrome_weight",
    "steane_syndrome",
    "extract_block_syndrome",
    "normalize_received_frame",
    "extract_syndrome_records",
    "extract_syndrome_statistics",
    "summarize_syndrome_records",
    "syndrome_position_histogram",
    "relevant_syndrome_for_basis",
    "relevant_correction_position_for_basis",
    "extract_qiskit_syndrome_record_objects",
    "extract_qiskit_syndrome_records",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        SyndromeExtractionError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[SYNDROME EXTRACTION ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error