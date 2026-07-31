"""
Steane CSS Encoder
FT-QuPAP Mobile Station

This module implements the scalable syndrome-level Steane [[7,1,3]]
encoding model used by the FT-QuPAP notebook.

Standard FT-QuPAP frame:

    128 payload logical blocks
     32 check logical blocks
    -------------------------
    160 total logical blocks

Each logical block is encoded into seven physical qubits:

    160 × 7 = 1120 physical qubits

Important research boundary:
    This is a syndrome-level CSS simulation for complete protocol
    sessions. It is not a hardware-level fault-tolerant quantum
    computer implementation.

Representative Qiskit circuits should be used separately to validate
selected Steane block operations.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from .control_schedule import LogicalSpec
except ImportError:
    from control_schedule import LogicalSpec


# ============================================================
# FT-QuPAP frame configuration
# ============================================================

PAYLOAD_LOGICAL_BLOCK_COUNT = 128
CHECK_LOGICAL_BLOCK_COUNT = 32

TOTAL_LOGICAL_BLOCK_COUNT = (
    PAYLOAD_LOGICAL_BLOCK_COUNT
    + CHECK_LOGICAL_BLOCK_COUNT
)

STEANE_BLOCK_SIZE = 7

TOTAL_PHYSICAL_QUBIT_COUNT = (
    TOTAL_LOGICAL_BLOCK_COUNT
    * STEANE_BLOCK_SIZE
)

SUPPORTED_BASES = (
    "Z",
    "X",
)


# ============================================================
# Steane [[7,1,3]] syndrome-level configuration
# ============================================================

STEANE_H = np.array(
    [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.int8,
)

STEANE_LOGICAL_X = np.ones(
    STEANE_BLOCK_SIZE,
    dtype=np.int8,
)


class SteaneEncodingError(Exception):
    """Base exception for Steane encoding failures."""


class LogicalSpecificationError(
    SteaneEncodingError
):
    """Raised when a logical specification is invalid."""


class PhysicalBlockValidationError(
    SteaneEncodingError
):
    """Raised when an encoded physical block is invalid."""


class SteaneFrameValidationError(
    SteaneEncodingError
):
    """Raised when the complete encoded frame is invalid."""


# Backward-compatible type name used by __init__.py.
LogicalQubitSpec = LogicalSpec


def binary_span(
    generators: np.ndarray,
) -> np.ndarray:
    """
    Generate the binary span of a generator matrix.

    The notebook applies this function to STEANE_H to construct the
    eight C2 codewords used by the syndrome-level encoder.
    """

    if not isinstance(
        generators,
        np.ndarray,
    ):
        raise TypeError(
            "generators must be a NumPy array."
        )

    if generators.ndim != 2:
        raise ValueError(
            "generators must be a two-dimensional array."
        )

    if generators.shape[1] == 0:
        raise ValueError(
            "generators cannot have zero columns."
        )

    normalized = (
        generators.astype(
            np.int8,
            copy=True,
        )
    )

    if not np.all(
        np.isin(
            normalized,
            [0, 1],
        )
    ):
        raise ValueError(
            "generators must contain only binary values."
        )

    words: list[np.ndarray] = []

    for mask in range(
        2 ** len(normalized)
    ):
        word = np.zeros(
            normalized.shape[1],
            dtype=np.int8,
        )

        for index in range(
            len(normalized)
        ):
            if (
                mask >> index
            ) & 1:
                word ^= normalized[index]

        words.append(word)

    return np.unique(
        np.array(
            words,
            dtype=np.int8,
        ),
        axis=0,
    )


STEANE_C2_CODEWORDS = binary_span(
    STEANE_H
)


def syndrome_tuple(
    vector: np.ndarray,
) -> tuple[int, int, int]:
    """
    Convert a three-bit syndrome vector to a tuple.
    """

    if not isinstance(vector, np.ndarray):
        raise TypeError(
            "vector must be a NumPy array."
        )

    if vector.shape != (3,):
        raise ValueError(
            "A Steane syndrome must contain three bits."
        )

    if not np.all(
        np.isin(vector, [0, 1])
    ):
        raise ValueError(
            "Syndrome values must be binary."
        )

    return tuple(
        int(value)
        for value in vector.tolist()
    )


STEANE_SYNDROME_TO_POSITION: dict[
    tuple[int, int, int],
    int,
] = {
    syndrome_tuple(
        STEANE_H[:, position]
    ): position
    for position in range(
        STEANE_H.shape[1]
    )
}

STEANE_SYNDROME_TO_POSITION[
    (0, 0, 0)
] = -1


@dataclass
class PhysicalBlock:
    """
    One encoded FT-QuPAP physical block.

    Attributes:
        spec:
            Deep-copied logical payload/check specification.

        reference_bits:
            Expected physical measurement/reference pattern.

        use_css:
            True for a seven-qubit Steane block.
            False for a one-bit no-CSS baseline.

        x_errors:
            Physical X-error indicators.

        z_errors:
            Physical Z-error indicators.

        erasures:
            Lost or erased physical-qubit indicators.

        attacked_mask:
            Simulator-only record showing which physical positions
            were intercepted.

    The Authentication Server's GP feature vector must never receive
    attacked_mask because it represents hidden simulator knowledge.
    """

    spec: LogicalSpec
    reference_bits: np.ndarray
    use_css: bool = True

    x_errors: np.ndarray = field(
        default_factory=lambda: np.array(
            [],
            dtype=np.int8,
        )
    )

    z_errors: np.ndarray = field(
        default_factory=lambda: np.array(
            [],
            dtype=np.int8,
        )
    )

    erasures: np.ndarray = field(
        default_factory=lambda: np.array(
            [],
            dtype=bool,
        )
    )

    attacked_mask: np.ndarray = field(
        default_factory=lambda: np.array(
            [],
            dtype=bool,
        )
    )

    def __post_init__(self) -> None:
        self.reference_bits = np.asarray(
            self.reference_bits,
            dtype=np.int8,
        ).copy()

        self.x_errors = np.asarray(
            self.x_errors,
            dtype=np.int8,
        ).copy()

        self.z_errors = np.asarray(
            self.z_errors,
            dtype=np.int8,
        ).copy()

        self.erasures = np.asarray(
            self.erasures,
            dtype=bool,
        ).copy()

        self.attacked_mask = np.asarray(
            self.attacked_mask,
            dtype=bool,
        ).copy()

        validate_physical_block(self)

    @property
    def physical_qubit_count(self) -> int:
        """Return the number of physical positions."""

        return len(self.reference_bits)

    @property
    def block_id(self) -> str:
        """Return the logical block identifier."""

        return self.spec.block_id

    @property
    def role(self) -> str:
        """Return payload or check role."""

        return self.spec.role

    def safe_summary(self) -> dict[str, Any]:
        """
        Return non-sensitive encoded-block metadata.
        """

        return {
            "block_id": self.spec.block_id,
            "role": self.spec.role,
            "logical_index":
                self.spec.logical_index,
            "basis": self.spec.basis,
            "use_css": self.use_css,
            "physical_qubit_count":
                self.physical_qubit_count,
        }


@dataclass
class SteaneEncodedFrame:
    """
    Complete encoded FT-QuPAP transmission frame.

    Attributes:
        ordered_specs:
            Interleaved logical block specifications.

        payload_blocks:
            Encoded payload blocks.

        check_blocks:
            Encoded check blocks.

        frame:
            Physical blocks arranged in transmission order.

        use_css:
            Whether the Steane CSS configuration is active.
    """

    ordered_specs: list[LogicalSpec]
    payload_blocks: list[PhysicalBlock]
    check_blocks: list[PhysicalBlock]
    frame: list[PhysicalBlock]
    use_css: bool = True

    def __post_init__(self) -> None:
        validate_encoded_frame(self)

    @property
    def physical_blocks(
        self,
    ) -> list[PhysicalBlock]:
        """
        Alias used by quantum-transmission modules.
        """

        return self.frame

    @property
    def transmission_frame(
        self,
    ) -> list[PhysicalBlock]:
        """
        Alias for the ordered physical frame.
        """

        return self.frame

    @property
    def blocks(
        self,
    ) -> list[PhysicalBlock]:
        """Additional frame alias."""

        return self.frame

    @property
    def logical_block_count(self) -> int:
        """Return the number of logical blocks."""

        return len(self.frame)

    @property
    def total_physical_qubits(self) -> int:
        """Return total physical positions in the frame."""

        return sum(
            len(block.reference_bits)
            for block in self.frame
        )

    @property
    def physical_qubits_per_block(
        self,
    ) -> int:
        """Return the expected block expansion."""

        return (
            STEANE_BLOCK_SIZE
            if self.use_css
            else 1
        )

    def safe_summary(self) -> dict[str, Any]:
        """Return public resource information."""

        return {
            "encoder":
                "Steane [[7,1,3]]"
                if self.use_css
                else "No-CSS baseline",
            "payload_logical_blocks":
                len(self.payload_blocks),
            "check_logical_blocks":
                len(self.check_blocks),
            "total_logical_blocks":
                self.logical_block_count,
            "physical_qubits_per_block":
                self.physical_qubits_per_block,
            "total_physical_qubits":
                self.total_physical_qubits,
            "css_expansion_factor":
                self.physical_qubits_per_block,
        }


def validate_logical_spec(
    spec: Any,
) -> None:
    """
    Validate one control_schedule.LogicalSpec-compatible object.
    """

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
            raise LogicalSpecificationError(
                "Logical specification is missing "
                f"{attribute!r}."
            )

    if not isinstance(spec.block_id, str):
        raise TypeError(
            "spec.block_id must be a string."
        )

    if not spec.block_id:
        raise ValueError(
            "spec.block_id cannot be empty."
        )

    if spec.role not in (
        "payload",
        "check",
    ):
        raise LogicalSpecificationError(
            "spec.role must be 'payload' or 'check'."
        )

    if isinstance(
        spec.logical_index,
        bool,
    ) or not isinstance(
        spec.logical_index,
        int,
    ):
        raise TypeError(
            "spec.logical_index must be an integer."
        )

    if spec.logical_index < 0:
        raise ValueError(
            "spec.logical_index cannot be negative."
        )

    if spec.logical_bit not in (
        0,
        1,
    ):
        raise LogicalSpecificationError(
            "spec.logical_bit must be 0 or 1."
        )

    if spec.basis not in SUPPORTED_BASES:
        raise LogicalSpecificationError(
            "spec.basis must be 'Z' or 'X'."
        )

    if (
        spec.role == "payload"
        and spec.basis != "Z"
    ):
        raise LogicalSpecificationError(
            "Payload logical blocks must use the Z basis."
        )

    if spec.position is not None:
        if isinstance(
            spec.position,
            bool,
        ) or not isinstance(
            spec.position,
            int,
        ):
            raise TypeError(
                "spec.position must be an integer or None."
            )

        if spec.position < 0:
            raise ValueError(
                "spec.position cannot be negative."
            )


def validate_ordered_specs(
    ordered_specs: Sequence[Any],
    require_standard_counts: bool = True,
) -> None:
    """
    Validate an interleaved logical frame.
    """

    if isinstance(
        ordered_specs,
        (str, bytes, bytearray),
    ):
        raise TypeError(
            "ordered_specs must be a sequence."
        )

    if not isinstance(
        ordered_specs,
        Sequence,
    ):
        raise TypeError(
            "ordered_specs must be a sequence."
        )

    if len(ordered_specs) == 0:
        raise ValueError(
            "ordered_specs cannot be empty."
        )

    if (
        require_standard_counts
        and len(ordered_specs)
        != TOTAL_LOGICAL_BLOCK_COUNT
    ):
        raise SteaneFrameValidationError(
            "FT-QuPAP requires exactly "
            f"{TOTAL_LOGICAL_BLOCK_COUNT} "
            "interleaved logical blocks."
        )

    seen_block_ids: set[str] = set()
    used_positions: set[int] = set()

    payload_count = 0
    check_count = 0

    for expected_position, spec in enumerate(
        ordered_specs
    ):
        validate_logical_spec(spec)

        if spec.block_id in seen_block_ids:
            raise SteaneFrameValidationError(
                f"Duplicate logical block ID: "
                f"{spec.block_id!r}."
            )

        seen_block_ids.add(
            spec.block_id
        )

        if spec.role == "payload":
            payload_count += 1
        else:
            check_count += 1

        if spec.position is None:
            raise SteaneFrameValidationError(
                f"Block {spec.block_id!r} has no "
                "interleaved frame position."
            )

        if spec.position != expected_position:
            raise SteaneFrameValidationError(
                f"Block {spec.block_id!r} declares "
                f"position {spec.position}, but appears "
                f"at position {expected_position}."
            )

        if spec.position in used_positions:
            raise SteaneFrameValidationError(
                "Duplicate logical frame position."
            )

        used_positions.add(
            spec.position
        )

    if require_standard_counts:
        if payload_count != (
            PAYLOAD_LOGICAL_BLOCK_COUNT
        ):
            raise SteaneFrameValidationError(
                "Expected exactly "
                f"{PAYLOAD_LOGICAL_BLOCK_COUNT} "
                "payload logical blocks."
            )

        if check_count != (
            CHECK_LOGICAL_BLOCK_COUNT
        ):
            raise SteaneFrameValidationError(
                "Expected exactly "
                f"{CHECK_LOGICAL_BLOCK_COUNT} "
                "check logical blocks."
            )


def validate_binary_array(
    values: np.ndarray,
    field_name: str,
    expected_length: int,
) -> None:
    """Validate a one-dimensional binary array."""

    if not isinstance(values, np.ndarray):
        raise TypeError(
            f"{field_name} must be a NumPy array."
        )

    if values.ndim != 1:
        raise ValueError(
            f"{field_name} must be one-dimensional."
        )

    if len(values) != expected_length:
        raise ValueError(
            f"{field_name} must contain exactly "
            f"{expected_length} values."
        )

    if not np.all(
        np.isin(
            values.astype(np.int8),
            [0, 1],
        )
    ):
        raise ValueError(
            f"{field_name} must contain only 0 or 1."
        )


def validate_physical_block(
    block: PhysicalBlock,
) -> None:
    """Validate one encoded physical block."""

    if not isinstance(block, PhysicalBlock):
        raise TypeError(
            "block must be a PhysicalBlock."
        )

    validate_logical_spec(
        block.spec
    )

    if not isinstance(block.use_css, bool):
        raise TypeError(
            "block.use_css must be boolean."
        )

    expected_length = (
        STEANE_BLOCK_SIZE
        if block.use_css
        else 1
    )

    validate_binary_array(
        block.reference_bits,
        "reference_bits",
        expected_length,
    )

    validate_binary_array(
        block.x_errors,
        "x_errors",
        expected_length,
    )

    validate_binary_array(
        block.z_errors,
        "z_errors",
        expected_length,
    )

    if not isinstance(
        block.erasures,
        np.ndarray,
    ):
        raise TypeError(
            "erasures must be a NumPy array."
        )

    if (
        block.erasures.ndim != 1
        or len(block.erasures)
        != expected_length
    ):
        raise ValueError(
            "erasures has an invalid shape."
        )

    if not isinstance(
        block.attacked_mask,
        np.ndarray,
    ):
        raise TypeError(
            "attacked_mask must be a NumPy array."
        )

    if (
        block.attacked_mask.ndim != 1
        or len(block.attacked_mask)
        != expected_length
    ):
        raise ValueError(
            "attacked_mask has an invalid shape."
        )

    if block.use_css:
        decoded_bit = decode_steane_logical_bit(
            block.reference_bits
        )

        if decoded_bit != block.spec.logical_bit:
            raise PhysicalBlockValidationError(
                f"Encoded block {block.spec.block_id!r} "
                "does not represent its declared logical bit."
            )

    elif (
        int(block.reference_bits[0])
        != block.spec.logical_bit
    ):
        raise PhysicalBlockValidationError(
            "No-CSS reference bit does not match "
            "the logical specification."
        )


def encode_one_logical_qubit(
    spec: LogicalSpec,
    use_css: bool = True,
    rng: np.random.Generator | None = None,
) -> PhysicalBlock:
    """
    Encode one logical payload/check state.

    Notebook-aligned syndrome-level encoding:

        base_word = random C2 codeword

        reference_bits =
            base_word XOR
            (logical-X × logical_bit)

    For the no-CSS baseline, one physical bit is used.
    """

    validate_logical_spec(spec)

    if not isinstance(use_css, bool):
        raise TypeError(
            "use_css must be boolean."
        )

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    if use_css:
        codeword_index = int(
            rng.integers(
                0,
                len(
                    STEANE_C2_CODEWORDS
                ),
            )
        )

        base_word = (
            STEANE_C2_CODEWORDS[
                codeword_index
            ].copy()
        )

        reference_bits = (
            base_word
            ^ (
                STEANE_LOGICAL_X
                * int(spec.logical_bit)
            )
        ).astype(
            np.int8
        )

    else:
        reference_bits = np.array(
            [
                int(
                    spec.logical_bit
                )
            ],
            dtype=np.int8,
        )

    physical_count = len(
        reference_bits
    )

    return PhysicalBlock(
        spec=copy.deepcopy(spec),
        reference_bits=
            reference_bits,
        use_css=use_css,
        x_errors=np.zeros(
            physical_count,
            dtype=np.int8,
        ),
        z_errors=np.zeros(
            physical_count,
            dtype=np.int8,
        ),
        erasures=np.zeros(
            physical_count,
            dtype=bool,
        ),
        attacked_mask=np.zeros(
            physical_count,
            dtype=bool,
        ),
    )


def encode_payload_blocks(
    ordered_specs: Sequence[LogicalSpec],
    use_css: bool = True,
    rng: np.random.Generator | None = None,
) -> list[PhysicalBlock]:
    """
    Encode all logical KMAC payload blocks.
    """

    if rng is None:
        rng = np.random.default_rng()

    return [
        encode_one_logical_qubit(
            spec=spec,
            use_css=use_css,
            rng=rng,
        )
        for spec in ordered_specs
        if spec.role == "payload"
    ]


def encode_check_blocks(
    ordered_specs: Sequence[LogicalSpec],
    use_css: bool = True,
    rng: np.random.Generator | None = None,
) -> list[PhysicalBlock]:
    """
    Encode all independent logical check blocks.
    """

    if rng is None:
        rng = np.random.default_rng()

    return [
        encode_one_logical_qubit(
            spec=spec,
            use_css=use_css,
            rng=rng,
        )
        for spec in ordered_specs
        if spec.role == "check"
    ]


def frame_and_interleave_blocks(
    ordered_specs: Sequence[LogicalSpec],
    payload_blocks: Sequence[PhysicalBlock],
    check_blocks: Sequence[PhysicalBlock],
) -> list[PhysicalBlock]:
    """
    Reconstruct the physical transmission frame in schedule order.
    """

    all_blocks = (
        list(payload_blocks)
        + list(check_blocks)
    )

    blocks_by_id: dict[
        str,
        PhysicalBlock,
    ] = {}

    for block in all_blocks:
        validate_physical_block(block)

        block_id = block.spec.block_id

        if block_id in blocks_by_id:
            raise SteaneFrameValidationError(
                f"Duplicate encoded block ID: "
                f"{block_id!r}."
            )

        blocks_by_id[block_id] = block

    ordered_ids = [
        spec.block_id
        for spec in ordered_specs
    ]

    missing_ids = [
        block_id
        for block_id in ordered_ids
        if block_id not in blocks_by_id
    ]

    if missing_ids:
        raise SteaneFrameValidationError(
            "Encoded blocks are missing for IDs: "
            f"{missing_ids[:10]}"
        )

    extra_ids = set(
        blocks_by_id
    ).difference(
        ordered_ids
    )

    if extra_ids:
        raise SteaneFrameValidationError(
            "Unexpected encoded block IDs: "
            f"{sorted(extra_ids)[:10]}"
        )

    return [
        blocks_by_id[
            spec.block_id
        ]
        for spec in ordered_specs
    ]


def encode_ft_qupap_frame(
    ordered_specs: Sequence[LogicalSpec],
    rng: np.random.Generator | None = None,
    use_css: bool = True,
    require_standard_counts: bool = True,
) -> SteaneEncodedFrame:
    """
    Encode the complete interleaved FT-QuPAP frame.

    This is the primary function imported by mobile_station.py.
    """

    validate_ordered_specs(
        ordered_specs,
        require_standard_counts=
            require_standard_counts,
    )

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    payload_blocks = (
        encode_payload_blocks(
            ordered_specs=ordered_specs,
            use_css=use_css,
            rng=rng,
        )
    )

    check_blocks = (
        encode_check_blocks(
            ordered_specs=ordered_specs,
            use_css=use_css,
            rng=rng,
        )
    )

    frame = frame_and_interleave_blocks(
        ordered_specs=ordered_specs,
        payload_blocks=
            payload_blocks,
        check_blocks=
            check_blocks,
    )

    encoded_frame = SteaneEncodedFrame(
        ordered_specs=[
            copy.deepcopy(spec)
            for spec in ordered_specs
        ],
        payload_blocks=
            payload_blocks,
        check_blocks=
            check_blocks,
        frame=frame,
        use_css=use_css,
    )

    return encoded_frame


def validate_encoded_frame(
    encoded_frame: SteaneEncodedFrame,
    require_standard_counts: bool = True,
) -> None:
    """
    Validate a complete FT-QuPAP encoded frame.
    """

    if not isinstance(
        encoded_frame,
        SteaneEncodedFrame,
    ):
        raise TypeError(
            "encoded_frame must be a SteaneEncodedFrame."
        )

    validate_ordered_specs(
        encoded_frame.ordered_specs,
        require_standard_counts=
            require_standard_counts,
    )

    if len(encoded_frame.frame) != len(
        encoded_frame.ordered_specs
    ):
        raise SteaneFrameValidationError(
            "Physical frame length does not match "
            "logical frame length."
        )

    payload_count = len(
        encoded_frame.payload_blocks
    )

    check_count = len(
        encoded_frame.check_blocks
    )

    if require_standard_counts:
        if payload_count != (
            PAYLOAD_LOGICAL_BLOCK_COUNT
        ):
            raise SteaneFrameValidationError(
                "Encoded frame must contain exactly "
                f"{PAYLOAD_LOGICAL_BLOCK_COUNT} "
                "payload blocks."
            )

        if check_count != (
            CHECK_LOGICAL_BLOCK_COUNT
        ):
            raise SteaneFrameValidationError(
                "Encoded frame must contain exactly "
                f"{CHECK_LOGICAL_BLOCK_COUNT} "
                "check blocks."
            )

    expected_ids = [
        spec.block_id
        for spec
        in encoded_frame.ordered_specs
    ]

    frame_ids = [
        block.spec.block_id
        for block in encoded_frame.frame
    ]

    if frame_ids != expected_ids:
        raise SteaneFrameValidationError(
            "Physical frame ordering does not match "
            "the interleaved logical schedule."
        )

    for block in encoded_frame.frame:
        validate_physical_block(block)

        if block.use_css != (
            encoded_frame.use_css
        ):
            raise SteaneFrameValidationError(
                "Frame contains inconsistent CSS modes."
            )

    expected_physical_count = (
        len(encoded_frame.frame)
        * (
            STEANE_BLOCK_SIZE
            if encoded_frame.use_css
            else 1
        )
    )

    if (
        encoded_frame.total_physical_qubits
        != expected_physical_count
    ):
        raise SteaneFrameValidationError(
            "Encoded frame contains an unexpected "
            "number of physical positions."
        )


def decode_steane_logical_bit(
    codeword: np.ndarray | None,
) -> int | None:
    """
    Decode an exact corrected Steane codeword.

    This function expects errors to have already been corrected.
    """

    if codeword is None:
        return None

    if not isinstance(
        codeword,
        np.ndarray,
    ):
        try:
            codeword = np.asarray(
                codeword,
                dtype=np.int8,
            )
        except Exception:
            return None

    if (
        codeword.ndim != 1
        or len(codeword)
        != STEANE_BLOCK_SIZE
    ):
        return None

    if not np.all(
        np.isin(
            codeword,
            [0, 1],
        )
    ):
        return None

    normalized = codeword.astype(
        np.int8,
        copy=False,
    )

    for logical_bit in (
        0,
        1,
    ):
        for base_word in (
            STEANE_C2_CODEWORDS
        ):
            candidate = (
                base_word
                ^ (
                    STEANE_LOGICAL_X
                    * logical_bit
                )
            )

            if np.array_equal(
                normalized,
                candidate,
            ):
                return logical_bit

    return None


def calculate_syndrome(
    error_vector: np.ndarray,
) -> tuple[int, int, int]:
    """
    Calculate the three-bit Steane/Hamming syndrome.
    """

    if not isinstance(
        error_vector,
        np.ndarray,
    ):
        error_vector = np.asarray(
            error_vector,
            dtype=np.int8,
        )

    if error_vector.shape != (
        STEANE_BLOCK_SIZE,
    ):
        raise ValueError(
            "error_vector must contain seven bits."
        )

    if not np.all(
        np.isin(
            error_vector,
            [0, 1],
        )
    ):
        raise ValueError(
            "error_vector must be binary."
        )

    syndrome = (
        STEANE_H
        @ error_vector.astype(
            np.int8
        )
    ) % 2

    return syndrome_tuple(
        syndrome.astype(
            np.int8
        )
    )


def correction_position_from_syndrome(
    syndrome: tuple[int, int, int],
) -> int | None:
    """
    Map a syndrome to its single-error correction position.

    Returns:
        -1:
            Zero syndrome; no correction required.

        0-6:
            Physical position indicated by the syndrome.

        None:
            Invalid syndrome.
    """

    if not isinstance(
        syndrome,
        tuple,
    ):
        raise TypeError(
            "syndrome must be a tuple."
        )

    if len(syndrome) != 3:
        raise ValueError(
            "syndrome must contain three bits."
        )

    normalized = tuple(
        int(value)
        for value in syndrome
    )

    if any(
        value not in (0, 1)
        for value in normalized
    ):
        raise ValueError(
            "syndrome values must be binary."
        )

    return STEANE_SYNDROME_TO_POSITION.get(
        normalized
    )


def flatten_reference_bits(
    frame: SteaneEncodedFrame
    | Sequence[PhysicalBlock],
) -> np.ndarray:
    """
    Flatten all physical reference patterns into one array.
    """

    if isinstance(
        frame,
        SteaneEncodedFrame,
    ):
        physical_blocks = frame.frame
    else:
        physical_blocks = list(frame)

    if len(physical_blocks) == 0:
        return np.array(
            [],
            dtype=np.int8,
        )

    for block in physical_blocks:
        validate_physical_block(block)

    return np.concatenate(
        [
            block.reference_bits
            for block in physical_blocks
        ]
    ).astype(
        np.int8
    )


def create_self_test_specs(
    seed: int,
) -> list[LogicalSpec]:
    """Create one valid standard interleaved logical frame."""

    rng = np.random.default_rng(
        seed
    )

    payload_specs = [
        LogicalSpec(
            block_id=f"P{index:04d}",
            role="payload",
            logical_index=index,
            logical_bit=int(
                rng.integers(0, 2)
            ),
            basis="Z",
        )
        for index in range(
            PAYLOAD_LOGICAL_BLOCK_COUNT
        )
    ]

    check_specs = [
        LogicalSpec(
            block_id=f"C{index:04d}",
            role="check",
            logical_index=index,
            logical_bit=int(
                rng.integers(0, 2)
            ),
            basis=str(
                rng.choice(
                    SUPPORTED_BASES
                )
            ),
        )
        for index in range(
            CHECK_LOGICAL_BLOCK_COUNT
        )
    ]

    all_specs = (
        payload_specs
        + check_specs
    )

    permutation = rng.permutation(
        len(all_specs)
    )

    ordered_specs = [
        copy.deepcopy(
            all_specs[
                int(index)
            ]
        )
        for index in permutation
    ]

    for position, spec in enumerate(
        ordered_specs
    ):
        spec.position = position

    return ordered_specs


def run_self_test() -> None:
    """
    Test standard Steane frame encoding and reproducibility.
    """

    print("=" * 70)
    print("FT-QuPAP Steane Encoder Self-Test")
    print("=" * 70)

    ordered_specs = (
        create_self_test_specs(
            seed=20260701
        )
    )

    first_frame = (
        encode_ft_qupap_frame(
            ordered_specs=
                ordered_specs,
            rng=np.random.default_rng(
                9102
            ),
            use_css=True,
        )
    )

    second_frame = (
        encode_ft_qupap_frame(
            ordered_specs=
                ordered_specs,
            rng=np.random.default_rng(
                9102
            ),
            use_css=True,
        )
    )

    reproducible = all(
        np.array_equal(
            first.reference_bits,
            second.reference_bits,
        )
        for first, second in zip(
            first_frame.frame,
            second_frame.frame,
            strict=True,
        )
    )

    all_blocks_decode = all(
        decode_steane_logical_bit(
            block.reference_bits
        )
        == block.spec.logical_bit
        for block in first_frame.frame
    )

    error_arrays_zero = all(
        not np.any(
            block.x_errors
        )
        and not np.any(
            block.z_errors
        )
        and not np.any(
            block.erasures
        )
        and not np.any(
            block.attacked_mask
        )
        for block in first_frame.frame
    )

    flattened = flatten_reference_bits(
        first_frame
    )

    sample_error = np.zeros(
        STEANE_BLOCK_SIZE,
        dtype=np.int8,
    )

    sample_error[4] = 1

    sample_syndrome = (
        calculate_syndrome(
            sample_error
        )
    )

    located_position = (
        correction_position_from_syndrome(
            sample_syndrome
        )
    )

    print(
        f"Steane C2 codewords       : "
        f"{len(STEANE_C2_CODEWORDS)}"
    )
    print(
        f"Payload logical blocks    : "
        f"{len(first_frame.payload_blocks)}"
    )
    print(
        f"Check logical blocks      : "
        f"{len(first_frame.check_blocks)}"
    )
    print(
        f"Total logical blocks      : "
        f"{first_frame.logical_block_count}"
    )
    print(
        f"Physical qubits per block : "
        f"{first_frame.physical_qubits_per_block}"
    )
    print(
        f"Total physical qubits     : "
        f"{first_frame.total_physical_qubits}"
    )
    print(
        f"Flattened physical bits   : "
        f"{len(flattened)}"
    )
    print(
        f"All blocks decode         : "
        f"{all_blocks_decode}"
    )
    print(
        f"Initial error arrays zero : "
        f"{error_arrays_zero}"
    )
    print(
        f"Fixed-seed reproducible   : "
        f"{reproducible}"
    )
    print(
        f"Test error position       : "
        f"4"
    )
    print(
        f"Detected syndrome         : "
        f"{sample_syndrome}"
    )
    print(
        f"Located error position    : "
        f"{located_position}"
    )

    if (
        first_frame.logical_block_count
        != TOTAL_LOGICAL_BLOCK_COUNT
    ):
        raise SteaneEncodingError(
            "Incorrect logical block count."
        )

    if (
        first_frame.total_physical_qubits
        != TOTAL_PHYSICAL_QUBIT_COUNT
    ):
        raise SteaneEncodingError(
            "Incorrect physical-qubit count."
        )

    if not all_blocks_decode:
        raise SteaneEncodingError(
            "An encoded block failed logical decoding."
        )

    if not error_arrays_zero:
        raise SteaneEncodingError(
            "A newly encoded block contains an error."
        )

    if not reproducible:
        raise SteaneEncodingError(
            "Fixed-seed encoding is not reproducible."
        )

    if located_position != 4:
        raise SteaneEncodingError(
            "Syndrome lookup returned the wrong position."
        )

    print("\nSafe frame summary:")

    print(
        first_frame.safe_summary()
    )

    print(
        "\nSteane encoder self-test "
        "completed successfully."
    )


__all__ = [
    "LogicalSpec",
    "LogicalQubitSpec",
    "PhysicalBlock",
    "SteaneEncodedFrame",
    "SteaneEncodingError",
    "STEANE_H",
    "STEANE_LOGICAL_X",
    "STEANE_C2_CODEWORDS",
    "STEANE_SYNDROME_TO_POSITION",
    "PAYLOAD_LOGICAL_BLOCK_COUNT",
    "CHECK_LOGICAL_BLOCK_COUNT",
    "TOTAL_LOGICAL_BLOCK_COUNT",
    "STEANE_BLOCK_SIZE",
    "TOTAL_PHYSICAL_QUBIT_COUNT",
    "binary_span",
    "syndrome_tuple",
    "calculate_syndrome",
    "correction_position_from_syndrome",
    "encode_one_logical_qubit",
    "encode_payload_blocks",
    "encode_check_blocks",
    "frame_and_interleave_blocks",
    "encode_ft_qupap_frame",
    "decode_steane_logical_bit",
    "flatten_reference_bits",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        SteaneEncodingError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"\n[STEANE ENCODER ERROR] "
            f"{error}"
        )