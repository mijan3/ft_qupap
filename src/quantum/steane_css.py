"""
FT-QuPAP Steane [[7,1,3]] CSS Encoding

This module implements the scalable syndrome-level Steane CSS model
used by the FT-QuPAP complete-session simulator.

Standard FT-QuPAP frame:

    128 payload logical blocks
     32 check logical blocks
    -------------------------
    160 logical blocks

With Steane [[7,1,3]] encoding:

    160 logical blocks × 7 physical qubits
    = 1120 physical qubits

Research boundary:

This is a syndrome-level simulation of Steane-code behavior. It does
not claim hardware-level or threshold-theorem fault tolerance. Qiskit
circuits are used separately for representative encoded-block
validation.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .block_interleaver import (
    frame_and_interleave_blocks,
    validate_interleaved_blocks,
)
from .logical_qubit import (
    CHECK_ROLE,
    PAYLOAD_ROLE,
    LogicalQubit,
)


STEANE_BLOCK_SIZE = 7

PAYLOAD_LOGICAL_BLOCK_COUNT = 128
CHECK_LOGICAL_BLOCK_COUNT = 32
TOTAL_LOGICAL_BLOCK_COUNT = 160

TOTAL_STEANE_PHYSICAL_QUBITS = (
    TOTAL_LOGICAL_BLOCK_COUNT
    * STEANE_BLOCK_SIZE
)


# Hamming-code parity-check matrix used by the notebook.
STEANE_H = np.array(
    [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.int8,
)


# Transversal logical-X representation.
STEANE_LOGICAL_X = np.ones(
    STEANE_BLOCK_SIZE,
    dtype=np.int8,
)


class SteaneCSSError(Exception):
    """Base exception for Steane CSS processing."""


class InvalidSteaneBlockError(
    SteaneCSSError
):
    """Raised when a physical Steane block is invalid."""


class InvalidSteaneFrameError(
    SteaneCSSError
):
    """Raised when an encoded FT-QuPAP frame is invalid."""


class LogicalDecodingError(
    SteaneCSSError
):
    """Raised when a physical word cannot be logically decoded."""


def validate_binary_vector(
    vector: Any,
    field_name: str,
    expected_length: int,
) -> np.ndarray:
    """
    Validate and normalize a one-dimensional binary vector.
    """

    if isinstance(
        expected_length,
        bool,
    ) or not isinstance(
        expected_length,
        int,
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
        raise InvalidSteaneBlockError(
            f"{field_name} cannot be converted "
            "to a NumPy array."
        ) from error

    if normalized.ndim != 1:
        raise InvalidSteaneBlockError(
            f"{field_name} must be one-dimensional."
        )

    if len(normalized) != expected_length:
        raise InvalidSteaneBlockError(
            f"{field_name} must contain exactly "
            f"{expected_length} values."
        )

    if not np.all(
        np.isin(
            normalized,
            [0, 1],
        )
    ):
        raise InvalidSteaneBlockError(
            f"{field_name} must contain only 0 or 1."
        )

    return normalized.copy()


def validate_boolean_vector(
    vector: Any,
    field_name: str,
    expected_length: int,
) -> np.ndarray:
    """
    Validate and normalize a one-dimensional boolean vector.
    """

    try:
        normalized = np.asarray(
            vector,
            dtype=bool,
        )

    except Exception as error:
        raise InvalidSteaneBlockError(
            f"{field_name} cannot be converted "
            "to a boolean NumPy array."
        ) from error

    if normalized.ndim != 1:
        raise InvalidSteaneBlockError(
            f"{field_name} must be one-dimensional."
        )

    if len(normalized) != expected_length:
        raise InvalidSteaneBlockError(
            f"{field_name} must contain exactly "
            f"{expected_length} values."
        )

    return normalized.copy()


def binary_span(
    generators: np.ndarray,
) -> np.ndarray:
    """
    Generate the binary span of a generator matrix.

    For the three Steane/Hamming generators, this produces the eight
    C2 codewords used by the notebook's syndrome-level encoder.
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
            "generators must be two-dimensional."
        )

    if generators.shape[0] == 0:
        raise ValueError(
            "generators cannot have zero rows."
        )

    if generators.shape[1] == 0:
        raise ValueError(
            "generators cannot have zero columns."
        )

    normalized_generators = (
        generators.astype(
            np.int8,
            copy=True,
        )
    )

    if not np.all(
        np.isin(
            normalized_generators,
            [0, 1],
        )
    ):
        raise ValueError(
            "generators must contain only binary values."
        )

    words: list[np.ndarray] = []

    for mask in range(
        2 ** len(
            normalized_generators
        )
    ):
        word = np.zeros(
            normalized_generators.shape[1],
            dtype=np.int8,
        )

        for index in range(
            len(
                normalized_generators
            )
        ):
            if (
                mask >> index
            ) & 1:
                word ^= (
                    normalized_generators[
                        index
                    ]
                )

        words.append(word)

    return np.unique(
        np.asarray(
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
    Convert a three-bit syndrome array into a tuple.
    """

    normalized = validate_binary_vector(
        vector=vector,
        field_name="syndrome",
        expected_length=3,
    )

    return tuple(
        int(value)
        for value in normalized.tolist()
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
    One syndrome-level physical FT-QuPAP block.

    Attributes:
        spec:
            Logical payload/check specification.

        reference_bits:
            Expected seven-bit measurement pattern in the declared
            logical basis.

        use_css:
            True for Steane encoding and False for the no-CSS
            comparison baseline.

        x_errors:
            Physical X-error indicators.

        z_errors:
            Physical Z-error indicators.

        erasures:
            Physical loss/erasure indicators.

        attacked_mask:
            Hidden simulator-only record of intercepted positions.

    attacked_mask must never be supplied to the Authentication
    Server's GP feature extractor.
    """

    spec: LogicalQubit
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
        if not isinstance(
            self.spec,
            LogicalQubit,
        ):
            raise TypeError(
                "spec must be a LogicalQubit."
            )

        if not isinstance(
            self.use_css,
            bool,
        ):
            raise TypeError(
                "use_css must be boolean."
            )

        expected_length = (
            STEANE_BLOCK_SIZE
            if self.use_css
            else 1
        )

        self.reference_bits = (
            validate_binary_vector(
                vector=self.reference_bits,
                field_name="reference_bits",
                expected_length=(
                    expected_length
                ),
            )
        )

        if len(self.x_errors) == 0:
            self.x_errors = np.zeros(
                expected_length,
                dtype=np.int8,
            )

        if len(self.z_errors) == 0:
            self.z_errors = np.zeros(
                expected_length,
                dtype=np.int8,
            )

        if len(self.erasures) == 0:
            self.erasures = np.zeros(
                expected_length,
                dtype=bool,
            )

        if len(self.attacked_mask) == 0:
            self.attacked_mask = (
                np.zeros(
                    expected_length,
                    dtype=bool,
                )
            )

        self.x_errors = (
            validate_binary_vector(
                vector=self.x_errors,
                field_name="x_errors",
                expected_length=(
                    expected_length
                ),
            )
        )

        self.z_errors = (
            validate_binary_vector(
                vector=self.z_errors,
                field_name="z_errors",
                expected_length=(
                    expected_length
                ),
            )
        )

        self.erasures = (
            validate_boolean_vector(
                vector=self.erasures,
                field_name="erasures",
                expected_length=(
                    expected_length
                ),
            )
        )

        self.attacked_mask = (
            validate_boolean_vector(
                vector=self.attacked_mask,
                field_name="attacked_mask",
                expected_length=(
                    expected_length
                ),
            )
        )

        validate_physical_block(
            self
        )

    @property
    def block_id(self) -> str:
        """Return the logical block identifier."""

        return self.spec.block_id

    @property
    def role(self) -> str:
        """Return payload or check role."""

        return self.spec.role

    @property
    def basis(self) -> str:
        """Return the declared preparation basis."""

        return self.spec.basis

    @property
    def physical_qubit_count(self) -> int:
        """Return the number of physical positions."""

        return len(
            self.reference_bits
        )

    @property
    def has_erasure(self) -> bool:
        """Return whether any physical position was erased."""

        return bool(
            np.any(
                self.erasures
            )
        )

    @property
    def x_error_weight(self) -> int:
        """Return final physical X-error weight."""

        return int(
            np.sum(
                self.x_errors
            )
        )

    @property
    def z_error_weight(self) -> int:
        """Return final physical Z-error weight."""

        return int(
            np.sum(
                self.z_errors
            )
        )

    def copy(self) -> PhysicalBlock:
        """Return an independent block copy."""

        return copy.deepcopy(self)

    def reset_channel_state(self) -> None:
        """
        Reset channel errors, erasures, and hidden attack metadata.
        """

        physical_count = (
            self.physical_qubit_count
        )

        self.x_errors = np.zeros(
            physical_count,
            dtype=np.int8,
        )

        self.z_errors = np.zeros(
            physical_count,
            dtype=np.int8,
        )

        self.erasures = np.zeros(
            physical_count,
            dtype=bool,
        )

        self.attacked_mask = np.zeros(
            physical_count,
            dtype=bool,
        )

    def safe_summary(self) -> dict[str, Any]:
        """
        Return metadata without hidden attack information.
        """

        return {
            "block_id":
                self.block_id,
            "role":
                self.role,
            "logical_index":
                self.spec.logical_index,
            "basis":
                self.basis,
            "use_css":
                self.use_css,
            "physical_qubit_count":
                self.physical_qubit_count,
        }


@dataclass
class SteaneEncodedFrame:
    """
    Complete encoded FT-QuPAP transmission frame.
    """

    ordered_specs: list[LogicalQubit]
    payload_blocks: list[PhysicalBlock]
    check_blocks: list[PhysicalBlock]
    frame: list[PhysicalBlock]
    use_css: bool = True

    def __post_init__(self) -> None:
        validate_encoded_frame(
            self
        )

    @property
    def physical_blocks(
        self,
    ) -> list[PhysicalBlock]:
        """Return the frame in transmission order."""

        return self.frame

    @property
    def logical_block_count(self) -> int:
        """Return total logical-block count."""

        return len(
            self.frame
        )

    @property
    def physical_qubits_per_block(
        self,
    ) -> int:
        """Return physical expansion per logical block."""

        return (
            STEANE_BLOCK_SIZE
            if self.use_css
            else 1
        )

    @property
    def total_physical_qubits(
        self,
    ) -> int:
        """Return total transmitted physical positions."""

        return sum(
            block.physical_qubit_count
            for block in self.frame
        )

    def safe_summary(self) -> dict[str, Any]:
        """Return public resource information."""

        return {
            "encoding":
                (
                    "Steane [[7,1,3]] CSS"
                    if self.use_css
                    else "No-CSS baseline"
                ),
            "payload_logical_blocks":
                len(
                    self.payload_blocks
                ),
            "check_logical_blocks":
                len(
                    self.check_blocks
                ),
            "total_logical_blocks":
                self.logical_block_count,
            "physical_qubits_per_block":
                self.physical_qubits_per_block,
            "total_physical_qubits":
                self.total_physical_qubits,
        }


def validate_physical_block(
    block: PhysicalBlock,
) -> None:
    """
    Validate one physical block and its logical representation.
    """

    if not isinstance(
        block,
        PhysicalBlock,
    ):
        raise TypeError(
            "block must be a PhysicalBlock."
        )

    expected_length = (
        STEANE_BLOCK_SIZE
        if block.use_css
        else 1
    )

    arrays = (
        block.reference_bits,
        block.x_errors,
        block.z_errors,
        block.erasures,
        block.attacked_mask,
    )

    if any(
        len(array) != expected_length
        for array in arrays
    ):
        raise InvalidSteaneBlockError(
            f"Block {block.block_id!r} contains "
            "inconsistent physical-array lengths."
        )

    if block.use_css:
        decoded_bit = (
            decode_logical_bit(
                block.reference_bits
            )
        )

        if (
            decoded_bit
            != block.spec.logical_bit
        ):
            raise InvalidSteaneBlockError(
                f"Block {block.block_id!r} does not "
                "encode its declared logical bit."
            )

    elif (
        int(
            block.reference_bits[0]
        )
        != block.spec.logical_bit
    ):
        raise InvalidSteaneBlockError(
            f"No-CSS block {block.block_id!r} "
            "does not match its logical bit."
        )


def encode_reference_word(
    logical_bit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one seven-bit Steane reference word.

    Notebook-compatible operation:

        base_word = random C2 codeword

        reference_bits =
            base_word XOR
            (STEANE_LOGICAL_X × logical_bit)
    """

    if isinstance(
        logical_bit,
        bool,
    ) or not isinstance(
        logical_bit,
        int,
    ):
        raise TypeError(
            "logical_bit must be an integer."
        )

    if logical_bit not in (
        0,
        1,
    ):
        raise ValueError(
            "logical_bit must be 0 or 1."
        )

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    selected_index = int(
        rng.integers(
            0,
            len(
                STEANE_C2_CODEWORDS
            ),
        )
    )

    base_word = (
        STEANE_C2_CODEWORDS[
            selected_index
        ].copy()
    )

    return (
        base_word
        ^ (
            STEANE_LOGICAL_X
            * logical_bit
        )
    ).astype(
        np.int8
    )


def encode_one_logical_qubit(
    spec: LogicalQubit,
    use_css: bool = True,
    rng: np.random.Generator | None = None,
) -> PhysicalBlock:
    """
    Encode one logical payload or check block.

    The declared basis remains in spec.basis. In the scalable
    syndrome-level model, reference_bits represent the expected
    physical outcomes when measured in that declared basis.
    """

    if not isinstance(
        spec,
        LogicalQubit,
    ):
        raise TypeError(
            "spec must be a LogicalQubit."
        )

    if not isinstance(
        use_css,
        bool,
    ):
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
        reference_bits = (
            encode_reference_word(
                logical_bit=(
                    spec.logical_bit
                ),
                rng=rng,
            )
        )

    else:
        reference_bits = np.array(
            [
                spec.logical_bit
            ],
            dtype=np.int8,
        )

    physical_count = len(
        reference_bits
    )

    return PhysicalBlock(
        spec=copy.deepcopy(
            spec
        ),
        reference_bits=(
            reference_bits
        ),
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
    ordered_specs: Sequence[LogicalQubit],
    use_css: bool = True,
    rng: np.random.Generator | None = None,
) -> list[PhysicalBlock]:
    """
    Encode all payload logical blocks.
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
        if spec.role == PAYLOAD_ROLE
    ]


def encode_check_blocks(
    ordered_specs: Sequence[LogicalQubit],
    use_css: bool = True,
    rng: np.random.Generator | None = None,
) -> list[PhysicalBlock]:
    """
    Encode all independent check logical blocks.
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
        if spec.role == CHECK_ROLE
    ]


def encode_steane_frame(
    ordered_specs: Sequence[LogicalQubit],
    use_css: bool = True,
    rng: np.random.Generator | None = None,
) -> SteaneEncodedFrame:
    """
    Encode the complete interleaved FT-QuPAP logical frame.
    """

    validate_interleaved_blocks(
        ordered_specs,
        require_standard_counts=True,
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
        payload_blocks=payload_blocks,
        check_blocks=check_blocks,
    )

    return SteaneEncodedFrame(
        ordered_specs=[
            copy.deepcopy(spec)
            for spec in ordered_specs
        ],
        payload_blocks=(
            payload_blocks
        ),
        check_blocks=(
            check_blocks
        ),
        frame=frame,
        use_css=use_css,
    )


# Compatibility name used by the Mobile Station package.
encode_ft_qupap_frame = encode_steane_frame


def validate_encoded_frame(
    encoded_frame: SteaneEncodedFrame,
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

    validate_interleaved_blocks(
        encoded_frame.ordered_specs,
        require_standard_counts=True,
    )

    if len(
        encoded_frame.payload_blocks
    ) != PAYLOAD_LOGICAL_BLOCK_COUNT:
        raise InvalidSteaneFrameError(
            "Encoded frame must contain exactly "
            f"{PAYLOAD_LOGICAL_BLOCK_COUNT} "
            "payload blocks."
        )

    if len(
        encoded_frame.check_blocks
    ) != CHECK_LOGICAL_BLOCK_COUNT:
        raise InvalidSteaneFrameError(
            "Encoded frame must contain exactly "
            f"{CHECK_LOGICAL_BLOCK_COUNT} "
            "check blocks."
        )

    if len(
        encoded_frame.frame
    ) != TOTAL_LOGICAL_BLOCK_COUNT:
        raise InvalidSteaneFrameError(
            "Encoded frame must contain exactly "
            f"{TOTAL_LOGICAL_BLOCK_COUNT} "
            "logical blocks."
        )

    expected_ids = [
        spec.block_id
        for spec in (
            encoded_frame
            .ordered_specs
        )
    ]

    frame_ids = [
        block.block_id
        for block in (
            encoded_frame.frame
        )
    ]

    if frame_ids != expected_ids:
        raise InvalidSteaneFrameError(
            "Physical frame order does not match "
            "the interleaved logical schedule."
        )

    for block in encoded_frame.frame:
        validate_physical_block(
            block
        )

        if block.use_css != (
            encoded_frame.use_css
        ):
            raise InvalidSteaneFrameError(
                "Encoded frame contains mixed CSS modes."
            )

    expected_physical_count = (
        TOTAL_STEANE_PHYSICAL_QUBITS
        if encoded_frame.use_css
        else TOTAL_LOGICAL_BLOCK_COUNT
    )

    if (
        encoded_frame.total_physical_qubits
        != expected_physical_count
    ):
        raise InvalidSteaneFrameError(
            "Encoded frame contains an unexpected "
            "number of physical qubits."
        )


def is_c2_codeword(
    word: np.ndarray,
) -> bool:
    """
    Return whether a word belongs to the Steane C2 code.
    """

    normalized = validate_binary_vector(
        vector=word,
        field_name="word",
        expected_length=(
            STEANE_BLOCK_SIZE
        ),
    )

    return any(
        np.array_equal(
            normalized,
            codeword,
        )
        for codeword in (
            STEANE_C2_CODEWORDS
        )
    )


def decode_logical_bit(
    reference_word: np.ndarray,
) -> int | None:
    """
    Decode an exact corrected Steane reference word.

    Returns:
        0:
            Word belongs to C2.

        1:
            Word belongs to the logical-X coset.

        None:
            Word is not an exact valid logical codeword.
    """

    try:
        normalized = (
            validate_binary_vector(
                vector=reference_word,
                field_name=(
                    "reference_word"
                ),
                expected_length=(
                    STEANE_BLOCK_SIZE
                ),
            )
        )

    except (
        InvalidSteaneBlockError,
        TypeError,
        ValueError,
    ):
        return None

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


def calculate_steane_syndrome(
    error_vector: np.ndarray,
    use_css: bool = True,
) -> np.ndarray:
    """
    Calculate a three-bit Steane/Hamming syndrome.

    This helper is retained for compatibility. Detailed syndrome
    extraction is implemented in syndrome_extraction.py.
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
            3,
            dtype=np.int8,
        )

    normalized_error = (
        validate_binary_vector(
            vector=error_vector,
            field_name="error_vector",
            expected_length=(
                STEANE_BLOCK_SIZE
            ),
        )
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


def correction_position(
    syndrome: np.ndarray
    | tuple[int, int, int],
) -> int | None:
    """
    Map a syndrome to a single-error correction position.

    Returns:
        -1:
            Zero syndrome; no correction required.

        0 through 6:
            Correctable physical position.

        None:
            Invalid or unknown syndrome.
    """

    if isinstance(
        syndrome,
        tuple,
    ):
        if len(syndrome) != 3:
            return None

        try:
            syndrome_key = tuple(
                int(value)
                for value in syndrome
            )

        except Exception:
            return None

    else:
        try:
            syndrome_key = syndrome_tuple(
                np.asarray(
                    syndrome,
                    dtype=np.int8,
                )
            )

        except Exception:
            return None

    if any(
        value not in (
            0,
            1,
        )
        for value in syndrome_key
    ):
        return None

    return (
        STEANE_SYNDROME_TO_POSITION
        .get(
            syndrome_key
        )
    )


def flatten_reference_bits(
    encoded_frame: SteaneEncodedFrame
    | Sequence[PhysicalBlock],
) -> np.ndarray:
    """
    Flatten a physical frame into one binary vector.
    """

    if isinstance(
        encoded_frame,
        SteaneEncodedFrame,
    ):
        blocks = encoded_frame.frame
    else:
        blocks = list(
            encoded_frame
        )

    if not blocks:
        return np.array(
            [],
            dtype=np.int8,
        )

    for block in blocks:
        validate_physical_block(
            block
        )

    return np.concatenate(
        [
            block.reference_bits
            for block in blocks
        ]
    ).astype(
        np.int8
    )


def run_self_test() -> None:
    """
    Verify Steane codewords, syndrome lookup, encoding, and framing.
    """

    from .block_interleaver import (
        create_interleaving_result,
    )
    from .check_qubit_generator import (
        generate_check_blocks,
    )
    from .payload_generator import (
        generate_payload_blocks,
    )

    print("=" * 72)
    print("FT-QuPAP Steane CSS Self-Test")
    print("=" * 72)

    sample_tag = bytes.fromhex(
        "00112233445566778899aabbccddeeff"
    )

    payload_specs = (
        generate_payload_blocks(
            sample_tag
        )
    )

    check_specs = (
        generate_check_blocks(
            check_count=(
                CHECK_LOGICAL_BLOCK_COUNT
            ),
            rng=np.random.default_rng(
                20260701
            ),
            require_standard_count=True,
        )
    )

    interleaving_result = (
        create_interleaving_result(
            payload_blocks=(
                payload_specs
            ),
            check_blocks=(
                check_specs
            ),
            rng=np.random.default_rng(
                9102
            ),
            require_standard_counts=True,
        )
    )

    ordered_specs = list(
        interleaving_result
        .ordered_blocks
    )

    first_frame = (
        encode_steane_frame(
            ordered_specs=ordered_specs,
            use_css=True,
            rng=np.random.default_rng(
                7001
            ),
        )
    )

    second_frame = (
        encode_steane_frame(
            ordered_specs=ordered_specs,
            use_css=True,
            rng=np.random.default_rng(
                7001
            ),
        )
    )

    fixed_seed_reproducible = all(
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
        decode_logical_bit(
            block.reference_bits
        )
        == block.spec.logical_bit
        for block in (
            first_frame.frame
        )
    )

    all_error_arrays_zero = all(
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
        for block in (
            first_frame.frame
        )
    )

    test_error = np.zeros(
        STEANE_BLOCK_SIZE,
        dtype=np.int8,
    )

    test_error[4] = 1

    test_syndrome = (
        calculate_steane_syndrome(
            test_error,
            use_css=True,
        )
    )

    located_position = (
        correction_position(
            test_syndrome
        )
    )

    flattened_bits = (
        flatten_reference_bits(
            first_frame
        )
    )

    print(
        "Steane C2 codewords       : "
        f"{len(STEANE_C2_CODEWORDS)}"
    )

    print(
        "Syndrome lookup entries   : "
        f"{len(STEANE_SYNDROME_TO_POSITION)}"
    )

    print(
        "Payload logical blocks    : "
        f"{len(first_frame.payload_blocks)}"
    )

    print(
        "Check logical blocks      : "
        f"{len(first_frame.check_blocks)}"
    )

    print(
        "Total logical blocks      : "
        f"{first_frame.logical_block_count}"
    )

    print(
        "Physical qubits per block : "
        f"{first_frame.physical_qubits_per_block}"
    )

    print(
        "Total physical qubits     : "
        f"{first_frame.total_physical_qubits}"
    )

    print(
        "Flattened reference bits  : "
        f"{len(flattened_bits)}"
    )

    print(
        "All logical blocks decode : "
        f"{all_blocks_decode}"
    )

    print(
        "Initial channel state zero: "
        f"{all_error_arrays_zero}"
    )

    print(
        "Fixed-seed reproducible   : "
        f"{fixed_seed_reproducible}"
    )

    print(
        "Test error position       : 4"
    )

    print(
        "Calculated syndrome       : "
        f"{test_syndrome.tolist()}"
    )

    print(
        "Located correction position: "
        f"{located_position}"
    )

    if len(
        STEANE_C2_CODEWORDS
    ) != 8:
        raise SteaneCSSError(
            "Steane C2 codeword count must be 8."
        )

    if (
        first_frame.logical_block_count
        != TOTAL_LOGICAL_BLOCK_COUNT
    ):
        raise SteaneCSSError(
            "Incorrect logical-block count."
        )

    if (
        first_frame.total_physical_qubits
        != TOTAL_STEANE_PHYSICAL_QUBITS
    ):
        raise SteaneCSSError(
            "Incorrect physical-qubit count."
        )

    if not all_blocks_decode:
        raise SteaneCSSError(
            "A Steane block failed logical decoding."
        )

    if not all_error_arrays_zero:
        raise SteaneCSSError(
            "A new encoded block contains channel errors."
        )

    if not fixed_seed_reproducible:
        raise SteaneCSSError(
            "Fixed-seed encoding is not reproducible."
        )

    if located_position != 4:
        raise SteaneCSSError(
            "Syndrome lookup returned the wrong position."
        )

    print(
        "\nSteane CSS self-test "
        "completed successfully."
    )


__all__ = [
    "STEANE_BLOCK_SIZE",
    "STEANE_H",
    "STEANE_LOGICAL_X",
    "STEANE_C2_CODEWORDS",
    "STEANE_SYNDROME_TO_POSITION",
    "TOTAL_STEANE_PHYSICAL_QUBITS",
    "PhysicalBlock",
    "SteaneEncodedFrame",
    "SteaneCSSError",
    "InvalidSteaneBlockError",
    "InvalidSteaneFrameError",
    "LogicalDecodingError",
    "binary_span",
    "syndrome_tuple",
    "encode_reference_word",
    "encode_one_logical_qubit",
    "encode_payload_blocks",
    "encode_check_blocks",
    "encode_steane_frame",
    "encode_ft_qupap_frame",
    "validate_physical_block",
    "validate_encoded_frame",
    "is_c2_codeword",
    "decode_logical_bit",
    "calculate_steane_syndrome",
    "correction_position",
    "flatten_reference_bits",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        SteaneCSSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[STEANE CSS ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error