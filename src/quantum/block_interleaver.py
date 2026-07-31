"""
FT-QuPAP Logical and Physical Block Interleaver

This module randomly interleaves FT-QuPAP payload and check blocks.

Standard protocol configuration:

    128 payload logical blocks
     32 check logical blocks
    -------------------------
    160 total logical blocks

The logical interleaver performs these operations:

1. Deep-copy payload and check logical blocks.
2. Combine them into one collection.
3. Generate a random permutation.
4. Arrange the blocks according to that permutation.
5. Assign frame positions from 0 through 159.
6. Create the control-schedule structure.

The control schedule records:

    ordered_block_ids
    check block positions
    check preparation bases
    expected check logical bits
    payload positions
    payload logical indices

The schedule is later completed with expected physical check patterns
and encrypted using K_ctrl.

The physical-frame interleaver reconstructs the final transmission
frame after payload and check blocks have been Steane encoded.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from .check_qubit_generator import (
    CHECK_LOGICAL_BLOCK_COUNT,
    validate_check_blocks,
)
from .logical_qubit import (
    CHECK_ROLE,
    PAYLOAD_ROLE,
    LogicalQubit,
    validate_logical_qubit_collection,
)
from .payload_generator import (
    PAYLOAD_LOGICAL_BLOCK_COUNT,
    validate_payload_blocks,
)


TOTAL_LOGICAL_BLOCK_COUNT = (
    PAYLOAD_LOGICAL_BLOCK_COUNT
    + CHECK_LOGICAL_BLOCK_COUNT
)


class BlockInterleavingError(Exception):
    """Base exception for FT-QuPAP block interleaving."""


class InvalidInterleavingInputError(
    BlockInterleavingError
):
    """Raised when payload or check inputs are invalid."""


class InvalidInterleavingScheduleError(
    BlockInterleavingError
):
    """Raised when an interleaving schedule is inconsistent."""


class PhysicalFrameInterleavingError(
    BlockInterleavingError
):
    """Raised when encoded physical blocks cannot be interleaved."""


PhysicalBlockType = TypeVar(
    "PhysicalBlockType"
)


@dataclass(frozen=True)
class InterleavingResult:
    """
    Result of FT-QuPAP logical-block interleaving.

    Attributes:
        ordered_blocks:
            Payload and check logical blocks in transmission order.

        schedule:
            Plaintext control schedule. It must later be completed
            with expected physical check patterns and encrypted.

        permutation:
            Source collection indices in randomized order.
    """

    ordered_blocks: tuple[LogicalQubit, ...]
    schedule: dict[str, Any]
    permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_interleaved_blocks(
            self.ordered_blocks,
            require_standard_counts=True,
        )

        validate_interleaving_schedule(
            self.schedule,
            require_standard_counts=True,
        )

        if len(self.permutation) != len(
            self.ordered_blocks
        ):
            raise InvalidInterleavingScheduleError(
                "Permutation length does not match "
                "the interleaved block count."
            )

        expected_permutation_values = set(
            range(len(self.ordered_blocks))
        )

        if set(self.permutation) != (
            expected_permutation_values
        ):
            raise InvalidInterleavingScheduleError(
                "Permutation does not contain every "
                "source block index exactly once."
            )

        scheduled_ids = self.schedule[
            "ordered_block_ids"
        ]

        actual_ids = [
            block.block_id
            for block in self.ordered_blocks
        ]

        if scheduled_ids != actual_ids:
            raise InvalidInterleavingScheduleError(
                "Schedule order does not match "
                "the ordered logical blocks."
            )

    @property
    def payload_count(self) -> int:
        """Return the number of payload blocks."""

        return sum(
            block.role == PAYLOAD_ROLE
            for block in self.ordered_blocks
        )

    @property
    def check_count(self) -> int:
        """Return the number of check blocks."""

        return sum(
            block.role == CHECK_ROLE
            for block in self.ordered_blocks
        )

    @property
    def total_count(self) -> int:
        """Return total logical-block count."""

        return len(
            self.ordered_blocks
        )

    def safe_summary(self) -> dict[str, Any]:
        """
        Return metadata without revealing randomized check positions.

        Check locations and expected check values are sensitive
        control information and must remain inside the encrypted
        schedule.
        """

        return {
            "payload_logical_blocks":
                self.payload_count,
            "check_logical_blocks":
                self.check_count,
            "total_logical_blocks":
                self.total_count,
            "positions_assigned":
                True,
            "check_schedule_protected":
                False,
        }


@dataclass(frozen=True)
class PhysicalFrameResult(
    Generic[PhysicalBlockType]
):
    """
    Final physical transmission-frame result.

    Attributes:
        ordered_logical_blocks:
            Logical schedule used to determine the physical order.

        physical_frame:
            Encoded blocks arranged in transmission order.
    """

    ordered_logical_blocks: tuple[LogicalQubit, ...]
    physical_frame: tuple[PhysicalBlockType, ...]

    def __post_init__(self) -> None:
        validate_interleaved_blocks(
            self.ordered_logical_blocks,
            require_standard_counts=True,
        )

        if len(self.physical_frame) != len(
            self.ordered_logical_blocks
        ):
            raise PhysicalFrameInterleavingError(
                "Physical-frame length does not match "
                "the logical-block order."
            )

        logical_ids = [
            block.block_id
            for block in self.ordered_logical_blocks
        ]

        physical_ids = [
            physical_block_id(block)
            for block in self.physical_frame
        ]

        if physical_ids != logical_ids:
            raise PhysicalFrameInterleavingError(
                "Physical-frame order does not match "
                "the logical schedule."
            )

    @property
    def logical_block_count(self) -> int:
        """Return the logical-frame size."""

        return len(
            self.ordered_logical_blocks
        )

    @property
    def physical_block_count(self) -> int:
        """Return the number of encoded blocks."""

        return len(
            self.physical_frame
        )

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


def validate_source_collections(
    payload_blocks: Sequence[LogicalQubit],
    check_blocks: Sequence[LogicalQubit],
    require_standard_counts: bool = True,
) -> None:
    """
    Validate payload and check collections before interleaving.
    """

    payload_expected_count = (
        PAYLOAD_LOGICAL_BLOCK_COUNT
        if require_standard_counts
        else None
    )

    check_expected_count = (
        CHECK_LOGICAL_BLOCK_COUNT
        if require_standard_counts
        else None
    )

    validate_logical_qubit_collection(
        logical_qubits=payload_blocks,
        expected_role=PAYLOAD_ROLE,
        expected_count=payload_expected_count,
        require_positions=False,
        require_unique_ids=True,
        require_unique_positions=False,
    )

    validate_logical_qubit_collection(
        logical_qubits=check_blocks,
        expected_role=CHECK_ROLE,
        expected_count=check_expected_count,
        require_positions=False,
        require_unique_ids=True,
        require_unique_positions=False,
    )

    if require_standard_counts:
        validate_payload_blocks(
            payload_blocks=payload_blocks,
            require_standard_count=True,
            require_unpositioned=True,
        )

        validate_check_blocks(
            check_blocks=check_blocks,
            expected_count=(
                CHECK_LOGICAL_BLOCK_COUNT
            ),
            require_unpositioned=True,
        )

    combined_ids = [
        block.block_id
        for block in (
            list(payload_blocks)
            + list(check_blocks)
        )
    ]

    if len(combined_ids) != len(
        set(combined_ids)
    ):
        raise InvalidInterleavingInputError(
            "Payload and check block IDs must be "
            "globally unique."
        )

    for block in (
        list(payload_blocks)
        + list(check_blocks)
    ):
        if block.position is not None:
            raise InvalidInterleavingInputError(
                f"Source block {block.block_id!r} "
                "already has an interleaved position."
            )


def create_random_permutation(
    item_count: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    Generate a random permutation of source indices.
    """

    if isinstance(
        item_count,
        bool,
    ) or not isinstance(
        item_count,
        int,
    ):
        raise TypeError(
            "item_count must be an integer."
        )

    if item_count <= 0:
        raise ValueError(
            "item_count must be greater than zero."
        )

    validate_rng(rng)

    permutation = [
        int(index)
        for index in rng.permutation(
            item_count
        )
    ]

    if len(permutation) != item_count:
        raise BlockInterleavingError(
            "Random permutation has an "
            "unexpected length."
        )

    if set(permutation) != set(
        range(item_count)
    ):
        raise BlockInterleavingError(
            "Random permutation is invalid."
        )

    return permutation


def assign_interleaved_positions(
    ordered_blocks: Sequence[LogicalQubit],
    in_place: bool = False,
) -> list[LogicalQubit]:
    """
    Assign positions 0 through n-1 to ordered logical blocks.

    By default, the supplied collection is not modified.
    """

    validate_logical_qubit_collection(
        logical_qubits=ordered_blocks,
        require_positions=False,
        require_unique_ids=True,
        require_unique_positions=False,
    )

    positioned_blocks = (
        list(ordered_blocks)
        if in_place
        else [
            copy.deepcopy(block)
            for block in ordered_blocks
        ]
    )

    for position, block in enumerate(
        positioned_blocks
    ):
        block.position = position

    return positioned_blocks


def build_control_schedule(
    ordered_blocks: Sequence[LogicalQubit],
) -> dict[str, Any]:
    """
    Build the notebook-compatible plaintext control schedule.

    Check entries contain:

        block_id
        position
        basis
        expected_logical_bit

    Payload entries contain:

        block_id
        position
        logical_index

    Expected physical check patterns are inserted after Steane
    encoding by the appropriate control-schedule module.
    """

    validate_interleaved_blocks(
        ordered_blocks,
        require_standard_counts=False,
    )

    schedule = {
        "ordered_block_ids": [
            block.block_id
            for block in ordered_blocks
        ],
        "check_blocks": [
            {
                "block_id":
                    block.block_id,
                "position":
                    block.position,
                "basis":
                    block.basis,
                "expected_logical_bit":
                    block.logical_bit,
            }
            for block in ordered_blocks
            if block.role == CHECK_ROLE
        ],
        "payload_blocks": [
            {
                "block_id":
                    block.block_id,
                "position":
                    block.position,
                "logical_index":
                    block.logical_index,
            }
            for block in ordered_blocks
            if block.role == PAYLOAD_ROLE
        ],
    }

    validate_interleaving_schedule(
        schedule,
        require_standard_counts=False,
    )

    return schedule


def create_interleaving_result(
    payload_blocks: Sequence[LogicalQubit],
    check_blocks: Sequence[LogicalQubit],
    rng: np.random.Generator | None = None,
    require_standard_counts: bool = True,
) -> InterleavingResult:
    """
    Randomly interleave payload and check logical blocks.

    Source collections are deep-copied and remain unchanged.
    """

    validate_source_collections(
        payload_blocks=payload_blocks,
        check_blocks=check_blocks,
        require_standard_counts=(
            require_standard_counts
        ),
    )

    if rng is None:
        rng = np.random.default_rng()

    validate_rng(rng)

    combined_blocks = [
        copy.deepcopy(block)
        for block in (
            list(payload_blocks)
            + list(check_blocks)
        )
    ]

    permutation = create_random_permutation(
        item_count=len(
            combined_blocks
        ),
        rng=rng,
    )

    randomized_blocks = [
        combined_blocks[index]
        for index in permutation
    ]

    ordered_blocks = (
        assign_interleaved_positions(
            randomized_blocks,
            in_place=True,
        )
    )

    validate_interleaved_blocks(
        ordered_blocks,
        require_standard_counts=(
            require_standard_counts
        ),
    )

    schedule = build_control_schedule(
        ordered_blocks
    )

    validate_interleaving_schedule(
        schedule,
        require_standard_counts=(
            require_standard_counts
        ),
    )

    return InterleavingResult(
        ordered_blocks=tuple(
            ordered_blocks
        ),
        schedule=schedule,
        permutation=tuple(
            permutation
        ),
    )


def create_interleaved_schedule(
    payload_specs: Sequence[LogicalQubit],
    check_specs: Sequence[LogicalQubit],
    rng: np.random.Generator | None = None,
    require_standard_counts: bool = True,
) -> tuple[
    list[LogicalQubit],
    dict[str, Any],
]:
    """
    Notebook-compatible interleaving function.

    Returns:

        ordered_specs, schedule
    """

    result = create_interleaving_result(
        payload_blocks=payload_specs,
        check_blocks=check_specs,
        rng=rng,
        require_standard_counts=(
            require_standard_counts
        ),
    )

    return (
        [
            copy.deepcopy(block)
            for block in result.ordered_blocks
        ],
        copy.deepcopy(
            result.schedule
        ),
    )


def validate_interleaved_blocks(
    ordered_blocks: Sequence[LogicalQubit],
    require_standard_counts: bool = True,
) -> None:
    """
    Validate a positioned interleaved logical frame.
    """

    expected_count = (
        TOTAL_LOGICAL_BLOCK_COUNT
        if require_standard_counts
        else None
    )

    validate_logical_qubit_collection(
        logical_qubits=ordered_blocks,
        expected_count=expected_count,
        require_positions=True,
        require_unique_ids=True,
        require_unique_positions=True,
    )

    actual_positions = [
        block.position
        for block in ordered_blocks
    ]

    expected_positions = list(
        range(len(ordered_blocks))
    )

    if actual_positions != (
        expected_positions
    ):
        raise InvalidInterleavingScheduleError(
            "Logical blocks are not arranged according "
            "to their declared positions."
        )

    payload_count = sum(
        block.role == PAYLOAD_ROLE
        for block in ordered_blocks
    )

    check_count = sum(
        block.role == CHECK_ROLE
        for block in ordered_blocks
    )

    if (
        payload_count
        + check_count
        != len(ordered_blocks)
    ):
        raise InvalidInterleavingScheduleError(
            "Interleaved frame contains an "
            "unsupported logical role."
        )

    if require_standard_counts:
        if payload_count != (
            PAYLOAD_LOGICAL_BLOCK_COUNT
        ):
            raise InvalidInterleavingScheduleError(
                "Interleaved frame must contain "
                f"{PAYLOAD_LOGICAL_BLOCK_COUNT} "
                "payload blocks."
            )

        if check_count != (
            CHECK_LOGICAL_BLOCK_COUNT
        ):
            raise InvalidInterleavingScheduleError(
                "Interleaved frame must contain "
                f"{CHECK_LOGICAL_BLOCK_COUNT} "
                "check blocks."
            )


def validate_interleaving_schedule(
    schedule: Mapping[str, Any],
    require_standard_counts: bool = True,
) -> None:
    """
    Validate the logical control schedule before encryption.
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
        "check_blocks",
        "payload_blocks",
    }

    missing_fields = (
        required_fields.difference(
            schedule.keys()
        )
    )

    if missing_fields:
        raise InvalidInterleavingScheduleError(
            "Schedule is missing fields: "
            f"{sorted(missing_fields)}"
        )

    ordered_ids = schedule[
        "ordered_block_ids"
    ]

    check_entries = schedule[
        "check_blocks"
    ]

    payload_entries = schedule[
        "payload_blocks"
    ]

    if not isinstance(
        ordered_ids,
        list,
    ):
        raise TypeError(
            "ordered_block_ids must be a list."
        )

    if not isinstance(
        check_entries,
        list,
    ):
        raise TypeError(
            "check_blocks must be a list."
        )

    if not isinstance(
        payload_entries,
        list,
    ):
        raise TypeError(
            "payload_blocks must be a list."
        )

    if len(ordered_ids) != len(
        check_entries
    ) + len(
        payload_entries
    ):
        raise InvalidInterleavingScheduleError(
            "Schedule block counts are inconsistent."
        )

    if len(ordered_ids) != len(
        set(ordered_ids)
    ):
        raise InvalidInterleavingScheduleError(
            "ordered_block_ids contains duplicates."
        )

    if require_standard_counts:
        if len(ordered_ids) != (
            TOTAL_LOGICAL_BLOCK_COUNT
        ):
            raise InvalidInterleavingScheduleError(
                "Schedule must contain exactly "
                f"{TOTAL_LOGICAL_BLOCK_COUNT} "
                "ordered block IDs."
            )

        if len(payload_entries) != (
            PAYLOAD_LOGICAL_BLOCK_COUNT
        ):
            raise InvalidInterleavingScheduleError(
                "Schedule must contain exactly "
                f"{PAYLOAD_LOGICAL_BLOCK_COUNT} "
                "payload entries."
            )

        if len(check_entries) != (
            CHECK_LOGICAL_BLOCK_COUNT
        ):
            raise InvalidInterleavingScheduleError(
                "Schedule must contain exactly "
                f"{CHECK_LOGICAL_BLOCK_COUNT} "
                "check entries."
            )

    scheduled_positions: set[int] = set()
    scheduled_ids: set[str] = set()

    for entry in check_entries:
        validate_schedule_entry(
            entry=entry,
            ordered_ids=ordered_ids,
            expected_role=CHECK_ROLE,
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
            raise InvalidInterleavingScheduleError(
                "A check schedule entry is missing "
                "required fields."
            )

        if entry["basis"] not in (
            "Z",
            "X",
        ):
            raise InvalidInterleavingScheduleError(
                "A check entry contains an invalid basis."
            )

        if entry[
            "expected_logical_bit"
        ] not in (
            0,
            1,
        ):
            raise InvalidInterleavingScheduleError(
                "A check entry contains an invalid "
                "expected logical bit."
            )

        register_schedule_entry(
            entry,
            scheduled_positions,
            scheduled_ids,
        )

    payload_indices: set[int] = set()

    for entry in payload_entries:
        validate_schedule_entry(
            entry=entry,
            ordered_ids=ordered_ids,
            expected_role=PAYLOAD_ROLE,
        )

        required_payload_fields = {
            "block_id",
            "position",
            "logical_index",
        }

        if not required_payload_fields.issubset(
            entry.keys()
        ):
            raise InvalidInterleavingScheduleError(
                "A payload schedule entry is missing "
                "required fields."
            )

        logical_index = entry[
            "logical_index"
        ]

        if isinstance(
            logical_index,
            bool,
        ) or not isinstance(
            logical_index,
            int,
        ):
            raise InvalidInterleavingScheduleError(
                "Payload logical_index must be an integer."
            )

        if logical_index < 0:
            raise InvalidInterleavingScheduleError(
                "Payload logical_index cannot be negative."
            )

        if logical_index in payload_indices:
            raise InvalidInterleavingScheduleError(
                "Payload logical indices must be unique."
            )

        payload_indices.add(
            logical_index
        )

        register_schedule_entry(
            entry,
            scheduled_positions,
            scheduled_ids,
        )

    expected_positions = set(
        range(len(ordered_ids))
    )

    if scheduled_positions != (
        expected_positions
    ):
        raise InvalidInterleavingScheduleError(
            "Schedule does not cover every "
            "interleaved position."
        )

    if scheduled_ids != set(
        ordered_ids
    ):
        raise InvalidInterleavingScheduleError(
            "Schedule entries do not cover every "
            "ordered block ID."
        )


def validate_schedule_entry(
    entry: Any,
    ordered_ids: Sequence[str],
    expected_role: str,
) -> None:
    """
    Validate common fields in one schedule entry.
    """

    if not isinstance(
        entry,
        Mapping,
    ):
        raise TypeError(
            "Every schedule entry must be a mapping."
        )

    if "block_id" not in entry:
        raise InvalidInterleavingScheduleError(
            "Schedule entry is missing block_id."
        )

    if "position" not in entry:
        raise InvalidInterleavingScheduleError(
            "Schedule entry is missing position."
        )

    block_id = entry[
        "block_id"
    ]

    position = entry[
        "position"
    ]

    if not isinstance(
        block_id,
        str,
    ):
        raise InvalidInterleavingScheduleError(
            "Schedule block_id must be a string."
        )

    if isinstance(
        position,
        bool,
    ) or not isinstance(
        position,
        int,
    ):
        raise InvalidInterleavingScheduleError(
            "Schedule position must be an integer."
        )

    if not 0 <= position < len(
        ordered_ids
    ):
        raise InvalidInterleavingScheduleError(
            "Schedule position is outside "
            "the logical frame."
        )

    if ordered_ids[position] != block_id:
        raise InvalidInterleavingScheduleError(
            "Schedule block ID does not match "
            "its declared position."
        )

    required_prefix = (
        "P"
        if expected_role == PAYLOAD_ROLE
        else "C"
    )

    if not block_id.startswith(
        required_prefix
    ):
        raise InvalidInterleavingScheduleError(
            f"{expected_role} schedule entries must "
            f"use {required_prefix}-prefixed block IDs."
        )


def register_schedule_entry(
    entry: Mapping[str, Any],
    used_positions: set[int],
    used_ids: set[str],
) -> None:
    """
    Register a validated schedule entry and detect duplicates.
    """

    position = int(
        entry["position"]
    )

    block_id = str(
        entry["block_id"]
    )

    if position in used_positions:
        raise InvalidInterleavingScheduleError(
            f"Duplicate schedule position: {position}."
        )

    if block_id in used_ids:
        raise InvalidInterleavingScheduleError(
            f"Duplicate scheduled block ID: {block_id!r}."
        )

    used_positions.add(
        position
    )

    used_ids.add(
        block_id
    )


def physical_block_id(
    physical_block: Any,
) -> str:
    """
    Extract the logical block ID from an encoded physical block.

    Supported encoded-block layouts:

        block.block_id
        block.spec.block_id
        block.logical_qubit.block_id
    """

    if hasattr(
        physical_block,
        "block_id",
    ):
        block_id = getattr(
            physical_block,
            "block_id",
        )

    elif (
        hasattr(
            physical_block,
            "spec",
        )
        and hasattr(
            physical_block.spec,
            "block_id",
        )
    ):
        block_id = (
            physical_block
            .spec
            .block_id
        )

    elif (
        hasattr(
            physical_block,
            "logical_qubit",
        )
        and hasattr(
            physical_block.logical_qubit,
            "block_id",
        )
    ):
        block_id = (
            physical_block
            .logical_qubit
            .block_id
        )

    else:
        raise PhysicalFrameInterleavingError(
            "Encoded physical block does not expose "
            "a logical block ID."
        )

    if not isinstance(
        block_id,
        str,
    ) or not block_id:
        raise PhysicalFrameInterleavingError(
            "Encoded physical block contains an "
            "invalid block ID."
        )

    return block_id


def build_physical_block_lookup(
    physical_blocks: Sequence[
        PhysicalBlockType
    ],
) -> dict[str, PhysicalBlockType]:
    """
    Build a lookup table for encoded physical blocks.
    """

    if isinstance(
        physical_blocks,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "physical_blocks must be a sequence."
        )

    if not isinstance(
        physical_blocks,
        Sequence,
    ):
        raise TypeError(
            "physical_blocks must be a sequence."
        )

    lookup: dict[
        str,
        PhysicalBlockType,
    ] = {}

    for block in physical_blocks:
        block_id = physical_block_id(
            block
        )

        if block_id in lookup:
            raise PhysicalFrameInterleavingError(
                "Duplicate encoded physical block ID: "
                f"{block_id!r}."
            )

        lookup[block_id] = block

    return lookup


def frame_and_interleave_blocks(
    ordered_specs: Sequence[LogicalQubit],
    payload_blocks: Sequence[
        PhysicalBlockType
    ],
    check_blocks: Sequence[
        PhysicalBlockType
    ],
) -> list[PhysicalBlockType]:
    """
    Reconstruct the physical transmission frame.

    This follows the logical order created before Steane encoding.

    The notebook first encodes payload and check blocks separately,
    then uses this operation to restore the randomized transmission
    order.
    """

    validate_interleaved_blocks(
        ordered_specs,
        require_standard_counts=True,
    )

    if len(payload_blocks) != (
        PAYLOAD_LOGICAL_BLOCK_COUNT
    ):
        raise PhysicalFrameInterleavingError(
            "Expected exactly "
            f"{PAYLOAD_LOGICAL_BLOCK_COUNT} "
            "encoded payload blocks."
        )

    if len(check_blocks) != (
        CHECK_LOGICAL_BLOCK_COUNT
    ):
        raise PhysicalFrameInterleavingError(
            "Expected exactly "
            f"{CHECK_LOGICAL_BLOCK_COUNT} "
            "encoded check blocks."
        )

    all_physical_blocks = (
        list(payload_blocks)
        + list(check_blocks)
    )

    blocks_by_id = (
        build_physical_block_lookup(
            all_physical_blocks
        )
    )

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
        raise PhysicalFrameInterleavingError(
            "Encoded blocks are missing for IDs: "
            f"{missing_ids[:10]}"
        )

    unexpected_ids = set(
        blocks_by_id
    ).difference(
        ordered_ids
    )

    if unexpected_ids:
        raise PhysicalFrameInterleavingError(
            "Unexpected encoded block IDs: "
            f"{sorted(unexpected_ids)[:10]}"
        )

    physical_frame = [
        blocks_by_id[
            spec.block_id
        ]
        for spec in ordered_specs
    ]

    physical_frame_ids = [
        physical_block_id(block)
        for block in physical_frame
    ]

    if physical_frame_ids != ordered_ids:
        raise PhysicalFrameInterleavingError(
            "Physical-frame reconstruction failed."
        )

    return physical_frame


def create_physical_frame_result(
    ordered_specs: Sequence[LogicalQubit],
    payload_blocks: Sequence[
        PhysicalBlockType
    ],
    check_blocks: Sequence[
        PhysicalBlockType
    ],
) -> PhysicalFrameResult[
    PhysicalBlockType
]:
    """
    Reconstruct the physical frame and return a result object.
    """

    physical_frame = (
        frame_and_interleave_blocks(
            ordered_specs=ordered_specs,
            payload_blocks=payload_blocks,
            check_blocks=check_blocks,
        )
    )

    return PhysicalFrameResult(
        ordered_logical_blocks=tuple(
            copy.deepcopy(
                list(ordered_specs)
            )
        ),
        physical_frame=tuple(
            physical_frame
        ),
    )


def payload_positions(
    schedule: Mapping[str, Any],
) -> list[int]:
    """
    Return payload positions in logical-index order.
    """

    validate_interleaving_schedule(
        schedule,
        require_standard_counts=False,
    )

    ordered_entries = sorted(
        schedule["payload_blocks"],
        key=lambda entry:
            entry["logical_index"],
    )

    return [
        int(entry["position"])
        for entry in ordered_entries
    ]


def check_positions(
    schedule: Mapping[str, Any],
) -> list[int]:
    """
    Return check positions in frame order.
    """

    validate_interleaving_schedule(
        schedule,
        require_standard_counts=False,
    )

    return sorted(
        int(entry["position"])
        for entry in schedule[
            "check_blocks"
        ]
    )


def run_self_test() -> None:
    """
    Verify randomized logical and physical block interleaving.
    """

    from dataclasses import dataclass

    from .check_qubit_generator import (
        generate_check_blocks,
    )
    from .payload_generator import (
        generate_payload_blocks,
    )

    print("=" * 72)
    print("FT-QuPAP Block Interleaver Self-Test")
    print("=" * 72)

    sample_tag = bytes.fromhex(
        "00112233445566778899aabbccddeeff"
    )

    payload_blocks = (
        generate_payload_blocks(
            sample_tag
        )
    )

    check_blocks = (
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

    first_result = (
        create_interleaving_result(
            payload_blocks=payload_blocks,
            check_blocks=check_blocks,
            rng=np.random.default_rng(
                9102
            ),
            require_standard_counts=True,
        )
    )

    second_result = (
        create_interleaving_result(
            payload_blocks=payload_blocks,
            check_blocks=check_blocks,
            rng=np.random.default_rng(
                9102
            ),
            require_standard_counts=True,
        )
    )

    source_blocks_unchanged = all(
        block.position is None
        for block in (
            payload_blocks
            + check_blocks
        )
    )

    reproducible = (
        first_result.permutation
        == second_result.permutation
    )

    ordered_ids = [
        block.block_id
        for block
        in first_result.ordered_blocks
    ]

    source_ids = [
        block.block_id
        for block in (
            payload_blocks
            + check_blocks
        )
    ]

    every_block_preserved = (
        set(ordered_ids)
        == set(source_ids)
        and len(ordered_ids)
        == len(source_ids)
    )

    positions_correct = all(
        block.position == position
        for position, block in enumerate(
            first_result.ordered_blocks
        )
    )

    check_location_count = len(
        first_result.schedule[
            "check_blocks"
        ]
    )

    payload_location_count = len(
        first_result.schedule[
            "payload_blocks"
        ]
    )

    @dataclass
    class DummyPhysicalBlock:
        spec: LogicalQubit
        reference_bits: tuple[int, ...]

        @property
        def block_id(self) -> str:
            return self.spec.block_id

    encoded_payload = [
        DummyPhysicalBlock(
            spec=copy.deepcopy(block),
            reference_bits=(
                block.logical_bit,
            ) * 7,
        )
        for block in first_result.ordered_blocks
        if block.role == PAYLOAD_ROLE
    ]

    encoded_checks = [
        DummyPhysicalBlock(
            spec=copy.deepcopy(block),
            reference_bits=(
                block.logical_bit,
            ) * 7,
        )
        for block in first_result.ordered_blocks
        if block.role == CHECK_ROLE
    ]

    physical_frame = (
        frame_and_interleave_blocks(
            ordered_specs=(
                first_result.ordered_blocks
            ),
            payload_blocks=(
                encoded_payload
            ),
            check_blocks=(
                encoded_checks
            ),
        )
    )

    physical_order_correct = [
        block.block_id
        for block in physical_frame
    ] == ordered_ids

    check_positions_hidden_from_summary = (
        "check_positions"
        not in first_result.safe_summary()
    )

    print(
        "Payload source blocks     : "
        f"{len(payload_blocks)}"
    )

    print(
        "Check source blocks       : "
        f"{len(check_blocks)}"
    )

    print(
        "Interleaved logical blocks: "
        f"{first_result.total_count}"
    )

    print(
        "Scheduled payload blocks  : "
        f"{payload_location_count}"
    )

    print(
        "Scheduled check blocks    : "
        f"{check_location_count}"
    )

    print(
        "Positions sequential      : "
        f"{positions_correct}"
    )

    print(
        "Every block preserved     : "
        f"{every_block_preserved}"
    )

    print(
        "Source blocks unchanged   : "
        f"{source_blocks_unchanged}"
    )

    print(
        "Fixed-seed reproducible   : "
        f"{reproducible}"
    )

    print(
        "Physical order restored   : "
        f"{physical_order_correct}"
    )

    print(
        "Safe summary hides checks : "
        f"{check_positions_hidden_from_summary}"
    )

    print(
        "First ten ordered IDs     : "
        f"{ordered_ids[:10]}"
    )

    if (
        first_result.total_count
        != TOTAL_LOGICAL_BLOCK_COUNT
    ):
        raise BlockInterleavingError(
            "Incorrect total logical-block count."
        )

    if payload_location_count != (
        PAYLOAD_LOGICAL_BLOCK_COUNT
    ):
        raise BlockInterleavingError(
            "Incorrect scheduled payload count."
        )

    if check_location_count != (
        CHECK_LOGICAL_BLOCK_COUNT
    ):
        raise BlockInterleavingError(
            "Incorrect scheduled check count."
        )

    if not positions_correct:
        raise BlockInterleavingError(
            "Interleaved positions are incorrect."
        )

    if not every_block_preserved:
        raise BlockInterleavingError(
            "Interleaving lost or duplicated a block."
        )

    if not source_blocks_unchanged:
        raise BlockInterleavingError(
            "Interleaving modified a source block."
        )

    if not reproducible:
        raise BlockInterleavingError(
            "Identical random seeds produced "
            "different permutations."
        )

    if not physical_order_correct:
        raise BlockInterleavingError(
            "Physical-frame order was not restored."
        )

    if not check_positions_hidden_from_summary:
        raise BlockInterleavingError(
            "Safe summary exposed check positions."
        )

    print(
        "\nBlock interleaver self-test "
        "completed successfully."
    )


__all__ = [
    "TOTAL_LOGICAL_BLOCK_COUNT",
    "BlockInterleavingError",
    "InvalidInterleavingInputError",
    "InvalidInterleavingScheduleError",
    "PhysicalFrameInterleavingError",
    "InterleavingResult",
    "PhysicalFrameResult",
    "validate_rng",
    "validate_source_collections",
    "create_random_permutation",
    "assign_interleaved_positions",
    "build_control_schedule",
    "create_interleaving_result",
    "create_interleaved_schedule",
    "validate_interleaved_blocks",
    "validate_interleaving_schedule",
    "physical_block_id",
    "build_physical_block_lookup",
    "frame_and_interleave_blocks",
    "create_physical_frame_result",
    "payload_positions",
    "check_positions",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        BlockInterleavingError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[BLOCK INTERLEAVING ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error