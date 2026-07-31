"""
Logical Qubit Models
FT-QuPAP Quantum Simulation Package

This module defines the logical-qubit representation shared by the
FT-QuPAP payload, check-block, interleaving, Steane CSS, channel,
measurement, and QBER modules.

Notebook-compatible logical specification:

    LogicalSpec(
        block_id,
        role,
        logical_index,
        logical_bit,
        basis,
        position,
    )

Protocol rules:

1. Payload logical qubits represent KMAC tag bits.
2. Payload logical qubits always use the Z basis.
3. Check logical qubits are independent random states.
4. Check logical qubits may use the Z or X basis.
5. Position remains None until random interleaving is complete.
6. One logical qubit later becomes one seven-qubit Steane block.

This is a syndrome-level logical representation. It does not create
or execute a physical quantum circuit.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .classical_to_qubit import (
    QubitStateDescription,
    state_label,
    validate_basis,
    validate_classical_bit,
)


LogicalRole = Literal[
    "payload",
    "check",
]

LogicalBasis = Literal[
    "Z",
    "X",
]


PAYLOAD_ROLE = "payload"
CHECK_ROLE = "check"

SUPPORTED_ROLES = (
    PAYLOAD_ROLE,
    CHECK_ROLE,
)

SUPPORTED_BASES = (
    "Z",
    "X",
)

PAYLOAD_BLOCK_PREFIX = "P"
CHECK_BLOCK_PREFIX = "C"

PAYLOAD_BLOCK_COUNT = 128
CHECK_BLOCK_COUNT = 32
TOTAL_LOGICAL_BLOCK_COUNT = 160

_BLOCK_ID_PATTERN = re.compile(
    r"^[PC]\d{4,}$"
)


class LogicalQubitError(Exception):
    """Base exception for logical-qubit processing."""


class InvalidLogicalRoleError(
    LogicalQubitError
):
    """Raised when a logical-qubit role is invalid."""


class InvalidLogicalIndexError(
    LogicalQubitError
):
    """Raised when a logical index is invalid."""


class InvalidLogicalPositionError(
    LogicalQubitError
):
    """Raised when an interleaved position is invalid."""


class InvalidLogicalBlockIDError(
    LogicalQubitError
):
    """Raised when a block identifier is malformed."""


class LogicalQubitCollectionError(
    LogicalQubitError
):
    """Raised when a logical-qubit collection is inconsistent."""


def normalize_role(
    role: Any,
) -> LogicalRole:
    """
    Validate and normalize a logical-qubit role.
    """

    if not isinstance(role, str):
        raise InvalidLogicalRoleError(
            "role must be a string."
        )

    normalized_role = (
        role.strip().lower()
    )

    if normalized_role not in (
        SUPPORTED_ROLES
    ):
        raise InvalidLogicalRoleError(
            "role must be either "
            "'payload' or 'check'."
        )

    return normalized_role  # type: ignore[return-value]


def validate_logical_index(
    logical_index: Any,
) -> int:
    """
    Validate one non-negative logical index.
    """

    if isinstance(
        logical_index,
        bool,
    ):
        raise InvalidLogicalIndexError(
            "logical_index cannot be boolean."
        )

    if not isinstance(
        logical_index,
        int,
    ):
        raise InvalidLogicalIndexError(
            "logical_index must be an integer."
        )

    if logical_index < 0:
        raise InvalidLogicalIndexError(
            "logical_index cannot be negative."
        )

    return logical_index


def validate_position(
    position: Any,
) -> int | None:
    """
    Validate an optional interleaved frame position.
    """

    if position is None:
        return None

    if isinstance(position, bool):
        raise InvalidLogicalPositionError(
            "position cannot be boolean."
        )

    if not isinstance(position, int):
        raise InvalidLogicalPositionError(
            "position must be an integer or None."
        )

    if position < 0:
        raise InvalidLogicalPositionError(
            "position cannot be negative."
        )

    return position


def validate_block_id(
    block_id: Any,
    role: str | None = None,
) -> str:
    """
    Validate a logical block identifier.

    Standard FT-QuPAP identifiers:

        Payload:
            P0000, P0001, ...

        Check:
            C0000, C0001, ...
    """

    if not isinstance(block_id, str):
        raise InvalidLogicalBlockIDError(
            "block_id must be a string."
        )

    normalized_id = (
        block_id.strip().upper()
    )

    if not normalized_id:
        raise InvalidLogicalBlockIDError(
            "block_id cannot be empty."
        )

    if not _BLOCK_ID_PATTERN.fullmatch(
        normalized_id
    ):
        raise InvalidLogicalBlockIDError(
            "block_id must begin with P or C "
            "and contain at least four digits, "
            "for example P0000 or C0000."
        )

    if role is not None:
        normalized_role = normalize_role(
            role
        )

        required_prefix = (
            PAYLOAD_BLOCK_PREFIX
            if normalized_role
            == PAYLOAD_ROLE
            else CHECK_BLOCK_PREFIX
        )

        if not normalized_id.startswith(
            required_prefix
        ):
            raise InvalidLogicalBlockIDError(
                f"{normalized_role} block IDs "
                f"must begin with "
                f"{required_prefix!r}."
            )

    return normalized_id


def block_id_for(
    role: str,
    logical_index: int,
) -> str:
    """
    Construct the standard FT-QuPAP logical block ID.
    """

    normalized_role = normalize_role(
        role
    )

    normalized_index = (
        validate_logical_index(
            logical_index
        )
    )

    prefix = (
        PAYLOAD_BLOCK_PREFIX
        if normalized_role
        == PAYLOAD_ROLE
        else CHECK_BLOCK_PREFIX
    )

    return (
        f"{prefix}"
        f"{normalized_index:04d}"
    )


def ket_for(
    logical_bit: int,
    basis: str,
) -> str:
    """
    Return the human-readable logical state label.

    Z basis:

        0 -> |0_L>
        1 -> |1_L>

    X basis:

        0 -> |+_L>
        1 -> |-_L>
    """

    normalized_bit = (
        validate_classical_bit(
            logical_bit
        )
    )

    normalized_basis = (
        validate_basis(
            basis
        )
    )

    if normalized_basis == "Z":
        return (
            "|0_L>"
            if normalized_bit == 0
            else "|1_L>"
        )

    return (
        "|+_L>"
        if normalized_bit == 0
        else "|-_L>"
    )


@dataclass
class LogicalQubit:
    """
    One FT-QuPAP logical payload or check state.

    Attributes:
        block_id:
            Unique block identifier.

        role:
            Either "payload" or "check".

        logical_index:
            Original position inside the payload or check collection.

        logical_bit:
            Encoded classical value, either 0 or 1.

        basis:
            Logical preparation basis, either Z or X.

        position:
            Position after random payload/check interleaving.
            It remains None before interleaving.
    """

    block_id: str
    role: LogicalRole
    logical_index: int
    logical_bit: int
    basis: LogicalBasis
    position: int | None = None

    def __post_init__(self) -> None:
        self.role = normalize_role(
            self.role
        )

        self.logical_index = (
            validate_logical_index(
                self.logical_index
            )
        )

        self.logical_bit = (
            validate_classical_bit(
                self.logical_bit
            )
        )

        self.basis = validate_basis(
            self.basis
        )  # type: ignore[assignment]

        self.position = (
            validate_position(
                self.position
            )
        )

        self.block_id = (
            validate_block_id(
                self.block_id,
                self.role,
            )
        )

        if (
            self.role == PAYLOAD_ROLE
            and self.basis != "Z"
        ):
            raise LogicalQubitError(
                "FT-QuPAP payload logical qubits "
                "must use the Z basis."
            )

    @property
    def ket(self) -> str:
        """
        Return the logical state label.
        """

        return ket_for(
            self.logical_bit,
            self.basis,
        )

    @property
    def classical_ket(self) -> str:
        """
        Return the corresponding unencoded state label.

        Examples:

            |0>, |1>, |+>, |->
        """

        return state_label(
            self.logical_bit,
            self.basis,
        )

    @property
    def is_payload(self) -> bool:
        """Return True for a payload block."""

        return (
            self.role == PAYLOAD_ROLE
        )

    @property
    def is_check(self) -> bool:
        """Return True for a check block."""

        return (
            self.role == CHECK_ROLE
        )

    @property
    def is_positioned(self) -> bool:
        """
        Return True after interleaving assigns a position.
        """

        return self.position is not None

    def assign_position(
        self,
        position: int,
    ) -> None:
        """
        Assign an interleaved frame position in place.
        """

        self.position = (
            validate_position(
                position
            )
        )

    def clear_position(self) -> None:
        """
        Remove the assigned interleaved position.
        """

        self.position = None

    def copy(
        self,
    ) -> LogicalQubit:
        """
        Return an independent copy.
        """

        return copy.deepcopy(self)

    def copy_with_position(
        self,
        position: int,
    ) -> LogicalQubit:
        """
        Return an independent copy with a new position.
        """

        copied = self.copy()

        copied.assign_position(
            position
        )

        return copied

    def to_dictionary(
        self,
    ) -> dict[str, Any]:
        """
        Return a serializable notebook-compatible dictionary.
        """

        return {
            "block_id":
                self.block_id,
            "role":
                self.role,
            "logical_index":
                self.logical_index,
            "logical_bit":
                self.logical_bit,
            "basis":
                self.basis,
            "position":
                self.position,
        }

    def to_schedule_entry(
        self,
    ) -> dict[str, Any]:
        """
        Convert this logical qubit into its control-schedule entry.

        Check entries reveal the basis and expected logical bit only
        inside the encrypted control schedule.

        Payload entries identify the original tag-bit index.
        """

        if self.position is None:
            raise InvalidLogicalPositionError(
                "A logical qubit must be interleaved "
                "before creating a schedule entry."
            )

        if self.is_check:
            return {
                "block_id":
                    self.block_id,
                "position":
                    self.position,
                "basis":
                    self.basis,
                "expected_logical_bit":
                    self.logical_bit,
            }

        return {
            "block_id":
                self.block_id,
            "position":
                self.position,
            "logical_index":
                self.logical_index,
        }

    @classmethod
    def from_dictionary(
        cls,
        data: dict[str, Any],
    ) -> LogicalQubit:
        """
        Reconstruct a logical qubit from a dictionary.
        """

        if not isinstance(data, dict):
            raise TypeError(
                "data must be a dictionary."
            )

        required_fields = {
            "block_id",
            "role",
            "logical_index",
            "logical_bit",
            "basis",
        }

        missing_fields = (
            required_fields.difference(
                data.keys()
            )
        )

        if missing_fields:
            raise LogicalQubitError(
                "Logical-qubit dictionary is "
                "missing fields: "
                f"{sorted(missing_fields)}"
            )

        return cls(
            block_id=data[
                "block_id"
            ],
            role=data[
                "role"
            ],
            logical_index=data[
                "logical_index"
            ],
            logical_bit=data[
                "logical_bit"
            ],
            basis=data[
                "basis"
            ],
            position=data.get(
                "position"
            ),
        )

    @classmethod
    def from_state_description(
        cls,
        state: QubitStateDescription,
        role: str,
        block_id: str | None = None,
        position: int | None = None,
    ) -> LogicalQubit:
        """
        Construct a logical qubit from classical_to_qubit output.
        """

        if not isinstance(
            state,
            QubitStateDescription,
        ):
            raise TypeError(
                "state must be a QubitStateDescription."
            )

        normalized_role = normalize_role(
            role
        )

        selected_block_id = (
            block_id
            if block_id is not None
            else block_id_for(
                normalized_role,
                state.bit_index,
            )
        )

        return cls(
            block_id=selected_block_id,
            role=normalized_role,
            logical_index=(
                state.bit_index
            ),
            logical_bit=(
                state.classical_bit
            ),
            basis=state.basis,
            position=position,
        )


# Notebook-compatible name.
LogicalSpec = LogicalQubit


def create_payload_logical_qubit(
    logical_bit: int,
    logical_index: int,
    position: int | None = None,
) -> LogicalQubit:
    """
    Create one standard Z-basis payload logical qubit.
    """

    normalized_index = (
        validate_logical_index(
            logical_index
        )
    )

    return LogicalQubit(
        block_id=block_id_for(
            PAYLOAD_ROLE,
            normalized_index,
        ),
        role=PAYLOAD_ROLE,
        logical_index=(
            normalized_index
        ),
        logical_bit=logical_bit,
        basis="Z",
        position=position,
    )


def create_check_logical_qubit(
    logical_bit: int,
    basis: str,
    logical_index: int,
    position: int | None = None,
) -> LogicalQubit:
    """
    Create one independent Z- or X-basis check logical qubit.
    """

    normalized_index = (
        validate_logical_index(
            logical_index
        )
    )

    return LogicalQubit(
        block_id=block_id_for(
            CHECK_ROLE,
            normalized_index,
        ),
        role=CHECK_ROLE,
        logical_index=(
            normalized_index
        ),
        logical_bit=logical_bit,
        basis=validate_basis(
            basis
        ),
        position=position,
    )


def validate_logical_qubit_collection(
    logical_qubits: Sequence[LogicalQubit],
    expected_role: str | None = None,
    expected_count: int | None = None,
    require_positions: bool = False,
    require_unique_ids: bool = True,
    require_unique_positions: bool = True,
) -> None:
    """
    Validate a collection of logical qubits.
    """

    if isinstance(
        logical_qubits,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "logical_qubits must be a sequence."
        )

    if not isinstance(
        logical_qubits,
        Sequence,
    ):
        raise TypeError(
            "logical_qubits must be a sequence."
        )

    if expected_count is not None:
        if isinstance(
            expected_count,
            bool,
        ) or not isinstance(
            expected_count,
            int,
        ):
            raise TypeError(
                "expected_count must be an integer or None."
            )

        if expected_count < 0:
            raise ValueError(
                "expected_count cannot be negative."
            )

        if len(logical_qubits) != (
            expected_count
        ):
            raise LogicalQubitCollectionError(
                "Expected "
                f"{expected_count} logical qubits, "
                f"received {len(logical_qubits)}."
            )

    normalized_expected_role = (
        normalize_role(
            expected_role
        )
        if expected_role is not None
        else None
    )

    seen_ids: set[str] = set()
    seen_positions: set[int] = set()

    for index, logical_qubit in enumerate(
        logical_qubits
    ):
        if not isinstance(
            logical_qubit,
            LogicalQubit,
        ):
            raise TypeError(
                "Every item must be a LogicalQubit. "
                f"Invalid item at index {index}."
            )

        if (
            normalized_expected_role
            is not None
            and logical_qubit.role
            != normalized_expected_role
        ):
            raise LogicalQubitCollectionError(
                f"Logical qubit {logical_qubit.block_id!r} "
                f"has role {logical_qubit.role!r}; "
                f"expected "
                f"{normalized_expected_role!r}."
            )

        if require_unique_ids:
            if (
                logical_qubit.block_id
                in seen_ids
            ):
                raise LogicalQubitCollectionError(
                    "Duplicate logical block ID: "
                    f"{logical_qubit.block_id!r}."
                )

            seen_ids.add(
                logical_qubit.block_id
            )

        if require_positions:
            if logical_qubit.position is None:
                raise LogicalQubitCollectionError(
                    f"Logical qubit "
                    f"{logical_qubit.block_id!r} "
                    "has no interleaved position."
                )

        if (
            logical_qubit.position
            is not None
            and require_unique_positions
        ):
            if (
                logical_qubit.position
                in seen_positions
            ):
                raise LogicalQubitCollectionError(
                    "Duplicate interleaved position: "
                    f"{logical_qubit.position}."
                )

            seen_positions.add(
                logical_qubit.position
            )


def assign_sequential_positions(
    logical_qubits: Sequence[LogicalQubit],
    start_position: int = 0,
    in_place: bool = False,
) -> list[LogicalQubit]:
    """
    Assign sequential positions to logical qubits.

    The actual FT-QuPAP block interleaver should randomize the block
    order first and then call this operation.
    """

    normalized_start = (
        validate_position(
            start_position
        )
    )

    if normalized_start is None:
        raise InvalidLogicalPositionError(
            "start_position cannot be None."
        )

    validate_logical_qubit_collection(
        logical_qubits,
        require_unique_ids=True,
        require_unique_positions=False,
    )

    positioned_qubits = (
        list(logical_qubits)
        if in_place
        else [
            logical_qubit.copy()
            for logical_qubit
            in logical_qubits
        ]
    )

    for offset, logical_qubit in enumerate(
        positioned_qubits
    ):
        logical_qubit.assign_position(
            normalized_start + offset
        )

    return positioned_qubits


def clear_positions(
    logical_qubits: Sequence[LogicalQubit],
    in_place: bool = False,
) -> list[LogicalQubit]:
    """
    Remove all interleaved positions.
    """

    validate_logical_qubit_collection(
        logical_qubits,
        require_unique_ids=True,
        require_unique_positions=False,
    )

    result = (
        list(logical_qubits)
        if in_place
        else [
            logical_qubit.copy()
            for logical_qubit
            in logical_qubits
        ]
    )

    for logical_qubit in result:
        logical_qubit.clear_position()

    return result


def extract_logical_bits(
    logical_qubits: Iterable[LogicalQubit],
    sort_by_logical_index: bool = False,
) -> list[int]:
    """
    Extract logical bit values.

    Payload recovery should normally sort by logical_index after
    removing the randomly interleaved check blocks.
    """

    qubit_list = list(
        logical_qubits
    )

    validate_logical_qubit_collection(
        qubit_list,
        require_unique_ids=True,
        require_unique_positions=False,
    )

    if sort_by_logical_index:
        qubit_list = sorted(
            qubit_list,
            key=lambda qubit:
                qubit.logical_index,
        )

    return [
        qubit.logical_bit
        for qubit in qubit_list
    ]


def separate_by_role(
    logical_qubits: Sequence[LogicalQubit],
) -> tuple[
    list[LogicalQubit],
    list[LogicalQubit],
]:
    """
    Separate payload and check logical qubits.

    Returns:
        payload_qubits, check_qubits
    """

    validate_logical_qubit_collection(
        logical_qubits,
        require_unique_ids=True,
        require_unique_positions=False,
    )

    payload_qubits = [
        logical_qubit
        for logical_qubit
        in logical_qubits
        if logical_qubit.is_payload
    ]

    check_qubits = [
        logical_qubit
        for logical_qubit
        in logical_qubits
        if logical_qubit.is_check
    ]

    return (
        payload_qubits,
        check_qubits,
    )


def run_self_test() -> None:
    """
    Verify logical payload/check models and positioning.
    """

    print("=" * 70)
    print("FT-QuPAP Logical Qubit Self-Test")
    print("=" * 70)

    payload_qubits = [
        create_payload_logical_qubit(
            logical_bit=index % 2,
            logical_index=index,
        )
        for index in range(4)
    ]

    check_qubits = [
        create_check_logical_qubit(
            logical_bit=0,
            basis="Z",
            logical_index=0,
        ),
        create_check_logical_qubit(
            logical_bit=1,
            basis="X",
            logical_index=1,
        ),
    ]

    combined = (
        payload_qubits
        + check_qubits
    )

    positioned = (
        assign_sequential_positions(
            combined,
            start_position=0,
            in_place=False,
        )
    )

    separated_payload, separated_check = (
        separate_by_role(
            positioned
        )
    )

    payload_bits = (
        extract_logical_bits(
            separated_payload,
            sort_by_logical_index=True,
        )
    )

    payload_kets = [
        qubit.ket
        for qubit
        in payload_qubits
    ]

    check_kets = [
        qubit.ket
        for qubit
        in check_qubits
    ]

    source_positions_unchanged = all(
        qubit.position is None
        for qubit in combined
    )

    copied_positions_assigned = all(
        qubit.position == index
        for index, qubit
        in enumerate(positioned)
    )

    dictionary_round_trip = (
        LogicalQubit.from_dictionary(
            check_qubits[1]
            .to_dictionary()
        )
        == check_qubits[1]
    )

    print(
        f"Payload blocks created     : "
        f"{len(payload_qubits)}"
    )

    print(
        f"Check blocks created       : "
        f"{len(check_qubits)}"
    )

    print(
        f"Payload states             : "
        f"{payload_kets}"
    )

    print(
        f"Check states               : "
        f"{check_kets}"
    )

    print(
        f"Recovered payload bits     : "
        f"{payload_bits}"
    )

    print(
        f"Source positions unchanged : "
        f"{source_positions_unchanged}"
    )

    print(
        f"Copied positions assigned  : "
        f"{copied_positions_assigned}"
    )

    print(
        f"Dictionary round-trip      : "
        f"{dictionary_round_trip}"
    )

    if payload_kets != [
        "|0_L>",
        "|1_L>",
        "|0_L>",
        "|1_L>",
    ]:
        raise LogicalQubitError(
            "Payload logical-state mapping failed."
        )

    if check_kets != [
        "|0_L>",
        "|-_L>",
    ]:
        raise LogicalQubitError(
            "Check logical-state mapping failed."
        )

    if payload_bits != [
        0,
        1,
        0,
        1,
    ]:
        raise LogicalQubitError(
            "Payload bit extraction failed."
        )

    if not source_positions_unchanged:
        raise LogicalQubitError(
            "Position assignment modified the source collection."
        )

    if not copied_positions_assigned:
        raise LogicalQubitError(
            "Sequential position assignment failed."
        )

    if not dictionary_round_trip:
        raise LogicalQubitError(
            "Logical-qubit dictionary round-trip failed."
        )

    print(
        "\nLogical qubit self-test "
        "completed successfully."
    )


__all__ = [
    "LogicalRole",
    "LogicalBasis",
    "PAYLOAD_ROLE",
    "CHECK_ROLE",
    "SUPPORTED_ROLES",
    "SUPPORTED_BASES",
    "PAYLOAD_BLOCK_COUNT",
    "CHECK_BLOCK_COUNT",
    "TOTAL_LOGICAL_BLOCK_COUNT",
    "LogicalQubit",
    "LogicalSpec",
    "LogicalQubitError",
    "InvalidLogicalRoleError",
    "InvalidLogicalIndexError",
    "InvalidLogicalPositionError",
    "InvalidLogicalBlockIDError",
    "LogicalQubitCollectionError",
    "normalize_role",
    "validate_logical_index",
    "validate_position",
    "validate_block_id",
    "block_id_for",
    "ket_for",
    "create_payload_logical_qubit",
    "create_check_logical_qubit",
    "validate_logical_qubit_collection",
    "assign_sequential_positions",
    "clear_positions",
    "extract_logical_bits",
    "separate_by_role",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        LogicalQubitError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[LOGICAL QUBIT ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error