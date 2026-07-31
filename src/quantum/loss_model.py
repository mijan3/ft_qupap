"""
FT-QuPAP Physical-Qubit Loss and Erasure Model

This module models physical-qubit loss separately from bit, phase,
depolarizing, and eavesdropping disturbances.

For every physical position:

    erasure = Bernoulli(loss_probability)

An erased position is marked in:

    PhysicalBlock.erasures

Protocol interpretation:

1. Loss is receiver-observable channel evidence.
2. Loss rate is calculated over physical positions.
3. A block containing any erasure cannot be silently accepted.
4. Required payload blocks containing erasures fail decoding.
5. Check blocks containing erasures are excluded from raw-QBER
   observations.
6. Excessive session loss causes deterministic rejection.
7. Loss rate may be supplied to the GP because it is observable.
8. Hidden Eve configuration is never used by this module.

Notebook policy:

    maximum acceptable loss rate = 0.15

Research boundary:

This is a syndrome-level erasure model. It does not simulate photon
detectors, dark counts, optical attenuation, or real quantum hardware.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .steane_css import (
    PhysicalBlock,
    SteaneEncodedFrame,
    validate_physical_block,
)


MAX_ACCEPTABLE_LOSS_RATE = 0.15

LOSS_POLICY_ACCEPTED = "loss_rate_acceptable"
LOSS_POLICY_REJECTED = "loss_rate_exceeds_policy"

PAYLOAD_ROLE = "payload"
CHECK_ROLE = "check"


class LossModelError(Exception):
    """Base exception for FT-QuPAP loss-model failures."""


class InvalidLossProbabilityError(LossModelError):
    """Raised when a loss probability is invalid."""


class InvalidLossFrameError(LossModelError):
    """Raised when a physical frame is invalid."""


class InvalidErasurePositionError(LossModelError):
    """Raised when an erasure position is invalid."""


@dataclass(frozen=True)
class LossStatistics:
    """
    Receiver-observable loss statistics for one physical frame.

    Attributes:
        total_blocks:
            Number of logical/physical blocks in the frame.

        total_physical_positions:
            Number of physical positions transmitted.

        erased_physical_positions:
            Number of physical positions marked as erased.

        observed_physical_positions:
            Number of physical positions successfully observed.

        loss_rate:
            erased_physical_positions divided by
            total_physical_positions.

        erased_blocks:
            Number of blocks containing one or more erasures.

        fully_erased_blocks:
            Number of blocks where every physical position is erased.

        fully_observed_blocks:
            Number of blocks containing no erasures.

        payload_erased_blocks:
            Number of payload blocks containing erasures.

        check_erased_blocks:
            Number of check blocks containing erasures.

        payload_fully_observed_blocks:
            Number of payload blocks containing no erasures.

        check_fully_observed_blocks:
            Number of check blocks containing no erasures.

        block_erasure_rate:
            erased_blocks divided by total_blocks.
    """

    total_blocks: int
    total_physical_positions: int
    erased_physical_positions: int
    observed_physical_positions: int
    loss_rate: float

    erased_blocks: int
    fully_erased_blocks: int
    fully_observed_blocks: int

    payload_erased_blocks: int
    check_erased_blocks: int

    payload_fully_observed_blocks: int
    check_fully_observed_blocks: int

    block_erasure_rate: float

    def __post_init__(self) -> None:
        integer_fields = {
            "total_blocks":
                self.total_blocks,
            "total_physical_positions":
                self.total_physical_positions,
            "erased_physical_positions":
                self.erased_physical_positions,
            "observed_physical_positions":
                self.observed_physical_positions,
            "erased_blocks":
                self.erased_blocks,
            "fully_erased_blocks":
                self.fully_erased_blocks,
            "fully_observed_blocks":
                self.fully_observed_blocks,
            "payload_erased_blocks":
                self.payload_erased_blocks,
            "check_erased_blocks":
                self.check_erased_blocks,
            "payload_fully_observed_blocks":
                self.payload_fully_observed_blocks,
            "check_fully_observed_blocks":
                self.check_fully_observed_blocks,
        }

        for field_name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{field_name} must be an integer."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if self.total_blocks <= 0:
            raise ValueError(
                "total_blocks must be greater than zero."
            )

        if self.total_physical_positions <= 0:
            raise ValueError(
                "total_physical_positions must be greater than zero."
            )

        if (
            self.erased_physical_positions
            + self.observed_physical_positions
            != self.total_physical_positions
        ):
            raise ValueError(
                "Erased and observed physical-position counts "
                "are inconsistent."
            )

        if (
            self.erased_blocks
            + self.fully_observed_blocks
            != self.total_blocks
        ):
            raise ValueError(
                "Erased and fully observed block counts "
                "are inconsistent."
            )

        if not 0.0 <= self.loss_rate <= 1.0:
            raise ValueError(
                "loss_rate must be between zero and one."
            )

        if not 0.0 <= self.block_erasure_rate <= 1.0:
            raise ValueError(
                "block_erasure_rate must be between zero and one."
            )

        expected_loss_rate = (
            self.erased_physical_positions
            / self.total_physical_positions
        )

        if not np.isclose(
            self.loss_rate,
            expected_loss_rate,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "loss_rate is inconsistent with physical counts."
            )

        expected_block_rate = (
            self.erased_blocks
            / self.total_blocks
        )

        if not np.isclose(
            self.block_erasure_rate,
            expected_block_rate,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "block_erasure_rate is inconsistent "
                "with block counts."
            )

    @property
    def has_loss(self) -> bool:
        """Return whether any physical position was erased."""

        return self.erased_physical_positions > 0

    @property
    def all_positions_erased(self) -> bool:
        """Return whether every physical position was erased."""

        return (
            self.erased_physical_positions
            == self.total_physical_positions
        )

    @property
    def within_default_policy(self) -> bool:
        """
        Return whether loss satisfies the notebook policy.
        """

        return (
            self.loss_rate
            <= MAX_ACCEPTABLE_LOSS_RATE
        )

    def to_dictionary(self) -> dict[str, Any]:
        """Return serializable receiver-observable statistics."""

        return {
            "total_blocks":
                self.total_blocks,
            "total_physical_positions":
                self.total_physical_positions,
            "erased_physical_positions":
                self.erased_physical_positions,
            "observed_physical_positions":
                self.observed_physical_positions,
            "loss_rate":
                self.loss_rate,
            "erased_blocks":
                self.erased_blocks,
            "fully_erased_blocks":
                self.fully_erased_blocks,
            "fully_observed_blocks":
                self.fully_observed_blocks,
            "payload_erased_blocks":
                self.payload_erased_blocks,
            "check_erased_blocks":
                self.check_erased_blocks,
            "payload_fully_observed_blocks":
                self.payload_fully_observed_blocks,
            "check_fully_observed_blocks":
                self.check_fully_observed_blocks,
            "block_erasure_rate":
                self.block_erasure_rate,
            "within_default_policy":
                self.within_default_policy,
        }


@dataclass(frozen=True)
class LossPolicyDecision:
    """
    Deterministic loss-policy decision.

    This decision remains separate from GP attack detection.
    """

    accepted: bool
    reason: str
    observed_loss_rate: float
    maximum_loss_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError(
                "accepted must be boolean."
            )

        if self.reason not in {
            LOSS_POLICY_ACCEPTED,
            LOSS_POLICY_REJECTED,
        }:
            raise ValueError(
                "Unsupported loss-policy reason."
            )

        validate_loss_probability(
            self.observed_loss_rate,
            field_name="observed_loss_rate",
        )

        validate_loss_probability(
            self.maximum_loss_rate,
            field_name="maximum_loss_rate",
        )

        expected_accepted = (
            self.observed_loss_rate
            <= self.maximum_loss_rate
        )

        if self.accepted != expected_accepted:
            raise ValueError(
                "Loss-policy decision is inconsistent."
            )

    def to_dictionary(self) -> dict[str, Any]:
        """Return a serializable policy decision."""

        return {
            "accepted":
                self.accepted,
            "reason":
                self.reason,
            "observed_loss_rate":
                self.observed_loss_rate,
            "maximum_loss_rate":
                self.maximum_loss_rate,
        }


@dataclass(frozen=True)
class LossModelResult:
    """
    Result of applying physical loss to an encoded frame.

    Attributes:
        sent_blocks:
            Independent copies of the frame before this loss model.

        received_blocks:
            Independent copies after erasures were applied.

        configured_loss_probability:
            Bernoulli probability used for newly generated erasures.

        preserve_existing_erasures:
            Whether earlier erasure indicators were preserved.

        statistics:
            Receiver-observable loss statistics.
    """

    sent_blocks: tuple[PhysicalBlock, ...]
    received_blocks: tuple[PhysicalBlock, ...]

    configured_loss_probability: float
    preserve_existing_erasures: bool

    statistics: LossStatistics

    def __post_init__(self) -> None:
        validate_loss_probability(
            self.configured_loss_probability
        )

        if not isinstance(
            self.preserve_existing_erasures,
            bool,
        ):
            raise TypeError(
                "preserve_existing_erasures must be boolean."
            )

        validate_physical_frame(
            self.sent_blocks
        )

        validate_physical_frame(
            self.received_blocks
        )

        if len(self.sent_blocks) != len(
            self.received_blocks
        ):
            raise InvalidLossFrameError(
                "Sent and received frames have different lengths."
            )

        sent_ids = [
            block.block_id
            for block in self.sent_blocks
        ]

        received_ids = [
            block.block_id
            for block in self.received_blocks
        ]

        if sent_ids != received_ids:
            raise InvalidLossFrameError(
                "Loss model changed block order or identifiers."
            )

        calculated_statistics = calculate_loss_statistics(
            self.received_blocks
        )

        if calculated_statistics != self.statistics:
            raise InvalidLossFrameError(
                "Stored loss statistics do not match "
                "the received frame."
            )

    @property
    def loss_rate(self) -> float:
        """Return the receiver-observable physical loss rate."""

        return self.statistics.loss_rate

    @property
    def erased_physical_positions(self) -> int:
        """Return erased physical-position count."""

        return (
            self.statistics
            .erased_physical_positions
        )

    def receiver_visible_summary(self) -> dict[str, Any]:
        """
        Return receiver-observable loss evidence.

        Configured probability is deliberately excluded because the
        receiver observes erasure outcomes, not simulator ground truth.
        """

        return self.statistics.to_dictionary()

    def simulator_summary(self) -> dict[str, Any]:
        """
        Return full offline simulator information.
        """

        return {
            "configured_loss_probability":
                self.configured_loss_probability,
            "preserve_existing_erasures":
                self.preserve_existing_erasures,
            **self.statistics.to_dictionary(),
        }


def validate_loss_probability(
    loss_probability: Any,
    field_name: str = "loss_probability",
) -> float:
    """
    Validate a finite probability in the interval [0, 1].
    """

    if isinstance(loss_probability, bool):
        raise InvalidLossProbabilityError(
            f"{field_name} cannot be boolean."
        )

    if not isinstance(
        loss_probability,
        (int, float),
    ):
        raise InvalidLossProbabilityError(
            f"{field_name} must be numeric."
        )

    normalized_probability = float(
        loss_probability
    )

    if not np.isfinite(
        normalized_probability
    ):
        raise InvalidLossProbabilityError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= normalized_probability <= 1.0:
        raise InvalidLossProbabilityError(
            f"{field_name} must be between zero and one."
        )

    return normalized_probability


def validate_rng(
    rng: Any,
) -> np.random.Generator:
    """Validate a NumPy random generator."""

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    return rng


def normalize_physical_frame(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> list[PhysicalBlock]:
    """
    Normalize an encoded frame or physical-block sequence.
    """

    if isinstance(
        frame,
        SteaneEncodedFrame,
    ):
        blocks = list(
            frame.frame
        )

    else:
        if isinstance(
            frame,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            raise TypeError(
                "frame must be a physical-block sequence."
            )

        if not isinstance(
            frame,
            Sequence,
        ):
            raise TypeError(
                "frame must be a sequence."
            )

        blocks = list(frame)

    validate_physical_frame(
        blocks
    )

    return blocks


def validate_physical_frame(
    frame: Sequence[PhysicalBlock],
) -> None:
    """
    Validate an FT-QuPAP physical frame.
    """

    if isinstance(
        frame,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "frame must contain PhysicalBlock objects."
        )

    if not isinstance(
        frame,
        Sequence,
    ):
        raise TypeError(
            "frame must be a sequence."
        )

    if not frame:
        raise InvalidLossFrameError(
            "frame cannot be empty."
        )

    seen_ids: set[str] = set()

    for expected_position, block in enumerate(frame):
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
            raise InvalidLossFrameError(
                "Duplicate block ID: "
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
            raise InvalidLossFrameError(
                f"Block {block.block_id!r} is not "
                "at its declared position."
            )


def generate_erasure_mask(
    physical_position_count: int,
    loss_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate independent Bernoulli erasure indicators.
    """

    if isinstance(
        physical_position_count,
        bool,
    ) or not isinstance(
        physical_position_count,
        int,
    ):
        raise TypeError(
            "physical_position_count must be an integer."
        )

    if physical_position_count <= 0:
        raise ValueError(
            "physical_position_count must be positive."
        )

    normalized_probability = (
        validate_loss_probability(
            loss_probability
        )
    )

    validate_rng(rng)

    if normalized_probability == 0.0:
        return np.zeros(
            physical_position_count,
            dtype=bool,
        )

    if normalized_probability == 1.0:
        return np.ones(
            physical_position_count,
            dtype=bool,
        )

    return (
        rng.random(
            physical_position_count
        )
        < normalized_probability
    )


def apply_loss_to_block(
    block: PhysicalBlock,
    loss_probability: float,
    rng: np.random.Generator,
    preserve_existing_erasures: bool = True,
) -> PhysicalBlock:
    """
    Apply physical loss to one block without modifying the source.

    When preserve_existing_erasures=True, newly generated erasures
    are combined with existing erasures using logical OR.
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

    normalized_probability = (
        validate_loss_probability(
            loss_probability
        )
    )

    validate_rng(rng)

    if not isinstance(
        preserve_existing_erasures,
        bool,
    ):
        raise TypeError(
            "preserve_existing_erasures must be boolean."
        )

    received_block = copy.deepcopy(
        block
    )

    generated_mask = generate_erasure_mask(
        physical_position_count=(
            received_block.physical_qubit_count
        ),
        loss_probability=(
            normalized_probability
        ),
        rng=rng,
    )

    if preserve_existing_erasures:
        received_block.erasures = np.logical_or(
            received_block.erasures,
            generated_mask,
        )

    else:
        received_block.erasures = (
            generated_mask.copy()
        )

    validate_physical_block(
        received_block
    )

    return received_block


def apply_loss_model(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    loss_probability: float,
    rng: np.random.Generator | None = None,
    preserve_existing_erasures: bool = True,
) -> LossModelResult:
    """
    Apply independent physical erasures to a complete frame.

    Source blocks remain unchanged.
    """

    normalized_probability = (
        validate_loss_probability(
            loss_probability
        )
    )

    if rng is None:
        rng = np.random.default_rng()

    validate_rng(rng)

    if not isinstance(
        preserve_existing_erasures,
        bool,
    ):
        raise TypeError(
            "preserve_existing_erasures must be boolean."
        )

    source_blocks = normalize_physical_frame(
        frame
    )

    sent_blocks = copy.deepcopy(
        source_blocks
    )

    received_blocks = [
        apply_loss_to_block(
            block=block,
            loss_probability=(
                normalized_probability
            ),
            rng=rng,
            preserve_existing_erasures=(
                preserve_existing_erasures
            ),
        )
        for block in source_blocks
    ]

    statistics = calculate_loss_statistics(
        received_blocks
    )

    return LossModelResult(
        sent_blocks=tuple(
            sent_blocks
        ),
        received_blocks=tuple(
            received_blocks
        ),
        configured_loss_probability=(
            normalized_probability
        ),
        preserve_existing_erasures=(
            preserve_existing_erasures
        ),
        statistics=statistics,
    )


def mark_specific_erasures(
    block: PhysicalBlock,
    erasure_positions: Sequence[int],
    in_place: bool = False,
) -> PhysicalBlock:
    """
    Mark selected physical positions as erased.

    This helper is useful for deterministic unit tests and controlled
    demonstration scenarios.
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

    if isinstance(
        erasure_positions,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "erasure_positions must be an integer sequence."
        )

    if not isinstance(
        erasure_positions,
        Sequence,
    ):
        raise TypeError(
            "erasure_positions must be a sequence."
        )

    target_block = (
        block
        if in_place
        else copy.deepcopy(block)
    )

    normalized_positions: set[int] = set()

    for position in erasure_positions:
        if isinstance(position, bool) or not isinstance(
            position,
            int,
        ):
            raise InvalidErasurePositionError(
                "Every erasure position must be an integer."
            )

        if not 0 <= position < (
            target_block.physical_qubit_count
        ):
            raise InvalidErasurePositionError(
                f"Erasure position {position} is outside "
                f"block {target_block.block_id!r}."
            )

        normalized_positions.add(
            position
        )

    for position in normalized_positions:
        target_block.erasures[
            position
        ] = True

    validate_physical_block(
        target_block
    )

    return target_block


def clear_block_erasures(
    block: PhysicalBlock,
    in_place: bool = False,
) -> PhysicalBlock:
    """
    Remove all erasure indicators from one block.
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

    target_block = (
        block
        if in_place
        else copy.deepcopy(block)
    )

    target_block.erasures = np.zeros(
        target_block.physical_qubit_count,
        dtype=bool,
    )

    return target_block


def calculate_loss_statistics(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> LossStatistics:
    """
    Calculate receiver-observable physical and block loss statistics.

    The main session loss rate is:

        erased physical positions
        -------------------------
        total physical positions
    """

    blocks = normalize_physical_frame(
        frame
    )

    total_physical_positions = sum(
        block.physical_qubit_count
        for block in blocks
    )

    erased_physical_positions = sum(
        int(
            np.count_nonzero(
                block.erasures
            )
        )
        for block in blocks
    )

    observed_physical_positions = (
        total_physical_positions
        - erased_physical_positions
    )

    erased_blocks = [
        block
        for block in blocks
        if np.any(
            block.erasures
        )
    ]

    fully_erased_blocks = [
        block
        for block in blocks
        if np.all(
            block.erasures
        )
    ]

    fully_observed_blocks = [
        block
        for block in blocks
        if not np.any(
            block.erasures
        )
    ]

    payload_erased_blocks = [
        block
        for block in erased_blocks
        if block.role == PAYLOAD_ROLE
    ]

    check_erased_blocks = [
        block
        for block in erased_blocks
        if block.role == CHECK_ROLE
    ]

    payload_fully_observed_blocks = [
        block
        for block in fully_observed_blocks
        if block.role == PAYLOAD_ROLE
    ]

    check_fully_observed_blocks = [
        block
        for block in fully_observed_blocks
        if block.role == CHECK_ROLE
    ]

    total_blocks = len(blocks)

    loss_rate = float(
        erased_physical_positions
        / total_physical_positions
    )

    block_erasure_rate = float(
        len(erased_blocks)
        / total_blocks
    )

    return LossStatistics(
        total_blocks=total_blocks,
        total_physical_positions=(
            total_physical_positions
        ),
        erased_physical_positions=(
            erased_physical_positions
        ),
        observed_physical_positions=(
            observed_physical_positions
        ),
        loss_rate=loss_rate,
        erased_blocks=len(
            erased_blocks
        ),
        fully_erased_blocks=len(
            fully_erased_blocks
        ),
        fully_observed_blocks=len(
            fully_observed_blocks
        ),
        payload_erased_blocks=len(
            payload_erased_blocks
        ),
        check_erased_blocks=len(
            check_erased_blocks
        ),
        payload_fully_observed_blocks=len(
            payload_fully_observed_blocks
        ),
        check_fully_observed_blocks=len(
            check_fully_observed_blocks
        ),
        block_erasure_rate=(
            block_erasure_rate
        ),
    )


def calculate_loss_rate(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> float:
    """
    Return the physical-position loss rate.

    This function produces the notebook-compatible session value
    stored under the `loss_rate` field.
    """

    return calculate_loss_statistics(
        frame
    ).loss_rate


def evaluate_loss_policy(
    loss: float | LossStatistics,
    maximum_loss_rate: float = MAX_ACCEPTABLE_LOSS_RATE,
) -> LossPolicyDecision:
    """
    Apply the deterministic FT-QuPAP session-loss policy.
    """

    if isinstance(
        loss,
        LossStatistics,
    ):
        observed_loss_rate = (
            loss.loss_rate
        )

    else:
        observed_loss_rate = (
            validate_loss_probability(
                loss,
                field_name="loss",
            )
        )

    normalized_maximum = (
        validate_loss_probability(
            maximum_loss_rate,
            field_name="maximum_loss_rate",
        )
    )

    accepted = (
        observed_loss_rate
        <= normalized_maximum
    )

    return LossPolicyDecision(
        accepted=accepted,
        reason=(
            LOSS_POLICY_ACCEPTED
            if accepted
            else LOSS_POLICY_REJECTED
        ),
        observed_loss_rate=(
            observed_loss_rate
        ),
        maximum_loss_rate=(
            normalized_maximum
        ),
    )


def erased_block_ids(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> list[str]:
    """
    Return IDs of blocks containing at least one erasure.
    """

    blocks = normalize_physical_frame(
        frame
    )

    return [
        block.block_id
        for block in blocks
        if np.any(
            block.erasures
        )
    ]


def fully_observed_block_ids(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> list[str]:
    """
    Return IDs of blocks containing no erasures.
    """

    blocks = normalize_physical_frame(
        frame
    )

    return [
        block.block_id
        for block in blocks
        if not np.any(
            block.erasures
        )
    ]


def observed_check_block_count(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> int:
    """
    Count check blocks that contain no erased physical position.

    The final raw-QBER calculator may impose additional schedule and
    measurement validity checks.
    """

    blocks = normalize_physical_frame(
        frame
    )

    return sum(
        block.role == CHECK_ROLE
        and not np.any(
            block.erasures
        )
        for block in blocks
    )


def all_payload_blocks_observed(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> bool:
    """
    Return whether every payload block contains no erasure.

    Payload blocks are mandatory because all 128 logical tag bits are
    required to reconstruct the authentication tag.
    """

    blocks = normalize_physical_frame(
        frame
    )

    payload_blocks = [
        block
        for block in blocks
        if block.role == PAYLOAD_ROLE
    ]

    if not payload_blocks:
        return False

    return all(
        not np.any(
            block.erasures
        )
        for block in payload_blocks
    )


def extract_erasure_records(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> list[dict[str, Any]]:
    """
    Return per-block receiver-observable erasure records.
    """

    blocks = normalize_physical_frame(
        frame
    )

    records: list[dict[str, Any]] = []

    for position, block in enumerate(blocks):
        erasure_count = int(
            np.count_nonzero(
                block.erasures
            )
        )

        records.append(
            {
                "position":
                    position,
                "block_id":
                    block.block_id,
                "role":
                    block.role,
                "physical_position_count":
                    block.physical_qubit_count,
                "erasure_count":
                    erasure_count,
                "erased":
                    erasure_count > 0,
                "fully_erased":
                    erasure_count
                    == block.physical_qubit_count,
                "fully_observed":
                    erasure_count == 0,
                "block_loss_rate":
                    float(
                        erasure_count
                        / block.physical_qubit_count
                    ),
            }
        )

    return records


def run_self_test() -> None:
    """
    Verify ideal, complete-loss, seeded-loss, and policy behavior.
    """

    from .logical_qubit import (
        create_check_logical_qubit,
        create_payload_logical_qubit,
    )
    from .steane_css import (
        encode_one_logical_qubit,
    )

    print("=" * 72)
    print("FT-QuPAP Loss Model Self-Test")
    print("=" * 72)

    logical_specs = [
        create_payload_logical_qubit(
            logical_bit=0,
            logical_index=0,
            position=0,
        ),
        create_payload_logical_qubit(
            logical_bit=1,
            logical_index=1,
            position=1,
        ),
        create_check_logical_qubit(
            logical_bit=0,
            basis="Z",
            logical_index=0,
            position=2,
        ),
        create_check_logical_qubit(
            logical_bit=1,
            basis="X",
            logical_index=1,
            position=3,
        ),
    ]

    encoded_blocks = [
        encode_one_logical_qubit(
            spec=spec,
            use_css=True,
            rng=np.random.default_rng(
                5000 + index
            ),
        )
        for index, spec in enumerate(
            logical_specs
        )
    ]

    ideal_result = apply_loss_model(
        frame=encoded_blocks,
        loss_probability=0.0,
        rng=np.random.default_rng(
            6001
        ),
    )

    complete_loss_result = apply_loss_model(
        frame=encoded_blocks,
        loss_probability=1.0,
        rng=np.random.default_rng(
            6002
        ),
    )

    first_seeded_result = apply_loss_model(
        frame=encoded_blocks,
        loss_probability=0.25,
        rng=np.random.default_rng(
            6003
        ),
    )

    second_seeded_result = apply_loss_model(
        frame=encoded_blocks,
        loss_probability=0.25,
        rng=np.random.default_rng(
            6003
        ),
    )

    seeded_reproducible = all(
        np.array_equal(
            first_block.erasures,
            second_block.erasures,
        )
        for first_block, second_block in zip(
            first_seeded_result.received_blocks,
            second_seeded_result.received_blocks,
            strict=True,
        )
    )

    source_frame_unchanged = all(
        not np.any(
            block.erasures
        )
        for block in encoded_blocks
    )

    marked_block = mark_specific_erasures(
        encoded_blocks[0],
        erasure_positions=[
            1,
            4,
        ],
        in_place=False,
    )

    specific_erasures_correct = (
        int(
            np.count_nonzero(
                marked_block.erasures
            )
        )
        == 2
        and marked_block.erasures[1]
        and marked_block.erasures[4]
    )

    preserved_result = apply_loss_model(
        frame=[
            marked_block,
            *encoded_blocks[1:],
        ],
        loss_probability=0.0,
        rng=np.random.default_rng(
            6004
        ),
        preserve_existing_erasures=True,
    )

    existing_erasure_preserved = (
        preserved_result
        .received_blocks[0]
        .erasures[1]
        and preserved_result
        .received_blocks[0]
        .erasures[4]
    )

    ideal_policy = evaluate_loss_policy(
        ideal_result.statistics
    )

    complete_loss_policy = (
        evaluate_loss_policy(
            complete_loss_result.statistics
        )
    )

    safe_summary = (
        first_seeded_result
        .receiver_visible_summary()
    )

    configured_probability_hidden = (
        "configured_loss_probability"
        not in safe_summary
    )

    print(
        "Physical positions tested : "
        f"{ideal_result.statistics.total_physical_positions}"
    )

    print(
        "Ideal loss rate           : "
        f"{ideal_result.loss_rate:.6f}"
    )

    print(
        "Complete loss rate        : "
        f"{complete_loss_result.loss_rate:.6f}"
    )

    print(
        "Seeded loss rate          : "
        f"{first_seeded_result.loss_rate:.6f}"
    )

    print(
        "Seeded result reproducible: "
        f"{seeded_reproducible}"
    )

    print(
        "Source frame unchanged    : "
        f"{source_frame_unchanged}"
    )

    print(
        "Specific erasures correct : "
        f"{specific_erasures_correct}"
    )

    print(
        "Existing erasures retained: "
        f"{existing_erasure_preserved}"
    )

    print(
        "Ideal loss policy accepts : "
        f"{ideal_policy.accepted}"
    )

    print(
        "Full-loss policy rejects  : "
        f"{not complete_loss_policy.accepted}"
    )

    print(
        "Safe summary hides config : "
        f"{configured_probability_hidden}"
    )

    if ideal_result.loss_rate != 0.0:
        raise LossModelError(
            "Ideal loss model generated an erasure."
        )

    if complete_loss_result.loss_rate != 1.0:
        raise LossModelError(
            "Complete-loss model failed to erase "
            "every physical position."
        )

    if not seeded_reproducible:
        raise LossModelError(
            "Identical seeds produced different "
            "erasure masks."
        )

    if not source_frame_unchanged:
        raise LossModelError(
            "Loss simulation modified source blocks."
        )

    if not specific_erasures_correct:
        raise LossModelError(
            "Specific-erasure helper failed."
        )

    if not existing_erasure_preserved:
        raise LossModelError(
            "Existing erasures were not preserved."
        )

    if not ideal_policy.accepted:
        raise LossModelError(
            "Ideal channel failed the loss policy."
        )

    if complete_loss_policy.accepted:
        raise LossModelError(
            "Complete loss passed the loss policy."
        )

    if complete_loss_policy.reason != (
        LOSS_POLICY_REJECTED
    ):
        raise LossModelError(
            "Incorrect excessive-loss rejection reason."
        )

    if not configured_probability_hidden:
        raise LossModelError(
            "Receiver summary exposed simulator configuration."
        )

    print(
        "\nLoss model self-test "
        "completed successfully."
    )


__all__ = [
    "MAX_ACCEPTABLE_LOSS_RATE",
    "LOSS_POLICY_ACCEPTED",
    "LOSS_POLICY_REJECTED",
    "LossModelError",
    "InvalidLossProbabilityError",
    "InvalidLossFrameError",
    "InvalidErasurePositionError",
    "LossStatistics",
    "LossPolicyDecision",
    "LossModelResult",
    "validate_loss_probability",
    "validate_rng",
    "normalize_physical_frame",
    "validate_physical_frame",
    "generate_erasure_mask",
    "apply_loss_to_block",
    "apply_loss_model",
    "mark_specific_erasures",
    "clear_block_erasures",
    "calculate_loss_statistics",
    "calculate_loss_rate",
    "evaluate_loss_policy",
    "erased_block_ids",
    "fully_observed_block_ids",
    "observed_check_block_count",
    "all_payload_blocks_observed",
    "extract_erasure_records",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        LossModelError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[LOSS MODEL ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error