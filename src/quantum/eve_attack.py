"""
FT-QuPAP Eve Intercept-Measure-Resend Attack Model

This module implements the simulator-only Eve attack layer used by
FT-QuPAP.

Supported attack mode:

    intercept_resend

For each non-erased physical position:

1. Eve intercepts with probability eve_fraction.
2. Eve independently selects the Z or X basis.
3. Eve measures the physical qubit in that basis.
4. Eve resends the measured state.
5. A wrong-basis measurement can introduce observable disturbance.

Standard notebook scenarios:

    PARTIAL_EVE_CHANNEL:
        bit_flip_prob     = 0.010
        phase_flip_prob   = 0.010
        depolarizing_prob = 0.005
        loss_prob         = 0.005
        eve_fraction      = 0.35

    FULL_EVE_CHANNEL:
        eve_fraction      = 1.00

Security boundary:

The following values are hidden simulator ground truth:

    eve_fraction
    Eve-selected bases
    attacked physical positions
    attacked_mask
    attack event counts
    actual attack label

These values must never be supplied directly to the Authentication
Server Gaussian Process feature extractor. The detector may use only
receiver-observable evidence such as raw QBER, syndrome statistics,
correction failures, loss rate, trusted noise estimate, and context.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .quantum_channel import (
    ChannelConfig,
    apply_bit_flip,
    apply_phase_flip,
    validate_physical_frame,
)
from .steane_css import (
    PhysicalBlock,
    SteaneEncodedFrame,
    validate_physical_block,
)


EVE_MODE_NONE = "none"
EVE_MODE_INTERCEPT_RESEND = "intercept_resend"

SUPPORTED_EVE_MODES = (
    EVE_MODE_NONE,
    EVE_MODE_INTERCEPT_RESEND,
)

SUPPORTED_EVE_BASES = (
    "Z",
    "X",
)


# ============================================================
# Notebook-aligned Eve scenarios
# ============================================================

PARTIAL_EVE_CHANNEL = ChannelConfig(
    name="partial_eve",
    bit_flip_prob=0.010,
    phase_flip_prob=0.010,
    depolarizing_prob=0.005,
    loss_prob=0.005,
    eve_fraction=0.35,
    eve_mode=EVE_MODE_INTERCEPT_RESEND,
    context="urban",
)


FULL_EVE_CHANNEL = ChannelConfig(
    name="full_eve",
    bit_flip_prob=0.0,
    phase_flip_prob=0.0,
    depolarizing_prob=0.0,
    loss_prob=0.0,
    eve_fraction=1.0,
    eve_mode=EVE_MODE_INTERCEPT_RESEND,
    context="urban",
)


STANDARD_EVE_CHANNELS: dict[str, ChannelConfig] = {
    PARTIAL_EVE_CHANNEL.name:
        PARTIAL_EVE_CHANNEL,
    FULL_EVE_CHANNEL.name:
        FULL_EVE_CHANNEL,
}


class EveAttackError(Exception):
    """Base exception for Eve attack simulation."""


class InvalidEveConfigurationError(
    EveAttackError
):
    """Raised when Eve attack settings are invalid."""


class InvalidEveAttackPlanError(
    EveAttackError
):
    """Raised when an attack plan is malformed."""


class EveAttackApplicationError(
    EveAttackError
):
    """Raised when an attack cannot be applied."""


@dataclass(frozen=True)
class EveAttackConfig:
    """
    Standalone Eve attack configuration.

    Attributes:
        mode:
            Either "none" or "intercept_resend".

        fraction:
            Probability that Eve intercepts each non-erased physical
            position.
    """

    mode: str = EVE_MODE_NONE
    fraction: float = 0.0

    def __post_init__(self) -> None:
        validate_eve_mode(
            self.mode
        )

        validate_eve_fraction(
            self.fraction
        )

        if (
            self.mode == EVE_MODE_NONE
            and self.fraction != 0.0
        ):
            raise InvalidEveConfigurationError(
                "fraction must be zero when mode='none'."
            )

        if (
            self.mode
            == EVE_MODE_INTERCEPT_RESEND
            and self.fraction <= 0.0
        ):
            raise InvalidEveConfigurationError(
                "intercept_resend requires a positive fraction."
            )

    @property
    def enabled(self) -> bool:
        """Return whether Eve interception is enabled."""

        return (
            self.mode
            == EVE_MODE_INTERCEPT_RESEND
            and self.fraction > 0.0
        )

    def hidden_dictionary(
        self,
    ) -> dict[str, Any]:
        """
        Return simulator-only attack configuration.
        """

        return {
            "eve_mode":
                self.mode,
            "eve_fraction":
                self.fraction,
            "attack_enabled":
                self.enabled,
        }


@dataclass(frozen=True)
class EveAttackPlanEntry:
    """
    One planned physical interception.

    This object is simulator-only ground truth.
    """

    frame_position: int
    block_id: str
    physical_position: int
    preparation_basis: str
    eve_basis: str

    def __post_init__(self) -> None:
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

        if (
            isinstance(
                self.physical_position,
                bool,
            )
            or not isinstance(
                self.physical_position,
                int,
            )
        ):
            raise TypeError(
                "physical_position must be an integer."
            )

        if self.physical_position < 0:
            raise ValueError(
                "physical_position cannot be negative."
            )

        validate_eve_basis(
            self.preparation_basis
        )

        validate_eve_basis(
            self.eve_basis
        )

    @property
    def basis_match(self) -> bool:
        """
        Return whether Eve selected the preparation basis.
        """

        return (
            self.preparation_basis
            == self.eve_basis
        )

    def hidden_dictionary(
        self,
    ) -> dict[str, Any]:
        """Return simulator-only plan information."""

        return {
            "frame_position":
                self.frame_position,
            "block_id":
                self.block_id,
            "physical_position":
                self.physical_position,
            "preparation_basis":
                self.preparation_basis,
            "eve_basis":
                self.eve_basis,
            "basis_match":
                self.basis_match,
        }


@dataclass(frozen=True)
class EveAttackEvent:
    """
    Simulator-only result of one physical interception.
    """

    frame_position: int
    block_id: str
    physical_position: int

    preparation_basis: str
    eve_basis: str
    basis_match: bool

    x_error_changed: bool
    z_error_changed: bool

    def hidden_dictionary(
        self,
    ) -> dict[str, Any]:
        """Return internal attack-event information."""

        return {
            "frame_position":
                self.frame_position,
            "block_id":
                self.block_id,
            "physical_position":
                self.physical_position,
            "preparation_basis":
                self.preparation_basis,
            "eve_basis":
                self.eve_basis,
            "basis_match":
                self.basis_match,
            "x_error_changed":
                self.x_error_changed,
            "z_error_changed":
                self.z_error_changed,
        }


@dataclass(frozen=True)
class EveAttackResult:
    """
    Result of applying Eve interception to one physical frame.

    sent_blocks:
        Independent copies of the frame before Eve interception.

    attacked_blocks:
        Independent copies after Eve interception.

    attack_plan:
        Hidden selected positions and Eve bases.

    attack_events:
        Hidden generated disturbance details.

    elapsed_time_s:
        Simulator runtime.
    """

    sent_blocks: tuple[PhysicalBlock, ...]
    attacked_blocks: tuple[PhysicalBlock, ...]

    configuration: EveAttackConfig

    attack_plan: tuple[
        EveAttackPlanEntry,
        ...,
    ]

    attack_events: tuple[
        EveAttackEvent,
        ...,
    ]

    elapsed_time_s: float

    def __post_init__(self) -> None:
        validate_physical_frame(
            self.sent_blocks
        )

        validate_physical_frame(
            self.attacked_blocks
        )

        if len(self.sent_blocks) != len(
            self.attacked_blocks
        ):
            raise EveAttackApplicationError(
                "Sent and attacked frames have different lengths."
            )

        sent_ids = [
            block.block_id
            for block in self.sent_blocks
        ]

        attacked_ids = [
            block.block_id
            for block in self.attacked_blocks
        ]

        if sent_ids != attacked_ids:
            raise EveAttackApplicationError(
                "Eve attack changed block IDs or frame order."
            )

        if not isinstance(
            self.configuration,
            EveAttackConfig,
        ):
            raise TypeError(
                "configuration must be EveAttackConfig."
            )

        if len(self.attack_plan) != len(
            self.attack_events
        ):
            raise EveAttackApplicationError(
                "Attack-plan and attack-event counts differ."
            )

        if (
            isinstance(
                self.elapsed_time_s,
                bool,
            )
            or not isinstance(
                self.elapsed_time_s,
                (int, float),
            )
        ):
            raise TypeError(
                "elapsed_time_s must be numeric."
            )

        if self.elapsed_time_s < 0:
            raise ValueError(
                "elapsed_time_s cannot be negative."
            )

    @property
    def logical_block_count(self) -> int:
        """Return logical block count."""

        return len(
            self.attacked_blocks
        )

    @property
    def physical_position_count(self) -> int:
        """Return total physical positions."""

        return sum(
            block.physical_qubit_count
            for block in self.attacked_blocks
        )

    @property
    def intercepted_position_count(self) -> int:
        """
        Return hidden intercepted-position count.
        """

        return len(
            self.attack_events
        )

    @property
    def intercepted_fraction_realized(self) -> float:
        """
        Return hidden realized attack fraction.
        """

        eligible_positions = sum(
            int(
                np.count_nonzero(
                    np.logical_not(
                        block.erasures
                    )
                )
            )
            for block in self.sent_blocks
        )

        if eligible_positions == 0:
            return 0.0

        return float(
            self.intercepted_position_count
            / eligible_positions
        )

    @property
    def wrong_basis_interceptions(self) -> int:
        """
        Return hidden wrong-basis interception count.
        """

        return sum(
            not event.basis_match
            for event in self.attack_events
        )

    @property
    def disturbed_interceptions(self) -> int:
        """
        Return events that changed an X or Z error component.
        """

        return sum(
            event.x_error_changed
            or event.z_error_changed
            for event in self.attack_events
        )

    def receiver_visible_summary(
        self,
    ) -> dict[str, Any]:
        """
        Return information safe for receiver-side use.

        Attack configuration, attacked positions, Eve bases, and
        attack counts are deliberately excluded.
        """

        erased_positions = sum(
            int(
                np.count_nonzero(
                    block.erasures
                )
            )
            for block in self.attacked_blocks
        )

        return {
            "logical_blocks":
                self.logical_block_count,
            "physical_positions":
                self.physical_position_count,
            "erased_physical_positions":
                erased_positions,
            "processing_time_s":
                float(
                    self.elapsed_time_s
                ),
        }

    def simulator_summary(
        self,
        include_hidden_attack_truth: bool = False,
    ) -> dict[str, Any]:
        """
        Return experiment diagnostics.

        Hidden attack truth is provided only when explicitly enabled.
        """

        result = {
            **self.receiver_visible_summary(),
        }

        if include_hidden_attack_truth:
            result.update(
                {
                    **self.configuration
                    .hidden_dictionary(),
                    "planned_interceptions":
                        len(
                            self.attack_plan
                        ),
                    "realized_interceptions":
                        self.intercepted_position_count,
                    "realized_eve_fraction":
                        self.intercepted_fraction_realized,
                    "wrong_basis_interceptions":
                        self.wrong_basis_interceptions,
                    "disturbed_interceptions":
                        self.disturbed_interceptions,
                }
            )

        return result


def validate_eve_fraction(
    eve_fraction: Any,
) -> float:
    """
    Validate an Eve interception probability in [0, 1].
    """

    if isinstance(
        eve_fraction,
        bool,
    ):
        raise InvalidEveConfigurationError(
            "eve_fraction cannot be boolean."
        )

    if not isinstance(
        eve_fraction,
        (int, float),
    ):
        raise InvalidEveConfigurationError(
            "eve_fraction must be numeric."
        )

    normalized_fraction = float(
        eve_fraction
    )

    if not np.isfinite(
        normalized_fraction
    ):
        raise InvalidEveConfigurationError(
            "eve_fraction must be finite."
        )

    if not 0.0 <= normalized_fraction <= 1.0:
        raise InvalidEveConfigurationError(
            "eve_fraction must be between zero and one."
        )

    return normalized_fraction


def validate_eve_mode(
    eve_mode: Any,
) -> str:
    """Validate an Eve attack mode."""

    if not isinstance(
        eve_mode,
        str,
    ):
        raise InvalidEveConfigurationError(
            "eve_mode must be a string."
        )

    normalized_mode = (
        eve_mode.strip().lower()
    )

    if normalized_mode not in (
        SUPPORTED_EVE_MODES
    ):
        raise InvalidEveConfigurationError(
            "eve_mode must be 'none' or "
            "'intercept_resend'."
        )

    return normalized_mode


def validate_eve_basis(
    eve_basis: Any,
) -> str:
    """Validate a Z- or X-basis value."""

    if not isinstance(
        eve_basis,
        str,
    ):
        raise InvalidEveConfigurationError(
            "Eve basis must be a string."
        )

    normalized_basis = (
        eve_basis.strip().upper()
    )

    if normalized_basis not in (
        SUPPORTED_EVE_BASES
    ):
        raise InvalidEveConfigurationError(
            "Eve basis must be Z or X."
        )

    return normalized_basis


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


def attack_config_from_channel(
    channel: ChannelConfig,
) -> EveAttackConfig:
    """
    Extract simulator-only Eve settings from a channel.
    """

    if not isinstance(
        channel,
        ChannelConfig,
    ):
        raise TypeError(
            "channel must be a ChannelConfig."
        )

    return EveAttackConfig(
        mode=channel.eve_mode,
        fraction=channel.eve_fraction,
    )


def get_eve_channel(
    channel_name: str,
) -> ChannelConfig:
    """
    Return a standard notebook Eve channel.

    Accepted names:

        partial_eve
        partial
        full_eve
        full
    """

    if not isinstance(
        channel_name,
        str,
    ):
        raise TypeError(
            "channel_name must be a string."
        )

    normalized_name = (
        channel_name.strip().lower()
    )

    aliases = {
        "partial":
            "partial_eve",
        "partial_eve":
            "partial_eve",
        "full":
            "full_eve",
        "full_eve":
            "full_eve",
    }

    canonical_name = aliases.get(
        normalized_name
    )

    if canonical_name is None:
        raise InvalidEveConfigurationError(
            "Unknown Eve channel. Use partial_eve or full_eve."
        )

    return STANDARD_EVE_CHANNELS[
        canonical_name
    ]


def normalize_frame(
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

        blocks = list(
            frame
        )

    validate_physical_frame(
        blocks
    )

    return blocks


def choose_eve_basis(
    rng: np.random.Generator,
) -> str:
    """
    Select Eve's Z or X measurement basis independently.
    """

    validate_rng(
        rng
    )

    return str(
        rng.choice(
            SUPPORTED_EVE_BASES
        )
    )


def generate_eve_attack_plan(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    configuration: EveAttackConfig,
    rng: np.random.Generator | None = None,
) -> list[EveAttackPlanEntry]:
    """
    Generate a hidden intercept-resend plan.

    Erased physical positions are not eligible for interception.
    """

    if not isinstance(
        configuration,
        EveAttackConfig,
    ):
        raise TypeError(
            "configuration must be EveAttackConfig."
        )

    if rng is None:
        rng = np.random.default_rng()

    validate_rng(
        rng
    )

    blocks = normalize_frame(
        frame
    )

    if not configuration.enabled:
        return []

    plan: list[
        EveAttackPlanEntry
    ] = []

    for frame_position, block in enumerate(
        blocks
    ):
        for physical_position in range(
            block.physical_qubit_count
        ):
            if block.erasures[
                physical_position
            ]:
                continue

            if rng.random() >= (
                configuration.fraction
            ):
                continue

            plan.append(
                EveAttackPlanEntry(
                    frame_position=(
                        frame_position
                    ),
                    block_id=(
                        block.block_id
                    ),
                    physical_position=(
                        physical_position
                    ),
                    preparation_basis=(
                        block.spec.basis
                    ),
                    eve_basis=(
                        choose_eve_basis(
                            rng
                        )
                    ),
                )
            )

    validate_eve_attack_plan(
        frame=blocks,
        attack_plan=plan,
    )

    return plan


def validate_eve_attack_plan(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    attack_plan: Sequence[EveAttackPlanEntry],
) -> None:
    """
    Validate attack-plan positions and block IDs.
    """

    blocks = normalize_frame(
        frame
    )

    if isinstance(
        attack_plan,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "attack_plan must be a sequence."
        )

    if not isinstance(
        attack_plan,
        Sequence,
    ):
        raise TypeError(
            "attack_plan must be a sequence."
        )

    used_positions: set[
        tuple[int, int]
    ] = set()

    for entry in attack_plan:
        if not isinstance(
            entry,
            EveAttackPlanEntry,
        ):
            raise TypeError(
                "Every plan item must be EveAttackPlanEntry."
            )

        if not 0 <= entry.frame_position < len(
            blocks
        ):
            raise InvalidEveAttackPlanError(
                "Attack-plan frame position is outside the frame."
            )

        block = blocks[
            entry.frame_position
        ]

        if block.block_id != entry.block_id:
            raise InvalidEveAttackPlanError(
                "Attack-plan block ID does not match "
                "its frame position."
            )

        if not 0 <= entry.physical_position < (
            block.physical_qubit_count
        ):
            raise InvalidEveAttackPlanError(
                "Attack-plan physical position is outside "
                f"block {block.block_id!r}."
            )

        if block.erasures[
            entry.physical_position
        ]:
            raise InvalidEveAttackPlanError(
                "An erased physical position cannot be intercepted."
            )

        if (
            block.spec.basis
            != entry.preparation_basis
        ):
            raise InvalidEveAttackPlanError(
                "Attack-plan preparation basis does not "
                "match the physical block."
            )

        position_key = (
            entry.frame_position,
            entry.physical_position,
        )

        if position_key in used_positions:
            raise InvalidEveAttackPlanError(
                "Attack plan contains a duplicate physical position."
            )

        used_positions.add(
            position_key
        )


def apply_intercept_resend_event(
    block: PhysicalBlock,
    physical_position: int,
    eve_basis: str,
    rng: np.random.Generator,
) -> EveAttackEvent:
    """
    Apply one syndrome-level intercept-measure-resend event.

    When Eve selects the correct basis, the resend operation produces
    no additional modeled disturbance.

    When Eve selects the wrong basis:

    - a 50% probability introduces an error relevant to the original
      preparation basis;
    - an additional 20% probability introduces the complementary
      Pauli component.

    This is the scalable syndrome-level approximation used for the
    full-session simulator. Representative Qiskit validation performs
    explicit measurement and resend operations.
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

    validate_rng(
        rng
    )

    selected_basis = validate_eve_basis(
        eve_basis
    )

    if (
        isinstance(
            physical_position,
            bool,
        )
        or not isinstance(
            physical_position,
            int,
        )
    ):
        raise TypeError(
            "physical_position must be an integer."
        )

    if not 0 <= physical_position < (
        block.physical_qubit_count
    ):
        raise ValueError(
            "physical_position is outside the block."
        )

    if block.erasures[
        physical_position
    ]:
        raise EveAttackApplicationError(
            "Cannot intercept an erased physical position."
        )

    before_x = int(
        block.x_errors[
            physical_position
        ]
    )

    before_z = int(
        block.z_errors[
            physical_position
        ]
    )

    block.attacked_mask[
        physical_position
    ] = True

    preparation_basis = (
        block.spec.basis
    )

    basis_match = (
        selected_basis
        == preparation_basis
    )

    if not basis_match:
        # Main wrong-basis disturbance.
        if rng.random() < 0.50:
            if preparation_basis == "Z":
                apply_bit_flip(
                    block,
                    physical_position,
                )

            else:
                apply_phase_flip(
                    block,
                    physical_position,
                )

        # Secondary complementary disturbance.
        if rng.random() < 0.20:
            if preparation_basis == "Z":
                apply_phase_flip(
                    block,
                    physical_position,
                )

            else:
                apply_bit_flip(
                    block,
                    physical_position,
                )

    after_x = int(
        block.x_errors[
            physical_position
        ]
    )

    after_z = int(
        block.z_errors[
            physical_position
        ]
    )

    return EveAttackEvent(
        frame_position=(
            block.spec.position
            if block.spec.position is not None
            else 0
        ),
        block_id=block.block_id,
        physical_position=(
            physical_position
        ),
        preparation_basis=(
            preparation_basis
        ),
        eve_basis=selected_basis,
        basis_match=basis_match,
        x_error_changed=(
            before_x != after_x
        ),
        z_error_changed=(
            before_z != after_z
        ),
    )


def apply_eve_attack_plan(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    attack_plan: Sequence[EveAttackPlanEntry],
    rng: np.random.Generator | None = None,
    clear_existing_attack_marks: bool = True,
) -> tuple[
    list[PhysicalBlock],
    list[EveAttackEvent],
]:
    """
    Apply a validated attack plan without modifying source blocks.
    """

    if rng is None:
        rng = np.random.default_rng()

    validate_rng(
        rng
    )

    if not isinstance(
        clear_existing_attack_marks,
        bool,
    ):
        raise TypeError(
            "clear_existing_attack_marks must be boolean."
        )

    source_blocks = normalize_frame(
        frame
    )

    validate_eve_attack_plan(
        frame=source_blocks,
        attack_plan=attack_plan,
    )

    attacked_blocks = copy.deepcopy(
        source_blocks
    )

    if clear_existing_attack_marks:
        for block in attacked_blocks:
            block.attacked_mask = np.zeros(
                block.physical_qubit_count,
                dtype=bool,
            )

    events: list[
        EveAttackEvent
    ] = []

    for entry in attack_plan:
        block = attacked_blocks[
            entry.frame_position
        ]

        event = apply_intercept_resend_event(
            block=block,
            physical_position=(
                entry.physical_position
            ),
            eve_basis=(
                entry.eve_basis
            ),
            rng=rng,
        )

        # Preserve the explicit plan frame position.
        event = EveAttackEvent(
            frame_position=(
                entry.frame_position
            ),
            block_id=event.block_id,
            physical_position=(
                event.physical_position
            ),
            preparation_basis=(
                event.preparation_basis
            ),
            eve_basis=(
                event.eve_basis
            ),
            basis_match=(
                event.basis_match
            ),
            x_error_changed=(
                event.x_error_changed
            ),
            z_error_changed=(
                event.z_error_changed
            ),
        )

        events.append(
            event
        )

    validate_physical_frame(
        attacked_blocks
    )

    return (
        attacked_blocks,
        events,
    )


def apply_eve_attack(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    configuration: EveAttackConfig,
    rng: np.random.Generator | None = None,
) -> EveAttackResult:
    """
    Generate and apply an intercept-measure-resend attack.
    """

    if not isinstance(
        configuration,
        EveAttackConfig,
    ):
        raise TypeError(
            "configuration must be EveAttackConfig."
        )

    if rng is None:
        rng = np.random.default_rng()

    validate_rng(
        rng
    )

    source_blocks = normalize_frame(
        frame
    )

    sent_blocks = copy.deepcopy(
        source_blocks
    )

    started_at = time.perf_counter()

    attack_plan = generate_eve_attack_plan(
        frame=source_blocks,
        configuration=configuration,
        rng=rng,
    )

    attacked_blocks, events = (
        apply_eve_attack_plan(
            frame=source_blocks,
            attack_plan=attack_plan,
            rng=rng,
            clear_existing_attack_marks=True,
        )
    )

    elapsed_time_s = (
        time.perf_counter()
        - started_at
    )

    return EveAttackResult(
        sent_blocks=tuple(
            sent_blocks
        ),
        attacked_blocks=tuple(
            attacked_blocks
        ),
        configuration=configuration,
        attack_plan=tuple(
            attack_plan
        ),
        attack_events=tuple(
            events
        ),
        elapsed_time_s=(
            elapsed_time_s
        ),
    )


def apply_channel_eve_attack(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    channel: ChannelConfig,
    rng: np.random.Generator | None = None,
) -> EveAttackResult:
    """
    Apply only the Eve portion of a ChannelConfig.

    Ordinary bit, phase, depolarizing, and loss effects remain the
    responsibility of quantum_channel.py.
    """

    return apply_eve_attack(
        frame=frame,
        configuration=(
            attack_config_from_channel(
                channel
            )
        ),
        rng=rng,
    )


def count_attacked_positions(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> int:
    """
    Count hidden attacked-mask positions.

    Never use this value as a GP input feature.
    """

    blocks = normalize_frame(
        frame
    )

    return sum(
        int(
            np.count_nonzero(
                block.attacked_mask
            )
        )
        for block in blocks
    )


def clear_attack_metadata(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    in_place: bool = False,
) -> list[PhysicalBlock]:
    """
    Clear hidden attacked-mask values.

    Physical X/Z disturbances are retained because they are observable
    through later measurement and syndrome processing.
    """

    blocks = normalize_frame(
        frame
    )

    result = (
        blocks
        if in_place
        else copy.deepcopy(
            blocks
        )
    )

    for block in result:
        block.attacked_mask = np.zeros(
            block.physical_qubit_count,
            dtype=bool,
        )

    return result


# ============================================================
# Representative Qiskit attack helpers
# ============================================================

def generate_qiskit_attack_plan(
    data_qubits: Sequence[int],
    eve_fraction: float,
    rng: np.random.Generator,
) -> dict[int, str]:
    """
    Generate the notebook-compatible Qiskit attack plan.

    Returns:

        {
            data_qubit_index: "Z" or "X"
        }
    """

    normalized_fraction = (
        validate_eve_fraction(
            eve_fraction
        )
    )

    validate_rng(
        rng
    )

    if isinstance(
        data_qubits,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "data_qubits must be an integer sequence."
        )

    if not isinstance(
        data_qubits,
        Sequence,
    ):
        raise TypeError(
            "data_qubits must be a sequence."
        )

    normalized_qubits: list[int] = []

    for qubit in data_qubits:
        if (
            isinstance(
                qubit,
                bool,
            )
            or not isinstance(
                qubit,
                int,
            )
        ):
            raise TypeError(
                "Every data-qubit index must be an integer."
            )

        if qubit < 0:
            raise ValueError(
                "Data-qubit indices cannot be negative."
            )

        normalized_qubits.append(
            qubit
        )

    if len(normalized_qubits) != len(
        set(normalized_qubits)
    ):
        raise ValueError(
            "data_qubits contains duplicate indices."
        )

    attack_plan: dict[
        int,
        str,
    ] = {}

    for qubit in normalized_qubits:
        if rng.random() < normalized_fraction:
            attack_plan[qubit] = (
                choose_eve_basis(
                    rng
                )
            )

    return attack_plan


def apply_qiskit_attack_plan(
    circuit: Any,
    attack_plan: Mapping[int, str],
    eve_classical_bits: Sequence[int],
) -> None:
    """
    Add notebook-compatible Eve measurement and resend operations.

    For a Z-basis interception:

        measure

    For an X-basis interception:

        H
        measure
        H

    The function uses duck typing so importing this module does not
    require Qiskit unless circuit validation is executed.
    """

    if circuit is None:
        raise TypeError(
            "circuit cannot be None."
        )

    if not isinstance(
        attack_plan,
        Mapping,
    ):
        raise TypeError(
            "attack_plan must be a mapping."
        )

    if not isinstance(
        eve_classical_bits,
        Sequence,
    ):
        raise TypeError(
            "eve_classical_bits must be a sequence."
        )

    for qubit, eve_basis in attack_plan.items():
        if (
            isinstance(
                qubit,
                bool,
            )
            or not isinstance(
                qubit,
                int,
            )
        ):
            raise TypeError(
                "Attack-plan qubit keys must be integers."
            )

        selected_basis = (
            validate_eve_basis(
                eve_basis
            )
        )

        if not 0 <= qubit < len(
            eve_classical_bits
        ):
            raise ValueError(
                "No Eve classical bit exists for "
                f"data qubit {qubit}."
            )

        if selected_basis == "X":
            circuit.h(
                qubit
            )

        circuit.measure(
            qubit,
            eve_classical_bits[
                qubit
            ],
        )

        # Measurement leaves the qubit in Eve's measured state.
        # Applying H restores the X-basis resend representation.
        if selected_basis == "X":
            circuit.h(
                qubit
            )


def run_self_test() -> None:
    """
    Verify no-attack, partial, full, and Qiskit-plan behavior.
    """

    from .logical_qubit import (
        create_check_logical_qubit,
        create_payload_logical_qubit,
    )
    from .steane_css import (
        encode_one_logical_qubit,
    )

    print("=" * 72)
    print("FT-QuPAP Eve Attack Self-Test")
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
                7000 + index
            ),
        )
        for index, spec in enumerate(
            logical_specs
        )
    ]

    no_attack_result = apply_eve_attack(
        frame=encoded_blocks,
        configuration=EveAttackConfig(),
        rng=np.random.default_rng(
            8001
        ),
    )

    first_partial_result = (
        apply_channel_eve_attack(
            frame=encoded_blocks,
            channel=PARTIAL_EVE_CHANNEL,
            rng=np.random.default_rng(
                8002
            ),
        )
    )

    second_partial_result = (
        apply_channel_eve_attack(
            frame=encoded_blocks,
            channel=PARTIAL_EVE_CHANNEL,
            rng=np.random.default_rng(
                8002
            ),
        )
    )

    full_result = apply_channel_eve_attack(
        frame=encoded_blocks,
        channel=FULL_EVE_CHANNEL,
        rng=np.random.default_rng(
            8003
        ),
    )

    total_physical_positions = sum(
        block.physical_qubit_count
        for block in encoded_blocks
    )

    no_attack_is_clean = (
        no_attack_result
        .intercepted_position_count
        == 0
    )

    partial_reproducible = (
        first_partial_result.attack_plan
        == second_partial_result.attack_plan
        and first_partial_result.attack_events
        == second_partial_result.attack_events
    )

    full_attack_intercepted_all = (
        full_result
        .intercepted_position_count
        == total_physical_positions
    )

    full_attacked_masks_set = (
        count_attacked_positions(
            full_result.attacked_blocks
        )
        == total_physical_positions
    )

    source_frame_unchanged = all(
        not np.any(
            block.attacked_mask
        )
        and not np.any(
            block.x_errors
        )
        and not np.any(
            block.z_errors
        )
        for block in encoded_blocks
    )

    safe_summary = (
        full_result
        .receiver_visible_summary()
    )

    hidden_fields_excluded = all(
        field_name not in safe_summary
        for field_name in (
            "eve_mode",
            "eve_fraction",
            "attack_enabled",
            "attack_plan",
            "attack_events",
            "intercepted_position_count",
            "realized_eve_fraction",
            "wrong_basis_interceptions",
            "disturbed_interceptions",
            "attacked_mask",
        )
    )

    full_qiskit_plan = (
        generate_qiskit_attack_plan(
            data_qubits=list(
                range(7)
            ),
            eve_fraction=1.0,
            rng=np.random.default_rng(
                8004
            ),
        )
    )

    qiskit_full_plan_valid = (
        len(full_qiskit_plan) == 7
        and all(
            basis in SUPPORTED_EVE_BASES
            for basis in (
                full_qiskit_plan.values()
            )
        )
    )

    partial_profile_correct = (
        PARTIAL_EVE_CHANNEL
        .bit_flip_prob
        == 0.010
        and PARTIAL_EVE_CHANNEL
        .phase_flip_prob
        == 0.010
        and PARTIAL_EVE_CHANNEL
        .depolarizing_prob
        == 0.005
        and PARTIAL_EVE_CHANNEL
        .loss_prob
        == 0.005
        and PARTIAL_EVE_CHANNEL
        .eve_fraction
        == 0.35
        and PARTIAL_EVE_CHANNEL
        .eve_mode
        == EVE_MODE_INTERCEPT_RESEND
    )

    full_profile_correct = (
        FULL_EVE_CHANNEL
        .eve_fraction
        == 1.0
        and FULL_EVE_CHANNEL
        .eve_mode
        == EVE_MODE_INTERCEPT_RESEND
    )

    print(
        "Physical positions tested : "
        f"{total_physical_positions}"
    )

    print(
        "No-attack interceptions   : "
        f"{no_attack_result.intercepted_position_count}"
    )

    print(
        "Partial interceptions     : "
        f"{first_partial_result.intercepted_position_count}"
    )

    print(
        "Full interceptions        : "
        f"{full_result.intercepted_position_count}"
    )

    print(
        "Partial seeded reproducible: "
        f"{partial_reproducible}"
    )

    print(
        "Full attack intercepted all: "
        f"{full_attack_intercepted_all}"
    )

    print(
        "Full attacked masks set   : "
        f"{full_attacked_masks_set}"
    )

    print(
        "Source frame unchanged    : "
        f"{source_frame_unchanged}"
    )

    print(
        "Safe summary hides Eve    : "
        f"{hidden_fields_excluded}"
    )

    print(
        "Partial profile correct   : "
        f"{partial_profile_correct}"
    )

    print(
        "Full profile correct      : "
        f"{full_profile_correct}"
    )

    print(
        "Qiskit full plan valid    : "
        f"{qiskit_full_plan_valid}"
    )

    if not no_attack_is_clean:
        raise EveAttackError(
            "No-attack mode generated an interception."
        )

    if not partial_reproducible:
        raise EveAttackError(
            "Identical seeds generated different "
            "partial attack results."
        )

    if not full_attack_intercepted_all:
        raise EveAttackError(
            "Full Eve attack did not intercept "
            "every eligible position."
        )

    if not full_attacked_masks_set:
        raise EveAttackError(
            "Full Eve attack did not mark all positions."
        )

    if not source_frame_unchanged:
        raise EveAttackError(
            "Eve simulation modified the source frame."
        )

    if not hidden_fields_excluded:
        raise EveAttackError(
            "Receiver-visible summary exposed Eve truth."
        )

    if not partial_profile_correct:
        raise EveAttackError(
            "Partial Eve channel differs from the notebook."
        )

    if not full_profile_correct:
        raise EveAttackError(
            "Full Eve channel differs from the notebook."
        )

    if not qiskit_full_plan_valid:
        raise EveAttackError(
            "Qiskit attack-plan generation failed."
        )

    print(
        "\nEve attack self-test "
        "completed successfully."
    )


__all__ = [
    "EVE_MODE_NONE",
    "EVE_MODE_INTERCEPT_RESEND",
    "SUPPORTED_EVE_MODES",
    "SUPPORTED_EVE_BASES",
    "PARTIAL_EVE_CHANNEL",
    "FULL_EVE_CHANNEL",
    "STANDARD_EVE_CHANNELS",
    "EveAttackError",
    "InvalidEveConfigurationError",
    "InvalidEveAttackPlanError",
    "EveAttackApplicationError",
    "EveAttackConfig",
    "EveAttackPlanEntry",
    "EveAttackEvent",
    "EveAttackResult",
    "validate_eve_fraction",
    "validate_eve_mode",
    "validate_eve_basis",
    "validate_rng",
    "attack_config_from_channel",
    "get_eve_channel",
    "normalize_frame",
    "choose_eve_basis",
    "generate_eve_attack_plan",
    "validate_eve_attack_plan",
    "apply_intercept_resend_event",
    "apply_eve_attack_plan",
    "apply_eve_attack",
    "apply_channel_eve_attack",
    "count_attacked_positions",
    "clear_attack_metadata",
    "generate_qiskit_attack_plan",
    "apply_qiskit_attack_plan",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        EveAttackError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[EVE ATTACK ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error