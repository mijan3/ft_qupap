"""
FT-QuPAP Syndrome-Level Quantum Channel

This module transmits Steane-encoded FT-QuPAP blocks through a noisy
and untrusted syndrome-level quantum channel.

Supported channel effects:

1. Independent physical bit-flip errors
2. Independent physical phase-flip errors
3. Depolarizing X, Y, or Z errors
4. Physical-qubit loss and erasure
5. Optional Eve intercept-measure-resend disturbance

Channel-operation order for every physical position:

    loss/erasure
        -> bit flip
        -> phase flip
        -> depolarizing error
        -> optional Eve interception

A lost physical position receives no later simulated operation.

Security boundary:

The following values are simulator-only ground truth:

    eve_fraction
    eve_mode
    attacked_mask
    intercepted-position count

They must never be included in the Authentication Server's Gaussian
Process feature vector. The GP must use only receiver-observable
evidence such as raw QBER, syndrome statistics, correction failures,
loss rate, and trusted channel context.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .steane_css import (
    PhysicalBlock,
    SteaneEncodedFrame,
    validate_physical_block,
)


SUPPORTED_CONTEXTS = (
    "urban",
    "suburban",
    "rural",
)

SUPPORTED_EVE_MODES = (
    "none",
    "intercept_resend",
)

SUPPORTED_PAULI_ERRORS = (
    "X",
    "Y",
    "Z",
)


class QuantumChannelError(Exception):
    """Base exception for quantum-channel simulation failures."""


class InvalidChannelConfigurationError(QuantumChannelError):
    """Raised when a channel configuration is invalid."""


class InvalidQuantumFrameError(QuantumChannelError):
    """Raised when a physical transmission frame is invalid."""


class QuantumTransmissionError(QuantumChannelError):
    """Raised when quantum transmission cannot be completed."""


@dataclass(frozen=True)
class ChannelConfig:
    """
    Syndrome-level quantum-channel configuration.

    Attributes:
        name:
            Human-readable channel-profile name.

        bit_flip_prob:
            Probability of an independent physical X error.

        phase_flip_prob:
            Probability of an independent physical Z error.

        depolarizing_prob:
            Probability of a random X, Y, or Z Pauli error.

        loss_prob:
            Probability that a physical position is erased.

        eve_fraction:
            Hidden simulator probability that Eve intercepts a
            physical position.

        eve_mode:
            Either "none" or "intercept_resend".

        context:
            Trusted channel context used by the observable feature
            pipeline: urban, suburban, or rural.
    """

    name: str

    bit_flip_prob: float = 0.0
    phase_flip_prob: float = 0.0
    depolarizing_prob: float = 0.0
    loss_prob: float = 0.0

    eve_fraction: float = 0.0
    eve_mode: str = "none"

    context: str = "urban"

    def __post_init__(self) -> None:
        validate_channel_config(self)

    @property
    def attack_enabled(self) -> bool:
        """
        Return hidden simulator attack status.

        This value must not be used as an Authentication Server
        observable feature.
        """

        return (
            self.eve_mode == "intercept_resend"
            and self.eve_fraction > 0.0
        )

    def receiver_visible_dictionary(self) -> dict[str, Any]:
        """
        Return channel information safe for receiver-side processing.

        Eve configuration is deliberately excluded.
        """

        return {
            "name": self.name,
            "context": self.context,
        }

    def simulator_dictionary(
        self,
        include_hidden_attack_settings: bool = False,
    ) -> dict[str, Any]:
        """
        Return simulator channel settings.

        Hidden attack settings are returned only when explicitly
        requested for offline experiment evaluation.
        """

        result: dict[str, Any] = {
            "name": self.name,
            "bit_flip_prob": self.bit_flip_prob,
            "phase_flip_prob": self.phase_flip_prob,
            "depolarizing_prob": self.depolarizing_prob,
            "loss_prob": self.loss_prob,
            "context": self.context,
        }

        if include_hidden_attack_settings:
            result.update(
                {
                    "eve_fraction": self.eve_fraction,
                    "eve_mode": self.eve_mode,
                }
            )

        return result


@dataclass
class ChannelEventCounters:
    """
    Generated event counters for one simulated transmission.

    Eve interception count is hidden simulator information.
    """

    physical_positions: int = 0
    erasure_events: int = 0
    bit_flip_events: int = 0
    phase_flip_events: int = 0
    depolarizing_events: int = 0
    eve_interception_events: int = 0

    def receiver_visible_dictionary(self) -> dict[str, int]:
        """
        Return counters without Eve ground truth.
        """

        return {
            "physical_positions": self.physical_positions,
            "erasure_events": self.erasure_events,
        }

    def simulator_dictionary(
        self,
        include_hidden_attack_events: bool = False,
    ) -> dict[str, int]:
        """
        Return internal channel event counts.
        """

        result = {
            "physical_positions": self.physical_positions,
            "erasure_events": self.erasure_events,
            "bit_flip_events": self.bit_flip_events,
            "phase_flip_events": self.phase_flip_events,
            "depolarizing_events": self.depolarizing_events,
        }

        if include_hidden_attack_events:
            result["eve_interception_events"] = (
                self.eve_interception_events
            )

        return result


@dataclass
class QuantumChannelResult:
    """
    Detailed result of one quantum-channel transmission.

    Attributes:
        sent_blocks:
            Deep-copied original encoded blocks.

        received_blocks:
            Blocks after channel noise, loss, and optional attack.

        channel:
            Channel configuration used for the simulation.

        event_counters:
            Generated channel-event counters.

        simulation_time_s:
            Channel simulation runtime in seconds.
    """

    sent_blocks: list[PhysicalBlock]
    received_blocks: list[PhysicalBlock]
    channel: ChannelConfig
    event_counters: ChannelEventCounters
    simulation_time_s: float

    def __post_init__(self) -> None:
        validate_physical_frame(
            self.sent_blocks
        )

        validate_physical_frame(
            self.received_blocks
        )

        if len(self.sent_blocks) != len(
            self.received_blocks
        ):
            raise InvalidQuantumFrameError(
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
            raise InvalidQuantumFrameError(
                "Received block order does not match "
                "the transmitted frame."
            )

        if not isinstance(
            self.channel,
            ChannelConfig,
        ):
            raise TypeError(
                "channel must be a ChannelConfig."
            )

        if not isinstance(
            self.event_counters,
            ChannelEventCounters,
        ):
            raise TypeError(
                "event_counters must be ChannelEventCounters."
            )

        if isinstance(
            self.simulation_time_s,
            bool,
        ) or not isinstance(
            self.simulation_time_s,
            (int, float),
        ):
            raise TypeError(
                "simulation_time_s must be numeric."
            )

        if self.simulation_time_s < 0:
            raise ValueError(
                "simulation_time_s cannot be negative."
            )

    @property
    def logical_block_count(self) -> int:
        """Return transmitted logical-block count."""

        return len(
            self.received_blocks
        )

    @property
    def physical_position_count(self) -> int:
        """Return total transmitted physical positions."""

        return sum(
            block.physical_qubit_count
            for block in self.received_blocks
        )

    @property
    def erased_physical_positions(self) -> int:
        """Return receiver-observable erased-position count."""

        return sum(
            int(
                np.count_nonzero(
                    block.erasures
                )
            )
            for block in self.received_blocks
        )

    @property
    def received_physical_positions(self) -> int:
        """Return physical positions that were not erased."""

        return (
            self.physical_position_count
            - self.erased_physical_positions
        )

    @property
    def loss_rate(self) -> float:
        """Return receiver-observable physical loss rate."""

        if self.physical_position_count == 0:
            return 1.0

        return float(
            self.erased_physical_positions
            / self.physical_position_count
        )

    @property
    def erased_block_count(self) -> int:
        """Return blocks containing one or more erasures."""

        return sum(
            bool(
                np.any(
                    block.erasures
                )
            )
            for block in self.received_blocks
        )

    @property
    def hidden_intercepted_positions(self) -> int:
        """
        Return hidden simulator Eve interception count.

        Never use this value in the GP feature vector.
        """

        return sum(
            int(
                np.count_nonzero(
                    block.attacked_mask
                )
            )
            for block in self.received_blocks
        )

    def receiver_visible_summary(self) -> dict[str, Any]:
        """
        Return receiver-observable channel information only.
        """

        return {
            "channel_name": self.channel.name,
            "channel_context": self.channel.context,
            "logical_blocks": self.logical_block_count,
            "physical_positions": self.physical_position_count,
            "received_physical_positions": (
                self.received_physical_positions
            ),
            "erased_physical_positions": (
                self.erased_physical_positions
            ),
            "erased_blocks": self.erased_block_count,
            "loss_rate": self.loss_rate,
            "simulation_time_s": float(
                self.simulation_time_s
            ),
        }

    def simulator_summary(
        self,
        include_hidden_attack_truth: bool = False,
    ) -> dict[str, Any]:
        """
        Return experiment diagnostics.

        Hidden attack truth is included only when explicitly requested.
        """

        summary: dict[str, Any] = {
            **self.receiver_visible_summary(),
            "channel_configuration": (
                self.channel.simulator_dictionary(
                    include_hidden_attack_settings=(
                        include_hidden_attack_truth
                    )
                )
            ),
            "generated_events": (
                self.event_counters.simulator_dictionary(
                    include_hidden_attack_events=(
                        include_hidden_attack_truth
                    )
                )
            ),
        }

        if include_hidden_attack_truth:
            summary["hidden_intercepted_positions"] = (
                self.hidden_intercepted_positions
            )

        return summary


def validate_probability(
    value: Any,
    field_name: str,
) -> float:
    """
    Validate a finite probability in the closed interval [0, 1].
    """

    if isinstance(value, bool):
        raise InvalidChannelConfigurationError(
            f"{field_name} cannot be boolean."
        )

    if not isinstance(
        value,
        (int, float),
    ):
        raise InvalidChannelConfigurationError(
            f"{field_name} must be numeric."
        )

    normalized_value = float(value)

    if not np.isfinite(
        normalized_value
    ):
        raise InvalidChannelConfigurationError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= normalized_value <= 1.0:
        raise InvalidChannelConfigurationError(
            f"{field_name} must be between 0 and 1."
        )

    return normalized_value


def validate_channel_config(
    channel: ChannelConfig,
) -> None:
    """
    Validate a complete syndrome-level channel configuration.
    """

    if not isinstance(
        channel.name,
        str,
    ):
        raise InvalidChannelConfigurationError(
            "channel name must be a string."
        )

    if not channel.name.strip():
        raise InvalidChannelConfigurationError(
            "channel name cannot be empty."
        )

    validate_probability(
        channel.bit_flip_prob,
        "bit_flip_prob",
    )

    validate_probability(
        channel.phase_flip_prob,
        "phase_flip_prob",
    )

    validate_probability(
        channel.depolarizing_prob,
        "depolarizing_prob",
    )

    validate_probability(
        channel.loss_prob,
        "loss_prob",
    )

    validate_probability(
        channel.eve_fraction,
        "eve_fraction",
    )

    if channel.eve_mode not in SUPPORTED_EVE_MODES:
        raise InvalidChannelConfigurationError(
            "eve_mode must be 'none' or 'intercept_resend'."
        )

    if channel.context not in SUPPORTED_CONTEXTS:
        raise InvalidChannelConfigurationError(
            "context must be urban, suburban, or rural."
        )

    if (
        channel.eve_mode == "none"
        and channel.eve_fraction != 0.0
    ):
        raise InvalidChannelConfigurationError(
            "eve_fraction must be zero when eve_mode='none'."
        )

    if (
        channel.eve_mode == "intercept_resend"
        and channel.eve_fraction <= 0.0
    ):
        raise InvalidChannelConfigurationError(
            "intercept_resend requires a positive eve_fraction."
        )


def normalize_frame(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
) -> list[PhysicalBlock]:
    """
    Normalize a SteaneEncodedFrame or physical-block sequence.
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
    Validate a physical FT-QuPAP transmission frame.
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
            "frame must be a sequence of PhysicalBlock objects."
        )

    if not isinstance(
        frame,
        Sequence,
    ):
        raise TypeError(
            "frame must be a sequence."
        )

    if not frame:
        raise InvalidQuantumFrameError(
            "frame cannot be empty."
        )

    seen_ids: set[str] = set()

    for expected_position, block in enumerate(
        frame
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
            raise InvalidQuantumFrameError(
                "Duplicate physical block ID: "
                f"{block.block_id!r}."
            )

        seen_ids.add(
            block.block_id
        )

        if (
            block.spec.position is not None
            and block.spec.position != expected_position
        ):
            raise InvalidQuantumFrameError(
                f"Block {block.block_id!r} is not at "
                "its declared frame position."
            )


def apply_bit_flip(
    block: PhysicalBlock,
    qubit_index: int,
) -> None:
    """Apply one physical X-error component."""

    block.x_errors[
        qubit_index
    ] ^= 1


def apply_phase_flip(
    block: PhysicalBlock,
    qubit_index: int,
) -> None:
    """Apply one physical Z-error component."""

    block.z_errors[
        qubit_index
    ] ^= 1


def apply_depolarizing_error(
    block: PhysicalBlock,
    qubit_index: int,
    rng: np.random.Generator,
) -> str:
    """
    Apply one random depolarizing X, Y, or Z error.

    Returns the selected Pauli symbol.
    """

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    pauli = str(
        rng.choice(
            SUPPORTED_PAULI_ERRORS
        )
    )

    if pauli in (
        "X",
        "Y",
    ):
        apply_bit_flip(
            block,
            qubit_index,
        )

    if pauli in (
        "Y",
        "Z",
    ):
        apply_phase_flip(
            block,
            qubit_index,
        )

    return pauli


def apply_intercept_resend(
    block: PhysicalBlock,
    qubit_index: int,
    rng: np.random.Generator,
) -> str:
    """
    Apply the notebook's intercept-measure-resend disturbance.

    Eve randomly selects either the Z or X basis. When Eve's basis
    differs from the block's declared preparation basis, disturbance
    may be introduced.

    Returns:
        Eve's selected basis.

    attacked_mask is hidden simulator ground truth.
    """

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    block.attacked_mask[
        qubit_index
    ] = True

    eve_basis = str(
        rng.choice(
            (
                "Z",
                "X",
            )
        )
    )

    if eve_basis == block.spec.basis:
        return eve_basis

    # Main wrong-basis disturbance.
    if rng.random() < 0.50:
        if block.spec.basis == "Z":
            apply_bit_flip(
                block,
                qubit_index,
            )
        else:
            apply_phase_flip(
                block,
                qubit_index,
            )

    # Secondary disturbance used by the notebook model.
    if rng.random() < 0.20:
        if block.spec.basis == "Z":
            apply_phase_flip(
                block,
                qubit_index,
            )
        else:
            apply_bit_flip(
                block,
                qubit_index,
            )

    return eve_basis


def transmit_one_block(
    sent_block: PhysicalBlock,
    channel: ChannelConfig,
    rng: np.random.Generator,
    event_counters: ChannelEventCounters | None = None,
) -> PhysicalBlock:
    """
    Transmit one encoded physical block through the channel.

    The original block is not modified.
    """

    if not isinstance(
        sent_block,
        PhysicalBlock,
    ):
        raise TypeError(
            "sent_block must be a PhysicalBlock."
        )

    validate_physical_block(
        sent_block
    )

    if not isinstance(
        channel,
        ChannelConfig,
    ):
        raise TypeError(
            "channel must be a ChannelConfig."
        )

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    if event_counters is None:
        event_counters = ChannelEventCounters()

    if not isinstance(
        event_counters,
        ChannelEventCounters,
    ):
        raise TypeError(
            "event_counters must be ChannelEventCounters."
        )

    block = copy.deepcopy(
        sent_block
    )

    # Reset any previous channel state before this transmission.
    block.reset_channel_state()

    physical_count = (
        block.physical_qubit_count
    )

    for qubit_index in range(
        physical_count
    ):
        event_counters.physical_positions += 1

        # -----------------------------------------------------
        # 1. Physical loss / erasure
        # -----------------------------------------------------
        if rng.random() < channel.loss_prob:
            block.erasures[
                qubit_index
            ] = True

            event_counters.erasure_events += 1

            # Lost positions receive no later operation.
            continue

        # -----------------------------------------------------
        # 2. Independent physical bit flip
        # -----------------------------------------------------
        if rng.random() < channel.bit_flip_prob:
            apply_bit_flip(
                block,
                qubit_index,
            )

            event_counters.bit_flip_events += 1

        # -----------------------------------------------------
        # 3. Independent physical phase flip
        # -----------------------------------------------------
        if rng.random() < channel.phase_flip_prob:
            apply_phase_flip(
                block,
                qubit_index,
            )

            event_counters.phase_flip_events += 1

        # -----------------------------------------------------
        # 4. Depolarizing Pauli error
        # -----------------------------------------------------
        if rng.random() < channel.depolarizing_prob:
            apply_depolarizing_error(
                block=block,
                qubit_index=qubit_index,
                rng=rng,
            )

            event_counters.depolarizing_events += 1

        # -----------------------------------------------------
        # 5. Optional Eve intercept-measure-resend
        # -----------------------------------------------------
        if (
            channel.eve_mode == "intercept_resend"
            and rng.random() < channel.eve_fraction
        ):
            apply_intercept_resend(
                block=block,
                qubit_index=qubit_index,
                rng=rng,
            )

            event_counters.eve_interception_events += 1

    return block


def simulate_quantum_channel(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    channel: ChannelConfig,
    rng: np.random.Generator | None = None,
) -> QuantumChannelResult:
    """
    Transmit a complete encoded frame and return detailed results.
    """

    if not isinstance(
        channel,
        ChannelConfig,
    ):
        raise TypeError(
            "channel must be a ChannelConfig."
        )

    validate_channel_config(
        channel
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

    original_blocks = normalize_frame(
        frame
    )

    sent_blocks = copy.deepcopy(
        original_blocks
    )

    event_counters = (
        ChannelEventCounters()
    )

    started_at = time.perf_counter()

    received_blocks = [
        transmit_one_block(
            sent_block=block,
            channel=channel,
            rng=rng,
            event_counters=event_counters,
        )
        for block in original_blocks
    ]

    simulation_time_s = (
        time.perf_counter()
        - started_at
    )

    return QuantumChannelResult(
        sent_blocks=sent_blocks,
        received_blocks=received_blocks,
        channel=channel,
        event_counters=event_counters,
        simulation_time_s=simulation_time_s,
    )


def transmit_blocks_through_channel(
    frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    channel: ChannelConfig,
    rng: np.random.Generator | None = None,
) -> list[PhysicalBlock]:
    """
    Notebook-compatible quantum-channel transmission function.

    Returns only the received physical blocks.
    """

    result = simulate_quantum_channel(
        frame=frame,
        channel=channel,
        rng=rng,
    )

    return result.received_blocks


class QuantumChannel:
    """
    Stateful FT-QuPAP syndrome-level quantum channel.
    """

    def __init__(
        self,
        config: ChannelConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not isinstance(
            config,
            ChannelConfig,
        ):
            raise TypeError(
                "config must be a ChannelConfig."
            )

        validate_channel_config(
            config
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

        self.config = config
        self.rng = rng

    def transmit(
        self,
        frame: SteaneEncodedFrame | Sequence[PhysicalBlock],
    ) -> QuantumChannelResult:
        """Transmit one complete encoded frame."""

        return simulate_quantum_channel(
            frame=frame,
            channel=self.config,
            rng=self.rng,
        )


def count_hidden_intercepted_positions(
    received_blocks: Sequence[PhysicalBlock],
) -> int:
    """
    Count simulator-only Eve-intercepted physical positions.

    Do not use this function in Authentication Server feature
    extraction.
    """

    validate_physical_frame(
        received_blocks
    )

    return sum(
        int(
            np.count_nonzero(
                block.attacked_mask
            )
        )
        for block in received_blocks
    )


def calculate_loss_rate(
    received_blocks: Sequence[PhysicalBlock],
) -> float:
    """
    Calculate receiver-observable physical loss rate.
    """

    validate_physical_frame(
        received_blocks
    )

    total_physical = sum(
        block.physical_qubit_count
        for block in received_blocks
    )

    erased_physical = sum(
        int(
            np.count_nonzero(
                block.erasures
            )
        )
        for block in received_blocks
    )

    if total_physical == 0:
        return 1.0

    return float(
        erased_physical
        / total_physical
    )


def run_self_test() -> None:
    """
    Verify ideal, noisy, lossy, and Eve-channel behavior.
    """

    from .logical_qubit import (
        create_check_logical_qubit,
        create_payload_logical_qubit,
    )
    from .steane_css import (
        encode_one_logical_qubit,
    )

    print("=" * 72)
    print("FT-QuPAP Quantum Channel Self-Test")
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
                2000 + index
            ),
        )
        for index, spec in enumerate(
            logical_specs
        )
    ]

    ideal_channel = ChannelConfig(
        name="ideal_test",
        context="urban",
    )

    deterministic_noise_channel = ChannelConfig(
        name="deterministic_noise_test",
        bit_flip_prob=1.0,
        phase_flip_prob=1.0,
        depolarizing_prob=1.0,
        loss_prob=0.0,
        context="suburban",
    )

    complete_loss_channel = ChannelConfig(
        name="complete_loss_test",
        loss_prob=1.0,
        context="rural",
    )

    full_eve_channel = ChannelConfig(
        name="full_eve_test",
        eve_fraction=1.0,
        eve_mode="intercept_resend",
        context="urban",
    )

    ideal_result = simulate_quantum_channel(
        frame=encoded_blocks,
        channel=ideal_channel,
        rng=np.random.default_rng(
            3001
        ),
    )

    noisy_result = simulate_quantum_channel(
        frame=encoded_blocks,
        channel=deterministic_noise_channel,
        rng=np.random.default_rng(
            3002
        ),
    )

    loss_result = simulate_quantum_channel(
        frame=encoded_blocks,
        channel=complete_loss_channel,
        rng=np.random.default_rng(
            3003
        ),
    )

    eve_result = QuantumChannel(
        config=full_eve_channel,
        rng=np.random.default_rng(
            3004
        ),
    ).transmit(
        encoded_blocks
    )

    ideal_unchanged = all(
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
        for block in ideal_result.received_blocks
    )

    noise_events_generated = (
        noisy_result.event_counters.bit_flip_events > 0
        and noisy_result.event_counters.phase_flip_events > 0
        and noisy_result.event_counters.depolarizing_events > 0
    )

    complete_loss_applied = (
        loss_result.loss_rate == 1.0
    )

    all_positions_intercepted = (
        eve_result.hidden_intercepted_positions
        == eve_result.physical_position_count
    )

    original_frame_unchanged = all(
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
        for block in encoded_blocks
    )

    safe_summary = (
        eve_result.receiver_visible_summary()
    )

    hidden_eve_information_excluded = all(
        field_name not in safe_summary
        for field_name in (
            "eve_fraction",
            "eve_mode",
            "attacked_mask",
            "hidden_intercepted_positions",
            "eve_interception_events",
        )
    )

    print(
        "Logical blocks transmitted : "
        f"{ideal_result.logical_block_count}"
    )

    print(
        "Physical positions         : "
        f"{ideal_result.physical_position_count}"
    )

    print(
        "Ideal frame unchanged      : "
        f"{ideal_unchanged}"
    )

    print(
        "Noise events generated     : "
        f"{noise_events_generated}"
    )

    print(
        "Complete loss applied      : "
        f"{complete_loss_applied}"
    )

    print(
        "Full Eve intercepted all   : "
        f"{all_positions_intercepted}"
    )

    print(
        "Original frame unchanged   : "
        f"{original_frame_unchanged}"
    )

    print(
        "Safe summary hides Eve     : "
        f"{hidden_eve_information_excluded}"
    )

    if not ideal_unchanged:
        raise QuantumTransmissionError(
            "Ideal channel modified the received frame."
        )

    if not noise_events_generated:
        raise QuantumTransmissionError(
            "Configured channel did not generate expected noise."
        )

    if not complete_loss_applied:
        raise QuantumTransmissionError(
            "Complete-loss channel did not erase all positions."
        )

    if not all_positions_intercepted:
        raise QuantumTransmissionError(
            "Full-Eve channel did not intercept every position."
        )

    if not original_frame_unchanged:
        raise QuantumTransmissionError(
            "Transmission modified the original encoded frame."
        )

    if not hidden_eve_information_excluded:
        raise QuantumTransmissionError(
            "Receiver-visible summary exposed hidden Eve data."
        )

    print(
        "\nQuantum channel self-test "
        "completed successfully."
    )


__all__ = [
    "SUPPORTED_CONTEXTS",
    "SUPPORTED_EVE_MODES",
    "SUPPORTED_PAULI_ERRORS",
    "QuantumChannelError",
    "InvalidChannelConfigurationError",
    "InvalidQuantumFrameError",
    "QuantumTransmissionError",
    "ChannelConfig",
    "ChannelEventCounters",
    "QuantumChannelResult",
    "QuantumChannel",
    "validate_probability",
    "validate_channel_config",
    "normalize_frame",
    "validate_physical_frame",
    "apply_bit_flip",
    "apply_phase_flip",
    "apply_depolarizing_error",
    "apply_intercept_resend",
    "transmit_one_block",
    "simulate_quantum_channel",
    "transmit_blocks_through_channel",
    "count_hidden_intercepted_positions",
    "calculate_loss_rate",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        QuantumChannelError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[QUANTUM CHANNEL ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error