"""
Quantum Transmitter Module
FT-QuPAP Mobile Station

This module simulates transmission of Steane-encoded FT-QuPAP
physical blocks through a noisy and untrusted quantum channel.

Supported channel effects:

1. Independent physical bit-flip errors
2. Independent physical phase-flip errors
3. Depolarizing Pauli errors
4. Physical-qubit loss and erasure
5. Eve intercept-measure-resend disturbance

Research boundary:
    Complete FT-QuPAP sessions use this scalable syndrome-level
    channel model. It does not represent physical quantum hardware.

Security boundary:
    eve_fraction and attacked_mask are simulator-only ground-truth
    values. They must never be supplied to the Authentication
    Server's observable GP feature vector.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

try:
    from .steane_encoder import (
        PhysicalBlock,
        SteaneEncodedFrame,
        STEANE_BLOCK_SIZE,
        TOTAL_LOGICAL_BLOCK_COUNT,
        TOTAL_PHYSICAL_QUBIT_COUNT,
        encode_ft_qupap_frame,
    )
except ImportError:
    from steane_encoder import (
        PhysicalBlock,
        SteaneEncodedFrame,
        STEANE_BLOCK_SIZE,
        TOTAL_LOGICAL_BLOCK_COUNT,
        TOTAL_PHYSICAL_QUBIT_COUNT,
        encode_ft_qupap_frame,
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


class QuantumTransmissionError(Exception):
    """Base exception for quantum-channel transmission failures."""


class ChannelConfigurationError(
    QuantumTransmissionError
):
    """Raised when a channel profile is invalid."""


class QuantumFrameValidationError(
    QuantumTransmissionError
):
    """Raised when a physical transmission frame is invalid."""


@dataclass(frozen=True)
class ChannelConfig:
    """
    Syndrome-level quantum-channel configuration.

    Attributes:
        name:
            Human-readable channel-profile name.

        bit_flip_prob:
            Independent physical X-error probability.

        phase_flip_prob:
            Independent physical Z-error probability.

        depolarizing_prob:
            Probability of a random X, Y, or Z Pauli error.

        loss_prob:
            Physical-qubit loss/erasure probability.

        eve_fraction:
            Hidden simulator probability that Eve intercepts a
            physical position.

        eve_mode:
            Either "none" or "intercept_resend".

        context:
            Trusted simulation context: urban, suburban, or rural.
    """

    name: str = "ideal"

    bit_flip_prob: float = 0.0
    phase_flip_prob: float = 0.0
    depolarizing_prob: float = 0.0
    loss_prob: float = 0.0

    # Simulator-only hidden attack configuration.
    eve_fraction: float = 0.0
    eve_mode: str = "none"

    context: str = "urban"

    def __post_init__(self) -> None:
        validate_channel_config(self)

    @property
    def contains_attack(self) -> bool:
        """
        Return simulator ground truth.

        Do not use this property as an Authentication Server feature.
        """

        return (
            self.eve_mode == "intercept_resend"
            and self.eve_fraction > 0.0
        )

    def to_dictionary(
        self,
        include_hidden_attack_settings: bool = False,
    ) -> dict[str, Any]:
        """
        Convert the channel configuration into a dictionary.

        Hidden Eve settings are excluded by default.
        """

        result = {
            "name": self.name,
            "bit_flip_prob":
                self.bit_flip_prob,
            "phase_flip_prob":
                self.phase_flip_prob,
            "depolarizing_prob":
                self.depolarizing_prob,
            "loss_prob":
                self.loss_prob,
            "context":
                self.context,
        }

        if include_hidden_attack_settings:
            result.update(
                {
                    "eve_fraction":
                        self.eve_fraction,
                    "eve_mode":
                        self.eve_mode,
                }
            )

        return result


# ============================================================
# Notebook-aligned channel profiles
# ============================================================

IDEAL_CHANNEL = ChannelConfig(
    name="ideal",
    context="urban",
)

NOISY_CHANNEL = ChannelConfig(
    name="benign_noisy",
    bit_flip_prob=0.001,
    phase_flip_prob=0.001,
    depolarizing_prob=0.0005,
    loss_prob=0.001,
    context="urban",
)

LOSSY_CHANNEL = ChannelConfig(
    name="lossy",
    bit_flip_prob=0.010,
    phase_flip_prob=0.010,
    depolarizing_prob=0.005,
    loss_prob=0.100,
    context="rural",
)

PARTIAL_EVE_CHANNEL = ChannelConfig(
    name="partial_eve",
    bit_flip_prob=0.010,
    phase_flip_prob=0.010,
    depolarizing_prob=0.005,
    loss_prob=0.005,
    eve_fraction=0.35,
    eve_mode="intercept_resend",
    context="urban",
)

FULL_EVE_CHANNEL = ChannelConfig(
    name="full_eve",
    eve_fraction=1.0,
    eve_mode="intercept_resend",
    context="urban",
)


@dataclass
class ChannelEventCounters:
    """
    Internal simulator event counters.

    These values describe generated channel events. The hidden Eve
    count must not be used as a receiver-observable feature.
    """

    physical_positions: int = 0
    loss_events: int = 0
    bit_flip_events: int = 0
    phase_flip_events: int = 0
    depolarizing_events: int = 0

    # Hidden simulator-only value.
    eve_interceptions: int = 0

    def public_dictionary(self) -> dict[str, int]:
        """
        Return counters without hidden Eve information.
        """

        return {
            "physical_positions":
                self.physical_positions,
            "loss_events":
                self.loss_events,
            "bit_flip_events":
                self.bit_flip_events,
            "phase_flip_events":
                self.phase_flip_events,
            "depolarizing_events":
                self.depolarizing_events,
        }


@dataclass
class QuantumTransmissionResult:
    """
    Result of one syndrome-level quantum-channel transmission.

    Attributes:
        sent_blocks:
            Original encoded physical blocks.

        received_blocks:
            Deep-copied blocks after channel effects.

        channel:
            Channel profile used by the simulator.

        transmission_time_ms:
            Runtime measured by the simulator.

        event_counters:
            Internal generated-event counters.
    """

    sent_blocks: list[PhysicalBlock]
    received_blocks: list[PhysicalBlock]
    channel: ChannelConfig
    transmission_time_ms: float
    event_counters: ChannelEventCounters = field(
        repr=False
    )

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
            raise QuantumFrameValidationError(
                "Sent and received frame lengths differ."
            )

        if not isinstance(
            self.channel,
            ChannelConfig,
        ):
            raise TypeError(
                "channel must be a ChannelConfig."
            )

        if not isinstance(
            self.transmission_time_ms,
            (int, float),
        ):
            raise TypeError(
                "transmission_time_ms must be numeric."
            )

        if self.transmission_time_ms < 0:
            raise ValueError(
                "transmission_time_ms cannot be negative."
            )

        if not isinstance(
            self.event_counters,
            ChannelEventCounters,
        ):
            raise TypeError(
                "event_counters must be "
                "ChannelEventCounters."
            )

    @property
    def sent_frame(self) -> list[PhysicalBlock]:
        """Backward-compatible sent-frame alias."""

        return self.sent_blocks

    @property
    def received_frame(self) -> list[PhysicalBlock]:
        """Backward-compatible received-frame alias."""

        return self.received_blocks

    @property
    def frame(self) -> list[PhysicalBlock]:
        """Return the received physical frame."""

        return self.received_blocks

    @property
    def transmitted_logical_blocks(self) -> int:
        """Return the transmitted logical-block count."""

        return len(self.sent_blocks)

    @property
    def transmitted_physical_qubits(self) -> int:
        """Return the number of physical transmission positions."""

        return sum(
            len(block.reference_bits)
            for block in self.sent_blocks
        )

    @property
    def lost_physical_qubits(self) -> int:
        """Return receiver-observable lost physical positions."""

        return sum(
            int(np.count_nonzero(block.erasures))
            for block in self.received_blocks
        )

    @property
    def received_physical_qubits(self) -> int:
        """Return physical positions not marked as erased."""

        return (
            self.transmitted_physical_qubits
            - self.lost_physical_qubits
        )

    @property
    def loss_rate(self) -> float:
        """Return receiver-observable physical loss rate."""

        total = self.transmitted_physical_qubits

        if total == 0:
            return 0.0

        return float(
            self.lost_physical_qubits
            / total
        )

    @property
    def final_x_error_indicators(self) -> int:
        """
        Return final X-error indicators after XOR cancellation.

        This is simulator state, not direct Authentication Server
        channel knowledge.
        """

        return sum(
            int(np.count_nonzero(block.x_errors))
            for block in self.received_blocks
        )

    @property
    def final_z_error_indicators(self) -> int:
        """
        Return final Z-error indicators after XOR cancellation.
        """

        return sum(
            int(np.count_nonzero(block.z_errors))
            for block in self.received_blocks
        )

    @property
    def _hidden_intercepted_positions(self) -> int:
        """
        Return hidden simulator attack count.

        This property must not be used by the GP detector.
        """

        return sum(
            int(
                np.count_nonzero(
                    block.attacked_mask
                )
            )
            for block in self.received_blocks
        )

    def receiver_visible_summary(
        self,
    ) -> dict[str, Any]:
        """
        Return only receiver-safe transmission metadata.

        Hidden Eve configuration, attacked masks, and interception
        counts are intentionally excluded.
        """

        return {
            "channel_name":
                self.channel.name,
            "service_context":
                self.channel.context,
            "logical_blocks":
                self.transmitted_logical_blocks,
            "physical_positions":
                self.transmitted_physical_qubits,
            "received_physical_positions":
                self.received_physical_qubits,
            "lost_physical_positions":
                self.lost_physical_qubits,
            "loss_rate":
                self.loss_rate,
            "transmission_time_ms":
                self.transmission_time_ms,
        }

    def simulator_summary(
        self,
        include_hidden_attack_truth: bool = False,
    ) -> dict[str, Any]:
        """
        Return simulator diagnostics.

        Hidden Eve information is included only when explicitly
        requested for offline evaluation.
        """

        summary = {
            **self.receiver_visible_summary(),
            "final_x_error_indicators":
                self.final_x_error_indicators,
            "final_z_error_indicators":
                self.final_z_error_indicators,
            "generated_channel_events":
                self.event_counters.public_dictionary(),
        }

        if include_hidden_attack_truth:
            summary[
                "hidden_eve_mode"
            ] = self.channel.eve_mode

            summary[
                "hidden_eve_fraction"
            ] = self.channel.eve_fraction

            summary[
                "hidden_intercepted_positions"
            ] = self._hidden_intercepted_positions

        return summary


def validate_probability(
    value: float,
    field_name: str,
) -> None:
    """Validate a channel probability in [0, 1]."""

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    if not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    if not np.isfinite(value):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )


def validate_channel_config(
    channel: ChannelConfig,
) -> None:
    """Validate a complete channel configuration."""

    if not isinstance(channel.name, str):
        raise TypeError(
            "channel.name must be a string."
        )

    if not channel.name.strip():
        raise ValueError(
            "channel.name cannot be empty."
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

    if channel.eve_mode not in (
        SUPPORTED_EVE_MODES
    ):
        raise ChannelConfigurationError(
            "eve_mode must be 'none' or "
            "'intercept_resend'."
        )

    if channel.context not in (
        SUPPORTED_CONTEXTS
    ):
        raise ChannelConfigurationError(
            "context must be urban, suburban, or rural."
        )

    if (
        channel.eve_mode == "none"
        and channel.eve_fraction != 0.0
    ):
        raise ChannelConfigurationError(
            "eve_fraction must be zero when "
            "eve_mode='none'."
        )

    if (
        channel.eve_mode == "intercept_resend"
        and channel.eve_fraction <= 0.0
    ):
        raise ChannelConfigurationError(
            "intercept_resend requires a positive "
            "eve_fraction."
        )


def validate_physical_frame(
    frame: Sequence[PhysicalBlock],
) -> None:
    """Validate an encoded physical-block frame."""

    if isinstance(
        frame,
        (str, bytes, bytearray),
    ):
        raise TypeError(
            "frame must be a sequence of PhysicalBlock objects."
        )

    if not isinstance(frame, Sequence):
        raise TypeError(
            "frame must be a sequence."
        )

    if len(frame) == 0:
        raise QuantumFrameValidationError(
            "Physical frame cannot be empty."
        )

    seen_ids: set[str] = set()

    for expected_position, block in enumerate(
        frame
    ):
        if not isinstance(block, PhysicalBlock):
            raise TypeError(
                "Every frame item must be a PhysicalBlock."
            )

        if block.spec.block_id in seen_ids:
            raise QuantumFrameValidationError(
                "Physical frame contains duplicate "
                f"block ID {block.spec.block_id!r}."
            )

        seen_ids.add(
            block.spec.block_id
        )

        if block.spec.position is not None:
            if (
                block.spec.position
                != expected_position
            ):
                raise QuantumFrameValidationError(
                    f"Block {block.spec.block_id!r} is "
                    "outside its declared frame position."
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
            raise QuantumFrameValidationError(
                f"Block {block.spec.block_id!r} "
                "contains inconsistent physical arrays."
            )


def extract_physical_blocks(
    frame: SteaneEncodedFrame
    | Sequence[PhysicalBlock],
) -> list[PhysicalBlock]:
    """
    Extract a physical frame from either supported input form.
    """

    if isinstance(
        frame,
        SteaneEncodedFrame,
    ):
        blocks = frame.frame

    else:
        blocks = list(frame)

    validate_physical_frame(blocks)

    return blocks


def apply_depolarizing_error(
    block: PhysicalBlock,
    qubit_index: int,
    rng: np.random.Generator,
) -> str:
    """
    Apply a random X, Y, or Z Pauli error.

    Returns:
        Selected Pauli symbol.
    """

    pauli = str(
        rng.choice(
            ["X", "Y", "Z"]
        )
    )

    if pauli in ("X", "Y"):
        block.x_errors[
            qubit_index
        ] ^= 1

    if pauli in ("Y", "Z"):
        block.z_errors[
            qubit_index
        ] ^= 1

    return pauli


def apply_intercept_resend(
    block: PhysicalBlock,
    qubit_index: int,
    rng: np.random.Generator,
) -> None:
    """
    Apply the notebook's syndrome-level Eve model.

    Eve randomly selects the Z or X basis. When Eve's basis differs
    from the block preparation basis, disturbance may be introduced.

    attacked_mask is simulator-only hidden ground truth.
    """

    block.attacked_mask[
        qubit_index
    ] = True

    eve_basis = str(
        rng.choice(
            ["Z", "X"]
        )
    )

    if eve_basis == block.spec.basis:
        return

    # Main disturbance caused by wrong-basis measurement.
    if rng.random() < 0.5:
        if block.spec.basis == "Z":
            block.x_errors[
                qubit_index
            ] ^= 1
        else:
            block.z_errors[
                qubit_index
            ] ^= 1

    # Secondary component used by the notebook's scalable model.
    if rng.random() < 0.2:
        if block.spec.basis == "Z":
            block.z_errors[
                qubit_index
            ] ^= 1
        else:
            block.x_errors[
                qubit_index
            ] ^= 1


def transmit_blocks_through_channel(
    frame: SteaneEncodedFrame
    | Sequence[PhysicalBlock],
    channel: ChannelConfig,
    rng: np.random.Generator | None = None,
) -> list[PhysicalBlock]:
    """
    Notebook-compatible channel-transmission function.

    The original frame is never modified. A deep-copied received
    frame is returned.
    """

    result = transmit_quantum_frame(
        frame=frame,
        channel=channel,
        rng=rng,
    )

    return result.received_blocks


def transmit_quantum_frame(
    frame: SteaneEncodedFrame
    | Sequence[PhysicalBlock],
    channel: ChannelConfig,
    rng: np.random.Generator | None = None,
) -> QuantumTransmissionResult:
    """
    Transmit all encoded physical blocks through the channel.

    Channel operation order for each physical position:

        1. Loss/erasure
        2. Independent X error
        3. Independent Z error
        4. Depolarizing Pauli error
        5. Optional Eve intercept-measure-resend

    Lost positions do not receive later simulated operations.
    """

    if not isinstance(channel, ChannelConfig):
        raise TypeError(
            "channel must be a ChannelConfig."
        )

    validate_channel_config(channel)

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    sent_blocks = extract_physical_blocks(
        frame
    )

    received_blocks = copy.deepcopy(
        sent_blocks
    )

    counters = ChannelEventCounters()

    started_at = time.perf_counter()

    for block in received_blocks:
        physical_count = len(
            block.reference_bits
        )

        for qubit_index in range(
            physical_count
        ):
            counters.physical_positions += 1

            # -------------------------------------------------
            # Physical loss / erasure
            # -------------------------------------------------
            if (
                rng.random()
                < channel.loss_prob
            ):
                block.erasures[
                    qubit_index
                ] = True

                counters.loss_events += 1

                # No further operation occurs on a lost position.
                continue

            # -------------------------------------------------
            # Independent physical bit flip
            # -------------------------------------------------
            if (
                rng.random()
                < channel.bit_flip_prob
            ):
                block.x_errors[
                    qubit_index
                ] ^= 1

                counters.bit_flip_events += 1

            # -------------------------------------------------
            # Independent physical phase flip
            # -------------------------------------------------
            if (
                rng.random()
                < channel.phase_flip_prob
            ):
                block.z_errors[
                    qubit_index
                ] ^= 1

                counters.phase_flip_events += 1

            # -------------------------------------------------
            # Depolarizing Pauli disturbance
            # -------------------------------------------------
            if (
                rng.random()
                < channel.depolarizing_prob
            ):
                apply_depolarizing_error(
                    block=block,
                    qubit_index=qubit_index,
                    rng=rng,
                )

                counters.depolarizing_events += 1

            # -------------------------------------------------
            # Hidden simulator Eve model
            # -------------------------------------------------
            if (
                channel.eve_mode
                == "intercept_resend"
                and rng.random()
                < channel.eve_fraction
            ):
                apply_intercept_resend(
                    block=block,
                    qubit_index=qubit_index,
                    rng=rng,
                )

                counters.eve_interceptions += 1

    elapsed_ms = (
        time.perf_counter()
        - started_at
    ) * 1000.0

    validate_physical_frame(
        received_blocks
    )

    return QuantumTransmissionResult(
        sent_blocks=copy.deepcopy(
            sent_blocks
        ),
        received_blocks=
            received_blocks,
        channel=channel,
        transmission_time_ms=
            elapsed_ms,
        event_counters=counters,
    )


class QuantumTransmitter:
    """
    Stateful FT-QuPAP quantum transmitter.

    Compatible usage:

        transmitter = QuantumTransmitter(
            channel=NOISY_CHANNEL,
            rng=rng,
        )

        result = transmitter.transmit(
            encoded_frame
        )

    The channel can also be provided to transmit().
    """

    def __init__(
        self,
        channel: ChannelConfig
        | np.random.Generator
        | None = None,
        rng: np.random.Generator
        | None = None,
    ) -> None:
        # Supports QuantumTransmitter(rng) for compatibility.
        if (
            isinstance(
                channel,
                np.random.Generator,
            )
            and rng is None
        ):
            rng = channel
            channel = None

        if channel is not None:
            if not isinstance(
                channel,
                ChannelConfig,
            ):
                raise TypeError(
                    "channel must be a ChannelConfig or None."
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

        self.channel = channel
        self.rng = rng

    def transmit(
        self,
        frame: SteaneEncodedFrame
        | Sequence[PhysicalBlock],
        channel: ChannelConfig
        | None = None,
    ) -> QuantumTransmissionResult:
        """
        Transmit one complete encoded frame.
        """

        selected_channel = (
            channel
            if channel is not None
            else self.channel
        )

        if selected_channel is None:
            raise ChannelConfigurationError(
                "No quantum-channel configuration was supplied."
            )

        return transmit_quantum_frame(
            frame=frame,
            channel=selected_channel,
            rng=self.rng,
        )


def count_hidden_interceptions(
    received_blocks: Sequence[PhysicalBlock],
) -> int:
    """
    Return hidden Eve ground-truth count for offline evaluation.

    Never pass this value to the Authentication Server or GP model.
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


def run_self_test() -> None:
    """
    Test ideal, noisy, lossy, and full-Eve channel behavior.
    """

    print("=" * 72)
    print("FT-QuPAP Quantum Transmitter Self-Test")
    print("=" * 72)

    try:
        from .control_schedule import LogicalSpec
    except ImportError:
        from control_schedule import LogicalSpec

    logical_rng = np.random.default_rng(
        20260701
    )

    payload_specs = [
        LogicalSpec(
            block_id=f"P{index:04d}",
            role="payload",
            logical_index=index,
            logical_bit=int(
                logical_rng.integers(0, 2)
            ),
            basis="Z",
        )
        for index in range(128)
    ]

    check_specs = [
        LogicalSpec(
            block_id=f"C{index:04d}",
            role="check",
            logical_index=index,
            logical_bit=int(
                logical_rng.integers(0, 2)
            ),
            basis=str(
                logical_rng.choice(
                    ["Z", "X"]
                )
            ),
        )
        for index in range(32)
    ]

    combined_specs = (
        payload_specs
        + check_specs
    )

    permutation = logical_rng.permutation(
        len(combined_specs)
    )

    ordered_specs = [
        copy.deepcopy(
            combined_specs[int(index)]
        )
        for index in permutation
    ]

    for position, spec in enumerate(
        ordered_specs
    ):
        spec.position = position

    encoded_frame = encode_ft_qupap_frame(
        ordered_specs=ordered_specs,
        rng=np.random.default_rng(
            9102
        ),
        use_css=True,
    )

    ideal_result = transmit_quantum_frame(
        frame=encoded_frame,
        channel=IDEAL_CHANNEL,
        rng=np.random.default_rng(
            1001
        ),
    )

    ideal_unchanged = all(
        np.array_equal(
            sent.reference_bits,
            received.reference_bits,
        )
        and not np.any(
            received.x_errors
        )
        and not np.any(
            received.z_errors
        )
        and not np.any(
            received.erasures
        )
        and not np.any(
            received.attacked_mask
        )
        for sent, received in zip(
            ideal_result.sent_blocks,
            ideal_result.received_blocks,
            strict=True,
        )
    )

    deterministic_test_channel = ChannelConfig(
        name="deterministic_test_noise",
        bit_flip_prob=0.20,
        phase_flip_prob=0.15,
        depolarizing_prob=0.10,
        loss_prob=0.05,
        context="suburban",
    )

    noisy_result = transmit_quantum_frame(
        frame=encoded_frame,
        channel=
            deterministic_test_channel,
        rng=np.random.default_rng(
            1002
        ),
    )

    lossy_result = QuantumTransmitter(
        channel=LOSSY_CHANNEL,
        rng=np.random.default_rng(
            1003
        ),
    ).transmit(
        encoded_frame
    )

    full_eve_result = (
        QuantumTransmitter(
            channel=FULL_EVE_CHANNEL,
            rng=np.random.default_rng(
                1004
            ),
        ).transmit(
            encoded_frame
        )
    )

    noisy_events_present = (
        noisy_result.event_counters.bit_flip_events > 0
        and noisy_result.event_counters.phase_flip_events > 0
        and noisy_result.event_counters.depolarizing_events > 0
    )

    lossy_events_present = (
        lossy_result.lost_physical_qubits > 0
    )

    full_eve_attacked_all = (
        count_hidden_interceptions(
            full_eve_result.received_blocks
        )
        == TOTAL_PHYSICAL_QUBIT_COUNT
    )

    receiver_summary_hides_eve = all(
        key not in (
            full_eve_result
            .receiver_visible_summary()
        )
        for key in (
            "eve_fraction",
            "eve_mode",
            "attacked_mask",
            "hidden_intercepted_positions",
        )
    )

    original_frame_unchanged = all(
        not np.any(block.x_errors)
        and not np.any(block.z_errors)
        and not np.any(block.erasures)
        and not np.any(block.attacked_mask)
        for block in encoded_frame.frame
    )

    print(
        f"Logical blocks transmitted : "
        f"{ideal_result.transmitted_logical_blocks}"
    )
    print(
        f"Physical qubits transmitted: "
        f"{ideal_result.transmitted_physical_qubits}"
    )
    print(
        f"Ideal frame unchanged      : "
        f"{ideal_unchanged}"
    )
    print(
        f"Noisy events generated     : "
        f"{noisy_events_present}"
    )
    print(
        f"Loss events generated      : "
        f"{lossy_events_present}"
    )
    print(
        f"Full Eve intercepted all   : "
        f"{full_eve_attacked_all}"
    )
    print(
        f"Receiver summary hides Eve : "
        f"{receiver_summary_hides_eve}"
    )
    print(
        f"Original frame unchanged   : "
        f"{original_frame_unchanged}"
    )
    print(
        f"Lossy-channel loss rate    : "
        f"{lossy_result.loss_rate:.6f}"
    )

    if (
        ideal_result.transmitted_logical_blocks
        != TOTAL_LOGICAL_BLOCK_COUNT
    ):
        raise QuantumTransmissionError(
            "Incorrect logical transmission count."
        )

    if (
        ideal_result.transmitted_physical_qubits
        != TOTAL_PHYSICAL_QUBIT_COUNT
    ):
        raise QuantumTransmissionError(
            "Incorrect physical transmission count."
        )

    if not ideal_unchanged:
        raise QuantumTransmissionError(
            "Ideal channel modified the frame."
        )

    if not noisy_events_present:
        raise QuantumTransmissionError(
            "Configured noisy channel generated no errors."
        )

    if not lossy_events_present:
        raise QuantumTransmissionError(
            "Configured lossy channel generated no erasures."
        )

    if not full_eve_attacked_all:
        raise QuantumTransmissionError(
            "Full-Eve channel failed to intercept "
            "every available physical position."
        )

    if not receiver_summary_hides_eve:
        raise QuantumTransmissionError(
            "Receiver-visible summary exposed hidden Eve data."
        )

    if not original_frame_unchanged:
        raise QuantumTransmissionError(
            "Quantum transmission modified the original frame."
        )

    print("\nReceiver-visible ideal summary:")

    print(
        ideal_result
        .receiver_visible_summary()
    )

    print(
        "\nQuantum transmitter self-test "
        "completed successfully."
    )


__all__ = [
    "ChannelConfig",
    "ChannelEventCounters",
    "QuantumTransmissionResult",
    "QuantumTransmitter",
    "QuantumTransmissionError",
    "ChannelConfigurationError",
    "QuantumFrameValidationError",
    "IDEAL_CHANNEL",
    "NOISY_CHANNEL",
    "LOSSY_CHANNEL",
    "PARTIAL_EVE_CHANNEL",
    "FULL_EVE_CHANNEL",
    "apply_depolarizing_error",
    "apply_intercept_resend",
    "transmit_blocks_through_channel",
    "transmit_quantum_frame",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        QuantumTransmissionError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[QUANTUM TRANSMISSION ERROR] "
            f"{error}"
        )