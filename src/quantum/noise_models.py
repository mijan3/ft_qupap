"""
FT-QuPAP Quantum Noise Models

This module defines the standard syndrome-level and representative
Qiskit noise configurations used by the FT-QuPAP simulator.

Notebook-aligned named profiles:

1. IDEAL_CHANNEL
2. NOISY_CHANNEL
3. LOSSY_CHANNEL

Context assumptions:

    urban:
        noise scale = 1.20
        loss scale  = 0.90

    suburban:
        noise scale = 1.00
        loss scale  = 1.00

    rural:
        noise scale = 0.85
        loss scale  = 1.40

These context factors are simulation assumptions. They do not claim
to be measured 5G or 6G channel statistics.

Security boundary:

Eve attack parameters are not part of the normal noise model.
Intercept-measure-resend behavior is implemented separately in
eve_attack.py and quantum_channel.py.

The Authentication Server's GP feature extractor must not directly
read hidden channel probabilities. Receiver-observable evidence such
as raw QBER, syndrome weight, correction failure, and loss rate must
be used instead.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np

from .quantum_channel import (
    ChannelConfig,
    validate_channel_config,
)


# ============================================================
# Context assumptions from the FT-QuPAP notebook
# ============================================================

SUPPORTED_CONTEXTS = (
    "urban",
    "suburban",
    "rural",
)

CONTEXT_CHANNEL_PROFILE: dict[str, dict[str, float]] = {
    "urban": {
        "noise_scale": 1.20,
        "loss_scale": 0.90,
    },
    "suburban": {
        "noise_scale": 1.00,
        "loss_scale": 1.00,
    },
    "rural": {
        "noise_scale": 0.85,
        "loss_scale": 1.40,
    },
}


MAX_CONTEXT_NOISE_PROBABILITY = 0.05
MAX_ACCEPTABLE_LOSS_RATE = 0.15
MAX_QISKIT_TOTAL_ERROR_PROBABILITY = 0.999


# ============================================================
# Notebook-aligned named channel profiles
# ============================================================

IDEAL_CHANNEL = ChannelConfig(
    name="ideal",
    bit_flip_prob=0.0,
    phase_flip_prob=0.0,
    depolarizing_prob=0.0,
    loss_prob=0.0,
    eve_fraction=0.0,
    eve_mode="none",
    context="urban",
)


NOISY_CHANNEL = ChannelConfig(
    name="benign_noisy",
    bit_flip_prob=0.001,
    phase_flip_prob=0.001,
    depolarizing_prob=0.0005,
    loss_prob=0.001,
    eve_fraction=0.0,
    eve_mode="none",
    context="urban",
)


LOSSY_CHANNEL = ChannelConfig(
    name="lossy",
    bit_flip_prob=0.010,
    phase_flip_prob=0.010,
    depolarizing_prob=0.005,
    loss_prob=0.100,
    eve_fraction=0.0,
    eve_mode="none",
    context="rural",
)


STANDARD_NOISE_MODELS: dict[str, ChannelConfig] = {
    IDEAL_CHANNEL.name: IDEAL_CHANNEL,
    NOISY_CHANNEL.name: NOISY_CHANNEL,
    LOSSY_CHANNEL.name: LOSSY_CHANNEL,
}


class NoiseModelError(Exception):
    """Base exception for FT-QuPAP noise-model processing."""


class InvalidNoiseContextError(NoiseModelError):
    """Raised when an unsupported channel context is supplied."""


class InvalidBaseNoiseError(NoiseModelError):
    """Raised when a base-noise value is invalid."""


class QiskitNoiseModelUnavailableError(NoiseModelError):
    """
    Raised when the optional Qiskit Aer dependency is unavailable.
    """


def validate_context(
    context: Any,
) -> str:
    """
    Validate and normalize a channel context.
    """

    if not isinstance(
        context,
        str,
    ):
        raise InvalidNoiseContextError(
            "context must be a string."
        )

    normalized_context = (
        context.strip().lower()
    )

    if normalized_context not in (
        SUPPORTED_CONTEXTS
    ):
        raise InvalidNoiseContextError(
            "context must be urban, suburban, or rural."
        )

    return normalized_context


def validate_base_noise(
    base_noise: Any,
) -> float:
    """
    Validate a finite base-noise probability.

    The value must be in the closed interval [0, 1].
    Context adjustment later clips the effective probability to 0.05,
    following the notebook.
    """

    if isinstance(
        base_noise,
        bool,
    ):
        raise InvalidBaseNoiseError(
            "base_noise cannot be boolean."
        )

    if not isinstance(
        base_noise,
        (int, float),
    ):
        raise InvalidBaseNoiseError(
            "base_noise must be numeric."
        )

    normalized_noise = float(
        base_noise
    )

    if not np.isfinite(
        normalized_noise
    ):
        raise InvalidBaseNoiseError(
            "base_noise must be finite."
        )

    if not 0.0 <= normalized_noise <= 1.0:
        raise InvalidBaseNoiseError(
            "base_noise must be between 0 and 1."
        )

    return normalized_noise


def context_adjusted_channel_parameters(
    base_noise: float,
    context: str,
) -> dict[str, float]:
    """
    Apply the notebook's explicit context assumptions.

    Equations:

        effective_noise =
            clip(
                base_noise × noise_scale,
                0,
                0.05,
            )

        effective_loss =
            clip(
                (base_noise / 3) × loss_scale,
                0,
                0.15,
            )

    Returned parameters:

        bit_flip_prob
        phase_flip_prob
        depolarizing_prob
        loss_prob
    """

    normalized_noise = validate_base_noise(
        base_noise
    )

    normalized_context = validate_context(
        context
    )

    context_profile = (
        CONTEXT_CHANNEL_PROFILE[
            normalized_context
        ]
    )

    effective_noise = float(
        np.clip(
            normalized_noise
            * context_profile[
                "noise_scale"
            ],
            0.0,
            MAX_CONTEXT_NOISE_PROBABILITY,
        )
    )

    effective_loss = float(
        np.clip(
            (
                normalized_noise
                / 3.0
            )
            * context_profile[
                "loss_scale"
            ],
            0.0,
            MAX_ACCEPTABLE_LOSS_RATE,
        )
    )

    return {
        "bit_flip_prob":
            effective_noise,
        "phase_flip_prob":
            effective_noise,
        "depolarizing_prob":
            effective_noise / 2.0,
        "loss_prob":
            effective_loss,
    }


def create_context_adjusted_channel(
    base_noise: float,
    context: str,
    name: str | None = None,
) -> ChannelConfig:
    """
    Create a benign channel from a base-noise level and context.

    Eve settings are always disabled because this function builds a
    normal channel model rather than an attack scenario.
    """

    normalized_context = validate_context(
        context
    )

    parameters = (
        context_adjusted_channel_parameters(
            base_noise=base_noise,
            context=normalized_context,
        )
    )

    selected_name = (
        name
        if name is not None
        else (
            "context_adjusted_"
            f"{normalized_context}"
        )
    )

    if not isinstance(
        selected_name,
        str,
    ):
        raise TypeError(
            "name must be a string or None."
        )

    selected_name = (
        selected_name.strip()
    )

    if not selected_name:
        raise ValueError(
            "name cannot be empty."
        )

    channel = ChannelConfig(
        name=selected_name,
        bit_flip_prob=(
            parameters[
                "bit_flip_prob"
            ]
        ),
        phase_flip_prob=(
            parameters[
                "phase_flip_prob"
            ]
        ),
        depolarizing_prob=(
            parameters[
                "depolarizing_prob"
            ]
        ),
        loss_prob=(
            parameters[
                "loss_prob"
            ]
        ),
        eve_fraction=0.0,
        eve_mode="none",
        context=normalized_context,
    )

    validate_channel_config(
        channel
    )

    return channel


def clone_channel(
    channel: ChannelConfig,
    *,
    name: str | None = None,
    context: str | None = None,
    bit_flip_prob: float | None = None,
    phase_flip_prob: float | None = None,
    depolarizing_prob: float | None = None,
    loss_prob: float | None = None,
) -> ChannelConfig:
    """
    Create a modified copy of a benign noise profile.

    Eve parameters are preserved from the original channel. This
    helper should therefore not be used to silently convert a normal
    channel into an attack scenario.
    """

    if not isinstance(
        channel,
        ChannelConfig,
    ):
        raise TypeError(
            "channel must be a ChannelConfig."
        )

    selected_context = (
        validate_context(context)
        if context is not None
        else channel.context
    )

    selected_name = (
        name.strip()
        if name is not None
        else channel.name
    )

    if not selected_name:
        raise ValueError(
            "name cannot be empty."
        )

    updated_channel = replace(
        channel,
        name=selected_name,
        context=selected_context,
        bit_flip_prob=(
            channel.bit_flip_prob
            if bit_flip_prob is None
            else float(bit_flip_prob)
        ),
        phase_flip_prob=(
            channel.phase_flip_prob
            if phase_flip_prob is None
            else float(phase_flip_prob)
        ),
        depolarizing_prob=(
            channel.depolarizing_prob
            if depolarizing_prob is None
            else float(depolarizing_prob)
        ),
        loss_prob=(
            channel.loss_prob
            if loss_prob is None
            else float(loss_prob)
        ),
    )

    validate_channel_config(
        updated_channel
    )

    return updated_channel


def get_noise_model(
    model_name: str,
) -> ChannelConfig:
    """
    Return a named standard noise profile.

    Supported names:

        ideal
        benign_noisy
        lossy
    """

    if not isinstance(
        model_name,
        str,
    ):
        raise TypeError(
            "model_name must be a string."
        )

    normalized_name = (
        model_name.strip().lower()
    )

    aliases = {
        "ideal":
            "ideal",
        "clean":
            "ideal",
        "no_noise":
            "ideal",
        "benign_noisy":
            "benign_noisy",
        "noisy":
            "benign_noisy",
        "normal_noise":
            "benign_noisy",
        "lossy":
            "lossy",
        "high_loss":
            "lossy",
    }

    canonical_name = aliases.get(
        normalized_name
    )

    if canonical_name is None:
        raise NoiseModelError(
            "Unknown noise model. Supported models are "
            "ideal, benign_noisy, and lossy."
        )

    return STANDARD_NOISE_MODELS[
        canonical_name
    ]


def list_noise_models() -> list[str]:
    """
    Return standard noise-model names.
    """

    return list(
        STANDARD_NOISE_MODELS.keys()
    )


def channel_to_dictionary(
    channel: ChannelConfig,
    include_hidden_attack_settings: bool = False,
) -> dict[str, Any]:
    """
    Convert a channel configuration into a dictionary.

    Eve settings are omitted by default.
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

    channel_dictionary = asdict(
        channel
    )

    if not include_hidden_attack_settings:
        channel_dictionary.pop(
            "eve_fraction",
            None,
        )

        channel_dictionary.pop(
            "eve_mode",
            None,
        )

    return channel_dictionary


def qiskit_pauli_probabilities(
    channel: ChannelConfig,
) -> dict[str, float]:
    """
    Convert a ChannelConfig into Qiskit Pauli probabilities.

    Notebook-aligned equations:

        p_X =
            bit_flip_prob
            + depolarizing_prob / 3

        p_Y =
            depolarizing_prob / 3

        p_Z =
            phase_flip_prob
            + depolarizing_prob / 3

    When the total error exceeds 0.999, X, Y, and Z probabilities are
    scaled proportionally.

    Loss and Eve interception are not represented in this Pauli model.
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

    p_x = float(
        channel.bit_flip_prob
        + (
            channel.depolarizing_prob
            / 3.0
        )
    )

    p_y = float(
        channel.depolarizing_prob
        / 3.0
    )

    p_z = float(
        channel.phase_flip_prob
        + (
            channel.depolarizing_prob
            / 3.0
        )
    )

    total_error = (
        p_x
        + p_y
        + p_z
    )

    if total_error > (
        MAX_QISKIT_TOTAL_ERROR_PROBABILITY
    ):
        scale = (
            MAX_QISKIT_TOTAL_ERROR_PROBABILITY
            / total_error
        )

        p_x *= scale
        p_y *= scale
        p_z *= scale

        total_error = (
            p_x
            + p_y
            + p_z
        )

    p_identity = float(
        1.0
        - total_error
    )

    probabilities = {
        "I":
            p_identity,
        "X":
            float(p_x),
        "Y":
            float(p_y),
        "Z":
            float(p_z),
    }

    if not np.isclose(
        sum(
            probabilities.values()
        ),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise NoiseModelError(
            "Qiskit Pauli probabilities do not sum to one."
        )

    if any(
        probability < 0.0
        for probability in (
            probabilities.values()
        )
    ):
        raise NoiseModelError(
            "Qiskit Pauli probabilities cannot be negative."
        )

    return probabilities


def _load_qiskit_noise_dependencies() -> tuple[Any, Any]:
    """
    Import optional Qiskit Aer noise components.

    Importing lazily allows the syndrome-level simulator to run when
    Qiskit is not installed.
    """

    try:
        from qiskit_aer.noise import (
            NoiseModel,
            pauli_error,
        )

    except ImportError as error:
        raise QiskitNoiseModelUnavailableError(
            "Qiskit Aer is required for representative "
            "circuit noise-model validation. Install it with: "
            "python -m pip install qiskit qiskit-aer"
        ) from error

    return (
        NoiseModel,
        pauli_error,
    )


def build_qiskit_noise_model(
    channel: ChannelConfig,
) -> Any:
    """
    Build a Qiskit Aer Pauli noise model.

    The Pauli error is attached only to identity gates used as
    quantum-channel transmission markers.

    Excluded from this model:

    - physical loss and erasure
    - Eve attack fraction
    - Eve measurement basis
    - intercept-measure-resend operations

    Loss is handled by the syndrome-level loss model. Eve is handled
    by explicit circuit operations.
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

    NoiseModel, pauli_error = (
        _load_qiskit_noise_dependencies()
    )

    probabilities = (
        qiskit_pauli_probabilities(
            channel
        )
    )

    noise_model = NoiseModel()

    total_error = (
        probabilities["X"]
        + probabilities["Y"]
        + probabilities["Z"]
    )

    if total_error > 0.0:
        channel_error = pauli_error(
            [
                (
                    "I",
                    probabilities["I"],
                ),
                (
                    "X",
                    probabilities["X"],
                ),
                (
                    "Y",
                    probabilities["Y"],
                ),
                (
                    "Z",
                    probabilities["Z"],
                ),
            ]
        )

        noise_model.add_all_qubit_quantum_error(
            channel_error,
            ["id"],
        )

    return noise_model


def make_qiskit_backend(
    channel: ChannelConfig,
) -> Any:
    """
    Create the notebook's representative Aer stabilizer backend.

    The Steane validation circuit uses Clifford operations such as
    H, X, Z, and CX, allowing the stabilizer simulation method.
    """

    if not isinstance(
        channel,
        ChannelConfig,
    ):
        raise TypeError(
            "channel must be a ChannelConfig."
        )

    try:
        from qiskit_aer import (
            AerSimulator,
        )

    except ImportError as error:
        raise QiskitNoiseModelUnavailableError(
            "Qiskit Aer is required to create the "
            "representative stabilizer backend. Install it with: "
            "python -m pip install qiskit qiskit-aer"
        ) from error

    return AerSimulator(
        method="stabilizer",
        noise_model=(
            build_qiskit_noise_model(
                channel
            )
        ),
    )


def compare_contexts(
    base_noise: float,
) -> dict[str, dict[str, float]]:
    """
    Calculate context-adjusted parameters for every context.
    """

    normalized_noise = validate_base_noise(
        base_noise
    )

    return {
        context:
            context_adjusted_channel_parameters(
                base_noise=normalized_noise,
                context=context,
            )
        for context in (
            SUPPORTED_CONTEXTS
        )
    }


def run_self_test() -> None:
    """
    Verify named profiles, context scaling, and Pauli probabilities.
    """

    print("=" * 72)
    print("FT-QuPAP Noise Models Self-Test")
    print("=" * 72)

    profile_names = list_noise_models()

    ideal_is_clean = all(
        probability == 0.0
        for probability in (
            IDEAL_CHANNEL.bit_flip_prob,
            IDEAL_CHANNEL.phase_flip_prob,
            IDEAL_CHANNEL.depolarizing_prob,
            IDEAL_CHANNEL.loss_prob,
        )
    )

    noisy_matches_notebook = (
        NOISY_CHANNEL.bit_flip_prob
        == 0.001
        and NOISY_CHANNEL.phase_flip_prob
        == 0.001
        and NOISY_CHANNEL.depolarizing_prob
        == 0.0005
        and NOISY_CHANNEL.loss_prob
        == 0.001
        and NOISY_CHANNEL.context
        == "urban"
    )

    lossy_matches_notebook = (
        LOSSY_CHANNEL.bit_flip_prob
        == 0.010
        and LOSSY_CHANNEL.phase_flip_prob
        == 0.010
        and LOSSY_CHANNEL.depolarizing_prob
        == 0.005
        and LOSSY_CHANNEL.loss_prob
        == 0.100
        and LOSSY_CHANNEL.context
        == "rural"
    )

    context_comparison = (
        compare_contexts(
            base_noise=0.01
        )
    )

    urban_parameters = (
        context_comparison[
            "urban"
        ]
    )

    suburban_parameters = (
        context_comparison[
            "suburban"
        ]
    )

    rural_parameters = (
        context_comparison[
            "rural"
        ]
    )

    context_noise_order_correct = (
        urban_parameters[
            "bit_flip_prob"
        ]
        >
        suburban_parameters[
            "bit_flip_prob"
        ]
        >
        rural_parameters[
            "bit_flip_prob"
        ]
    )

    context_loss_order_correct = (
        rural_parameters[
            "loss_prob"
        ]
        >
        suburban_parameters[
            "loss_prob"
        ]
        >
        urban_parameters[
            "loss_prob"
        ]
    )

    pauli_probabilities = (
        qiskit_pauli_probabilities(
            NOISY_CHANNEL
        )
    )

    pauli_sum_valid = np.isclose(
        sum(
            pauli_probabilities.values()
        ),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

    safe_noisy_dictionary = (
        channel_to_dictionary(
            NOISY_CHANNEL,
            include_hidden_attack_settings=False,
        )
    )

    hidden_fields_excluded = all(
        field_name
        not in safe_noisy_dictionary
        for field_name in (
            "eve_fraction",
            "eve_mode",
        )
    )

    qiskit_status = "available"

    try:
        qiskit_noise_model = (
            build_qiskit_noise_model(
                NOISY_CHANNEL
            )
        )

        qiskit_model_created = (
            qiskit_noise_model
            is not None
        )

    except QiskitNoiseModelUnavailableError:
        qiskit_status = (
            "not installed; "
            "syndrome-level tests still valid"
        )

        qiskit_model_created = True

    print(
        "Named noise models        : "
        f"{profile_names}"
    )

    print(
        "Ideal profile is clean    : "
        f"{ideal_is_clean}"
    )

    print(
        "Noisy profile matches     : "
        f"{noisy_matches_notebook}"
    )

    print(
        "Lossy profile matches     : "
        f"{lossy_matches_notebook}"
    )

    print(
        "Context noise order valid : "
        f"{context_noise_order_correct}"
    )

    print(
        "Context loss order valid  : "
        f"{context_loss_order_correct}"
    )

    print(
        "Noisy Pauli probabilities : "
        f"{pauli_probabilities}"
    )

    print(
        "Pauli probabilities sum 1 : "
        f"{bool(pauli_sum_valid)}"
    )

    print(
        "Safe profile hides Eve    : "
        f"{hidden_fields_excluded}"
    )

    print(
        "Qiskit status             : "
        f"{qiskit_status}"
    )

    if not ideal_is_clean:
        raise NoiseModelError(
            "Ideal profile contains nonzero noise."
        )

    if not noisy_matches_notebook:
        raise NoiseModelError(
            "Benign-noisy profile differs from the notebook."
        )

    if not lossy_matches_notebook:
        raise NoiseModelError(
            "Lossy profile differs from the notebook."
        )

    if not context_noise_order_correct:
        raise NoiseModelError(
            "Context noise scaling is incorrect."
        )

    if not context_loss_order_correct:
        raise NoiseModelError(
            "Context loss scaling is incorrect."
        )

    if not pauli_sum_valid:
        raise NoiseModelError(
            "Pauli probabilities do not sum to one."
        )

    if not hidden_fields_excluded:
        raise NoiseModelError(
            "Safe channel dictionary exposed Eve settings."
        )

    if not qiskit_model_created:
        raise NoiseModelError(
            "Qiskit noise model was not created."
        )

    print(
        "\nNoise models self-test "
        "completed successfully."
    )


__all__ = [
    "SUPPORTED_CONTEXTS",
    "CONTEXT_CHANNEL_PROFILE",
    "MAX_CONTEXT_NOISE_PROBABILITY",
    "MAX_ACCEPTABLE_LOSS_RATE",
    "IDEAL_CHANNEL",
    "NOISY_CHANNEL",
    "LOSSY_CHANNEL",
    "STANDARD_NOISE_MODELS",
    "NoiseModelError",
    "InvalidNoiseContextError",
    "InvalidBaseNoiseError",
    "QiskitNoiseModelUnavailableError",
    "validate_context",
    "validate_base_noise",
    "context_adjusted_channel_parameters",
    "create_context_adjusted_channel",
    "clone_channel",
    "get_noise_model",
    "list_noise_models",
    "channel_to_dictionary",
    "qiskit_pauli_probabilities",
    "build_qiskit_noise_model",
    "make_qiskit_backend",
    "compare_contexts",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        NoiseModelError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[NOISE MODEL ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error