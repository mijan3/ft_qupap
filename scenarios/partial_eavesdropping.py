"""
FT-QuPAP Partial Eavesdropping Scenario
=======================================

Defines a controlled partial eavesdropping attack against the FT-QuPAP
v5.1 quantum authentication transmission.

In this scenario, Eve interacts with only a selected fraction of the
Steane-encoded payload and independent check blocks.

Compared with a full intercept-resend attack, the disturbance may be
smaller and harder to detect using a fixed QBER threshold alone.
Therefore, the FT-QuPAP decision process combines:

- Independent check-block QBER
- Check-basis mismatch evidence
- Syndrome activity
- Corrected and uncorrectable block counts
- Observed check-block availability
- Quantum-channel loss
- Payload reconstruction evidence
- KMAC tag verification
- Calibrated Gaussian Process attack probability

Expected result:

    REJECTED

Deterministic verification remains mandatory. The GP detector only
supplements the protocol evidence and never overrides a deterministic
security failure.
"""

from __future__ import annotations

from typing import Any, Final, Mapping

from config import ApplicationConfig, get_config

from .scenario_config import (
    ChannelProfile,
    EveAttackMode,
    EveProfile,
    ExpectedOutcome,
    NoiseModelName,
    RetryProfile,
    ScenarioCategory,
    ScenarioConfig,
    TamperingProfile,
    create_scenario_config,
)


SCENARIO_NAME: Final[str] = "partial_eavesdropping"

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Partial Eavesdropping"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "Eve interacts with only part of the encoded quantum "
    "transmission. The attack introduces lower disturbance than "
    "full eavesdropping, requiring combined deterministic evidence "
    "and calibrated GP attack detection."
)

DEFAULT_RANDOM_SEED: Final[int] = 9202

DEFAULT_INTERACTION_FRACTION: Final[float] = 0.30

DEFAULT_BASIS_ERROR_PROBABILITY: Final[float] = 0.50

DEFAULT_RESEND_ERROR_PROBABILITY: Final[float] = 0.015

EXPECTED_DECISION: Final[str] = "rejected"

EXPECTED_MINIMUM_ATTACK_PROBABILITY: Final[float] = 0.25


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    interaction_fraction: float = (
        DEFAULT_INTERACTION_FRACTION
    ),
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the partial-eavesdropping attack scenario.

    Args:
        context:
            Network environment such as urban, suburban, or rural.

        random_seed:
            Reproducible seed used by the quantum simulation.

        interaction_fraction:
            Fraction of transmitted quantum information selected by
            Eve. The value must be greater than zero and less than one.

        app_config:
            Optional FT-QuPAP application configuration.

    Returns:
        Validated partial-eavesdropping scenario configuration.
    """

    config = app_config or get_config()

    validate_partial_interaction_fraction(
        interaction_fraction
    )

    scenario = create_scenario_config(
        name=SCENARIO_NAME,
        display_name=SCENARIO_DISPLAY_NAME,
        description=SCENARIO_DESCRIPTION,
        category=ScenarioCategory.QUANTUM_ATTACK,
        expected_outcome=ExpectedOutcome.REJECTED,
        context=context,
        random_seed=random_seed,
        channel=ChannelProfile(
            noise_model=NoiseModelName.COMBINED.value,

            # Normal environmental noise remains active.
            bit_flip_probability=0.004,
            phase_flip_probability=0.004,
            depolarizing_probability=0.006,
            measurement_error_probability=0.002,

            # Loss remains below the protocol rejection limit.
            loss_rate=0.025,

            burst_error_probability=0.0,
            burst_length=0,

            forced_lost_check_blocks=0,
            forced_uncorrectable_blocks=0,
        ),
        eve=EveProfile(
            enabled=True,
            attack_mode=(
                EveAttackMode
                .PARTIAL_EAVESDROPPING
                .value
            ),
            interaction_fraction=interaction_fraction,
            measurement_basis_error_probability=(
                DEFAULT_BASIS_ERROR_PROBABILITY
            ),
            resend_error_probability=(
                DEFAULT_RESEND_ERROR_PROBABILITY
            ),
            target_check_blocks=True,
            target_payload_blocks=True,
        ),
        tampering=TamperingProfile(
            replay_authentication_request=False,
            reuse_nonce=False,
            stale_timestamp=False,
            forge_server_signature=False,
            modify_authentication_request=False,
            tamper_mlkem_ciphertext=False,
            forge_kmac_tag=False,
            modify_mobile_identifier=False,
            modify_session_identifier=False,
            modify_network_identifier=False,
            modify_protocol_version=False,
            modify_control_schedule=False,
        ),
        retry=RetryProfile(
            enabled=False,
            force_retry_on_attempts=(),
            noise_multiplier_after_retry=1.0,
            loss_multiplier_after_retry=1.0,
            change_random_seed_per_attempt=True,
        ),
        deterministic_verification_expected=True,
        gp_evaluation_expected=True,
        notes=(
            "The subscriber and classical credentials are valid.",
            "The attack affects only part of the quantum transmission.",
            "Eve does not know the encrypted control schedule.",
            "The observed QBER may be lower than a full attack.",
            "Syndrome and check-block evidence remain important.",
            "The calibrated GP detector supports low-disturbance "
            "attack detection.",
            "Clear attack evidence must not receive a retry.",
            "The expected final decision is rejection.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": (
                "partial_quantum_eavesdropping"
            ),
            "attack_enabled": True,
            "attacker": "Eve",
            "attack_mode": (
                EveAttackMode
                .PARTIAL_EAVESDROPPING
                .value
            ),
            "interaction_fraction": (
                interaction_fraction
            ),
            "measurement_basis_error_probability": (
                DEFAULT_BASIS_ERROR_PROBABILITY
            ),
            "resend_error_probability": (
                DEFAULT_RESEND_ERROR_PROBABILITY
            ),
            "expected_decision": EXPECTED_DECISION,
            "expected_attack_detected": True,
            "expected_minimum_attack_probability": (
                EXPECTED_MINIMUM_ATTACK_PROBABILITY
            ),
            "retry_expected": False,
            "classical_request_valid": True,
            "server_signature_valid": True,
            "mlkem_ciphertext_valid": True,
            "original_kmac_tag_valid": True,
            "fixed_qber_threshold": (
                config.quantum.fixed_qber_threshold
            ),
            "maximum_loss_rate": (
                config.quantum.maximum_loss_rate
            ),
            "minimum_observed_check_blocks": (
                config.quantum
                .minimum_observed_check_blocks
            ),
            "raw_calibrated_gp_threshold": (
                config.machine_learning
                .raw_calibrated_threshold
            ),
            "minimum_operational_gp_threshold": (
                config.machine_learning
                .minimum_operational_threshold
            ),
            "operational_gp_threshold": (
                config.machine_learning
                .operational_threshold
            ),
            "gp_retry_upper_probability": (
                config.machine_learning
                .retry_upper_probability
            ),
            "steane_code": "[[7,1,3]]",
            "logical_payload_blocks": (
                config.quantum.logical_payload_blocks
            ),
            "independent_check_blocks": (
                config.quantum
                .independent_check_blocks
            ),
            "total_logical_blocks": (
                config.quantum.total_logical_blocks
            ),
            "total_physical_qubits": (
                config.quantum.total_physical_qubits
            ),
        },
    )

    scenario.validate(config)

    return scenario


def create_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    interaction_fraction: float = (
        DEFAULT_INTERACTION_FRACTION
    ),
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """Alias for building the partial-eavesdropping scenario."""

    return build_scenario(
        context=context,
        random_seed=random_seed,
        interaction_fraction=interaction_fraction,
        app_config=app_config,
    )


def get_scenario_summary(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    interaction_fraction: float = (
        DEFAULT_INTERACTION_FRACTION
    ),
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return dashboard-friendly attack information."""

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        interaction_fraction=interaction_fraction,
        app_config=app_config,
    )

    summary = scenario.dashboard_summary()

    summary.update(
        {
            "attacker": "Eve",
            "attack_stage": "quantum_transmission",
            "attack_scope": "partial",
            "expected_decision": EXPECTED_DECISION,
            "expected_attack_detected": True,
            "retry_allowed": False,
            "classical_tampering": False,
            "cryptographic_forgery": False,
            "expected_detection_sources": [
                "independent_check_qber",
                "check_basis_mismatch",
                "syndrome_activity",
                "corrected_block_ratio",
                "uncorrectable_block_ratio",
                "payload_reconstruction_evidence",
                "kmac_tag_verification",
                "calibrated_gp_probability",
            ],
        }
    )

    return summary


def get_attempt_configuration(
    attempt_number: int = 1,
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    interaction_fraction: float = (
        DEFAULT_INTERACTION_FRACTION
    ),
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return effective attack conditions for one attempt."""

    config = app_config or get_config()

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        interaction_fraction=interaction_fraction,
        app_config=config,
    )

    attempt = scenario.for_attempt(
        attempt_number=attempt_number,
        app_config=config,
    )

    result = attempt.to_dictionary()

    result.update(
        {
            "expected_attempt_decision": (
                EXPECTED_DECISION
            ),
            "expected_attack_detected": True,
            "retry_allowed": False,
            "attack_scope": "partial",
        }
    )

    return result


def estimate_interacted_blocks(
    interaction_fraction: float = (
        DEFAULT_INTERACTION_FRACTION
    ),
    *,
    app_config: ApplicationConfig | None = None,
) -> dict[str, int]:
    """
    Estimate how many logical and physical blocks Eve interacts with.

    The simulator may perform randomized selection. These values are
    intended for dashboard presentation.
    """

    config = app_config or get_config()

    validate_partial_interaction_fraction(
        interaction_fraction
    )

    interacted_logical_blocks = round(
        config.quantum.total_logical_blocks
        * interaction_fraction
    )

    interacted_physical_qubits = round(
        config.quantum.total_physical_qubits
        * interaction_fraction
    )

    interacted_payload_blocks = round(
        config.quantum.logical_payload_blocks
        * interaction_fraction
    )

    interacted_check_blocks = round(
        config.quantum.independent_check_blocks
        * interaction_fraction
    )

    return {
        "interacted_logical_blocks": (
            interacted_logical_blocks
        ),
        "interacted_physical_qubits": (
            interacted_physical_qubits
        ),
        "interacted_payload_blocks": (
            interacted_payload_blocks
        ),
        "interacted_check_blocks": (
            interacted_check_blocks
        ),
    }


def validate_attack_result(
    result: Mapping[str, Any],
    *,
    app_config: ApplicationConfig | None = None,
) -> bool:
    """
    Validate whether a result represents successful detection.

    Partial attacks may produce a QBER below the fixed QBER threshold.
    Therefore, rejection may also be supported by GP probability,
    syndrome evidence, payload decoding failure, or KMAC mismatch.
    """

    if not isinstance(result, Mapping):
        raise TypeError(
            "result must be a mapping."
        )

    config = app_config or get_config()

    decision = normalize_text(
        result.get(
            "decision",
            result.get("status"),
        )
    )

    rejected = decision in {
        "rejected",
        "rejected_gp",
        "rejected_quantum",
        "rejected_qber",
        "rejected_deterministic",
        "rejected_tag",
        "failed",
    }

    qber = safe_float(
        result.get(
            "qber",
            result.get("check_qber"),
        )
    )

    attack_probability = safe_float(
        result.get(
            "attack_probability",
            result.get("gp_probability"),
        )
    )

    syndrome_rate = safe_float(
        result.get(
            "syndrome_rate",
            result.get("syndrome_activity"),
        )
    )

    corrected_blocks = safe_integer(
        result.get("corrected_blocks")
    )

    uncorrectable_blocks = safe_integer(
        result.get("uncorrectable_blocks")
    )

    tag_valid = result.get(
        "tag_valid",
        result.get("kmac_valid"),
    )

    payload_valid = result.get(
        "payload_valid",
        result.get("decoded_payload_valid"),
    )

    qber_alert = (
        qber is not None
        and qber
        > config.quantum.fixed_qber_threshold
    )

    gp_alert = (
        attack_probability is not None
        and attack_probability
        >= config.machine_learning
        .operational_threshold
    )

    syndrome_alert = (
        syndrome_rate is not None
        and syndrome_rate > 0.10
    )

    decoder_alert = (
        corrected_blocks > 0
        or uncorrectable_blocks > 0
    )

    tag_alert = tag_valid is False
    payload_alert = payload_valid is False

    evidence_detected = any(
        (
            qber_alert,
            gp_alert,
            syndrome_alert,
            decoder_alert,
            tag_alert,
            payload_alert,
        )
    )

    return rejected and evidence_detected


def validate_partial_interaction_fraction(
    interaction_fraction: float,
) -> float:
    """Validate a partial eavesdropping fraction."""

    if isinstance(interaction_fraction, bool):
        raise TypeError(
            "interaction_fraction must be numeric."
        )

    if not isinstance(
        interaction_fraction,
        (int, float),
    ):
        raise TypeError(
            "interaction_fraction must be numeric."
        )

    normalized_fraction = float(
        interaction_fraction
    )

    if normalized_fraction != normalized_fraction:
        raise ValueError(
            "interaction_fraction cannot be NaN."
        )

    if normalized_fraction in {
        float("inf"),
        float("-inf"),
    }:
        raise ValueError(
            "interaction_fraction must be finite."
        )

    if not 0.0 < normalized_fraction < 1.0:
        raise ValueError(
            "Partial eavesdropping requires an "
            "interaction_fraction greater than 0 and "
            "less than 1."
        )

    return normalized_fraction


def normalize_text(value: Any) -> str:
    """Normalize an optional value as lowercase text."""

    if value is None:
        return ""

    return str(value).strip().lower()


def safe_float(value: Any) -> float | None:
    """Convert a finite numeric value to float."""

    if value is None or isinstance(value, bool):
        return None

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    if converted != converted:
        return None

    if converted in {
        float("inf"),
        float("-inf"),
    }:
        return None

    return converted


def safe_integer(
    value: Any,
    default: int = 0,
) -> int:
    """Convert an integer-like value to int."""

    if value is None or isinstance(value, bool):
        return default

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default

    if not converted.is_integer():
        return default

    return int(converted)


PARTIAL_EAVESDROPPING_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run partial-eavesdropping consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert scenario.name == "partial_eavesdropping"

    assert (
        scenario.category
        == "quantum_attack"
    )

    assert scenario.expected_outcome == "rejected"
    assert scenario.attack_enabled is True
    assert scenario.retry_expected is False

    assert scenario.eve.enabled is True

    assert (
        scenario.eve.attack_mode
        == "partial_eavesdropping"
    )

    assert (
        scenario.eve.interaction_fraction
        == DEFAULT_INTERACTION_FRACTION
    )

    assert (
        0.0
        < scenario.eve.interaction_fraction
        < 1.0
    )

    assert scenario.tampering.enabled is False
    assert scenario.retry.enabled is False

    attempt = scenario.for_attempt(
        1,
        config,
    )

    assert attempt.eve.enabled is True

    assert (
        attempt.force_retry_gray_zone
        is False
    )

    estimates = estimate_interacted_blocks(
        app_config=config
    )

    assert (
        estimates["interacted_logical_blocks"]
        == round(
            config.quantum.total_logical_blocks
            * DEFAULT_INTERACTION_FRACTION
        )
    )

    assert (
        estimates["interacted_physical_qubits"]
        == round(
            config.quantum.total_physical_qubits
            * DEFAULT_INTERACTION_FRACTION
        )
    )

    valid_low_qber_result = validate_attack_result(
        {
            "decision": "rejected_gp",
            "check_qber": 0.09,
            "gp_probability": 0.82,
            "syndrome_rate": 0.18,
            "corrected_blocks": 6,
            "uncorrectable_blocks": 0,
            "tag_valid": True,
            "payload_valid": True,
        },
        app_config=config,
    )

    assert valid_low_qber_result is True

    valid_tag_failure_result = validate_attack_result(
        {
            "decision": "rejected_tag",
            "check_qber": 0.08,
            "gp_probability": 0.14,
            "syndrome_rate": 0.08,
            "corrected_blocks": 2,
            "uncorrectable_blocks": 1,
            "tag_valid": False,
            "payload_valid": False,
        },
        app_config=config,
    )

    assert valid_tag_failure_result is True

    invalid_result = validate_attack_result(
        {
            "decision": "accepted",
            "check_qber": 0.02,
            "gp_probability": 0.05,
            "syndrome_rate": 0.01,
            "corrected_blocks": 0,
            "uncorrectable_blocks": 0,
            "tag_valid": True,
            "payload_valid": True,
        },
        app_config=config,
    )

    assert invalid_result is False

    summary = get_scenario_summary(
        app_config=config
    )

    assert summary["attack_enabled"] is True
    assert summary["attack_scope"] == "partial"
    assert summary["attacker"] == "Eve"
    assert summary["retry_allowed"] is False

    print(
        "FT-QuPAP partial-eavesdropping "
        "scenario self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_INTERACTION_FRACTION",
    "DEFAULT_BASIS_ERROR_PROBABILITY",
    "DEFAULT_RESEND_ERROR_PROBABILITY",
    "EXPECTED_DECISION",
    "EXPECTED_MINIMUM_ATTACK_PROBABILITY",
    "PARTIAL_EAVESDROPPING_CONFIG",
    "build_scenario",
    "create_scenario",
    "get_scenario_summary",
    "get_attempt_configuration",
    "estimate_interacted_blocks",
    "validate_attack_result",
    "validate_partial_interaction_fraction",
]


if __name__ == "__main__":
    run_self_test()