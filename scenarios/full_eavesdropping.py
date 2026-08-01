"""
FT-QuPAP Full Eavesdropping Scenario
====================================

Defines a controlled full quantum eavesdropping attack against the
FT-QuPAP v5.1 authentication protocol.

In this scenario, Eve interacts with the complete Steane-encoded
quantum transmission, including:

- 128 logical authentication-payload blocks
- 32 independent check blocks
- 160 total logical blocks
- 1120 Steane-encoded physical qubits

Eve does not know the secret encrypted control schedule or the correct
preparation and measurement bases. Her interaction therefore introduces
significant disturbance into both payload and check blocks.

Expected evidence:

- High independent check-block QBER
- Significant syndrome activity
- Increased corrected block count
- Possible uncorrectable logical blocks
- Payload reconstruction errors
- Possible KMAC tag mismatch
- High calibrated Gaussian Process attack probability

Expected result:

    REJECTED

This attack does not modify the classical authentication request,
timestamp, nonce, ML-DSA signature, or ML-KEM ciphertext.
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


SCENARIO_NAME: Final[str] = "full_eavesdropping"

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Full Eavesdropping"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "Eve interacts with the complete encoded quantum transmission. "
    "Because Eve does not know the secret control schedule or correct "
    "measurement bases, the attack is expected to create strong QBER, "
    "syndrome, payload-decoding, and GP attack evidence."
)

DEFAULT_RANDOM_SEED: Final[int] = 9203

FULL_INTERACTION_FRACTION: Final[float] = 1.0

DEFAULT_BASIS_ERROR_PROBABILITY: Final[float] = 0.50

DEFAULT_RESEND_ERROR_PROBABILITY: Final[float] = 0.04

EXPECTED_DECISION: Final[str] = "rejected"

EXPECTED_MINIMUM_ATTACK_PROBABILITY: Final[float] = 0.80

EXPECTED_MINIMUM_QBER: Final[float] = 0.15


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the full-eavesdropping FT-QuPAP scenario.

    Args:
        context:
            Network environment such as urban, suburban, or rural.

        random_seed:
            Reproducible seed used by the quantum simulation.

        app_config:
            Optional FT-QuPAP application configuration.

    Returns:
        Validated full-eavesdropping scenario configuration.
    """

    config = app_config or get_config()

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

            # Normal channel noise remains active in addition to
            # Eve's full interaction with the transmission.
            bit_flip_probability=0.004,
            phase_flip_probability=0.004,
            depolarizing_probability=0.006,
            measurement_error_probability=0.002,

            # Environmental loss remains below the deterministic
            # maximum-loss rejection threshold.
            loss_rate=0.03,

            burst_error_probability=0.0,
            burst_length=0,

            forced_lost_check_blocks=0,
            forced_uncorrectable_blocks=0,
        ),
        eve=EveProfile(
            enabled=True,
            attack_mode=(
                EveAttackMode.FULL_EAVESDROPPING.value
            ),
            interaction_fraction=(
                FULL_INTERACTION_FRACTION
            ),
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
            "Eve interacts with the complete quantum transmission.",
            "Both payload and independent check blocks are targeted.",
            "Eve does not know the encrypted control schedule.",
            "Eve may select incompatible measurement bases.",
            "Independent check-block QBER should rise significantly.",
            "Steane syndrome activity should increase.",
            "The decoded authentication payload may be corrupted.",
            "The reconstructed KMAC tag may fail verification.",
            "The calibrated GP attack probability should be high.",
            "Clear attack evidence must produce immediate rejection.",
            "Retry is not permitted for this attack scenario.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": (
                "full_quantum_eavesdropping"
            ),
            "attack_enabled": True,
            "attacker": "Eve",
            "attack_mode": (
                EveAttackMode.FULL_EAVESDROPPING.value
            ),
            "interaction_fraction": (
                FULL_INTERACTION_FRACTION
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
            "expected_minimum_qber": (
                EXPECTED_MINIMUM_QBER
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
            "operational_gp_threshold": (
                config.machine_learning
                .operational_threshold
            ),
            "retry_upper_probability": (
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
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """Alias for building the full-eavesdropping scenario."""

    return build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )


def get_scenario_summary(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return dashboard-friendly attack information."""

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )

    summary = scenario.dashboard_summary()

    summary.update(
        {
            "attacker": "Eve",
            "attack_stage": "quantum_transmission",
            "attack_scope": "full",
            "interaction_fraction": (
                FULL_INTERACTION_FRACTION
            ),
            "expected_decision": EXPECTED_DECISION,
            "expected_attack_detected": True,
            "retry_allowed": False,
            "classical_tampering": False,
            "cryptographic_forgery": False,
            "expected_detection_sources": [
                "independent_check_qber",
                "basis_mismatch_rate",
                "syndrome_activity",
                "corrected_block_count",
                "uncorrectable_block_count",
                "payload_reconstruction_failure",
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
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return effective full-attack conditions for one attempt."""

    config = app_config or get_config()

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
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
            "attack_scope": "full",
            "all_payload_blocks_targeted": True,
            "all_check_blocks_targeted": True,
        }
    )

    return result


def get_attack_coverage(
    *,
    app_config: ApplicationConfig | None = None,
) -> dict[str, int | float]:
    """
    Return the complete quantum-transmission attack coverage.

    The full-eavesdropping scenario targets every logical block and
    every Steane-encoded physical qubit.
    """

    config = app_config or get_config()

    return {
        "interaction_fraction": (
            FULL_INTERACTION_FRACTION
        ),
        "targeted_payload_blocks": (
            config.quantum.logical_payload_blocks
        ),
        "targeted_check_blocks": (
            config.quantum.independent_check_blocks
        ),
        "targeted_logical_blocks": (
            config.quantum.total_logical_blocks
        ),
        "targeted_physical_qubits": (
            config.quantum.total_physical_qubits
        ),
    }


def validate_attack_result(
    result: Mapping[str, Any],
    *,
    app_config: ApplicationConfig | None = None,
) -> bool:
    """
    Validate whether a result represents full-attack detection.

    A valid result must contain a rejection decision and at least one
    strong quantum, decoding, tag, or GP attack indicator.
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
        "rejected_uncorrectable",
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

    strong_qber_alert = (
        qber is not None
        and qber >= EXPECTED_MINIMUM_QBER
    )

    gp_alert = (
        attack_probability is not None
        and attack_probability
        >= config.machine_learning
        .operational_threshold
    )

    strong_gp_alert = (
        attack_probability is not None
        and attack_probability
        >= EXPECTED_MINIMUM_ATTACK_PROBABILITY
    )

    syndrome_alert = (
        syndrome_rate is not None
        and syndrome_rate > 0.15
    )

    decoder_alert = (
        corrected_blocks > 0
        or uncorrectable_blocks > 0
    )

    tag_alert = tag_valid is False
    payload_alert = payload_valid is False

    attack_evidence = any(
        (
            qber_alert,
            strong_qber_alert,
            gp_alert,
            strong_gp_alert,
            syndrome_alert,
            decoder_alert,
            tag_alert,
            payload_alert,
        )
    )

    return rejected and attack_evidence


def normalize_text(value: Any) -> str:
    """Normalize an optional value as lowercase text."""

    if value is None:
        return ""

    return str(value).strip().lower()


def safe_float(value: Any) -> float | None:
    """Convert a finite numeric value into float."""

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
    """Convert an integer-like value into int."""

    if value is None or isinstance(value, bool):
        return default

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default

    if not converted.is_integer():
        return default

    return int(converted)


FULL_EAVESDROPPING_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run full-eavesdropping consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert scenario.name == "full_eavesdropping"

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
        == "full_eavesdropping"
    )

    assert (
        scenario.eve.interaction_fraction
        == 1.0
    )

    assert (
        scenario.eve.target_payload_blocks
        is True
    )

    assert (
        scenario.eve.target_check_blocks
        is True
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

    coverage = get_attack_coverage(
        app_config=config
    )

    assert (
        coverage["targeted_payload_blocks"]
        == 128
    )

    assert (
        coverage["targeted_check_blocks"]
        == 32
    )

    assert (
        coverage["targeted_logical_blocks"]
        == 160
    )

    assert (
        coverage["targeted_physical_qubits"]
        == 1120
    )

    valid_result = validate_attack_result(
        {
            "decision": "rejected_gp",
            "check_qber": 0.27,
            "gp_probability": 0.96,
            "syndrome_rate": 0.42,
            "corrected_blocks": 31,
            "uncorrectable_blocks": 12,
            "tag_valid": False,
            "payload_valid": False,
        },
        app_config=config,
    )

    assert valid_result is True

    valid_qber_result = validate_attack_result(
        {
            "decision": "rejected_qber",
            "check_qber": 0.24,
            "gp_probability": 0.72,
            "syndrome_rate": 0.33,
            "corrected_blocks": 20,
            "uncorrectable_blocks": 4,
            "tag_valid": False,
            "payload_valid": False,
        },
        app_config=config,
    )

    assert valid_qber_result is True

    invalid_result = validate_attack_result(
        {
            "decision": "accepted",
            "check_qber": 0.02,
            "gp_probability": 0.04,
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
    assert summary["attack_scope"] == "full"
    assert summary["attacker"] == "Eve"
    assert summary["retry_allowed"] is False

    print(
        "FT-QuPAP full-eavesdropping "
        "scenario self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "FULL_INTERACTION_FRACTION",
    "DEFAULT_BASIS_ERROR_PROBABILITY",
    "DEFAULT_RESEND_ERROR_PROBABILITY",
    "EXPECTED_DECISION",
    "EXPECTED_MINIMUM_ATTACK_PROBABILITY",
    "EXPECTED_MINIMUM_QBER",
    "FULL_EAVESDROPPING_CONFIG",
    "build_scenario",
    "create_scenario",
    "get_scenario_summary",
    "get_attempt_configuration",
    "get_attack_coverage",
    "validate_attack_result",
]


if __name__ == "__main__":
    run_self_test()