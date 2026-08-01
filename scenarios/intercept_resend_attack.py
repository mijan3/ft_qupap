"""
FT-QuPAP Intercept-Resend Attack Scenario
=========================================

Defines a controlled quantum intercept-resend attack against the
FT-QuPAP v5.1 authentication protocol.

Attack process:

1. The legitimate Mobile Station prepares the transcript-bound
   128-bit KMAC256 authentication tag.

2. The classical tag bits are converted into logical qubits.

3. Payload and independent check blocks are encoded using the
   Steane [[7,1,3]] CSS code.

4. Eve intercepts a configured portion of the transmitted quantum
   blocks.

5. Eve measures the intercepted qubits without knowing the secret
   control schedule and preparation bases.

6. Eve prepares replacement qubits from her measurement results.

7. The Authentication Server receives the disturbed sequence.

Expected evidence:

- Increased check-block QBER
- Increased syndrome activity
- More corrected and possibly uncorrectable blocks
- Possible payload reconstruction errors
- Possible KMAC tag mismatch
- Elevated calibrated GP attack probability

Expected result:

    REJECTED

The scenario does not alter the ML-DSA signature, ML-KEM ciphertext,
timestamp, nonce, or classical authentication request.
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


SCENARIO_NAME: Final[str] = (
    "intercept_resend_attack"
)

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Intercept-Resend Attack"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "Eve intercepts part of the encoded quantum transmission, "
    "measures the selected qubits without knowing their correct "
    "preparation bases, and sends replacement qubits to the "
    "Authentication Server."
)

DEFAULT_RANDOM_SEED: Final[int] = 9201

DEFAULT_INTERACTION_FRACTION: Final[float] = 0.60
DEFAULT_BASIS_ERROR_PROBABILITY: Final[float] = 0.50
DEFAULT_RESEND_ERROR_PROBABILITY: Final[float] = 0.03

EXPECTED_MINIMUM_ATTACK_PROBABILITY: Final[float] = 0.50
EXPECTED_DECISION: Final[str] = "rejected"


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
    Build the FT-QuPAP intercept-resend scenario.

    Args:
        context:
            Network environment such as urban, suburban, or rural.

        random_seed:
            Reproducible simulation seed.

        interaction_fraction:
            Fraction of transmitted quantum information intercepted
            and resent by Eve.

        app_config:
            Optional application configuration.

    Returns:
        Validated intercept-resend scenario configuration.
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

            # Normal environmental noise remains present in addition
            # to the disturbance introduced by Eve.
            bit_flip_probability=0.004,
            phase_flip_probability=0.004,
            depolarizing_probability=0.006,
            measurement_error_probability=0.002,
            loss_rate=0.025,

            burst_error_probability=0.0,
            burst_length=0,

            forced_lost_check_blocks=0,
            forced_uncorrectable_blocks=0,
        ),
        eve=EveProfile(
            enabled=True,
            attack_mode=(
                EveAttackMode.INTERCEPT_RESEND.value
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
            "The Mobile Station and server credentials are valid.",
            "The attack affects only the quantum transmission.",
            "Eve does not know the secret control schedule.",
            "Eve may select incompatible measurement bases.",
            "Independent check blocks should reveal disturbance.",
            "The Steane decoder may report increased syndrome activity.",
            "The calibrated GP detector should identify attack-like "
            "evidence.",
            "Retry must not be granted for clear attack evidence.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": "quantum_intercept_resend",
            "attack_enabled": True,
            "attack_mode": (
                EveAttackMode.INTERCEPT_RESEND.value
            ),
            "interaction_fraction": interaction_fraction,
            "basis_error_probability": (
                DEFAULT_BASIS_ERROR_PROBABILITY
            ),
            "resend_error_probability": (
                DEFAULT_RESEND_ERROR_PROBABILITY
            ),
            "expected_decision": EXPECTED_DECISION,
            "expected_minimum_attack_probability": (
                EXPECTED_MINIMUM_ATTACK_PROBABILITY
            ),
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
    interaction_fraction: float = (
        DEFAULT_INTERACTION_FRACTION
    ),
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """Alias for building the intercept-resend scenario."""

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
            "attack_operation": (
                "measure_and_prepare_replacement"
            ),
            "expected_decision": EXPECTED_DECISION,
            "retry_expected": False,
            "classical_request_valid": True,
            "server_signature_valid": True,
            "mlkem_ciphertext_valid": True,
            "original_kmac_tag_valid": True,
            "expected_detection_sources": [
                "check_block_qber",
                "syndrome_activity",
                "logical_block_errors",
                "decoded_tag_verification",
                "gp_attack_probability",
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
            "retry_allowed": False,
            "expected_attack_detected": True,
        }
    )

    return result


def estimate_intercepted_physical_qubits(
    interaction_fraction: float = (
        DEFAULT_INTERACTION_FRACTION
    ),
    *,
    app_config: ApplicationConfig | None = None,
) -> int:
    """
    Estimate how many physical qubits Eve selects.

    The actual simulator may use randomized selection, but this helper
    provides the expected count for dashboard presentation.
    """

    config = app_config or get_config()

    if isinstance(interaction_fraction, bool) or not isinstance(
        interaction_fraction,
        (int, float),
    ):
        raise TypeError(
            "interaction_fraction must be numeric."
        )

    normalized_fraction = float(
        interaction_fraction
    )

    if not 0.0 <= normalized_fraction <= 1.0:
        raise ValueError(
            "interaction_fraction must be between 0 and 1."
        )

    return round(
        config.quantum.total_physical_qubits
        * normalized_fraction
    )


def validate_attack_result(
    result: Mapping[str, Any],
    *,
    app_config: ApplicationConfig | None = None,
) -> bool:
    """
    Validate whether a result is consistent with attack rejection.

    At least one meaningful quantum or GP attack indicator must be
    present in addition to a rejection decision.
    """

    if not isinstance(result, Mapping):
        raise TypeError(
            "result must be a mapping."
        )

    config = app_config or get_config()

    decision = str(
        result.get(
            "decision",
            result.get("status", ""),
        )
    ).strip().lower()

    rejected = decision in {
        "rejected",
        "rejected_gp",
        "rejected_deterministic",
        "rejected_quantum",
        "rejected_qber",
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

    uncorrectable_blocks = safe_integer(
        result.get(
            "uncorrectable_blocks",
            0,
        )
    )

    tag_valid = result.get(
        "tag_valid",
        result.get("kmac_valid"),
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

    decoder_alert = uncorrectable_blocks > 0
    tag_alert = tag_valid is False

    return (
        rejected
        and (
            qber_alert
            or gp_alert
            or decoder_alert
            or tag_alert
        )
    )


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


INTERCEPT_RESEND_ATTACK_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run intercept-resend scenario consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert (
        scenario.name
        == "intercept_resend_attack"
    )

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
        == "intercept_resend"
    )

    assert (
        scenario.eve.interaction_fraction
        == DEFAULT_INTERACTION_FRACTION
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

    expected_intercepted = (
        estimate_intercepted_physical_qubits(
            app_config=config
        )
    )

    assert expected_intercepted == round(
        config.quantum.total_physical_qubits
        * DEFAULT_INTERACTION_FRACTION
    )

    valid_result = validate_attack_result(
        {
            "decision": "rejected_gp",
            "qber": 0.19,
            "attack_probability": 0.91,
            "uncorrectable_blocks": 3,
            "tag_valid": False,
        },
        app_config=config,
    )

    assert valid_result is True

    invalid_result = validate_attack_result(
        {
            "decision": "accepted",
            "qber": 0.02,
            "attack_probability": 0.05,
            "uncorrectable_blocks": 0,
            "tag_valid": True,
        },
        app_config=config,
    )

    assert invalid_result is False

    summary = get_scenario_summary(
        app_config=config
    )

    assert summary["attack_enabled"] is True
    assert summary["attacker"] == "Eve"
    assert summary["retry_expected"] is False

    print(
        "FT-QuPAP intercept-resend attack "
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
    "EXPECTED_MINIMUM_ATTACK_PROBABILITY",
    "EXPECTED_DECISION",
    "INTERCEPT_RESEND_ATTACK_CONFIG",
    "build_scenario",
    "create_scenario",
    "get_scenario_summary",
    "get_attempt_configuration",
    "estimate_intercepted_physical_qubits",
    "validate_attack_result",
]


if __name__ == "__main__":
    run_self_test()