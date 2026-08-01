"""
FT-QuPAP Normal Authentication Scenario
========================================

Defines the baseline successful authentication scenario for the
FT-QuPAP v5.1 capstone demonstration.

Scenario characteristics:

- Registered and valid Mobile Station
- Correct pseudonymous subscriber identity
- Fresh timestamp
- Unique authentication nonce
- Valid ML-DSA-65 server signature
- Valid ML-KEM-768 encapsulation and decapsulation
- Matching transcript-bound session keys
- Correct 128-bit KMAC256 authentication tag
- Low benign quantum-channel noise
- Low qubit-loss rate
- No eavesdropper
- No classical-message modification
- No cryptographic forgery
- No forced retry
- Correctable Steane [[7,1,3]] errors only

Expected result:

    ACCEPTED

This scenario does not bypass any deterministic or machine-learning
verification stage.
"""

from __future__ import annotations

from typing import Any, Final

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


SCENARIO_NAME: Final[str] = "normal_session"
SCENARIO_DISPLAY_NAME: Final[str] = "Normal Session"

SCENARIO_DESCRIPTION: Final[str] = (
    "A valid registered Mobile Station authenticates through a "
    "low-noise quantum channel without an attacker or message "
    "tampering."
)

DEFAULT_RANDOM_SEED: Final[int] = 9102


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the normal FT-QuPAP authentication scenario.

    Args:
        context:
            Network environment. Supported values are determined by
            the application configuration, normally urban, suburban,
            or rural.

        random_seed:
            Reproducible seed used by the channel and quantum
            simulation.

        app_config:
            Optional application configuration used for validation.

    Returns:
        Validated normal-session configuration.
    """

    config = app_config or get_config()

    scenario = create_scenario_config(
        name=SCENARIO_NAME,
        display_name=SCENARIO_DISPLAY_NAME,
        description=SCENARIO_DESCRIPTION,
        category=ScenarioCategory.BENIGN,
        expected_outcome=ExpectedOutcome.ACCEPTED,
        context=context,
        random_seed=random_seed,
        channel=ChannelProfile(
            noise_model=NoiseModelName.COMBINED.value,
            bit_flip_probability=0.002,
            phase_flip_probability=0.002,
            depolarizing_probability=0.003,
            measurement_error_probability=0.001,
            loss_rate=0.01,
            burst_error_probability=0.0,
            burst_length=0,
            forced_lost_check_blocks=0,
            forced_uncorrectable_blocks=0,
        ),
        eve=EveProfile(
            enabled=False,
            attack_mode=EveAttackMode.NONE.value,
            interaction_fraction=0.0,
            measurement_basis_error_probability=0.5,
            resend_error_probability=0.0,
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
            noise_multiplier_after_retry=0.55,
            loss_multiplier_after_retry=0.70,
            change_random_seed_per_attempt=True,
        ),
        deterministic_verification_expected=True,
        gp_evaluation_expected=True,
        notes=(
            "This is the baseline valid-user authentication case.",
            "All classical cryptographic verification steps must pass.",
            "The observed QBER should remain below the fixed threshold.",
            "The calibrated GP attack probability should remain low.",
            "No retry should be required under expected conditions.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "ml_dsa_parameter_set": (
                config.cryptography.ml_dsa_parameter_set
            ),
            "ml_kem_parameter_set": (
                config.cryptography.ml_kem_parameter_set
            ),
            "kmac_algorithm": (
                config.cryptography.kmac_algorithm
            ),
            "kmac_tag_bits": (
                config.cryptography.kmac_tag_bits
            ),
            "steane_code": "[[7,1,3]]",
            "logical_payload_blocks": (
                config.quantum.logical_payload_blocks
            ),
            "independent_check_blocks": (
                config.quantum.independent_check_blocks
            ),
            "total_logical_blocks": (
                config.quantum.total_logical_blocks
            ),
            "total_physical_qubits": (
                config.quantum.total_physical_qubits
            ),
            "fixed_qber_threshold": (
                config.quantum.fixed_qber_threshold
            ),
            "maximum_loss_rate": (
                config.quantum.maximum_loss_rate
            ),
            "operational_gp_threshold": (
                config.machine_learning.operational_threshold
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
    """Alias for building the normal scenario."""

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
    """Return dashboard-friendly normal-scenario information."""

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )

    return scenario.dashboard_summary()


def get_attempt_configuration(
    attempt_number: int = 1,
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """
    Return the effective normal-scenario values for one attempt.
    """

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

    return attempt.to_dictionary()


NORMAL_SESSION_CONFIG: Final[ScenarioConfig] = (
    build_scenario()
)


def run_self_test() -> None:
    """Run baseline scenario consistency checks."""

    config = get_config()
    scenario = build_scenario(
        app_config=config
    )

    assert scenario.name == "normal_session"
    assert scenario.category == "benign"
    assert scenario.expected_outcome == "accepted"

    assert scenario.attack_enabled is False
    assert scenario.retry_expected is False

    assert scenario.eve.enabled is False
    assert scenario.eve.attack_mode == "none"

    assert scenario.tampering.enabled is False
    assert (
        scenario.tampering.active_actions()
        == ()
    )

    assert scenario.retry.enabled is False
    assert (
        scenario.retry.force_retry_on_attempts
        == ()
    )

    assert scenario.channel.loss_rate == 0.01

    assert (
        scenario.channel.forced_lost_check_blocks
        == 0
    )

    assert (
        scenario.channel
        .forced_uncorrectable_blocks
        == 0
    )

    assert (
        scenario.channel.loss_rate
        < config.quantum.maximum_loss_rate
    )

    first_attempt = scenario.for_attempt(
        1,
        config,
    )

    assert first_attempt.attempt_number == 1
    assert first_attempt.random_seed == 9102
    assert (
        first_attempt.force_retry_gray_zone
        is False
    )

    summary = scenario.dashboard_summary()

    assert summary["attack_enabled"] is False
    assert summary["retry_expected"] is False
    assert summary["eve_attack_mode"] == "none"

    print(
        "FT-QuPAP normal-session scenario "
        "self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "NORMAL_SESSION_CONFIG",
    "build_scenario",
    "create_scenario",
    "get_scenario_summary",
    "get_attempt_configuration",
]


if __name__ == "__main__":
    run_self_test()