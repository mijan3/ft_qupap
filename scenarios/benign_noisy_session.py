"""
FT-QuPAP Benign Noisy Authentication Scenario
==============================================

Defines a valid-user authentication scenario affected by realistic
benign quantum-channel noise.

Scenario characteristics:

- Registered and legitimate Mobile Station
- Fresh timestamp and unique nonce
- Valid ML-DSA-65 server signature
- Valid ML-KEM-768 encapsulation and decapsulation
- Matching transcript-bound session keys
- Correct 128-bit KMAC256 authentication tag
- No eavesdropper
- No classical or cryptographic tampering
- Increased but non-malicious channel noise
- Moderate physical-qubit loss
- Possible correctable Steane [[7,1,3]] errors
- Retry permitted when evidence enters the protected gray zone
- Improved channel conditions on later attempts

Expected result:

    ACCEPTED or ACCEPTED_AFTER_RETRY

This scenario demonstrates that FT-QuPAP should distinguish benign
channel degradation from active eavesdropping while still enforcing all
deterministic checks, QBER limits, loss limits, GP analysis, and bounded
retry rules.
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


SCENARIO_NAME: Final[str] = "benign_noisy_session"

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Benign Noisy Session"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "A legitimate Mobile Station authenticates through a degraded "
    "but non-malicious quantum channel. Increased noise and loss may "
    "cause an initial retry, but all classical credentials and "
    "transcript-bound cryptographic values remain valid."
)

DEFAULT_RANDOM_SEED: Final[int] = 9102


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the benign noisy FT-QuPAP scenario.

    Args:
        context:
            Network environment such as urban, suburban, or rural.

        random_seed:
            Reproducible seed used by quantum-channel simulation.

        app_config:
            Optional application configuration.

    Returns:
        Validated benign noisy scenario configuration.
    """

    config = app_config or get_config()

    scenario = create_scenario_config(
        name=SCENARIO_NAME,
        display_name=SCENARIO_DISPLAY_NAME,
        description=SCENARIO_DESCRIPTION,
        category=ScenarioCategory.BENIGN,
        expected_outcome=(
            ExpectedOutcome.RETRY_OR_ACCEPT
        ),
        context=context,
        random_seed=random_seed,
        channel=ChannelProfile(
            noise_model=NoiseModelName.COMBINED.value,

            # Increased benign channel errors.
            bit_flip_probability=0.012,
            phase_flip_probability=0.010,
            depolarizing_probability=0.018,
            measurement_error_probability=0.006,

            # Below the configured maximum loss rate of 0.15.
            loss_rate=0.08,

            # Small correlated error bursts may occur naturally.
            burst_error_probability=0.025,
            burst_length=2,

            # No artificial loss or uncorrectable error is forced.
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
            enabled=True,

            # Retry is determined by measured evidence rather than
            # being artificially forced in this scenario.
            force_retry_on_attempts=(),

            # Later attempts model an improved channel condition.
            noise_multiplier_after_retry=0.55,
            loss_multiplier_after_retry=0.70,

            # Each attempt remains reproducible but receives a
            # different deterministic seed.
            change_random_seed_per_attempt=True,
        ),
        deterministic_verification_expected=True,
        gp_evaluation_expected=True,
        notes=(
            "The subscriber and server credentials remain valid.",
            "No active attacker is introduced.",
            "The channel has higher benign noise than normal_session.",
            "The first attempt may be accepted or enter the retry zone.",
            "Retry attempts use reduced noise and loss conditions.",
            "Deterministic verification remains mandatory.",
            "A retry must not override replay, signature, tag, or "
            "ciphertext failures.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": "benign_channel_degradation",
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
            "maximum_attempts": (
                config.protocol
                .maximum_authentication_attempts
            ),
            "maximum_retries": (
                config.protocol.maximum_retries
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
    """Alias for building the benign noisy scenario."""

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
    """Return dashboard-friendly scenario information."""

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )

    summary = scenario.dashboard_summary()

    summary.update(
        {
            "expected_behavior": (
                "Accept directly or request a bounded retry "
                "when benign evidence enters the gray zone."
            ),
            "deterministic_checks_required": True,
            "active_attacker": False,
            "retry_condition": (
                "Determined by protocol evidence rather than "
                "forced by the scenario."
            ),
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
    """
    Return effective noisy-channel values for one attempt.

    Later attempts contain reduced benign noise and loss according to
    the retry profile.
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


def compare_attempt_conditions(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Return effective channel conditions for all allowed attempts.

    This function is useful for the dashboard retry visualization.
    """

    config = app_config or get_config()

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=config,
    )

    attempt_conditions: list[dict[str, Any]] = []

    for attempt_number in range(
        1,
        (
            config.protocol
            .maximum_authentication_attempts
            + 1
        ),
    ):
        attempt = scenario.for_attempt(
            attempt_number=attempt_number,
            app_config=config,
        )

        attempt_conditions.append(
            {
                "attempt_number": attempt.attempt_number,
                "random_seed": attempt.random_seed,
                "bit_flip_probability": (
                    attempt.channel
                    .bit_flip_probability
                ),
                "phase_flip_probability": (
                    attempt.channel
                    .phase_flip_probability
                ),
                "depolarizing_probability": (
                    attempt.channel
                    .depolarizing_probability
                ),
                "measurement_error_probability": (
                    attempt.channel
                    .measurement_error_probability
                ),
                "loss_rate": attempt.channel.loss_rate,
                "burst_error_probability": (
                    attempt.channel
                    .burst_error_probability
                ),
                "force_retry_gray_zone": (
                    attempt.force_retry_gray_zone
                ),
            }
        )

    return attempt_conditions


BENIGN_NOISY_SESSION_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run benign noisy scenario consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert scenario.name == "benign_noisy_session"
    assert scenario.category == "benign"

    assert (
        scenario.expected_outcome
        == "retry_or_accept"
    )

    assert scenario.attack_enabled is False
    assert scenario.retry_expected is True

    assert scenario.eve.enabled is False
    assert scenario.eve.attack_mode == "none"

    assert scenario.tampering.enabled is False
    assert (
        scenario.tampering.active_actions()
        == ()
    )

    assert scenario.retry.enabled is True

    assert (
        scenario.retry.force_retry_on_attempts
        == ()
    )

    assert scenario.channel.loss_rate == 0.08

    assert (
        scenario.channel.loss_rate
        < config.quantum.maximum_loss_rate
    )

    assert (
        scenario.channel.forced_lost_check_blocks
        == 0
    )

    assert (
        scenario.channel
        .forced_uncorrectable_blocks
        == 0
    )

    first_attempt = scenario.for_attempt(
        1,
        config,
    )

    second_attempt = scenario.for_attempt(
        2,
        config,
    )

    third_attempt = scenario.for_attempt(
        3,
        config,
    )

    assert first_attempt.random_seed == 9102
    assert second_attempt.random_seed == 9103
    assert third_attempt.random_seed == 9104

    assert (
        first_attempt.force_retry_gray_zone
        is False
    )

    assert (
        second_attempt.channel.loss_rate
        < first_attempt.channel.loss_rate
    )

    assert (
        third_attempt.channel.loss_rate
        < second_attempt.channel.loss_rate
    )

    assert (
        second_attempt.channel
        .depolarizing_probability
        < first_attempt.channel
        .depolarizing_probability
    )

    attempt_comparison = compare_attempt_conditions(
        app_config=config
    )

    assert len(attempt_comparison) == 3

    assert (
        attempt_comparison[0]["attempt_number"]
        == 1
    )

    assert (
        attempt_comparison[-1]["attempt_number"]
        == 3
    )

    summary = get_scenario_summary(
        app_config=config
    )

    assert summary["attack_enabled"] is False
    assert summary["retry_expected"] is True
    assert summary["active_attacker"] is False

    print(
        "FT-QuPAP benign noisy-session "
        "scenario self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "BENIGN_NOISY_SESSION_CONFIG",
    "build_scenario",
    "create_scenario",
    "get_scenario_summary",
    "get_attempt_configuration",
    "compare_attempt_conditions",
]


if __name__ == "__main__":
    run_self_test()