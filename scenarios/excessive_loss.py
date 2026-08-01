"""
FT-QuPAP Excessive Quantum-Channel Loss Scenario
=================================================

Defines a controlled channel-failure scenario in which quantum loss
exceeds the safe operational limits of FT-QuPAP v5.1.

FT-QuPAP requires sufficient received quantum evidence before making an
authentication decision. The Authentication Server checks:

- Total quantum-channel loss rate
- Number of observed independent check blocks
- Number of missing check blocks
- Availability of payload blocks
- Steane-decoding results
- QBER measurement reliability

Established protocol limits:

- Maximum accepted loss rate: 0.15
- Independent check blocks: 32
- Minimum observed check blocks: 24

This scenario intentionally produces:

- A channel-loss rate above 0.15
- More than eight missing independent check blocks
- Fewer than 24 observable check blocks

Expected result:

    REJECTED_EXCESSIVE_LOSS

The rejection is deterministic. Insufficient quantum evidence must not
be interpreted as a successful authentication, low-risk GP result, or
retry-compatible condition.
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


SCENARIO_NAME: Final[str] = "excessive_loss"

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Excessive Quantum Loss"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "The quantum channel loses more physical qubits and independent "
    "check blocks than FT-QuPAP permits. The Authentication Server "
    "must reject the session because the available quantum evidence "
    "is insufficient for secure authentication."
)

DEFAULT_RANDOM_SEED: Final[int] = 9401

DEFAULT_CHANNEL_LOSS_RATE: Final[float] = 0.22

DEFAULT_FORCED_LOST_CHECK_BLOCKS: Final[int] = 10

EXPECTED_DECISION: Final[str] = (
    "rejected_excessive_loss"
)

EXPECTED_REJECTION_STAGE: Final[str] = (
    "quantum_loss_validation"
)


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    loss_rate: float = DEFAULT_CHANNEL_LOSS_RATE,
    forced_lost_check_blocks: int = (
        DEFAULT_FORCED_LOST_CHECK_BLOCKS
    ),
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the FT-QuPAP excessive-loss scenario.

    Args:
        context:
            Network environment such as urban, suburban, or rural.

        random_seed:
            Reproducible quantum-channel simulation seed.

        loss_rate:
            Physical-qubit loss probability. It must exceed the
            configured maximum loss rate for this scenario.

        forced_lost_check_blocks:
            Number of independent check blocks intentionally marked
            unavailable. The remaining observed check blocks must be
            below the configured minimum.

        app_config:
            Optional FT-QuPAP application configuration.

    Returns:
        Validated excessive-loss scenario configuration.
    """

    config = app_config or get_config()

    validate_excessive_loss_parameters(
        loss_rate=loss_rate,
        forced_lost_check_blocks=(
            forced_lost_check_blocks
        ),
        app_config=config,
    )

    observed_check_blocks = (
        config.quantum.independent_check_blocks
        - forced_lost_check_blocks
    )

    scenario = create_scenario_config(
        name=SCENARIO_NAME,
        display_name=SCENARIO_DISPLAY_NAME,
        description=SCENARIO_DESCRIPTION,
        category=ScenarioCategory.CHANNEL_FAILURE,
        expected_outcome=ExpectedOutcome.REJECTED,
        context=context,
        random_seed=random_seed,
        channel=ChannelProfile(
            noise_model=NoiseModelName.COMBINED.value,

            # Noise remains moderate so the rejection can be clearly
            # attributed to excessive quantum loss.
            bit_flip_probability=0.004,
            phase_flip_probability=0.004,
            depolarizing_probability=0.006,
            measurement_error_probability=0.002,

            # Intentionally above the protocol limit of 0.15.
            loss_rate=loss_rate,

            burst_error_probability=0.0,
            burst_length=0,

            # Leaves fewer than 24 of the 32 independent check
            # blocks available to the Authentication Server.
            forced_lost_check_blocks=(
                forced_lost_check_blocks
            ),

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
            noise_multiplier_after_retry=1.0,
            loss_multiplier_after_retry=1.0,
            change_random_seed_per_attempt=False,
        ),
        deterministic_verification_expected=True,

        # The protocol must reject the session before trusting a GP
        # prediction derived from incomplete quantum evidence.
        gp_evaluation_expected=False,
        notes=(
            "The Mobile Station and classical credentials are valid.",
            "No active attacker is introduced.",
            "The quantum-channel loss rate exceeds the safe limit.",
            "Too many independent check blocks are unavailable.",
            "The observed check-block count is below the minimum.",
            "QBER calculated from insufficient checks is not trusted.",
            "The GP detector must not override loss validation.",
            "Authentication must be rejected deterministically.",
            "Retry is not permitted for this controlled failure.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": (
                "excessive_quantum_channel_loss"
            ),
            "attack_enabled": False,
            "channel_failure_enabled": True,
            "failure_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "configured_loss_rate": loss_rate,
            "maximum_loss_rate": (
                config.quantum.maximum_loss_rate
            ),
            "independent_check_blocks": (
                config.quantum
                .independent_check_blocks
            ),
            "forced_lost_check_blocks": (
                forced_lost_check_blocks
            ),
            "expected_observed_check_blocks": (
                observed_check_blocks
            ),
            "minimum_observed_check_blocks": (
                config.quantum
                .minimum_observed_check_blocks
            ),
            "expected_decision": EXPECTED_DECISION,
            "deterministic_rejection_expected": True,
            "qber_trusted_expected": False,
            "gp_evaluation_required": False,
            "retry_expected": False,
            "steane_code": "[[7,1,3]]",
            "logical_payload_blocks": (
                config.quantum.logical_payload_blocks
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
    loss_rate: float = DEFAULT_CHANNEL_LOSS_RATE,
    forced_lost_check_blocks: int = (
        DEFAULT_FORCED_LOST_CHECK_BLOCKS
    ),
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """Alias for building the excessive-loss scenario."""

    return build_scenario(
        context=context,
        random_seed=random_seed,
        loss_rate=loss_rate,
        forced_lost_check_blocks=(
            forced_lost_check_blocks
        ),
        app_config=app_config,
    )


def get_scenario_summary(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    loss_rate: float = DEFAULT_CHANNEL_LOSS_RATE,
    forced_lost_check_blocks: int = (
        DEFAULT_FORCED_LOST_CHECK_BLOCKS
    ),
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return dashboard-friendly channel-failure information."""

    config = app_config or get_config()

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        loss_rate=loss_rate,
        forced_lost_check_blocks=(
            forced_lost_check_blocks
        ),
        app_config=config,
    )

    observed_check_blocks = (
        config.quantum.independent_check_blocks
        - forced_lost_check_blocks
    )

    summary = scenario.dashboard_summary()

    summary.update(
        {
            "failure_type": (
                "excessive_quantum_channel_loss"
            ),
            "failure_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "configured_loss_rate": loss_rate,
            "maximum_loss_rate": (
                config.quantum.maximum_loss_rate
            ),
            "loss_limit_exceeded": (
                loss_rate
                > config.quantum.maximum_loss_rate
            ),
            "total_check_blocks": (
                config.quantum
                .independent_check_blocks
            ),
            "lost_check_blocks": (
                forced_lost_check_blocks
            ),
            "observed_check_blocks": (
                observed_check_blocks
            ),
            "minimum_observed_check_blocks": (
                config.quantum
                .minimum_observed_check_blocks
            ),
            "minimum_check_requirement_met": (
                observed_check_blocks
                >= config.quantum
                .minimum_observed_check_blocks
            ),
            "expected_decision": EXPECTED_DECISION,
            "qber_trusted_expected": False,
            "gp_evaluation_required": False,
            "retry_allowed": False,
            "expected_detection_sources": [
                "physical_qubit_loss_count",
                "channel_loss_rate",
                "observed_check_block_count",
                "minimum_check_block_requirement",
                "deterministic_loss_validation",
            ],
        }
    )

    return summary


def get_attempt_configuration(
    attempt_number: int = 1,
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    loss_rate: float = DEFAULT_CHANNEL_LOSS_RATE,
    forced_lost_check_blocks: int = (
        DEFAULT_FORCED_LOST_CHECK_BLOCKS
    ),
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return effective excessive-loss conditions."""

    config = app_config or get_config()

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        loss_rate=loss_rate,
        forced_lost_check_blocks=(
            forced_lost_check_blocks
        ),
        app_config=config,
    )

    attempt = scenario.for_attempt(
        attempt_number=attempt_number,
        app_config=config,
    )

    observed_check_blocks = (
        config.quantum.independent_check_blocks
        - forced_lost_check_blocks
    )

    result = attempt.to_dictionary()

    result.update(
        {
            "configured_loss_rate": loss_rate,
            "maximum_loss_rate": (
                config.quantum.maximum_loss_rate
            ),
            "loss_limit_exceeded": True,
            "forced_lost_check_blocks": (
                forced_lost_check_blocks
            ),
            "expected_observed_check_blocks": (
                observed_check_blocks
            ),
            "minimum_observed_check_blocks": (
                config.quantum
                .minimum_observed_check_blocks
            ),
            "sufficient_check_blocks_expected": False,
            "qber_trusted_expected": False,
            "expected_attempt_decision": (
                EXPECTED_DECISION
            ),
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "continue_to_gp_evaluation": False,
            "retry_allowed": False,
        }
    )

    return result


def create_loss_failure_plan(
    *,
    app_config: ApplicationConfig | None = None,
    loss_rate: float = DEFAULT_CHANNEL_LOSS_RATE,
    forced_lost_check_blocks: int = (
        DEFAULT_FORCED_LOST_CHECK_BLOCKS
    ),
) -> list[dict[str, Any]]:
    """
    Return the controlled excessive-loss demonstration sequence.
    """

    config = app_config or get_config()

    validate_excessive_loss_parameters(
        loss_rate=loss_rate,
        forced_lost_check_blocks=(
            forced_lost_check_blocks
        ),
        app_config=config,
    )

    observed_check_blocks = (
        config.quantum.independent_check_blocks
        - forced_lost_check_blocks
    )

    return [
        {
            "step": 1,
            "actor": "mobile_station",
            "operation": (
                "prepare_steane_encoded_quantum_blocks"
            ),
            "logical_payload_blocks": (
                config.quantum.logical_payload_blocks
            ),
            "independent_check_blocks": (
                config.quantum
                .independent_check_blocks
            ),
            "physical_qubits": (
                config.quantum.total_physical_qubits
            ),
        },
        {
            "step": 2,
            "actor": "quantum_channel",
            "operation": (
                "apply_excessive_loss"
            ),
            "loss_rate": loss_rate,
            "maximum_allowed_loss_rate": (
                config.quantum.maximum_loss_rate
            ),
            "loss_limit_exceeded": True,
        },
        {
            "step": 3,
            "actor": "quantum_channel",
            "operation": (
                "remove_independent_check_blocks"
            ),
            "lost_check_blocks": (
                forced_lost_check_blocks
            ),
            "remaining_check_blocks": (
                observed_check_blocks
            ),
        },
        {
            "step": 4,
            "actor": "authentication_server",
            "operation": (
                "validate_received_quantum_evidence"
            ),
            "minimum_required_check_blocks": (
                config.quantum
                .minimum_observed_check_blocks
            ),
            "observed_check_blocks": (
                observed_check_blocks
            ),
            "validation_result": "fail",
        },
        {
            "step": 5,
            "actor": "authentication_server",
            "operation": "reject_authentication",
            "expected_decision": EXPECTED_DECISION,
            "qber_trusted": False,
            "gp_evaluation_started": False,
            "retry_requested": False,
        },
    ]


def validate_excessive_loss_result(
    result: Mapping[str, Any],
    *,
    app_config: ApplicationConfig | None = None,
) -> bool:
    """
    Validate whether excessive loss was correctly rejected.

    A valid result must show:

    - Authentication rejection
    - Excessive loss or insufficient check-block evidence
    - No successful authentication
    - No GP override
    - No retry
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

    reason = normalize_text(
        result.get(
            "reason",
            result.get("rejection_reason"),
        )
    )

    observed_loss_rate = safe_float(
        result.get(
            "loss_rate",
            result.get(
                "observed_loss_rate",
            ),
        )
    )

    observed_check_blocks = safe_integer_or_none(
        result.get(
            "observed_check_blocks",
            result.get(
                "received_check_blocks",
            ),
        )
    )

    excessive_loss_detected = read_boolean(
        result,
        (
            "excessive_loss_detected",
            "loss_limit_exceeded",
            "channel_loss_failure",
        ),
        default=False,
    )

    insufficient_checks_detected = read_boolean(
        result,
        (
            "insufficient_check_blocks",
            "minimum_check_requirement_failed",
            "check_block_shortage_detected",
        ),
        default=False,
    )

    gp_started = read_boolean(
        result,
        (
            "gp_evaluation_started",
            "gp_prediction_performed",
        ),
        default=False,
    )

    retry_used = read_boolean(
        result,
        (
            "retry_used",
            "retry_requested",
        ),
        default=False,
    )

    accepted = read_boolean(
        result,
        ("accepted",),
        default=False,
    )

    valid_decisions = {
        "rejected",
        "rejected_loss",
        "rejected_excessive_loss",
        "rejected_insufficient_checks",
        "rejected_quantum_evidence",
        "failed",
    }

    valid_reasons = {
        "excessive_loss",
        "excessive_quantum_loss",
        "loss_rate_exceeded",
        "insufficient_check_blocks",
        "minimum_observed_checks_not_met",
        "insufficient_quantum_evidence",
    }

    loss_rate_failure = (
        observed_loss_rate is not None
        and observed_loss_rate
        > config.quantum.maximum_loss_rate
    )

    check_count_failure = (
        observed_check_blocks is not None
        and observed_check_blocks
        < config.quantum
        .minimum_observed_check_blocks
    )

    failure_evidence = any(
        (
            excessive_loss_detected,
            insufficient_checks_detected,
            loss_rate_failure,
            check_count_failure,
            reason in valid_reasons,
        )
    )

    return (
        decision in valid_decisions
        and failure_evidence
        and not accepted
        and not gp_started
        and not retry_used
    )


def validate_excessive_loss_parameters(
    *,
    loss_rate: float,
    forced_lost_check_blocks: int,
    app_config: ApplicationConfig,
) -> None:
    """Validate controlled excessive-loss parameters."""

    if isinstance(loss_rate, bool) or not isinstance(
        loss_rate,
        (int, float),
    ):
        raise TypeError(
            "loss_rate must be numeric."
        )

    normalized_loss_rate = float(loss_rate)

    if not 0.0 <= normalized_loss_rate <= 1.0:
        raise ValueError(
            "loss_rate must be between 0 and 1."
        )

    if (
        normalized_loss_rate
        <= app_config.quantum.maximum_loss_rate
    ):
        raise ValueError(
            "The excessive-loss scenario requires a loss rate "
            "greater than the configured maximum loss rate."
        )

    if isinstance(
        forced_lost_check_blocks,
        bool,
    ) or not isinstance(
        forced_lost_check_blocks,
        int,
    ):
        raise TypeError(
            "forced_lost_check_blocks must be an integer."
        )

    if forced_lost_check_blocks < 0:
        raise ValueError(
            "forced_lost_check_blocks cannot be negative."
        )

    total_check_blocks = (
        app_config.quantum.independent_check_blocks
    )

    if forced_lost_check_blocks > total_check_blocks:
        raise ValueError(
            "forced_lost_check_blocks cannot exceed the "
            "number of independent check blocks."
        )

    observed_check_blocks = (
        total_check_blocks
        - forced_lost_check_blocks
    )

    if (
        observed_check_blocks
        >= app_config.quantum
        .minimum_observed_check_blocks
    ):
        raise ValueError(
            "The excessive-loss scenario requires fewer "
            "observed check blocks than the configured minimum."
        )


def normalize_text(value: Any) -> str:
    """Normalize an optional value as lowercase identifier text."""

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
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


def safe_integer_or_none(
    value: Any,
) -> int | None:
    """Convert an integer-like value to int."""

    if value is None or isinstance(value, bool):
        return None

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    if not converted.is_integer():
        return None

    return int(converted)


def read_boolean(
    mapping: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    default: bool,
) -> bool:
    """Read the first boolean-compatible mapping field."""

    for field_name in field_names:
        if field_name not in mapping:
            continue

        value = mapping[field_name]

        if isinstance(value, bool):
            return value

        if isinstance(value, int) and value in {
            0,
            1,
        }:
            return bool(value)

        if isinstance(value, str):
            normalized_value = (
                value.strip().lower()
            )

            if normalized_value in {
                "true",
                "yes",
                "1",
                "detected",
                "failed",
                "exceeded",
                "performed",
            }:
                return True

            if normalized_value in {
                "false",
                "no",
                "0",
                "not_detected",
                "passed",
                "within_limit",
                "not_performed",
            }:
                return False

    return default


EXCESSIVE_LOSS_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run excessive-loss scenario consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert scenario.name == "excessive_loss"

    assert (
        scenario.category
        == "channel_failure"
    )

    assert scenario.expected_outcome == "rejected"
    assert scenario.attack_enabled is False
    assert scenario.retry_expected is False

    assert scenario.eve.enabled is False
    assert scenario.tampering.enabled is False
    assert scenario.retry.enabled is False

    assert (
        scenario.gp_evaluation_expected
        is False
    )

    assert (
        scenario.channel.loss_rate
        > config.quantum.maximum_loss_rate
    )

    observed_check_blocks = (
        config.quantum.independent_check_blocks
        - scenario.channel
        .forced_lost_check_blocks
    )

    assert (
        observed_check_blocks
        < config.quantum
        .minimum_observed_check_blocks
    )

    assert (
        scenario.channel
        .forced_uncorrectable_blocks
        == 0
    )

    attempt = scenario.for_attempt(
        1,
        config,
    )

    assert (
        attempt.channel.loss_rate
        == DEFAULT_CHANNEL_LOSS_RATE
    )

    assert (
        attempt.channel
        .forced_lost_check_blocks
        == DEFAULT_FORCED_LOST_CHECK_BLOCKS
    )

    assert (
        attempt.force_retry_gray_zone
        is False
    )

    failure_plan = create_loss_failure_plan(
        app_config=config
    )

    assert len(failure_plan) == 5

    assert (
        failure_plan[1]["loss_limit_exceeded"]
        is True
    )

    assert (
        failure_plan[3]["validation_result"]
        == "fail"
    )

    assert (
        failure_plan[4]["expected_decision"]
        == EXPECTED_DECISION
    )

    valid_loss_result = (
        validate_excessive_loss_result(
            {
                "decision": (
                    "rejected_excessive_loss"
                ),
                "reason": "loss_rate_exceeded",
                "loss_rate": 0.22,
                "observed_check_blocks": 22,
                "loss_limit_exceeded": True,
                "insufficient_check_blocks": True,
                "gp_evaluation_started": False,
                "retry_requested": False,
                "accepted": False,
            },
            app_config=config,
        )
    )

    assert valid_loss_result is True

    valid_check_result = (
        validate_excessive_loss_result(
            {
                "decision": (
                    "rejected_insufficient_checks"
                ),
                "reason": (
                    "minimum_observed_checks_not_met"
                ),
                "observed_loss_rate": 0.18,
                "received_check_blocks": 21,
                "minimum_check_requirement_failed": True,
                "gp_prediction_performed": False,
                "retry_used": False,
                "accepted": False,
            },
            app_config=config,
        )
    )

    assert valid_check_result is True

    invalid_retry_result = (
        validate_excessive_loss_result(
            {
                "decision": "rejected_loss",
                "reason": "excessive_quantum_loss",
                "loss_rate": 0.25,
                "observed_check_blocks": 20,
                "excessive_loss_detected": True,
                "gp_evaluation_started": False,
                "retry_requested": True,
                "accepted": False,
            },
            app_config=config,
        )
    )

    assert invalid_retry_result is False

    invalid_gp_result = (
        validate_excessive_loss_result(
            {
                "decision": "rejected_loss",
                "reason": "loss_rate_exceeded",
                "loss_rate": 0.22,
                "observed_check_blocks": 22,
                "excessive_loss_detected": True,
                "gp_prediction_performed": True,
                "retry_requested": False,
                "accepted": False,
            },
            app_config=config,
        )
    )

    assert invalid_gp_result is False

    invalid_accept_result = (
        validate_excessive_loss_result(
            {
                "decision": "accepted",
                "loss_rate": 0.22,
                "observed_check_blocks": 22,
                "loss_limit_exceeded": True,
                "gp_evaluation_started": False,
                "retry_requested": False,
                "accepted": True,
            },
            app_config=config,
        )
    )

    assert invalid_accept_result is False

    summary = get_scenario_summary(
        app_config=config
    )

    assert summary["attack_enabled"] is False
    assert summary["loss_limit_exceeded"] is True

    assert (
        summary["minimum_check_requirement_met"]
        is False
    )

    assert (
        summary["expected_decision"]
        == "rejected_excessive_loss"
    )

    assert summary["retry_allowed"] is False

    print(
        "FT-QuPAP excessive-loss scenario "
        "self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_CHANNEL_LOSS_RATE",
    "DEFAULT_FORCED_LOST_CHECK_BLOCKS",
    "EXPECTED_DECISION",
    "EXPECTED_REJECTION_STAGE",
    "EXCESSIVE_LOSS_CONFIG",
    "build_scenario",
    "create_scenario",
    "get_scenario_summary",
    "get_attempt_configuration",
    "create_loss_failure_plan",
    "validate_excessive_loss_result",
    "validate_excessive_loss_parameters",
]


if __name__ == "__main__":
    run_self_test()