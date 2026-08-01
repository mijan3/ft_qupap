"""
FT-QuPAP Uncorrectable Quantum Error Scenario
=============================================

Defines a controlled fault-tolerance failure scenario for FT-QuPAP
v5.1.

FT-QuPAP protects every logical authentication and check bit using the
Steane [[7,1,3]] CSS quantum error-correcting code.

The Steane code has:

- 7 physical qubits per logical qubit
- 1 encoded logical qubit
- Code distance 3
- Guaranteed correction of one physical-qubit error per block

In this scenario, one or more selected logical blocks receive an error
pattern beyond the supported single-qubit correction capability.

Possible consequences include:

- Nonzero Steane syndrome
- Multiple physical-qubit errors in one logical block
- Ambiguous error location
- Decoder failure
- Logical-bit reconstruction failure
- Invalid reconstructed KMAC authentication tag

Expected result:

    REJECTED_UNCORRECTABLE_ERROR

The Authentication Server must reject the session deterministically.
A known uncorrectable logical block must never be hidden by the
Gaussian Process classifier or converted into a retry.
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
    "uncorrectable_quantum_error"
)

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Uncorrectable Quantum Error"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "One or more Steane [[7,1,3]] logical blocks receive a "
    "multi-qubit error pattern outside the guaranteed correction "
    "capability. The Authentication Server must reject the session "
    "during deterministic quantum-decoding validation."
)

DEFAULT_RANDOM_SEED: Final[int] = 9402

DEFAULT_FORCED_UNCORRECTABLE_BLOCKS: Final[int] = 3

DEFAULT_ERRORS_PER_AFFECTED_BLOCK: Final[int] = 2

EXPECTED_DECISION: Final[str] = (
    "rejected_uncorrectable_error"
)

EXPECTED_REJECTION_STAGE: Final[str] = (
    "steane_decoding_validation"
)

STEANE_PHYSICAL_QUBITS_PER_BLOCK: Final[int] = 7
STEANE_CODE_DISTANCE: Final[int] = 3
STEANE_CORRECTION_CAPABILITY: Final[int] = 1


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    forced_uncorrectable_blocks: int = (
        DEFAULT_FORCED_UNCORRECTABLE_BLOCKS
    ),
    errors_per_affected_block: int = (
        DEFAULT_ERRORS_PER_AFFECTED_BLOCK
    ),
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the uncorrectable-quantum-error scenario.

    Args:
        context:
            Network environment such as urban, suburban, or rural.

        random_seed:
            Reproducible quantum-simulation seed.

        forced_uncorrectable_blocks:
            Number of logical blocks intentionally given an error
            pattern beyond the supported correction capability.

        errors_per_affected_block:
            Number of physical-qubit errors introduced into each
            affected Steane block. This value must exceed one.

        app_config:
            Optional FT-QuPAP application configuration.

    Returns:
        Validated uncorrectable-error scenario configuration.
    """

    config = app_config or get_config()

    validate_uncorrectable_error_parameters(
        forced_uncorrectable_blocks=(
            forced_uncorrectable_blocks
        ),
        errors_per_affected_block=(
            errors_per_affected_block
        ),
        app_config=config,
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

            # Ordinary environmental noise remains low so rejection
            # is attributable to the forced uncorrectable blocks.
            bit_flip_probability=0.002,
            phase_flip_probability=0.002,
            depolarizing_probability=0.003,
            measurement_error_probability=0.001,

            # Loss remains below the maximum permitted value.
            loss_rate=0.01,

            burst_error_probability=0.0,
            burst_length=0,

            forced_lost_check_blocks=0,

            # Main controlled fault condition.
            forced_uncorrectable_blocks=(
                forced_uncorrectable_blocks
            ),
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

        # Known decoder failure is sufficient for immediate rejection.
        gp_evaluation_expected=False,
        notes=(
            "The Mobile Station and classical credentials are valid.",
            "No active eavesdropper is introduced.",
            "ML-KEM and transcript-bound key derivation remain valid.",
            "The original KMAC authentication tag is valid.",
            "Selected Steane blocks receive multi-qubit errors.",
            "The error weight exceeds the guaranteed correction limit.",
            "The decoder must report uncorrectable logical blocks.",
            "A known decoder failure requires deterministic rejection.",
            "GP analysis must not override uncorrectable-error evidence.",
            "Retry is forbidden for this controlled failure scenario.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": (
                "uncorrectable_steane_error"
            ),
            "attack_enabled": False,
            "channel_failure_enabled": True,
            "failure_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "steane_code": "[[7,1,3]]",
            "steane_physical_qubits_per_block": (
                STEANE_PHYSICAL_QUBITS_PER_BLOCK
            ),
            "steane_code_distance": (
                STEANE_CODE_DISTANCE
            ),
            "steane_correction_capability": (
                STEANE_CORRECTION_CAPABILITY
            ),
            "forced_uncorrectable_blocks": (
                forced_uncorrectable_blocks
            ),
            "errors_per_affected_block": (
                errors_per_affected_block
            ),
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
            "expected_decision": EXPECTED_DECISION,
            "deterministic_rejection_expected": True,
            "decoder_success_expected": False,
            "all_logical_blocks_recovered_expected": False,
            "kmac_tag_valid_expected": False,
            "gp_evaluation_required": False,
            "retry_expected": False,
        },
    )

    scenario.validate(config)

    return scenario


def create_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    forced_uncorrectable_blocks: int = (
        DEFAULT_FORCED_UNCORRECTABLE_BLOCKS
    ),
    errors_per_affected_block: int = (
        DEFAULT_ERRORS_PER_AFFECTED_BLOCK
    ),
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """Alias for building the uncorrectable-error scenario."""

    return build_scenario(
        context=context,
        random_seed=random_seed,
        forced_uncorrectable_blocks=(
            forced_uncorrectable_blocks
        ),
        errors_per_affected_block=(
            errors_per_affected_block
        ),
        app_config=app_config,
    )


def create_uncorrectable_error_pattern(
    *,
    errors_per_block: int = (
        DEFAULT_ERRORS_PER_AFFECTED_BLOCK
    ),
    physical_qubits_per_block: int = (
        STEANE_PHYSICAL_QUBITS_PER_BLOCK
    ),
) -> tuple[int, ...]:
    """
    Create a deterministic multi-qubit error-position pattern.

    The returned tuple contains physical-qubit indexes inside one
    Steane block. The pattern is intended for scenario simulation only.

    Example:

        (0, 1)

    represents errors on the first and second physical qubits.
    """

    validate_positive_integer(
        "physical_qubits_per_block",
        physical_qubits_per_block,
    )

    validate_positive_integer(
        "errors_per_block",
        errors_per_block,
    )

    if errors_per_block <= STEANE_CORRECTION_CAPABILITY:
        raise ValueError(
            "An uncorrectable demonstration requires more than "
            f"{STEANE_CORRECTION_CAPABILITY} physical-qubit error "
            "per logical block."
        )

    if errors_per_block > physical_qubits_per_block:
        raise ValueError(
            "errors_per_block cannot exceed the number of physical "
            "qubits in one logical block."
        )

    return tuple(range(errors_per_block))


def select_affected_logical_blocks(
    *,
    total_logical_blocks: int,
    affected_block_count: int,
    start_index: int = 0,
) -> tuple[int, ...]:
    """
    Select deterministic logical-block indexes for error injection.

    The real quantum simulator may randomize block selection using the
    configured scenario seed. This helper provides reproducible indexes
    for dashboard and unit-test use.
    """

    validate_positive_integer(
        "total_logical_blocks",
        total_logical_blocks,
    )

    validate_positive_integer(
        "affected_block_count",
        affected_block_count,
    )

    validate_nonnegative_integer(
        "start_index",
        start_index,
    )

    if affected_block_count > total_logical_blocks:
        raise ValueError(
            "affected_block_count cannot exceed "
            "total_logical_blocks."
        )

    if start_index >= total_logical_blocks:
        raise ValueError(
            "start_index must be inside the logical-block range."
        )

    if (
        start_index + affected_block_count
        > total_logical_blocks
    ):
        raise ValueError(
            "The selected affected-block range exceeds the "
            "logical-block sequence."
        )

    return tuple(
        range(
            start_index,
            start_index + affected_block_count,
        )
    )


def get_scenario_summary(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    forced_uncorrectable_blocks: int = (
        DEFAULT_FORCED_UNCORRECTABLE_BLOCKS
    ),
    errors_per_affected_block: int = (
        DEFAULT_ERRORS_PER_AFFECTED_BLOCK
    ),
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return dashboard-friendly fault information."""

    config = app_config or get_config()

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        forced_uncorrectable_blocks=(
            forced_uncorrectable_blocks
        ),
        errors_per_affected_block=(
            errors_per_affected_block
        ),
        app_config=config,
    )

    affected_blocks = select_affected_logical_blocks(
        total_logical_blocks=(
            config.quantum.total_logical_blocks
        ),
        affected_block_count=(
            forced_uncorrectable_blocks
        ),
    )

    error_pattern = create_uncorrectable_error_pattern(
        errors_per_block=errors_per_affected_block
    )

    summary = scenario.dashboard_summary()

    summary.update(
        {
            "failure_type": (
                "uncorrectable_quantum_error"
            ),
            "failure_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "steane_code": "[[7,1,3]]",
            "correction_capability": (
                STEANE_CORRECTION_CAPABILITY
            ),
            "errors_per_affected_block": (
                errors_per_affected_block
            ),
            "forced_uncorrectable_blocks": (
                forced_uncorrectable_blocks
            ),
            "affected_logical_block_indexes": list(
                affected_blocks
            ),
            "physical_error_positions": list(
                error_pattern
            ),
            "decoder_success_expected": False,
            "all_logical_blocks_recovered_expected": False,
            "expected_decision": EXPECTED_DECISION,
            "gp_evaluation_required": False,
            "retry_allowed": False,
            "expected_detection_sources": [
                "steane_syndrome",
                "physical_error_weight",
                "decoder_status",
                "uncorrectable_block_count",
                "logical_payload_reconstruction",
                "kmac_tag_verification",
            ],
        }
    )

    return summary


def get_attempt_configuration(
    attempt_number: int = 1,
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    forced_uncorrectable_blocks: int = (
        DEFAULT_FORCED_UNCORRECTABLE_BLOCKS
    ),
    errors_per_affected_block: int = (
        DEFAULT_ERRORS_PER_AFFECTED_BLOCK
    ),
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return effective uncorrectable-error conditions."""

    config = app_config or get_config()

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        forced_uncorrectable_blocks=(
            forced_uncorrectable_blocks
        ),
        errors_per_affected_block=(
            errors_per_affected_block
        ),
        app_config=config,
    )

    attempt = scenario.for_attempt(
        attempt_number=attempt_number,
        app_config=config,
    )

    affected_blocks = select_affected_logical_blocks(
        total_logical_blocks=(
            config.quantum.total_logical_blocks
        ),
        affected_block_count=(
            forced_uncorrectable_blocks
        ),
    )

    error_pattern = create_uncorrectable_error_pattern(
        errors_per_block=errors_per_affected_block
    )

    result = attempt.to_dictionary()

    result.update(
        {
            "steane_code": "[[7,1,3]]",
            "forced_uncorrectable_blocks": (
                forced_uncorrectable_blocks
            ),
            "errors_per_affected_block": (
                errors_per_affected_block
            ),
            "affected_logical_block_indexes": list(
                affected_blocks
            ),
            "physical_error_positions": list(
                error_pattern
            ),
            "decoder_success_expected": False,
            "logical_payload_valid_expected": False,
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


def create_error_failure_plan(
    *,
    forced_uncorrectable_blocks: int = (
        DEFAULT_FORCED_UNCORRECTABLE_BLOCKS
    ),
    errors_per_affected_block: int = (
        DEFAULT_ERRORS_PER_AFFECTED_BLOCK
    ),
    app_config: ApplicationConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Return the controlled Steane-decoding failure sequence.
    """

    config = app_config or get_config()

    validate_uncorrectable_error_parameters(
        forced_uncorrectable_blocks=(
            forced_uncorrectable_blocks
        ),
        errors_per_affected_block=(
            errors_per_affected_block
        ),
        app_config=config,
    )

    affected_blocks = select_affected_logical_blocks(
        total_logical_blocks=(
            config.quantum.total_logical_blocks
        ),
        affected_block_count=(
            forced_uncorrectable_blocks
        ),
    )

    error_pattern = create_uncorrectable_error_pattern(
        errors_per_block=errors_per_affected_block
    )

    return [
        {
            "step": 1,
            "actor": "mobile_station",
            "operation": (
                "prepare_authentication_tag"
            ),
            "kmac_tag_valid": True,
            "logical_payload_blocks": (
                config.quantum.logical_payload_blocks
            ),
        },
        {
            "step": 2,
            "actor": "mobile_station",
            "operation": (
                "steane_encode_logical_blocks"
            ),
            "steane_code": "[[7,1,3]]",
            "total_logical_blocks": (
                config.quantum.total_logical_blocks
            ),
            "total_physical_qubits": (
                config.quantum.total_physical_qubits
            ),
        },
        {
            "step": 3,
            "actor": "quantum_channel",
            "operation": (
                "inject_multi_qubit_errors"
            ),
            "affected_logical_blocks": list(
                affected_blocks
            ),
            "errors_per_affected_block": (
                errors_per_affected_block
            ),
            "physical_error_positions": list(
                error_pattern
            ),
        },
        {
            "step": 4,
            "actor": "authentication_server",
            "operation": (
                "measure_steane_syndromes"
            ),
            "nonzero_syndromes_expected": True,
            "multiple_error_patterns_expected": True,
        },
        {
            "step": 5,
            "actor": "authentication_server",
            "operation": (
                "decode_logical_blocks"
            ),
            "decoder_success_expected": False,
            "uncorrectable_block_count_expected": (
                forced_uncorrectable_blocks
            ),
        },
        {
            "step": 6,
            "actor": "authentication_server",
            "operation": (
                "validate_quantum_decoding_result"
            ),
            "validation_result": "fail",
            "expected_decision": EXPECTED_DECISION,
        },
        {
            "step": 7,
            "actor": "authentication_server",
            "operation": "stop_authentication",
            "accepted": False,
            "gp_evaluation_started": False,
            "retry_requested": False,
        },
    ]


def validate_uncorrectable_error_result(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate correct rejection of uncorrectable Steane errors.

    A valid result must show:

    - Authentication rejection
    - At least one uncorrectable logical block or decoder failure
    - No successful authentication
    - No GP override
    - No retry
    """

    if not isinstance(result, Mapping):
        raise TypeError(
            "result must be a mapping."
        )

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

    decoder_performed = read_boolean(
        result,
        (
            "decoder_performed",
            "steane_decoding_performed",
            "quantum_decoding_performed",
        ),
        default=False,
    )

    decoder_success = read_boolean(
        result,
        (
            "decoder_success",
            "steane_decoding_success",
            "quantum_decoding_success",
        ),
        default=True,
    )

    uncorrectable_blocks = safe_nonnegative_integer(
        result.get(
            "uncorrectable_blocks",
            result.get(
                "uncorrectable_block_count",
                0,
            ),
        )
    )

    logical_payload_valid = read_boolean(
        result,
        (
            "logical_payload_valid",
            "decoded_payload_valid",
            "payload_reconstruction_valid",
        ),
        default=True,
    )

    kmac_valid = read_boolean(
        result,
        (
            "kmac_valid",
            "tag_valid",
            "authentication_tag_valid",
        ),
        default=True,
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
        "rejected_quantum",
        "rejected_decoder",
        "rejected_uncorrectable",
        "rejected_uncorrectable_error",
        "rejected_quantum_error",
        "failed",
    }

    valid_reasons = {
        "uncorrectable_quantum_error",
        "uncorrectable_steane_error",
        "steane_decoding_failed",
        "decoder_failure",
        "uncorrectable_logical_block",
        "multiple_physical_qubit_errors",
        "logical_payload_reconstruction_failed",
    }

    decoder_failure_evidence = (
        (
            decoder_performed
            and not decoder_success
        )
        or uncorrectable_blocks > 0
        or not logical_payload_valid
        or reason in valid_reasons
    )

    return (
        decision in valid_decisions
        and decoder_failure_evidence
        and not accepted
        and not gp_started
        and not retry_used
        and (
            uncorrectable_blocks > 0
            or not decoder_success
            or not logical_payload_valid
            or not kmac_valid
        )
    )


def validate_uncorrectable_error_parameters(
    *,
    forced_uncorrectable_blocks: int,
    errors_per_affected_block: int,
    app_config: ApplicationConfig,
) -> None:
    """Validate controlled Steane failure parameters."""

    validate_positive_integer(
        "forced_uncorrectable_blocks",
        forced_uncorrectable_blocks,
    )

    validate_positive_integer(
        "errors_per_affected_block",
        errors_per_affected_block,
    )

    if (
        forced_uncorrectable_blocks
        > app_config.quantum.total_logical_blocks
    ):
        raise ValueError(
            "forced_uncorrectable_blocks cannot exceed the "
            "configured total logical blocks."
        )

    if (
        errors_per_affected_block
        <= STEANE_CORRECTION_CAPABILITY
    ):
        raise ValueError(
            "errors_per_affected_block must exceed the Steane "
            "single-qubit correction capability."
        )

    if (
        errors_per_affected_block
        > STEANE_PHYSICAL_QUBITS_PER_BLOCK
    ):
        raise ValueError(
            "errors_per_affected_block cannot exceed seven "
            "physical qubits."
        )


def validate_positive_integer(
    name: str,
    value: int,
) -> int:
    """Validate and return a positive integer."""

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value < 1:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return value


def validate_nonnegative_integer(
    name: str,
    value: int,
) -> int:
    """Validate and return a nonnegative integer."""

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return value


def safe_nonnegative_integer(
    value: Any,
    default: int = 0,
) -> int:
    """Convert an integer-like value to a nonnegative integer."""

    if value is None or isinstance(value, bool):
        return default

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default

    if not converted.is_integer():
        return default

    integer_value = int(converted)

    if integer_value < 0:
        return default

    return integer_value


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
                "valid",
                "passed",
                "successful",
                "performed",
            }:
                return True

            if normalized_value in {
                "false",
                "no",
                "0",
                "invalid",
                "failed",
                "unsuccessful",
                "not_performed",
            }:
                return False

    return default


UNCORRECTABLE_QUANTUM_ERROR_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run uncorrectable-error consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert (
        scenario.name
        == "uncorrectable_quantum_error"
    )

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
        scenario.channel
        .forced_uncorrectable_blocks
        == DEFAULT_FORCED_UNCORRECTABLE_BLOCKS
    )

    assert (
        scenario.channel
        .forced_lost_check_blocks
        == 0
    )

    assert (
        scenario.channel.loss_rate
        < config.quantum.maximum_loss_rate
    )

    error_pattern = (
        create_uncorrectable_error_pattern()
    )

    assert error_pattern == (0, 1)

    affected_blocks = (
        select_affected_logical_blocks(
            total_logical_blocks=(
                config.quantum.total_logical_blocks
            ),
            affected_block_count=(
                DEFAULT_FORCED_UNCORRECTABLE_BLOCKS
            ),
        )
    )

    assert affected_blocks == (0, 1, 2)

    attempt = scenario.for_attempt(
        1,
        config,
    )

    assert (
        attempt.channel
        .forced_uncorrectable_blocks
        == DEFAULT_FORCED_UNCORRECTABLE_BLOCKS
    )

    assert (
        attempt.force_retry_gray_zone
        is False
    )

    failure_plan = create_error_failure_plan(
        app_config=config
    )

    assert len(failure_plan) == 7

    assert (
        failure_plan[2]
        ["errors_per_affected_block"]
        == DEFAULT_ERRORS_PER_AFFECTED_BLOCK
    )

    assert (
        failure_plan[4]
        ["decoder_success_expected"]
        is False
    )

    assert (
        failure_plan[5]["expected_decision"]
        == EXPECTED_DECISION
    )

    valid_decoder_failure = (
        validate_uncorrectable_error_result(
            {
                "decision": (
                    "rejected_uncorrectable_error"
                ),
                "reason": (
                    "steane_decoding_failed"
                ),
                "steane_decoding_performed": True,
                "steane_decoding_success": False,
                "uncorrectable_blocks": 3,
                "logical_payload_valid": False,
                "kmac_valid": False,
                "gp_evaluation_started": False,
                "retry_requested": False,
                "accepted": False,
            }
        )
    )

    assert valid_decoder_failure is True

    valid_uncorrectable_count = (
        validate_uncorrectable_error_result(
            {
                "decision": (
                    "rejected_quantum_error"
                ),
                "reason": (
                    "uncorrectable_logical_block"
                ),
                "decoder_performed": True,
                "decoder_success": False,
                "uncorrectable_block_count": 2,
                "decoded_payload_valid": False,
                "tag_valid": False,
                "gp_prediction_performed": False,
                "retry_used": False,
                "accepted": False,
            }
        )
    )

    assert valid_uncorrectable_count is True

    invalid_retry_result = (
        validate_uncorrectable_error_result(
            {
                "decision": (
                    "rejected_uncorrectable"
                ),
                "reason": "decoder_failure",
                "decoder_performed": True,
                "decoder_success": False,
                "uncorrectable_blocks": 3,
                "logical_payload_valid": False,
                "kmac_valid": False,
                "gp_evaluation_started": False,
                "retry_requested": True,
                "accepted": False,
            }
        )
    )

    assert invalid_retry_result is False

    invalid_gp_result = (
        validate_uncorrectable_error_result(
            {
                "decision": (
                    "rejected_uncorrectable"
                ),
                "reason": "decoder_failure",
                "decoder_performed": True,
                "decoder_success": False,
                "uncorrectable_blocks": 3,
                "logical_payload_valid": False,
                "kmac_valid": False,
                "gp_prediction_performed": True,
                "retry_requested": False,
                "accepted": False,
            }
        )
    )

    assert invalid_gp_result is False

    invalid_accept_result = (
        validate_uncorrectable_error_result(
            {
                "decision": "accepted",
                "decoder_performed": True,
                "decoder_success": True,
                "uncorrectable_blocks": 0,
                "logical_payload_valid": True,
                "kmac_valid": True,
                "gp_evaluation_started": False,
                "retry_requested": False,
                "accepted": True,
            }
        )
    )

    assert invalid_accept_result is False

    summary = get_scenario_summary(
        app_config=config
    )

    assert summary["attack_enabled"] is False

    assert (
        summary["decoder_success_expected"]
        is False
    )

    assert (
        summary["forced_uncorrectable_blocks"]
        == DEFAULT_FORCED_UNCORRECTABLE_BLOCKS
    )

    assert (
        summary["expected_decision"]
        == "rejected_uncorrectable_error"
    )

    assert summary["retry_allowed"] is False

    print(
        "FT-QuPAP uncorrectable-quantum-error "
        "scenario self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_FORCED_UNCORRECTABLE_BLOCKS",
    "DEFAULT_ERRORS_PER_AFFECTED_BLOCK",
    "EXPECTED_DECISION",
    "EXPECTED_REJECTION_STAGE",
    "STEANE_PHYSICAL_QUBITS_PER_BLOCK",
    "STEANE_CODE_DISTANCE",
    "STEANE_CORRECTION_CAPABILITY",
    "UNCORRECTABLE_QUANTUM_ERROR_CONFIG",
    "build_scenario",
    "create_scenario",
    "create_uncorrectable_error_pattern",
    "select_affected_logical_blocks",
    "get_scenario_summary",
    "get_attempt_configuration",
    "create_error_failure_plan",
    "validate_uncorrectable_error_result",
    "validate_uncorrectable_error_parameters",
]


if __name__ == "__main__":
    run_self_test()