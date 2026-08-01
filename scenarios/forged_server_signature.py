"""
FT-QuPAP Forged Server Signature Scenario
=========================================

Defines a controlled forged-server-signature attack against the
FT-QuPAP v5.1 classical bootstrapping phase.

Normal server-package verification:

1. The Authentication Server prepares a server package containing
   session-bound protocol information and its ML-KEM-768 public key.

2. The server signs the canonical serialized package using ML-DSA-65.

3. The Mobile Station verifies the signature using the trusted server
   ML-DSA public key.

Attack process:

1. An attacker replaces or modifies the server package.

2. The attacker supplies a random, corrupted, or otherwise invalid
   ML-DSA signature.

3. The Mobile Station performs deterministic ML-DSA verification.

4. Signature verification fails.

Expected result:

    REJECTED_SERVER_SIGNATURE

The Mobile Station must stop immediately. It must not continue to:

- ML-KEM encapsulation
- Shared-secret derivation
- Authentication-key derivation
- Control-key derivation
- KMAC tag generation
- Steane encoding
- Quantum transmission
- GP attack evaluation
- Retry processing

A forged server signature is a deterministic cryptographic failure and
must never be treated as benign channel noise.
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


SCENARIO_NAME: Final[str] = "forged_server_signature"

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Forged Server Signature"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "The Mobile Station receives a server package carrying an invalid "
    "ML-DSA-65 signature. The package must be rejected before "
    "ML-KEM encapsulation, session-key derivation, or quantum "
    "authentication begins."
)

DEFAULT_RANDOM_SEED: Final[int] = 9302

EXPECTED_DECISION: Final[str] = (
    "rejected_server_signature"
)

EXPECTED_REJECTION_STAGE: Final[str] = (
    "server_package_verification"
)

EXPECTED_SIGNATURE_VALID: Final[bool] = False


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the forged-server-signature scenario.

    Args:
        context:
            Network context such as urban, suburban, or rural.

        random_seed:
            Reproducible demonstration seed.

        app_config:
            Optional FT-QuPAP application configuration.

    Returns:
        Validated forged-signature scenario configuration.
    """

    config = app_config or get_config()

    scenario = create_scenario_config(
        name=SCENARIO_NAME,
        display_name=SCENARIO_DISPLAY_NAME,
        description=SCENARIO_DESCRIPTION,
        category=(
            ScenarioCategory.CRYPTOGRAPHIC_ATTACK
        ),
        expected_outcome=ExpectedOutcome.REJECTED,
        context=context,
        random_seed=random_seed,
        channel=ChannelProfile(
            # These values represent an ordinary channel, but the
            # quantum channel should never be reached in this attack.
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

            # Main attack switch.
            forge_server_signature=True,

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

        # The Mobile Station must reject the package before any
        # quantum evidence or GP feature vector exists.
        gp_evaluation_expected=False,
        notes=(
            "The trusted server ML-DSA public key remains unchanged.",
            "The received server signature is intentionally invalid.",
            "Canonical package serialization must be verified.",
            "Signature verification must use the trusted public key.",
            "ML-KEM encapsulation must not begin after failure.",
            "No session keys should be derived.",
            "No KMAC authentication tag should be generated.",
            "No logical or physical qubits should be prepared.",
            "The GP detector is not used for this deterministic failure.",
            "Retry is forbidden for an invalid server signature.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": (
                "forged_mldsa_server_signature"
            ),
            "attack_enabled": True,
            "attack_stage": (
                "server_package_verification"
            ),
            "forged_component": (
                "server_package_signature"
            ),
            "signature_algorithm": (
                config.cryptography
                .ml_dsa_parameter_set
            ),
            "expected_signature_valid": (
                EXPECTED_SIGNATURE_VALID
            ),
            "expected_decision": EXPECTED_DECISION,
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "deterministic_rejection_expected": True,
            "mlkem_encapsulation_expected": False,
            "shared_secret_expected": False,
            "session_key_derivation_expected": False,
            "kmac_generation_expected": False,
            "quantum_payload_generation_expected": False,
            "quantum_transmission_expected": False,
            "qber_evaluation_expected": False,
            "gp_evaluation_expected": False,
            "retry_expected": False,
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
    """Alias for building the forged-signature scenario."""

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
            "attacker": "classical_network_attacker",
            "attack_stage": (
                "server_package_verification"
            ),
            "attacked_component": (
                "ml_dsa_server_signature"
            ),
            "signature_algorithm": (
                "ML-DSA-65"
            ),
            "expected_signature_valid": False,
            "expected_decision": EXPECTED_DECISION,
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "mlkem_encapsulation_expected": False,
            "session_key_derivation_expected": False,
            "quantum_transmission_expected": False,
            "gp_evaluation_expected": False,
            "retry_allowed": False,
            "expected_detection_sources": [
                "trusted_server_public_key",
                "canonical_server_package_bytes",
                "received_mldsa_signature",
                "mldsa_signature_verification",
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
    """Return effective forged-signature conditions."""

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
            "server_package_received": True,
            "trusted_server_key_available": True,
            "signature_present": True,
            "signature_valid": False,
            "expected_attempt_decision": (
                EXPECTED_DECISION
            ),
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "continue_to_mlkem_encapsulation": False,
            "continue_to_session_key_derivation": False,
            "continue_to_quantum_transmission": False,
            "continue_to_gp_evaluation": False,
            "retry_allowed": False,
        }
    )

    return result


def create_signature_attack_plan(
    *,
    package_modified: bool = False,
    signature_modified: bool = True,
) -> list[dict[str, Any]]:
    """
    Return the controlled forged-signature demonstration sequence.

    At least one of the signed package or signature must be modified.
    """

    if not isinstance(package_modified, bool):
        raise TypeError(
            "package_modified must be boolean."
        )

    if not isinstance(signature_modified, bool):
        raise TypeError(
            "signature_modified must be boolean."
        )

    if not package_modified and not signature_modified:
        raise ValueError(
            "The scenario must modify either the signed package "
            "or the signature."
        )

    return [
        {
            "step": 1,
            "actor": "authentication_server",
            "operation": (
                "create_and_sign_server_package"
            ),
            "signature_algorithm": "ML-DSA-65",
            "expected_result": (
                "valid_original_package"
            ),
        },
        {
            "step": 2,
            "actor": "attacker",
            "operation": (
                "tamper_with_server_package"
            ),
            "package_modified": package_modified,
            "signature_modified": signature_modified,
            "expected_result": (
                "signature_package_mismatch"
            ),
        },
        {
            "step": 3,
            "actor": "mobile_station",
            "operation": (
                "verify_server_signature"
            ),
            "trusted_public_key_used": True,
            "expected_signature_valid": False,
            "expected_decision": EXPECTED_DECISION,
        },
        {
            "step": 4,
            "actor": "mobile_station",
            "operation": "stop_authentication",
            "mlkem_encapsulation_started": False,
            "quantum_transmission_started": False,
            "retry_requested": False,
        },
    ]


def validate_forged_signature_result(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate whether the forged signature was correctly rejected.

    A valid result must indicate:

    - ML-DSA verification failure
    - Authentication rejection
    - No ML-KEM encapsulation
    - No quantum transmission
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

    signature_valid = read_boolean(
        result,
        (
            "signature_valid",
            "server_signature_valid",
            "mldsa_signature_valid",
        ),
        default=True,
    )

    signature_verification_performed = read_boolean(
        result,
        (
            "signature_verification_performed",
            "server_signature_checked",
            "mldsa_verification_performed",
        ),
        default=False,
    )

    mlkem_started = read_boolean(
        result,
        (
            "mlkem_encapsulation_started",
            "encapsulation_started",
        ),
        default=False,
    )

    quantum_started = read_boolean(
        result,
        (
            "quantum_transmission_started",
            "quantum_processing_started",
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

    valid_rejection_decisions = {
        "rejected",
        "rejected_server_signature",
        "rejected_signature",
        "rejected_mldsa",
        "failed",
    }

    valid_rejection_reasons = {
        "forged_server_signature",
        "invalid_server_signature",
        "signature_verification_failed",
        "mldsa_verification_failed",
        "server_signature_invalid",
    }

    rejected = decision in valid_rejection_decisions

    signature_failure_identified = (
        reason in valid_rejection_reasons
        or (
            signature_verification_performed
            and not signature_valid
        )
    )

    return (
        rejected
        and signature_failure_identified
        and not signature_valid
        and not accepted
        and not mlkem_started
        and not quantum_started
        and not retry_used
    )


def normalize_text(value: Any) -> str:
    """Normalize an optional value as lowercase text."""

    if value is None:
        return ""

    return str(value).strip().lower()


def read_boolean(
    mapping: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    default: bool,
) -> bool:
    """Read the first boolean-compatible field."""

    for field_name in field_names:
        if field_name not in mapping:
            continue

        value = mapping[field_name]

        if isinstance(value, bool):
            return value

        if isinstance(value, int) and value in {0, 1}:
            return bool(value)

        if isinstance(value, str):
            normalized_value = value.strip().lower()

            if normalized_value in {
                "true",
                "yes",
                "1",
                "valid",
                "passed",
            }:
                return True

            if normalized_value in {
                "false",
                "no",
                "0",
                "invalid",
                "failed",
            }:
                return False

    return default


FORGED_SERVER_SIGNATURE_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run forged-signature scenario consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert (
        scenario.name
        == "forged_server_signature"
    )

    assert (
        scenario.category
        == "cryptographic_attack"
    )

    assert scenario.expected_outcome == "rejected"
    assert scenario.attack_enabled is True
    assert scenario.retry_expected is False

    assert scenario.eve.enabled is False

    assert (
        scenario.tampering
        .forge_server_signature
        is True
    )

    assert scenario.tampering.enabled is True
    assert scenario.retry.enabled is False

    assert (
        scenario.gp_evaluation_expected
        is False
    )

    attempt = scenario.for_attempt(
        1,
        config,
    )

    assert (
        attempt.tampering
        .forge_server_signature
        is True
    )

    assert (
        attempt.force_retry_gray_zone
        is False
    )

    attack_plan = create_signature_attack_plan()

    assert len(attack_plan) == 4

    assert (
        attack_plan[2]["expected_signature_valid"]
        is False
    )

    assert (
        attack_plan[2]["expected_decision"]
        == EXPECTED_DECISION
    )

    valid_result = validate_forged_signature_result(
        {
            "decision": (
                "rejected_server_signature"
            ),
            "reason": (
                "signature_verification_failed"
            ),
            "signature_verification_performed": True,
            "signature_valid": False,
            "mlkem_encapsulation_started": False,
            "quantum_transmission_started": False,
            "retry_used": False,
            "accepted": False,
        }
    )

    assert valid_result is True

    invalid_retry_result = (
        validate_forged_signature_result(
            {
                "decision": "rejected_signature",
                "reason": "invalid_server_signature",
                "signature_verification_performed": True,
                "signature_valid": False,
                "mlkem_encapsulation_started": False,
                "quantum_transmission_started": False,
                "retry_requested": True,
                "accepted": False,
            }
        )
    )

    assert invalid_retry_result is False

    invalid_continuation_result = (
        validate_forged_signature_result(
            {
                "decision": "rejected_signature",
                "reason": "invalid_server_signature",
                "signature_verification_performed": True,
                "signature_valid": False,
                "mlkem_encapsulation_started": True,
                "quantum_transmission_started": False,
                "retry_used": False,
                "accepted": False,
            }
        )
    )

    assert invalid_continuation_result is False

    summary = get_scenario_summary(
        app_config=config
    )

    assert summary["attack_enabled"] is True

    assert (
        summary["expected_decision"]
        == "rejected_server_signature"
    )

    assert (
        summary["mlkem_encapsulation_expected"]
        is False
    )

    assert (
        summary["quantum_transmission_expected"]
        is False
    )

    assert summary["retry_allowed"] is False

    print(
        "FT-QuPAP forged-server-signature "
        "scenario self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "EXPECTED_DECISION",
    "EXPECTED_REJECTION_STAGE",
    "EXPECTED_SIGNATURE_VALID",
    "FORGED_SERVER_SIGNATURE_CONFIG",
    "build_scenario",
    "create_scenario",
    "get_scenario_summary",
    "get_attempt_configuration",
    "create_signature_attack_plan",
    "validate_forged_signature_result",
]


if __name__ == "__main__":
    run_self_test()