"""
FT-QuPAP Tampered ML-KEM Ciphertext Scenario
============================================

Defines a controlled ciphertext-integrity attack against the ML-KEM-768
bootstrapping phase of FT-QuPAP v5.1.

Normal ML-KEM process:

1. The Authentication Server supplies its authenticated ML-KEM-768
   public key.

2. The Mobile Station encapsulates a shared secret and produces an
   ML-KEM ciphertext.

3. The Mobile Station derives transcript-bound authentication and
   control keys from its shared secret.

4. The Authentication Server decapsulates the received ciphertext and
   derives the same shared secret.

Attack process:

1. A classical network attacker intercepts the ML-KEM ciphertext.

2. At least one ciphertext bit is changed before the ciphertext reaches
   the Authentication Server.

3. The Authentication Server performs ML-KEM decapsulation using the
   modified ciphertext.

Possible secure outcomes:

- The implementation reports a decapsulation failure, or
- Implicit rejection produces a different shared secret.

When a different shared secret is produced, the Authentication Server
derives different authentication and control keys. Consequently, the
received quantum authentication payload cannot produce a valid
transcript-bound KMAC256 tag.

Expected result:

    REJECTED_MLKEM_CIPHERTEXT

The attack must never result in successful authentication or retry.
Gaussian Process evaluation is not required because cryptographic
verification provides deterministic rejection evidence.
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
    "tampered_mlkem_ciphertext"
)

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Tampered ML-KEM Ciphertext"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "A classical network attacker changes one or more bits of the "
    "ML-KEM-768 ciphertext before server decapsulation. The server "
    "must reject the session through decapsulation failure or a "
    "downstream transcript-bound KMAC verification failure."
)

DEFAULT_RANDOM_SEED: Final[int] = 9304

DEFAULT_TAMPER_BYTE_INDEX: Final[int] = 0
DEFAULT_TAMPER_BIT_MASK: Final[int] = 0x01

EXPECTED_DECISION: Final[str] = (
    "rejected_mlkem_ciphertext"
)

EXPECTED_PRIMARY_REJECTION_STAGE: Final[str] = (
    "mlkem_decapsulation"
)

EXPECTED_FALLBACK_REJECTION_STAGE: Final[str] = (
    "kmac_tag_verification"
)


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the tampered ML-KEM ciphertext scenario.

    Args:
        context:
            Network environment such as urban, suburban, or rural.

        random_seed:
            Reproducible demonstration seed.

        app_config:
            Optional FT-QuPAP application configuration.

    Returns:
        Validated tampered-ciphertext scenario configuration.
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
            # Normal channel conditions are configured because the
            # attack targets the classical ML-KEM ciphertext.
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

            # Main attack switch.
            tamper_mlkem_ciphertext=True,

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

        # The cryptographic failure is sufficient for rejection.
        # GP evaluation may be skipped even if later quantum evidence
        # becomes available.
        gp_evaluation_expected=False,
        notes=(
            "The server ML-DSA signature remains valid.",
            "The Mobile Station uses the authentic ML-KEM public key.",
            "The Mobile Station creates a valid original ciphertext.",
            "The attacker modifies the ciphertext during transmission.",
            "The Authentication Server decapsulates the modified value.",
            "Decapsulation may fail explicitly or use implicit rejection.",
            "A changed shared secret causes derived-key disagreement.",
            "Derived-key disagreement causes KMAC verification failure.",
            "Authentication must be rejected deterministically.",
            "Retry is forbidden for ciphertext-integrity failure.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": (
                "tampered_mlkem_ciphertext"
            ),
            "attack_enabled": True,
            "attacker": (
                "classical_network_attacker"
            ),
            "attack_stage": (
                "mlkem_ciphertext_transmission"
            ),
            "attacked_component": (
                "mlkem_ciphertext"
            ),
            "mlkem_parameter_set": (
                config.cryptography
                .ml_kem_parameter_set
            ),
            "default_tamper_byte_index": (
                DEFAULT_TAMPER_BYTE_INDEX
            ),
            "default_tamper_bit_mask": (
                DEFAULT_TAMPER_BIT_MASK
            ),
            "expected_decision": EXPECTED_DECISION,
            "expected_primary_rejection_stage": (
                EXPECTED_PRIMARY_REJECTION_STAGE
            ),
            "expected_fallback_rejection_stage": (
                EXPECTED_FALLBACK_REJECTION_STAGE
            ),
            "server_signature_valid": True,
            "original_ciphertext_valid": True,
            "received_ciphertext_modified": True,
            "shared_secret_match_expected": False,
            "authentication_key_match_expected": False,
            "control_key_match_expected": False,
            "kmac_verification_expected": False,
            "deterministic_rejection_expected": True,
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
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """Alias for building the ciphertext-tampering scenario."""

    return build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )


def tamper_ciphertext(
    ciphertext: bytes,
    *,
    byte_index: int = DEFAULT_TAMPER_BYTE_INDEX,
    bit_mask: int = DEFAULT_TAMPER_BIT_MASK,
) -> bytes:
    """
    Return a modified copy of an ML-KEM ciphertext.

    The selected ciphertext byte is XORed with a nonzero bit mask.

    Args:
        ciphertext:
            Original ML-KEM ciphertext.

        byte_index:
            Index of the ciphertext byte to modify.

        bit_mask:
            Integer mask in the range 1 to 255.

    Returns:
        Modified ciphertext bytes.

    Raises:
        TypeError:
            If the ciphertext or mutation parameters have invalid types.

        ValueError:
            If the ciphertext is empty, the index is outside the
            ciphertext, or the mask is zero or greater than 255.
    """

    if not isinstance(ciphertext, bytes):
        raise TypeError(
            "ciphertext must be bytes."
        )

    if not ciphertext:
        raise ValueError(
            "ciphertext cannot be empty."
        )

    if isinstance(byte_index, bool) or not isinstance(
        byte_index,
        int,
    ):
        raise TypeError(
            "byte_index must be an integer."
        )

    if not 0 <= byte_index < len(ciphertext):
        raise ValueError(
            "byte_index is outside the ciphertext."
        )

    if isinstance(bit_mask, bool) or not isinstance(
        bit_mask,
        int,
    ):
        raise TypeError(
            "bit_mask must be an integer."
        )

    if not 1 <= bit_mask <= 0xFF:
        raise ValueError(
            "bit_mask must be between 1 and 255."
        )

    modified = bytearray(ciphertext)
    modified[byte_index] ^= bit_mask

    tampered = bytes(modified)

    if tampered == ciphertext:
        raise RuntimeError(
            "Ciphertext tampering did not change the input."
        )

    if len(tampered) != len(ciphertext):
        raise RuntimeError(
            "Ciphertext tampering changed its length."
        )

    return tampered


def get_scenario_summary(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return dashboard-friendly ciphertext-attack information."""

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )

    summary = scenario.dashboard_summary()

    summary.update(
        {
            "attacker": (
                "classical_network_attacker"
            ),
            "attack_stage": (
                "mlkem_ciphertext_transmission"
            ),
            "attack_operation": (
                "flip_ciphertext_bit"
            ),
            "attacked_component": (
                "ML-KEM-768 ciphertext"
            ),
            "expected_decision": EXPECTED_DECISION,
            "possible_rejection_stages": [
                EXPECTED_PRIMARY_REJECTION_STAGE,
                EXPECTED_FALLBACK_REJECTION_STAGE,
            ],
            "server_signature_valid": True,
            "original_ciphertext_valid": True,
            "received_ciphertext_valid": False,
            "shared_secret_match_expected": False,
            "session_key_match_expected": False,
            "retry_allowed": False,
            "gp_evaluation_required": False,
            "expected_detection_sources": [
                "mlkem_decapsulation_status",
                "implicit_rejection_secret",
                "transcript_bound_key_derivation",
                "control_schedule_mismatch",
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
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return effective ciphertext-attack conditions."""

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
            "server_signature_valid": True,
            "original_ciphertext_valid": True,
            "ciphertext_tampering_enabled": True,
            "tamper_byte_index": (
                DEFAULT_TAMPER_BYTE_INDEX
            ),
            "tamper_bit_mask": (
                DEFAULT_TAMPER_BIT_MASK
            ),
            "expected_attempt_decision": (
                EXPECTED_DECISION
            ),
            "expected_primary_rejection_stage": (
                EXPECTED_PRIMARY_REJECTION_STAGE
            ),
            "expected_fallback_rejection_stage": (
                EXPECTED_FALLBACK_REJECTION_STAGE
            ),
            "shared_secret_match_expected": False,
            "derived_keys_match_expected": False,
            "retry_allowed": False,
        }
    )

    return result


def create_ciphertext_attack_plan() -> list[dict[str, Any]]:
    """
    Return the controlled ML-KEM ciphertext attack sequence.
    """

    return [
        {
            "step": 1,
            "actor": "authentication_server",
            "operation": (
                "send_authenticated_mlkem_public_key"
            ),
            "server_signature_valid": True,
            "expected_result": (
                "trusted_public_key_received"
            ),
        },
        {
            "step": 2,
            "actor": "mobile_station",
            "operation": "mlkem_encapsulation",
            "parameter_set": "ML-KEM-768",
            "original_ciphertext_valid": True,
            "shared_secret_generated": True,
        },
        {
            "step": 3,
            "actor": (
                "classical_network_attacker"
            ),
            "operation": (
                "modify_mlkem_ciphertext"
            ),
            "byte_index": (
                DEFAULT_TAMPER_BYTE_INDEX
            ),
            "bit_mask": (
                DEFAULT_TAMPER_BIT_MASK
            ),
            "expected_result": (
                "ciphertext_changed"
            ),
        },
        {
            "step": 4,
            "actor": "authentication_server",
            "operation": "mlkem_decapsulation",
            "received_ciphertext_modified": True,
            "possible_result": [
                "explicit_decapsulation_failure",
                "implicit_rejection_secret",
            ],
        },
        {
            "step": 5,
            "actor": "authentication_server",
            "operation": (
                "verify_derived_authentication_evidence"
            ),
            "shared_secret_match_expected": False,
            "kmac_tag_valid_expected": False,
            "expected_decision": EXPECTED_DECISION,
        },
        {
            "step": 6,
            "actor": "authentication_server",
            "operation": "stop_authentication",
            "accepted": False,
            "retry_requested": False,
        },
    ]


def validate_tampered_ciphertext_result(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate correct rejection of a tampered ML-KEM ciphertext.

    A valid result must show:

    - Authentication rejection
    - Ciphertext modification or ML-KEM/key mismatch evidence
    - No successful authentication
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

    ciphertext_modified = read_boolean(
        result,
        (
            "ciphertext_modified",
            "mlkem_ciphertext_modified",
            "ciphertext_tampered",
        ),
        default=False,
    )

    decapsulation_performed = read_boolean(
        result,
        (
            "decapsulation_performed",
            "mlkem_decapsulation_performed",
        ),
        default=False,
    )

    decapsulation_valid = read_boolean(
        result,
        (
            "decapsulation_valid",
            "mlkem_decapsulation_valid",
            "mlkem_ciphertext_valid",
        ),
        default=True,
    )

    shared_secret_match = read_boolean(
        result,
        (
            "shared_secret_match",
            "shared_secrets_match",
        ),
        default=True,
    )

    authentication_keys_match = read_boolean(
        result,
        (
            "authentication_keys_match",
            "derived_keys_match",
            "session_keys_match",
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
        "rejected_ciphertext",
        "rejected_mlkem",
        "rejected_mlkem_ciphertext",
        "rejected_decapsulation",
        "rejected_key_mismatch",
        "rejected_tag",
        "failed",
    }

    valid_reasons = {
        "tampered_mlkem_ciphertext",
        "invalid_mlkem_ciphertext",
        "mlkem_decapsulation_failed",
        "decapsulation_failure",
        "implicit_rejection",
        "shared_secret_mismatch",
        "derived_key_mismatch",
        "kmac_verification_failed",
    }

    rejected = decision in valid_decisions

    explicit_decapsulation_failure = (
        decapsulation_performed
        and not decapsulation_valid
    )

    implicit_rejection_evidence = (
        decapsulation_performed
        and (
            not shared_secret_match
            or not authentication_keys_match
            or not kmac_valid
        )
    )

    attack_evidence = (
        ciphertext_modified
        or reason in valid_reasons
    )

    cryptographic_failure = (
        explicit_decapsulation_failure
        or implicit_rejection_evidence
        or reason in valid_reasons
    )

    return (
        rejected
        and attack_evidence
        and cryptographic_failure
        and not accepted
        and not retry_used
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
                "matched",
                "performed",
            }:
                return True

            if normalized_value in {
                "false",
                "no",
                "0",
                "invalid",
                "failed",
                "mismatched",
                "not_performed",
            }:
                return False

    return default


TAMPERED_MLKEM_CIPHERTEXT_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run tampered-ciphertext consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert (
        scenario.name
        == "tampered_mlkem_ciphertext"
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
        .tamper_mlkem_ciphertext
        is True
    )

    assert scenario.tampering.enabled is True
    assert scenario.retry.enabled is False

    assert (
        scenario.gp_evaluation_expected
        is False
    )

    original_ciphertext = bytes(
        range(32)
    )

    modified_ciphertext = tamper_ciphertext(
        original_ciphertext
    )

    assert (
        modified_ciphertext
        != original_ciphertext
    )

    assert (
        len(modified_ciphertext)
        == len(original_ciphertext)
    )

    assert (
        modified_ciphertext[0]
        == original_ciphertext[0] ^ 0x01
    )

    attempt = scenario.for_attempt(
        1,
        config,
    )

    assert (
        attempt.tampering
        .tamper_mlkem_ciphertext
        is True
    )

    assert (
        attempt.force_retry_gray_zone
        is False
    )

    attack_plan = create_ciphertext_attack_plan()

    assert len(attack_plan) == 6

    assert (
        attack_plan[2]["operation"]
        == "modify_mlkem_ciphertext"
    )

    assert (
        attack_plan[4]["expected_decision"]
        == EXPECTED_DECISION
    )

    explicit_failure_result = (
        validate_tampered_ciphertext_result(
            {
                "decision": (
                    "rejected_mlkem_ciphertext"
                ),
                "reason": (
                    "mlkem_decapsulation_failed"
                ),
                "ciphertext_modified": True,
                "decapsulation_performed": True,
                "decapsulation_valid": False,
                "shared_secret_match": False,
                "authentication_keys_match": False,
                "kmac_valid": False,
                "retry_used": False,
                "accepted": False,
            }
        )
    )

    assert explicit_failure_result is True

    implicit_rejection_result = (
        validate_tampered_ciphertext_result(
            {
                "decision": (
                    "rejected_key_mismatch"
                ),
                "reason": (
                    "shared_secret_mismatch"
                ),
                "mlkem_ciphertext_modified": True,
                "mlkem_decapsulation_performed": True,
                "mlkem_decapsulation_valid": True,
                "shared_secrets_match": False,
                "derived_keys_match": False,
                "tag_valid": False,
                "retry_requested": False,
                "accepted": False,
            }
        )
    )

    assert implicit_rejection_result is True

    invalid_retry_result = (
        validate_tampered_ciphertext_result(
            {
                "decision": "rejected_mlkem",
                "reason": "decapsulation_failure",
                "ciphertext_tampered": True,
                "decapsulation_performed": True,
                "decapsulation_valid": False,
                "shared_secret_match": False,
                "kmac_valid": False,
                "retry_requested": True,
                "accepted": False,
            }
        )
    )

    assert invalid_retry_result is False

    invalid_accept_result = (
        validate_tampered_ciphertext_result(
            {
                "decision": "accepted",
                "ciphertext_modified": True,
                "decapsulation_performed": True,
                "decapsulation_valid": True,
                "shared_secret_match": True,
                "authentication_keys_match": True,
                "kmac_valid": True,
                "retry_used": False,
                "accepted": True,
            }
        )
    )

    assert invalid_accept_result is False

    summary = get_scenario_summary(
        app_config=config
    )

    assert summary["attack_enabled"] is True

    assert (
        summary["expected_decision"]
        == "rejected_mlkem_ciphertext"
    )

    assert (
        summary["shared_secret_match_expected"]
        is False
    )

    assert summary["retry_allowed"] is False

    print(
        "FT-QuPAP tampered-ML-KEM-ciphertext "
        "scenario self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TAMPER_BYTE_INDEX",
    "DEFAULT_TAMPER_BIT_MASK",
    "EXPECTED_DECISION",
    "EXPECTED_PRIMARY_REJECTION_STAGE",
    "EXPECTED_FALLBACK_REJECTION_STAGE",
    "TAMPERED_MLKEM_CIPHERTEXT_CONFIG",
    "build_scenario",
    "create_scenario",
    "tamper_ciphertext",
    "get_scenario_summary",
    "get_attempt_configuration",
    "create_ciphertext_attack_plan",
    "validate_tampered_ciphertext_result",
]


if __name__ == "__main__":
    run_self_test()