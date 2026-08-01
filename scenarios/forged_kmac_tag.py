"""
FT-QuPAP Forged KMAC Authentication Tag Scenario
=================================================

Defines a controlled forged-tag attack against the transcript-bound
KMAC256 authentication stage of FT-QuPAP v5.1.

Normal authentication-tag process:

1. ML-KEM-768 establishes a shared secret between the Mobile Station
   and Authentication Server.

2. Both parties derive transcript-bound authentication and control
   keys from the shared secret.

3. The Mobile Station calculates a 128-bit KMAC256 authentication tag.

4. The 128 classical tag bits are converted into logical qubits.

5. The payload is protected using Steane [[7,1,3]] encoding and sent
   through the quantum channel.

6. The Authentication Server performs Steane decoding, reconstructs
   the received tag, independently calculates the expected KMAC tag,
   and compares both tags using constant-time comparison.

Attack process:

1. A valid authentication tag is generated.

2. An attacker changes one or more tag bits before logical-qubit
   conversion and Steane encoding.

3. The forged tag is transmitted through an otherwise normal quantum
   channel.

4. The server reconstructs the forged tag successfully.

5. The reconstructed tag does not match the server-generated expected
   transcript-bound KMAC256 tag.

Expected result:

    REJECTED_KMAC_TAG

This deterministic integrity failure must never be accepted or changed
into a retry decision.
"""

from __future__ import annotations

import hmac
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


SCENARIO_NAME: Final[str] = "forged_kmac_tag"

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Forged KMAC Tag"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "An attacker changes one or more bits of the transcript-bound "
    "128-bit KMAC256 authentication tag before the tag is converted "
    "into logical qubits. The server must reject the decoded forged "
    "tag using deterministic constant-time tag verification."
)

DEFAULT_RANDOM_SEED: Final[int] = 9305

DEFAULT_TAG_LENGTH_BITS: Final[int] = 128
DEFAULT_TAG_LENGTH_BYTES: Final[int] = 16

DEFAULT_TAMPER_BYTE_INDEX: Final[int] = 0
DEFAULT_TAMPER_BIT_MASK: Final[int] = 0x01

EXPECTED_DECISION: Final[str] = (
    "rejected_kmac_tag"
)

EXPECTED_REJECTION_STAGE: Final[str] = (
    "kmac_tag_verification"
)


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the forged-KMAC-tag scenario.

    Args:
        context:
            Network context such as urban, suburban, or rural.

        random_seed:
            Reproducible quantum-channel simulation seed.

        app_config:
            Optional FT-QuPAP application configuration.

    Returns:
        Validated forged-KMAC-tag scenario configuration.
    """

    config = app_config or get_config()

    configured_tag_bits = (
        config.cryptography.kmac_tag_bits
    )

    if configured_tag_bits != DEFAULT_TAG_LENGTH_BITS:
        raise ValueError(
            "The forged-KMAC scenario expects a "
            f"{DEFAULT_TAG_LENGTH_BITS}-bit tag, but the "
            f"application configuration uses "
            f"{configured_tag_bits} bits."
        )

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
            # The channel remains close to normal so rejection is
            # attributable to the forged authentication tag.
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

            # Main attack switch.
            forge_kmac_tag=True,

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

        # GP analysis may still be recorded by the complete protocol
        # pipeline, but it is not needed to reject an invalid tag.
        gp_evaluation_expected=False,
        notes=(
            "The Mobile Station and server ML-KEM keys remain valid.",
            "Both sides derive matching session keys.",
            "The control schedule remains valid.",
            "The original KMAC256 tag is valid before modification.",
            "At least one authentication-tag bit is changed.",
            "The forged tag is encoded using Steane [[7,1,3]].",
            "Quantum transmission may otherwise appear normal.",
            "The server independently recomputes the expected tag.",
            "Tag comparison must use constant-time comparison.",
            "An invalid KMAC tag must cause deterministic rejection.",
            "Retry is forbidden for authentication-tag failure.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": "forged_kmac_tag",
            "attack_enabled": True,
            "attacker": (
                "authentication_payload_attacker"
            ),
            "attack_stage": (
                "authentication_tag_preparation"
            ),
            "attacked_component": (
                "transcript_bound_kmac_tag"
            ),
            "kmac_algorithm": (
                config.cryptography.kmac_algorithm
            ),
            "kmac_tag_bits": configured_tag_bits,
            "kmac_tag_bytes": (
                configured_tag_bits // 8
            ),
            "default_tamper_byte_index": (
                DEFAULT_TAMPER_BYTE_INDEX
            ),
            "default_tamper_bit_mask": (
                DEFAULT_TAMPER_BIT_MASK
            ),
            "expected_decision": EXPECTED_DECISION,
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "mlkem_ciphertext_valid": True,
            "shared_secret_match_expected": True,
            "authentication_key_match_expected": True,
            "control_key_match_expected": True,
            "control_schedule_match_expected": True,
            "original_tag_valid": True,
            "received_tag_modified": True,
            "decoded_tag_matches_expected": False,
            "constant_time_comparison_expected": True,
            "deterministic_rejection_expected": True,
            "retry_expected": False,
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
    """Alias for building the forged-KMAC-tag scenario."""

    return build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )


def forge_kmac_tag(
    valid_tag: bytes,
    *,
    byte_index: int = DEFAULT_TAMPER_BYTE_INDEX,
    bit_mask: int = DEFAULT_TAMPER_BIT_MASK,
    expected_length: int = DEFAULT_TAG_LENGTH_BYTES,
) -> bytes:
    """
    Return a modified copy of a KMAC authentication tag.

    The selected byte is XORed with a nonzero mask. The function never
    modifies the original bytes object.

    Args:
        valid_tag:
            Original valid KMAC tag.

        byte_index:
            Byte position to modify.

        bit_mask:
            Nonzero XOR mask between 1 and 255.

        expected_length:
            Required tag length in bytes.

    Returns:
        Forged tag with the same length as the original tag.
    """

    if not isinstance(valid_tag, bytes):
        raise TypeError(
            "valid_tag must be bytes."
        )

    if isinstance(expected_length, bool) or not isinstance(
        expected_length,
        int,
    ):
        raise TypeError(
            "expected_length must be an integer."
        )

    if expected_length < 1:
        raise ValueError(
            "expected_length must be positive."
        )

    if len(valid_tag) != expected_length:
        raise ValueError(
            "Invalid KMAC tag length. Expected "
            f"{expected_length} bytes but received "
            f"{len(valid_tag)} bytes."
        )

    if isinstance(byte_index, bool) or not isinstance(
        byte_index,
        int,
    ):
        raise TypeError(
            "byte_index must be an integer."
        )

    if not 0 <= byte_index < len(valid_tag):
        raise ValueError(
            "byte_index is outside the authentication tag."
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

    modified_tag = bytearray(valid_tag)
    modified_tag[byte_index] ^= bit_mask

    forged_tag = bytes(modified_tag)

    if forged_tag == valid_tag:
        raise RuntimeError(
            "Tag forgery did not modify the original tag."
        )

    if len(forged_tag) != len(valid_tag):
        raise RuntimeError(
            "Tag forgery changed the tag length."
        )

    return forged_tag


def compare_tags(
    received_tag: bytes,
    expected_tag: bytes,
) -> bool:
    """
    Compare authentication tags using constant-time comparison.
    """

    if not isinstance(received_tag, bytes):
        raise TypeError(
            "received_tag must be bytes."
        )

    if not isinstance(expected_tag, bytes):
        raise TypeError(
            "expected_tag must be bytes."
        )

    return hmac.compare_digest(
        received_tag,
        expected_tag,
    )


def get_scenario_summary(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return dashboard-friendly forged-tag information."""

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )

    summary = scenario.dashboard_summary()

    summary.update(
        {
            "attacker": (
                "authentication_payload_attacker"
            ),
            "attack_stage": (
                "authentication_tag_preparation"
            ),
            "attack_operation": (
                "flip_kmac_tag_bit"
            ),
            "attacked_component": (
                "128-bit KMAC256 tag"
            ),
            "expected_decision": EXPECTED_DECISION,
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "mlkem_ciphertext_valid": True,
            "shared_secret_match_expected": True,
            "session_keys_match_expected": True,
            "control_schedule_valid": True,
            "original_tag_valid": True,
            "received_tag_modified": True,
            "decoded_tag_valid_expected": False,
            "constant_time_comparison_required": True,
            "gp_evaluation_required": False,
            "retry_allowed": False,
            "expected_detection_sources": [
                "decoded_authentication_tag",
                "server_generated_expected_tag",
                "canonical_transcript",
                "derived_authentication_key",
                "constant_time_tag_comparison",
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
    """Return effective forged-tag conditions."""

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
            "valid_tag_generated": True,
            "tag_forgery_enabled": True,
            "tag_length_bits": (
                config.cryptography.kmac_tag_bits
            ),
            "tamper_byte_index": (
                DEFAULT_TAMPER_BYTE_INDEX
            ),
            "tamper_bit_mask": (
                DEFAULT_TAMPER_BIT_MASK
            ),
            "shared_secret_match_expected": True,
            "derived_keys_match_expected": True,
            "expected_tag_match": False,
            "expected_attempt_decision": (
                EXPECTED_DECISION
            ),
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "retry_allowed": False,
        }
    )

    return result


def create_tag_attack_plan() -> list[dict[str, Any]]:
    """
    Return the controlled forged-KMAC-tag demonstration sequence.
    """

    return [
        {
            "step": 1,
            "actor": "mobile_station",
            "operation": (
                "derive_transcript_bound_keys"
            ),
            "shared_secret_valid": True,
            "expected_result": (
                "authentication_and_control_keys_ready"
            ),
        },
        {
            "step": 2,
            "actor": "mobile_station",
            "operation": "generate_kmac_tag",
            "algorithm": "KMAC256",
            "tag_length_bits": (
                DEFAULT_TAG_LENGTH_BITS
            ),
            "original_tag_valid": True,
        },
        {
            "step": 3,
            "actor": (
                "authentication_payload_attacker"
            ),
            "operation": "modify_kmac_tag",
            "byte_index": (
                DEFAULT_TAMPER_BYTE_INDEX
            ),
            "bit_mask": (
                DEFAULT_TAMPER_BIT_MASK
            ),
            "expected_result": "forged_tag_created",
        },
        {
            "step": 4,
            "actor": "mobile_station",
            "operation": (
                "convert_tag_to_logical_qubits"
            ),
            "classical_tag_bits": (
                DEFAULT_TAG_LENGTH_BITS
            ),
            "expected_logical_payload_blocks": (
                DEFAULT_TAG_LENGTH_BITS
            ),
        },
        {
            "step": 5,
            "actor": "mobile_station",
            "operation": "steane_encode_and_transmit",
            "steane_code": "[[7,1,3]]",
            "quantum_channel_attack": False,
        },
        {
            "step": 6,
            "actor": "authentication_server",
            "operation": (
                "decode_received_authentication_tag"
            ),
            "expected_result": (
                "forged_tag_reconstructed"
            ),
        },
        {
            "step": 7,
            "actor": "authentication_server",
            "operation": (
                "compare_received_and_expected_kmac_tags"
            ),
            "constant_time_comparison": True,
            "expected_tag_match": False,
            "expected_decision": EXPECTED_DECISION,
        },
        {
            "step": 8,
            "actor": "authentication_server",
            "operation": "stop_authentication",
            "accepted": False,
            "retry_requested": False,
        },
    ]


def validate_forged_tag_result(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate correct rejection of a forged KMAC tag.

    A valid result must show:

    - Authentication rejection
    - KMAC comparison failure
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

    tag_verification_performed = read_boolean(
        result,
        (
            "tag_verification_performed",
            "kmac_verification_performed",
            "authentication_tag_checked",
        ),
        default=False,
    )

    tag_valid = read_boolean(
        result,
        (
            "tag_valid",
            "kmac_valid",
            "authentication_tag_valid",
            "tags_match",
        ),
        default=True,
    )

    constant_time_comparison = read_boolean(
        result,
        (
            "constant_time_comparison",
            "constant_time_comparison_used",
            "compare_digest_used",
        ),
        default=False,
    )

    shared_secret_match = read_boolean(
        result,
        (
            "shared_secret_match",
            "shared_secrets_match",
        ),
        default=True,
    )

    derived_keys_match = read_boolean(
        result,
        (
            "derived_keys_match",
            "session_keys_match",
            "authentication_keys_match",
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
        "rejected_tag",
        "rejected_kmac",
        "rejected_kmac_tag",
        "rejected_authentication_tag",
        "failed",
    }

    valid_reasons = {
        "forged_kmac_tag",
        "invalid_kmac_tag",
        "kmac_verification_failed",
        "authentication_tag_mismatch",
        "tag_verification_failed",
        "tags_do_not_match",
    }

    rejected = decision in valid_decisions

    tag_failure_detected = (
        reason in valid_reasons
        or (
            tag_verification_performed
            and not tag_valid
        )
    )

    return (
        rejected
        and tag_failure_detected
        and not tag_valid
        and shared_secret_match
        and derived_keys_match
        and constant_time_comparison
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
                "used",
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
                "not_used",
            }:
                return False

    return default


FORGED_KMAC_TAG_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run forged-KMAC-tag consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert scenario.name == "forged_kmac_tag"

    assert (
        scenario.category
        == "cryptographic_attack"
    )

    assert scenario.expected_outcome == "rejected"
    assert scenario.attack_enabled is True
    assert scenario.retry_expected is False

    assert scenario.eve.enabled is False

    assert (
        scenario.tampering.forge_kmac_tag
        is True
    )

    assert scenario.tampering.enabled is True
    assert scenario.retry.enabled is False

    assert (
        scenario.gp_evaluation_expected
        is False
    )

    valid_tag = bytes(
        range(DEFAULT_TAG_LENGTH_BYTES)
    )

    forged_tag = forge_kmac_tag(valid_tag)

    assert forged_tag != valid_tag

    assert (
        len(forged_tag)
        == DEFAULT_TAG_LENGTH_BYTES
    )

    assert (
        forged_tag[0]
        == valid_tag[0] ^ 0x01
    )

    assert compare_tags(
        valid_tag,
        valid_tag,
    ) is True

    assert compare_tags(
        forged_tag,
        valid_tag,
    ) is False

    attempt = scenario.for_attempt(
        1,
        config,
    )

    assert (
        attempt.tampering.forge_kmac_tag
        is True
    )

    assert (
        attempt.force_retry_gray_zone
        is False
    )

    attack_plan = create_tag_attack_plan()

    assert len(attack_plan) == 8

    assert (
        attack_plan[2]["operation"]
        == "modify_kmac_tag"
    )

    assert (
        attack_plan[6]["expected_tag_match"]
        is False
    )

    assert (
        attack_plan[6]["expected_decision"]
        == EXPECTED_DECISION
    )

    valid_result = validate_forged_tag_result(
        {
            "decision": "rejected_kmac_tag",
            "reason": "authentication_tag_mismatch",
            "tag_verification_performed": True,
            "tag_valid": False,
            "constant_time_comparison_used": True,
            "shared_secret_match": True,
            "derived_keys_match": True,
            "retry_used": False,
            "accepted": False,
        }
    )

    assert valid_result is True

    invalid_retry_result = (
        validate_forged_tag_result(
            {
                "decision": "rejected_tag",
                "reason": "invalid_kmac_tag",
                "kmac_verification_performed": True,
                "kmac_valid": False,
                "compare_digest_used": True,
                "shared_secrets_match": True,
                "session_keys_match": True,
                "retry_requested": True,
                "accepted": False,
            }
        )
    )

    assert invalid_retry_result is False

    invalid_key_mismatch_result = (
        validate_forged_tag_result(
            {
                "decision": "rejected_kmac",
                "reason": "kmac_verification_failed",
                "tag_verification_performed": True,
                "tag_valid": False,
                "constant_time_comparison": True,
                "shared_secret_match": False,
                "derived_keys_match": False,
                "retry_used": False,
                "accepted": False,
            }
        )
    )

    assert invalid_key_mismatch_result is False

    invalid_accept_result = (
        validate_forged_tag_result(
            {
                "decision": "accepted",
                "tag_verification_performed": True,
                "tag_valid": True,
                "constant_time_comparison": True,
                "shared_secret_match": True,
                "derived_keys_match": True,
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
        == "rejected_kmac_tag"
    )

    assert (
        summary["shared_secret_match_expected"]
        is True
    )

    assert (
        summary["decoded_tag_valid_expected"]
        is False
    )

    assert summary["retry_allowed"] is False

    print(
        "FT-QuPAP forged-KMAC-tag scenario "
        "self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TAG_LENGTH_BITS",
    "DEFAULT_TAG_LENGTH_BYTES",
    "DEFAULT_TAMPER_BYTE_INDEX",
    "DEFAULT_TAMPER_BIT_MASK",
    "EXPECTED_DECISION",
    "EXPECTED_REJECTION_STAGE",
    "FORGED_KMAC_TAG_CONFIG",
    "build_scenario",
    "create_scenario",
    "forge_kmac_tag",
    "compare_tags",
    "get_scenario_summary",
    "get_attempt_configuration",
    "create_tag_attack_plan",
    "validate_forged_tag_result",
]


if __name__ == "__main__":
    run_self_test()