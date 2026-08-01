"""
FT-QuPAP Scenario Configuration
================================

Shared configuration models for controlled FT-QuPAP v5.1 demonstration
scenarios.

This module describes only the environment in which an authentication
session is executed. It does not replace or bypass any security check
performed by the protocol.

A scenario may configure:

- Network context
- Benign quantum-channel noise
- Quantum-channel loss
- Eve interaction
- Classical-message tampering
- Cryptographic forgery attempts
- Forced Steane-code error conditions
- Reproducible random seeds
- Retry demonstration behavior
- Expected experimental outcome

Security thresholds such as freshness, QBER, loss, GP probability, and
maximum retry count remain controlled by ``config.py`` and the protocol
decision engine.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from config import ApplicationConfig, get_config


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ScenarioCategory(str, Enum):
    """High-level controlled-scenario categories."""

    BENIGN = "benign"
    RETRY = "retry"
    QUANTUM_ATTACK = "quantum_attack"
    CLASSICAL_ATTACK = "classical_attack"
    CRYPTOGRAPHIC_ATTACK = "cryptographic_attack"
    CHANNEL_FAILURE = "channel_failure"


class ExpectedOutcome(str, Enum):
    """Expected protocol result for an experiment."""

    ACCEPTED = "accepted"
    ACCEPTED_AFTER_RETRY = "accepted_after_retry"
    RETRY_OR_ACCEPT = "retry_or_accept"
    REJECTED = "rejected"


class EveAttackMode(str, Enum):
    """Supported Eve interaction strategies."""

    NONE = "none"
    INTERCEPT_RESEND = "intercept_resend"
    PARTIAL_EAVESDROPPING = "partial_eavesdropping"
    FULL_EAVESDROPPING = "full_eavesdropping"


class NoiseModelName(str, Enum):
    """Supported simulated quantum-noise profiles."""

    NONE = "none"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    DEPOLARIZING = "depolarizing"
    COMBINED = "combined"


# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT = "urban"
DEFAULT_RANDOM_SEED = 9102

MINIMUM_PROBABILITY = 0.0
MAXIMUM_PROBABILITY = 1.0

SUPPORTED_EXPECTED_OUTCOMES = {
    outcome.value
    for outcome in ExpectedOutcome
}

SUPPORTED_CATEGORIES = {
    category.value
    for category in ScenarioCategory
}

SUPPORTED_EVE_ATTACK_MODES = {
    mode.value
    for mode in EveAttackMode
}

SUPPORTED_NOISE_MODELS = {
    model.value
    for model in NoiseModelName
}


# ---------------------------------------------------------------------------
# Quantum-channel profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelProfile:
    """
    Controlled quantum-channel conditions.

    All probability values must remain in the inclusive range [0, 1].

    Attributes:
        noise_model:
            Name of the simulated noise model.

        bit_flip_probability:
            Probability of an X-type error.

        phase_flip_probability:
            Probability of a Z-type error.

        depolarizing_probability:
            Probability of a depolarizing error.

        measurement_error_probability:
            Probability that a measured classical bit is inverted.

        loss_rate:
            Probability that a transmitted physical qubit is lost.

        burst_error_probability:
            Probability of a correlated error burst.

        burst_length:
            Number of consecutive physical qubits affected when a
            burst occurs.

        forced_lost_check_blocks:
            Additional independent check blocks intentionally hidden
            from the server for a controlled demonstration.

        forced_uncorrectable_blocks:
            Number of logical Steane blocks forced beyond the
            single-qubit correction capability.
    """

    noise_model: str = NoiseModelName.COMBINED.value

    bit_flip_probability: float = 0.002
    phase_flip_probability: float = 0.002
    depolarizing_probability: float = 0.003
    measurement_error_probability: float = 0.001

    loss_rate: float = 0.01

    burst_error_probability: float = 0.0
    burst_length: int = 0

    forced_lost_check_blocks: int = 0
    forced_uncorrectable_blocks: int = 0

    def validate(
        self,
        app_config: ApplicationConfig | None = None,
    ) -> None:
        """Validate channel-profile consistency."""

        config = app_config or get_config()

        if self.noise_model not in SUPPORTED_NOISE_MODELS:
            raise ValueError(
                "Unsupported noise model: "
                f"{self.noise_model}"
            )

        probability_fields = {
            "bit_flip_probability":
                self.bit_flip_probability,
            "phase_flip_probability":
                self.phase_flip_probability,
            "depolarizing_probability":
                self.depolarizing_probability,
            "measurement_error_probability":
                self.measurement_error_probability,
            "loss_rate": self.loss_rate,
            "burst_error_probability":
                self.burst_error_probability,
        }

        for name, value in probability_fields.items():
            validate_probability(name, value)

        validate_nonnegative_integer(
            "burst_length",
            self.burst_length,
        )

        validate_nonnegative_integer(
            "forced_lost_check_blocks",
            self.forced_lost_check_blocks,
        )

        validate_nonnegative_integer(
            "forced_uncorrectable_blocks",
            self.forced_uncorrectable_blocks,
        )

        if (
            self.burst_error_probability > 0.0
            and self.burst_length < 1
        ):
            raise ValueError(
                "burst_length must be positive when "
                "burst_error_probability is enabled."
            )

        if (
            self.burst_error_probability == 0.0
            and self.burst_length != 0
        ):
            raise ValueError(
                "burst_length must be zero when burst errors "
                "are disabled."
            )

        if (
            self.forced_lost_check_blocks
            > config.quantum.independent_check_blocks
        ):
            raise ValueError(
                "forced_lost_check_blocks cannot exceed the "
                "number of independent check blocks."
            )

        if (
            self.forced_uncorrectable_blocks
            > config.quantum.total_logical_blocks
        ):
            raise ValueError(
                "forced_uncorrectable_blocks cannot exceed "
                "the total number of logical blocks."
            )

    def scaled(
        self,
        *,
        noise_multiplier: float = 1.0,
        loss_multiplier: float = 1.0,
    ) -> "ChannelProfile":
        """
        Return a channel profile adjusted for a retry attempt.

        Probability values are clamped to [0, 1].
        """

        validate_nonnegative_number(
            "noise_multiplier",
            noise_multiplier,
        )

        validate_nonnegative_number(
            "loss_multiplier",
            loss_multiplier,
        )

        return replace(
            self,
            bit_flip_probability=clamp_probability(
                self.bit_flip_probability
                * noise_multiplier
            ),
            phase_flip_probability=clamp_probability(
                self.phase_flip_probability
                * noise_multiplier
            ),
            depolarizing_probability=clamp_probability(
                self.depolarizing_probability
                * noise_multiplier
            ),
            measurement_error_probability=(
                clamp_probability(
                    self.measurement_error_probability
                    * noise_multiplier
                )
            ),
            loss_rate=clamp_probability(
                self.loss_rate * loss_multiplier
            ),
            burst_error_probability=clamp_probability(
                self.burst_error_probability
                * noise_multiplier
            ),
        )


# ---------------------------------------------------------------------------
# Eve profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EveProfile:
    """
    Quantum eavesdropping configuration.

    Attributes:
        enabled:
            Whether Eve is active.

        attack_mode:
            Controlled Eve strategy.

        interaction_fraction:
            Fraction of transmitted physical qubits or logical blocks
            selected for interaction.

        measurement_basis_error_probability:
            Probability that Eve chooses an incompatible basis.

        resend_error_probability:
            Additional error probability introduced while preparing
            and resending replacement qubits.

        target_check_blocks:
            Whether Eve may interact with independent check blocks.

        target_payload_blocks:
            Whether Eve may interact with encoded payload blocks.
    """

    enabled: bool = False
    attack_mode: str = EveAttackMode.NONE.value

    interaction_fraction: float = 0.0
    measurement_basis_error_probability: float = 0.5
    resend_error_probability: float = 0.0

    target_check_blocks: bool = True
    target_payload_blocks: bool = True

    def validate(self) -> None:
        """Validate the eavesdropping configuration."""

        if not isinstance(self.enabled, bool):
            raise TypeError(
                "Eve enabled flag must be boolean."
            )

        if (
            self.attack_mode
            not in SUPPORTED_EVE_ATTACK_MODES
        ):
            raise ValueError(
                "Unsupported Eve attack mode: "
                f"{self.attack_mode}"
            )

        validate_probability(
            "interaction_fraction",
            self.interaction_fraction,
        )

        validate_probability(
            "measurement_basis_error_probability",
            self.measurement_basis_error_probability,
        )

        validate_probability(
            "resend_error_probability",
            self.resend_error_probability,
        )

        if not isinstance(self.target_check_blocks, bool):
            raise TypeError(
                "target_check_blocks must be boolean."
            )

        if not isinstance(self.target_payload_blocks, bool):
            raise TypeError(
                "target_payload_blocks must be boolean."
            )

        if not self.enabled:
            if self.attack_mode != EveAttackMode.NONE.value:
                raise ValueError(
                    "Disabled Eve profile must use attack mode "
                    "'none'."
                )

            if self.interaction_fraction != 0.0:
                raise ValueError(
                    "Disabled Eve profile must use zero "
                    "interaction_fraction."
                )

        if self.enabled:
            if self.attack_mode == EveAttackMode.NONE.value:
                raise ValueError(
                    "Enabled Eve profile requires an attack mode."
                )

            if self.interaction_fraction <= 0.0:
                raise ValueError(
                    "Enabled Eve profile requires a positive "
                    "interaction_fraction."
                )

            if not (
                self.target_check_blocks
                or self.target_payload_blocks
            ):
                raise ValueError(
                    "Enabled Eve profile must target at least "
                    "one block category."
                )

        if (
            self.attack_mode
            == EveAttackMode.FULL_EAVESDROPPING.value
            and self.interaction_fraction != 1.0
        ):
            raise ValueError(
                "Full eavesdropping requires an "
                "interaction_fraction of 1."
                )

        if (
            self.attack_mode
            == EveAttackMode.FULL_EAVESDROPPING.value
            and self.interaction_fraction != 1.0
        ):
            raise ValueError(
                "Full eavesdropping requires an "
                "interaction_fraction of 1."
                )

# ---------------------------------------------------------------------------
# Classical and cryptographic tampering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TamperingProfile:
    """
    Controlled classical and cryptographic attack switches.

    These switches instruct scenario orchestration code to alter a
    message before it reaches the relevant verifier. The verifiers
    themselves must remain unchanged.
    """

    replay_authentication_request: bool = False
    reuse_nonce: bool = False
    stale_timestamp: bool = False

    forge_server_signature: bool = False
    modify_authentication_request: bool = False
    tamper_mlkem_ciphertext: bool = False
    forge_kmac_tag: bool = False

    modify_mobile_identifier: bool = False
    modify_session_identifier: bool = False
    modify_network_identifier: bool = False
    modify_protocol_version: bool = False
    modify_control_schedule: bool = False

    def validate(self) -> None:
        """Validate all tampering switches."""

        for field_name, field_value in asdict(self).items():
            if not isinstance(field_value, bool):
                raise TypeError(
                    f"{field_name} must be boolean."
                )

        if (
            self.replay_authentication_request
            and not self.reuse_nonce
        ):
            raise ValueError(
                "A replayed authentication request must reuse "
                "the original nonce."
            )

    @property
    def enabled(self) -> bool:
        """Return whether any tampering action is active."""

        return any(asdict(self).values())

    def active_actions(self) -> tuple[str, ...]:
        """Return enabled tampering-action names."""

        return tuple(
            name
            for name, enabled in asdict(self).items()
            if enabled
        )


# ---------------------------------------------------------------------------
# Retry demonstration profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetryProfile:
    """
    Controlled retry behavior for reproducible demonstrations.

    This profile does not authorize a retry. The protocol retry engine
    must still determine whether the retry conditions are satisfied.

    Attributes:
        enabled:
            Whether the scenario may demonstrate retry processing.

        force_retry_on_attempts:
            Attempts for which generated evidence should intentionally
            fall into a retry-compatible gray zone.

        noise_multiplier_after_retry:
            Multiplier applied to benign noise on each later attempt.

        loss_multiplier_after_retry:
            Multiplier applied to loss on each later attempt.

        change_random_seed_per_attempt:
            Whether each retry receives a deterministic but distinct
            random seed.
    """

    enabled: bool = False

    force_retry_on_attempts: tuple[int, ...] = ()

    noise_multiplier_after_retry: float = 0.55
    loss_multiplier_after_retry: float = 0.70

    change_random_seed_per_attempt: bool = True

    def validate(
        self,
        app_config: ApplicationConfig | None = None,
    ) -> None:
        """Validate retry demonstration settings."""

        config = app_config or get_config()

        if not isinstance(self.enabled, bool):
            raise TypeError(
                "Retry enabled flag must be boolean."
            )

        validate_nonnegative_number(
            "noise_multiplier_after_retry",
            self.noise_multiplier_after_retry,
        )

        validate_nonnegative_number(
            "loss_multiplier_after_retry",
            self.loss_multiplier_after_retry,
        )

        if not isinstance(
            self.change_random_seed_per_attempt,
            bool,
        ):
            raise TypeError(
                "change_random_seed_per_attempt must be "
                "boolean."
            )

        seen_attempts: set[int] = set()

        for attempt in self.force_retry_on_attempts:
            validate_positive_integer(
                "force_retry_on_attempts item",
                attempt,
            )

            if (
                attempt
                > config.protocol
                .maximum_authentication_attempts
            ):
                raise ValueError(
                    "Forced retry attempt exceeds the protocol "
                    "maximum authentication attempts."
                )

            if attempt in seen_attempts:
                raise ValueError(
                    "force_retry_on_attempts cannot contain "
                    "duplicates."
                )

            seen_attempts.add(attempt)

        if (
            not self.enabled
            and self.force_retry_on_attempts
        ):
            raise ValueError(
                "Disabled retry profile cannot force retries."
            )

        if (
            config.protocol.maximum_retries == 0
            and self.enabled
        ):
            raise ValueError(
                "Retry profile cannot be enabled when the "
                "protocol permits no retries."
            )

    def should_force_retry(
        self,
        attempt_number: int,
    ) -> bool:
        """Return whether an attempt targets the retry gray zone."""

        validate_positive_integer(
            "attempt_number",
            attempt_number,
        )

        return (
            self.enabled
            and attempt_number
            in self.force_retry_on_attempts
        )


# ---------------------------------------------------------------------------
# Attempt-specific configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttemptScenarioConfig:
    """
    Effective scenario values for one authentication attempt.
    """

    scenario_name: str
    attempt_number: int
    random_seed: int

    context: str
    channel: ChannelProfile
    eve: EveProfile
    tampering: TamperingProfile

    force_retry_gray_zone: bool

    def to_dictionary(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


# ---------------------------------------------------------------------------
# Complete scenario configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioConfig:
    """
    Complete controlled FT-QuPAP scenario configuration.
    """

    name: str
    display_name: str
    description: str

    category: str
    expected_outcome: str

    context: str = DEFAULT_CONTEXT
    random_seed: int = DEFAULT_RANDOM_SEED

    channel: ChannelProfile = field(
        default_factory=ChannelProfile
    )

    eve: EveProfile = field(
        default_factory=EveProfile
    )

    tampering: TamperingProfile = field(
        default_factory=TamperingProfile
    )

    retry: RetryProfile = field(
        default_factory=RetryProfile
    )

    deterministic_verification_expected: bool = True
    gp_evaluation_expected: bool = True

    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
        app_config: ApplicationConfig | None = None,
    ) -> None:
        """Validate the complete scenario configuration."""

        config = app_config or get_config()

        normalized_name = normalize_identifier(
            self.name
        )

        if normalized_name != self.name:
            raise ValueError(
                "Scenario name must already be normalized as "
                f"'{normalized_name}'."
            )

        validate_required_string(
            "display_name",
            self.display_name,
        )

        validate_required_string(
            "description",
            self.description,
        )

        if self.category not in SUPPORTED_CATEGORIES:
            raise ValueError(
                "Unsupported scenario category: "
                f"{self.category}"
            )

        if (
            self.expected_outcome
            not in SUPPORTED_EXPECTED_OUTCOMES
        ):
            raise ValueError(
                "Unsupported expected outcome: "
                f"{self.expected_outcome}"
            )

        if self.context not in (
            config.protocol.supported_contexts
        ):
            raise ValueError(
                "Unsupported network context: "
                f"{self.context}"
            )

        validate_nonnegative_integer(
            "random_seed",
            self.random_seed,
        )

        if not isinstance(
            self.deterministic_verification_expected,
            bool,
        ):
            raise TypeError(
                "deterministic_verification_expected must "
                "be boolean."
            )

        if not isinstance(
            self.gp_evaluation_expected,
            bool,
        ):
            raise TypeError(
                "gp_evaluation_expected must be boolean."
            )

        self.channel.validate(config)
        self.eve.validate()
        self.tampering.validate()
        self.retry.validate(config)

        self._validate_category_consistency()

        for note in self.notes:
            validate_required_string(
                "scenario note",
                note,
            )

        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping."
            )

    def _validate_category_consistency(self) -> None:
        """Validate attack and category relationships."""

        quantum_attack = (
            self.category
            == ScenarioCategory.QUANTUM_ATTACK.value
        )

        if quantum_attack and not self.eve.enabled:
            raise ValueError(
                "Quantum-attack scenarios must enable Eve."
            )

        if (
            self.eve.enabled
            and not quantum_attack
        ):
            raise ValueError(
                "An enabled Eve profile requires the "
                "quantum_attack category."
            )

        tampering_attack_categories = {
            ScenarioCategory.CLASSICAL_ATTACK.value,
            ScenarioCategory.CRYPTOGRAPHIC_ATTACK.value,
        }

        if (
            self.category in tampering_attack_categories
            and not self.tampering.enabled
        ):
            raise ValueError(
                "Classical or cryptographic attack scenarios "
                "must enable a tampering action."
            )

        if (
            self.tampering.enabled
            and self.category
            not in tampering_attack_categories
        ):
            raise ValueError(
                "Tampering actions require a classical_attack "
                "or cryptographic_attack category."
            )

        if (
            self.category
            == ScenarioCategory.RETRY.value
            and not self.retry.enabled
        ):
            raise ValueError(
                "Retry-category scenarios must enable the "
                "retry profile."
            )

        if (
            self.expected_outcome
            == ExpectedOutcome.ACCEPTED_AFTER_RETRY.value
            and not self.retry.enabled
        ):
            raise ValueError(
                "accepted_after_retry requires retry to be "
                "enabled."
            )

        if (
            self.expected_outcome
            == ExpectedOutcome.REJECTED.value
            and self.category
            == ScenarioCategory.BENIGN.value
        ):
            raise ValueError(
                "A benign scenario should not declare rejection "
                "as its expected outcome."
            )

    @property
    def attack_enabled(self) -> bool:
        """Return whether any controlled attack is active."""

        return self.eve.enabled or self.tampering.enabled

    @property
    def retry_expected(self) -> bool:
        """Return whether retry activity is expected."""

        return (
            self.retry.enabled
            or self.expected_outcome
            in {
                ExpectedOutcome.ACCEPTED_AFTER_RETRY.value,
                ExpectedOutcome.RETRY_OR_ACCEPT.value,
            }
        )

    def for_attempt(
        self,
        attempt_number: int,
        app_config: ApplicationConfig | None = None,
    ) -> AttemptScenarioConfig:
        """
        Resolve effective conditions for an authentication attempt.

        The protocol maximum-attempt rule is enforced.
        """

        config = app_config or get_config()

        validate_positive_integer(
            "attempt_number",
            attempt_number,
        )

        if (
            attempt_number
            > config.protocol
            .maximum_authentication_attempts
        ):
            raise ValueError(
                "Attempt number exceeds the configured "
                "FT-QuPAP maximum."
            )

        retry_index = attempt_number - 1

        if retry_index == 0:
            effective_channel = self.channel
        else:
            noise_multiplier = (
                self.retry
                .noise_multiplier_after_retry
                ** retry_index
            )

            loss_multiplier = (
                self.retry
                .loss_multiplier_after_retry
                ** retry_index
            )

            effective_channel = self.channel.scaled(
                noise_multiplier=noise_multiplier,
                loss_multiplier=loss_multiplier,
            )

        effective_seed = self.random_seed

        if (
            self.retry.change_random_seed_per_attempt
            and attempt_number > 1
        ):
            effective_seed = (
                self.random_seed
                + attempt_number
                - 1
            )

        return AttemptScenarioConfig(
            scenario_name=self.name,
            attempt_number=attempt_number,
            random_seed=effective_seed,
            context=self.context,
            channel=effective_channel,
            eve=self.eve,
            tampering=self.tampering,
            force_retry_gray_zone=(
                self.retry.should_force_retry(
                    attempt_number
                )
            ),
        )

    def with_overrides(
        self,
        **changes: Any,
    ) -> "ScenarioConfig":
        """
        Return a validated copy with selected top-level values changed.
        """

        updated = replace(self, **changes)
        updated.validate()

        return updated

    def to_dictionary(self) -> dict[str, Any]:
        """Return a deep JSON-compatible dictionary."""

        result = asdict(self)
        result["metadata"] = copy.deepcopy(
            dict(self.metadata)
        )

        return result

    def dashboard_summary(self) -> dict[str, Any]:
        """Return concise information for scenario-selection UI."""

        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "expected_outcome": self.expected_outcome,
            "context": self.context,
            "random_seed": self.random_seed,
            "attack_enabled": self.attack_enabled,
            "eve_attack_mode": self.eve.attack_mode,
            "eve_interaction_fraction": (
                self.eve.interaction_fraction
            ),
            "tampering_actions": list(
                self.tampering.active_actions()
            ),
            "retry_expected": self.retry_expected,
            "noise_model": self.channel.noise_model,
            "loss_rate": self.channel.loss_rate,
            "forced_uncorrectable_blocks": (
                self.channel
                .forced_uncorrectable_blocks
            ),
        }


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------

def create_scenario_config(
    *,
    name: str,
    display_name: str,
    description: str,
    category: str | ScenarioCategory,
    expected_outcome: str | ExpectedOutcome,
    context: str = DEFAULT_CONTEXT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    channel: ChannelProfile | None = None,
    eve: EveProfile | None = None,
    tampering: TamperingProfile | None = None,
    retry: RetryProfile | None = None,
    deterministic_verification_expected: bool = True,
    gp_evaluation_expected: bool = True,
    notes: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ScenarioConfig:
    """
    Build and validate a complete scenario configuration.
    """

    normalized_category = enum_or_string_value(
        category
    )

    normalized_expected_outcome = (
        enum_or_string_value(expected_outcome)
    )

    scenario = ScenarioConfig(
        name=normalize_identifier(name),
        display_name=display_name.strip(),
        description=description.strip(),
        category=normalized_category,
        expected_outcome=(
            normalized_expected_outcome
        ),
        context=context.strip().lower(),
        random_seed=random_seed,
        channel=channel or ChannelProfile(),
        eve=eve or EveProfile(),
        tampering=(
            tampering or TamperingProfile()
        ),
        retry=retry or RetryProfile(),
        deterministic_verification_expected=(
            deterministic_verification_expected
        ),
        gp_evaluation_expected=(
            gp_evaluation_expected
        ),
        notes=tuple(notes),
        metadata=dict(metadata or {}),
    )

    scenario.validate()

    return scenario


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def enum_or_string_value(
    value: str | Enum,
) -> str:
    """Return a normalized value from a string or enum."""

    if isinstance(value, Enum):
        return str(value.value)

    if not isinstance(value, str):
        raise TypeError(
            "Expected a string or Enum value."
        )

    normalized_value = value.strip().lower()

    if not normalized_value:
        raise ValueError(
            "Enum-compatible value cannot be empty."
        )

    return normalized_value


def normalize_identifier(value: str) -> str:
    """Convert text into a safe scenario identifier."""

    validate_required_string(
        "scenario identifier",
        value,
    )

    normalized_value = value.strip().lower()
    normalized_value = re.sub(
        r"[^a-z0-9]+",
        "_",
        normalized_value,
    )

    return normalized_value.strip("_")


def validate_required_string(
    name: str,
    value: str,
) -> str:
    """Validate and return a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return normalized_value


def validate_probability(
    name: str,
    value: int | float,
) -> float:
    """Validate a finite probability in [0, 1]."""

    normalized_value = validate_nonnegative_number(
        name,
        value,
    )

    if normalized_value > MAXIMUM_PROBABILITY:
        raise ValueError(
            f"{name} must not exceed 1.0."
        )

    return normalized_value


def validate_nonnegative_number(
    name: str,
    value: int | float,
) -> float:
    """Validate and return a finite nonnegative number."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    normalized_value = float(value)

    if normalized_value != normalized_value:
        raise ValueError(
            f"{name} cannot be NaN."
        )

    if normalized_value in {
        float("inf"),
        float("-inf"),
    }:
        raise ValueError(
            f"{name} must be finite."
        )

    if normalized_value < 0.0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return normalized_value


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


def clamp_probability(value: float) -> float:
    """Clamp a numeric value to [0, 1]."""

    return max(
        MINIMUM_PROBABILITY,
        min(MAXIMUM_PROBABILITY, float(value)),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    """Run scenario-configuration consistency checks."""

    normal = create_scenario_config(
        name="normal_session",
        display_name="Normal Session",
        description=(
            "Valid authentication under a low-noise channel."
        ),
        category=ScenarioCategory.BENIGN,
        expected_outcome=ExpectedOutcome.ACCEPTED,
    )

    assert normal.name == "normal_session"
    assert normal.attack_enabled is False
    assert normal.retry_expected is False

    first_attempt = normal.for_attempt(1)

    assert first_attempt.attempt_number == 1
    assert first_attempt.random_seed == 9102

    retry_scenario = create_scenario_config(
        name="accept_after_retry",
        display_name="Accept After Retry",
        description=(
            "First attempt enters the retry region."
        ),
        category=ScenarioCategory.RETRY,
        expected_outcome=(
            ExpectedOutcome.ACCEPTED_AFTER_RETRY
        ),
        channel=ChannelProfile(
            depolarizing_probability=0.04,
            loss_rate=0.05,
        ),
        retry=RetryProfile(
            enabled=True,
            force_retry_on_attempts=(1,),
            noise_multiplier_after_retry=0.5,
            loss_multiplier_after_retry=0.5,
        ),
    )

    attempt_one = retry_scenario.for_attempt(1)
    attempt_two = retry_scenario.for_attempt(2)

    assert attempt_one.force_retry_gray_zone is True
    assert attempt_two.force_retry_gray_zone is False

    assert (
        attempt_two.channel.loss_rate
        < attempt_one.channel.loss_rate
    )

    eve_scenario = create_scenario_config(
        name="partial_eavesdropping",
        display_name="Partial Eavesdropping",
        description=(
            "Eve interacts with part of the transmission."
        ),
        category=(
            ScenarioCategory.QUANTUM_ATTACK
        ),
        expected_outcome=ExpectedOutcome.REJECTED,
        eve=EveProfile(
            enabled=True,
            attack_mode=(
                EveAttackMode
                .PARTIAL_EAVESDROPPING
                .value
            ),
            interaction_fraction=0.4,
        ),
    )

    assert eve_scenario.attack_enabled is True
    assert eve_scenario.eve.interaction_fraction == 0.4

    replay_scenario = create_scenario_config(
        name="replay_attack",
        display_name="Replay Attack",
        description=(
            "A previous request is submitted again."
        ),
        category=(
            ScenarioCategory.CLASSICAL_ATTACK
        ),
        expected_outcome=ExpectedOutcome.REJECTED,
        tampering=TamperingProfile(
            replay_authentication_request=True,
            reuse_nonce=True,
        ),
    )

    assert replay_scenario.tampering.enabled is True
    assert (
        "reuse_nonce"
        in replay_scenario
        .tampering
        .active_actions()
    )

    print(
        "FT-QuPAP scenario configuration "
        "self-test passed."
    )


__all__ = [
    "ScenarioCategory",
    "ExpectedOutcome",
    "EveAttackMode",
    "NoiseModelName",
    "ChannelProfile",
    "EveProfile",
    "TamperingProfile",
    "RetryProfile",
    "AttemptScenarioConfig",
    "ScenarioConfig",
    "create_scenario_config",
    "normalize_identifier",
]


if __name__ == "__main__":
    run_self_test()