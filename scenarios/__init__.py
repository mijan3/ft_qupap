"""
FT-QuPAP Controlled Scenarios Package
=====================================

Controlled authentication, channel, and attack scenarios for the
FT-QuPAP v5.1 capstone demonstration.

The scenario package provides reproducible configurations for testing:

- Normal authentication
- Benign channel noise
- Successful authentication after retry
- Eavesdropping attacks
- Replay attacks
- Classical-message modification
- Forged server signatures
- Modified ML-KEM ciphertexts
- Forged KMAC tags
- Excessive quantum-channel loss
- Uncorrectable Steane-code errors

Each scenario configures the existing protocol engine. Scenarios must
not directly bypass or replace:

- Subscriber verification
- Timestamp freshness checking
- Nonce replay detection
- ML-DSA signature verification
- ML-KEM encapsulation and decapsulation
- Transcript-bound session-key derivation
- KMAC tag verification
- Check-block QBER analysis
- Steane [[7,1,3]] syndrome processing
- Deterministic verification
- Calibrated Gaussian Process detection
- Bounded retry policy
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, Final


SCENARIO_PACKAGE_VERSION: Final[str] = "5.1.0"


# ---------------------------------------------------------------------------
# Scenario categories
# ---------------------------------------------------------------------------

CATEGORY_BENIGN: Final[str] = "benign"
CATEGORY_RETRY: Final[str] = "retry"
CATEGORY_QUANTUM_ATTACK: Final[str] = "quantum_attack"
CATEGORY_CLASSICAL_ATTACK: Final[str] = "classical_attack"
CATEGORY_CRYPTOGRAPHIC_ATTACK: Final[str] = (
    "cryptographic_attack"
)
CATEGORY_CHANNEL_FAILURE: Final[str] = "channel_failure"


# ---------------------------------------------------------------------------
# Expected outcomes
# ---------------------------------------------------------------------------

EXPECTED_ACCEPT: Final[str] = "accepted"
EXPECTED_ACCEPT_AFTER_RETRY: Final[str] = (
    "accepted_after_retry"
)
EXPECTED_RETRY_OR_ACCEPT: Final[str] = "retry_or_accept"
EXPECTED_REJECT: Final[str] = "rejected"


@dataclass(frozen=True)
class ScenarioDefinition:
    """
    Public metadata describing one controlled scenario.

    Attributes:
        name:
            Unique machine-readable scenario identifier.

        display_name:
            Human-readable dashboard label.

        module_name:
            Python module containing the implementation.

        category:
            High-level scenario category.

        expected_outcome:
            Expected protocol result.

        attack_enabled:
            Whether the scenario intentionally introduces an attacker.

        retry_expected:
            Whether retry processing is expected.

        description:
            Short explanation displayed in the dashboard.
    """

    name: str
    display_name: str
    module_name: str
    category: str
    expected_outcome: str
    attack_enabled: bool
    retry_expected: bool
    description: str

    def to_dictionary(self) -> dict[str, Any]:
        """Return a JSON-compatible scenario description."""

        return asdict(self)


SCENARIO_DEFINITIONS: Final[
    tuple[ScenarioDefinition, ...]
] = (
    ScenarioDefinition(
        name="normal_session",
        display_name="Normal Session",
        module_name="scenarios.normal_session",
        category=CATEGORY_BENIGN,
        expected_outcome=EXPECTED_ACCEPT,
        attack_enabled=False,
        retry_expected=False,
        description=(
            "Valid subscriber authentication under a low-noise "
            "quantum channel."
        ),
    ),
    ScenarioDefinition(
        name="benign_noisy_session",
        display_name="Benign Noisy Session",
        module_name="scenarios.benign_noisy_session",
        category=CATEGORY_BENIGN,
        expected_outcome=EXPECTED_RETRY_OR_ACCEPT,
        attack_enabled=False,
        retry_expected=True,
        description=(
            "Valid authentication affected by realistic benign "
            "channel noise without an eavesdropper."
        ),
    ),
    ScenarioDefinition(
        name="accept_after_retry",
        display_name="Accept After Retry",
        module_name="scenarios.accept_after_retry",
        category=CATEGORY_RETRY,
        expected_outcome=EXPECTED_ACCEPT_AFTER_RETRY,
        attack_enabled=False,
        retry_expected=True,
        description=(
            "The first attempt enters the protected retry region "
            "and a later attempt is accepted."
        ),
    ),
    ScenarioDefinition(
        name="intercept_resend_attack",
        display_name="Intercept-Resend Attack",
        module_name="scenarios.intercept_resend_attack",
        category=CATEGORY_QUANTUM_ATTACK,
        expected_outcome=EXPECTED_REJECT,
        attack_enabled=True,
        retry_expected=False,
        description=(
            "Eve measures transmitted qubits and prepares "
            "replacement qubits for the server."
        ),
    ),
    ScenarioDefinition(
        name="partial_eavesdropping",
        display_name="Partial Eavesdropping",
        module_name="scenarios.partial_eavesdropping",
        category=CATEGORY_QUANTUM_ATTACK,
        expected_outcome=EXPECTED_REJECT,
        attack_enabled=True,
        retry_expected=False,
        description=(
            "Eve interacts with only a controlled fraction of "
            "the transmitted quantum blocks."
        ),
    ),
    ScenarioDefinition(
        name="full_eavesdropping",
        display_name="Full Eavesdropping",
        module_name="scenarios.full_eavesdropping",
        category=CATEGORY_QUANTUM_ATTACK,
        expected_outcome=EXPECTED_REJECT,
        attack_enabled=True,
        retry_expected=False,
        description=(
            "Eve attacks the complete transmitted quantum sequence."
        ),
    ),
    ScenarioDefinition(
        name="replay_attack",
        display_name="Replay Attack",
        module_name="scenarios.replay_attack",
        category=CATEGORY_CLASSICAL_ATTACK,
        expected_outcome=EXPECTED_REJECT,
        attack_enabled=True,
        retry_expected=False,
        description=(
            "A previously transmitted authentication request "
            "or nonce is submitted again."
        ),
    ),
    ScenarioDefinition(
        name="forged_server_signature",
        display_name="Forged Server Signature",
        module_name="scenarios.forged_server_signature",
        category=CATEGORY_CRYPTOGRAPHIC_ATTACK,
        expected_outcome=EXPECTED_REJECT,
        attack_enabled=True,
        retry_expected=False,
        description=(
            "The Mobile Station receives a server package carrying "
            "an invalid ML-DSA signature."
        ),
    ),
    ScenarioDefinition(
        name="modified_authentication_request",
        display_name="Modified Authentication Request",
        module_name=(
            "scenarios.modified_authentication_request"
        ),
        category=CATEGORY_CLASSICAL_ATTACK,
        expected_outcome=EXPECTED_REJECT,
        attack_enabled=True,
        retry_expected=False,
        description=(
            "One or more transcript-bound authentication-request "
            "fields are altered in transit."
        ),
    ),
    ScenarioDefinition(
        name="tampered_mlkem_ciphertext",
        display_name="Tampered ML-KEM Ciphertext",
        module_name="scenarios.tampered_mlkem_ciphertext",
        category=CATEGORY_CRYPTOGRAPHIC_ATTACK,
        expected_outcome=EXPECTED_REJECT,
        attack_enabled=True,
        retry_expected=False,
        description=(
            "The ML-KEM-768 ciphertext is modified before server "
            "decapsulation."
        ),
    ),
    ScenarioDefinition(
        name="forged_kmac_tag",
        display_name="Forged KMAC Tag",
        module_name="scenarios.forged_kmac_tag",
        category=CATEGORY_CRYPTOGRAPHIC_ATTACK,
        expected_outcome=EXPECTED_REJECT,
        attack_enabled=True,
        retry_expected=False,
        description=(
            "The received 128-bit KMAC256 authentication tag does "
            "not match the reconstructed transcript."
        ),
    ),
    ScenarioDefinition(
        name="excessive_loss",
        display_name="Excessive Quantum Loss",
        module_name="scenarios.excessive_loss",
        category=CATEGORY_CHANNEL_FAILURE,
        expected_outcome=EXPECTED_REJECT,
        attack_enabled=False,
        retry_expected=False,
        description=(
            "Quantum-channel loss exceeds the maximum permitted "
            "loss rate or leaves too few observed check blocks."
        ),
    ),
    ScenarioDefinition(
        name="uncorrectable_quantum_error",
        display_name="Uncorrectable Quantum Error",
        module_name=(
            "scenarios.uncorrectable_quantum_error"
        ),
        category=CATEGORY_CHANNEL_FAILURE,
        expected_outcome=EXPECTED_REJECT,
        attack_enabled=False,
        retry_expected=False,
        description=(
            "A Steane logical block contains an error pattern "
            "outside the supported correction capability."
        ),
    ),
)


SCENARIO_REGISTRY: Final[
    dict[str, ScenarioDefinition]
] = {
    scenario.name: scenario
    for scenario in SCENARIO_DEFINITIONS
}


DEFAULT_SCENARIO_NAME: Final[str] = "normal_session"


def normalize_scenario_name(name: str) -> str:
    """Validate and normalize a scenario identifier."""

    if not isinstance(name, str):
        raise TypeError("Scenario name must be a string.")

    normalized_name = (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if not normalized_name:
        raise ValueError("Scenario name cannot be empty.")

    return normalized_name


def is_registered_scenario(name: str) -> bool:
    """Return whether a scenario exists in the registry."""

    try:
        normalized_name = normalize_scenario_name(name)
    except (TypeError, ValueError):
        return False

    return normalized_name in SCENARIO_REGISTRY


def get_scenario_definition(
    name: str,
) -> ScenarioDefinition:
    """
    Return metadata for a registered scenario.

    Raises:
        KeyError:
            If the scenario is not registered.
    """

    normalized_name = normalize_scenario_name(name)

    try:
        return SCENARIO_REGISTRY[normalized_name]
    except KeyError as error:
        available_names = ", ".join(
            sorted(SCENARIO_REGISTRY)
        )

        raise KeyError(
            f"Unknown FT-QuPAP scenario "
            f"'{normalized_name}'. Available scenarios: "
            f"{available_names}"
        ) from error


def list_scenarios(
    *,
    category: str | None = None,
    attack_only: bool = False,
) -> list[ScenarioDefinition]:
    """
    Return registered scenarios with optional filtering.

    Args:
        category:
            Optional scenario category.

        attack_only:
            Return only scenarios that intentionally enable an attack.
    """

    if not isinstance(attack_only, bool):
        raise TypeError("attack_only must be a boolean.")

    normalized_category: str | None = None

    if category is not None:
        if not isinstance(category, str):
            raise TypeError(
                "category must be a string or None."
            )

        normalized_category = category.strip().lower()

        if not normalized_category:
            raise ValueError(
                "category cannot be empty."
            )

    selected_scenarios: list[
        ScenarioDefinition
    ] = []

    for scenario in SCENARIO_DEFINITIONS:
        if (
            normalized_category is not None
            and scenario.category != normalized_category
        ):
            continue

        if attack_only and not scenario.attack_enabled:
            continue

        selected_scenarios.append(scenario)

    return selected_scenarios


def list_scenario_names() -> list[str]:
    """Return scenario identifiers in dashboard order."""

    return [
        scenario.name
        for scenario in SCENARIO_DEFINITIONS
    ]


def get_scenario_choices() -> dict[str, str]:
    """
    Return dashboard labels mapped to scenario identifiers.

    Example:

        {
            "Normal Session": "normal_session",
            "Replay Attack": "replay_attack"
        }
    """

    return {
        scenario.display_name: scenario.name
        for scenario in SCENARIO_DEFINITIONS
    }


def load_scenario_module(
    name: str,
) -> ModuleType:
    """
    Dynamically import a registered scenario module.

    Dynamic loading keeps package initialization lightweight and avoids
    importing the complete protocol stack before a scenario is selected.
    """

    definition = get_scenario_definition(name)

    try:
        return importlib.import_module(
            definition.module_name
        )
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Scenario implementation could not be loaded: "
            f"{definition.module_name}"
        ) from error


def get_scenario_callable(
    name: str,
    *,
    preferred_names: tuple[str, ...] = (
        "run_scenario",
        "build_scenario",
        "create_scenario",
    ),
) -> Any:
    """
    Return the first supported callable from a scenario module.

    Scenario implementations should normally expose one of:

        run_scenario()
        build_scenario()
        create_scenario()
    """

    module = load_scenario_module(name)

    for callable_name in preferred_names:
        candidate = getattr(
            module,
            callable_name,
            None,
        )

        if callable(candidate):
            return candidate

    raise AttributeError(
        f"Scenario module '{module.__name__}' does not "
        "expose a supported scenario function."
    )


def get_scenario_summary() -> list[dict[str, Any]]:
    """Return all scenario metadata as dictionaries."""

    return [
        scenario.to_dictionary()
        for scenario in SCENARIO_DEFINITIONS
    ]


def run_self_test() -> None:
    """Run scenario-registry consistency checks."""

    assert len(SCENARIO_DEFINITIONS) == 13
    assert len(SCENARIO_REGISTRY) == 13

    assert DEFAULT_SCENARIO_NAME in SCENARIO_REGISTRY

    normal = get_scenario_definition(
        "normal_session"
    )

    assert normal.attack_enabled is False
    assert normal.expected_outcome == EXPECTED_ACCEPT

    retry = get_scenario_definition(
        "accept-after-retry"
    )

    assert retry.retry_expected is True
    assert (
        retry.expected_outcome
        == EXPECTED_ACCEPT_AFTER_RETRY
    )

    replay = get_scenario_definition(
        "Replay Attack"
    )

    assert replay.attack_enabled is True
    assert replay.expected_outcome == EXPECTED_REJECT

    attack_scenarios = list_scenarios(
        attack_only=True
    )

    assert len(attack_scenarios) == 8

    assert is_registered_scenario(
        "forged_kmac_tag"
    )

    assert not is_registered_scenario(
        "unknown_scenario"
    )

    print("FT-QuPAP scenario registry self-test passed.")


__all__ = [
    "SCENARIO_PACKAGE_VERSION",
    "CATEGORY_BENIGN",
    "CATEGORY_RETRY",
    "CATEGORY_QUANTUM_ATTACK",
    "CATEGORY_CLASSICAL_ATTACK",
    "CATEGORY_CRYPTOGRAPHIC_ATTACK",
    "CATEGORY_CHANNEL_FAILURE",
    "EXPECTED_ACCEPT",
    "EXPECTED_ACCEPT_AFTER_RETRY",
    "EXPECTED_RETRY_OR_ACCEPT",
    "EXPECTED_REJECT",
    "ScenarioDefinition",
    "SCENARIO_DEFINITIONS",
    "SCENARIO_REGISTRY",
    "DEFAULT_SCENARIO_NAME",
    "normalize_scenario_name",
    "is_registered_scenario",
    "get_scenario_definition",
    "list_scenarios",
    "list_scenario_names",
    "get_scenario_choices",
    "load_scenario_module",
    "get_scenario_callable",
    "get_scenario_summary",
]


if __name__ == "__main__":
    run_self_test()