"""
FT-QuPAP Mobile Station Package

This package implements the Mobile Station side of the
Fault-Tolerant Quantum Authentication Protocol.

Main protocol stages:

1. Pseudonymous authentication request
2. ML-DSA server credential verification
3. ML-KEM encapsulation
4. Transcript-bound session-key derivation
5. KMAC-256 authentication tag generation
6. Payload and check-qubit preparation
7. K_ctrl-protected control schedule
8. Steane [[7,1,3]] CSS encoding
9. Quantum-channel transmission
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


__version__ = "1.0.0"
__protocol_name__ = "FT-QuPAP"
__protocol_version__ = "FT-QuPAP-1.0"


# Lazy import table:
# public_name -> (module_name, attribute_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Main coordinator
    "MobileStation": (
        ".mobile_station",
        "MobileStation",
    ),
    "PreparedMobileSession": (
        ".mobile_station",
        "PreparedMobileSession",
    ),
    "MobileStationError": (
        ".mobile_station",
        "MobileStationError",
    ),
    "ServerCredentialRejectedError": (
        ".mobile_station",
        "ServerCredentialRejectedError",
    ),
    "print_safe_session_summary": (
        ".mobile_station",
        "print_safe_session_summary",
    ),

    # Registration
    "RegistrationClient": (
        ".registration_client",
        "RegistrationClient",
    ),
    "RegistrationBundle": (
        ".registration_client",
        "RegistrationBundle",
    ),
    "MobileTrustAnchor": (
        ".registration_client",
        "MobileTrustAnchor",
    ),
    "SubscriberRecord": (
        ".registration_client",
        "SubscriberRecord",
    ),

    # Authentication request
    "create_authentication_request": (
        ".authentication_request",
        "create_authentication_request",
    ),

    # Server verification
    "ServerPackageVerificationResult": (
        ".server_package_verifier",
        "ServerPackageVerificationResult",
    ),
    "verify_server_credential": (
        ".server_package_verifier",
        "verify_server_credential",
    ),
    "verify_server_credential_detailed": (
        ".server_package_verifier",
        "verify_server_credential_detailed",
    ),

    # ML-KEM
    "MLKEMEncapsulationResult": (
        ".mlkem_encapsulation",
        "MLKEMEncapsulationResult",
    ),
    "encapsulate_session_secret": (
        ".mlkem_encapsulation",
        "encapsulate_session_secret",
    ),

    # Session-key derivation
    "derive_session_key_material": (
        ".session_key_derivation",
        "derive_session_key_material",
    ),
    "split_session_keys": (
        ".session_key_derivation",
        "split_session_keys",
    ),

    # KMAC authentication tag
    "compute_authentication_tag": (
        ".kmac_tag_generator",
        "compute_authentication_tag",
    ),

    # Payload preparation
    "map_tag_to_logical_specs": (
        ".payload_preparation",
        "map_tag_to_logical_specs",
    ),

    # Check-qubit preparation
    "generate_check_specs": (
        ".check_qubit_preparation",
        "generate_check_specs",
    ),

    # Control schedule
    "LogicalSpec": (
        ".control_schedule",
        "LogicalSpec",
    ),
    "ProtectedControlSchedule": (
        ".control_schedule",
        "ProtectedControlSchedule",
    ),
    "create_interleaved_schedule": (
        ".control_schedule",
        "create_interleaved_schedule",
    ),
    "attach_expected_reference_bits": (
        ".control_schedule",
        "attach_expected_reference_bits",
    ),
    "protect_control_schedule": (
        ".control_schedule",
        "protect_control_schedule",
    ),

    # Steane CSS encoding
    "LogicalQubitSpec": (
        ".steane_encoder",
        "LogicalQubitSpec",
    ),
    "PhysicalBlock": (
        ".steane_encoder",
        "PhysicalBlock",
    ),
    "SteaneEncodedFrame": (
        ".steane_encoder",
        "SteaneEncodedFrame",
    ),
    "encode_ft_qupap_frame": (
        ".steane_encoder",
        "encode_ft_qupap_frame",
    ),

    # Quantum transmission
    "ChannelConfig": (
        ".quantum_transmitter",
        "ChannelConfig",
    ),
    "QuantumTransmitter": (
        ".quantum_transmitter",
        "QuantumTransmitter",
    ),
    "QuantumTransmissionResult": (
        ".quantum_transmitter",
        "QuantumTransmissionResult",
    ),
    "IDEAL_CHANNEL": (
        ".quantum_transmitter",
        "IDEAL_CHANNEL",
    ),
    "NOISY_CHANNEL": (
        ".quantum_transmitter",
        "NOISY_CHANNEL",
    ),
    "LOSSY_CHANNEL": (
        ".quantum_transmitter",
        "LOSSY_CHANNEL",
    ),
    "PARTIAL_EVE_CHANNEL": (
        ".quantum_transmitter",
        "PARTIAL_EVE_CHANNEL",
    ),
    "FULL_EVE_CHANNEL": (
        ".quantum_transmitter",
        "FULL_EVE_CHANNEL",
    ),
}


__all__ = [
    "__version__",
    "__protocol_name__",
    "__protocol_version__",

    "MobileStation",
    "PreparedMobileSession",
    "MobileStationError",
    "ServerCredentialRejectedError",
    "print_safe_session_summary",

    "RegistrationClient",
    "RegistrationBundle",
    "MobileTrustAnchor",
    "SubscriberRecord",

    "create_authentication_request",

    "ServerPackageVerificationResult",
    "verify_server_credential",
    "verify_server_credential_detailed",

    "MLKEMEncapsulationResult",
    "encapsulate_session_secret",

    "derive_session_key_material",
    "split_session_keys",

    "compute_authentication_tag",
    "map_tag_to_logical_specs",
    "generate_check_specs",

    "LogicalSpec",
    "ProtectedControlSchedule",
    "create_interleaved_schedule",
    "attach_expected_reference_bits",
    "protect_control_schedule",

    "LogicalQubitSpec",
    "PhysicalBlock",
    "SteaneEncodedFrame",
    "encode_ft_qupap_frame",

    "ChannelConfig",
    "QuantumTransmitter",
    "QuantumTransmissionResult",
    "IDEAL_CHANNEL",
    "NOISY_CHANNEL",
    "LOSSY_CHANNEL",
    "PARTIAL_EVE_CHANNEL",
    "FULL_EVE_CHANNEL",
]


def __getattr__(name: str) -> Any:
    """
    Lazily import public package components.

    This prevents optional cryptographic and quantum libraries from
    loading when only a small part of the package is required.
    """

    if name not in _LAZY_IMPORTS:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = _LAZY_IMPORTS[name]

    module = import_module(
        module_name,
        package=__name__,
    )

    attribute = getattr(
        module,
        attribute_name,
    )

    # Cache the loaded value for later access.
    globals()[name] = attribute

    return attribute


def __dir__() -> list[str]:
    """Return all publicly available package names."""

    return sorted(
        set(globals())
        | set(__all__)
        | set(_LAZY_IMPORTS)
    )


if TYPE_CHECKING:
    from .authentication_request import (
        create_authentication_request,
    )
    from .check_qubit_preparation import (
        generate_check_specs,
    )
    from .control_schedule import (
        LogicalSpec,
        ProtectedControlSchedule,
        attach_expected_reference_bits,
        create_interleaved_schedule,
        protect_control_schedule,
    )
    from .kmac_tag_generator import (
        compute_authentication_tag,
    )
    from .mlkem_encapsulation import (
        MLKEMEncapsulationResult,
        encapsulate_session_secret,
    )
    from .mobile_station import (
        MobileStation,
        MobileStationError,
        PreparedMobileSession,
        ServerCredentialRejectedError,
        print_safe_session_summary,
    )
    from .payload_preparation import (
        map_tag_to_logical_specs,
    )
    from .quantum_transmitter import (
        FULL_EVE_CHANNEL,
        IDEAL_CHANNEL,
        LOSSY_CHANNEL,
        NOISY_CHANNEL,
        PARTIAL_EVE_CHANNEL,
        ChannelConfig,
        QuantumTransmissionResult,
        QuantumTransmitter,
    )
    from .registration_client import (
        MobileTrustAnchor,
        RegistrationBundle,
        RegistrationClient,
        SubscriberRecord,
    )
    from .server_package_verifier import (
        ServerPackageVerificationResult,
        verify_server_credential,
        verify_server_credential_detailed,
    )
    from .session_key_derivation import (
        derive_session_key_material,
        split_session_keys,
    )
    from .steane_encoder import (
        LogicalQubitSpec,
        PhysicalBlock,
        SteaneEncodedFrame,
        encode_ft_qupap_frame,
    )