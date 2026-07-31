"""
Authentication Server components for FT-QuPAP v5.1.

This package implements server-side registration, post-quantum
bootstrapping, quantum-frame processing, deterministic verification,
Gaussian Process attack detection, and retry decisions.
"""

AUTHENTICATION_SERVER_PACKAGE_VERSION = "5.1.0"


AUTHENTICATION_SERVER_MODULES = (
    "check_block_analyzer",
    "control_schedule_decryptor",
    "deterministic_verifier",
    "freshness_checker",
    "gp_attack_detector",
    "gp_feature_extractor",
    "mldsa_key_manager",
    "mlkem_decapsulation",
    "mlkem_key_manager",
    "payload_decoder",
    "registration_manager",
    "replay_detector",
    "retry_policy",
    "server_package_signer",
    "session_key_derivation",
    "subscriber_verifier",
    "syndrome_processor",
    "tag_verifier",
)


def get_authentication_server_package_info() -> dict[str, object]:
    """Return Authentication Server package information."""

    return {
        "package": "src.authentication_server",
        "version": AUTHENTICATION_SERVER_PACKAGE_VERSION,
        "modules": list(AUTHENTICATION_SERVER_MODULES),
    }


__all__ = [
    "AUTHENTICATION_SERVER_PACKAGE_VERSION",
    "AUTHENTICATION_SERVER_MODULES",
    "get_authentication_server_package_info",
]