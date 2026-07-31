"""
Cryptographic components of the FT-QuPAP protocol.

This package implements:

- ML-DSA server authentication
- ML-KEM session-key establishment
- Transcript hashing
- Session-key derivation
- KMAC authentication-tag generation
- Secure constant-time comparison
- Nonce generation and replay protection
- Pseudonymous subscriber identity handling
- Shared cryptographic data models
"""

CRYPTOGRAPHY_PACKAGE_VERSION = "5.1.0"


CRYPTOGRAPHY_MODULES = (
    "crypto_models",
    "kdf_module",
    "kmac_module",
    "mldsa_module",
    "mlkem_module",
    "nonce_manager",
    "pseudonymous_identity",
    "secure_compare",
    "transcript_hash",
)


def get_cryptography_package_info() -> dict[str, object]:
    """
    Return basic information about the cryptography package.
    """

    return {
        "package": "src.cryptography",
        "version": CRYPTOGRAPHY_PACKAGE_VERSION,
        "modules": list(CRYPTOGRAPHY_MODULES),
    }


__all__ = [
    "CRYPTOGRAPHY_PACKAGE_VERSION",
    "CRYPTOGRAPHY_MODULES",
    "get_cryptography_package_info",
]