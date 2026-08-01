"""
FT-QuPAP Protocol Package.

This package contains the high-level protocol models and engines used
to coordinate complete FT-QuPAP authentication sessions.

Modules:
    message_models:
        Defines protocol request, response, and transport messages.

    session_models:
        Defines authentication-session data structures.

    result_models:
        Defines verification, decision, retry, and final result models.

    transcript:
        Maintains the ordered protocol transcript and transcript hash.

    protocol_state:
        Defines valid FT-QuPAP protocol states and transitions.

    protocol_engine:
        Coordinates the complete end-to-end authentication workflow.

    verification_engine:
        Combines deterministic cryptographic and quantum verification.

    decision_engine:
        Combines deterministic verification with GP attack probability.

    retry_engine:
        Manages bounded fresh-session retry.

    protocol_logger:
        Records protocol events without exposing secret key material.
"""

__version__ = "1.0.0"

__all__ = [
    "__version__",
]