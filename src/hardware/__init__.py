"""
FT-QuPAP Hardware Integration Package
=====================================

Optional physical-indicator support for the FT-QuPAP v5.1 capstone
demonstration.

The hardware layer can communicate with an ESP32 or another serial
controller to display the final protocol state:

    GREEN  -> Authentication accepted
    YELLOW -> Retry requested
    RED    -> Authentication rejected or failed

Important:
    This package is not part of the FT-QuPAP security decision logic.

    Authentication decisions must first be produced by the protocol
    engine using:

    - Deterministic verification
    - QBER and loss evidence
    - Steane decoder results
    - Calibrated GP attack probability
    - Retry policy

    The hardware package only visualizes the resulting decision.

The package also provides a no-operation fallback so that the complete
project can run without an ESP32, serial port, or physical LEDs.
"""

from __future__ import annotations

from .hardware_fallback import NullLEDController
from .led_controller import send_decision
from .serial_connection import open_serial


HARDWARE_PACKAGE_VERSION = "1.0.0"

ACCEPT_COMMAND = b"GREEN\n"
RETRY_COMMAND = b"YELLOW\n"
REJECT_COMMAND = b"RED\n"


__all__ = [
    # Package information
    "HARDWARE_PACKAGE_VERSION",

    # Hardware commands
    "ACCEPT_COMMAND",
    "RETRY_COMMAND",
    "REJECT_COMMAND",

    # Serial connection
    "open_serial",

    # Decision indicator
    "send_decision",

    # Hardware-free fallback
    "NullLEDController",
]