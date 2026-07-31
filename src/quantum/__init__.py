"""
FT-QuPAP Quantum Simulation Package

This package contains the syndrome-level quantum components used by
the FT-QuPAP research simulator.

Main responsibilities:

1. Convert classical KMAC tag bits into logical qubit descriptions.
2. Generate independent check logical qubits.
3. Randomly interleave payload and check blocks.
4. Apply Steane [[7,1,3]] CSS encoding.
5. Simulate physical quantum-channel noise and loss.
6. Simulate intercept-measure-resend attacks.
7. Perform logical-basis measurements.
8. Calculate check-block QBER.
9. Extract syndromes and perform error correction.

Research boundary:

The complete FT-QuPAP experiment uses a scalable syndrome-level
simulation. It does not claim to implement a physical fault-tolerant
quantum computer.

Security boundary:

Hidden simulator information, including Eve's configured attack
fraction and attacked physical positions, must never be used as
Authentication Server GP model features.
"""

from __future__ import annotations


__version__ = "1.0.0"

__protocol_name__ = "FT-QuPAP"

__simulation_type__ = "syndrome-level"

__quantum_code__ = "Steane [[7,1,3]] CSS"


PAYLOAD_LOGICAL_BLOCKS = 128
CHECK_LOGICAL_BLOCKS = 32

TOTAL_LOGICAL_BLOCKS = (
    PAYLOAD_LOGICAL_BLOCKS
    + CHECK_LOGICAL_BLOCKS
)

STEANE_PHYSICAL_QUBITS_PER_BLOCK = 7

TOTAL_STEANE_PHYSICAL_QUBITS = (
    TOTAL_LOGICAL_BLOCKS
    * STEANE_PHYSICAL_QUBITS_PER_BLOCK
)


__all__ = [
    "__version__",
    "__protocol_name__",
    "__simulation_type__",
    "__quantum_code__",
    "PAYLOAD_LOGICAL_BLOCKS",
    "CHECK_LOGICAL_BLOCKS",
    "TOTAL_LOGICAL_BLOCKS",
    "STEANE_PHYSICAL_QUBITS_PER_BLOCK",
    "TOTAL_STEANE_PHYSICAL_QUBITS",
]