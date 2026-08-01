"""
FT-QuPAP Hardware Fallback
==========================

Software-only hardware controller for FT-QuPAP v5.1.

The fallback controller emulates the small subset of the PySerial
interface required by the hardware package. It allows the complete
FT-QuPAP application to run when:

- No ESP32 is connected
- No serial port is available
- PySerial is not installed
- The project is running in simulation or test mode

Supported indicator commands:

    GREEN   -> Authentication accepted
    YELLOW  -> Authentication retry requested
    RED     -> Authentication rejected or failed
    OFF     -> Turn off all indicators

This module does not perform any security decision. It only receives and
records a decision already produced by the FT-QuPAP protocol engine.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


GREEN_COMMAND = b"GREEN\n"
YELLOW_COMMAND = b"YELLOW\n"
RED_COMMAND = b"RED\n"
OFF_COMMAND = b"OFF\n"

SUPPORTED_COMMANDS = {
    GREEN_COMMAND,
    YELLOW_COMMAND,
    RED_COMMAND,
    OFF_COMMAND,
}

SUPPORTED_DECISIONS = {
    "accepted",
    "accepted_after_retry",
    "retry",
    "retry_pending",
    "rejected",
    "rejected_replay",
    "rejected_credential",
    "rejected_ciphertext",
    "rejected_deterministic",
    "rejected_gp",
    "rejected_retry_exhausted",
    "failed",
    "aborted",
}


def current_utc_timestamp() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FallbackCommandRecord:
    """
    Record of one command handled by the fallback controller.
    """

    sequence_number: int
    timestamp: str
    command: str
    byte_length: int

    def to_dictionary(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


class NullLEDController:
    """
    In-memory replacement for an ESP32 serial LED controller.

    The object follows the basic serial-port interface used by the
    hardware integration layer:

        controller.write(data)
        controller.flush()
        controller.close()
        controller.is_open

    It may also be used directly:

        controller.send_decision("accepted")
        controller.send_command("GREEN")
    """

    def __init__(
        self,
        *,
        echo: bool = False,
        start_open: bool = True,
        maximum_history: int = 1000,
    ) -> None:
        """
        Initialize the software-only controller.

        Args:
            echo:
                Print each command to the terminal.

            start_open:
                Start the fallback controller in an open state.

            maximum_history:
                Maximum number of command records retained in memory.
        """

        if not isinstance(echo, bool):
            raise TypeError("echo must be a boolean.")

        if not isinstance(start_open, bool):
            raise TypeError("start_open must be a boolean.")

        if (
            isinstance(maximum_history, bool)
            or not isinstance(maximum_history, int)
        ):
            raise TypeError(
                "maximum_history must be an integer."
            )

        if maximum_history < 1:
            raise ValueError(
                "maximum_history must be greater than zero."
            )

        self._echo = echo
        self._is_open = start_open
        self._maximum_history = maximum_history

        self._history: list[FallbackCommandRecord] = []
        self._sequence_number = 0
        self._last_command: str | None = None

        self._lock = threading.RLock()

    @property
    def is_open(self) -> bool:
        """Return whether the fallback connection is open."""

        with self._lock:
            return self._is_open

    @property
    def port(self) -> str:
        """Return the virtual serial-port name."""

        return "NULL-HARDWARE"

    @property
    def baudrate(self) -> int:
        """Return a placeholder baud rate."""

        return 0

    @property
    def in_waiting(self) -> int:
        """Return zero because no serial input is available."""

        return 0

    @property
    def last_command(self) -> str | None:
        """Return the most recently processed command."""

        with self._lock:
            return self._last_command

    @property
    def command_count(self) -> int:
        """Return the number of processed commands."""

        with self._lock:
            return self._sequence_number

    def open(self) -> None:
        """Open the fallback connection."""

        with self._lock:
            self._is_open = True

    def close(self) -> None:
        """Close the fallback connection."""

        with self._lock:
            self._is_open = False

    def flush(self) -> None:
        """
        Provide PySerial-compatible flush behavior.

        No operation is needed because commands are stored immediately.
        """

        self._require_open()

    def reset_input_buffer(self) -> None:
        """Provide a PySerial-compatible no-operation method."""

        self._require_open()

    def reset_output_buffer(self) -> None:
        """Provide a PySerial-compatible no-operation method."""

        self._require_open()

    def write(
        self,
        data: bytes | bytearray | memoryview,
    ) -> int:
        """
        Process a serial-style LED command.

        Args:
            data:
                Command bytes such as b"GREEN\\n".

        Returns:
            Number of bytes processed.
        """

        self._require_open()

        if not isinstance(
            data,
            (bytes, bytearray, memoryview),
        ):
            raise TypeError(
                "Serial command must be bytes-like."
            )

        command_bytes = bytes(data)

        if not command_bytes:
            raise ValueError(
                "Serial command cannot be empty."
            )

        normalized_command = normalize_command_bytes(
            command_bytes
        )

        if normalized_command not in SUPPORTED_COMMANDS:
            readable_command = normalized_command.decode(
                "ascii",
                errors="replace",
            ).strip()

            raise ValueError(
                "Unsupported hardware command: "
                f"{readable_command}"
            )

        command_text = normalized_command.decode(
            "ascii"
        ).strip()

        self._record_command(
            command=command_text,
            byte_length=len(command_bytes),
        )

        return len(command_bytes)

    def send_command(
        self,
        command: str | bytes,
    ) -> str:
        """
        Send a named indicator command.

        Supported values:

            GREEN
            YELLOW
            RED
            OFF
        """

        if isinstance(command, str):
            normalized_text = command.strip().upper()

            if not normalized_text:
                raise ValueError(
                    "command cannot be empty."
                )

            command_bytes = (
                normalized_text.encode("ascii") + b"\n"
            )

        elif isinstance(command, bytes):
            command_bytes = command

        else:
            raise TypeError(
                "command must be a string or bytes."
            )

        normalized_command = normalize_command_bytes(
            command_bytes
        )

        self.write(normalized_command)

        return normalized_command.decode(
            "ascii"
        ).strip()

    def send_decision(
        self,
        decision: str | bool,
    ) -> str:
        """
        Convert a protocol decision into an indicator command.

        Decision mapping:

            accepted / accepted_after_retry / True
                -> GREEN

            retry / retry_pending
                -> YELLOW

            rejected / failed / aborted / False
                -> RED
        """

        command = command_for_decision(decision)

        return self.send_command(command)

    def turn_off(self) -> str:
        """Record an OFF command."""

        return self.send_command("OFF")

    def get_history(
        self,
        limit: int | None = None,
    ) -> list[FallbackCommandRecord]:
        """
        Return a deep copy of the command history.

        Args:
            limit:
                Optional maximum number of most recent records.
        """

        if limit is not None:
            if isinstance(limit, bool) or not isinstance(
                limit,
                int,
            ):
                raise TypeError(
                    "limit must be an integer or None."
                )

            if limit < 1:
                raise ValueError(
                    "limit must be greater than zero."
                )

        with self._lock:
            records = (
                self._history
                if limit is None
                else self._history[-limit:]
            )

            return copy.deepcopy(records)

    def get_history_as_dictionaries(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return command history as dictionaries."""

        return [
            record.to_dictionary()
            for record in self.get_history(limit)
        ]

    def clear_history(self) -> None:
        """Remove all stored command records."""

        with self._lock:
            self._history.clear()
            self._last_command = None

    def readline(self) -> bytes:
        """
        Return a virtual acknowledgement message.

        This method provides compatibility with code that expects an
        ESP32 acknowledgement after a command.
        """

        self._require_open()

        with self._lock:
            if self._last_command is None:
                return b"READY\n"

            return (
                f"ACK:{self._last_command}\n"
            ).encode("ascii")

    def __enter__(self) -> "NullLEDController":
        """Open and return the fallback controller."""

        self.open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Close the fallback controller."""

        self.close()

    def _record_command(
        self,
        *,
        command: str,
        byte_length: int,
    ) -> None:
        """Store one processed command."""

        with self._lock:
            self._sequence_number += 1
            self._last_command = command

            record = FallbackCommandRecord(
                sequence_number=self._sequence_number,
                timestamp=current_utc_timestamp(),
                command=command,
                byte_length=byte_length,
            )

            self._history.append(record)

            if (
                len(self._history)
                > self._maximum_history
            ):
                excess_count = (
                    len(self._history)
                    - self._maximum_history
                )

                del self._history[:excess_count]

            if self._echo:
                print(
                    "[FT-QuPAP hardware fallback] "
                    f"{command}"
                )

    def _require_open(self) -> None:
        """Raise an error when the connection is closed."""

        with self._lock:
            if not self._is_open:
                raise RuntimeError(
                    "Hardware fallback connection is closed."
                )


def normalize_command_bytes(
    command: bytes,
) -> bytes:
    """Normalize command bytes to uppercase ASCII with newline."""

    if not isinstance(command, bytes):
        raise TypeError("command must be bytes.")

    try:
        command_text = command.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(
            "Hardware command must contain ASCII data."
        ) from error

    normalized_text = command_text.strip().upper()

    if not normalized_text:
        raise ValueError(
            "Hardware command cannot be empty."
        )

    return normalized_text.encode("ascii") + b"\n"


def command_for_decision(
    decision: str | bool,
) -> str:
    """Map an FT-QuPAP decision to an LED command."""

    if isinstance(decision, bool):
        return "GREEN" if decision else "RED"

    if not isinstance(decision, str):
        raise TypeError(
            "decision must be a string or boolean."
        )

    normalized_decision = decision.strip().lower()

    if not normalized_decision:
        raise ValueError(
            "decision cannot be empty."
        )

    if normalized_decision not in SUPPORTED_DECISIONS:
        raise ValueError(
            "Unsupported FT-QuPAP decision: "
            f"{normalized_decision}"
        )

    if normalized_decision in {
        "accepted",
        "accepted_after_retry",
    }:
        return "GREEN"

    if normalized_decision in {
        "retry",
        "retry_pending",
    }:
        return "YELLOW"

    return "RED"


def run_self_test() -> None:
    """Run fallback-controller behavior tests."""

    controller = NullLEDController(
        echo=False,
        start_open=True,
    )

    assert controller.is_open
    assert controller.port == "NULL-HARDWARE"

    assert controller.send_decision(
        "accepted"
    ) == "GREEN"

    assert controller.send_decision(
        "retry_pending"
    ) == "YELLOW"

    assert controller.send_decision(
        "rejected_gp"
    ) == "RED"

    assert controller.send_decision(
        True
    ) == "GREEN"

    assert controller.send_decision(
        False
    ) == "RED"

    assert controller.turn_off() == "OFF"
    assert controller.command_count == 6
    assert controller.last_command == "OFF"
    assert controller.readline() == b"ACK:OFF\n"

    history = controller.get_history()

    assert len(history) == 6
    assert history[0].command == "GREEN"
    assert history[1].command == "YELLOW"
    assert history[2].command == "RED"
    assert history[-1].command == "OFF"

    controller.clear_history()

    assert controller.get_history() == []
    assert controller.last_command is None
    assert controller.readline() == b"READY\n"

    controller.close()

    assert not controller.is_open

    try:
        controller.send_command("GREEN")
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Closed controller accepted a command."
        )

    with NullLEDController() as context_controller:
        context_controller.send_command("GREEN")
        assert context_controller.is_open

    assert not context_controller.is_open

    print("Hardware fallback self-test passed.")


if __name__ == "__main__":
    run_self_test()