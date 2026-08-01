"""
FT-QuPAP LED Decision Controller
================================

Controls the optional ESP32/Arduino LED indicator used in the
FT-QuPAP v5.1 capstone demonstration.

Decision-to-color mapping:

    accepted
    accepted_after_retry
        -> GREEN

    retry
    retry_pending
        -> YELLOW

    rejected
    rejected_replay
    rejected_credential
    rejected_ciphertext
    rejected_deterministic
    rejected_gp
    rejected_retry_exhausted
    failed
    aborted
        -> RED

    reset
    off
        -> OFF

The hardware indicator is only a presentation layer. It does not make
authentication decisions and does not replace deterministic verification,
QBER analysis, Steane decoding, GP inference, or retry policy.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .hardware_fallback import NullLEDController
from .serial_connection import (
    close_serial,
    get_connection_description,
    is_connection_open,
    open_serial,
)


GREEN_COMMAND = b"GREEN\n"
YELLOW_COMMAND = b"YELLOW\n"
RED_COMMAND = b"RED\n"
OFF_COMMAND = b"OFF\n"
PING_COMMAND = b"PING\n"

DEFAULT_ACK_TIMEOUT_SECONDS = 1.0
DEFAULT_POST_WRITE_DELAY_SECONDS = 0.05


ACCEPTED_DECISIONS = {
    "accepted",
    "accepted_after_retry",
}

RETRY_DECISIONS = {
    "retry",
    "retry_pending",
    "retry_started",
    "retry_requested",
}

REJECTED_DECISIONS = {
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

OFF_DECISIONS = {
    "off",
    "reset",
    "idle",
    "initialized",
}


@dataclass(frozen=True)
class LEDCommandResult:
    """
    Result of one hardware-indicator command.

    Attributes:
        success:
            True when the command was written successfully.

        decision:
            Normalized FT-QuPAP decision.

        command:
            Command sent to the controller.

        indicator:
            GREEN, YELLOW, RED, or OFF.

        timestamp:
            UTC time at which the command was sent.

        connection_mode:
            physical_serial or software_fallback.

        port:
            Serial port or NULL-HARDWARE.

        bytes_written:
            Number of command bytes written.

        acknowledgement:
            Optional device response.

        error:
            Error message when command transmission failed.
    """

    success: bool
    decision: str
    command: str
    indicator: str
    timestamp: str
    connection_mode: str
    port: str | None
    bytes_written: int
    acknowledgement: str | None = None
    error: str | None = None

    def to_dictionary(self) -> dict[str, Any]:
        """Return a JSON-compatible command result."""

        return asdict(self)


class LEDController:
    """
    High-level FT-QuPAP hardware indicator controller.

    The controller may use a real PySerial connection or the
    NullLEDController fallback.

    Example:
        controller = LEDController(use_fallback=True)
        controller.connect()
        controller.send_decision("accepted")
        controller.close()
    """

    def __init__(
        self,
        connection: Any | None = None,
        *,
        port: str | None = None,
        baudrate: int = 115_200,
        timeout: float = 1.0,
        write_timeout: float = 1.0,
        reset_delay: float = 2.0,
        use_fallback: bool = True,
        fallback_echo: bool = False,
        wait_for_acknowledgement: bool = False,
        acknowledgement_timeout: float = (
            DEFAULT_ACK_TIMEOUT_SECONDS
        ),
        post_write_delay: float = (
            DEFAULT_POST_WRITE_DELAY_SECONDS
        ),
    ) -> None:
        """Initialize the LED controller."""

        self._connection = connection
        self._port = port
        self._baudrate = validate_positive_integer(
            "baudrate",
            baudrate,
        )
        self._timeout = validate_nonnegative_number(
            "timeout",
            timeout,
        )
        self._write_timeout = validate_nonnegative_number(
            "write_timeout",
            write_timeout,
        )
        self._reset_delay = validate_nonnegative_number(
            "reset_delay",
            reset_delay,
        )

        self._use_fallback = validate_boolean(
            "use_fallback",
            use_fallback,
        )
        self._fallback_echo = validate_boolean(
            "fallback_echo",
            fallback_echo,
        )
        self._wait_for_acknowledgement = validate_boolean(
            "wait_for_acknowledgement",
            wait_for_acknowledgement,
        )

        self._acknowledgement_timeout = (
            validate_nonnegative_number(
                "acknowledgement_timeout",
                acknowledgement_timeout,
            )
        )

        self._post_write_delay = validate_nonnegative_number(
            "post_write_delay",
            post_write_delay,
        )

        self._owns_connection = connection is None
        self._last_result: LEDCommandResult | None = None

    @property
    def connection(self) -> Any | None:
        """Return the active hardware connection."""

        return self._connection

    @property
    def is_connected(self) -> bool:
        """Return whether the connection is open."""

        return is_connection_open(self._connection)

    @property
    def is_fallback(self) -> bool:
        """Return True when software fallback is active."""

        return isinstance(
            self._connection,
            NullLEDController,
        )

    @property
    def last_result(self) -> LEDCommandResult | None:
        """Return the most recent command result."""

        return self._last_result

    def connect(self) -> Any:
        """Open or return the hardware connection."""

        if self.is_connected:
            return self._connection

        if (
            self._connection is not None
            and hasattr(self._connection, "open")
        ):
            self._connection.open()

            if self.is_connected:
                return self._connection

        self._connection = open_serial(
            port=self._port,
            baudrate=self._baudrate,
            timeout=self._timeout,
            write_timeout=self._write_timeout,
            reset_delay=self._reset_delay,
            use_fallback=self._use_fallback,
            fallback_echo=self._fallback_echo,
        )

        self._owns_connection = True

        return self._connection

    def close(self) -> None:
        """Close the hardware connection."""

        if self._connection is None:
            return

        close_serial(self._connection)

    def send_decision(
        self,
        decision: str | bool | Mapping[str, Any],
        *,
        wait_for_acknowledgement: bool | None = None,
    ) -> LEDCommandResult:
        """
        Send the appropriate indicator command for a protocol decision.

        The method accepts:

        - A decision string
        - A boolean accepted value
        - A complete decision/result mapping
        """

        normalized_decision = normalize_decision(
            decision
        )

        command = command_for_decision(
            normalized_decision
        )

        return self.send_command(
            command=command,
            decision=normalized_decision,
            wait_for_acknowledgement=(
                wait_for_acknowledgement
            ),
        )

    def send_command(
        self,
        command: str | bytes,
        *,
        decision: str | None = None,
        wait_for_acknowledgement: bool | None = None,
    ) -> LEDCommandResult:
        """Send a GREEN, YELLOW, RED, OFF, or PING command."""

        connection = self.connect()

        command_bytes = normalize_command(
            command
        )

        indicator = command_bytes.decode(
            "ascii"
        ).strip()

        normalized_decision = (
            normalize_required_string(
                "decision",
                decision,
            ).lower()
            if decision is not None
            else indicator.lower()
        )

        should_wait_for_acknowledgement = (
            self._wait_for_acknowledgement
            if wait_for_acknowledgement is None
            else validate_boolean(
                "wait_for_acknowledgement",
                wait_for_acknowledgement,
            )
        )

        description = get_connection_description(
            connection
        )

        bytes_written = 0
        acknowledgement: str | None = None
        error_message: str | None = None
        success = False

        try:
            bytes_written = connection.write(
                command_bytes
            )

            flush_method = getattr(
                connection,
                "flush",
                None,
            )

            if callable(flush_method):
                flush_method()

            if self._post_write_delay > 0:
                time.sleep(self._post_write_delay)

            if should_wait_for_acknowledgement:
                acknowledgement = (
                    read_acknowledgement(
                        connection,
                        timeout_seconds=(
                            self._acknowledgement_timeout
                        ),
                    )
                )

            success = True

        except Exception as error:
            error_message = str(error)

        result = LEDCommandResult(
            success=success,
            decision=normalized_decision,
            command=indicator,
            indicator=indicator,
            timestamp=current_utc_timestamp(),
            connection_mode=str(
                description.get("mode", "unknown")
            ),
            port=normalize_optional_string(
                description.get("port")
            ),
            bytes_written=int(bytes_written),
            acknowledgement=acknowledgement,
            error=error_message,
        )

        self._last_result = result

        return result

    def accepted(self) -> LEDCommandResult:
        """Display the accepted state."""

        return self.send_decision("accepted")

    def accepted_after_retry(
        self,
    ) -> LEDCommandResult:
        """Display acceptance after retry."""

        return self.send_decision(
            "accepted_after_retry"
        )

    def retry(self) -> LEDCommandResult:
        """Display the retry-pending state."""

        return self.send_decision("retry_pending")

    def rejected(
        self,
        reason: str = "rejected",
    ) -> LEDCommandResult:
        """Display a rejection state."""

        normalized_reason = (
            normalize_required_string(
                "reason",
                reason,
            ).lower()
        )

        if normalized_reason not in REJECTED_DECISIONS:
            normalized_reason = "rejected"

        return self.send_decision(
            normalized_reason
        )

    def turn_off(self) -> LEDCommandResult:
        """Turn off all decision LEDs."""

        return self.send_decision("off")

    def ping(self) -> LEDCommandResult:
        """Send a device connectivity command."""

        return self.send_command(
            PING_COMMAND,
            decision="ping",
            wait_for_acknowledgement=True,
        )

    def connection_information(
        self,
    ) -> dict[str, Any]:
        """Return dashboard-friendly connection information."""

        return get_connection_description(
            self._connection
        )

    def __enter__(self) -> "LEDController":
        """Connect and return the controller."""

        self.connect()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Close the controller connection."""

        self.close()


def send_decision(
    decision: str | bool | Mapping[str, Any],
    connection: Any | None = None,
    *,
    port: str | None = None,
    baudrate: int = 115_200,
    use_fallback: bool = True,
    fallback_echo: bool = False,
    close_after_send: bool = False,
    wait_for_acknowledgement: bool = False,
) -> LEDCommandResult:
    """
    Convenience function for sending one FT-QuPAP decision.

    Example:
        result = send_decision("accepted")

        result = send_decision(
            {
                "accepted": False,
                "reason": "rejected_gp",
            }
        )
    """

    controller = LEDController(
        connection=connection,
        port=port,
        baudrate=baudrate,
        use_fallback=use_fallback,
        fallback_echo=fallback_echo,
        wait_for_acknowledgement=(
            wait_for_acknowledgement
        ),
    )

    try:
        return controller.send_decision(
            decision
        )
    finally:
        if close_after_send:
            controller.close()


def command_for_decision(
    decision: str | bool | Mapping[str, Any],
) -> bytes:
    """Convert an FT-QuPAP decision into an LED command."""

    normalized_decision = normalize_decision(
        decision
    )

    if normalized_decision in ACCEPTED_DECISIONS:
        return GREEN_COMMAND

    if normalized_decision in RETRY_DECISIONS:
        return YELLOW_COMMAND

    if normalized_decision in REJECTED_DECISIONS:
        return RED_COMMAND

    if normalized_decision in OFF_DECISIONS:
        return OFF_COMMAND

    raise ValueError(
        "Unsupported FT-QuPAP decision: "
        f"{normalized_decision}"
    )


def normalize_decision(
    decision: str | bool | Mapping[str, Any],
) -> str:
    """
    Normalize a string, boolean, or decision-result mapping.

    Mapping resolution order:

    1. outcome
    2. reason
    3. status
    4. accepted
    5. decision.accepted / decision.reason
    """

    if isinstance(decision, bool):
        return "accepted" if decision else "rejected"

    if isinstance(decision, str):
        normalized_value = decision.strip().lower()

        if not normalized_value:
            raise ValueError(
                "decision cannot be empty."
            )

        return normalize_decision_alias(
            normalized_value
        )

    if not isinstance(decision, Mapping):
        raise TypeError(
            "decision must be a string, boolean, "
            "or mapping."
        )

    for field_name in (
        "outcome",
        "reason",
        "status",
    ):
        field_value = decision.get(field_name)

        if isinstance(field_value, str):
            normalized_value = (
                field_value.strip().lower()
            )

            if normalized_value:
                resolved = normalize_decision_alias(
                    normalized_value
                )

                if is_supported_decision(resolved):
                    return resolved

    accepted_value = decision.get("accepted")

    if isinstance(accepted_value, bool):
        if accepted_value:
            retry_attempts = safe_integer(
                decision.get(
                    "retry_attempts",
                    1,
                ),
                default=1,
            )

            retry_used = bool(
                decision.get(
                    "retry_used",
                    retry_attempts > 1,
                )
            )

            return (
                "accepted_after_retry"
                if retry_used or retry_attempts > 1
                else "accepted"
            )

        return "rejected"

    nested_decision = decision.get("decision")

    if isinstance(nested_decision, Mapping):
        return normalize_decision(
            nested_decision
        )

    raise ValueError(
        "Decision mapping does not contain a supported "
        "outcome, reason, status, or accepted value."
    )


def normalize_decision_alias(
    decision: str,
) -> str:
    """Normalize common application decision aliases."""

    aliases = {
        "accept": "accepted",
        "success": "accepted",
        "authenticated": "accepted",
        "authentication_accepted": "accepted",
        "accepted after retry":
            "accepted_after_retry",
        "accepted-after-retry":
            "accepted_after_retry",

        "yellow": "retry_pending",
        "retry_required": "retry_pending",
        "retry_allowed": "retry_pending",
        "low_risk_gray_zone":
            "retry_pending",

        "reject": "rejected",
        "denied": "rejected",
        "authentication_rejected":
            "rejected",
        "red": "rejected",

        "green": "accepted",

        "clear": "off",
        "none": "off",
    }

    return aliases.get(decision, decision)


def is_supported_decision(
    decision: str,
) -> bool:
    """Return True for a recognized decision."""

    return decision in (
        ACCEPTED_DECISIONS
        | RETRY_DECISIONS
        | REJECTED_DECISIONS
        | OFF_DECISIONS
    )


def normalize_command(
    command: str | bytes,
) -> bytes:
    """Normalize and validate a controller command."""

    if isinstance(command, str):
        normalized_text = command.strip().upper()

        if not normalized_text:
            raise ValueError(
                "command cannot be empty."
            )

        command_bytes = (
            normalized_text.encode("ascii")
            + b"\n"
        )

    elif isinstance(command, bytes):
        try:
            normalized_text = (
                command.decode("ascii")
                .strip()
                .upper()
            )
        except UnicodeDecodeError as error:
            raise ValueError(
                "command must contain ASCII data."
            ) from error

        command_bytes = (
            normalized_text.encode("ascii")
            + b"\n"
        )

    else:
        raise TypeError(
            "command must be a string or bytes."
        )

    supported_commands = {
        GREEN_COMMAND,
        YELLOW_COMMAND,
        RED_COMMAND,
        OFF_COMMAND,
        PING_COMMAND,
    }

    if command_bytes not in supported_commands:
        raise ValueError(
            "Unsupported LED command: "
            f"{normalized_text}"
        )

    return command_bytes


def read_acknowledgement(
    connection: Any,
    *,
    timeout_seconds: float = (
        DEFAULT_ACK_TIMEOUT_SECONDS
    ),
) -> str | None:
    """Read a short acknowledgement from the controller."""

    validate_nonnegative_number(
        "timeout_seconds",
        timeout_seconds,
    )

    readline_method = getattr(
        connection,
        "readline",
        None,
    )

    if not callable(readline_method):
        return None

    start_time = time.monotonic()

    while (
        time.monotonic() - start_time
        <= timeout_seconds
    ):
        response = readline_method()

        if response:
            if isinstance(response, bytes):
                return response.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

            return str(response).strip()

        time.sleep(0.01)

    return None


def current_utc_timestamp() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def normalize_required_string(
    name: str,
    value: str,
) -> str:
    """Validate and normalize a required string."""

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


def normalize_optional_string(
    value: Any,
) -> str | None:
    """Normalize optional text."""

    if value is None:
        return None

    normalized_value = str(value).strip()

    return normalized_value or None


def validate_boolean(
    name: str,
    value: bool,
) -> bool:
    """Validate and return a boolean."""

    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be a boolean."
        )

    return value


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

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return value


def validate_nonnegative_number(
    name: str,
    value: int | float,
) -> float:
    """Validate and return a finite nonnegative value."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    normalized_value = float(value)

    if normalized_value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

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

    return normalized_value


def safe_integer(
    value: Any,
    *,
    default: int = 0,
) -> int:
    """Convert an integer-like value to int."""

    if value is None or isinstance(value, bool):
        return default

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default

    if not converted.is_integer():
        return default

    return int(converted)


def run_self_test() -> None:
    """Run LED-controller tests without physical hardware."""

    fallback = NullLEDController(
        echo=False,
        start_open=True,
    )

    controller = LEDController(
        connection=fallback,
        wait_for_acknowledgement=True,
        post_write_delay=0.0,
    )

    accepted_result = controller.send_decision(
        "accepted"
    )

    assert accepted_result.success is True
    assert accepted_result.indicator == "GREEN"
    assert accepted_result.command == "GREEN"
    assert accepted_result.bytes_written == len(
        GREEN_COMMAND
    )
    assert accepted_result.acknowledgement == (
        "ACK:GREEN"
    )

    retry_result = controller.send_decision(
        {
            "accepted": False,
            "status": "retry_pending",
        }
    )

    assert retry_result.success is True
    assert retry_result.indicator == "YELLOW"

    rejected_result = controller.send_decision(
        {
            "decision": {
                "accepted": False,
                "reason": "rejected_gp",
            }
        }
    )

    assert rejected_result.success is True
    assert rejected_result.indicator == "RED"

    accepted_after_retry = (
        controller.send_decision(
            {
                "accepted": True,
                "retry_attempts": 2,
                "retry_used": True,
            }
        )
    )

    assert (
        accepted_after_retry.decision
        == "accepted_after_retry"
    )
    assert accepted_after_retry.indicator == "GREEN"

    off_result = controller.turn_off()

    assert off_result.success is True
    assert off_result.indicator == "OFF"

    assert command_for_decision(
        "accepted"
    ) == GREEN_COMMAND

    assert command_for_decision(
        "retry_pending"
    ) == YELLOW_COMMAND

    assert command_for_decision(
        "rejected_replay"
    ) == RED_COMMAND

    assert command_for_decision(
        False
    ) == RED_COMMAND

    history = fallback.get_history()

    assert len(history) == 5
    assert history[0].command == "GREEN"
    assert history[1].command == "YELLOW"
    assert history[2].command == "RED"
    assert history[3].command == "GREEN"
    assert history[4].command == "OFF"

    controller.close()

    print("LED controller self-test passed.")


if __name__ == "__main__":
    run_self_test()