"""
FT-QuPAP Serial Connection
==========================

Serial-port connection management for the optional FT-QuPAP physical
demonstration hardware.

The serial connection may be used to communicate with:

- ESP32
- Arduino-compatible board
- USB-to-serial LED controller

The hardware only displays the decision already produced by the
FT-QuPAP protocol engine:

    GREEN  -> accepted
    YELLOW -> retry
    RED    -> rejected or failed

When PySerial is unavailable, no suitable port is detected, or the
physical device cannot be opened, the module can return a
NullLEDController fallback.

This module does not perform authentication, cryptographic operations,
QBER calculation, Gaussian Process inference, or retry decisions.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from .hardware_fallback import NullLEDController


DEFAULT_BAUD_RATE = 115_200
DEFAULT_TIMEOUT_SECONDS = 1.0
DEFAULT_WRITE_TIMEOUT_SECONDS = 1.0
DEFAULT_RESET_DELAY_SECONDS = 2.0

SERIAL_PORT_ENVIRONMENT_VARIABLE = "FT_QUPAP_SERIAL_PORT"

COMMON_DEVICE_KEYWORDS = (
    "esp32",
    "arduino",
    "usb serial",
    "usb-serial",
    "cp210",
    "cp210x",
    "ch340",
    "ch341",
    "silicon labs",
    "ftdi",
    "uart",
)


class SerialConnectionError(RuntimeError):
    """Raised when the physical serial connection cannot be created."""


@dataclass(frozen=True)
class SerialPortInformation:
    """Non-secret information about one detected serial device."""

    device: str
    description: str
    manufacturer: str | None = None
    product: str | None = None
    serial_number: str | None = None
    hardware_id: str | None = None
    score: int = 0

    def to_dictionary(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


def open_serial(
    port: str | None = None,
    *,
    baudrate: int = DEFAULT_BAUD_RATE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    write_timeout: float = DEFAULT_WRITE_TIMEOUT_SECONDS,
    reset_delay: float = DEFAULT_RESET_DELAY_SECONDS,
    use_fallback: bool = True,
    fallback_echo: bool = False,
    exclusive: bool | None = None,
) -> Any:
    """
    Open the FT-QuPAP hardware serial connection.

    Port resolution order:

    1. Explicit ``port`` argument
    2. ``FT_QUPAP_SERIAL_PORT`` environment variable
    3. Automatically detected ESP32/Arduino-compatible serial port
    4. NullLEDController fallback, when enabled

    Args:
        port:
            Explicit serial device such as ``COM5`` or
            ``/dev/ttyUSB0``.

        baudrate:
            Serial communication speed.

        timeout:
            Read timeout in seconds.

        write_timeout:
            Write timeout in seconds.

        reset_delay:
            Delay after opening the serial port. ESP32 and Arduino
            boards may reset when the connection is established.

        use_fallback:
            Return NullLEDController when hardware is unavailable.

        fallback_echo:
            Print fallback LED commands to the terminal.

        exclusive:
            Optional PySerial exclusive-access setting on platforms
            that support it.

    Returns:
        A PySerial ``Serial`` object or ``NullLEDController``.

    Raises:
        SerialConnectionError:
            When physical serial communication fails and fallback mode
            is disabled.
    """

    normalized_baudrate = validate_positive_integer(
        "baudrate",
        baudrate,
    )

    normalized_timeout = validate_nonnegative_number(
        "timeout",
        timeout,
    )

    normalized_write_timeout = validate_nonnegative_number(
        "write_timeout",
        write_timeout,
    )

    normalized_reset_delay = validate_nonnegative_number(
        "reset_delay",
        reset_delay,
    )

    if not isinstance(use_fallback, bool):
        raise TypeError("use_fallback must be a boolean.")

    if not isinstance(fallback_echo, bool):
        raise TypeError("fallback_echo must be a boolean.")

    if exclusive is not None and not isinstance(exclusive, bool):
        raise TypeError(
            "exclusive must be a boolean or None."
        )

    try:
        serial_module = import_pyserial()
    except SerialConnectionError:
        return _fallback_or_raise(
            use_fallback=use_fallback,
            fallback_echo=fallback_echo,
            message=(
                "PySerial is not installed. Install it with "
                "'pip install pyserial'."
            ),
        )

    resolved_port = resolve_serial_port(port)

    if resolved_port is None:
        return _fallback_or_raise(
            use_fallback=use_fallback,
            fallback_echo=fallback_echo,
            message=(
                "No suitable ESP32 or serial controller "
                "was detected."
            ),
        )

    serial_arguments: dict[str, Any] = {
        "port": resolved_port,
        "baudrate": normalized_baudrate,
        "timeout": normalized_timeout,
        "write_timeout": normalized_write_timeout,
    }

    if exclusive is not None:
        serial_arguments["exclusive"] = exclusive

    try:
        connection = serial_module.Serial(
            **serial_arguments
        )

        if not connection.is_open:
            connection.open()

        if normalized_reset_delay > 0.0:
            time.sleep(normalized_reset_delay)

        reset_serial_buffers(connection)

        return connection

    except Exception as error:
        return _fallback_or_raise(
            use_fallback=use_fallback,
            fallback_echo=fallback_echo,
            message=(
                "Unable to open FT-QuPAP hardware port "
                f"'{resolved_port}': {error}"
            ),
            cause=error,
        )


def resolve_serial_port(
    port: str | None = None,
) -> str | None:
    """
    Resolve the physical serial device to use.

    An explicit argument has priority over the environment variable and
    automatic device discovery.
    """

    if port is not None:
        return normalize_port_name(port)

    environment_port = os.getenv(
        SERIAL_PORT_ENVIRONMENT_VARIABLE
    )

    if environment_port:
        return normalize_port_name(environment_port)

    detected_port = detect_preferred_serial_port()

    if detected_port is None:
        return None

    return detected_port.device


def list_serial_ports() -> list[SerialPortInformation]:
    """
    Return detected serial ports ordered by FT-QuPAP suitability.

    Ports whose descriptions resemble ESP32, Arduino, CP210x, CH340,
    FTDI, or USB-UART devices receive a higher score.
    """

    try:
        import serial.tools.list_ports
    except ImportError:
        return []

    detected_ports: list[SerialPortInformation] = []

    try:
        raw_ports = serial.tools.list_ports.comports()
    except Exception:
        return []

    for raw_port in raw_ports:
        device = str(
            getattr(raw_port, "device", "")
        ).strip()

        if not device:
            continue

        description = str(
            getattr(
                raw_port,
                "description",
                "Unknown serial device",
            )
            or "Unknown serial device"
        ).strip()

        manufacturer = normalize_optional_text(
            getattr(raw_port, "manufacturer", None)
        )

        product = normalize_optional_text(
            getattr(raw_port, "product", None)
        )

        serial_number = normalize_optional_text(
            getattr(raw_port, "serial_number", None)
        )

        hardware_id = normalize_optional_text(
            getattr(raw_port, "hwid", None)
        )

        score = calculate_port_score(
            device=device,
            description=description,
            manufacturer=manufacturer,
            product=product,
            hardware_id=hardware_id,
        )

        detected_ports.append(
            SerialPortInformation(
                device=device,
                description=description,
                manufacturer=manufacturer,
                product=product,
                serial_number=serial_number,
                hardware_id=hardware_id,
                score=score,
            )
        )

    return sorted(
        detected_ports,
        key=lambda information: (
            -information.score,
            information.device.lower(),
        ),
    )


def detect_preferred_serial_port(
) -> SerialPortInformation | None:
    """
    Select the most suitable detected hardware port.

    The highest-ranked device is returned. A device with a score of zero
    is used only when it is the sole detected serial port.
    """

    ports = list_serial_ports()

    if not ports:
        return None

    best_port = ports[0]

    if best_port.score > 0:
        return best_port

    if len(ports) == 1:
        return best_port

    return None


def calculate_port_score(
    *,
    device: str,
    description: str,
    manufacturer: str | None,
    product: str | None,
    hardware_id: str | None,
) -> int:
    """Calculate a suitability score for a serial device."""

    searchable_text = " ".join(
        value
        for value in (
            device,
            description,
            manufacturer,
            product,
            hardware_id,
        )
        if value
    ).lower()

    score = 0

    for keyword in COMMON_DEVICE_KEYWORDS:
        if keyword in searchable_text:
            score += 10

    device_lower = device.lower()

    if device_lower.startswith("com"):
        score += 1

    if "ttyusb" in device_lower:
        score += 3

    if "ttyacm" in device_lower:
        score += 3

    if "cu.usb" in device_lower:
        score += 3

    if "bluetooth" in searchable_text:
        score -= 10

    return score


def reset_serial_buffers(connection: Any) -> None:
    """
    Clear available serial input and output buffers.

    Not every serial implementation exposes both methods, so each method
    is invoked only when available.
    """

    reset_input = getattr(
        connection,
        "reset_input_buffer",
        None,
    )

    if callable(reset_input):
        try:
            reset_input()
        except Exception:
            pass

    reset_output = getattr(
        connection,
        "reset_output_buffer",
        None,
    )

    if callable(reset_output):
        try:
            reset_output()
        except Exception:
            pass


def close_serial(connection: Any) -> None:
    """
    Safely close a physical or fallback connection.
    """

    if connection is None:
        return

    close_method = getattr(connection, "close", None)

    if callable(close_method):
        close_method()


def is_fallback_connection(
    connection: Any,
) -> bool:
    """Return True when the connection is NullLEDController."""

    return isinstance(connection, NullLEDController)


def is_connection_open(
    connection: Any,
) -> bool:
    """Return whether a serial or fallback connection is open."""

    if connection is None:
        return False

    try:
        return bool(connection.is_open)
    except (AttributeError, RuntimeError):
        return False


def get_connection_description(
    connection: Any,
) -> dict[str, Any]:
    """
    Return basic connection information for the dashboard.
    """

    if connection is None:
        return {
            "connected": False,
            "mode": "none",
            "port": None,
            "baudrate": None,
        }

    fallback = is_fallback_connection(connection)

    return {
        "connected": is_connection_open(connection),
        "mode": (
            "software_fallback"
            if fallback
            else "physical_serial"
        ),
        "port": getattr(connection, "port", None),
        "baudrate": getattr(
            connection,
            "baudrate",
            None,
        ),
    }


def import_pyserial() -> Any:
    """
    Import PySerial without making it a mandatory dependency.
    """

    try:
        import serial
    except ImportError as error:
        raise SerialConnectionError(
            "PySerial is unavailable."
        ) from error

    if not hasattr(serial, "Serial"):
        raise SerialConnectionError(
            "The imported 'serial' package does not provide "
            "serial.Serial. Remove the unrelated 'serial' package "
            "and install 'pyserial'."
        )

    return serial


def normalize_port_name(port: str) -> str:
    """Validate and normalize a serial port name."""

    if not isinstance(port, str):
        raise TypeError("port must be a string.")

    normalized_port = port.strip()

    if not normalized_port:
        raise ValueError("port cannot be empty.")

    return normalized_port


def normalize_optional_text(
    value: Any,
) -> str | None:
    """Normalize optional serial-device metadata."""

    if value is None:
        return None

    normalized_value = str(value).strip()

    return normalized_value or None


def validate_positive_integer(
    name: str,
    value: int,
) -> int:
    """Validate and return a positive integer."""

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return value


def validate_nonnegative_number(
    name: str,
    value: int | float,
) -> float:
    """Validate and return a nonnegative finite number."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"{name} must be numeric.")

    normalized_value = float(value)

    if normalized_value < 0.0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    if normalized_value != normalized_value:
        raise ValueError(f"{name} cannot be NaN.")

    if normalized_value in {
        float("inf"),
        float("-inf"),
    }:
        raise ValueError(f"{name} must be finite.")

    return normalized_value


def _fallback_or_raise(
    *,
    use_fallback: bool,
    fallback_echo: bool,
    message: str,
    cause: Exception | None = None,
) -> NullLEDController:
    """Return fallback hardware or raise SerialConnectionError."""

    if use_fallback:
        return NullLEDController(
            echo=fallback_echo,
            start_open=True,
        )

    if cause is None:
        raise SerialConnectionError(message)

    raise SerialConnectionError(message) from cause


def run_self_test() -> None:
    """
    Run tests that do not require physical hardware.
    """

    fallback = open_serial(
        port="NON-EXISTENT-FT-QUPAP-PORT",
        reset_delay=0.0,
        use_fallback=True,
        fallback_echo=False,
    )

    assert isinstance(
        fallback,
        NullLEDController,
    )

    assert is_fallback_connection(fallback)
    assert is_connection_open(fallback)

    description = get_connection_description(
        fallback
    )

    assert description["connected"] is True
    assert description["mode"] == (
        "software_fallback"
    )
    assert description["port"] == "NULL-HARDWARE"

    fallback.send_decision("accepted")

    assert fallback.last_command == "GREEN"

    close_serial(fallback)

    assert not is_connection_open(fallback)

    manual_port = resolve_serial_port("COM99")

    assert manual_port == "COM99"

    assert calculate_port_score(
        device="/dev/ttyUSB0",
        description="CP210x USB to UART Bridge",
        manufacturer="Silicon Labs",
        product="CP2102",
        hardware_id="USB VID:PID",
    ) > 0

    print("Serial connection self-test passed.")


if __name__ == "__main__":
    run_self_test()