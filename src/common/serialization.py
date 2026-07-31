"""
Deterministic serialization utilities for FT-QuPAP v5.1.

Cryptographic signing, transcript hashing, KMAC generation, and key
derivation require both protocol parties to serialize the same data in
exactly the same byte order.

This module provides:

- Canonical JSON serialization
- Base64 encoding and decoding
- Hexadecimal encoding and decoding
- Byte-to-bit conversion
- Bit-to-byte conversion
- JSON-safe conversion of dataclasses, enums, NumPy values, and paths
- Recursive restoration of specially encoded byte values
"""

from __future__ import annotations

import base64
import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.common.exceptions import ProtocolValidationError
from src.common.validators import (
    validate_bit_sequence,
    validate_bit_string,
    validate_bytes,
)


# Reserved field used when bytes are represented inside JSON.
BYTES_MARKER = "__ft_qupap_bytes_base64__"

# Reserved field used when tuples are represented inside JSON.
TUPLE_MARKER = "__ft_qupap_tuple__"


def encode_base64(value: bytes) -> str:
    """
    Convert bytes into an ASCII Base64 string.

    Example:

        encode_base64(b"FT-QuPAP")
    """

    validated = validate_bytes(
        value,
        field_name="base64_input",
        minimum_length=0,
    )

    return base64.b64encode(validated).decode("ascii")


def decode_base64(value: str) -> bytes:
    """
    Convert a Base64 string back into bytes.

    Strict validation is used so malformed characters are rejected.
    """

    if not isinstance(value, str):
        raise ProtocolValidationError(
            "Base64 input must be a string.",
            details={
                "received_type": type(value).__name__,
            },
        )

    try:
        return base64.b64decode(
            value.encode("ascii"),
            validate=True,
        )
    except (ValueError, UnicodeEncodeError) as exc:
        raise ProtocolValidationError(
            "Invalid Base64 data.",
            details={
                "value_length": len(value),
            },
        ) from exc


def encode_hex(value: bytes) -> str:
    """
    Convert bytes into lowercase hexadecimal text.
    """

    validated = validate_bytes(
        value,
        field_name="hex_input",
        minimum_length=0,
    )

    return validated.hex()


def decode_hex(value: str) -> bytes:
    """
    Convert hexadecimal text into bytes.
    """

    if not isinstance(value, str):
        raise ProtocolValidationError(
            "Hex input must be a string.",
            details={
                "received_type": type(value).__name__,
            },
        )

    normalized = value.strip()

    if len(normalized) % 2 != 0:
        raise ProtocolValidationError(
            "Hexadecimal text must have an even length.",
            details={
                "length": len(normalized),
            },
        )

    try:
        return bytes.fromhex(normalized)
    except ValueError as exc:
        raise ProtocolValidationError(
            "Invalid hexadecimal data.",
            details={
                "value_length": len(normalized),
            },
        ) from exc


def bytes_to_bits(value: bytes) -> list[int]:
    """
    Convert bytes into a big-endian list of classical bits.

    Example:

        bytes_to_bits(bytes([5]))

    returns:

        [0, 0, 0, 0, 0, 1, 0, 1]
    """

    validated = validate_bytes(
        value,
        field_name="byte_value",
        minimum_length=0,
    )

    bits: list[int] = []

    for byte_value in validated:
        for shift in range(7, -1, -1):
            bits.append(
                (byte_value >> shift) & 1
            )

    return bits


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    """
    Convert a big-endian bit sequence into bytes.

    The number of bits must be divisible by eight.
    """

    normalized = validate_bit_sequence(
        bits,
        field_name="bits",
    )

    if len(normalized) % 8 != 0:
        raise ProtocolValidationError(
            "The number of bits must be divisible by eight.",
            details={
                "bit_count": len(normalized),
            },
        )

    output = bytearray()

    for start_index in range(0, len(normalized), 8):
        byte_bits = normalized[
            start_index:start_index + 8
        ]

        byte_value = 0

        for bit in byte_bits:
            byte_value = (
                byte_value << 1
            ) | bit

        output.append(byte_value)

    return bytes(output)


def bytes_to_bit_string(value: bytes) -> str:
    """
    Convert bytes into text containing only 0 and 1.
    """

    return "".join(
        str(bit)
        for bit in bytes_to_bits(value)
    )


def bit_string_to_bytes(bit_string: str) -> bytes:
    """
    Convert a textual bit string into bytes.
    """

    normalized = validate_bit_string(
        bit_string,
        field_name="bit_string",
    )

    return bits_to_bytes(
        [int(bit) for bit in normalized]
    )


def _convert_numpy_value(value: Any) -> Any:
    """
    Convert NumPy values without making NumPy a mandatory import here.

    Detection is based on the object's module name.
    """

    value_type = type(value)
    module_name = getattr(
        value_type,
        "__module__",
        "",
    )

    if not module_name.startswith("numpy"):
        return value

    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass

    if hasattr(value, "tolist"):
        return value.tolist()

    return value


def to_json_safe(
    value: Any,
    *,
    preserve_tuples: bool = False,
) -> Any:
    """
    Recursively convert a Python value into a JSON-compatible value.

    Bytes are encoded using a reserved Base64 marker:

        {
            "__ft_qupap_bytes_base64__": "..."
        }

    This prevents byte values from being confused with ordinary strings.
    """

    value = _convert_numpy_value(value)

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, bytes):
        return {
            BYTES_MARKER: encode_base64(value),
        }

    if isinstance(value, bytearray):
        return {
            BYTES_MARKER: encode_base64(
                bytes(value)
            ),
        }

    if isinstance(value, memoryview):
        return {
            BYTES_MARKER: encode_base64(
                value.tobytes()
            ),
        }

    if isinstance(value, Enum):
        return to_json_safe(
            value.value,
            preserve_tuples=preserve_tuples,
        )

    if isinstance(value, Path):
        return str(value)

    if dataclasses.is_dataclass(value):
        return to_json_safe(
            dataclasses.asdict(value),
            preserve_tuples=preserve_tuples,
        )

    if isinstance(value, Mapping):
        converted_mapping: dict[str, Any] = {}

        for key, item in value.items():
            if isinstance(key, Enum):
                normalized_key = str(key.value)
            else:
                normalized_key = str(key)

            converted_mapping[normalized_key] = (
                to_json_safe(
                    item,
                    preserve_tuples=preserve_tuples,
                )
            )

        return converted_mapping

    if isinstance(value, tuple):
        converted_items = [
            to_json_safe(
                item,
                preserve_tuples=preserve_tuples,
            )
            for item in value
        ]

        if preserve_tuples:
            return {
                TUPLE_MARKER: converted_items,
            }

        return converted_items

    if isinstance(value, (list, set, frozenset)):
        return [
            to_json_safe(
                item,
                preserve_tuples=preserve_tuples,
            )
            for item in value
        ]

    if hasattr(value, "to_dict"):
        try:
            converted = value.to_dict()
        except TypeError:
            converted = None

        if converted is not None:
            return to_json_safe(
                converted,
                preserve_tuples=preserve_tuples,
            )

    if hasattr(value, "__dict__"):
        return to_json_safe(
            vars(value),
            preserve_tuples=preserve_tuples,
        )

    raise TypeError(
        "Value cannot be converted to JSON safely: "
        f"{type(value).__name__}"
    )


def restore_json_types(value: Any) -> Any:
    """
    Restore values created by `to_json_safe`.

    Specifically restores:

    - Base64-marked byte values
    - Tuple-marked values
    """

    if isinstance(value, list):
        return [
            restore_json_types(item)
            for item in value
        ]

    if not isinstance(value, dict):
        return value

    if set(value.keys()) == {BYTES_MARKER}:
        encoded_value = value[BYTES_MARKER]

        if not isinstance(encoded_value, str):
            raise ProtocolValidationError(
                "Encoded byte marker must contain a string."
            )

        return decode_base64(encoded_value)

    if set(value.keys()) == {TUPLE_MARKER}:
        tuple_values = value[TUPLE_MARKER]

        if not isinstance(tuple_values, list):
            raise ProtocolValidationError(
                "Encoded tuple marker must contain a list."
            )

        return tuple(
            restore_json_types(item)
            for item in tuple_values
        )

    return {
        key: restore_json_types(item)
        for key, item in value.items()
    }


def canonical_json_text(value: Any) -> str:
    """
    Serialize a value into deterministic canonical JSON text.

    Deterministic settings:

    - Keys sorted alphabetically
    - No unnecessary whitespace
    - UTF-8 characters preserved
    - NaN and Infinity rejected
    """

    json_safe_value = to_json_safe(
        value,
        preserve_tuples=False,
    )

    try:
        return json.dumps(
            json_safe_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError(
            "Unable to create canonical JSON.",
            details={
                "reason": str(exc),
            },
        ) from exc


def canonical_json_bytes(value: Any) -> bytes:
    """
    Serialize a value into deterministic UTF-8 JSON bytes.

    These bytes can safely be used for:

    - ML-DSA signing
    - Transcript hashing
    - KMAC input
    - HKDF information
    """

    return canonical_json_text(value).encode(
        "utf-8"
    )


def pretty_json_text(
    value: Any,
    *,
    indent: int = 2,
) -> str:
    """
    Serialize a value into human-readable JSON.

    This function is intended for logs, files, and dashboards.
    It must not be used as cryptographic canonical input.
    """

    json_safe_value = to_json_safe(
        value,
        preserve_tuples=True,
    )

    return json.dumps(
        json_safe_value,
        indent=indent,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )


def parse_json_text(
    text: str,
    *,
    restore_special_types: bool = True,
) -> Any:
    """
    Parse JSON text and optionally restore bytes and tuples.
    """

    if not isinstance(text, str):
        raise ProtocolValidationError(
            "JSON input must be a string.",
            details={
                "received_type": type(text).__name__,
            },
        )

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolValidationError(
            "Invalid JSON text.",
            details={
                "line": exc.lineno,
                "column": exc.colno,
                "reason": exc.msg,
            },
        ) from exc

    if restore_special_types:
        return restore_json_types(value)

    return value


def parse_json_bytes(
    data: bytes,
    *,
    restore_special_types: bool = True,
) -> Any:
    """
    Parse UTF-8 JSON bytes.
    """

    validated = validate_bytes(
        data,
        field_name="json_bytes",
        minimum_length=1,
    )

    try:
        text = validated.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolValidationError(
            "JSON bytes must contain valid UTF-8 text."
        ) from exc

    return parse_json_text(
        text,
        restore_special_types=restore_special_types,
    )


def save_json_file(
    path: Path | str,
    value: Any,
    *,
    indent: int = 2,
) -> Path:
    """
    Save a value as human-readable JSON.

    Parent directories are created automatically.
    """

    target_path = Path(path)

    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_path.write_text(
        pretty_json_text(
            value,
            indent=indent,
        ),
        encoding="utf-8",
    )

    return target_path


def load_json_file(
    path: Path | str,
    *,
    restore_special_types: bool = True,
) -> Any:
    """
    Load a JSON file.

    Raises FileNotFoundError when the requested file does not exist.
    """

    source_path = Path(path)

    text = source_path.read_text(
        encoding="utf-8"
    )

    return parse_json_text(
        text,
        restore_special_types=restore_special_types,
    )


# Short aliases used by protocol modules.
b64e = encode_base64
b64d = decode_base64

bits_from_bytes = bytes_to_bits
bytes_from_bits = bits_to_bytes


__all__ = [
    "BYTES_MARKER",
    "TUPLE_MARKER",
    "encode_base64",
    "decode_base64",
    "encode_hex",
    "decode_hex",
    "bytes_to_bits",
    "bits_to_bytes",
    "bytes_to_bit_string",
    "bit_string_to_bytes",
    "to_json_safe",
    "restore_json_types",
    "canonical_json_text",
    "canonical_json_bytes",
    "pretty_json_text",
    "parse_json_text",
    "parse_json_bytes",
    "save_json_file",
    "load_json_file",
    "b64e",
    "b64d",
    "bits_from_bytes",
    "bytes_from_bits",
]