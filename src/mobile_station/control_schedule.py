"""
Control Schedule Module
FT-QuPAP Mobile Station

This module implements the protected FT-QuPAP control schedule.

Protocol sequence:

1. Randomly interleave:
       - 128 logical KMAC payload blocks
       - 32 independent logical check blocks

2. Record:
       - complete block ordering
       - check-block positions
       - check measurement bases
       - expected logical check bits
       - payload logical indices

3. After Steane encoding, insert each check block's expected
   physical reference pattern.

4. Encrypt the completed schedule using AES-256-GCM with K_ctrl.

5. Bind the encrypted schedule to H(Transcript) using SHA3-256.

The protected schedule is delivered to the Authentication Server
before quantum-block arrival.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


PAYLOAD_BLOCK_COUNT = 128
CHECK_BLOCK_COUNT = 32

TOTAL_LOGICAL_BLOCK_COUNT = (
    PAYLOAD_BLOCK_COUNT
    + CHECK_BLOCK_COUNT
)

STEANE_BLOCK_SIZE = 7

K_CTRL_LENGTH_BYTES = 32
TRANSCRIPT_HASH_LENGTH_BYTES = 32
AES_GCM_NONCE_LENGTH_BYTES = 12

SUPPORTED_BASES = (
    "Z",
    "X",
)

DEFAULT_RANDOM_SEED = 20260701


class ControlScheduleError(Exception):
    """Base exception for control-schedule processing."""


class ControlScheduleValidationError(
    ControlScheduleError
):
    """Raised when a control schedule is structurally invalid."""


class ControlScheduleEncryptionError(
    ControlScheduleError
):
    """Raised when control-schedule encryption fails."""


@dataclass
class LogicalSpec:
    """
    Description of one logical FT-QuPAP block.

    Attributes:
        block_id:
            Unique block identifier.

        role:
            Either "payload" or "check".

        logical_index:
            Original index inside its payload/check collection.

        logical_bit:
            Logical bit value, either 0 or 1.

        basis:
            Logical preparation basis, either Z or X.

        position:
            Position assigned after random interleaving.
    """

    block_id: str
    role: str
    logical_index: int
    logical_bit: int
    basis: str
    position: int | None = None

    def __post_init__(self) -> None:
        validate_logical_spec(self)


@dataclass(frozen=True)
class ProtectedControlSchedule:
    """
    Encrypted and transcript-bound control-schedule package.

    Attributes:
        encrypted_schedule:
            Dictionary containing Base64 AES-GCM nonce and ciphertext.

        schedule_binding:
            SHA3-256 hexadecimal transcript/control binding.
    """

    encrypted_schedule: dict[str, str]
    schedule_binding: str

    def __post_init__(self) -> None:
        validate_encrypted_schedule(
            self.encrypted_schedule
        )

        validate_schedule_binding(
            self.schedule_binding
        )

    def to_transport_dictionary(
        self,
    ) -> dict[str, Any]:
        """
        Return the public classical transport representation.
        """

        return {
            "encrypted_schedule": {
                "nonce":
                    self.encrypted_schedule[
                        "nonce"
                    ],
                "ciphertext":
                    self.encrypted_schedule[
                        "ciphertext"
                    ],
            },
            "schedule_binding":
                self.schedule_binding,
        }

    def safe_summary(self) -> dict[str, Any]:
        """
        Return non-sensitive schedule metadata.

        Decrypted block positions, bases, and expected patterns are
        deliberately excluded.
        """

        nonce = decode_base64(
            self.encrypted_schedule["nonce"],
            "nonce",
        )

        ciphertext = decode_base64(
            self.encrypted_schedule[
                "ciphertext"
            ],
            "ciphertext",
        )

        return {
            "encryption_algorithm":
                "AES-256-GCM",
            "nonce_length_bytes":
                len(nonce),
            "ciphertext_length_bytes":
                len(ciphertext),
            "schedule_binding_prefix":
                self.schedule_binding[:16],
        }


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """
    Serialize a mapping into deterministic UTF-8 JSON.

    Both the Mobile Station and Authentication Server must use the
    same settings:

        sort_keys=True
        separators=(",", ":")
    """

    if not isinstance(value, Mapping):
        raise TypeError(
            "value must be a mapping."
        )

    try:
        serialized = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ControlScheduleValidationError(
            "Control schedule cannot be serialized."
        ) from error

    return serialized.encode("utf-8")


def encode_base64(
    data: bytes,
) -> str:
    """Encode non-empty bytes as Base64 text."""

    if not isinstance(data, bytes):
        raise TypeError(
            "data must be bytes."
        )

    if not data:
        raise ValueError(
            "data cannot be empty."
        )

    return base64.b64encode(
        data
    ).decode("ascii")


def decode_base64(
    encoded_value: str,
    field_name: str,
) -> bytes:
    """Decode strict Base64 text."""

    if not isinstance(encoded_value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    if not encoded_value:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    try:
        return base64.b64decode(
            encoded_value.encode("ascii"),
            validate=True,
        )

    except Exception as error:
        raise ValueError(
            f"{field_name} is not valid Base64."
        ) from error


def hash_hex(
    data: bytes,
) -> str:
    """Return a SHA3-256 hexadecimal digest."""

    if not isinstance(data, bytes):
        raise TypeError(
            "data must be bytes."
        )

    return hashlib.sha3_256(
        data
    ).hexdigest()


def validate_k_ctrl(
    k_ctrl: bytes,
) -> None:
    """
    Validate the 32-byte FT-QuPAP control key.
    """

    if not isinstance(k_ctrl, bytes):
        raise TypeError(
            "k_ctrl must be bytes."
        )

    if len(k_ctrl) != (
        K_CTRL_LENGTH_BYTES
    ):
        raise ValueError(
            "K_ctrl must contain exactly "
            f"{K_CTRL_LENGTH_BYTES} bytes."
        )


def validate_transcript_hash(
    transcript_hash: bytes,
) -> None:
    """
    Validate the SHA3-256 transcript hash.
    """

    if not isinstance(
        transcript_hash,
        bytes,
    ):
        raise TypeError(
            "transcript_hash must be bytes."
        )

    if len(transcript_hash) != (
        TRANSCRIPT_HASH_LENGTH_BYTES
    ):
        raise ValueError(
            "transcript_hash must contain exactly "
            f"{TRANSCRIPT_HASH_LENGTH_BYTES} bytes."
        )


def validate_logical_spec(
    spec: Any,
) -> None:
    """
    Validate one payload or check logical specification.
    """

    required_attributes = (
        "block_id",
        "role",
        "logical_index",
        "logical_bit",
        "basis",
        "position",
    )

    for attribute in required_attributes:
        if not hasattr(spec, attribute):
            raise TypeError(
                "Logical specification is missing "
                f"{attribute!r}."
            )

    if not isinstance(spec.block_id, str):
        raise TypeError(
            "block_id must be a string."
        )

    if not spec.block_id:
        raise ValueError(
            "block_id cannot be empty."
        )

    if spec.role not in (
        "payload",
        "check",
    ):
        raise ValueError(
            "role must be 'payload' or 'check'."
        )

    if isinstance(
        spec.logical_index,
        bool,
    ) or not isinstance(
        spec.logical_index,
        int,
    ):
        raise TypeError(
            "logical_index must be an integer."
        )

    if spec.logical_index < 0:
        raise ValueError(
            "logical_index cannot be negative."
        )

    if spec.logical_bit not in (
        0,
        1,
    ):
        raise ValueError(
            "logical_bit must be 0 or 1."
        )

    if spec.basis not in SUPPORTED_BASES:
        raise ValueError(
            "basis must be Z or X."
        )

    if (
        spec.role == "payload"
        and spec.basis != "Z"
    ):
        raise ValueError(
            "FT-QuPAP payload blocks must use "
            "the Z basis."
        )

    if spec.position is not None:
        if isinstance(
            spec.position,
            bool,
        ) or not isinstance(
            spec.position,
            int,
        ):
            raise TypeError(
                "position must be an integer or None."
            )

        if spec.position < 0:
            raise ValueError(
                "position cannot be negative."
            )


def _validate_logical_collection(
    specifications: Sequence[Any],
    expected_role: str,
    expected_count: int,
    require_standard_count: bool,
) -> None:
    """
    Validate one payload/check logical collection.
    """

    if isinstance(
        specifications,
        (str, bytes, bytearray),
    ):
        raise TypeError(
            "Logical specifications must be a sequence."
        )

    if not isinstance(
        specifications,
        Sequence,
    ):
        raise TypeError(
            "Logical specifications must be a sequence."
        )

    if len(specifications) == 0:
        raise ValueError(
            "Logical specification sequence cannot be empty."
        )

    if (
        require_standard_count
        and len(specifications) != expected_count
    ):
        raise ControlScheduleValidationError(
            f"FT-QuPAP requires exactly "
            f"{expected_count} {expected_role} blocks."
        )

    seen_ids: set[str] = set()
    seen_indices: set[int] = set()

    for spec in specifications:
        validate_logical_spec(spec)

        if spec.role != expected_role:
            raise ControlScheduleValidationError(
                f"Every item must use "
                f"role={expected_role!r}."
            )

        if spec.block_id in seen_ids:
            raise ControlScheduleValidationError(
                f"Duplicate block ID: "
                f"{spec.block_id!r}."
            )

        if spec.logical_index in seen_indices:
            raise ControlScheduleValidationError(
                f"Duplicate {expected_role} logical index: "
                f"{spec.logical_index}."
            )

        seen_ids.add(spec.block_id)
        seen_indices.add(
            spec.logical_index
        )


def create_interleaved_schedule(
    payload_specs: Sequence[Any],
    check_specs: Sequence[Any],
    rng: np.random.Generator | None = None,
    validate_standard_counts: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Randomly interleave payload and check logical blocks.

    Notebook-aligned output:

        ordered_specs
        schedule

    The schedule initially contains logical check information.
    Expected physical check patterns are inserted after Steane
    encoding by attach_expected_reference_bits().
    """

    _validate_logical_collection(
        specifications=payload_specs,
        expected_role="payload",
        expected_count=PAYLOAD_BLOCK_COUNT,
        require_standard_count=
            validate_standard_counts,
    )

    _validate_logical_collection(
        specifications=check_specs,
        expected_role="check",
        expected_count=CHECK_BLOCK_COUNT,
        require_standard_count=
            validate_standard_counts,
    )

    all_ids = [
        spec.block_id
        for spec in (
            list(payload_specs)
            + list(check_specs)
        )
    ]

    if len(all_ids) != len(set(all_ids)):
        raise ControlScheduleValidationError(
            "Payload and check block IDs must be "
            "globally unique."
        )

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a NumPy Generator."
        )

    all_specs = [
        copy.deepcopy(spec)
        for spec in (
            list(payload_specs)
            + list(check_specs)
        )
    ]

    permutation = rng.permutation(
        len(all_specs)
    )

    ordered_specs = [
        all_specs[int(index)]
        for index in permutation
    ]

    for position, spec in enumerate(
        ordered_specs
    ):
        spec.position = position

    schedule = {
        "ordered_block_ids": [
            spec.block_id
            for spec in ordered_specs
        ],
        "check_blocks": [
            {
                "block_id":
                    spec.block_id,
                "position":
                    int(spec.position),
                "basis":
                    spec.basis,
                "expected_logical_bit":
                    int(spec.logical_bit),
            }
            for spec in ordered_specs
            if spec.role == "check"
        ],
        "payload_blocks": [
            {
                "block_id":
                    spec.block_id,
                "position":
                    int(spec.position),
                "logical_index":
                    int(spec.logical_index),
            }
            for spec in ordered_specs
            if spec.role == "payload"
        ],
    }

    validate_control_schedule(
        schedule=schedule,
        require_reference_bits=False,
        require_standard_counts=
            validate_standard_counts,
    )

    return ordered_specs, schedule


def _normalize_reference_bits(
    reference_bits: Any,
) -> list[int]:
    """
    Convert a physical reference pattern into a binary list.
    """

    if isinstance(
        reference_bits,
        np.ndarray,
    ):
        values = (
            reference_bits
            .astype(np.int8)
            .tolist()
        )

    elif isinstance(
        reference_bits,
        Sequence,
    ) and not isinstance(
        reference_bits,
        (str, bytes, bytearray),
    ):
        values = list(reference_bits)

    else:
        raise TypeError(
            "reference_bits must be a sequence "
            "or NumPy array."
        )

    if len(values) == 0:
        raise ValueError(
            "reference_bits cannot be empty."
        )

    normalized: list[int] = []

    for index, value in enumerate(values):
        if isinstance(value, bool):
            bit = int(value)

        elif isinstance(
            value,
            (int, np.integer),
        ):
            bit = int(value)

        else:
            raise TypeError(
                f"Reference bit at index {index} "
                "must be an integer."
            )

        if bit not in (0, 1):
            raise ValueError(
                f"Reference bit at index {index} "
                "must be 0 or 1."
            )

        normalized.append(bit)

    return normalized


def attach_expected_reference_bits(
    schedule: Mapping[str, Any],
    encoded_check_blocks: Sequence[Any],
    require_steane_blocks: bool = True,
) -> dict[str, Any]:
    """
    Add encoded physical check patterns to the schedule.

    This operation must occur after check-block Steane encoding and
    before AES-GCM schedule encryption.

    Each encoded block must contain:

        block.spec.block_id
        block.reference_bits
    """

    validate_control_schedule(
        schedule=schedule,
        require_reference_bits=False,
        require_standard_counts=True,
    )

    if isinstance(
        encoded_check_blocks,
        (str, bytes, bytearray),
    ):
        raise TypeError(
            "encoded_check_blocks must be a sequence."
        )

    if not isinstance(
        encoded_check_blocks,
        Sequence,
    ):
        raise TypeError(
            "encoded_check_blocks must be a sequence."
        )

    if len(encoded_check_blocks) != (
        CHECK_BLOCK_COUNT
    ):
        raise ControlScheduleValidationError(
            "Expected exactly "
            f"{CHECK_BLOCK_COUNT} encoded check blocks."
        )

    blocks_by_id: dict[str, Any] = {}

    for block in encoded_check_blocks:
        if not hasattr(block, "spec"):
            raise TypeError(
                "Encoded check block is missing spec."
            )

        if not hasattr(
            block.spec,
            "block_id",
        ):
            raise TypeError(
                "Encoded check block spec is missing block_id."
            )

        if not hasattr(
            block,
            "reference_bits",
        ):
            raise TypeError(
                "Encoded check block is missing "
                "reference_bits."
            )

        block_id = block.spec.block_id

        if block_id in blocks_by_id:
            raise ControlScheduleValidationError(
                f"Duplicate encoded check block: "
                f"{block_id!r}."
            )

        normalized_bits = (
            _normalize_reference_bits(
                block.reference_bits
            )
        )

        expected_length = (
            STEANE_BLOCK_SIZE
            if require_steane_blocks
            else 1
        )

        if len(normalized_bits) != expected_length:
            raise ControlScheduleValidationError(
                f"Encoded check block {block_id!r} "
                f"must contain {expected_length} "
                "physical reference bits."
            )

        blocks_by_id[block_id] = (
            normalized_bits
        )

    completed_schedule = copy.deepcopy(
        dict(schedule)
    )

    for check_entry in completed_schedule[
        "check_blocks"
    ]:
        block_id = check_entry[
            "block_id"
        ]

        if block_id not in blocks_by_id:
            raise ControlScheduleValidationError(
                f"Encoded check block "
                f"{block_id!r} is missing."
            )

        check_entry[
            "expected_reference_bits"
        ] = list(
            blocks_by_id[block_id]
        )

    validate_control_schedule(
        schedule=completed_schedule,
        require_reference_bits=True,
        require_standard_counts=True,
    )

    return completed_schedule


def validate_control_schedule(
    schedule: Mapping[str, Any],
    require_reference_bits: bool = True,
    require_standard_counts: bool = True,
) -> None:
    """
    Validate the plaintext FT-QuPAP control schedule.
    """

    if not isinstance(schedule, Mapping):
        raise TypeError(
            "schedule must be a mapping."
        )

    required_top_level_fields = {
        "ordered_block_ids",
        "check_blocks",
        "payload_blocks",
    }

    if not required_top_level_fields.issubset(
        schedule.keys()
    ):
        missing = (
            required_top_level_fields
            .difference(schedule.keys())
        )

        raise ControlScheduleValidationError(
            "Control schedule is missing fields: "
            f"{sorted(missing)}"
        )

    ordered_ids = schedule[
        "ordered_block_ids"
    ]

    check_entries = schedule[
        "check_blocks"
    ]

    payload_entries = schedule[
        "payload_blocks"
    ]

    for field_name, value in (
        ("ordered_block_ids", ordered_ids),
        ("check_blocks", check_entries),
        ("payload_blocks", payload_entries),
    ):
        if not isinstance(value, list):
            raise TypeError(
                f"{field_name} must be a list."
            )

    if require_standard_counts:
        if len(payload_entries) != (
            PAYLOAD_BLOCK_COUNT
        ):
            raise ControlScheduleValidationError(
                "Control schedule must contain "
                f"{PAYLOAD_BLOCK_COUNT} payload entries."
            )

        if len(check_entries) != (
            CHECK_BLOCK_COUNT
        ):
            raise ControlScheduleValidationError(
                "Control schedule must contain "
                f"{CHECK_BLOCK_COUNT} check entries."
            )

        if len(ordered_ids) != (
            TOTAL_LOGICAL_BLOCK_COUNT
        ):
            raise ControlScheduleValidationError(
                "Control schedule must contain "
                f"{TOTAL_LOGICAL_BLOCK_COUNT} "
                "ordered block IDs."
            )

    if len(ordered_ids) != len(
        check_entries + payload_entries
    ):
        raise ControlScheduleValidationError(
            "Ordered block count does not match "
            "schedule entry count."
        )

    if len(ordered_ids) != len(
        set(ordered_ids)
    ):
        raise ControlScheduleValidationError(
            "ordered_block_ids contains duplicates."
        )

    used_positions: set[int] = set()
    used_ids: set[str] = set()

    def validate_entry_position(
        entry: Mapping[str, Any],
    ) -> None:
        if not isinstance(entry, Mapping):
            raise TypeError(
                "Every schedule entry must be a mapping."
            )

        if "block_id" not in entry:
            raise ControlScheduleValidationError(
                "Schedule entry is missing block_id."
            )

        if "position" not in entry:
            raise ControlScheduleValidationError(
                "Schedule entry is missing position."
            )

        block_id = entry["block_id"]
        position = entry["position"]

        if not isinstance(block_id, str):
            raise TypeError(
                "Schedule block_id must be a string."
            )

        if isinstance(
            position,
            bool,
        ) or not isinstance(
            position,
            int,
        ):
            raise TypeError(
                "Schedule position must be an integer."
            )

        if not 0 <= position < len(
            ordered_ids
        ):
            raise ControlScheduleValidationError(
                "Schedule position is outside the "
                "ordered frame."
            )

        if ordered_ids[position] != block_id:
            raise ControlScheduleValidationError(
                "Schedule position does not match "
                "ordered_block_ids."
            )

        if position in used_positions:
            raise ControlScheduleValidationError(
                "Duplicate schedule position."
            )

        if block_id in used_ids:
            raise ControlScheduleValidationError(
                "Duplicate scheduled block ID."
            )

        used_positions.add(position)
        used_ids.add(block_id)

    for entry in check_entries:
        validate_entry_position(entry)

        required_fields = {
            "block_id",
            "position",
            "basis",
            "expected_logical_bit",
        }

        if not required_fields.issubset(
            entry.keys()
        ):
            raise ControlScheduleValidationError(
                "Check entry is missing required fields."
            )

        if entry["basis"] not in (
            SUPPORTED_BASES
        ):
            raise ControlScheduleValidationError(
                "Check basis must be Z or X."
            )

        if entry[
            "expected_logical_bit"
        ] not in (0, 1):
            raise ControlScheduleValidationError(
                "Expected logical check bit must "
                "be 0 or 1."
            )

        if require_reference_bits:
            if (
                "expected_reference_bits"
                not in entry
            ):
                raise ControlScheduleValidationError(
                    "Check entry is missing "
                    "expected_reference_bits."
                )

            reference_bits = (
                _normalize_reference_bits(
                    entry[
                        "expected_reference_bits"
                    ]
                )
            )

            if len(reference_bits) not in (
                1,
                STEANE_BLOCK_SIZE,
            ):
                raise ControlScheduleValidationError(
                    "Expected physical check pattern "
                    "must contain one baseline bit or "
                    "seven Steane bits."
                )

    for entry in payload_entries:
        validate_entry_position(entry)

        required_fields = {
            "block_id",
            "position",
            "logical_index",
        }

        if not required_fields.issubset(
            entry.keys()
        ):
            raise ControlScheduleValidationError(
                "Payload entry is missing required fields."
            )

        logical_index = entry[
            "logical_index"
        ]

        if isinstance(
            logical_index,
            bool,
        ) or not isinstance(
            logical_index,
            int,
        ):
            raise TypeError(
                "Payload logical_index must be an integer."
            )

        if logical_index < 0:
            raise ControlScheduleValidationError(
                "Payload logical_index cannot be negative."
            )

    if used_positions != set(
        range(len(ordered_ids))
    ):
        raise ControlScheduleValidationError(
            "Control schedule does not cover every "
            "ordered frame position."
        )


def encrypt_control_schedule(
    schedule: Mapping[str, Any],
    k_ctrl: bytes,
    transcript_hash: bytes,
    nonce: bytes | None = None,
) -> dict[str, str]:
    """
    Encrypt the completed schedule using AES-256-GCM.

    H(Transcript) is used as authenticated associated data.

    Therefore, decryption fails when the encrypted schedule is moved
    to a different FT-QuPAP session.
    """

    validate_control_schedule(
        schedule=schedule,
        require_reference_bits=True,
        require_standard_counts=True,
    )

    validate_k_ctrl(k_ctrl)

    validate_transcript_hash(
        transcript_hash
    )

    if nonce is None:
        schedule_nonce = (
            secrets.token_bytes(
                AES_GCM_NONCE_LENGTH_BYTES
            )
        )
    else:
        if not isinstance(nonce, bytes):
            raise TypeError(
                "nonce must be bytes."
            )

        schedule_nonce = bytes(nonce)

    if len(schedule_nonce) != (
        AES_GCM_NONCE_LENGTH_BYTES
    ):
        raise ValueError(
            "AES-GCM nonce must contain exactly "
            f"{AES_GCM_NONCE_LENGTH_BYTES} bytes."
        )

    plaintext = canonical_json_bytes(
        schedule
    )

    try:
        ciphertext = AESGCM(
            k_ctrl
        ).encrypt(
            schedule_nonce,
            plaintext,
            transcript_hash,
        )

    except Exception as error:
        raise ControlScheduleEncryptionError(
            "AES-GCM control-schedule encryption failed."
        ) from error

    return {
        "nonce":
            encode_base64(
                schedule_nonce
            ),
        "ciphertext":
            encode_base64(
                ciphertext
            ),
    }


def validate_encrypted_schedule(
    encrypted_schedule: Mapping[str, Any],
) -> None:
    """
    Validate an encrypted schedule transport object.
    """

    if not isinstance(
        encrypted_schedule,
        Mapping,
    ):
        raise TypeError(
            "encrypted_schedule must be a mapping."
        )

    required_fields = {
        "nonce",
        "ciphertext",
    }

    if set(
        encrypted_schedule.keys()
    ) != required_fields:
        raise ControlScheduleValidationError(
            "encrypted_schedule must contain exactly "
            "'nonce' and 'ciphertext'."
        )

    nonce = decode_base64(
        encrypted_schedule["nonce"],
        "nonce",
    )

    ciphertext = decode_base64(
        encrypted_schedule["ciphertext"],
        "ciphertext",
    )

    if len(nonce) != (
        AES_GCM_NONCE_LENGTH_BYTES
    ):
        raise ControlScheduleValidationError(
            "Encrypted schedule contains an invalid "
            "AES-GCM nonce length."
        )

    if len(ciphertext) <= 16:
        raise ControlScheduleValidationError(
            "Encrypted schedule ciphertext is too short."
        )


def bind_control_schedule(
    encrypted_schedule: Mapping[str, Any],
    transcript_hash: bytes,
) -> str:
    """
    Bind the encrypted schedule to the session transcript.

    Notebook-aligned definition:

        SHA3-256(
            transcript_hash
            || decoded_nonce
            || decoded_ciphertext
        )
    """

    validate_encrypted_schedule(
        encrypted_schedule
    )

    validate_transcript_hash(
        transcript_hash
    )

    nonce = decode_base64(
        encrypted_schedule["nonce"],
        "nonce",
    )

    ciphertext = decode_base64(
        encrypted_schedule["ciphertext"],
        "ciphertext",
    )

    binding_input = (
        transcript_hash
        + nonce
        + ciphertext
    )

    return hash_hex(
        binding_input
    )


def validate_schedule_binding(
    schedule_binding: str,
) -> None:
    """Validate a SHA3-256 hexadecimal binding."""

    if not isinstance(
        schedule_binding,
        str,
    ):
        raise TypeError(
            "schedule_binding must be a string."
        )

    if len(schedule_binding) != 64:
        raise ValueError(
            "schedule_binding must contain exactly "
            "64 hexadecimal characters."
        )

    try:
        bytes.fromhex(
            schedule_binding
        )

    except ValueError as error:
        raise ValueError(
            "schedule_binding must be hexadecimal."
        ) from error


def verify_control_schedule_binding(
    encrypted_schedule: Mapping[str, Any],
    expected_binding: str,
    transcript_hash: bytes,
) -> bool:
    """
    Verify the transcript/control binding in constant time.
    """

    validate_schedule_binding(
        expected_binding
    )

    actual_binding = (
        bind_control_schedule(
            encrypted_schedule=
                encrypted_schedule,
            transcript_hash=
                transcript_hash,
        )
    )

    return hmac.compare_digest(
        actual_binding,
        expected_binding,
    )


def protect_control_schedule(
    schedule: Mapping[str, Any],
    k_ctrl: bytes,
    transcript_hash: bytes,
) -> ProtectedControlSchedule:
    """
    Encrypt and transcript-bind a completed schedule.
    """

    encrypted_schedule = (
        encrypt_control_schedule(
            schedule=schedule,
            k_ctrl=k_ctrl,
            transcript_hash=
                transcript_hash,
        )
    )

    schedule_binding = (
        bind_control_schedule(
            encrypted_schedule=
                encrypted_schedule,
            transcript_hash=
                transcript_hash,
        )
    )

    return ProtectedControlSchedule(
        encrypted_schedule=
            encrypted_schedule,
        schedule_binding=
            schedule_binding,
    )


def decrypt_for_self_test(
    encrypted_schedule: Mapping[str, Any],
    k_ctrl: bytes,
    transcript_hash: bytes,
) -> dict[str, Any]:
    """
    Decrypt a schedule for this module's local self-test.

    Authentication Server production decryption belongs in the
    Authentication Server package.
    """

    validate_encrypted_schedule(
        encrypted_schedule
    )

    validate_k_ctrl(k_ctrl)

    validate_transcript_hash(
        transcript_hash
    )

    try:
        plaintext = AESGCM(
            k_ctrl
        ).decrypt(
            decode_base64(
                encrypted_schedule["nonce"],
                "nonce",
            ),
            decode_base64(
                encrypted_schedule[
                    "ciphertext"
                ],
                "ciphertext",
            ),
            transcript_hash,
        )

        decoded = json.loads(
            plaintext.decode("utf-8")
        )

    except Exception as error:
        raise ControlScheduleEncryptionError(
            "Control-schedule self-test decryption failed."
        ) from error

    if not isinstance(decoded, dict):
        raise ControlScheduleValidationError(
            "Decrypted control schedule is not a dictionary."
        )

    validate_control_schedule(
        schedule=decoded,
        require_reference_bits=True,
        require_standard_counts=True,
    )

    return decoded


def run_self_test() -> None:
    """
    Test interleaving, reference-pattern attachment,
    AES-GCM protection, transcript binding, and tamper detection.
    """

    print("=" * 70)
    print("FT-QuPAP Control Schedule Self-Test")
    print("=" * 70)

    payload_specs = [
        LogicalSpec(
            block_id=f"P{index:04d}",
            role="payload",
            logical_index=index,
            logical_bit=index % 2,
            basis="Z",
        )
        for index in range(
            PAYLOAD_BLOCK_COUNT
        )
    ]

    check_rng = np.random.default_rng(
        DEFAULT_RANDOM_SEED
    )

    check_specs = [
        LogicalSpec(
            block_id=f"C{index:04d}",
            role="check",
            logical_index=index,
            logical_bit=int(
                check_rng.integers(
                    0,
                    2,
                )
            ),
            basis=str(
                check_rng.choice(
                    SUPPORTED_BASES
                )
            ),
        )
        for index in range(
            CHECK_BLOCK_COUNT
        )
    ]

    ordered_specs, schedule = (
        create_interleaved_schedule(
            payload_specs=payload_specs,
            check_specs=check_specs,
            rng=np.random.default_rng(
                DEFAULT_RANDOM_SEED
            ),
            validate_standard_counts=True,
        )
    )

    encoded_check_blocks = [
        SimpleNamespace(
            spec=SimpleNamespace(
                block_id=spec.block_id
            ),
            reference_bits=np.full(
                STEANE_BLOCK_SIZE,
                int(spec.logical_bit),
                dtype=np.int8,
            ),
        )
        for spec in ordered_specs
        if spec.role == "check"
    ]

    completed_schedule = (
        attach_expected_reference_bits(
            schedule=schedule,
            encoded_check_blocks=
                encoded_check_blocks,
            require_steane_blocks=True,
        )
    )

    k_ctrl = hashlib.sha3_256(
        b"FT-QuPAP K_ctrl self-test"
    ).digest()

    transcript_hash = (
        hashlib.sha3_256(
            b"FT-QuPAP transcript self-test"
        ).digest()
    )

    protected = protect_control_schedule(
        schedule=completed_schedule,
        k_ctrl=k_ctrl,
        transcript_hash=
            transcript_hash,
    )

    binding_valid = (
        verify_control_schedule_binding(
            encrypted_schedule=
                protected.encrypted_schedule,
            expected_binding=
                protected.schedule_binding,
            transcript_hash=
                transcript_hash,
        )
    )

    decrypted_schedule = (
        decrypt_for_self_test(
            encrypted_schedule=
                protected.encrypted_schedule,
            k_ctrl=k_ctrl,
            transcript_hash=
                transcript_hash,
        )
    )

    round_trip_valid = (
        canonical_json_bytes(
            completed_schedule
        )
        == canonical_json_bytes(
            decrypted_schedule
        )
    )

    tampered_encrypted_schedule = dict(
        protected.encrypted_schedule
    )

    tampered_ciphertext = bytearray(
        decode_base64(
            tampered_encrypted_schedule[
                "ciphertext"
            ],
            "ciphertext",
        )
    )

    tampered_ciphertext[0] ^= 0x01

    tampered_encrypted_schedule[
        "ciphertext"
    ] = encode_base64(
        bytes(tampered_ciphertext)
    )

    tampered_binding_rejected = not (
        verify_control_schedule_binding(
            encrypted_schedule=
                tampered_encrypted_schedule,
            expected_binding=
                protected.schedule_binding,
            transcript_hash=
                transcript_hash,
        )
    )

    reference_pattern_count = sum(
        int(
            "expected_reference_bits"
            in entry
        )
        for entry in completed_schedule[
            "check_blocks"
        ]
    )

    print(
        f"Payload schedule entries  : "
        f"{len(schedule['payload_blocks'])}"
    )
    print(
        f"Check schedule entries    : "
        f"{len(schedule['check_blocks'])}"
    )
    print(
        f"Interleaved logical blocks: "
        f"{len(ordered_specs)}"
    )
    print(
        f"Reference patterns added  : "
        f"{reference_pattern_count}"
    )
    print(
        f"AES-GCM nonce bytes       : "
        f"{len(decode_base64(protected.encrypted_schedule['nonce'], 'nonce'))}"
    )
    print(
        f"Schedule binding valid    : "
        f"{binding_valid}"
    )
    print(
        f"Encryption round-trip     : "
        f"{round_trip_valid}"
    )
    print(
        f"Tampered binding rejected : "
        f"{tampered_binding_rejected}"
    )

    if len(ordered_specs) != (
        TOTAL_LOGICAL_BLOCK_COUNT
    ):
        raise ControlScheduleError(
            "Incorrect interleaved block count."
        )

    if reference_pattern_count != (
        CHECK_BLOCK_COUNT
    ):
        raise ControlScheduleError(
            "Not every check block received an "
            "expected reference pattern."
        )

    if not binding_valid:
        raise ControlScheduleError(
            "Valid schedule binding was rejected."
        )

    if not round_trip_valid:
        raise ControlScheduleError(
            "Schedule encryption round-trip failed."
        )

    if not tampered_binding_rejected:
        raise ControlScheduleError(
            "Tampered encrypted schedule was accepted."
        )

    print(
        "\nControl schedule self-test "
        "completed successfully."
    )


__all__ = [
    "LogicalSpec",
    "ProtectedControlSchedule",
    "ControlScheduleError",
    "ControlScheduleValidationError",
    "ControlScheduleEncryptionError",
    "create_interleaved_schedule",
    "attach_expected_reference_bits",
    "validate_control_schedule",
    "encrypt_control_schedule",
    "bind_control_schedule",
    "verify_control_schedule_binding",
    "protect_control_schedule",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        ControlScheduleError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"\n[CONTROL SCHEDULE ERROR] "
            f"{error}"
        )