#!/usr/bin/env python3
"""
Initialize FT-QuPAP JSON databases and registration trust material.

Flowchart/notebook alignment:
- Pre-phase / Cells 6-7: generate the Authentication Server's long-term
  ML-DSA-65 credential and provision its public key as the Mobile Station
  trust anchor.
- Cells 8-9: register a pseudonymous subscriber and create the AS-side
  subscriber database.
- Cells 10-11: initialize the nonce-replay cache and session/audit storage.

Created project files:
    database/subscribers.json
    database/used_nonces.json
    database/registration_records.json
    database/trusted_server_keys.json
    database/demo_sessions.json

The AS private signing key is deliberately not written to
``trusted_server_keys.json``. For this capstone simulator it is stored in:
    database/.secrets/as_mldsa_65_private_key.json

Production deployments must replace this file-based private-key storage with
an HSM, secure element, KMS, or another protected operator key store.

Run from the project root:
    python scripts/initialize_database.py

Useful options:
    python scripts/initialize_database.py --validate-only
    python scripts/initialize_database.py --reset-runtime
    python scripts/initialize_database.py --force
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = PROJECT_ROOT / "database"
SECRETS_DIR = DATABASE_DIR / ".secrets"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

SUBSCRIBERS_FILE = DATABASE_DIR / "subscribers.json"
USED_NONCES_FILE = DATABASE_DIR / "used_nonces.json"
REGISTRATION_RECORDS_FILE = DATABASE_DIR / "registration_records.json"
TRUSTED_SERVER_KEYS_FILE = DATABASE_DIR / "trusted_server_keys.json"
DEMO_SESSIONS_FILE = DATABASE_DIR / "demo_sessions.json"
PRIVATE_KEY_FILE = SECRETS_DIR / "as_mldsa_65_private_key.json"

LOG_FILE = LOG_DIR / "initialize_database.log"
MANIFEST_FILE = LOG_DIR / "database_initialization.json"

PROTOCOL_NAME = "FT-QuPAP"
PROTOCOL_VERSION = "research-simulator-v5-1-large-ml-operational-threshold"
DEFAULT_SERVER_ID = "AS-6G-001"
DEFAULT_PSEUDONYM_ID = "PID-6G-UE-0001"
ML_DSA_ALGORITHM = "ML-DSA-65"
DEFAULT_CONTEXTS = ("urban", "suburban", "rural")
TRUST_ANCHOR_VERSION = 1
SCHEMA_VERSION = 1

PSEUDONYM_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._:-]{5,127}$")
SERVER_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._:-]{2,127}$")
ALLOWED_STATUSES = {"active", "suspended", "revoked"}


class DatabaseInitializationError(RuntimeError):
    """Raised when FT-QuPAP database initialization cannot be completed."""


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging() -> logging.Logger:
    """Configure console and file logging for this script."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ft_qupap.initialize_database")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize FT-QuPAP subscriber, trust-anchor, replay-cache, "
            "registration, and demo-session JSON stores."
        )
    )
    parser.add_argument(
        "--server-id",
        default=DEFAULT_SERVER_ID,
        help=f"Authentication Server identifier (default: {DEFAULT_SERVER_ID}).",
    )
    parser.add_argument(
        "--pseudonym-id",
        default=DEFAULT_PSEUDONYM_ID,
        help=(
            "Pseudonymous subscriber identity; no raw IMSI is stored "
            f"(default: {DEFAULT_PSEUDONYM_ID})."
        ),
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        default=list(DEFAULT_CONTEXTS),
        help="Registered service contexts (default: urban suburban rural).",
    )
    parser.add_argument(
        "--subscriber-status",
        choices=sorted(ALLOWED_STATUSES),
        default="active",
        help="Initial subscriber status (default: active).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite all managed database files and rotate the AS ML-DSA "
            "credential. Existing demo data will be lost."
        ),
    )
    parser.add_argument(
        "--reset-runtime",
        action="store_true",
        help=(
            "Clear used_nonces.json and demo_sessions.json while preserving "
            "subscriber registrations and ML-DSA trust material."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing database files without modifying them.",
    )
    return parser.parse_args()


def validate_identifier(value: str, pattern: re.Pattern[str], label: str) -> str:
    """Validate an identifier used as a persistent database key."""

    normalized = value.strip()
    if not pattern.fullmatch(normalized):
        raise DatabaseInitializationError(
            f"Invalid {label}: {value!r}. Use 3-128 uppercase letters, "
            "digits, dot, underscore, colon, or hyphen."
        )
    return normalized


def normalize_contexts(contexts: Sequence[str]) -> list[str]:
    """Normalize and validate registered channel/service contexts."""

    normalized: list[str] = []
    seen: set[str] = set()

    for value in contexts:
        context = value.strip().lower()
        if context not in DEFAULT_CONTEXTS:
            raise DatabaseInitializationError(
                f"Unsupported context {value!r}; allowed values are: "
                f"{', '.join(DEFAULT_CONTEXTS)}."
            )
        if context not in seen:
            normalized.append(context)
            seen.add(context)

    if not normalized:
        raise DatabaseInitializationError(
            "At least one registered context is required."
        )

    return normalized


def encode_bytes(value: bytes) -> str:
    """Encode binary cryptographic material using standard Base64."""

    return base64.b64encode(value).decode("ascii")


def decode_bytes(value: str, label: str) -> bytes:
    """Decode strict Base64 and return a clear validation error."""

    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise DatabaseInitializationError(
            f"{label} is not valid Base64."
        ) from exc


def sha256_hex(value: bytes) -> str:
    """Return the lowercase SHA-256 fingerprint of binary data."""

    return hashlib.sha256(value).hexdigest()


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    private: bool = False,
) -> None:
    """Atomically write JSON so an interruption cannot leave a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if private:
            try:
                temporary_path.chmod(0o600)
            except OSError:
                pass

        os.replace(temporary_path, path)

        if private:
            try:
                path.chmod(0o600)
            except OSError:
                pass
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    """Read one required UTF-8 JSON document."""

    if not path.exists():
        raise DatabaseInitializationError(
            f"Required database file does not exist: "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise DatabaseInitializationError(
            f"Invalid JSON in {path.relative_to(PROJECT_ROOT)}: "
            f"line {exc.lineno}, column {exc.colno}."
        ) from exc
    except OSError as exc:
        raise DatabaseInitializationError(
            f"Could not read {path.relative_to(PROJECT_ROOT)}."
        ) from exc


def import_ml_dsa_65() -> Any:
    """Load pqcrypto lazily so --validate-only can report clearer failures."""

    try:
        from pqcrypto.sign import ml_dsa_65  # type: ignore
    except ImportError as exc:
        raise DatabaseInitializationError(
            "The 'pqcrypto' package is not installed in the active Python "
            "environment. Run scripts/setup_environment.py, activate .venv, "
            "and run this command again."
        ) from exc

    return ml_dsa_65


def generate_server_credential(
    server_id: str,
    created_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate the AS long-term ML-DSA key and public trust-anchor record."""

    ml_dsa_65 = import_ml_dsa_65()

    try:
        public_key, secret_key = ml_dsa_65.generate_keypair()
    except Exception as exc:
        raise DatabaseInitializationError(
            "ML-DSA-65 key generation failed."
        ) from exc

    public_key_fingerprint = sha256_hex(public_key)
    key_id = f"mldsa65-{public_key_fingerprint[:16]}"

    trust_record = {
        "algorithm": ML_DSA_ALGORITHM,
        "created_at": created_at,
        "key_id": key_id,
        "public_key_b64": encode_bytes(public_key),
        "public_key_sha256": public_key_fingerprint,
        "server_id": server_id,
        "status": "active",
        "trust_anchor_version": TRUST_ANCHOR_VERSION,
    }

    private_record = {
        "algorithm": ML_DSA_ALGORITHM,
        "created_at": created_at,
        "key_id": key_id,
        "private_key_b64": encode_bytes(secret_key),
        "public_key_sha256": public_key_fingerprint,
        "server_id": server_id,
        "storage_warning": (
            "Capstone simulator only. Replace file storage with an HSM, "
            "secure element, or KMS in production."
        ),
    }

    return trust_record, private_record


def build_subscriber_record(
    pseudonym_id: str,
    status: str,
    contexts: Sequence[str],
    created_at: str,
) -> dict[str, Any]:
    """Create the pseudonymous subscriber record used by the AS verifier."""

    return {
        "pseudonym_id": pseudonym_id,
        "registered_at": created_at,
        "registered_contexts": list(contexts),
        "registration_version": 1,
        "subscriber_status": status,
    }


def build_registration_record(
    pseudonym_id: str,
    server_id: str,
    status: str,
    contexts: Sequence[str],
    key_id: str,
    created_at: str,
) -> dict[str, Any]:
    """Create an auditable pre-phase registration event."""

    registration_suffix = secrets.token_hex(6).upper()
    return {
        "event": "subscriber_registered",
        "key_id": key_id,
        "pseudonym_id": pseudonym_id,
        "registered_contexts": list(contexts),
        "registration_id": f"REG-{registration_suffix}",
        "server_id": server_id,
        "status": status,
        "timestamp": created_at,
        "trust_anchor_version": TRUST_ANCHOR_VERSION,
    }


def write_managed_file(
    path: Path,
    payload: Any,
    *,
    force: bool,
    logger: logging.Logger,
    private: bool = False,
) -> str:
    """Write a managed file or preserve it when initialization is idempotent."""

    relative_path = path.relative_to(PROJECT_ROOT)

    if path.exists() and not force:
        logger.info("Preserved existing file: %s", relative_path)
        return "preserved"

    atomic_write_json(path, payload, private=private)
    logger.info(
        "%s file: %s",
        "Replaced" if path.exists() and force else "Created",
        relative_path,
    )
    return "written"


def initialize_database(
    *,
    server_id: str,
    pseudonym_id: str,
    contexts: Sequence[str],
    subscriber_status: str,
    force: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Create all registration, trust, replay, and session stores."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    trust_exists = TRUSTED_SERVER_KEYS_FILE.exists()
    private_exists = PRIVATE_KEY_FILE.exists()

    if trust_exists != private_exists and not force:
        raise DatabaseInitializationError(
            "ML-DSA trust material is inconsistent: exactly one of "
            "trusted_server_keys.json and the private-key file exists. "
            "Restore the missing matching file or rerun with --force to rotate "
            "the credential and rebuild the managed databases."
        )

    created_at = utc_now_iso()
    actions: dict[str, str] = {}

    if force or not trust_exists:
        trust_record, private_record = generate_server_credential(
            server_id,
            created_at,
        )
        trusted_servers = {server_id: trust_record}

        actions["trusted_server_keys"] = write_managed_file(
            TRUSTED_SERVER_KEYS_FILE,
            trusted_servers,
            force=force,
            logger=logger,
        )
        actions["private_server_key"] = write_managed_file(
            PRIVATE_KEY_FILE,
            private_record,
            force=force,
            logger=logger,
            private=True,
        )
    else:
        trusted_servers = read_json(TRUSTED_SERVER_KEYS_FILE)
        trust_record = validate_trusted_server_keys(
            trusted_servers,
            expected_server_id=server_id,
        )
        validate_private_key(
            read_json(PRIVATE_KEY_FILE),
            trust_record=trust_record,
        )
        actions["trusted_server_keys"] = "preserved"
        actions["private_server_key"] = "preserved"
        logger.info("Preserved and validated existing ML-DSA trust material.")

    subscriber_record = build_subscriber_record(
        pseudonym_id,
        subscriber_status,
        contexts,
        created_at,
    )
    subscriber_database = {pseudonym_id: subscriber_record}

    registration_record = build_registration_record(
        pseudonym_id,
        server_id,
        subscriber_status,
        contexts,
        str(trust_record["key_id"]),
        created_at,
    )

    actions["subscribers"] = write_managed_file(
        SUBSCRIBERS_FILE,
        subscriber_database,
        force=force,
        logger=logger,
    )
    actions["used_nonces"] = write_managed_file(
        USED_NONCES_FILE,
        {},
        force=force,
        logger=logger,
    )
    actions["registration_records"] = write_managed_file(
        REGISTRATION_RECORDS_FILE,
        [registration_record],
        force=force,
        logger=logger,
    )
    actions["demo_sessions"] = write_managed_file(
        DEMO_SESSIONS_FILE,
        [],
        force=force,
        logger=logger,
    )

    validation_summary = validate_all_files(
        expected_server_id=server_id,
        expected_pseudonym_id=pseudonym_id,
    )

    manifest = {
        "actions": actions,
        "database_directory": str(DATABASE_DIR.relative_to(PROJECT_ROOT)),
        "initialized_at": created_at,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "validation": validation_summary,
    }
    atomic_write_json(MANIFEST_FILE, manifest)
    logger.info("Initialization manifest: %s", MANIFEST_FILE.relative_to(PROJECT_ROOT))

    return manifest


def reset_runtime_files(logger: logging.Logger) -> dict[str, Any]:
    """Clear replay and demo-session state without rotating trust material."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(USED_NONCES_FILE, {})
    atomic_write_json(DEMO_SESSIONS_FILE, [])

    logger.info("Reset runtime replay cache: %s", USED_NONCES_FILE.relative_to(PROJECT_ROOT))
    logger.info("Reset demo sessions: %s", DEMO_SESSIONS_FILE.relative_to(PROJECT_ROOT))

    return {
        "reset_at": utc_now_iso(),
        "reset_files": [
            str(USED_NONCES_FILE.relative_to(PROJECT_ROOT)),
            str(DEMO_SESSIONS_FILE.relative_to(PROJECT_ROOT)),
        ],
    }


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DatabaseInitializationError(f"{label} must be a JSON object.")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise DatabaseInitializationError(f"{label} must be a JSON array.")
    return value


def validate_subscribers(
    document: Any,
    expected_pseudonym_id: str | None = None,
) -> int:
    subscribers = require_mapping(document, "subscribers.json")

    for key, raw_record in subscribers.items():
        record = require_mapping(raw_record, f"subscriber {key!r}")
        pseudonym_id = str(record.get("pseudonym_id", ""))

        if key != pseudonym_id:
            raise DatabaseInitializationError(
                f"Subscriber key {key!r} does not match record pseudonym_id "
                f"{pseudonym_id!r}."
            )
        validate_identifier(pseudonym_id, PSEUDONYM_PATTERN, "pseudonym ID")

        status = record.get("subscriber_status")
        if status not in ALLOWED_STATUSES:
            raise DatabaseInitializationError(
                f"Subscriber {key!r} has unsupported status {status!r}."
            )

        contexts = record.get("registered_contexts")
        if not isinstance(contexts, list):
            raise DatabaseInitializationError(
                f"Subscriber {key!r} registered_contexts must be an array."
            )
        normalize_contexts([str(item) for item in contexts])

    if expected_pseudonym_id and expected_pseudonym_id not in subscribers:
        raise DatabaseInitializationError(
            f"Expected subscriber {expected_pseudonym_id!r} was not found."
        )

    return len(subscribers)


def validate_used_nonces(document: Any) -> int:
    nonces = require_mapping(document, "used_nonces.json")

    for nonce_key, timestamp in nonces.items():
        if not isinstance(nonce_key, str) or not nonce_key:
            raise DatabaseInitializationError(
                "Each replay-cache key must be a non-empty string."
            )
        if not isinstance(timestamp, (int, float)):
            raise DatabaseInitializationError(
                f"Replay timestamp for {nonce_key!r} must be numeric."
            )

    return len(nonces)


def validate_registration_records(document: Any) -> int:
    records = require_list(document, "registration_records.json")

    for index, raw_record in enumerate(records):
        record = require_mapping(raw_record, f"registration record {index}")
        required = {
            "event",
            "key_id",
            "pseudonym_id",
            "registered_contexts",
            "registration_id",
            "server_id",
            "status",
            "timestamp",
            "trust_anchor_version",
        }
        missing = sorted(required.difference(record))
        if missing:
            raise DatabaseInitializationError(
                f"Registration record {index} is missing: {', '.join(missing)}."
            )

    return len(records)


def validate_trusted_server_keys(
    document: Any,
    expected_server_id: str | None = None,
) -> Mapping[str, Any]:
    trusted_servers = require_mapping(document, "trusted_server_keys.json")

    if expected_server_id is not None:
        raw_record = trusted_servers.get(expected_server_id)
        if raw_record is None:
            raise DatabaseInitializationError(
                f"Expected server trust anchor {expected_server_id!r} was not found."
            )
        server_items = [(expected_server_id, raw_record)]
    else:
        server_items = list(trusted_servers.items())

    if not server_items:
        raise DatabaseInitializationError(
            "trusted_server_keys.json must contain at least one trust anchor."
        )

    selected_record: Mapping[str, Any] | None = None

    for key, raw_record in server_items:
        record = require_mapping(raw_record, f"trust anchor {key!r}")
        if key != record.get("server_id"):
            raise DatabaseInitializationError(
                f"Trust-anchor key {key!r} does not match server_id."
            )
        if record.get("algorithm") != ML_DSA_ALGORITHM:
            raise DatabaseInitializationError(
                f"Trust anchor {key!r} must use {ML_DSA_ALGORITHM}."
            )

        public_key = decode_bytes(
            str(record.get("public_key_b64", "")),
            f"Trust anchor {key!r} public key",
        )
        actual_fingerprint = sha256_hex(public_key)
        expected_fingerprint = record.get("public_key_sha256")
        if actual_fingerprint != expected_fingerprint:
            raise DatabaseInitializationError(
                f"Trust anchor {key!r} public-key fingerprint mismatch."
            )

        if not record.get("key_id"):
            raise DatabaseInitializationError(
                f"Trust anchor {key!r} has no key_id."
            )

        selected_record = record

    assert selected_record is not None
    return selected_record


def validate_private_key(
    document: Any,
    *,
    trust_record: Mapping[str, Any],
) -> None:
    private_record = require_mapping(document, "private ML-DSA key file")

    for field in (
        "algorithm",
        "key_id",
        "private_key_b64",
        "public_key_sha256",
        "server_id",
    ):
        if field not in private_record:
            raise DatabaseInitializationError(
                f"Private ML-DSA key file is missing {field!r}."
            )

    if private_record["algorithm"] != ML_DSA_ALGORITHM:
        raise DatabaseInitializationError(
            f"Private key must use {ML_DSA_ALGORITHM}."
        )
    if private_record["server_id"] != trust_record["server_id"]:
        raise DatabaseInitializationError(
            "Private key server_id does not match the trust anchor."
        )
    if private_record["key_id"] != trust_record["key_id"]:
        raise DatabaseInitializationError(
            "Private key key_id does not match the trust anchor."
        )
    if private_record["public_key_sha256"] != trust_record["public_key_sha256"]:
        raise DatabaseInitializationError(
            "Private key public-key fingerprint does not match the trust anchor."
        )

    private_key = decode_bytes(
        str(private_record["private_key_b64"]),
        "Private ML-DSA key",
    )
    if not private_key:
        raise DatabaseInitializationError("Private ML-DSA key is empty.")


def validate_demo_sessions(document: Any) -> int:
    sessions = require_list(document, "demo_sessions.json")

    for index, session in enumerate(sessions):
        if not isinstance(session, dict):
            raise DatabaseInitializationError(
                f"Demo session {index} must be a JSON object."
            )

    return len(sessions)


def validate_all_files(
    *,
    expected_server_id: str | None = None,
    expected_pseudonym_id: str | None = None,
) -> dict[str, Any]:
    """Validate all managed files and return record counts/fingerprints."""

    subscriber_count = validate_subscribers(
        read_json(SUBSCRIBERS_FILE),
        expected_pseudonym_id=expected_pseudonym_id,
    )
    used_nonce_count = validate_used_nonces(read_json(USED_NONCES_FILE))
    registration_count = validate_registration_records(
        read_json(REGISTRATION_RECORDS_FILE)
    )

    trust_record = validate_trusted_server_keys(
        read_json(TRUSTED_SERVER_KEYS_FILE),
        expected_server_id=expected_server_id,
    )
    validate_private_key(
        read_json(PRIVATE_KEY_FILE),
        trust_record=trust_record,
    )

    demo_session_count = validate_demo_sessions(read_json(DEMO_SESSIONS_FILE))

    return {
        "demo_session_count": demo_session_count,
        "key_id": trust_record["key_id"],
        "public_key_sha256": trust_record["public_key_sha256"],
        "registration_record_count": registration_count,
        "server_id": trust_record["server_id"],
        "subscriber_count": subscriber_count,
        "used_nonce_count": used_nonce_count,
        "valid": True,
    }


def print_summary(title: str, summary: Mapping[str, Any]) -> None:
    """Print a compact human-readable result followed by JSON details."""

    print(f"\n{title}")
    print("-" * len(title))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> int:
    arguments = parse_arguments()
    logger = configure_logging()

    try:
        server_id = validate_identifier(
            arguments.server_id,
            SERVER_ID_PATTERN,
            "server ID",
        )
        pseudonym_id = validate_identifier(
            arguments.pseudonym_id,
            PSEUDONYM_PATTERN,
            "pseudonym ID",
        )
        contexts = normalize_contexts(arguments.contexts)

        if arguments.validate_only:
            if arguments.force or arguments.reset_runtime:
                raise DatabaseInitializationError(
                    "--validate-only cannot be combined with --force or "
                    "--reset-runtime."
                )

            summary = validate_all_files(
                expected_server_id=server_id,
                expected_pseudonym_id=pseudonym_id,
            )
            logger.info("All FT-QuPAP database files are valid.")
            print_summary("Database validation passed", summary)
            return 0

        if arguments.reset_runtime:
            if arguments.force:
                raise DatabaseInitializationError(
                    "Choose either --reset-runtime or --force, not both."
                )

            reset_summary = reset_runtime_files(logger)
            print_summary("Runtime database reset completed", reset_summary)
            return 0

        manifest = initialize_database(
            server_id=server_id,
            pseudonym_id=pseudonym_id,
            contexts=contexts,
            subscriber_status=arguments.subscriber_status,
            force=arguments.force,
            logger=logger,
        )
        print_summary("FT-QuPAP database initialization completed", manifest)
        return 0

    except DatabaseInitializationError as exc:
        logger.error("%s", exc)
        print(f"\nInitialization failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.error("Initialization interrupted by the user.")
        print("\nInitialization interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # Defensive boundary for a setup utility.
        logger.exception("Unexpected initialization failure.")
        print(
            f"\nUnexpected initialization failure: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
