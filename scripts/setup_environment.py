#!/usr/bin/env python3
"""
FT-QuPAP Capstone environment setup.

Notebook alignment:
- Part A: environment, reproducibility, dependencies, random seed.
- Verifies ML-KEM-768, ML-DSA-65, KMAC256, GP support, and Qiskit Aer.
- Creates the runtime directories used by the dashboard, experiments, and logs.

Run:
    python scripts/setup_environment.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import venv
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
ENV_EXAMPLE_FILE = PROJECT_ROOT / ".env.example"
ENV_FILE = PROJECT_ROOT / ".env"
LOG_FILE = PROJECT_ROOT / "outputs" / "logs" / "setup_environment.log"
METADATA_FILE = PROJECT_ROOT / "outputs" / "logs" / "setup_metadata.json"

MINIMUM_PYTHON = (3, 10)
NOTEBOOK_SEED = 20260701

RUNTIME_DIRECTORIES = (
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "demo",
    PROJECT_ROOT / "data" / "results",
    PROJECT_ROOT / "database",
    PROJECT_ROOT / "outputs" / "figures",
    PROJECT_ROOT / "outputs" / "reports",
    PROJECT_ROOT / "outputs" / "logs",
)


class SetupError(RuntimeError):
    """Raised when environment setup cannot be completed."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the FT-QuPAP Capstone environment."
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Delete and recreate .venv.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip installation.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip dependency and cryptographic smoke tests.",
    )
    parser.add_argument(
        "--force-env",
        action="store_true",
        help="Overwrite .env using .env.example.",
    )
    return parser.parse_args()


def configure_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ft_qupap.setup")
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


def check_python_version() -> None:
    if sys.version_info[:2] < MINIMUM_PYTHON:
        required = ".".join(map(str, MINIMUM_PYTHON))
        detected = ".".join(map(str, sys.version_info[:3]))
        raise SetupError(
            f"Python {required}+ is required; detected {detected}."
        )


def run_command(
    command: Sequence[str],
    logger: logging.Logger,
) -> None:
    logger.info("Running: %s", " ".join(map(str, command)))

    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("PYTHONHASHSEED", str(NOTEBOOK_SEED))

    try:
        subprocess.run(
            [str(item) for item in command],
            cwd=str(PROJECT_ROOT),
            env=environment,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SetupError(
            f"Command was not found: {command[0]}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SetupError(
            f"Command failed with exit code {exc.returncode}: "
            f"{' '.join(map(str, command))}"
        ) from exc


def venv_python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def create_runtime_directories(logger: logging.Logger) -> None:
    for directory in RUNTIME_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Directory ready: %s",
            directory.relative_to(PROJECT_ROOT),
        )


def prepare_env_file(
    force: bool,
    logger: logging.Logger,
) -> None:
    if not ENV_EXAMPLE_FILE.exists():
        logger.warning(".env.example was not found; skipping .env creation.")
        return

    if ENV_FILE.exists() and not force:
        logger.info("Existing .env preserved.")
        return

    shutil.copy2(ENV_EXAMPLE_FILE, ENV_FILE)
    logger.info("Created .env from .env.example.")


def create_virtual_environment(
    recreate: bool,
    logger: logging.Logger,
) -> Path:
    if recreate and VENV_DIR.exists():
        logger.warning("Removing existing virtual environment.")
        shutil.rmtree(VENV_DIR)

    python_path = venv_python_path()

    if not python_path.exists():
        logger.info("Creating virtual environment: %s", VENV_DIR)
        try:
            venv.EnvBuilder(
                with_pip=True,
                symlinks=os.name != "nt",
            ).create(VENV_DIR)
        except Exception as exc:
            raise SetupError("Could not create .venv.") from exc
    else:
        logger.info("Reusing existing virtual environment.")

    python_path = venv_python_path()
    if not python_path.exists():
        raise SetupError(
            f"Virtual-environment Python is missing: {python_path}"
        )

    return python_path


def install_dependencies(
    python_path: Path,
    logger: logging.Logger,
) -> None:
    if not REQUIREMENTS_FILE.exists():
        raise SetupError(
            f"requirements.txt was not found: {REQUIREMENTS_FILE}"
        )

    run_command(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
        logger,
    )

    run_command(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS_FILE),
        ],
        logger,
    )


def verify_environment(
    python_path: Path,
    logger: logging.Logger,
) -> dict[str, object]:
    verification_code = r'''
import json

import joblib
import matplotlib
import numpy
import pandas
from Crypto.Hash import KMAC256
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pqcrypto.kem import ml_kem_768
from pqcrypto.sign import ml_dsa_65
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sklearn.gaussian_process import GaussianProcessClassifier

# KMAC256, 128-bit authentication tag.
kmac_tag = KMAC256.new(
    key=bytes(range(32)),
    data=b"FT-QuPAP",
    mac_len=16,
    custom=b"FT-QuPAP/KMAC256/v1",
).digest()

# ML-KEM encapsulation/decapsulation consistency.
kem_public_key, kem_secret_key = ml_kem_768.generate_keypair()
kem_ciphertext, shared_secret_ms = ml_kem_768.encrypt(kem_public_key)
shared_secret_as = ml_kem_768.decrypt(kem_secret_key, kem_ciphertext)

# ML-DSA signing and verification.
dsa_public_key, dsa_secret_key = ml_dsa_65.generate_keypair()
message = b"FT-QuPAP setup verification"
signature = ml_dsa_65.sign(dsa_secret_key, message)
ml_dsa_65.verify(dsa_public_key, message, signature)

# Qiskit Aer execution.
circuit = QuantumCircuit(1, 1)
circuit.h(0)
circuit.measure(0, 0)
AerSimulator().run(circuit, shots=1).result()

print(json.dumps({
    "imports_ok": True,
    "kmac_tag_bytes": len(kmac_tag),
    "ml_kem_public_key_bytes": len(kem_public_key),
    "ml_kem_ciphertext_bytes": len(kem_ciphertext),
    "ml_kem_shared_secret_bytes": len(shared_secret_ms),
    "ml_kem_shared_secret_match": shared_secret_ms == shared_secret_as,
    "ml_dsa_public_key_bytes": len(dsa_public_key),
    "ml_dsa_signature_bytes": len(signature),
    "qiskit_aer_execution": True,
    "gp_classifier_available": GaussianProcessClassifier is not None,
}))
'''

    completed = subprocess.run(
        [str(python_path), "-c", verification_code],
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONHASHSEED": str(NOTEBOOK_SEED),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise SetupError(
            "Environment verification failed:\n" + details
        )

    output_lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if not output_lines:
        raise SetupError("Verification returned no output.")

    try:
        result = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise SetupError(
            "Verification returned invalid JSON:\n"
            + completed.stdout
        ) from exc

    if not result["ml_kem_shared_secret_match"]:
        raise SetupError("ML-KEM shared-secret verification failed.")

    logger.info("PQC, KMAC, GP, and Qiskit verification passed.")
    return result


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(
    python_path: Path,
    verification: dict[str, object] | None,
    logger: logging.Logger,
) -> None:
    metadata = {
        "project": "FT-QuPAP-Capstone",
        "protocol": "FT-QuPAP-v5.1",
        "setup_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "host_python": sys.version,
        "environment_python": str(python_path),
        "notebook_seed": NOTEBOOK_SEED,
        "requirements_sha256": file_sha256(REQUIREMENTS_FILE),
        "verification": verification,
    }

    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_FILE.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    logger.info(
        "Metadata saved: %s",
        METADATA_FILE.relative_to(PROJECT_ROOT),
    )


def print_next_steps() -> None:
    print("\nFT-QuPAP environment is ready.")
    if os.name == "nt":
        print(r"Activate: .venv\Scripts\activate")
    else:
        print("Activate: source .venv/bin/activate")
    print("Start demo: python app.py")
    print("Run tests: python scripts/run_all_tests.py")


def main() -> int:
    args = parse_arguments()
    logger = configure_logging()

    try:
        check_python_version()
        logger.info("FT-QuPAP setup started.")
        logger.info("Project root: %s", PROJECT_ROOT)
        logger.info("Host Python: %s", sys.version.split()[0])

        create_runtime_directories(logger)
        prepare_env_file(args.force_env, logger)
        python_path = create_virtual_environment(
            args.recreate_venv,
            logger,
        )

        if args.skip_install:
            logger.warning("Dependency installation skipped.")
        else:
            install_dependencies(python_path, logger)

        verification = None
        if args.skip_verification:
            logger.warning("Environment verification skipped.")
        else:
            verification = verify_environment(
                python_path,
                logger,
            )

        write_metadata(python_path, verification, logger)
        logger.info("FT-QuPAP setup completed successfully.")
        print_next_steps()
        return 0

    except SetupError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Setup cancelled by user.")
        return 130
    except Exception:
        logger.exception("Unexpected setup failure.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
