"""
FT-QuPAP Mobile Station Coordinator

This module coordinates the complete Mobile Station side of one
FT-QuPAP authentication attempt.

Execution order:

1. Create pseudonymous authentication request.
2. Verify the ML-DSA-signed server package.
3. Extract the verified ephemeral ML-KEM public key.
4. Perform ML-KEM-768 encapsulation.
5. Construct H(Transcript).
6. Derive K_auth and K_ctrl.
7. Generate the 128-bit KMAC authentication tag.
8. Map the tag into 128 logical payload blocks.
9. Generate 32 independent logical check blocks.
10. Randomly interleave payload and check blocks.
11. Apply Steane [[7,1,3]] encoding.
12. Add expected physical check patterns to the schedule.
13. Encrypt and transcript-bind the schedule using K_ctrl.
14. Transmit the encoded frame through the quantum channel.

The Authentication Server performs QBER calculation, syndrome
correction, tag verification, GP inference, and final acceptance.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .authentication_request import (
    create_authentication_request,
    validate_authentication_request,
)
from .check_qubit_preparation import (
    STANDARD_CHECK_COUNT,
    generate_check_specs,
)
from .control_schedule import (
    ProtectedControlSchedule,
    attach_expected_reference_bits,
    create_interleaved_schedule,
    protect_control_schedule,
)
from .kmac_tag_generator import (
    compute_authentication_tag,
)
from .mlkem_encapsulation import (
    MLKEMEncapsulationResult,
    encapsulate_session_secret,
)
from .payload_preparation import (
    map_tag_to_logical_specs,
)
from .quantum_transmitter import (
    IDEAL_CHANNEL,
    ChannelConfig,
    QuantumTransmissionResult,
    QuantumTransmitter,
)
from .server_package_verifier import (
    ServerPackageVerificationResult,
    build_test_server_package,
    unsigned_server_information,
    verify_server_credential_detailed,
)
from .session_key_derivation import (
    derive_session_key_material,
    split_session_keys,
)
from .steane_encoder import (
    SteaneEncodedFrame,
    encode_ft_qupap_frame,
)


PROTOCOL_NAME = "FT-QuPAP"

PROTOCOL_VERSION = (
    "research-simulator-v5-1-"
    "large-ml-operational-threshold"
)

ML_KEM_ALGORITHM = "ML-KEM-768"

AUTHENTICATION_TAG_LENGTH_BYTES = 16
PAYLOAD_LOGICAL_BLOCK_COUNT = 128
CHECK_LOGICAL_BLOCK_COUNT = 32
TOTAL_LOGICAL_BLOCK_COUNT = 160
STEANE_PHYSICAL_QUBIT_COUNT = 1120


class MobileStationError(Exception):
    """Base exception for Mobile Station failures."""


class ServerCredentialRejectedError(MobileStationError):
    """Raised when the server package fails verification."""

    def __init__(
        self,
        verification_result: ServerPackageVerificationResult,
    ) -> None:
        self.verification_result = verification_result

        super().__init__(
            "Authentication Server credential rejected: "
            f"{verification_result.reason}"
        )


class MobileSessionPreparationError(MobileStationError):
    """Raised when session preparation is incomplete."""


@dataclass(frozen=True)
class MobileStationTimings:
    """
    Runtime measurements for one Mobile Station attempt.

    All values are measured in seconds.
    """

    server_verification_s: float
    mlkem_encapsulation_s: float
    transcript_and_kdf_s: float
    kmac_generation_s: float
    logical_preparation_s: float
    interleaving_s: float
    steane_encoding_s: float
    schedule_protection_s: float
    quantum_transmission_s: float
    total_s: float

    def to_dictionary(self) -> dict[str, float]:
        """Return timing values as a dictionary."""

        return {
            key: float(value)
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class MobileStationSessionSecrets:
    """
    Secret material retained only for the current simulation session.

    These values must never be inserted into:

    - classical transport messages
    - logs
    - CSV files
    - audit records
    - GP feature vectors
    """

    shared_secret: bytes = field(repr=False)
    k_auth: bytes = field(repr=False)
    k_ctrl: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.shared_secret, bytes):
            raise TypeError(
                "shared_secret must be bytes."
            )

        if not isinstance(self.k_auth, bytes):
            raise TypeError(
                "k_auth must be bytes."
            )

        if not isinstance(self.k_ctrl, bytes):
            raise TypeError(
                "k_ctrl must be bytes."
            )

        if len(self.shared_secret) != 32:
            raise ValueError(
                "ML-KEM shared secret must contain 32 bytes."
            )

        if len(self.k_auth) != 32:
            raise ValueError(
                "K_auth must contain 32 bytes."
            )

        if len(self.k_ctrl) != 32:
            raise ValueError(
                "K_ctrl must contain 32 bytes."
            )

        if hmac.compare_digest(
            self.k_auth,
            self.k_ctrl,
        ):
            raise ValueError(
                "K_auth and K_ctrl must be separated."
            )

    def safe_summary(self) -> dict[str, Any]:
        """
        Return only lengths and short testing fingerprints.
        """

        return {
            "shared_secret_bytes": len(
                self.shared_secret
            ),
            "k_auth_bytes": len(
                self.k_auth
            ),
            "k_ctrl_bytes": len(
                self.k_ctrl
            ),
            "shared_secret_fingerprint": (
                hashlib.sha3_256(
                    self.shared_secret
                ).hexdigest()[:16]
            ),
            "k_auth_fingerprint": (
                hashlib.sha3_256(
                    self.k_auth
                ).hexdigest()[:16]
            ),
            "k_ctrl_fingerprint": (
                hashlib.sha3_256(
                    self.k_ctrl
                ).hexdigest()[:16]
            ),
        }


@dataclass
class PreparedMobileSession:
    """
    Complete Mobile Station output for one FT-QuPAP attempt.

    The authentication tag, encoded frame, received frame, and
    session keys are excluded from the dataclass representation to
    prevent accidental logging.
    """

    request: dict[str, Any]
    server_information: dict[str, Any]

    ciphertext: bytes
    transcript_hash: bytes

    authentication_tag: bytes = field(
        repr=False
    )

    protected_schedule: ProtectedControlSchedule

    encoded_frame: SteaneEncodedFrame = field(
        repr=False
    )

    transmission_result: QuantumTransmissionResult = field(
        repr=False
    )

    server_verification: ServerPackageVerificationResult
    timings: MobileStationTimings

    _secrets: MobileStationSessionSecrets = field(
        repr=False
    )

    protocol_name: str = PROTOCOL_NAME
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        validate_authentication_request(
            self.request
        )

        if not isinstance(
            self.server_information,
            dict,
        ):
            raise TypeError(
                "server_information must be a dictionary."
            )

        if not isinstance(
            self.ciphertext,
            bytes,
        ):
            raise TypeError(
                "ciphertext must be bytes."
            )

        if not isinstance(
            self.transcript_hash,
            bytes,
        ):
            raise TypeError(
                "transcript_hash must be bytes."
            )

        if not isinstance(
            self.authentication_tag,
            bytes,
        ):
            raise TypeError(
                "authentication_tag must be bytes."
            )

        if not self.server_verification.valid:
            raise MobileSessionPreparationError(
                "Prepared session contains an invalid "
                "server-verification result."
            )

        if len(self.authentication_tag) != (
            AUTHENTICATION_TAG_LENGTH_BYTES
        ):
            raise MobileSessionPreparationError(
                "Authentication tag must contain 16 bytes."
            )

        if len(self.transcript_hash) != 32:
            raise MobileSessionPreparationError(
                "Transcript hash must contain 32 bytes."
            )

        if len(self.encoded_frame.frame) != (
            TOTAL_LOGICAL_BLOCK_COUNT
        ):
            raise MobileSessionPreparationError(
                "Encoded frame must contain "
                "160 logical blocks."
            )

        expected_physical_qubits = (
            STEANE_PHYSICAL_QUBIT_COUNT
            if self.encoded_frame.use_css
            else TOTAL_LOGICAL_BLOCK_COUNT
        )

        if (
            self.encoded_frame.total_physical_qubits
            != expected_physical_qubits
        ):
            raise MobileSessionPreparationError(
                "Encoded frame has an unexpected "
                "physical-qubit count."
            )

    @property
    def shared_secret(self) -> bytes:
        """
        Return the internal shared secret.

        This property is intended only for controlled integration
        testing. It must not be transmitted or logged.
        """

        return self._secrets.shared_secret

    @property
    def k_auth(self) -> bytes:
        """
        Return internal K_auth for controlled integration testing.
        """

        return self._secrets.k_auth

    @property
    def k_ctrl(self) -> bytes:
        """
        Return internal K_ctrl for controlled integration testing.
        """

        return self._secrets.k_ctrl

    def to_classical_transport_dictionary(
        self,
    ) -> dict[str, Any]:
        """
        Construct the public Mobile Station submission.

        Deliberately excluded:

        - shared secret
        - K_auth
        - K_ctrl
        - raw authentication tag
        - plaintext control schedule
        - hidden Eve information
        """

        protected_schedule = (
            self.protected_schedule
            .to_transport_dictionary()
        )

        return {
            "protocol": self.protocol_name,
            "version": self.protocol_version,
            "request": copy.deepcopy(
                self.request
            ),
            "ml_kem_algorithm": ML_KEM_ALGORITHM,
            "ciphertext": encode_base64(
                self.ciphertext
            ),
            "transcript_hash": encode_base64(
                self.transcript_hash
            ),
            "use_css": self.encoded_frame.use_css,
            "logical_block_count": (
                self.encoded_frame
                .logical_block_count
            ),
            "physical_qubit_count": (
                self.encoded_frame
                .total_physical_qubits
            ),
            **protected_schedule,
        }

    def safe_summary(self) -> dict[str, Any]:
        """
        Return a log-safe session summary.
        """

        return {
            "protocol": self.protocol_name,
            "version": self.protocol_version,
            "pseudonym_id": self.request[
                "pseudonym_id"
            ],
            "service_context": self.request[
                "service_context"
            ],
            "server_id": (
                self.server_verification
                .server_id
            ),
            "server_credential": (
                self.server_verification
                .reason
            ),
            "ml_kem_ciphertext_bytes": len(
                self.ciphertext
            ),
            "transcript_hash_bytes": len(
                self.transcript_hash
            ),
            "authentication_tag_bits": (
                len(self.authentication_tag)
                * 8
            ),
            "encoded_frame": (
                self.encoded_frame
                .safe_summary()
            ),
            "protected_schedule": (
                self.protected_schedule
                .safe_summary()
            ),
            "quantum_transmission": (
                self.transmission_result
                .receiver_visible_summary()
            ),
            "timings_seconds": (
                self.timings
                .to_dictionary()
            ),
        }


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


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """
    Serialize a mapping using notebook-compatible canonical JSON.
    """

    if not isinstance(value, Mapping):
        raise TypeError(
            "value must be a mapping."
        )

    try:
        serialized_value = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise MobileSessionPreparationError(
            "Unable to serialize transcript data."
        ) from error

    return serialized_value.encode(
        "utf-8"
    )


def build_transcript_hash(
    request: Mapping[str, Any],
    server_info_unsigned: Mapping[str, Any],
    protocol_name: str = PROTOCOL_NAME,
    protocol_version: str = PROTOCOL_VERSION,
) -> bytes:
    """
    Construct the FT-QuPAP transcript hash.

    Transcript:

        {
            "request": request,
            "server_info": unsigned server package,
            "protocol": "FT-QuPAP",
            "version": protocol version
        }

    The ML-DSA signature is excluded because server_info_unsigned
    contains only the signed server-information fields.
    """

    if not isinstance(request, Mapping):
        raise TypeError(
            "request must be a mapping."
        )

    if not isinstance(
        server_info_unsigned,
        Mapping,
    ):
        raise TypeError(
            "server_info_unsigned must be a mapping."
        )

    if not isinstance(
        protocol_name,
        str,
    ) or not protocol_name:
        raise ValueError(
            "protocol_name must be a non-empty string."
        )

    if not isinstance(
        protocol_version,
        str,
    ) or not protocol_version:
        raise ValueError(
            "protocol_version must be a non-empty string."
        )

    transcript = {
        "request": dict(request),
        "server_info": dict(
            server_info_unsigned
        ),
        "protocol": protocol_name,
        "version": protocol_version,
    }

    return hashlib.sha3_256(
        canonical_json_bytes(
            transcript
        )
    ).digest()


def normalize_trust_anchor(
    trust_anchor: Any,
) -> dict[str, Any]:
    """
    Normalize a registration trust anchor for server verification.

    Supported inputs:

    - dictionary-like object
    - object with to_verifier_dictionary()
    - object with to_dictionary()
    - object with server_id, algorithm, and public_key attributes
    """

    if isinstance(
        trust_anchor,
        Mapping,
    ):
        normalized = dict(
            trust_anchor
        )

    elif hasattr(
        trust_anchor,
        "to_verifier_dictionary",
    ):
        normalized = dict(
            trust_anchor
            .to_verifier_dictionary()
        )

    elif hasattr(
        trust_anchor,
        "to_dictionary",
    ):
        normalized = dict(
            trust_anchor
            .to_dictionary()
        )

    elif all(
        hasattr(
            trust_anchor,
            attribute,
        )
        for attribute in (
            "server_id",
            "algorithm",
            "public_key",
        )
    ):
        normalized = {
            "server_id": (
                trust_anchor.server_id
            ),
            "algorithm": (
                trust_anchor.algorithm
            ),
            "public_key": (
                trust_anchor.public_key
            ),
            "trust_anchor_version": getattr(
                trust_anchor,
                "trust_anchor_version",
                1,
            ),
        }

    else:
        raise TypeError(
            "trust_anchor must be a mapping or "
            "MobileTrustAnchor-compatible object."
        )

    if (
        "public_key" not in normalized
        and "public_key_base64" in normalized
    ):
        normalized["public_key"] = (
            normalized[
                "public_key_base64"
            ]
        )

    public_key = normalized.get(
        "public_key"
    )

    if isinstance(public_key, str):
        try:
            normalized["public_key"] = (
                base64.b64decode(
                    public_key.encode(
                        "ascii"
                    ),
                    validate=True,
                )
            )

        except Exception as error:
            raise ValueError(
                "Trust-anchor public key is not "
                "valid Base64."
            ) from error

    required_fields = {
        "server_id",
        "algorithm",
        "public_key",
    }

    missing_fields = (
        required_fields.difference(
            normalized.keys()
        )
    )

    if missing_fields:
        raise ValueError(
            "Trust anchor is missing fields: "
            f"{sorted(missing_fields)}"
        )

    if not isinstance(
        normalized["server_id"],
        str,
    ):
        raise TypeError(
            "Trust-anchor server_id must be a string."
        )

    if not isinstance(
        normalized["algorithm"],
        str,
    ):
        raise TypeError(
            "Trust-anchor algorithm must be a string."
        )

    if not isinstance(
        normalized["public_key"],
        bytes,
    ):
        raise TypeError(
            "Trust-anchor public_key must be bytes."
        )

    return normalized


class MobileStation:
    """
    FT-QuPAP Mobile Station implementation.

    Typical interaction:

        request = mobile.create_request("urban")

        # Send request to the Authentication Server.
        # Receive the signed ephemeral ML-KEM package.

        session = mobile.prepare_session(
            request=request,
            server_information=server_package,
            channel=IDEAL_CHANNEL,
        )
    """

    def __init__(
        self,
        pseudonym_id: str,
        trust_anchor: Any,
        rng: np.random.Generator | None = None,
        protocol_name: str = PROTOCOL_NAME,
        protocol_version: str = PROTOCOL_VERSION,
    ) -> None:
        if not isinstance(
            pseudonym_id,
            str,
        ):
            raise TypeError(
                "pseudonym_id must be a string."
            )

        normalized_pseudonym = (
            pseudonym_id.strip()
        )

        if not normalized_pseudonym:
            raise ValueError(
                "pseudonym_id cannot be empty."
            )

        if not normalized_pseudonym.startswith(
            "PID-"
        ):
            raise ValueError(
                "pseudonym_id must begin with 'PID-'."
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

        if not isinstance(
            protocol_name,
            str,
        ) or not protocol_name:
            raise ValueError(
                "protocol_name must be a non-empty string."
            )

        if not isinstance(
            protocol_version,
            str,
        ) or not protocol_version:
            raise ValueError(
                "protocol_version must be a non-empty string."
            )

        self.pseudonym_id = (
            normalized_pseudonym
        )

        self.trust_anchor = (
            normalize_trust_anchor(
                trust_anchor
            )
        )

        self.rng = rng
        self.protocol_name = protocol_name
        self.protocol_version = (
            protocol_version
        )

    def create_request(
        self,
        context: str = "urban",
        timestamp: int | None = None,
        nonce: bytes | None = None,
    ) -> dict[str, Any]:
        """
        Create one fresh pseudonymous authentication request.
        """

        return create_authentication_request(
            pseudonym_id=(
                self.pseudonym_id
            ),
            context=context,
            timestamp=timestamp,
            nonce=nonce,
        )

    def create_retry_request(
        self,
        context: str,
        timestamp: int | None = None,
    ) -> dict[str, Any]:
        """
        Create a retry request with a new random nonce.

        The earlier nonce and session keys must never be reused.
        """

        return self.create_request(
            context=context,
            timestamp=timestamp,
            nonce=None,
        )

    def prepare_session(
        self,
        request: Mapping[str, Any],
        server_information: Mapping[str, Any],
        channel: ChannelConfig,
        use_css: bool = True,
        verification_time: int | None = None,
    ) -> PreparedMobileSession:
        """
        Prepare and transmit one FT-QuPAP Mobile Station attempt.

        The server package must correspond to the supplied request.
        """

        total_started_at = (
            time.perf_counter()
        )

        if not isinstance(
            request,
            Mapping,
        ):
            raise TypeError(
                "request must be a mapping."
            )

        if not isinstance(
            server_information,
            Mapping,
        ):
            raise TypeError(
                "server_information must be a mapping."
            )

        if not isinstance(
            channel,
            ChannelConfig,
        ):
            raise TypeError(
                "channel must be a ChannelConfig."
            )

        if not isinstance(
            use_css,
            bool,
        ):
            raise TypeError(
                "use_css must be boolean."
            )

        request_copy = copy.deepcopy(
            dict(request)
        )

        server_information_copy = (
            copy.deepcopy(
                dict(server_information)
            )
        )

        validate_authentication_request(
            request_copy
        )

        if (
            request_copy["pseudonym_id"]
            != self.pseudonym_id
        ):
            raise MobileSessionPreparationError(
                "Request pseudonym does not match "
                "this Mobile Station."
            )

        if (
            request_copy["service_context"]
            != channel.context
        ):
            raise MobileSessionPreparationError(
                "Request service context does not "
                "match the channel context."
            )

        # =====================================================
        # Step 1: Verify signed server package
        # =====================================================

        stage_started_at = (
            time.perf_counter()
        )

        server_verification = (
            verify_server_credential_detailed(
                server_info=(
                    server_information_copy
                ),
                trust_anchor=(
                    self.trust_anchor
                ),
                request=request_copy,
                now=verification_time,
            )
        )

        server_verification_s = (
            time.perf_counter()
            - stage_started_at
        )

        if not server_verification.valid:
            raise ServerCredentialRejectedError(
                server_verification
            )

        verified_mlkem_public_key = (
            server_verification
            .require_valid_public_key()
        )

        # =====================================================
        # Step 2: ML-KEM encapsulation
        # =====================================================

        stage_started_at = (
            time.perf_counter()
        )

        kem_result: MLKEMEncapsulationResult = (
            encapsulate_session_secret(
                verified_mlkem_public_key
            )
        )

        mlkem_encapsulation_s = (
            time.perf_counter()
            - stage_started_at
        )

        # =====================================================
        # Step 3: Transcript hash and session keys
        # =====================================================

        stage_started_at = (
            time.perf_counter()
        )

        unsigned_server_info = (
            unsigned_server_information(
                server_information_copy
            )
        )

        transcript_hash = (
            build_transcript_hash(
                request=request_copy,
                server_info_unsigned=(
                    unsigned_server_info
                ),
                protocol_name=(
                    self.protocol_name
                ),
                protocol_version=(
                    self.protocol_version
                ),
            )
        )

        key_material = (
            derive_session_key_material(
                shared_secret=(
                    kem_result.shared_secret
                ),
                transcript_hash=(
                    transcript_hash
                ),
            )
        )

        k_auth, k_ctrl = (
            split_session_keys(
                key_material
            )
        )

        transcript_and_kdf_s = (
            time.perf_counter()
            - stage_started_at
        )

        # =====================================================
        # Step 4: Generate the 128-bit KMAC tag
        # =====================================================

        stage_started_at = (
            time.perf_counter()
        )

        authentication_tag = (
            compute_authentication_tag(
                k_auth=k_auth,
                request=request_copy,
                transcript_hash=(
                    transcript_hash
                ),
            )
        )

        kmac_generation_s = (
            time.perf_counter()
            - stage_started_at
        )

        # =====================================================
        # Step 5: Prepare payload and check logical blocks
        # =====================================================

        stage_started_at = (
            time.perf_counter()
        )

        payload_specs = (
            map_tag_to_logical_specs(
                authentication_tag
            )
        )

        check_specs = (
            generate_check_specs(
                check_count=(
                    STANDARD_CHECK_COUNT
                ),
                rng=self.rng,
                require_standard_count=True,
            )
        )

        logical_preparation_s = (
            time.perf_counter()
            - stage_started_at
        )

        if len(payload_specs) != (
            PAYLOAD_LOGICAL_BLOCK_COUNT
        ):
            raise MobileSessionPreparationError(
                "Payload preparation did not produce "
                "128 logical blocks."
            )

        if len(check_specs) != (
            CHECK_LOGICAL_BLOCK_COUNT
        ):
            raise MobileSessionPreparationError(
                "Check preparation did not produce "
                "32 logical blocks."
            )

        # =====================================================
        # Step 6: Random logical-block interleaving
        # =====================================================

        stage_started_at = (
            time.perf_counter()
        )

        ordered_specs, schedule = (
            create_interleaved_schedule(
                payload_specs=payload_specs,
                check_specs=check_specs,
                rng=self.rng,
                validate_standard_counts=True,
            )
        )

        interleaving_s = (
            time.perf_counter()
            - stage_started_at
        )

        # =====================================================
        # Step 7: Steane CSS encoding
        # =====================================================

        stage_started_at = (
            time.perf_counter()
        )

        encoded_frame = (
            encode_ft_qupap_frame(
                ordered_specs=ordered_specs,
                rng=self.rng,
                use_css=use_css,
                require_standard_counts=True,
            )
        )

        steane_encoding_s = (
            time.perf_counter()
            - stage_started_at
        )

        # =====================================================
        # Step 8: Complete and protect the control schedule
        # =====================================================

        stage_started_at = (
            time.perf_counter()
        )

        completed_schedule = (
            attach_expected_reference_bits(
                schedule=schedule,
                encoded_check_blocks=(
                    encoded_frame.check_blocks
                ),
                require_steane_blocks=(
                    use_css
                ),
            )
        )

        protected_schedule = (
            protect_control_schedule(
                schedule=completed_schedule,
                k_ctrl=k_ctrl,
                transcript_hash=(
                    transcript_hash
                ),
            )
        )

        schedule_protection_s = (
            time.perf_counter()
            - stage_started_at
        )

        # =====================================================
        # Step 9: Quantum-channel transmission
        # =====================================================

        stage_started_at = (
            time.perf_counter()
        )

        transmitter = QuantumTransmitter(
            channel=channel,
            rng=self.rng,
        )

        transmission_result = (
            transmitter.transmit(
                encoded_frame
            )
        )

        quantum_transmission_s = (
            time.perf_counter()
            - stage_started_at
        )

        total_s = (
            time.perf_counter()
            - total_started_at
        )

        session_secrets = (
            MobileStationSessionSecrets(
                shared_secret=(
                    kem_result.shared_secret
                ),
                k_auth=k_auth,
                k_ctrl=k_ctrl,
            )
        )

        timings = MobileStationTimings(
            server_verification_s=(
                server_verification_s
            ),
            mlkem_encapsulation_s=(
                mlkem_encapsulation_s
            ),
            transcript_and_kdf_s=(
                transcript_and_kdf_s
            ),
            kmac_generation_s=(
                kmac_generation_s
            ),
            logical_preparation_s=(
                logical_preparation_s
            ),
            interleaving_s=(
                interleaving_s
            ),
            steane_encoding_s=(
                steane_encoding_s
            ),
            schedule_protection_s=(
                schedule_protection_s
            ),
            quantum_transmission_s=(
                quantum_transmission_s
            ),
            total_s=total_s,
        )

        return PreparedMobileSession(
            request=request_copy,
            server_information=(
                server_information_copy
            ),
            ciphertext=(
                kem_result.ciphertext
            ),
            transcript_hash=(
                transcript_hash
            ),
            authentication_tag=(
                authentication_tag
            ),
            protected_schedule=(
                protected_schedule
            ),
            encoded_frame=(
                encoded_frame
            ),
            transmission_result=(
                transmission_result
            ),
            server_verification=(
                server_verification
            ),
            timings=timings,
            _secrets=session_secrets,
            protocol_name=(
                self.protocol_name
            ),
            protocol_version=(
                self.protocol_version
            ),
        )

    def prepare_authentication_attempt(
        self,
        request: Mapping[str, Any],
        server_information: Mapping[str, Any],
        channel: ChannelConfig,
        use_css: bool = True,
        verification_time: int | None = None,
    ) -> PreparedMobileSession:
        """
        Compatibility alias for prepare_session().
        """

        return self.prepare_session(
            request=request,
            server_information=(
                server_information
            ),
            channel=channel,
            use_css=use_css,
            verification_time=(
                verification_time
            ),
        )


def print_safe_session_summary(
    session: PreparedMobileSession,
) -> None:
    """
    Print a session summary without exposing secrets or raw tags.
    """

    if not isinstance(
        session,
        PreparedMobileSession,
    ):
        raise TypeError(
            "session must be a PreparedMobileSession."
        )

    print(
        json.dumps(
            session.safe_summary(),
            indent=4,
        )
    )


def run_self_test() -> None:
    """
    Run a complete Mobile Station integration self-test.

    Test sequence:

    1. Generate temporary Authentication Server ML-DSA keys.
    2. Generate temporary ephemeral ML-KEM keys.
    3. Create a pseudonymous authentication request.
    4. Build and sign the server package.
    5. Verify the server package.
    6. Perform ML-KEM encapsulation.
    7. Prepare the complete 160-block logical frame.
    8. Encode it into 1,120 physical qubits.
    9. Protect the control schedule.
    10. Transmit through the ideal channel.
    11. Verify Mobile Station and server shared secrets.
    """

    from pqcrypto.kem import ml_kem_768
    from pqcrypto.sign import ml_dsa_65

    print("=" * 72)
    print("FT-QuPAP Mobile Station Integration Self-Test")
    print("=" * 72)

    signing_public_key, signing_secret_key = (
        ml_dsa_65.generate_keypair()
    )

    kem_public_key, kem_secret_key = (
        ml_kem_768.generate_keypair()
    )

    test_timestamp = int(
        time.time()
    )

    trust_anchor = {
        "server_id": "AS-6G-001",
        "algorithm": "ML-DSA-65",
        "public_key": signing_public_key,
        "trust_anchor_version": 1,
    }

    mobile_station = MobileStation(
        pseudonym_id="PID-6G-UE-0001",
        trust_anchor=trust_anchor,
        rng=np.random.default_rng(
            20260701
        ),
    )

    request = mobile_station.create_request(
        context="urban",
        timestamp=test_timestamp,
        nonce=b"0123456789ABCDEF",
    )

    server_package = (
        build_test_server_package(
            request=request,
            server_id=(
                trust_anchor["server_id"]
            ),
            kem_public_key=(
                kem_public_key
            ),
            signing_secret_key=(
                signing_secret_key
            ),
            timestamp=test_timestamp,
        )
    )

    session = (
        mobile_station.prepare_session(
            request=request,
            server_information=(
                server_package
            ),
            channel=IDEAL_CHANNEL,
            use_css=True,
            verification_time=(
                test_timestamp
            ),
        )
    )

    server_shared_secret = (
        ml_kem_768.decrypt(
            kem_secret_key,
            session.ciphertext,
        )
    )

    server_shared_secret = bytes(
        server_shared_secret
    )

    shared_secrets_match = (
        hmac.compare_digest(
            session.shared_secret,
            server_shared_secret,
        )
    )

    server_key_material = (
        derive_session_key_material(
            shared_secret=(
                server_shared_secret
            ),
            transcript_hash=(
                session.transcript_hash
            ),
        )
    )

    server_k_auth, server_k_ctrl = (
        split_session_keys(
            server_key_material
        )
    )

    session_keys_match = (
        hmac.compare_digest(
            session.k_auth,
            server_k_auth,
        )
        and hmac.compare_digest(
            session.k_ctrl,
            server_k_ctrl,
        )
    )

    transport_package = (
        session
        .to_classical_transport_dictionary()
    )

    forbidden_transport_fields = {
        "shared_secret",
        "k_auth",
        "k_ctrl",
        "authentication_tag",
        "raw_tag",
        "plaintext_schedule",
        "eve_fraction",
        "attacked_mask",
    }

    transport_is_safe = not any(
        field_name in transport_package
        for field_name
        in forbidden_transport_fields
    )

    valid_frame_counts = (
        session.encoded_frame
        .logical_block_count
        == TOTAL_LOGICAL_BLOCK_COUNT
        and session.encoded_frame
        .total_physical_qubits
        == STEANE_PHYSICAL_QUBIT_COUNT
    )

    print(
        "Server credential valid   : "
        f"{session.server_verification.valid}"
    )

    print(
        "Server verification reason: "
        f"{session.server_verification.reason}"
    )

    print(
        "ML-KEM ciphertext bytes   : "
        f"{len(session.ciphertext)}"
    )

    print(
        "Shared secrets match      : "
        f"{shared_secrets_match}"
    )

    print(
        "Separated keys match      : "
        f"{session_keys_match}"
    )

    print(
        "KMAC tag bits             : "
        f"{len(session.authentication_tag) * 8}"
    )

    print(
        "Logical blocks            : "
        f"{session.encoded_frame.logical_block_count}"
    )

    print(
        "Physical qubits           : "
        f"{session.encoded_frame.total_physical_qubits}"
    )

    print(
        "Frame counts valid        : "
        f"{valid_frame_counts}"
    )

    print(
        "Classical transport safe  : "
        f"{transport_is_safe}"
    )

    if not session.server_verification.valid:
        raise MobileStationError(
            "Valid server package was rejected."
        )

    if not shared_secrets_match:
        raise MobileStationError(
            "ML-KEM shared secrets do not match."
        )

    if not session_keys_match:
        raise MobileStationError(
            "Mobile and server session keys do not match."
        )

    if not valid_frame_counts:
        raise MobileStationError(
            "Encoded FT-QuPAP frame has invalid counts."
        )

    if not transport_is_safe:
        raise MobileStationError(
            "Classical transport package exposed "
            "secret data."
        )

    print(
        "\nSafe session summary:"
    )

    print_safe_session_summary(
        session
    )

    print(
        "\nMobile Station integration self-test "
        "completed successfully."
    )


__all__ = [
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "MobileStation",
    "PreparedMobileSession",
    "MobileStationTimings",
    "MobileStationSessionSecrets",
    "MobileStationError",
    "MobileSessionPreparationError",
    "ServerCredentialRejectedError",
    "build_transcript_hash",
    "normalize_trust_anchor",
    "print_safe_session_summary",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        MobileStationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[MOBILE STATION ERROR] "
            f"{error}"
        )