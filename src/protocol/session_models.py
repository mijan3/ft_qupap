from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SessionSummary:
    """
    Compact non-secret summary of one FT-QuPAP authentication session.

    This model stores only values that are safe for dashboard display,
    session history, logging, and result export.

    Secret material such as K_ss, K_auth, K_ctrl, ML-KEM secret keys,
    ML-DSA secret keys, and raw KMAC tags must not be stored here.
    """

    accepted: bool
    reason: str
    qber_raw: float | None
    p_attack: float | None
    retry_attempts: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError(
                "accepted must be boolean."
            )

        if not isinstance(self.reason, str):
            raise TypeError(
                "reason must be a string."
            )

        if not self.reason.strip():
            raise ValueError(
                "reason cannot be empty."
            )

        if self.qber_raw is not None:
            if isinstance(self.qber_raw, bool) or not isinstance(
                self.qber_raw,
                (int, float),
            ):
                raise TypeError(
                    "qber_raw must be numeric or None."
                )

            self.qber_raw = float(
                self.qber_raw
            )

            if not 0.0 <= self.qber_raw <= 1.0:
                raise ValueError(
                    "qber_raw must be between 0 and 1."
                )

        if self.p_attack is not None:
            if isinstance(self.p_attack, bool) or not isinstance(
                self.p_attack,
                (int, float),
            ):
                raise TypeError(
                    "p_attack must be numeric or None."
                )

            self.p_attack = float(
                self.p_attack
            )

            if not 0.0 <= self.p_attack <= 1.0:
                raise ValueError(
                    "p_attack must be between 0 and 1."
                )

        if isinstance(self.retry_attempts, bool) or not isinstance(
            self.retry_attempts,
            int,
        ):
            raise TypeError(
                "retry_attempts must be an integer."
            )

        if self.retry_attempts < 1:
            raise ValueError(
                "retry_attempts must be at least 1."
            )

    @property
    def retry_used(self) -> bool:
        """Return whether the session required a fresh retry."""

        return self.retry_attempts > 1

    def as_dict(self) -> dict[str, Any]:
        """Convert the session summary into a dictionary."""

        return {
            **asdict(self),
            "retry_used": self.retry_used,
        }

    @classmethod
    def from_session_result(
        cls,
        session: dict[str, Any],
    ) -> "SessionSummary":
        """
        Create a summary from a complete protocol-session dictionary.
        """

        if not isinstance(session, dict):
            raise TypeError(
                "session must be a dictionary."
            )

        decision = session.get(
            "decision",
            {},
        )

        if not isinstance(decision, dict):
            decision = {}

        accepted = bool(
            decision.get(
                "accepted",
                session.get(
                    "accepted",
                    False,
                ),
            )
        )

        reason = str(
            decision.get(
                "reason",
                session.get(
                    "reason",
                    "unknown",
                ),
            )
        )

        qber_raw = session.get(
            "qber_raw"
        )

        p_attack = decision.get(
            "p_attack",
            session.get(
                "p_attack"
            ),
        )

        retry_attempts = session.get(
            "retry_attempts",
            1,
        )

        return cls(
            accepted=accepted,
            reason=reason,
            qber_raw=(
                None
                if qber_raw is None
                else float(qber_raw)
            ),
            p_attack=(
                None
                if p_attack is None
                else float(p_attack)
            ),
            retry_attempts=int(
                retry_attempts
            ),
        )


__all__ = [
    "SessionSummary",
]