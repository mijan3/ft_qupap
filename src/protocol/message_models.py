from dataclasses import dataclass
from typing import Any


@dataclass
class AuthenticationRequest:
    pseudonym_id: str
    timestamp: int
    nonce: str
    service_context: str
    request_type: str = "FT-QuPAP-Authentication"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()