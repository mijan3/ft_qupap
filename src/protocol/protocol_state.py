from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProtocolState:
    stages: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def set_stage(self, name: str, status: str) -> None:
        self.stages[name] = status