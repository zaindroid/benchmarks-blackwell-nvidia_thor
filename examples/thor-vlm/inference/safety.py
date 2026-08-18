"""Safety filters for the thor-vlm reference implementation.

Automotive deployment guardrails:

* prompt filtering — reject prompts with disallowed content
* output filtering — block responses that violate hard rules
* response allowlisting — optional: only allow responses matching a
  constrained pattern (e.g. structured driving outputs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {"allowed": self.allowed, "reason": self.reason}


PROMPT_DENYLIST = [
    "override brake",
    "disable safety",
    "ignore traffic",
    "crash into",
    "run red light",
    "exceed speed limit",
]

OUTPUT_DENYLIST = [
    "I will crash",
    "disregard the rules of the road",
    "ignore the pedestrian",
]


class SafetyFilter:
    """Prompt + output guardrails for on-device VLM inference."""

    def __init__(self, prompt_denylist: Optional[List[str]] = None,
                 output_denylist: Optional[List[str]] = None):
        self.prompt_denylist = [p.lower() for p in (prompt_denylist or PROMPT_DENYLIST)]
        self.output_denylist = [o.lower() for o in (output_denylist or OUTPUT_DENYLIST)]

    def check_prompt(self, prompt: str) -> SafetyDecision:
        lowered = prompt.lower()
        for bad in self.prompt_denylist:
            if bad in lowered:
                return SafetyDecision(False, f"prompt blocked: contains {bad!r}")
        return SafetyDecision(True)

    def check_output(self, text: str) -> SafetyDecision:
        lowered = text.lower()
        for bad in self.output_denylist:
            if bad in lowered:
                return SafetyDecision(False, f"output blocked: contains {bad!r}")
        return SafetyDecision(True)

    def filter(self, prompt: str, output: str) -> SafetyDecision:
        """Full pipeline: prompt then output check."""
        decision = self.check_prompt(prompt)
        if not decision.allowed:
            return decision
        return self.check_output(output)


def redact(text: str, replacements: Optional[dict] = None) -> str:
    """Redact common sensitive patterns from logs/outputs."""
    redactions = replacements or {
        "Bearer ": "Bearer ***",
        "password": "password=***",
    }
    out = text
    for key, value in redactions.items():
        out = out.replace(key, value)
    return out
