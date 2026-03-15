"""
persona-bench: Stress-test your AI app with synthetic users before you ship.
"""

__version__ = "0.1.0"
__author__ = "nkhalfe56-star"

from personabench.core import PersonaBench
from personabench.schemas import (
    BenchmarkReport,
    ConversationResult,
    Persona,
    PersonaArchetype,
    ScoreBreakdown,
    FailureMode,
)

__all__ = [
    "PersonaBench",
    "BenchmarkReport",
    "ConversationResult",
    "Persona",
    "PersonaArchetype",
    "ScoreBreakdown",
    "FailureMode",
]
