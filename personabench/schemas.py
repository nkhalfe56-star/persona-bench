"""
Pydantic schemas for persona-bench.
All data models used across the pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PersonaArchetype(str, Enum):
    ANGRY = "angry"
    CONFUSED = "confused"
    ADVERSARIAL = "adversarial"
    ELDERLY = "elderly"
    NON_NATIVE = "non_native"
    POWER_USER = "power_user"
    IMPATIENT = "impatient"
    VERBOSE = "verbose"
    SKEPTICAL = "skeptical"
    CASUAL = "casual"


class Persona(BaseModel):
    id: str
    name: str
    archetype: PersonaArchetype
    age: int
    background: str
    goal: str
    communication_style: str
    frustration_triggers: List[str]
    typical_phrases: List[str]


class ConversationTurn(BaseModel):
    turn_number: int
    user_message: str
    assistant_response: str
    latency_ms: Optional[float] = None


class FailureMode(str, Enum):
    OFF_TOPIC_DEFLECTION = "off_topic_deflection"
    CONTRADICTORY_ANSWER = "contradictory_answer"
    OVERLY_VERBOSE = "overly_verbose"
    IGNORED_USER_INTENT = "ignored_user_intent"
    HALLUCINATION = "hallucination"
    TONE_MISMATCH = "tone_mismatch"
    OVER_REFUSAL = "over_refusal"
    LOOP = "loop"
    NONE = "none"


class ScoreBreakdown(BaseModel):
    consistency: float = Field(..., ge=0, le=10)
    hallucination: float = Field(..., ge=0, le=10, description="10 = no hallucination")
    tone_calibration: float = Field(..., ge=0, le=10)
    graceful_failure: float = Field(..., ge=0, le=10)
    refusal_appropriateness: float = Field(..., ge=0, le=10)

    @property
    def overall(self) -> float:
        return round(
            (
                self.consistency
                + self.hallucination
                + self.tone_calibration
                + self.graceful_failure
                + self.refusal_appropriateness
            )
            / 5,
            2,
        )


class ConversationResult(BaseModel):
    persona: Persona
    turns: List[ConversationTurn]
    scores: ScoreBreakdown
    failure_modes: List[FailureMode]
    hallucination_examples: List[str] = []
    judge_notes: str = ""


class BenchmarkReport(BaseModel):
    app_url: str
    app_description: str
    run_at: datetime = Field(default_factory=datetime.utcnow)
    n_personas: int
    results: List[ConversationResult]

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.scores.overall for r in self.results) / len(self.results), 2)

    @property
    def hallucination_rate(self) -> float:
        total = len(self.results)
        if not total:
            return 0.0
        flagged = sum(1 for r in self.results if FailureMode.HALLUCINATION in r.failure_modes)
        return flagged / total

    @property
    def top_failure_mode(self) -> str:
        from collections import Counter
        all_modes = [m for r in self.results for m in r.failure_modes if m != FailureMode.NONE]
        if not all_modes:
            return "none"
        return Counter(all_modes).most_common(1)[0][0].value

    def save(self, path: str) -> None:
        import json
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    def to_heatmap_data(self) -> Dict[str, Any]:
        """Returns data for the failure heatmap: archetype x failure_mode matrix."""
        from collections import defaultdict
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for result in self.results:
            archetype = result.persona.archetype.value
            for mode in result.failure_modes:
                if mode != FailureMode.NONE:
                    matrix[archetype][mode.value] += 1
        return dict(matrix)
