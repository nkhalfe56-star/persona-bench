"""
ConversationScorer: LLM-as-judge scoring for all conversations.
Uses GPT-4o to evaluate 5 dimensions per conversation.
"""

from __future__ import annotations

import json
import logging
from typing import List, Tuple

from openai import OpenAI

from personabench.schemas import (
    ConversationResult,
    ConversationTurn,
    FailureMode,
    Persona,
    ScoreBreakdown,
)

logger = logging.getLogger(__name__)


JUDGE_SYSTEM = """You are an expert AI quality evaluator. 
You evaluate AI assistant responses for quality, accuracy, and appropriateness.
Always return valid JSON. Be strict but fair in your scoring."""


def _build_judge_prompt(
    persona: Persona, turns: List[ConversationTurn], app_description: str = ""
) -> str:
    conversation_text = "\n".join(
        [
            f"Turn {t.turn_number}:\nUSER: {t.user_message}\nASSISTANT: {t.assistant_response}"
            for t in turns
        ]
    )
    return f"""Evaluate this AI conversation on 5 dimensions (score 0-10 each).

App Being Tested: {app_description or "An AI assistant"}

User Persona:
- Name: {persona.name}
- Archetype: {persona.archetype.value}
- Communication style: {persona.communication_style}
- Goal: {persona.goal}

Conversation:
{conversation_text}

Score each dimension 0-10:
1. consistency (10 = perfectly consistent answers, 0 = contradicts itself)
2. hallucination (10 = no hallucination at all, 0 = many fabricated facts)
3. tone_calibration (10 = tone perfectly matches what this persona needs, 0 = completely wrong tone)
4. graceful_failure (10 = handles unknowns gracefully, 0 = crashes or gives bad errors)
5. refusal_appropriateness (10 = refuses exactly what should be refused, 0 = over-refuses or under-refuses)

Also identify failure modes present (list any that apply):
- off_topic_deflection
- contradictory_answer  
- overly_verbose
- ignored_user_intent
- hallucination
- tone_mismatch
- over_refusal
- loop
- none

Also note any specific hallucination examples found.

Return JSON:
{{
  "scores": {{
    "consistency": <0-10>,
    "hallucination": <0-10>,
    "tone_calibration": <0-10>,
    "graceful_failure": <0-10>,
    "refusal_appropriateness": <0-10>
  }},
  "failure_modes": ["mode1", "mode2"],
  "hallucination_examples": ["example if any"],
  "judge_notes": "Brief explanation of key issues found"
}}"""


class ConversationScorer:
    def __init__(self, api_key: str, judge_model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.judge_model = judge_model

    def score_all(
        self,
        raw_results: List[Tuple[Persona, List[ConversationTurn]]],
    ) -> List[ConversationResult]:
        """Score all conversations and return ConversationResult objects."""
        scored = []
        for persona, turns in raw_results:
            if not turns:
                result = self._empty_result(persona)
            else:
                result = self._score_one(persona, turns)
            scored.append(result)
        return scored

    def _score_one(self, persona: Persona, turns: List[ConversationTurn]) -> ConversationResult:
        try:
            prompt = _build_judge_prompt(persona, turns)
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            data = json.loads(response.choices[0].message.content)

            scores = ScoreBreakdown(**data["scores"])
            failure_modes = [
                FailureMode(m) for m in data.get("failure_modes", ["none"])
                if m in [f.value for f in FailureMode]
            ] or [FailureMode.NONE]

            return ConversationResult(
                persona=persona,
                turns=turns,
                scores=scores,
                failure_modes=failure_modes,
                hallucination_examples=data.get("hallucination_examples", []),
                judge_notes=data.get("judge_notes", ""),
            )
        except Exception as e:
            logger.warning(f"Scoring failed for {persona.name}: {e}")
            return self._fallback_result(persona, turns)

    def _empty_result(self, persona: Persona) -> ConversationResult:
        return ConversationResult(
            persona=persona,
            turns=[],
            scores=ScoreBreakdown(
                consistency=0, hallucination=10, tone_calibration=0,
                graceful_failure=0, refusal_appropriateness=5
            ),
            failure_modes=[FailureMode.NONE],
            judge_notes="No conversation turns recorded.",
        )

    def _fallback_result(self, persona: Persona, turns: List[ConversationTurn]) -> ConversationResult:
        return ConversationResult(
            persona=persona,
            turns=turns,
            scores=ScoreBreakdown(
                consistency=5, hallucination=5, tone_calibration=5,
                graceful_failure=5, refusal_appropriateness=5
            ),
            failure_modes=[FailureMode.NONE],
            judge_notes="Scoring failed — fallback scores applied.",
        )
