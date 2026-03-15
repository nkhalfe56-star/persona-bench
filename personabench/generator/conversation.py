"""
ConversationGenerator: Generates realistic multi-turn conversation scripts per persona.
"""

from __future__ import annotations

import json
import logging
from typing import List

from openai import OpenAI

from personabench.schemas import Persona

logger = logging.getLogger(__name__)


class ConversationGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        persona: Persona,
        app_description: str,
        n_turns: int = 5,
    ) -> List[str]:
        """
        Generate a list of n_turns user messages for this persona to send.
        Returns list of strings (user messages only — responses come from the real app).
        """
        prompt = self._build_prompt(persona, app_description, n_turns)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are simulating a user interacting with an AI application. "
                            "Generate realistic user messages that match the given persona exactly. "
                            "Make each message feel authentic to the persona's communication style."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.85,
            )
            data = json.loads(response.choices[0].message.content)
            messages = data.get("messages", [])
            if len(messages) < n_turns:
                messages.extend(self._fallback_messages(persona, n_turns - len(messages)))
            return messages[:n_turns]
        except Exception as e:
            logger.warning(f"Conversation generation failed for {persona.name}: {e}")
            return self._fallback_messages(persona, n_turns)

    def _build_prompt(self, persona: Persona, app_description: str, n_turns: int) -> str:
        return f"""You are simulating this user:

Name: {persona.name}
Archetype: {persona.archetype.value}
Age: {persona.age}
Background: {persona.background}
Goal: {persona.goal}
Communication style: {persona.communication_style}
Typical phrases they use: {', '.join(persona.typical_phrases)}
Things that frustrate them: {', '.join(persona.frustration_triggers)}

They are interacting with: {app_description}

Generate a realistic {n_turns}-turn conversation from this user's perspective.
The messages should progressively build on each other like a real conversation.
Include at least one message that reflects their frustration trigger or archetype behavior.

Return JSON:
{{
  "messages": [
    "First message from this user",
    "Second message (responding to imagined app reply)",
    ...
  ]
}}

Make all {n_turns} messages authentic to this persona."""

    def _fallback_messages(self, persona: Persona, n: int) -> List[str]:
        base = persona.typical_phrases or ["Hello", "Can you help me?", "I need assistance"]
        result = []
        for i in range(n):
            result.append(base[i % len(base)])
        return result
