"""
PersonaFactory: Generates diverse synthetic user personas using GPT-4o.
"""

from __future__ import annotations

import json
import uuid
import logging
from typing import List

from openai import OpenAI

from personabench.schemas import Persona, PersonaArchetype

logger = logging.getLogger(__name__)

ARCHETYPES = [a.value for a in PersonaArchetype]

SYSTEM_PROMPT = """You are an expert UX researcher and behavioral psychologist.
Your job is to generate realistic, diverse synthetic user personas for AI application testing.
Each persona should feel like a real human with consistent behavior patterns.
"""

def _persona_prompt(app_description: str, archetype: str, index: int) -> str:
    return f"""Generate a realistic synthetic user persona for testing an AI application.

App Description: {app_description}

Archetype: {archetype}
Persona #{index}

Return a JSON object with exactly these fields:
{{
  "name": "Full Name",
  "archetype": "{archetype}",
  "age": <integer between 18 and 75>,
  "background": "2-3 sentence professional/personal background",
  "goal": "What this user wants to achieve with the app",
  "communication_style": "How this person communicates",
  "frustration_triggers": ["trigger1", "trigger2", "trigger3"],
  "typical_phrases": ["phrase1", "phrase2", "phrase3", "phrase4"]
}}

Make the persona realistic, specific, and internally consistent with the {archetype} archetype.
"""


class PersonaFactory:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, n: int, app_description: str) -> List[Persona]:
        """Generate n diverse personas, cycling through archetypes."""
        personas = []
        for i in range(n):
            archetype = ARCHETYPES[i % len(ARCHETYPES)]
            try:
                persona = self._generate_one(
                    app_description=app_description,
                    archetype=archetype,
                    index=i + 1,
                )
                personas.append(persona)
            except Exception as e:
                logger.warning(f"Failed to generate persona {i+1}: {e}. Using fallback.")
                personas.append(self._fallback_persona(archetype, i))
        return personas

    def _generate_one(self, app_description: str, archetype: str, index: int) -> Persona:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _persona_prompt(app_description, archetype, index)},
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
        )
        data = json.loads(response.choices[0].message.content)
        data["id"] = str(uuid.uuid4())[:8]
        return Persona(**data)

    def _fallback_persona(self, archetype: str, index: int) -> Persona:
        """Fallback persona when API call fails."""
        return Persona(
            id=str(uuid.uuid4())[:8],
            name=f"Test User {index}",
            archetype=PersonaArchetype(archetype),
            age=30,
            background="A typical user of the application.",
            goal="Get help from the application.",
            communication_style="Direct and concise.",
            frustration_triggers=["Slow responses", "Vague answers"],
            typical_phrases=["Help me with this", "I need assistance", "Can you explain"],
        )
