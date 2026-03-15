"""
AttackRunner: Runs all personas against the target app in parallel using ThreadPoolExecutor.
"""

from __future__ import annotations

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import requests

from personabench.schemas import Persona, ConversationTurn

logger = logging.getLogger(__name__)


class AttackRunner:
    """
    Sends generated conversation scripts to the target app in parallel.
    Returns raw conversation results (turns with app responses).
    """

    def __init__(self, app_url: str, timeout: int = 30, max_workers: int = 10):
        self.app_url = app_url
        self.timeout = timeout
        self.max_workers = max_workers

    def run_all(
        self,
        personas: List[Persona],
        conversation_scripts: Dict[str, List[str]],
    ) -> List[Tuple[Persona, List[ConversationTurn]]]:
        """
        Run all personas against the app simultaneously.
        Returns list of (persona, conversation_turns) tuples.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_persona,
                    persona,
                    conversation_scripts.get(persona.id, []),
                ): persona
                for persona in personas
            }
            for future in as_completed(futures):
                persona = futures[future]
                try:
                    turns = future.result()
                    results.append((persona, turns))
                    logger.debug(f"Completed persona: {persona.name} ({persona.archetype.value})")
                except Exception as e:
                    logger.warning(f"Persona {persona.name} failed: {e}")
                    results.append((persona, []))

        return results

    def _run_persona(
        self, persona: Persona, messages: List[str]
    ) -> List[ConversationTurn]:
        """Run a single persona's conversation against the app."""
        turns = []
        for i, message in enumerate(messages):
            start = time.time()
            response_text = self._send_message(message)
            latency_ms = (time.time() - start) * 1000

            turns.append(
                ConversationTurn(
                    turn_number=i + 1,
                    user_message=message,
                    assistant_response=response_text,
                    latency_ms=round(latency_ms, 2),
                )
            )
        return turns

    def _send_message(self, message: str) -> str:
        """Send a single message to the target app and return its response."""
        try:
            resp = requests.post(
                self.app_url,
                json={"message": message},
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            # Support common response field names
            return (
                data.get("response")
                or data.get("message")
                or data.get("content")
                or data.get("answer")
                or data.get("text")
                or str(data)
            )
        except requests.Timeout:
            return "[ERROR: Request timed out]"
        except requests.HTTPError as e:
            return f"[ERROR: HTTP {e.response.status_code}]"
        except Exception as e:
            return f"[ERROR: {str(e)}]"
