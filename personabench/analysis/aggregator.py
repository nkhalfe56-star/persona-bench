"""
ResultAggregator: Builds the final BenchmarkReport from scored results.
"""

from __future__ import annotations

from typing import List

from personabench.schemas import BenchmarkReport, ConversationResult


class ResultAggregator:
    def build_report(
        self,
        app_url: str,
        app_description: str,
        results: List[ConversationResult],
    ) -> BenchmarkReport:
        """Build a complete BenchmarkReport from scored conversation results."""
        return BenchmarkReport(
            app_url=app_url,
            app_description=app_description,
            n_personas=len(results),
            results=results,
        )

    def per_archetype_scores(self, report: BenchmarkReport) -> dict:
        """Compute average scores grouped by persona archetype."""
        from collections import defaultdict
        archetype_scores = defaultdict(list)
        for result in report.results:
            archetype_scores[result.persona.archetype.value].append(result.scores.overall)

        return {
            archetype: round(sum(scores) / len(scores), 2)
            for archetype, scores in archetype_scores.items()
            if scores
        }
