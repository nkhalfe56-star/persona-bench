"""
Reporter: Saves BenchmarkReport to disk in JSON and Markdown formats.
"""

from __future__ import annotations

import os
import json
from datetime import datetime

from personabench.schemas import BenchmarkReport


class Reporter:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, report: BenchmarkReport) -> None:
        """Save report as JSON and Markdown."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._save_json(report, timestamp)
        self._save_markdown(report, timestamp)
        self._save_latest_symlink(report)

    def _save_json(self, report: BenchmarkReport, timestamp: str) -> None:
        path = os.path.join(self.output_dir, f"report_{timestamp}.json")
        with open(path, "w") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"JSON report saved: {path}")

    def _save_latest_symlink(self, report: BenchmarkReport) -> None:
        """Always keep results/latest.json pointing to the most recent run."""
        path = os.path.join(self.output_dir, "latest.json")
        with open(path, "w") as f:
            f.write(report.model_dump_json(indent=2))

    def _save_markdown(self, report: BenchmarkReport, timestamp: str) -> None:
        path = os.path.join(self.output_dir, f"report_{timestamp}.md")
        lines = [
            f"# persona-bench Report",
            f"",
            f"**Run at:** {report.run_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**App:** {report.app_url}",
            f"**Personas tested:** {report.n_personas}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Overall Score | **{report.overall_score}/10** |",
            f"| Hallucination Rate | {report.hallucination_rate:.1%} |",
            f"| Top Failure Mode | `{report.top_failure_mode}` |",
            f"",
            f"## Per-Persona Results",
            f"",
            f"| Persona | Archetype | Overall | Consistency | Hallucination | Tone | Failure Modes |",
            f"|---|---|---|---|---|---|---|",
        ]
        for result in report.results:
            s = result.scores
            modes = ", ".join([m.value for m in result.failure_modes])
            lines.append(
                f"| {result.persona.name} | {result.persona.archetype.value} "
                f"| {s.overall} | {s.consistency} | {s.hallucination} "
                f"| {s.tone_calibration} | {modes} |"
            )

        if any(result.hallucination_examples for result in report.results):
            lines += ["", "## Hallucination Examples", ""]
            for result in report.results:
                for ex in result.hallucination_examples:
                    lines.append(f"- **{result.persona.name}**: {ex}")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"Markdown report saved: {path}")
