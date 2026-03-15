"""
PersonaBench - Main orchestrator class.
Entry point for running the full benchmark pipeline.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from personabench.schemas import BenchmarkReport
from personabench.generator.persona_factory import PersonaFactory
from personabench.generator.conversation import ConversationGenerator
from personabench.runner.attack_runner import AttackRunner
from personabench.runner.scorer import ConversationScorer
from personabench.analysis.aggregator import ResultAggregator
from personabench.analysis.reporter import Reporter

logger = logging.getLogger(__name__)


class PersonaBench:
    """
    Main class for running the persona-bench benchmark.

    Usage:
        pb = PersonaBench(app_url="http://localhost:8000/chat")
        report = pb.run(n_personas=20)
        report.save("results/report.json")
    """

    def __init__(
        self,
        app_url: str,
        app_description: str = "",
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        judge_model: str = "gpt-4o",
        timeout: int = 30,
        verbose: bool = False,
    ):
        self.app_url = app_url
        self.app_description = app_description
        self.model = model
        self.judge_model = judge_model
        self.timeout = timeout

        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Pass openai_api_key= or set OPENAI_API_KEY env var."
            )
        self._api_key = api_key

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    def run(
        self,
        n_personas: int = 20,
        turns_per_persona: int = 5,
        max_workers: int = 10,
        output_dir: str = "results",
    ) -> BenchmarkReport:
        """
        Run the full benchmark pipeline.

        Args:
            n_personas: Number of synthetic personas to generate and test
            turns_per_persona: Number of conversation turns per persona
            max_workers: Parallelism for running personas simultaneously
            output_dir: Directory to save results

        Returns:
            BenchmarkReport with all results, scores, and heatmap data
        """
        logger.info(f"Starting persona-bench: {n_personas} personas x {turns_per_persona} turns")

        # Step 1: Generate personas
        logger.info("Generating synthetic personas...")
        factory = PersonaFactory(api_key=self._api_key, model=self.model)
        personas = factory.generate(n=n_personas, app_description=self.app_description)
        logger.info(f"Generated {len(personas)} personas")

        # Step 2: Generate conversations for each persona
        logger.info("Generating conversation scripts per persona...")
        conv_gen = ConversationGenerator(api_key=self._api_key, model=self.model)
        conversation_scripts = {
            p.id: conv_gen.generate(
                persona=p,
                app_description=self.app_description,
                n_turns=turns_per_persona,
            )
            for p in personas
        }

        # Step 3: Run conversations against the app in parallel
        logger.info(f"Running {n_personas} personas against {self.app_url} ...")
        runner = AttackRunner(
            app_url=self.app_url,
            timeout=self.timeout,
            max_workers=max_workers,
        )
        raw_results = runner.run_all(
            personas=personas,
            conversation_scripts=conversation_scripts,
        )

        # Step 4: Score each conversation
        logger.info("Scoring conversations with LLM-as-judge...")
        scorer = ConversationScorer(api_key=self._api_key, judge_model=self.judge_model)
        scored_results = scorer.score_all(raw_results)

        # Step 5: Aggregate and build report
        logger.info("Aggregating results...")
        aggregator = ResultAggregator()
        report = aggregator.build_report(
            app_url=self.app_url,
            app_description=self.app_description,
            results=scored_results,
        )

        # Step 6: Save results
        reporter = Reporter(output_dir=output_dir)
        reporter.save(report)
        logger.info(f"Results saved to {output_dir}/")
        logger.info(f"Overall Score: {report.overall_score}/10")
        logger.info(f"Hallucination Rate: {report.hallucination_rate:.1%}")
        logger.info(f"Top Failure Mode: {report.top_failure_mode}")

        return report
