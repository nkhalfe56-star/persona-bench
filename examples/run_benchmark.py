#!/usr/bin/env python3
"""
examples/run_benchmark.py

End-to-end example: generate 5 personas, simulate a 3-turn conversation
with GPT-4o, score each interaction, and print a Markdown report.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/run_benchmark.py
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("[error] OPENAI_API_KEY environment variable is not set.")

from personabench import PersonaBench


def main() -> None:
    print("=" * 60)
    print("  persona-bench  |  End-to-End Benchmark Example")
    print("=" * 60)

    bench = PersonaBench(api_key=api_key, model="gpt-4o")

    results = bench.run(
        target_model="gpt-4o",
        num_personas=5,
        turns_per_conversation=3,
        scenario="customer-support",
        output_format="markdown",
    )

    print("\n" + results.report)
    print(f"\nOverall benchmark score: {results.mean_score:.2f}/10")
    print(f"Personas tested         : {results.num_personas}")
    print(f"Total conversations     : {results.total_conversations}")
    print(f"Failure modes detected  : {len(results.failure_modes)}")

    if results.failure_modes:
        print("\nFailure modes:")
        for mode in results.failure_modes:
            print(f"  - {mode}")

    results.save_json("benchmark_results.json")
    print("\nFull results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
