"""
persona-bench CLI.
Install with: pip install -e .
Usage: pb run --help
"""

import os
import subprocess
import typer
from typing import Optional

app = typer.Typer(
    name="pb",
    help="persona-bench: Stress-test your AI app with synthetic users before you ship.",
    add_completion=False,
)


@app.command()
def run(
    app_url: str = typer.Option(..., "--app-url", "-u", help="URL of the AI app's chat endpoint"),
    description: str = typer.Option("", "--description", "-d", help="Description of your AI app"),
    personas: int = typer.Option(20, "--personas", "-n", help="Number of personas to generate"),
    turns: int = typer.Option(5, "--turns", "-t", help="Conversation turns per persona"),
    workers: int = typer.Option(10, "--workers", "-w", help="Parallel workers"),
    output_dir: str = typer.Option("results", "--output-dir", "-o", help="Output directory"),
    model: str = typer.Option("gpt-4o", "--model", help="OpenAI model for generation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run the persona-bench benchmark against your AI app."""
    from personabench import PersonaBench

    typer.echo(f"🤖 persona-bench starting...")
    typer.echo(f"   App URL: {app_url}")
    typer.echo(f"   Personas: {personas} | Turns: {turns} | Workers: {workers}")
    typer.echo("")

    pb = PersonaBench(
        app_url=app_url,
        app_description=description,
        model=model,
        verbose=verbose,
    )

    report = pb.run(
        n_personas=personas,
        turns_per_persona=turns,
        max_workers=workers,
        output_dir=output_dir,
    )

    typer.echo("")
    typer.echo("=" * 50)
    typer.echo(f"✅ Benchmark complete!")
    typer.echo(f"   Overall Score:      {report.overall_score}/10")
    typer.echo(f"   Hallucination Rate: {report.hallucination_rate:.1%}")
    typer.echo(f"   Top Failure Mode:   {report.top_failure_mode}")
    typer.echo(f"   Results saved to:   {output_dir}/")
    typer.echo("")
    typer.echo("Run `pb dashboard` to view the interactive dashboard.")


@app.command()
def dashboard(
    results_file: str = typer.Option("results/latest.json", "--results-file", "-f"),
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit port"),
):
    """Open the persona-bench Streamlit dashboard."""
    import sys
    dashboard_path = os.path.join(
        os.path.dirname(__file__), "dashboard", "app.py"
    )
    env = {**os.environ, "RESULTS_FILE": results_file}
    typer.echo(f"🖥️  Opening dashboard at http://localhost:{port}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dashboard_path, "--server.port", str(port)],
        env=env,
    )


def main():
    app()


if __name__ == "__main__":
    main()
