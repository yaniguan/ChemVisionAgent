"""CLI entry point for ChemVision Agent."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from rich.console import Console

from chemvision.core.log import get_logger

app = typer.Typer(
    name="chemvision",
    help="ChemVision Agent — multimodal scientific image reasoning platform.",
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)


def _version_callback(value: bool) -> None:
    if value:
        from chemvision import __version__

        console.print(f"chemvision {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ChemVision Agent — multimodal scientific image reasoning platform."""


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the API server."),
    port: int = typer.Option(8000, help="Port to bind the API server."),
    reload: bool = typer.Option(False, help="Enable hot-reload for development."),
) -> None:
    """Start the ChemVision Agent REST API server."""
    import uvicorn

    console.print(f"[bold green]Starting ChemVision API on {host}:{port}[/bold green]")
    uvicorn.run("chemvision.api:app", host=host, port=port, reload=reload)


@app.command()
def reason(
    image: str = typer.Argument(..., help="Path to scientific image"),
    question: str = typer.Option("What is shown in this image?", help="Analysis question"),
    skills: str = typer.Option("auto", help="Comma-separated skills or 'auto'"),
    output: str = typer.Option("json", help="Output format: json, text, or markdown"),
    verbose: bool = typer.Option(False, help="Show agent reasoning trace"),
) -> None:
    """Run the ChemVision agent on a scientific image."""
    # Validate image exists
    image_path = Path(image)
    if not image_path.exists():
        console.print(f"[red]Error:[/red] Image file not found: {image}")
        raise typer.Exit(1)

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it with:  export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "Or get one at: https://console.anthropic.com/settings/keys"
        )
        raise typer.Exit(1)

    from chemvision.agent.agent import ChemVisionAgent
    from chemvision.agent.config import AgentConfig

    # Build config
    skill_names: list[str] = []
    if skills != "auto":
        skill_names = [s.strip() for s in skills.split(",") if s.strip()]

    config = AgentConfig(
        anthropic_api_key=api_key,
        skill_names=skill_names,
        verbose=verbose,
    )

    agent = ChemVisionAgent(config)

    if verbose:
        console.print(f"[cyan]Image:[/cyan]  {image}")
        console.print(f"[cyan]Query:[/cyan]  {question}")
        console.print(f"[cyan]Skills:[/cyan] {skills}")
        console.print()

    try:
        if verbose:
            # Stream mode: print each step as it arrives
            from chemvision.agent.report import AnalysisReport
            from chemvision.agent.trace import AgentStep

            report = None
            for event in agent.run_stream(question=question, image_paths=[str(image_path)]):
                if isinstance(event, AnalysisReport):
                    report = event
                elif isinstance(event, AgentStep):
                    console.print(
                        f"[dim]{event.step_type.value.upper()}[/dim] "
                        f"{event.content[:200]}"
                    )
            if report is None:
                console.print("[red]Error:[/red] Agent did not produce a report.")
                raise typer.Exit(1)
        else:
            report = agent.run(question=question, image_paths=[str(image_path)])
    except Exception as exc:
        logger.error("Agent failed during reasoning", exc_info=exc)
        console.print(f"[red]Error:[/red] Agent failed: {exc}")
        raise typer.Exit(1)

    # Format output
    if output == "json":
        console.print_json(json.dumps(report.to_dict(), indent=2, default=str))
    elif output == "text":
        console.print(f"[bold]Answer:[/bold] {report.final_answer}")
        if report.low_confidence_flag:
            console.print(
                f"[yellow]Warning:[/yellow] Low confidence "
                f"(min={report.min_intermediate_confidence:.2f})"
            )
        console.print(f"[dim]Steps: {report.num_steps} | "
                       f"Tools: {len(report.tool_logs)}[/dim]")
    elif output == "markdown":
        console.print(f"# ChemVision Analysis\n")
        console.print(f"**Question:** {report.question}\n")
        console.print(f"**Answer:** {report.final_answer}\n")
        if report.low_confidence_flag:
            console.print(
                f"> **Warning:** Low confidence "
                f"(min={report.min_intermediate_confidence:.2f})\n"
            )
        console.print(f"**Steps:** {report.num_steps} | "
                       f"**Tools:** {len(report.tool_logs)}\n")
        if report.tool_logs:
            console.print("## Tool Calls\n")
            for log in report.tool_logs:
                conf = f" (conf={log.confidence:.2f})" if log.confidence is not None else ""
                console.print(f"- **{log.skill_name}**{conf}: {log.output_summary[:100]}")
    else:
        console.print(f"[red]Error:[/red] Unknown output format: {output}")
        raise typer.Exit(1)


@app.command()
def audit(
    output_dir: str = typer.Option("./audit_results", help="Output directory"),
    format: str = typer.Option("json", help="Output format: json or markdown"),
) -> None:
    """Run capability audit and generate report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Running capability audit...[/cyan]")

    try:
        from chemvision.audit.matrix import CapabilityMatrix, MatrixConfig

        matrix_cfg = MatrixConfig(output_dir=out_path)
        matrix = CapabilityMatrix(matrix_cfg)

        # Export the matrix structure (even without a live model evaluation,
        # this produces the empty/template matrix for inspection)
        matrix_data = matrix.to_dict()

        if format == "json":
            result = {
                "audit_type": "capability_matrix",
                "output_dir": str(out_path),
                "matrix": matrix_data,
                "task_types": matrix.TASK_TYPES,
                "difficulties": matrix.DIFFICULTIES,
            }
            result_path = out_path / "audit_report.json"
            result_path.write_text(json.dumps(result, indent=2, default=str))
            console.print(f"[green]Audit report saved to:[/green] {result_path}")
            console.print_json(json.dumps(result, indent=2, default=str))

        elif format == "markdown":
            lines = [
                "# ChemVision Capability Audit\n",
                "## Capability Matrix\n",
                "| Task Type | " + " | ".join(matrix.DIFFICULTIES) + " |",
                "| --- | " + " | ".join(["---"] * len(matrix.DIFFICULTIES)) + " |",
            ]
            for task in matrix.TASK_TYPES:
                row_vals = [
                    f"{matrix_data[task][d]:.2f}"
                    if not (matrix_data[task][d] != matrix_data[task][d])  # NaN check
                    else "N/A"
                    for d in matrix.DIFFICULTIES
                ]
                lines.append(f"| {task} | " + " | ".join(row_vals) + " |")
            lines.append("")

            report_text = "\n".join(lines)
            result_path = out_path / "audit_report.md"
            result_path.write_text(report_text)
            console.print(f"[green]Audit report saved to:[/green] {result_path}")
            console.print(report_text)
        else:
            console.print(f"[red]Error:[/red] Unknown format: {format}")
            raise typer.Exit(1)

        # Try to export heatmap
        try:
            heatmap_path = matrix.export_heatmap(output_dir=out_path)
            console.print(f"[green]Heatmap saved to:[/green] {heatmap_path}")
        except ImportError:
            console.print("[yellow]Skipping heatmap (matplotlib not installed).[/yellow]")

    except Exception as exc:
        logger.error("Capability audit failed", exc_info=exc)
        console.print(f"[red]Audit failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def evaluate(
    generated: str = typer.Argument(..., help="File with generated SMILES (one per line)"),
    training: str = typer.Option(None, help="Training SMILES file for novelty/FCD"),
    output: str = typer.Option("json", help="Output format: json, text, or markdown"),
) -> None:
    """Compute MOSES metrics on generated molecules."""
    gen_path = Path(generated)
    if not gen_path.exists():
        console.print(f"[red]Error:[/red] File not found: {generated}")
        raise typer.Exit(1)

    # Load generated SMILES
    gen_smiles = [
        line.strip() for line in gen_path.read_text().splitlines() if line.strip()
    ]
    if not gen_smiles:
        console.print("[red]Error:[/red] No SMILES found in the file.")
        raise typer.Exit(1)

    console.print(f"[cyan]Loaded {len(gen_smiles)} generated SMILES[/cyan]")

    # Load training SMILES if provided
    train_smiles: list[str] | None = None
    if training:
        train_path = Path(training)
        if not train_path.exists():
            console.print(f"[red]Error:[/red] Training file not found: {training}")
            raise typer.Exit(1)
        train_smiles = [
            line.strip() for line in train_path.read_text().splitlines() if line.strip()
        ]
        console.print(f"[cyan]Loaded {len(train_smiles)} training SMILES[/cyan]")

    from chemvision.eval.moses_metrics import compute_moses_metrics

    console.print("[cyan]Computing MOSES metrics...[/cyan]")
    metrics = compute_moses_metrics(gen_smiles, train_smiles)

    if output == "json":
        from dataclasses import asdict

        console.print_json(json.dumps(asdict(metrics), indent=2))
    elif output == "text":
        console.print(metrics.summary())
    elif output == "markdown":
        console.print("# MOSES Metrics\n")
        console.print(f"| Metric | Value |")
        console.print(f"| --- | --- |")
        console.print(f"| Validity | {metrics.validity:.1%} |")
        console.print(f"| Uniqueness | {metrics.uniqueness:.1%} |")
        console.print(f"| Novelty | {metrics.novelty:.1%} |")
        console.print(f"| IntDiv1 | {metrics.int_div_1:.4f} |")
        console.print(f"| IntDiv2 | {metrics.int_div_2:.4f} |")
        console.print(f"| FCD | {metrics.fcd:.2f} |")
        console.print(f"| FCD proxy | {metrics.fcd_proxy:.2f} |")
        console.print(f"| Scaffold Div | {metrics.scaffold_diversity:.4f} |")
        console.print(f"| MW | {metrics.mw_mean:.1f} +/- {metrics.mw_std:.1f} |")
        console.print(f"| LogP | {metrics.logp_mean:.2f} |")
        console.print(f"| QED | {metrics.qed_mean:.3f} |")
    else:
        console.print(f"[red]Error:[/red] Unknown output format: {output}")
        raise typer.Exit(1)


@app.command()
def train(
    dataset: str = typer.Option("zinc250k", help="Dataset: zinc250k or path to CSV"),
    max_molecules: int = typer.Option(10000, help="Max molecules to use"),
    epochs: int = typer.Option(50, help="Training epochs"),
    output_dir: str = typer.Option("./checkpoints", help="Save directory"),
    config: str = typer.Option(None, help="JSON config file for SELFIESGenConfig"),
) -> None:
    """Train the SELFIES molecular generator."""
    import numpy as np

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load config
    from chemvision.generation.selfies_gen import SELFIESGenConfig, SELFIESGenerator

    if config:
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[red]Error:[/red] Config file not found: {config}")
            raise typer.Exit(1)
        gen_config = SELFIESGenConfig(**json.loads(config_path.read_text()))
    else:
        gen_config = SELFIESGenConfig()

    # Load dataset
    console.print(f"[cyan]Loading dataset: {dataset}[/cyan]")
    if dataset == "zinc250k":
        from chemvision.data.zinc250k import load_zinc250k

        smiles_list, props = load_zinc250k(max_molecules=max_molecules)
    else:
        # Assume CSV with 'smiles' column
        import pandas as pd

        csv_path = Path(dataset)
        if not csv_path.exists():
            console.print(f"[red]Error:[/red] Dataset file not found: {dataset}")
            raise typer.Exit(1)
        df = pd.read_csv(csv_path)
        smiles_col = [c for c in df.columns if "smile" in c.lower()][0]
        smiles_list = df[smiles_col].tolist()[:max_molecules]

        # Try to extract numeric properties; fall back to dummy
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            props = df[numeric_cols].values[:max_molecules].astype(np.float32)
        else:
            props = np.zeros((len(smiles_list), gen_config.cond_dim), dtype=np.float32)

    console.print(f"[cyan]Loaded {len(smiles_list)} molecules[/cyan]")

    # Adjust cond_dim to match properties
    if props.ndim == 1:
        props = props.reshape(-1, 1)
    if props.shape[1] != gen_config.cond_dim:
        console.print(
            f"[yellow]Adjusting cond_dim from {gen_config.cond_dim} to "
            f"{props.shape[1]} to match properties[/yellow]"
        )
        gen_config.cond_dim = props.shape[1]

    # Train
    console.print(f"[cyan]Training for {epochs} epochs...[/cyan]")
    generator = SELFIESGenerator(gen_config)

    try:
        result = generator.train(
            smiles_list, props, epochs=epochs, batch_size=64, patience=10, verbose=True
        )
    except Exception as exc:
        logger.error("SELFIES generator training failed", exc_info=exc)
        console.print(f"[red]Training failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green]Training complete![/green]")
    console.print(f"  Epochs:     {result.epochs}")
    console.print(f"  Final loss: {result.final_loss:.4f}")
    console.print(f"  Vocab size: {result.vocab_size}")
    console.print(f"  Converged:  {result.converged}")

    # Save training info
    info = {
        "epochs": result.epochs,
        "final_loss": result.final_loss,
        "vocab_size": result.vocab_size,
        "converged": result.converged,
        "loss_history": result.loss_history,
        "config": gen_config.__dict__,
        "dataset": dataset,
        "n_molecules": len(smiles_list),
    }
    info_path = out_path / "train_info.json"
    info_path.write_text(json.dumps(info, indent=2, default=str))
    console.print(f"[green]Training info saved to:[/green] {info_path}")


@app.command()
def version() -> None:
    """Print the ChemVision Agent version."""
    from chemvision import __version__

    console.print(f"chemvision {__version__}")


if __name__ == "__main__":
    app()
