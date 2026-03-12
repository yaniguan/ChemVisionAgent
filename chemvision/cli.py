"""CLI entry point for ChemVision Agent."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="chemvision",
    help="ChemVision Agent — multimodal scientific image reasoning platform.",
    no_args_is_help=True,
)
console = Console()


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
def audit(
    image: str = typer.Argument(..., help="Path to a scientific image to audit."),
    skills: list[str] = typer.Option([], "--skill", "-s", help="Skills to evaluate."),
) -> None:
    """Run the capability audit framework on a single image."""
    console.print(f"[cyan]Auditing:[/cyan] {image}")
    console.print(f"[cyan]Skills:[/cyan] {skills or 'all'}")
    console.print("[yellow]Audit not yet implemented.[/yellow]")


@app.command()
def reason(
    image: str = typer.Argument(..., help="Path to a scientific image."),
    query: str = typer.Argument(..., help="Natural-language question about the image."),
) -> None:
    """Run the ReAct agent on an image + query."""
    console.print(f"[cyan]Image:[/cyan]  {image}")
    console.print(f"[cyan]Query:[/cyan]  {query}")
    console.print("[yellow]Agent reasoning not yet implemented.[/yellow]")


if __name__ == "__main__":
    app()
