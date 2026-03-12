"""Dataset statistics script.

Loads a saved HuggingFace ``DatasetDict`` and prints a rich summary covering
split sizes, domain distribution, difficulty distribution, source distribution,
and a random sample of QA pairs.

Usage
-----
Direct execution::

    python -m chemvision.data.data_stats --dataset-dir data/processed/hf_dataset

As a callable from Python::

    from chemvision.data.data_stats import print_stats
    from pathlib import Path
    print_stats(Path("data/processed/hf_dataset"))
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(name="data-stats", add_completion=False)
console = Console()


def print_stats(dataset_dir: Path, sample_size: int = 5) -> None:
    """Load a HuggingFace DatasetDict and print a rich summary to stdout.

    Parameters
    ----------
    dataset_dir:
        Path to a directory previously saved with ``DatasetDict.save_to_disk()``.
    sample_size:
        Number of random QA pairs to display as examples.
    """
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        console.print("[red]datasets package required. Install: pip install datasets[/red]")
        raise SystemExit(1) from exc

    ds_dict = load_from_disk(str(dataset_dir))

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]ChemVision Dataset Statistics[/bold cyan]\n"
            f"[dim]{dataset_dir}[/dim]",
        )
    )

    # ---- split sizes -------------------------------------------------------
    split_table = Table("Split", "Samples", title="Split Distribution", show_lines=True)
    total = 0
    all_records: list[dict] = []
    for split_name, ds in ds_dict.items():
        split_table.add_row(split_name, str(len(ds)))
        total += len(ds)
        all_records.extend(ds.to_list())
    split_table.add_section()
    split_table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
    console.print(split_table)

    if not all_records:
        console.print("[yellow]No records found in dataset.[/yellow]")
        return

    def _pct_table(title: str, counter: Counter) -> Table:
        t = Table("Value", "Count", "%", title=title, show_lines=True)
        for value, count in counter.most_common():
            pct = 100.0 * count / total
            t.add_row(str(value), str(count), f"{pct:.1f}%")
        return t

    # ---- distributions -----------------------------------------------------
    domain_counts: Counter = Counter(r.get("domain", "unknown") for r in all_records)
    console.print(_pct_table("Domain Distribution", domain_counts))

    difficulty_counts: Counter = Counter(
        r.get("difficulty") or "unset" for r in all_records
    )
    console.print(_pct_table("Difficulty Distribution", difficulty_counts))

    source_counts: Counter = Counter(r.get("source") or "unset" for r in all_records)
    console.print(_pct_table("Source Distribution", source_counts))

    # ---- answer length stats -----------------------------------------------
    lengths = [len(str(r.get("answer", ""))) for r in all_records]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        console.print(
            f"\n[dim]Answer length — min: {min(lengths)}  avg: {avg_len:.0f}  "
            f"max: {max(lengths)} chars[/dim]"
        )

    # ---- sample QA pairs ---------------------------------------------------
    console.print(f"\n[bold]Sample QA Pairs[/bold] (n={min(sample_size, len(all_records))})")
    sample = random.sample(all_records, min(sample_size, len(all_records)))
    for i, rec in enumerate(sample, 1):
        console.print(
            f"\n[dim]── Sample {i} / {min(sample_size, len(all_records))} ──[/dim]"
        )
        console.print(f"  [green]Q:[/green] {rec.get('question', 'N/A')}")
        console.print(f"  [yellow]A:[/yellow] {rec.get('answer', 'N/A')}")
        tags = (
            f"domain={rec.get('domain', '?')}  "
            f"difficulty={rec.get('difficulty', '?')}  "
            f"source={rec.get('source', '?')}"
        )
        console.print(f"  [dim]{tags}[/dim]")

    console.print()


@app.command()
def main(
    dataset_dir: Path = typer.Option(
        Path("data/processed/hf_dataset"),
        "--dataset-dir",
        "-d",
        help="Path to a HuggingFace DatasetDict saved with save_to_disk().",
        show_default=True,
    ),
    sample_size: int = typer.Option(
        5,
        "--sample",
        "-n",
        help="Number of random QA pairs to display as examples.",
    ),
) -> None:
    """Print summary statistics for a saved ChemVision HuggingFace dataset."""
    if not dataset_dir.exists():
        console.print(f"[red]Dataset directory not found: {dataset_dir}[/red]")
        raise typer.Exit(1)
    print_stats(dataset_dir, sample_size=sample_size)


if __name__ == "__main__":
    app()
