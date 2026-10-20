"""cli interface for dataset-dreamer."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.syntax import Syntax

from dataset_dreamer.config import (
    DatasetDreamerConfig,
    ModelType,
    OutputFormat,
    SignalSource,
    SignalConfig,
    TrainingConfig,
    ConditioningConfig,
    GenerationConfig,
)
from dataset_dreamer.core.pipeline import DatasetDreamer, create_default_config

app = typer.Typer(
    name="dreamer",
    help="dataset-dreamer: generative data augmentation conditioned on external signals",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓  ║
║  ▓▓         ▓▓         ▓▓         ▓▓         ▓▓         ▓▓  ▓▓             ║
║  ▓▓▓▓▓▓▓   ▓▓▓▓▓▓▓    ▓▓▓▓▓▓▓    ▓▓▓▓▓▓▓    ▓▓▓▓▓▓▓    ▓▓  ▓▓▓▓▓▓▓▓▓▓      ║
║  ▓▓         ▓▓         ▓▓         ▓▓         ▓▓         ▓▓         ▓▓      ║
║  ▓▓         ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓      ║
║                                                                             ║
║         GENERATIVE DATA AUGMENTATION CONDITIONED ON EXTERNAL SIGNALS       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="show version"),
):
    if version:
        console.print("dataset-dreamer 0.1.0")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print(ctx.get_help())


@app.command()
def train(
    data: Annotated[Path, typer.Argument(help="path to training data (parquet, csv, jsonl)")],
    model_type: Annotated[ModelType, typer.Option("--model", "-m", help="model type")] = ModelType.CTGAN,
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="training epochs")] = 300,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="batch size")] = 500,
    lr: Annotated[float, typer.Option("--lr", help="learning rate")] = 2e-4,
    latent_dim: Annotated[int, typer.Option("--latent-dim", "-l", help="latent dimension")] = 16,
    signals: Annotated[list[SignalSource], typer.Option("--signal", "-s", help="enabled signals")] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="model output directory")] = Path("models"),
    seed: Annotated[int, typer.Option("--seed", help="random seed")] = 42,
    verbose: Annotated[bool, typer.Option("--verbose", help="verbose output")] = False,
):
    """train a conditional generative model on data."""
    print_banner()

    # setup config
    sig_configs = [SignalConfig(source=s, weight=1.0) for s in (signals or [
        SignalSource.MOON_PHASE,
        SignalSource.SOLAR_WIND,
        SignalSource.GIT_ACTIVITY,
        SignalSource.CRYPTO_VOLATILITY,
    ])]

    config = DatasetDreamerConfig(
        training=TrainingConfig(
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        ),
        conditioning=ConditioningConfig(
            signals=sig_configs,
            latent_dim=latent_dim,
        ),
        generation=GenerationConfig(model_path=output),
        seed=seed,
    )

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    dreamer = DatasetDreamer(config)

    # load data
    with console.status("[cyan]loading data...[/cyan]"):
        dreamer.load_data(data)
    console.print(f"[green]loaded[/green] {len(dreamer.data)} rows, {len(dreamer.data.columns)} columns")

    # fetch signals
    with console.status("[cyan]fetching external signals...[/cyan]"):
        dreamer.fetch_signals_sync()

    _print_signals(dreamer.signals)

    # train
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"training {model_type.value.upper()}...", total=epochs)

        # we can't easily hook into sdv's training progress, so just run
        dreamer.train()

        progress.update(task, completed=epochs)

    # save model
    model_path = dreamer.save_model()
    console.print(f"[green]model saved to[/green] {model_path}")


@app.command()
def generate(
    model: Annotated[Path, typer.Argument(help="path to trained model")],
    rows: Annotated[int, typer.Option("--rows", "-n", help="number of rows to generate")] = 1000,
    temperature: Annotated[float, typer.Option("--temp", "-t", help="sampling temperature")] = 1.0,
    format: Annotated[OutputFormat, typer.Option("--format", "-f", help="output format")] = OutputFormat.PARQUET,
    output: Annotated[Path, typer.Option("--output", "-o", help="output directory")] = Path("output"),
    signal_override: Annotated[list[str], typer.Option("--signal", "-s", help="override signal value (source=value)")] = None,
    seed: Annotated[int, typer.Option("--seed", help="random seed")] = 42,
):
    """generate synthetic data from a trained model."""
    print_banner()

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    dreamer = DatasetDreamer()
    dreamer.load_model(model)

    # parse signal overrides
    overrides = {}
    if signal_override:
        for s in signal_override:
            if "=" in s:
                src, val = s.split("=", 1)
                try:
                    overrides[SignalSource(src.strip())] = float(val)
                except ValueError:
                    console.print(f"[red]invalid signal source: {src}[/red]")

    with console.status("[cyan]generating...[/cyan]"):
        generated = dreamer.generate(
            n=rows,
            temperature=temperature,
            signal_overrides=overrides if overrides else None,
        )

    out_path = dreamer.save_output(generated, output, format)
    console.print(f"[green]generated {len(generated)} rows →[/green] {out_path}")

    # show sample
    _print_sample(generated)


@app.command()
def interpolate(
    model: Annotated[Path, typer.Argument(help="path to trained model")],
    start: Annotated[Path, typer.Argument(help="start data point (csv/jsonl)")],
    end: Annotated[Path, typer.Argument(help="end data point (csv/jsonl)")],
    steps: Annotated[int, typer.Option("--steps", help="interpolation steps")] = 50,
    output: Annotated[Path, typer.Option("--output", "-o", help="output directory")] = Path("interpolations"),
):
    """interpolate in latent space between two data points."""
    print_banner()

    dreamer = DatasetDreamer()
    dreamer.load_model(model)

    start_df = pd.read_csv(start) if start.suffix == ".csv" else pd.read_json(start, lines=True)
    end_df = pd.read_csv(end) if end.suffix == ".csv" else pd.read_json(end, lines=True)

    with console.status("[cyan]interpolating in latent space...[/cyan]"):
        frames = dreamer.interpolate_latent(start_df, end_df, steps)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        frame.to_parquet(output / f"interp_{i:03d}.parquet", index=False)

    console.print(f"[green]saved {len(frames)} interpolation frames to[/green] {output}")


@app.command()
def signals(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="show full metadata")] = False,
):
    """fetch and display current external signal values."""
    print_banner()

    config = create_default_config()
    dreamer = DatasetDreamer(config)

    with console.status("[cyan]fetching signals...[/cyan]"):
        dreamer.fetch_signals_sync()

    _print_signals(dreamer.signals, verbose)


@app.command()
def config(
    output: Annotated[Path, typer.Option("--output", "-o", help="output config file")] = Path("dreamer_config.json"),
):
    """generate a default configuration file."""
    config = create_default_config()
    output.write_text(config.model_dump_json(indent=2, default=str))
    console.print(f"[green]config written to[/green] {output}")


@app.command()
def ui(
    host: Annotated[str, typer.Option("--host", help="host to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="port to bind")] = 8080,
    reload: Annotated[bool, typer.Option("--reload", help="auto-reload on changes")] = False,
):
    """launch the web ui."""
    from dataset_dreamer.ui.server import run
    run(host=host, port=port, reload=reload)


def _print_signals(signals: dict, verbose: bool = False):
    table = Table(title="external signals", show_header=True, header_style="bold cyan")
    table.add_column("source", style="cyan")
    table.add_column("value", justify="right", style="green")
    table.add_column("weight", justify="right", style="yellow")
    table.add_column("metadata", style="dim")

    for src, sig in signals.items():
        meta_str = ""
        if verbose:
            meta_str = json.dumps(sig.metadata, default=str, indent=2)
        else:
            meta_str = ", ".join(f"{k}={v}" for k, v in list(sig.metadata.items())[:3])
        table.add_row(src.value, f"{sig.value:.4f}", "1.0", meta_str)

    console.print(table)


def _print_sample(data: pd.DataFrame, n: int = 5):
    console.print("\n[bold]sample:[/bold]")
    sample = data.head(n)
    # convert to string representation
    buf = sample.to_string(max_colwidth=30)
    console.print(Syntax(buf, "text", theme="monokai", line_numbers=False))


if __name__ == "__main__":
    app()