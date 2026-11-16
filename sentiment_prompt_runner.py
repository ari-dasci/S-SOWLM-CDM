#!/usr/bin/env python
import dataclasses
import json
import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Annotated, ClassVar

import pandas as pd
import typer
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

from backends import BackendType, get_backend
from experiments import Experiment, load_experiments
from experiments.experiments import ChatResult
from models import ModelFamily, ModelRegistry


class PathHighlighter(RegexHighlighter):
    base_style = "repr."
    highlights: ClassVar[list[str]] = [
        r"(?P<path>\/[\w\-\.\/]+)",  # Unix paths
        r"(?P<path>[\w\-]+\.(?:json|csv|yaml|py|log))",  # Filenames with extensions
    ]


console = Console(theme=Theme({"logging.level.success": "green"}), record=True)


def setup_logging(
    log_path: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure logging with Rich console and optional file output."""

    logger = logging.getLogger("sentiment")
    logger.setLevel(level)
    logger.handlers.clear()  # Avoid duplicate handlers on re-init

    # Console handler with Rich (pretty output)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
        highlighter=PathHighlighter(),
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler (plain text, more detail)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Capture everything in file
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


REGISTRY = ModelRegistry(Path(__file__).parent / "config" / "models.yaml")
_CONFIG_DIR = Path(__file__).parent / "config"
# experiments.yaml holds the three schemes the paper reports. A second file of
# your own is picked up here if you add one; it may be absent.
EXPERIMENTS = load_experiments(
    _CONFIG_DIR / "experiments.yaml",
    _CONFIG_DIR / "experiments_extra.yaml",
)


def list_experiments() -> list[str]:
    return list(EXPERIMENTS.keys())


app = typer.Typer(help="Sentiment analysis experiment runner")


def generate_log_path(
    experiments: list[str],
    model_id: str,
    runs: int,
    output_dir: Path = Path("logs"),
) -> Path:
    """Generate descriptive log filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model_id.replace(":", "-")

    if len(experiments) == 1:
        exp_part = experiments[0]
    else:
        exp_part = f"{len(experiments)}exp"

    filename = f"{timestamp}_{model_safe}_{exp_part}_{runs}runs.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def validate_model_family(value: str) -> str:
    valid = [f.value for f in ModelFamily]
    if value not in valid:
        raise typer.BadParameter(f"Must be one of: {', '.join(valid)}")
    return value


@app.command()
def run(
    backend: Annotated[BackendType, typer.Option(help="Backend to run the LLM model")],
    model: Annotated[
        str,
        typer.Option(
            help="Model family",
            callback=validate_model_family,
            autocompletion=lambda: [f.value for f in ModelFamily],
        ),
    ],
    experiment: Annotated[
        str,
        typer.Option(
            "--experiment",
            "-e",
            help="Experiment name",
            autocompletion=lambda: list(EXPERIMENTS.keys()),
        ),
    ],
    size: Annotated[str | None, typer.Option(help="Model size (e.g., 8b)")] = None,
    runs: Annotated[
        int,
        typer.Option(
            "--runs", "-r", help="Number of runs for each experiment (Default is 1)"
        ),
    ] = 1,
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to TripAdvisor csv data.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("data/TripR-2020Large_FullReviews.csv"),  # pyright: ignore[reportCallInDefaultInitializer]
    results_dir: Annotated[
        Path,
        typer.Option(
            "--results-dir",
            help="Root the run JSON is written under: <dir>/<model>/<experiment>/.",
        ),
    ] = Path("collected"),  # pyright: ignore[reportCallInDefaultInitializer]
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            min=1,
            help=(
                "Concurrent requests to the backend. Serve Ollama with "
                "OLLAMA_NUM_PARALLEL >= this. Recorded in every run's config, "
                "since latency measured under concurrency is not per-request "
                "deployment latency."
            ),
        ),
    ] = 1,
    draft_num_predict: Annotated[
        int | None,
        typer.Option(
            "--draft-num-predict",
            min=0,
            help=(
                "Ollama speculative-decoding draft length; 0 disables the draft "
                "model. Omit to leave the model default. Recorded in the config."
            ),
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help=(
                "Only the first N rows of --input. Row positions are unchanged, "
                "so a later run without --limit resumes these files."
            ),
        ),
    ] = None,
):
    # Validate experiment
    if experiment not in EXPERIMENTS:
        raise typer.BadParameter(
            f"Unknown experiment '{experiment}'. "
            f"Available: {', '.join(list_experiments())}"
        )

    exp = EXPERIMENTS[experiment]
    spec = REGISTRY.get(model, size)
    try:
        model_id = spec.get_id(backend)
    except ValueError:
        available = ", ".join(backend_id.value for backend_id in spec.backend_ids)
        raise typer.BadParameter(
            f"Model {spec.canonical_name} not available for "
            f"{backend.value}. Available backends: {available}"
        ) from None
    backend_instance = get_backend(backend)
    backend_instance.pull(model_id)

    log_path = generate_log_path(
        experiments=[experiment],
        model_id=model_id,
        runs=runs,
    )

    logger = setup_logging(log_path)
    # Load data
    logger.info(f"[bold blue]Loading data from {input_file}")
    df = pd.read_csv(input_file)
    if limit is not None:
        df = df.head(limit)
        logger.info(f"[dim]First {len(df)} rows only (--limit)")

    logger.info(f"[bold green]Running experiment: {experiment}")
    logger.info(f"[dim]{exp.config.description}")
    logger.info(f"[dim]Model {model_id}")
    logger.info(f"[dim]Runs per review: {runs}")
    logger.info(f"[dim]Workers: {workers}; results under {results_dir}")

    _run_single_experiment(
        exp,
        backend_instance,
        model_id,
        df,
        runs,
        # Relative, so a run record does not carry a local directory.
        os.path.relpath(input_file),
        logger,
        results_dir=results_dir,
        workers=workers,
        draft_num_predict=draft_num_predict,
    )


def _check_resume(save_path: str, review_idx: str) -> None:
    """Refuse to resume when the stored run is a different review.

    Runs are keyed by position in the dataframe (review-<i>), not by review, so
    a rerun with a different --input lands a different review on an existing
    directory. Without this check the existing runs are skipped and the new
    ones written beside them, leaving two reviews mixed in one folder.
    """

    with open(save_path) as fh:
        stored = json.load(fh).get("review_idx")

    if stored is not None and stored != review_idx:
        raise typer.BadParameter(
            f"{save_path} holds review {stored}, but this input has "
            f"{review_idx} in that position. Write to a clean results directory."
        )


def _run_single_experiment(
    exp: Experiment,
    backend_instance,
    model_id: str,
    df: pd.DataFrame,
    runs: int,
    input_file: str,
    logger: logging.Logger,
    results_dir: Path = Path("collected"),
    workers: int = 1,
    draft_num_predict: int | None = None,
) -> None:
    """Run experiment on individual reviews.

    Every (review, run) pair without a JSON on disk is one request. With
    `workers > 1` the requests go to a thread pool; each writes its own file
    the moment it returns, so an interrupted collection resumes exactly as a
    sequential one does. With one worker the requests run in review, then
    run order, as before. The worker count is stored in each record's config,
    one included.
    """

    exp = dataclasses.replace(
        exp,
        config=dataclasses.replace(
            exp.config, workers=workers, draft_num_predict=draft_num_predict
        ),
    )

    rows = list(df.itertuples())
    pending: list[tuple[int, int, str]] = []
    for i, row in enumerate(rows):
        review_idx = f"{row.USER_ID}_{row.ENTITY_ID}"
        for run_number in range(runs):
            save_path = os.path.join(
                str(results_dir),
                model_id,
                exp.name,
                f"review-{i}",
                f"run-{run_number}.json",
            )
            if os.path.exists(save_path):
                _check_resume(save_path, review_idx)
                continue
            pending.append((i, run_number, save_path))

    skipped = len(rows) * runs - len(pending)
    if skipped:
        logger.warning(f"[yellow]{skipped} run(s) already on disk, skipped")

    remaining: dict[int, int] = defaultdict(int)
    for i, _, _ in pending:
        remaining[i] += 1
    done: dict[int, list[ChatResult]] = defaultdict(list)
    lock = threading.Lock()

    def one(i: int, run_number: int, save_path: str) -> tuple[int, ChatResult]:
        row = rows[i]
        result = exp.run(
            backend_instance,
            model_id=model_id,
            review=row.REVIEW_BODY,
            review_idx=f"{row.USER_ID}_{row.ENTITY_ID}",
            run_number=run_number,
            input_file=input_file,
        )
        result.save_json(Path(save_path))
        return i, result

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"{exp.name}", total=len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, *job) for job in pending]
            for future in as_completed(futures):
                i, result = future.result()
                progress.update(task, advance=1)
                with lock:
                    done[i].append(result)
                    remaining[i] -= 1
                    finished = remaining[i] == 0
                if finished:
                    row = rows[i]
                    line = f"{row.USER_ID}_{row.ENTITY_ID}"
                    for r in sorted(done.pop(i), key=lambda r: r.run_number):
                        line += f" - {r.parsed.display}"
                    logger.info(line)


@app.command()
def experiments():
    """List available experiments."""
    table = Table(title="Available Experiments")
    table.add_column("Name", style="bold")
    table.add_column("Description")

    for name, exp in EXPERIMENTS.items():
        table.add_row(name, exp.config.description)

    console.print(table)


@app.command()
def models():
    """List available models."""
    table = Table(title="Available Models")
    table.add_column("Family", style="bold")
    table.add_column("Size")
    table.add_column("Default", justify="center")
    table.add_column("Backends", style="dim")

    for family in ModelFamily:
        try:
            sizes = REGISTRY.available_sizes(family)
            default_size = REGISTRY.get(family).size

            for size in sizes:
                spec = REGISTRY.get(family, size)
                backends = ", ".join(b.value for b in spec.backend_ids)
                is_default = "✓" if size == default_size else ""

                table.add_row(family.value, size, is_default, backends)
        except (KeyError, ValueError) as exc:
            console.print(f"[yellow]Skipping {family.value}: {exc}")
            continue

    console.print(table)


if __name__ == "__main__":
    app()
