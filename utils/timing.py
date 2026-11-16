"""Re-measure per-request latency for a collection that did not record it.

    python -m utils.timing --model gemma4:e4b --out raw/london/latency_remeasured.csv

`utils.latency` reads `latency_s` from the archived run records. The London
Gemma 4 E4B collection predates that field, so its latency is measured here
instead: the first `--sample` reviews of the London corpus are replayed under
the three T = 1 schemes through the same experiment configuration and backend
call, one request at a time, and one row per request is written. Needs the
model served by Ollama. This is a re-measurement under the stated
configuration, not a recovery of what the original collection did, and nothing
is written into `raw/<corpus>/<model>/`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from backends import BackendType, get_backend
from experiments import load_experiments

from .parsing import SCHEMES
from .profiles import ROOT, load_profile

EXPERIMENTS_YAML = ROOT / "config" / "experiments.yaml"


def measure(
    raw_model: str,
    sample: int = 60,
    warmup: int = 3,
    backend_name: str = "ollama",
) -> pd.DataFrame:
    """One row per request: model, scheme, review position, latency.

    The first `warmup` requests of each scheme are timed and reported like the
    rest but flagged, because the first call after a model is loaded pays for
    the load and is not representative of a sweep. The summary drops them; the
    rows are kept so that the choice is visible rather than baked in.
    """

    data_csv = load_profile("london").data_csv
    experiments = load_experiments(EXPERIMENTS_YAML)
    backend = get_backend(BackendType(backend_name))
    reviews = pd.read_csv(data_csv).head(sample)

    rows = []
    for scheme, (experiment_name, _) in SCHEMES.items():
        experiment = experiments[experiment_name]
        for i, row in enumerate(reviews.itertuples()):
            started = time.perf_counter()
            experiment.run(
                backend,
                model_id=raw_model,
                review=row.REVIEW_BODY,
                review_idx=f"{row.USER_ID}_{row.ENTITY_ID}",
                run_number=0,
                input_file=str(data_csv),
            )
            rows.append(
                {
                    "model": raw_model,
                    "scheme": scheme,
                    "review": i,
                    "latency_s": round(time.perf_counter() - started, 4),
                    "think": experiment.config.think,
                    "warmup": i < warmup,
                }
            )
    return pd.DataFrame(rows)


def summarize(measured: pd.DataFrame) -> pd.DataFrame:
    """Median and spread per model, warmup requests excluded."""

    steady = measured[~measured["warmup"]]
    return (
        steady.groupby("model")
        .agg(
            n=("latency_s", "size"),
            mean_s=("latency_s", lambda s: round(s.mean(), 3)),
            median_s=("latency_s", lambda s: round(s.median(), 3)),
            p95_s=("latency_s", lambda s: round(s.quantile(0.95), 3)),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="Raw model id, e.g. gemma4:e4b")
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    measured = measure(args.model, args.sample, args.warmup, backend_name=args.backend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    measured.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(measured)} requests)")
    print(summarize(measured).to_string(index=False))


if __name__ == "__main__":
    main()
