"""Per-request wall-clock latency, read from the raw archives.

    python -m utils.latency london
    python -m utils.latency madrid --raw-dir raw/madrid_latency

`latency_s` is written by `experiments.experiments.Experiment.run`, measured
around the backend call. Writes `latency.csv` (one row per request),
`latency_summary.csv` (per model) and `tables/latency.tex`.

Two collections need something other than their own archives:

- London, Gemma 4 E4B: collected before the runner recorded `latency_s`. Its
  row comes from `raw/london/latency_remeasured.csv`, a replay of the first 60 reviews
  under the three schemes on the same hardware (`utils.timing`). GPT-5.2 has
  no comparable latency and is left out.
- Madrid: collected with four requests in flight, so its recorded latencies
  are not single-request ones. `raw/madrid_latency` holds a sequential
  re-measurement (60 reviews, one run, one request at a time), passed with
  `--raw-dir`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .profiles import ROOT, Profile, load_profile
from .runs import iter_records


def collect_latencies(profile: Profile, raw_dir: Path | None = None) -> pd.DataFrame:
    """One row per archived run: model, scheme, review, run, latency_s."""

    source = profile if raw_dir is None else replace(profile, raw_dir=raw_dir)
    rows: list[dict[str, object]] = []
    for entry in profile.models:
        for scheme, (scheme_dir, _) in profile.schemes.items():
            if not source.archive(entry.raw, scheme_dir).is_file():
                continue
            for review_dir, record in iter_records(source, entry.raw, scheme_dir):
                rows.append(
                    {
                        "model": entry.display,
                        "scheme": scheme,
                        "review_id": review_dir,
                        "run": record.get("run_number"),
                        "latency_s": record.get("latency_s"),
                    }
                )
    return pd.DataFrame(rows)


def summarize(latencies: pd.DataFrame) -> pd.DataFrame:
    """Per-model latency summary: n, missing, mean, median, p95.

    Median and p95 alongside the mean on purpose: a reasoning-capable model
    that emits a long trace on a minority of reviews has a mean pulled far
    above its typical request.
    """

    rows = []
    for model, g in latencies.groupby("model", sort=False):
        observed = g["latency_s"].dropna()
        rows.append(
            {
                "model": model,
                "n": len(g),
                "missing": int(g["latency_s"].isna().sum()),
                "mean_s": round(observed.mean(), 3) if len(observed) else float("nan"),
                "median_s": (
                    round(observed.median(), 3) if len(observed) else float("nan")
                ),
                "p95_s": (
                    round(observed.quantile(0.95), 3) if len(observed) else float("nan")
                ),
                "total_s": round(observed.sum(), 1) if len(observed) else float("nan"),
                "source": "instrumented" if len(observed) else "none",
            }
        )
    return pd.DataFrame(rows).set_index("model")


def merge_remeasured(
    summary: pd.DataFrame, remeasured: pd.DataFrame, display_of: dict[str, str]
) -> pd.DataFrame:
    """Fill in models whose archives predate the instrumentation.

    `remeasured` is the per-request output of `utils.timing`, which replays a
    sample of the corpus through the same experiment configuration on the same
    hardware. It fills a row only where the archive gave nothing, and the
    `source` column records which of the two the figure came from -- these are
    not the same measurement and the manuscript has to say which it is quoting.
    Warm-up requests are excluded; the load of the model into VRAM is paid once
    per sweep, not once per review.
    """

    if remeasured.empty:
        return summary

    out = summary.copy()
    steady = remeasured[~remeasured["warmup"]]
    for raw, g in steady.groupby("model", sort=False):
        model = display_of.get(str(raw), str(raw))
        if model not in out.index or out.loc[model, "source"] == "instrumented":
            continue
        observed = g["latency_s"].dropna()
        if observed.empty:
            continue
        out.loc[model, ["mean_s", "median_s", "p95_s"]] = [
            round(observed.mean(), 3),
            round(observed.median(), 3),
            round(observed.quantile(0.95), 3),
        ]
        out.loc[model, "source"] = "remeasured"
        out.loc[model, "remeasured_n"] = len(observed)
    return out


def latency_to_latex(summary: pd.DataFrame, out_path: Path) -> None:
    """Per-request latency table, replacing the old hand-written CPU timings."""

    measured = summary[summary["mean_s"].notna()]
    # The Source column only where the rows were not all obtained the same way.
    with_source = measured["source"].nunique() > 1
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Wall-clock latency per request on a single consumer GPU, "
            r"measured around the backend call over every collected run. $n$ is "
            r"the number of requests; median and the 95th percentile are "
            r"reported alongside the mean because a model that emits a "
            r"reasoning trace on a minority of reviews has a mean far above its "
            r"typical request. Source distinguishes the two ways a row can be "
            r"obtained: \emph{instrumented} means every request of the "
            r"collection was timed as it was made; \emph{remeasured} means the "
            r"collection predates that instrumentation and the figure comes "
            r"from replaying the first 60 reviews of the corpus under all "
            r"three schemes through the same experiment configuration on the "
            r"same hardware (180 requests, of which the first 3 per scheme are "
            r"discarded as warm-up, leaving 171). A "
            r"remeasured row states what the system costs now under the stated "
            r"configuration; it is not a recovery of what the original sweep "
            r"did. Systems with neither are omitted rather than estimated.}"
        ),
        r"\label{tab:latency}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr" + ("l" if with_source else "") + "}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{$n$} & \textbf{Mean (s)}"
            r" & \textbf{Median (s)} & \textbf{p95 (s)}"
            + (r" & \textbf{Source}" if with_source else "")
            + r" \\"
        ),
        r"\midrule",
    ]
    # A model whose runs carry no `latency_s` and no remeasurement has nothing
    # to report. Printing `nan` into a manuscript is worse than leaving the row
    # out.
    # Requests of a few hundredths of a second need the third decimal.
    places = 3 if measured["median_s"].min() < 0.1 else 2
    for model, row in measured.iterrows():
        instrumented = row.get("source", "instrumented") == "instrumented"
        n = int(row["n"]) if instrumented else int(row.get("remeasured_n", 0))
        lines.append(
            f"  {model} & {n:,}".replace(",", "{,}")
            + f" & {row['mean_s']:.{places}f}"
            + f" & {row['median_s']:.{places}f} & {row['p95_s']:.{places}f}"
            + (f" & {row['source']}" if with_source else "")
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("profile")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Archives to time instead of the profile's own (a sequential "
        "re-measurement).",
    )
    args = parser.parse_args()

    profile = load_profile(args.profile)
    out = profile.out_dir
    raw_dir = None if args.raw_dir is None else (ROOT / args.raw_dir)
    latencies = collect_latencies(profile, raw_dir)
    latencies.to_csv(out / "latency.csv", index=False)

    summary = summarize(latencies)
    remeasured = profile.raw_dir / "latency_remeasured.csv"
    if remeasured.is_file():
        summary = merge_remeasured(
            summary,
            pd.read_csv(remeasured),
            {e.raw: e.display for e in profile.models},
        )
    print(summary.to_string())
    summary.to_csv(out / "latency_summary.csv")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    latency_to_latex(summary, out / "tables" / "latency.tex")


if __name__ == "__main__":
    main()
