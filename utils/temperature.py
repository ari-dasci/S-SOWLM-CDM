"""Is any of this an artifact of sampling at $T = 1$?

The study rests on repeated inference producing different answers, and a reader
is entitled to ask whether greedy decoding would simply remove the problem. The
`single_label_5c_t0` collection answers it: same prompt, same vocabulary, same
parser, same 474 reviews, same five systems, ten runs each, with the sampling
temperature the only thing that changes.

The contrast has to separate three quantities that the phrase "model
variability" runs together, because they do not behave the same way.

    run-to-run spread   How much the ten runs of one review disagree. This is
                        what greedy decoding is expected to remove, and it
                        does.

    resolution          How many distinct values the review-level aggregate can
                        take. Counter-intuitively this *falls* at $T = 0$: when
                        the ten runs agree, their mean is the term index itself,
                        so the aggregate lands back on the grid. Sampling is
                        not only a defect here, it is also the mechanism that
                        gives the per-review aggregate its fine granularity.

    the decision        Whether the alternatives separate, and in what order.
                        This is the quantity the paper is about, and it is the
                        one greedy decoding does not fix.

`contrast` reports all three side by side so they cannot be conflated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

from .cdm import alternative_betas, rank_alternatives, scale_for
from .fusion import fuse_models, fuse_runs
from .stability import _min_margin, build_panels
from .twotuple import delta

SCHEME = "LABEL-5"


def _run_level(runs: pd.DataFrame) -> pd.DataFrame:
    """Per (model, review) spread of the raw runs, and whether they all agree."""

    rows = []
    for panel in build_panels(runs):
        spreads = np.array([a.std(ddof=0) for a in panel.runs])
        identical = np.array([len(set(a)) == 1 for a in panel.runs])
        rows.append(
            {
                "model": panel.model,
                "within_sd": round(float(spreads.mean()), 4),
                "identical_pct": round(100.0 * float(identical.mean()), 1),
                "reviews": len(panel.runs),
            }
        )
    return pd.DataFrame(rows)


def _decision_level(
    runs: pd.DataFrame,
    operator: str,
    hierarchy: str,
    fusion_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Review-level evaluations and the ranked alternatives, fusion included."""

    reviews = fuse_runs(runs, operator)
    members = sorted(reviews["model"].unique())
    fused = fuse_models(reviews, operator, members, fusion_label)
    both = pd.concat([reviews, fused], ignore_index=True)
    return both, rank_alternatives(alternative_betas(both, hierarchy))


def _terms_used(betas: np.ndarray, scheme: str) -> int:
    """How many distinct linguistic terms the four alternatives occupy.

    One means the term is carrying no part of the decision, which is the
    finding this collection was gathered to test for robustness.
    """

    scale = scale_for(scheme)
    return len({delta(float(b), scale).index for b in betas})


def contrast(
    sampled: pd.DataFrame,
    greedy: pd.DataFrame,
    operator: str = "two_tuple",
    hierarchy: str = "two_level",
    fusion_label: str = "Fusion (open-weight)",
) -> pd.DataFrame:
    """One row per system, the two temperatures side by side.

    `sampled` and `greedy` are runs tables restricted to the same scheme and
    the same systems.

    **The contrast is paired.** Each system is scored on the reviews that
    survived parsing in *both* collections, because rejected outputs do not
    fall in the same places at the two temperatures and the difference is not
    small: a deterministic model that answers outside the vocabulary answers
    outside it all ten times, so greedy decoding converts partial missingness
    into complete missingness and removes whole reviews from the corpus. Scored
    on its own reviews, Ministral 3 8B appears to go from a margin of 0.00014
    to 0.01196; scored on the reviews both collections share, it goes from
    0.00722 to 0.01196, and the missing eight reviews rather than the
    temperature accounted for most of the apparent change. `dropped` reports
    how many reviews the pairing cost each system.
    """

    shared: dict[str, set[str]] = {}
    for model in set(sampled["model"]) | set(greedy["model"]):
        shared[model] = set(sampled.loc[sampled["model"] == model, "review_id"]) & set(
            greedy.loc[greedy["model"] == model, "review_id"]
        )

    def pair(table: pd.DataFrame) -> pd.DataFrame:
        keep = [
            review_id in shared.get(model, ())
            for model, review_id in zip(table["model"], table["review_id"], strict=True)
        ]
        return table[keep]

    dropped = {
        model: len(set(sampled.loc[sampled["model"] == model, "review_id"])) - len(ids)
        for model, ids in shared.items()
    }
    sampled, greedy = pair(sampled), pair(greedy)

    spread = {
        "sampled": _run_level(sampled).set_index("model"),
        "greedy": _run_level(greedy).set_index("model"),
    }
    reviews, alternatives = {}, {}
    for name, table in (("sampled", sampled), ("greedy", greedy)):
        reviews[name], alternatives[name] = _decision_level(
            table, operator, hierarchy, fusion_label
        )

    scheme = str(sampled["scheme"].iloc[0])
    models = list(alternatives["sampled"]["model"].unique())

    rows = []
    for model in models:
        row: dict[str, object] = {"model": model, "dropped": dropped.get(model, 0)}
        for name in ("sampled", "greedy"):
            suffix = "t1" if name == "sampled" else "t0"
            g = alternatives[name]
            g = g[g["model"] == model]
            if g.empty:
                continue
            betas = g["beta"].to_numpy()
            row[f"margin_{suffix}"] = round(_min_margin(betas), 5)
            row[f"terms_{suffix}"] = _terms_used(betas, scheme)
            row[f"winner_{suffix}"] = g.iloc[0]["alternative"]
            row[f"order_{suffix}"] = " > ".join(g["alternative"])

            per_review = reviews[name]
            per_review = per_review[per_review["model"] == model]["beta"]
            row[f"distinct_{suffix}"] = len(set(np.round(per_review, 9)))
            row[f"reviews_{suffix}"] = len(per_review)

            if model in spread[name].index:
                record = spread[name].loc[model]
                row[f"within_sd_{suffix}"] = record["within_sd"]
                row[f"identical_pct_{suffix}"] = record["identical_pct"]

        ranks = {}
        for name, suffix in (("sampled", "t1"), ("greedy", "t0")):
            g = alternatives[name]
            ranks[suffix] = g[g["model"] == model].set_index("alternative")["rank"]
        order = list(ranks["t1"].index)
        if set(order) == set(ranks["t0"].index):
            tau = kendalltau(
                ranks["t1"].reindex(order), ranks["t0"].reindex(order)
            ).statistic
            row["tau_t1_vs_t0"] = round(float(tau), 4)
            row["same_winner"] = row.get("winner_t1") == row.get("winner_t0")
        rows.append(row)

    return pd.DataFrame(rows)


def review_correlation(
    sampled: pd.DataFrame,
    greedy: pd.DataFrame,
    operator: str = "two_tuple",
) -> pd.DataFrame:
    """Per-review agreement between the two collections, per system.

    The two temperatures are two measurements of the same review by the same
    model, so this says how much of the review-level evaluation the sampling
    temperature actually moves.
    """

    left = fuse_runs(sampled, operator).set_index(["model", "review_id"])["beta"]
    right = fuse_runs(greedy, operator).set_index(["model", "review_id"])["beta"]
    shared = left.index.intersection(right.index)

    rows = []
    for model in sorted({m for m, _ in shared}):
        keys = [k for k in shared if k[0] == model]
        a, b = left.loc[keys], right.loc[keys]
        rows.append(
            {
                "model": model,
                "n": len(keys),
                "pearson_r": round(float(pearsonr(a, b)[0]), 4),
                "mean_abs_diff": round(
                    float(np.abs(a.to_numpy() - b.to_numpy()).mean()), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def contrast_to_latex(
    table: pd.DataFrame, out_path: Path, model_order: list[str]
) -> None:
    """The temperature contrast, run level beside decision level."""

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Sampling temperature: $T = 1.0$ against greedy decoding, "
            r"under " + SCHEME + r". Same prompt, vocabulary, parser and reviews; "
            r"for Gemma 4 E4B the two collections also come from different "
            r"campaigns (Section~\ref{sec:protocol}). $\mathit{SD}_w$ is the mean "
            r"spread of the ten runs of a review and Ident.\ the share of "
            r"reviews whose ten runs agree; Distinct counts the different "
            r"review-level aggregates, Terms the linguistic terms the collective "
            r"evaluations occupy, and Min gap the smallest adjacent gap. Each "
            r"system is scored on the reviews that survived parsing at both "
            r"temperatures; Drop is how many that cost. Greedy decoding removes "
            r"the run-level spread, reduces resolution, and moves the margin in "
            r"both directions.}"
        ),
        r"\label{tab:cdm_temperature}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrr}",
        r"\toprule",
        (
            r" & \multicolumn{4}{c}{\textbf{Run level}}"
            r" & \multicolumn{2}{c}{\textbf{Resolution}}"
            r" & \multicolumn{4}{c}{\textbf{Decision}} & \\"
        ),
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-11}",
        (
            r"\textbf{Model} & $\mathit{SD}_w^{T1}$ & $\mathit{SD}_w^{T0}$"
            r" & Ident.$^{T1}$ & Ident.$^{T0}$"
            r" & Dist.$^{T1}$ & Dist.$^{T0}$"
            r" & Terms$^{T1}$ & Terms$^{T0}$"
            r" & Min gap$^{T1}$ & Min gap$^{T0}$ & Drop \\"
        ),
        r"\midrule",
    ]
    for model in model_order:
        row = table[table["model"] == model]
        if row.empty:
            continue
        r = row.iloc[0]
        sd_t1 = f"{r['within_sd_t1']:.3f}" if pd.notna(r.get("within_sd_t1")) else "--"
        sd_t0 = f"{r['within_sd_t0']:.3f}" if pd.notna(r.get("within_sd_t0")) else "--"
        id_t1 = (
            f"{r['identical_pct_t1']:.0f}\\%"
            if pd.notna(r.get("identical_pct_t1"))
            else "--"
        )
        id_t0 = (
            f"{r['identical_pct_t0']:.0f}\\%"
            if pd.notna(r.get("identical_pct_t0"))
            else "--"
        )
        lines.append(
            f"  {model} & {sd_t1} & {sd_t0} & {id_t1} & {id_t0}"
            f" & {int(r['distinct_t1'])} & {int(r['distinct_t0'])}"
            f" & {int(r['terms_t1'])} & {int(r['terms_t0'])}"
            f" & {r['margin_t1']:.4f} & {r['margin_t0']:.4f}"
            f" & {int(r['dropped'])}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    import argparse

    from .cdm import attach_entities
    from .profiles import load_profile
    from .runs import load_runs

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("profile", help="london or madrid")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    greedy_profile = load_profile(f"{profile.name}_t0")

    def load(p) -> pd.DataFrame:
        table = load_runs(p)
        return attach_entities(table[table["scheme"] == SCHEME], profile.data_csv)

    # The hosted reference has no greedy collection: the contrast is between
    # the local systems only.
    shared = profile.members
    sampled = load(profile)
    sampled = sampled[sampled["model"].isin(shared)]
    greedy = load(greedy_profile)
    greedy = greedy[greedy["model"].isin(shared)]

    table = contrast(sampled, greedy)
    correlation = review_correlation(sampled, greedy)
    print(table.drop(columns=["order_t1", "order_t0"]).to_string(index=False))
    print(correlation.to_string(index=False))

    out = greedy_profile.out_dir
    table.to_csv(out / "cdm_temperature.csv", index=False)
    correlation.to_csv(out / "cdm_temperature_reviews.csv", index=False)
    tables = profile.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    contrast_to_latex(
        table, tables / "cdm_temperature.tex", [*sorted(shared), "Fusion (open-weight)"]
    )


if __name__ == "__main__":
    main()
