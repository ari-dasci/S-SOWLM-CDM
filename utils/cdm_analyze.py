"""The decision analysis of one corpus: every CSV, table and figure but the
temperature contrast (`utils.temperature`) and the paired comparisons
(`utils.paired`).

    python -m utils.cdm_analyze london
    python -m utils.cdm_analyze madrid

Reads the profile's `runs_*.csv` and writes into its `out_dir`:

    cdm_alternatives.csv            collective evaluation and rank of each
                                    alternative, per system, scheme, hierarchy
    cdm_stability.csv               ranking stability under run resampling, per
                                    aggregation operator
    cdm_crowd_stability.csv         the same under participant resampling
    cdm_joint.csv, _intervals.csv   run, participant and joint resampling by
                                    one procedure
    cdm_candidates.csv, cdm_rival_gaps.csv, cdm_simultaneous_*.csv
                                    candidate sets and simultaneous intervals
    cdm_discriminance.csv           between-review spread against the noise
                                    each operator leaves
    cdm_operator_scope.csv          rounding at the review stages or everywhere
    cdm_mapping.csv, cdm_missing_data.csv, cdm_rejection_bias.csv,
    cdm_tiebreak.csv                sensitivity to term placement, rejected
                                    outputs and the majority tie rule
    cdm_star_*.csv                  agreement with the reviewers' star ratings
    cdm_budget.csv, cdm_budget_calls.csv, cdm_fusion_vs_best.csv
                                    inference budgets and fusion against members
    cdm_validity.csv                replies collected, accepted and rejected

with the LaTeX rendering of each in `tables/` and the figures in `figures/`.

The fused system is built from the local models only and appended as an extra
`model`, so it flows through the same code as the individual systems. The
reviewers' star ratings are scored the same way, as one more system
(`Stars (reference)`), in the candidate-set tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .budget import budget_curve, calls_comparison
from .cdm import (
    alternative_betas,
    attach_entities,
    rank_alternatives,
    review_position,
    scale_for,
)
from .fusion import LOSSLESS, SNAPPING, fuse_models, fuse_runs
from .parsing import SCHEMES
from .plot_cdm import (
    plot_alternative_profile,
    plot_decision_stability,
    plot_joint_intervals,
    plot_resolution,
)
from .profiles import load_profile
from .runs import load_runs
from .stability import (
    DEFAULT_PANEL_REPLICAS,
    DEFAULT_REPLICAS,
    DEFAULT_SEED,
    candidate_sets,
    crowd_stability,
    decision_stability,
    discriminance,
    draw_crowd,
    joint_stability,
    mapping_sensitivity,
    missing_data_sensitivity,
    operator_scope_contrast,
    rejection_bias,
    rival_gaps,
    simultaneous_candidates,
    tiebreak_sensitivity,
)
from .twotuple import delta, format_tuple
from .validity import (
    ranking_agreement,
    review_agreement,
    star_alternatives,
    star_frame,
)

SCHEME = "LABEL-5"
FUSION_LABEL = "Fusion (open-weight)"
STARS_LABEL = "Stars (reference)"


def _tt(operator: str) -> str:
    """Operator name as a LaTeX \\texttt, underscores escaped."""

    return r"\texttt{" + operator.replace("_", r"\_") + "}"


def _ci(lo: float, hi: float) -> str:
    """A percentage interval, small and in brackets under the point estimate."""

    return rf"\tiny[{lo:.1f}, {hi:.1f}]"


HIERARCHIES = ("two_level", "three_level")
OPERATORS_REPORTED = (*LOSSLESS, *SNAPPING)

# Long restaurant names blow out every table; the dataset's own short forms.
ALTERNATIVE_SHORT = {
    "Oxo Tower Restaurant, Bar and Brasserie": "Oxo Tower",
}


def short(alternative: str) -> str:
    """Table form of an alternative's name, LaTeX-escaped.

    Restaurant names carry `&` (``Zenith Brunch & Cocktails''), which would
    otherwise end a table cell.
    """

    name = ALTERNATIVE_SHORT.get(alternative, alternative).replace(" - ", "--")
    for char in ("&", "%", "#", "_", "$"):
        name = name.replace(char, "\\" + char)
    return name


def build_alternatives(
    reviews: pd.DataFrame, fused: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Ranked alternatives for both hierarchies, individual systems and fusion."""

    level = reviews if fused is None else pd.concat([reviews, fused], ignore_index=True)
    return pd.concat(
        [rank_alternatives(alternative_betas(level, h)) for h in HIERARCHIES],
        ignore_index=True,
    )


def alternatives_to_latex(
    alternatives: pd.DataFrame,
    out_path: Path,
    scheme: str,
    hierarchy: str,
    model_order: list[str],
) -> None:
    """Collective 2-tuple evaluation per alternative, one column per system."""

    sub = alternatives[
        (alternatives["scheme"] == scheme) & (alternatives["hierarchy"] == hierarchy)
    ]
    models = [m for m in model_order if m in set(sub["model"])]
    names = list(dict.fromkeys(sub["alternative"]))

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Collective evaluation of each alternative under "
            f"{scheme}, aggregated with the 2-tuple operator over the "
            f"{hierarchy.replace('_', '-')} hierarchy. Each cell is "
            r"$(s_i, \alpha)$; the rank within each column is given in "
            r"brackets. The fused system aggregates the open-weight models "
            r"only.}"
        ),
        r"\label{tab:cdm_alternatives_"
        + scheme.lower().replace("-", "")
        + f"_{hierarchy}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(models) + "}",
        r"\toprule",
        r"\textbf{Alternative} & "
        + " & ".join(rf"\textbf{{{m}}}" for m in models)
        + r" \\",
        r"\midrule",
    ]
    for name in names:
        cells = []
        for model in models:
            row = sub[(sub["model"] == model) & (sub["alternative"] == name)]
            if row.empty:
                cells.append("--")
                continue
            record = row.iloc[0]
            tuple_ = delta(record["beta"], scale_for(scheme))
            cells.append(
                f"{format_tuple(tuple_, scale_for(scheme))} [{int(record['rank'])}]"
            )
        lines.append(f"  {short(name)} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def stability_to_latex(
    stability: pd.DataFrame, out_path: Path, model_order: list[str]
) -> None:
    """Decision stability per operator: the answer to 'stable at what?'."""

    replicas = int(stability["replicas"].iloc[0]) if not stability.empty else 0
    # table*, not table: six columns do not fit a two-column journal measure,
    # and a single-column float here overflows into the neighbouring column.
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Stability of the collective ranking under resampling of "
            r"each review's repeated runs, with the crowd held fixed: this is "
            r"the decision's exposure to the models' own inference "
            r"variability, and says nothing about its exposure to who happened "
            r"to write the reviews (Table~\ref{tab:cdm_crowd}). $\tau$ is the "
            r"mean Kendall correlation between a replica's ranking and the "
            r"full-data ranking; Exact is the percentage of replicas "
            r"reproducing that ranking outright; Top-1 the percentage in which "
            r"no alternative strictly beats Winner, the alternative ranked first "
            r"on the full data (a tie for first keeps it); Top the $\beta$ lead "
            r"of Winner over the runner-up and Min "
            r"the smallest $\beta$ gap between any two adjacently ranked "
            r"alternatives, both on the full data. "
            r"Brackets give the 95\% Wilson interval over "
            + f"{replicas:,}".replace(",", "{,}")
            + r" replicas; an interval printed as [100.0, 100.0] has its lower "
            r"limit above 99.95\%. Reproducibility under a resampling and the "
            r"size of the margin are different properties and are reported "
            r"separately.}"
        ),
        r"\label{tab:cdm_stability}",
        r"\small",
        r"\begin{tabular}{llccclrr}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{Operator} & \textbf{$\tau$}"
            r" & \textbf{Exact} & \textbf{Top-1} & \textbf{Winner}"
            r" & \textbf{Lead} & \textbf{Min gap} \\"
        ),
        r"\midrule",
    ]
    rendered = False
    for model in model_order:
        rows = stability[stability["model"] == model]
        if rows.empty:
            continue
        if rendered:
            lines.append(r"\midrule")
        rendered = True
        for j, operator in enumerate(OPERATORS_REPORTED):
            row = rows[rows["operator"] == operator]
            if row.empty:
                continue
            record = row.iloc[0]
            prefix = (
                rf"\multirow{{{len(OPERATORS_REPORTED)}}}{{*}}{{{model}}}"
                if j == 0
                else ""
            )
            lines.append(
                f"  {prefix} & {_tt(operator)}"
                f" & {record['tau_mean']:.3f}"
                f" & {record['exact_pct']:.1f}\\%"
                f" {_ci(record['exact_lo'], record['exact_hi'])}"
                f" & {record['top1_pct']:.1f}\\%"
                f" {_ci(record['top1_lo'], record['top1_hi'])}"
                f" & {short(record['reference_winner'])}"
                f" & {record['margin_top']:.4f}"
                f" & {record['margin_min']:.4f}" + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def joint_to_latex(
    joint: pd.DataFrame, out_path: Path, scheme: str, model_order: list[str]
) -> None:
    """Runs, crowd, both: where the ranking's uncertainty comes from."""

    sub = joint[joint["scheme"] == scheme]
    replicas = int(sub["replicas"].iloc[0]) if not sub.empty else 0
    conditions = {"runs": "runs", "experts": "crowd", "both": "both"}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Stability of the collective ranking ("
            + scheme
            + r", 2-tuple operator, two-level hierarchy) under three "
            r"resamplings computed by one procedure: \emph{runs} redraws each "
            r"review's repeated runs with the crowd held fixed; \emph{crowd} "
            r"redraws the experts as clusters, each bringing all of their "
            r"reviews, with the runs held fixed; \emph{both} does the two in "
            r"the same replica, on the same redrawn crowds. For the fused "
            r"system each member's runs are redrawn independently. Top-1 is "
            r"the share of replicas in which no alternative strictly beats the "
            r"system's full-data winner; Exact the share reproducing its "
            r"full-data ranking. Brackets are 95\% Wilson intervals, one "
            r"decimal (a printed 100.0 lower limit is above 99.95\%), over "
            + f"{replicas:,}".replace(",", "{,}")
            + r" replicas. Which rivals the crowd does not separate from the "
            r"winner is in Table~\ref{tab:cdm_candidates}.}"
        ),
        r"\label{tab:cdm_joint}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Resampled} & \textbf{Top-1} & \textbf{Exact} \\",
        r"\midrule",
    ]
    rendered = False
    for model in model_order:
        rows = sub[sub["model"] == model]
        if rows.empty:
            continue
        if rendered:
            lines.append(r"\midrule")
        rendered = True
        for j, (condition, label) in enumerate(conditions.items()):
            row = rows[rows["condition"] == condition]
            if row.empty:
                continue
            r = row.iloc[0]
            prefix = rf"\multirow{{3}}{{*}}{{{model}}}" if j == 0 else ""
            lines.append(
                f"  {prefix} & {label}"
                f" & {r['top1_pct']:.1f}\\% {_ci(r['top1_lo'], r['top1_hi'])}"
                f" & {r['exact_pct']:.1f}\\% {_ci(r['exact_lo'], r['exact_hi'])}"
                r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


def _c_range(sub: pd.DataFrame, mode: str) -> str:
    """The max-t critical values of one crowd mode, as `2.56` or `2.56 to 2.57`."""

    c = sub[sub["crowd_mode"] == mode]["c_simult"]
    lo, hi = f"{c.min():.2f}", f"{c.max():.2f}"
    return lo if lo == hi else f"{lo} to {hi}"


def candidates_to_latex(
    candidates: pd.DataFrame, out_path: Path, scheme: str, model_order: list[str]
) -> None:
    """The reported candidate sets (simultaneous) beside the pairwise rule."""

    sub = candidates[candidates["scheme"] == scheme]
    replicas = int(sub["replicas"].iloc[0]) if not sub.empty else 0
    many = int(sub["rivals"].max()) + 1 > 5 if not sub.empty else False
    modes = list(dict.fromkeys(sub["crowd_mode"]))
    by_mode = len(modes) > 1
    mode_label = {"experts": "participants", "stratified": "strat."}
    c_text = (
        "; ".join(f"{_c_range(sub, m)} under \\emph{{{mode_label[m]}}}" for m in modes)
        if by_mode
        else _c_range(sub, modes[0])
        if modes
        else ""
    )
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Candidate sets under resampling of the crowd ("
            + scheme
            + r", 2-tuple operator, two-level hierarchy). Every difference "
            r"between two alternatives of a system is recomputed on "
            + f"{replicas:,}".replace(",", "{,}")
            + r" redrawn crowds, the same crowds for every system. Set "
            r"(simult.)\ is the reported candidate set, a confidence set for the "
            r"best alternative: the alternatives no rival is simultaneously "
            r"better than, from bootstrap max-$t$ intervals over all pairwise "
            r"differences of the system (critical value $c$ = "
            + c_text
            + r"); Not excluded lists its members other than the winner. "
            r"Coverage is simultaneous over all pairwise differences of a "
            r"system under this resampling design, not with respect to a "
            r"population of platform users. Set (pairw.)\ is the pairwise "
            r"sensitivity statement: the winner and every rival whose gap to "
            r"it has its 2.5th percentile at or below zero, one pair at a time. "
            r"Neither set shows its members to be equivalent to each other. "
            r"Lead is the gap to the runner-up with its 2.5th and 97.5th "
            r"percentiles; Adjacent pairs separated applies the pairwise test "
            r"to pairs consecutive in the full-data ranking and lists the ranks "
            r"of those it separates."
            + (
                r" Alternatives are given by their rank in the system's own "
                r"full-data ranking."
                if many
                else ""
            )
            + (
                r" \emph{participants} redraws participants across the corpus; "
                r"\emph{strat.}\ redraws them within each alternative, keeping "
                r"its number of reviewers."
                if by_mode
                else ""
            )
            + r"}"
        ),
        r"\label{tab:cdm_candidates}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + ("l" if by_mode else "") + "lccclcl}",
        r"\toprule",
        (
            r"\textbf{System}"
            + (r" & \textbf{Crowd}" if by_mode else "")
            + r" & \textbf{Winner} & \textbf{Top-1} & \textbf{Set (pairw.)}"
            r" & \textbf{Set (simult.)} & \textbf{Not excluded (simult.)}"
            r" & \textbf{Lead [95\%]} & \textbf{Adjacent pairs separated} \\"
        ),
        r"\midrule",
    ]
    for model in model_order:
        rows = sub[sub["model"] == model]
        for mode in modes:
            row = rows[rows["crowd_mode"] == mode]
            if row.empty:
                continue
            r = row.iloc[0]
            total = 1 + int(r["rivals"])
            if many:
                ranks = str(r["candidate_ranks_simult"]).split(", ")[1:]
                kept = ", ".join(ranks) if ranks else "--"
            else:
                names = str(r["candidates_simult"]).split("; ")[1:]
                kept = ", ".join(short(c) for c in names) if names else "--"
            tokens = str(r["partial_order_ranks"]).split()
            separated = [
                f"{tokens[i - 1]}--{tokens[i + 1]}"
                for i in range(1, len(tokens), 2)
                if tokens[i] == ">"
            ]
            adjacent = (
                f"{int(r['adjacent_distinguished'])} of {int(r['adjacent_pairs'])}"
                + (f": {', '.join(separated)}" if separated else "")
            )
            lines.append(
                f"  {model if mode == modes[0] else ''}"
                + (f" & {mode_label.get(mode, mode)}" if by_mode else "")
                + f" & {short(r['winner'])}"
                f" & {r['top1_pct']:.1f}\\%"
                f" & {1 + int(r['rivals_kept'])} of {total}"
                f" & {int(r['set_size_simult'])} of {total}"
                f" & {kept}"
                f" & {r['gap_runner_up']:+.4f} [{r['gap_runner_up_lo']:+.4f},"
                f" {r['gap_runner_up_hi']:+.4f}]"
                f" & {adjacent}" + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def simultaneous_pairs_to_latex(
    pairs: pd.DataFrame,
    sets: pd.DataFrame,
    out_path: Path,
    scheme: str,
    model_order: list[str],
) -> None:
    """Every pairwise difference with its simultaneous interval, per system."""

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[
            (frame["scheme"] == scheme)
            & (frame["crowd_mode"] == "experts")
            & (frame["method"] == "max_t")
        ]

    sub = keep(pairs)
    c_of = keep(sets).set_index("model")["c"]
    alternatives = len(set(sub["i"]) | set(sub["j"]))
    many = alternatives > 5
    # Twelve alternatives give 66 pairs per system: only the pairs whose
    # interval excludes zero are listed, with the count of the rest.
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Simultaneous 95\% intervals for the pairwise "
            r"differences $\beta_i-\beta_j$ under resampling of the crowd "
            r"(\emph{participants}; " + scheme + r", 2-tuple operator, two-level "
            r"hierarchy), $i$ ranked above $j$ in the system's full data. "
            r"$\hat d$ is the full-data difference, se its standard deviation "
            r"over the redrawn crowds, and the interval $\hat d \mp c\,$se with "
            r"$c$ the system's max-$t$ critical value. "
            + (
                r"Alternatives are given by their full-data rank. Only the "
                r"pairs whose interval lies above zero are listed; the "
                r"remaining pairs of each system have intervals containing "
                r"zero.}"
                if many
                else r"A pair whose interval lies above zero has $i$ "
                r"simultaneously better than $j$.}"
            )
        ),
        r"\label{tab:cdm_simultaneous_pairs}",
        r"\small",
    ]
    if many:
        lines += [
            r"\begin{tabular}{lrrl}",
            r"\toprule",
            (
                r"\textbf{System} & \textbf{$c$} & \textbf{Pairs above zero}"
                r" & \textbf{$(i, j)$: $\hat d$ [interval]} \\"
            ),
            r"\midrule",
        ]
        for model in model_order:
            g = sub[sub["model"] == model]
            # The stars' intervals are in the CSV; this table lists systems.
            if g.empty or model == STARS_LABEL:
                continue
            above = g[g["excluded_by"]]
            listed = "; ".join(
                f"({r.i_rank},~{r.j_rank}): {r.gap_full:.3f} "
                f"[{r.lo_simult:.3f}, {r.hi_simult:.3f}]"
                for r in above.itertuples()
            )
            lines.append(
                f"  {model} & {c_of[model]:.2f} & {len(above)} of {len(g)}"
                f" & \\parbox[t]{{0.56\\textwidth}}{{{listed or '--'}}}" + r" \\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    else:
        lines += [
            r"\begin{tabular}{lllrrr}",
            r"\toprule",
            (
                r"\textbf{System} & \textbf{$i$} & \textbf{$j$} & \textbf{$\hat d$}"
                r" & \textbf{se} & \textbf{Interval} \\"
            ),
            r"\midrule",
        ]
        for model in model_order:
            g = sub[sub["model"] == model]
            for k, r in enumerate(g.itertuples()):
                lines.append(
                    f"  {model if k == 0 else ''} & {short(r.i)} & {short(r.j)}"
                    f" & {r.gap_full:+.4f} & {r.se:.4f}"
                    f" & [{r.lo_simult:+.3f}, {r.hi_simult:+.3f}]" + r" \\"
                )
            if not g.empty and model != model_order[-1]:
                lines.append(r"\addlinespace")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


RUNS_PER_REVIEW = 10


def fusion_vs_best(
    joint: pd.DataFrame,
    alternatives: pd.DataFrame,
    agreement: pd.DataFrame,
    latency: pd.DataFrame | None,
    scheme: str,
    members: list[str],
    fusion_label: str,
    candidates: pd.DataFrame,
    runs_per_review: int = RUNS_PER_REVIEW,
) -> pd.DataFrame:
    """Each system against the fusion on the figures a decision rests on.

    Not on movement: the comparison is the ranking's margins, how often run
    and crowd resampling keep its winner, how it tracks the authors' stars,
    and what one review costs. Cost is the median request latency times the
    runs per review; the fused system costs the sum of its members.
    """

    two = alternatives[
        (alternatives["scheme"] == scheme) & (alternatives["hierarchy"] == "two_level")
    ]
    median = (
        latency.set_index("model")["median_s"].dropna()
        if latency is not None
        else pd.Series(dtype=float)
    )
    cand = candidates.set_index(["model", "scheme", "crowd_mode"])
    rows = []
    for model in joint["model"].drop_duplicates():
        betas = two[two["model"] == model]["beta"].sort_values(ascending=False)
        gaps = -np.diff(betas.to_numpy())
        j = joint[(joint["model"] == model) & (joint["scheme"] == scheme)].set_index(
            "condition"
        )
        star = agreement[
            (agreement["model"] == model) & (agreement["scheme"] == scheme)
        ]
        if model == fusion_label:
            cost = (
                float(sum(median[m] for m in members)) * runs_per_review
                if all(m in median for m in members)
                else np.nan
            )
        else:
            cost = float(median[model]) * runs_per_review if model in median else np.nan
        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "winner": j.loc["runs", "reference_winner"],
                "margin_top": round(float(gaps[0]), 5),
                "margin_min": round(float(gaps.min()), 5),
                "top1_runs": j.loc["runs", "top1_pct"],
                "exact_runs": j.loc["runs", "exact_pct"],
                "top1_crowd": j.loc["experts", "top1_pct"],
                "inverts_crowd": j.loc["experts", "inverts_pct"],
                # The reported candidate set is the simultaneous one; the
                # pairwise size is kept beside it as the sensitivity statement.
                "candidates_crowd": int(
                    cand.loc[(model, scheme, "experts"), "set_size_simult"]
                ),
                "candidates_crowd_pairwise": int(
                    1 + cand.loc[(model, scheme, "experts"), "rivals_kept"]
                ),
                "r_stars": float(star["pearson_r"].iloc[0]) if len(star) else np.nan,
                "cost_s_per_review": round(cost, 2) if np.isfinite(cost) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fusion_vs_best_to_latex(
    table: pd.DataFrame, out_path: Path, model_order: list[str]
) -> None:
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{The fused system against each of its members and the "
            r"proprietary reference, on the quantities a decision rests on "
            r"(LABEL-5, 2-tuple operator, two-level hierarchy). Top and Min are "
            r"the $\beta$ lead of the winner over the runner-up and the smallest "
            r"adjacent gap. Top-1 and Exact under runs resample each review's "
            r"runs with the crowd fixed; Top-1 and Cand.\ under crowd resample "
            r"the experts with the runs fixed: Cand.\ is the size of the "
            r"simultaneous candidate set, the alternatives no rival is "
            r"simultaneously better than (Table~\ref{tab:cdm_candidates}). $r$ "
            r"is the review-level Pearson correlation with the authors' star "
            r"ratings. Cost is the median request latency times ten runs, in "
            r"seconds per review; the fusion's is the sum over its members. "
            r"The hosted reference has no comparable latency.}"
        ),
        r"\label{tab:cdm_fusion_vs_best}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        (
            r" & & & & \multicolumn{2}{c}{\textbf{Runs}} & "
            r"\multicolumn{2}{c}{\textbf{Crowd}} & & \\"
        ),
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        (
            r"\textbf{System} & \textbf{Winner} & \textbf{Lead} & \textbf{Min gap}"
            r" & \textbf{Top-1} & \textbf{Exact} & \textbf{Top-1} & \textbf{Cand.}"
            r" & \textbf{$r$} & \textbf{Cost (s)} \\"
        ),
        r"\midrule",
    ]
    for model in model_order:
        row = table[table["model"] == model]
        if row.empty:
            continue
        r = row.iloc[0]
        if model.startswith("Fusion"):
            lines.append(r"\midrule")
        cost = r["cost_s_per_review"]
        cost = "--" if pd.isna(cost) else f"{cost:.2f}"
        lines.append(
            f"  {model} & {short(r['winner'])}"
            f" & {r['margin_top']:.4f} & {r['margin_min']:.4f}"
            f" & {r['top1_runs']:.1f}\\% & {r['exact_runs']:.1f}\\%"
            f" & {r['top1_crowd']:.1f}\\% & {r['candidates_crowd']:d}"
            f" & {r['r_stars']:.3f} & {cost}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def build_budget(
    scheme_runs: pd.DataFrame,
    alternatives: pd.DataFrame,
    members: list[str],
    latency: pd.DataFrame,
    scheme: str,
    greedy: pd.DataFrame,
    data_csv: Path,
    seed: int,
    draws=None,
) -> pd.DataFrame:
    """The budget table's systems: each local model, their fusion, and the
    same at T = 0, one greedy execution at a time."""

    median = latency.set_index("model")["median_s"].dropna().to_dict()
    two = alternatives[
        (alternatives["scheme"] == scheme) & (alternatives["hierarchy"] == "two_level")
    ]
    winner = {
        m: g.sort_values("rank")["alternative"].iloc[0] for m, g in two.groupby("model")
    }
    systems = {m: scheme_runs[scheme_runs["model"] == m] for m in members}
    systems[FUSION_LABEL] = scheme_runs[scheme_runs["model"].isin(members)]
    winners = {k: winner[k] for k in systems}
    costs = {k: [k] if k != FUSION_LABEL else members for k in systems}
    sampled = budget_curve(systems, winners, median, costs, seed=seed, draws=draws)
    greedy = attach_entities(greedy, data_csv)
    greedy = greedy[(greedy["scheme"] == scheme) & greedy["model"].isin(members)]
    present = [m for m in members if m in set(greedy["model"])]
    g_systems = {f"{m} (T=0)": greedy[greedy["model"] == m] for m in present}
    g_systems[f"{FUSION_LABEL} (T=0)"] = greedy
    g_winners = {k: winner[k.replace(" (T=0)", "")] for k in g_systems}
    g_costs = {
        k: [k.replace(" (T=0)", "")] if not k.startswith(FUSION_LABEL) else present
        for k in g_systems
    }
    # The greedy rows are scored on the same reviews and crowd replicas as the
    # sampled ones: budget_curve takes its review list from the first system.
    first = {next(iter(systems)): systems[next(iter(systems))]}
    greedy_rows = budget_curve(
        {**first, **g_systems},
        {**winners, **g_winners},
        median,
        {**costs, **g_costs},
        budgets=(1,),
        seed=seed,
        draws=draws,
    )
    greedy_rows = greedy_rows[greedy_rows["system"].str.contains("T=0", regex=False)]
    out = []
    for system in [*systems]:
        out.append(sampled[sampled["system"] == system])
        out.append(greedy_rows[greedy_rows["system"] == f"{system} (T=0)"])
    return pd.concat(out, ignore_index=True)


def tiebreak_to_latex(
    ties: pd.DataFrame, out_path: Path, model_order: list[str]
) -> None:
    """The majority operator under both tie rules."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{The tie rule of the \texttt{majority} operator (LABEL-5, "
            r"two-level hierarchy). The reported rule breaks a tie between "
            r"modal terms towards the lower term; the alternative breaks it "
            r"towards the upper one. Changed is the number of reviews whose "
            r"review-level value differs between the rules (for the fusion, "
            r"whose mode is taken over its members' review values). Min is the "
            r"smallest adjacent gap of the collective ranking under each rule.}"
        ),
        r"\label{tab:cdm_tiebreak}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrllcrr}",
        r"\toprule",
        (
            r"\textbf{System} & \textbf{Changed} & \textbf{Winner (lower)}"
            r" & \textbf{Winner (upper)} & \textbf{Same order}"
            r" & \textbf{Min gap (lower)} & \textbf{Min gap (upper)} \\"
        ),
        r"\midrule",
    ]
    for model in model_order:
        row = ties[ties["model"] == model]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"  {model} & {int(r['reviews_changed'])}"
            f" & {short(r['winner_lower'])} & {short(r['winner_upper'])}"
            f" & {'yes' if r['same_order'] else 'no'}"
            f" & {r['margin_lower']:.4f} & {r['margin_upper']:.4f}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


def budget_calls_to_latex(calls: pd.DataFrame, out_path: Path) -> None:
    """At each number of model calls per review: best single model and fusion."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{The budget counted in model calls per review (LABEL-5, "
            r"2-tuple operator, two-level hierarchy, $T = 1.0$): one model with "
            r"$k$ runs makes $k$ calls, the five-member fusion $5k$, its "
            r"$k$-subsets applied by rounds. Single is the single model with the "
            r"highest crowd Top-1 at that count, chosen with hindsight; Top-1 is "
            r"the median over $k$-subsets of the share of redrawn crowds in "
            r"which no alternative strictly beats the subset's winner. Seconds "
            r"are estimated from per-request medians. Full curve: "
            r"Table~\ref{tab:cdm_budget}.}"
        ),
        r"\label{tab:cdm_budget_calls}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{rlrrrr}",
        r"\toprule",
        (
            r" & \multicolumn{3}{c}{\textbf{Best single model}}"
            r" & \multicolumn{2}{c}{\textbf{Fusion}} \\"
        ),
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        (
            r"\textbf{Calls} & \textbf{System} & \textbf{Top-1} & \textbf{s}"
            r" & \textbf{Top-1} & \textbf{s} \\"
        ),
        r"\midrule",
    ]
    for r in calls.itertuples():
        single = isinstance(getattr(r, "single_system", None), str)
        fused = pd.notna(getattr(r, "fusion_top1", np.nan))
        lines.append(
            f"  {r.calls_per_review}"
            + (
                f" & {r.single_system} & {r.single_top1:.1f}\\% & {r.single_cost_s:.2f}"
                if single
                else " & -- & -- & --"
            )
            + (
                f" & {r.fusion_top1:.1f}\\% & {r.fusion_cost_s:.2f}"
                if fused
                else " & -- & --"
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


def budget_to_latex(budget: pd.DataFrame, out_path: Path) -> None:
    replicas = int(budget["replicas"].iloc[0]) if not budget.empty else 0
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Inference budget at common cost (LABEL-5, 2-tuple "
            r"operator, two-level hierarchy). A budget of $k$ runs per model is "
            r"emulated by every $k$-subset of the ten collected runs (Subsets), "
            r"the same run indices for every review; figures are the median "
            r"over subsets with the 5th and 95th percentiles in brackets. Top-1 "
            r"is the share of crowd replicas (experts redrawn as clusters, "
            + f"{replicas:,}".replace(",", "{,}")
            + r" replicas shared by every row) in which no alternative strictly "
            r"beats the subset's own winner; "
            r"Top is that winner's $\beta$ lead over the runner-up; Same "
            r"winner is the share of subsets whose winner is the one the "
            r"system reports at $T = 1.0$ with ten runs. Calls is the number of "
            r"model calls per review, $k$ for one model and $m k$ for a fusion "
            r"of $m$ members, whose $k$-subset is applied by rounds (the same "
            r"run positions for every member). Cost is estimated from "
            r"per-request medians: median request latency times $k$, summed "
            r"over members for the fusion. Greedy rows ($T = 0$) use one run "
            r"and the sampled collection's latency.}"
        ),
        r"\label{tab:cdm_budget}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrllr}",
        r"\toprule",
        (
            r"\textbf{System} & \textbf{$k$} & \textbf{Subsets}"
            r" & \textbf{Calls} & \textbf{Cost (s)} & \textbf{Top-1} & \textbf{Lead}"
            r" & \textbf{Same winner} \\"
        ),
        r"\midrule",
    ]
    previous = None
    for r in budget.itertuples():
        base = r.system.replace(" (T=0)", "")
        if previous is not None and base != previous:
            lines.append(r"\midrule")
        previous = base
        lines.append(
            f"  {r.system} & {r.k} & {r.subsets} & {r.calls_per_review}"
            f" & {r.cost_s_per_review:.2f}"
            f" & {r.top1_crowd:.1f}\\% \\tiny[{r.top1_crowd_p5:.1f}, "
            f"{r.top1_crowd_p95:.1f}]"
            f" & {r.margin_top:.4f} \\tiny[{r.margin_top_p5:.4f}, "
            f"{r.margin_top_p95:.4f}]"
            f" & {r.winner_ok_pct:.0f}\\%" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def crowd_to_latex(crowd: pd.DataFrame, out_path: Path, model_order: list[str]) -> None:
    """Stability of the ranking when the crowd, not the runs, is resampled."""

    replicas = int(crowd["replicas"].iloc[0]) if not crowd.empty else 0
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Stability of the collective ranking under resampling of "
            r"the \emph{experts}, with each expert carrying all of their "
            r"reviews and the model outputs held fixed. Reviews by one expert "
            r"are not independent observations of the crowd, so they are drawn "
            r"as a cluster. This is the uncertainty a decision maker actually "
            r"faces (a different sample of diners might have ranked the "
            r"restaurants differently), and it is a different quantity from "
            r"Table~\ref{tab:cdm_stability}, not a robustness check on it. "
            r"Every row is scored on the same redrawn crowds, those of the "
            r"crowd rows of Table~\ref{tab:cdm_joint}; Top-1 and Exact are "
            r"defined as in Table~\ref{tab:cdm_stability}. "
            r"Min gap gives the full-data value with the 2.5th and 97.5th "
            r"percentiles across "
            + f"{replicas:,}".replace(",", "{,}")
            + r" replicas, measured between alternatives consecutive in the "
            r"full-data ranking so that it goes negative when a replica "
            r"inverts a pair; Inv.\ is the share of replicas in which it does. "
            r"An interval reaching below zero means the surveyed crowd does not "
            r"determine the complete ranking; which rivals it leaves "
            r"undistinguished from the winner is in "
            r"Table~\ref{tab:cdm_candidates}.}"
        ),
        r"\label{tab:cdm_crowd}",
        r"\small",
        r"\begin{tabular}{llccclr}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{Operator} & \textbf{$\tau$}"
            r" & \textbf{Exact} & \textbf{Top-1} & \textbf{Min gap [95\%]}"
            r" & \textbf{Inv.} \\"
        ),
        r"\midrule",
    ]
    rendered = False
    for model in model_order:
        rows = crowd[crowd["model"] == model]
        if rows.empty:
            continue
        if rendered:
            lines.append(r"\midrule")
        rendered = True
        for j, operator in enumerate(OPERATORS_REPORTED):
            row = rows[rows["operator"] == operator]
            if row.empty:
                continue
            record = row.iloc[0]
            prefix = (
                rf"\multirow{{{len(OPERATORS_REPORTED)}}}{{*}}{{{model}}}"
                if j == 0
                else ""
            )
            lines.append(
                f"  {prefix} & {_tt(operator)}"
                f" & {record['tau_mean']:.3f}"
                f" & {record['exact_pct']:.1f}\\%"
                f" {_ci(record['exact_lo'], record['exact_hi'])}"
                f" & {record['top1_pct']:.1f}\\%"
                f" {_ci(record['top1_lo'], record['top1_hi'])}"
                f" & {record['margin_min']:.4f} "
                f"[{record['margin_lo']:.4f}, {record['margin_hi']:.4f}]"
                f" & {record['inverts_pct']:.0f}\\%" + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def scope_to_latex(
    scope: pd.DataFrame, out_path: Path, scheme: str, model_order: list[str]
) -> None:
    """What changes when the operator acts at every level, not just the first."""

    sub = scope[scope["scheme"] == scheme]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Where the operator is allowed to act, under "
            + scheme
            + r". \texttt{review} is the reported pipeline: the operator "
            r"aggregates the ten runs of a review, and every level above "
            r"carries $\beta$ forward as a continuous value. \texttt{all} "
            r"applies the same operator again at review~$\to$~expert and "
            r"expert~$\to$~alternative, which is what a study aggregating "
            r"``by majority vote'' throughout would be doing. Distinct is how "
            r"many different collective values the four alternatives receive; "
            r"an operator acting at every level cannot express a difference "
            r"smaller than one term, so alternatives come back tied and the "
            r"ranking stops existing.}"
        ),
        r"\label{tab:cdm_scope}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llccr}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{Operator} & \textbf{Scope}"
            r" & \textbf{Distinct} & \textbf{Min gap} \\"
        ),
        r"\midrule",
    ]
    rendered = False
    for model in model_order:
        rows = sub[sub["model"] == model]
        if rows.empty:
            continue
        if rendered:
            lines.append(r"\midrule")
        rendered = True
        first = True
        for operator in OPERATORS_REPORTED:
            for scope_name in ("review", "all_levels"):
                row = rows[
                    (rows["operator"] == operator) & (rows["scope"] == scope_name)
                ]
                if row.empty:
                    continue
                record = row.iloc[0]
                prefix = (
                    rf"\multirow{{{2 * len(OPERATORS_REPORTED)}}}{{*}}{{{model}}}"
                    if first
                    else ""
                )
                first = False
                margin = (
                    "--" if record["margin_min"] == 0 else f"{record['margin_min']:.4f}"
                )
                lines.append(
                    f"  {prefix} & {_tt(operator)}"
                    f" & {_tt('review' if scope_name == 'review' else 'all')}"
                    f" & {int(record['distinct_values'])} & {margin}" + r" \\"
                )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


def discriminance_to_latex(
    disc: pd.DataFrame, out_path: Path, model_order: list[str]
) -> None:
    """Signal against noise, with the resolution column that tells them apart."""

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{Between-review spread against the noise each operator "
            r"leaves, in $\beta$ units. $\mathit{SD}_b$ is the spread of the "
            r"aggregated per-review evaluation across reviews; "
            r"$\mathit{SD}_{\mathrm{in}}$ the spread of the raw runs within a "
            r"review, identical across operators by construction; "
            r"$\mathit{SD}_{\mathrm{res}}$ the spread of the operator's own "
            r"output under resampling of the runs, the noise that survives "
            r"aggregation. Both within-review spreads are the square root of the "
            r"mean of the per-review variances. Ratio is "
            r"$\mathit{SD}_b/\mathit{SD}_{\mathrm{res}}$ and Share "
            r"$\sigma^2_b/(\sigma^2_b+\sigma^2_{\mathrm{res}})$, descriptive "
            r"ratios, not an ICC. Distinct counts the different review-level "
            r"aggregates and the different collective values of the "
            r"alternatives (two-level hierarchy).}"
        ),
        r"\label{tab:cdm_discriminance}",
        r"\small",
        r"\begin{tabular}{llcccccrr}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{Operator} & \textbf{$\mathit{SD}_b$}"
            r" & \textbf{$\mathit{SD}_{\mathrm{in}}$}"
            r" & \textbf{$\mathit{SD}_{\mathrm{res}}$}"
            r" & \textbf{Ratio} & \textbf{Share}"
            r" & \textbf{Distinct (review)} & \textbf{Distinct (collective)} \\"
        ),
        r"\midrule",
    ]
    rendered = False
    for model in model_order:
        rows = disc[disc["model"] == model]
        if rows.empty:
            continue
        if rendered:
            lines.append(r"\midrule")
        rendered = True
        for j, operator in enumerate(OPERATORS_REPORTED):
            row = rows[rows["operator"] == operator]
            if row.empty:
                continue
            record = row.iloc[0]
            prefix = (
                rf"\multirow{{{len(OPERATORS_REPORTED)}}}{{*}}{{{model}}}"
                if j == 0
                else ""
            )
            snr = (
                r"$\infty$"
                if not np.isfinite(record["snr"])
                else f"{record['snr']:.1f}"
            )
            lines.append(
                f"  {prefix} & {_tt(operator)}"
                f" & {record['between_sd']:.3f} & {record['input_sd']:.3f}"
                f" & {record['residual_sd']:.3f} & {snr}"
                f" & {record['signal_share']:.3f}"
                f" & {int(record['distinct_values'])}"
                f" & {int(record['distinct_collective'])}" + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def validity_to_latex(
    validity: pd.DataFrame, out_path: Path, model_order: list[str]
) -> None:
    """Accepted and rejected outputs per system, over the three schemes."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Output validity per system over the three schemes under "
            r"the stated parsing policy, with the LABEL-5 share separately. "
            r"Rejected replies leave the requested vocabulary or name more than "
            r"one of its labels; they are excluded from every reported metric.}"
        ),
        r"\label{tab:cdm_validity}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{Collected} & \textbf{Rejected}"
            r" & \textbf{\%} & \textbf{LABEL-5 \%} \\"
        ),
        r"\midrule",
    ]
    for model in model_order:
        rows = validity[validity["model"] == model]
        if rows.empty:
            continue
        collected = int(rows["collected"].sum())
        rejected = int(rows["rejected"].sum())
        label5 = rows[rows["scheme"] == "LABEL-5"]
        label5_pct = (
            f"{label5['rejected_pct'].iloc[0]:.2f}\\%" if not label5.empty else "--"
        )
        lines.append(
            f"  {model} & {collected:,}".replace(",", "{,}")
            + f" & {rejected} & {100.0 * rejected / collected:.2f}\\%"
            + f" & {label5_pct}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


def build_validity(runs: pd.DataFrame, invalid: pd.DataFrame) -> pd.DataFrame:
    """Collected against accepted, per system and scheme.

    The runs table holds only what parsed; `invalid` is the log of what did
    not (`invalid_outputs.csv`), so collected = accepted + rejected.
    """

    scheme_of = {dirname: scheme for scheme, (dirname, _) in SCHEMES.items()}
    rejected = invalid.assign(scheme=invalid["experiment"].map(scheme_of))
    rows = []
    for (model, scheme), g in runs.groupby(["model", "scheme"], sort=False):
        n_rejected = int(
            ((rejected["model"] == model) & (rejected["scheme"] == scheme)).sum()
        )
        collected = len(g) + n_rejected
        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "collected": collected,
                "accepted": len(g),
                "rejected": n_rejected,
                "rejected_pct": 100.0 * n_rejected / collected if collected else 0.0,
            }
        )
    return pd.DataFrame(rows)


def stars_to_latex(
    agreement: pd.DataFrame,
    ranking: pd.DataFrame,
    star_rank: pd.DataFrame,
    out_path: Path,
    scheme: str,
    model_order: list[str],
) -> None:
    """Agreement with the rating each review's own author gave it.

    The one external anchor in the study, and the only table here that compares
    the pipeline against something it did not produce.
    """

    sub = agreement[agreement["scheme"] == scheme]
    n = int(sub["n"].iloc[0]) if not sub.empty else 0
    order = " $>$ ".join(short(a) for a in star_rank["alternative"])
    star_margin = (
        float(ranking["star_margin_min"].iloc[0]) if not ranking.empty else float("nan")
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Agreement with the 1--5 star rating the author of each "
            r"review attached to it, over all "
            f"{n} reviews under {scheme}"
            r". A rating of $k$ stars is $\beta_{\star} = k-1$, so Diff.\ (the "
            r"mean signed difference from the stars, negative when the pipeline "
            r"scores a review below its author) and MAD are in $\beta$ units. "
            r"Same term is the share of reviews whose aggregate rounds onto the "
            r"author's own term; $\tau$ compares each collective ranking with "
            r"the star ranking given beneath. The stars rate a visit, the "
            r"answer rates the text: related constructs, not one quantity "
            r"measured twice.}"
        ),
        r"\label{tab:cdm_stars}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccrrcr}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{$r$} & \textbf{$\rho$}"
            r" & \textbf{Diff.} & \textbf{MAD} & \textbf{Same term}"
            r" & \textbf{$\tau$} \\"
        ),
        r"\midrule",
    ]
    ranks = ranking.set_index("model") if not ranking.empty else None
    for model in model_order:
        row = sub[sub["model"] == model]
        if row.empty:
            continue
        record = row.iloc[0]
        tau = (
            f"{ranks.loc[model, 'tau_vs_stars']:+.2f}"
            if ranks is not None and model in ranks.index
            else "--"
        )
        lines.append(
            f"  {model} & {record['pearson_r']:.3f} & {record['spearman_rho']:.3f}"
            f" & {record['bias_beta']:+.3f} & {record['mad_beta']:.3f}"
            f" & {record['same_term_pct']:.1f}\\% & {tau}" + r" \\"
        )
    lines += [
        r"\midrule",
        (
            rf"  \multicolumn{{7}}{{l}}{{Star ranking: {order}"
            rf" \quad (margin {star_margin:.5f})}} \\"
        ),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines))


def missing_to_latex(
    missing: pd.DataFrame,
    bias: pd.DataFrame,
    out_path: Path,
    scheme: str,
    model_order: list[str],
) -> None:
    """Ranking and margin under each treatment of rejected outputs."""

    sub = missing[missing["scheme"] == scheme]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Sensitivity of the collective ranking to the treatment "
            r"of rejected outputs under " + scheme + r". \texttt{available} "
            r"keeps a review and averages whatever runs parsed; "
            r"\texttt{listwise} drops the review unless all ten did. Neither is "
            r"neutral: the first compares reviews aggregated from different "
            r"numbers of runs, the second removes reviews from the corpus. "
            r"For the fused system \texttt{listwise} keeps a review only if "
            r"every member returned all ten runs for it. Only systems with "
            r"rejections can differ.}"
        ),
        r"\label{tab:cdm_missing}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrlr}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{Policy} & \textbf{Reviews}"
            r" & \textbf{Winner} & \textbf{Min gap} \\"
        ),
        r"\midrule",
    ]
    rendered = False
    for model in model_order:
        rows = sub[sub["model"] == model]
        if rows.empty:
            continue
        if rendered:
            lines.append(r"\midrule")
        rendered = True
        for j, policy in enumerate(("available", "listwise")):
            row = rows[rows["policy"] == policy]
            if row.empty:
                continue
            record = row.iloc[0]
            prefix = rf"\multirow{{2}}{{*}}{{{model}}}" if j == 0 else ""
            lines.append(
                f"  {prefix} & {_tt(policy)} & {int(record['reviews'])}"
                f" & {short(record['winner'])} & {record['margin_min']:.5f}"
                r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}"]

    affected = bias[bias["scheme"] == scheme]
    if not affected.empty:
        record = affected.iloc[0]
        lines.append(
            r"\\[2pt]\footnotesize Rejection is selective: "
            f"for {record['model']} the {int(record['reviews_affected'])} "
            r"affected reviews average $"
            f"{record['mean_y_affected']:+.2f}$ against $"
            f"{record['mean_y_rest']:+.2f}$ for the rest, and the share of an "
            r"alternative's reviews affected ranges from "
            f"{record['min_alt_rate']:.1f}\\% to {record['max_alt_rate']:.1f}\\%."
        )
    lines += [r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


def mapping_to_latex(
    mapping: pd.DataFrame, out_path: Path, model_order: list[str]
) -> None:
    """Does the ranking survive moving the terms off their uniform positions?"""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Sensitivity of the collective ranking to the linguistic "
            r"mapping. The 2-tuple model places the terms at equally spaced "
            r"indices; \texttt{compressed} moves the interior terms towards "
            r"the center and \texttt{stretched} away from it, holding the "
            r"endpoints and the ordering fixed. $\tau$ is the Kendall "
            r"correlation of the resulting ranking against the uniform one.}"
        ),
        r"\label{tab:cdm_mapping}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lllcc}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{Scheme} & \textbf{Placement}"
            r" & \textbf{Winner} & \textbf{$\tau$ vs uniform} \\"
        ),
        r"\midrule",
    ]
    for model in model_order:
        rows = mapping[mapping["model"] == model]
        for _, record in rows.iterrows():
            lines.append(
                f"  {model} & {record['scheme']}"
                f" & {_tt(str(record['placement']))}"
                f" & {short(record['winner'])}"
                f" & {record['tau_vs_uniform']:.3f}" + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


def star_system(reviews: pd.DataFrame, stars: pd.DataFrame) -> pd.DataFrame:
    """The reviewers' ratings as one more system, on the same reviews.

    `reviews` is the review-level frame of any one system; its `beta` is
    replaced by the author's `beta_star`, so the stars are aggregated and
    resampled on exactly the same (alternative, participant) clusters.
    """

    by_position = stars.set_index("position")["beta_star"]
    out = reviews.copy()
    out["model"] = STARS_LABEL
    out["beta"] = by_position.loc[out["review_id"].map(review_position)].to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("profile", help="london or madrid")
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write here instead of the profile's out_dir (e.g. a rerun under "
        "another seed or replica count).",
    )
    args = parser.parse_args()

    profile = load_profile(args.profile)
    out = args.out or profile.out_dir
    tables, figures = out / "tables", out / "figures"
    for directory in (out, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    stars_source = star_frame(profile.data_csv)
    runs = attach_entities(load_runs(profile), profile.data_csv)
    present = profile.model_order
    open_weight = profile.members
    model_order = [*present, FUSION_LABEL]
    print(f"{profile.name}: {len(runs):,} runs over {len(present)} systems")
    print(f"  fusion members: {', '.join(open_weight)}")

    print("\nReview-level aggregation and fusion...")
    reviews = fuse_runs(runs, "two_tuple")
    fused = fuse_models(reviews, "two_tuple", open_weight, FUSION_LABEL)
    alternatives = build_alternatives(reviews, fused)
    alternatives.to_csv(out / "cdm_alternatives.csv", index=False)
    for hierarchy in HIERARCHIES:
        alternatives_to_latex(
            alternatives,
            tables / f"cdm_alternatives_{hierarchy}.tex",
            SCHEME,
            hierarchy,
            model_order,
        )

    scheme_runs = runs[runs["scheme"] == SCHEME]

    print(f"\nRun resampling per operator ({args.replicas} replicas)...")
    stability = pd.concat(
        [
            decision_stability(scheme_runs, op, "two_level", args.replicas, args.seed)
            for op in OPERATORS_REPORTED
        ],
        ignore_index=True,
    )
    stability.to_csv(out / "cdm_stability.csv", index=False)
    stability_to_latex(stability, tables / "cdm_stability.tex", present)

    print("\nParticipant resampling...")
    # Review-level frames, per operator, with the fused system appended: the
    # fusion is scored by the same code as an individual system.
    by_operator = {op: fuse_runs(scheme_runs, op) for op in OPERATORS_REPORTED}
    with_fusion = {
        op: pd.concat(
            [frame, fuse_models(frame, op, open_weight, FUSION_LABEL)],
            ignore_index=True,
        )
        for op, frame in by_operator.items()
    }
    # One set of crowd replicas for the corpus, shared by every system and
    # operator, by the joint table and by the candidate sets.
    draws = draw_crowd(with_fusion["two_tuple"], args.replicas, args.seed)
    crowd = pd.concat(
        [
            crowd_stability(frame, "two_level", draws=draws)
            for frame in with_fusion.values()
        ],
        ignore_index=True,
    )
    crowd.to_csv(out / "cdm_crowd_stability.csv", index=False)
    crowd_to_latex(crowd, tables / "cdm_crowd_stability.tex", model_order)

    print("\nJoint resampling: runs, participants, both...")
    joint, joint_intervals = joint_stability(
        scheme_runs,
        "two_tuple",
        args.replicas,
        args.seed,
        open_weight,
        FUSION_LABEL,
        draws=draws,
    )
    joint.to_csv(out / "cdm_joint.csv", index=False)
    joint_intervals.to_csv(out / "cdm_joint_intervals.csv", index=False)
    joint_to_latex(joint, tables / "cdm_joint.tex", SCHEME, model_order)

    print("\nTie rule of the majority operator...")
    ties = tiebreak_sensitivity(scheme_runs, open_weight, FUSION_LABEL)
    ties.to_csv(out / "cdm_tiebreak.csv", index=False)
    tiebreak_to_latex(ties, tables / "cdm_tiebreak.tex", model_order)

    print("\nCandidate sets, pairwise and simultaneous, stars included...")
    first = with_fusion["two_tuple"]
    first = first[first["model"] == present[0]]
    with_stars = pd.concat(
        [with_fusion["two_tuple"], star_system(first, stars_source)],
        ignore_index=True,
    )
    crowd_draws = [draws]
    if profile.stratified:
        crowd_draws.append(
            draw_crowd(with_fusion["two_tuple"], args.replicas, args.seed, "stratified")
        )
    gaps, candidates, simult_pairs, simult_sets = [], [], [], []
    for d in crowd_draws:
        g = rival_gaps(with_stars, d)
        top1 = crowd_stability(with_stars, "two_level", draws=d)
        gaps.append(g)
        candidates.append(
            candidate_sets(g).merge(
                top1[
                    ["model", "scheme", "crowd_mode", "top1_pct", "top1_lo", "top1_hi"]
                ],
                on=["model", "scheme", "crowd_mode"],
                how="left",
            )
        )
        for studentized in (True, False):
            pairs, sets = simultaneous_candidates(
                with_stars, d, studentized=studentized
            )
            simult_pairs.append(pairs)
            simult_sets.append(sets)
    gaps = pd.concat(gaps, ignore_index=True)
    candidates = pd.concat(candidates, ignore_index=True)
    simult_pairs = pd.concat(simult_pairs, ignore_index=True)
    simult_sets = pd.concat(simult_sets, ignore_index=True)
    max_t = simult_sets[simult_sets["method"] == "max_t"].rename(
        columns={
            "c": "c_simult",
            "candidates": "candidates_simult",
            "candidate_ranks": "candidate_ranks_simult",
            "set_size": "set_size_simult",
        }
    )
    candidates = candidates.merge(
        max_t[
            [
                "model",
                "scheme",
                "crowd_mode",
                "c_simult",
                "candidates_simult",
                "candidate_ranks_simult",
                "set_size_simult",
            ]
        ],
        on=["model", "scheme", "crowd_mode"],
        how="left",
    )
    gaps.to_csv(out / "cdm_rival_gaps.csv", index=False)
    candidates.to_csv(out / "cdm_candidates.csv", index=False)
    simult_pairs.to_csv(out / "cdm_simultaneous_pairs.csv", index=False)
    simult_sets.to_csv(out / "cdm_simultaneous_candidates.csv", index=False)
    with_stars_order = [*model_order, STARS_LABEL]
    candidates_to_latex(
        candidates, tables / "cdm_candidates.tex", SCHEME, with_stars_order
    )
    simultaneous_pairs_to_latex(
        simult_pairs,
        simult_sets,
        tables / "cdm_simultaneous_pairs.tex",
        SCHEME,
        with_stars_order,
    )
    print(
        candidates[
            [
                "model",
                "crowd_mode",
                "winner",
                "top1_pct",
                "rivals_kept",
                "set_size_simult",
            ]
        ].to_string(index=False)
    )

    print("\nOperator scope (review stages against every stage)...")
    scope = operator_scope_contrast(by_operator, "two_level")
    scope.to_csv(out / "cdm_operator_scope.csv", index=False)
    scope_to_latex(scope, tables / "cdm_operator_scope.tex", SCHEME, present)

    print("\nBetween-review against residual spread...")
    disc = pd.concat(
        [
            discriminance(scheme_runs, op, DEFAULT_PANEL_REPLICAS, args.seed)
            for op in OPERATORS_REPORTED
        ],
        ignore_index=True,
    )
    disc.to_csv(out / "cdm_discriminance.csv", index=False)
    discriminance_to_latex(disc, tables / "cdm_discriminance.tex", present)

    print("\nRejected-output treatment...")
    missing = missing_data_sensitivity(
        scheme_runs,
        "two_tuple",
        "two_level",
        fusion_members=open_weight,
        fusion_label=FUSION_LABEL,
    )
    bias = rejection_bias(runs)
    missing.to_csv(out / "cdm_missing_data.csv", index=False)
    bias.to_csv(out / "cdm_rejection_bias.csv", index=False)
    missing_to_latex(
        missing, bias, tables / "cdm_missing_data.tex", SCHEME, model_order
    )

    print("\nTerm placement...")
    mapping = mapping_sensitivity(
        runs, "two_tuple", "two_level", open_weight, FUSION_LABEL
    )
    mapping.to_csv(out / "cdm_mapping.csv", index=False)
    mapping_to_latex(mapping, tables / "cdm_mapping.tex", model_order)

    print("\nAgreement with the reviewers' star ratings...")
    agreement = review_agreement(
        pd.concat([reviews, fused], ignore_index=True), stars_source
    )
    star_rank = star_alternatives(stars_source, hierarchy="two_level")
    against_stars = ranking_agreement(alternatives, SCHEME, "two_level", stars_source)
    agreement.to_csv(out / "cdm_star_agreement.csv", index=False)
    star_rank.to_csv(out / "cdm_star_ranking.csv", index=False)
    against_stars.to_csv(out / "cdm_star_vs_ranking.csv", index=False)
    stars_to_latex(
        agreement,
        against_stars,
        star_rank,
        tables / "cdm_stars.tex",
        SCHEME,
        model_order,
    )

    print("\nInference budgets...")
    latency = pd.read_csv(profile.out_dir / "latency_summary.csv")
    greedy = load_runs(load_profile(f"{profile.name}_t0"))
    budget = build_budget(
        scheme_runs,
        alternatives,
        open_weight,
        latency,
        SCHEME,
        greedy,
        profile.data_csv,
        args.seed,
        draws,
    )
    budget.to_csv(out / "cdm_budget.csv", index=False)
    calls = calls_comparison(budget)
    calls.to_csv(out / "cdm_budget_calls.csv", index=False)
    budget_calls_to_latex(calls, tables / "cdm_budget_calls.tex")
    budget_to_latex(budget, tables / "cdm_budget.tex")

    print("\nFusion against each member...")
    versus = fusion_vs_best(
        joint,
        alternatives,
        agreement,
        latency,
        SCHEME,
        open_weight,
        FUSION_LABEL,
        candidates,
    )
    versus.to_csv(out / "cdm_fusion_vs_best.csv", index=False)
    fusion_vs_best_to_latex(versus, tables / "cdm_fusion_vs_best.tex", model_order)
    print(versus.to_string(index=False))

    print("\nOutput validity...")
    invalid = pd.read_csv(profile.out_dir / "invalid_outputs.csv")
    validity = build_validity(runs, invalid)
    validity.to_csv(out / "cdm_validity.csv", index=False)
    validity_to_latex(validity, tables / "cdm_validity.tex", present)

    print("\nFigures...")
    plot_alternative_profile(
        alternatives, SCHEME, "two_level", model_order, figures / "alternative_profile"
    )
    plot_joint_intervals(joint_intervals, SCHEME, model_order, figures / "sensitivity")
    plot_resolution(disc, present, figures / "resolution")
    plot_decision_stability(stability, present, figures / "ranking_stability")
    print(f"  written to {out}/")


if __name__ == "__main__":
    main()
