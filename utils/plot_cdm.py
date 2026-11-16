"""The supplementary figures, each saved as .pdf and .png.

plot_alternative_profile   collective evaluations on the term-index domain
plot_joint_intervals       intervals under run, participant and joint resampling
plot_resolution            between-review against residual spread, per operator
plot_decision_stability    ranking stability under run resampling, per operator
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .fusion import LOSSLESS, SNAPPING

OPERATOR_ORDER = (*LOSSLESS, *SNAPPING)
OPERATOR_COLORS = {
    "two_tuple": "#55a868",
    "mean_round": "#4c72b0",
    "median_round": "#8172b2",
    "majority": "#c44e52",
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig, out_path: Path) -> None:
    """Save `out_path.png` and `out_path.pdf`."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path.with_name(out_path.name + ".png"), bbox_inches="tight", dpi=200
    )
    # No creation date, so that a rebuild leaves the PDF unchanged.
    fig.savefig(
        out_path.with_name(out_path.name + ".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    plt.close(fig)


def _bars(ax, frame: pd.DataFrame, models: Sequence[str], column: str) -> None:
    """Grouped bars: one cluster per model, one bar per operator."""

    width = 0.8 / len(OPERATOR_ORDER)
    positions = np.arange(len(models))
    for i, operator in enumerate(OPERATOR_ORDER):
        rows = frame[frame["operator"] == operator].set_index("model")
        values = [rows[column].get(m, np.nan) for m in models]
        ax.bar(
            positions + i * width - 0.4 + width / 2,
            values,
            width=width,
            label=operator.replace("_", " "),
            color=OPERATOR_COLORS.get(operator, "#888888"),
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=15, ha="right")


def _legend_below(fig, source_ax) -> None:
    """One figure-level legend under the panels.

    Both panels reach 100% on the most stable system, so an in-axes legend
    covers bars whichever corner it is placed in.
    """

    handles, labels = source_ax.get_legend_handles_labels()
    # Reserve the strip first: the tick labels are rotated and would otherwise
    # sit in the same band as the legend, which bbox_inches="tight" preserves
    # rather than resolves.
    fig.subplots_adjust(bottom=0.24)
    fig.legend(
        handles,
        labels,
        title="Operator",
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize="small",
        bbox_to_anchor=(0.5, 0.0),
    )


def plot_decision_stability(
    stability: pd.DataFrame, model_order: Sequence[str], out_path: Path
) -> None:
    """How often the collective ranking survives resampling, per operator."""

    apply_style()
    models = [m for m in model_order if m in set(stability["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    _bars(axes[0], stability, models, "exact_pct")
    axes[0].set_ylabel("Replicas reproducing the ranking (%)")
    axes[0].set_title("A", loc="left", fontweight="bold")
    axes[0].set_ylim(0, 100)

    _bars(axes[1], stability, models, "top1_pct")
    axes[1].set_ylabel("Replicas keeping the winner (%)")
    axes[1].set_title("B", loc="left", fontweight="bold")
    axes[1].set_ylim(0, 100)

    _legend_below(fig, axes[0])
    _save(fig, out_path)


def plot_resolution(
    disc: pd.DataFrame, model_order: Sequence[str], out_path: Path
) -> None:
    """Signal against surviving noise, beside the resolution that survives too.

    The two panels are the two things aggregation is supposed to do, and the
    lossless operator wins both. Panel A is signal over the noise the operator
    actually leaves behind, measured by resampling rather than inherited from
    the raw runs: snapping *amplifies* residual noise, because a small shift in
    the mean moves the rounded term by a whole unit, so the ratio is two to
    three times worse. Panel B is the resolution the collective decision
    inherits, where the gap is an order of magnitude. Neither panel alone would
    settle it -- an operator can score well on A by refusing to discriminate --
    which is why they are shown together.
    """

    apply_style()
    models = [m for m in model_order if m in set(disc["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    _bars(axes[0], disc, models, "snr")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\mathit{SD}_b / \mathit{SD}_{\mathrm{res}}$ (log)")
    axes[0].set_title("A", loc="left", fontweight="bold")

    _bars(axes[1], disc, models, "distinct_values")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Distinct review-level values (log)")
    axes[1].set_title("B", loc="left", fontweight="bold")

    _legend_below(fig, axes[1])
    _save(fig, out_path)


def plot_alternative_profile(
    alternatives: pd.DataFrame,
    scheme: str,
    hierarchy: str,
    model_order: Sequence[str],
    out_path: Path,
) -> None:
    """Collective evaluation of each alternative, one line per system."""

    apply_style()
    sub = alternatives[
        (alternatives["scheme"] == scheme) & (alternatives["hierarchy"] == hierarchy)
    ]
    models = [m for m in model_order if m in set(sub["model"])]
    names = list(sub[sub["model"] == models[0]].sort_values("rank")["alternative"])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(names))
    for model in models:
        rows = sub[sub["model"] == model].set_index("alternative")
        fused = model.startswith("Fusion")
        ax.plot(
            positions,
            [rows["beta"].get(n, np.nan) for n in names],
            marker="D" if fused else "o",
            linewidth=2.4 if fused else 1.3,
            markersize=8 if fused else 5,
            color="#000000" if fused else None,
            zorder=3 if fused else 2,
            label=model,
        )
    # Term boundaries: where the rounded term would change. Between them every
    # alternative carries the same label and only alpha orders them.
    low, high = ax.get_ylim()
    # Widen to the boundaries that enclose the data, so the band the
    # alternatives share is visible whole.
    low = min(low, np.floor(low - 0.5) + 0.5 - 0.05)
    high = max(high, np.ceil(high - 0.5) + 0.5 + 0.05)
    for boundary in np.arange(0.5, 6.5, 1.0):
        if low < boundary < high:
            ax.axhline(boundary, color="#888888", linewidth=0.8, linestyle=":")
            ax.annotate(
                f"term boundary {boundary:g}",
                (positions[-1], boundary),
                xytext=(0, 3),
                textcoords="offset points",
                ha="right",
                fontsize="x-small",
                color="#666666",
            )
    ax.set_ylim(low, high)
    ax.set_xticks(positions)
    ax.set_xticklabels([n.split(",")[0] for n in names], rotation=15, ha="right")
    ax.set_ylabel(r"Collective evaluation $\beta$")
    ax.legend(frameon=True, fontsize="small", ncol=2, loc="lower left")
    _save(fig, out_path)


CONDITION_STYLE = {
    "runs": ("Runs resampled", "#4c72b0"),
    "experts": ("Experts resampled", "#dd8452"),
    "both": ("Both", "#000000"),
}


def plot_joint_intervals(
    intervals: pd.DataFrame,
    scheme: str,
    model_order: Sequence[str],
    out_path: Path,
) -> None:
    """Each alternative's evaluation with its 95% interval under the three
    resamplings, one panel per system, alternatives in each system's own
    full-data order."""

    apply_style()
    sub = intervals[intervals["scheme"] == scheme]
    models = [m for m in model_order if m in set(sub["model"])]
    n_alternatives = sub["alternative"].nunique()
    # Four alternatives fit small multiples four abreast; twelve need room for
    # their names, so the panels go two abreast and wider.
    many = n_alternatives > 6
    cols = min(2 if many else 4, len(models))
    rows_n = int(np.ceil(len(models) / cols))
    panel = (7.0, 4.2) if many else (3.4, 3.0)
    fig, axes = plt.subplots(
        rows_n,
        cols,
        figsize=(panel[0] * cols, panel[1] * rows_n),
        sharey=True,
        squeeze=False,
    )
    offsets = {c: (k - 1) * 0.22 for k, c in enumerate(CONDITION_STYLE)}
    for ax, model in zip(axes.flat, models, strict=False):
        rows = sub[sub["model"] == model]
        names = list(
            rows[rows["condition"] == "runs"].sort_values("rank")["alternative"]
        )
        x = np.arange(len(names))
        for condition, (label, color) in CONDITION_STYLE.items():
            r = rows[rows["condition"] == condition].set_index("alternative")
            beta = r["beta"].reindex(names).to_numpy()
            ax.errorbar(
                x + offsets[condition],
                beta,
                yerr=[
                    beta - r["beta_lo"].reindex(names).to_numpy(),
                    r["beta_hi"].reindex(names).to_numpy() - beta,
                ],
                fmt="o",
                markersize=3.5,
                capsize=2,
                linewidth=1.1,
                color=color,
                label=label,
            )
        ax.set_title(model, fontsize="medium")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.split(",")[0].split(" - ")[0] for n in names],
            rotation=45 if many else 35,
            ha="right",
            fontsize="small",
        )
    for ax in axes.flat[len(models) :]:
        ax.set_visible(False)
    # Term boundaries, as in the alternative profile: an interval that stays
    # between two of them never changes the rounded term. The axis is shared,
    # so the limits are read once after every panel is drawn.
    low, high = axes.flat[0].get_ylim()
    for ax in axes.flat[: len(models)]:
        for boundary in np.arange(0.5, 6.5, 1.0):
            if low < boundary < high:
                ax.axhline(
                    boundary,
                    color="#888888",
                    linewidth=0.8,
                    linestyle=":",
                    zorder=0,
                )
    axes.flat[0].set_ylim(low, high)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\beta$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    _save(fig, out_path)
