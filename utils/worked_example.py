"""The manuscript's worked example, chosen by rule rather than by hand.

The review shown is the one on which the systems disagree most under LABEL-5:
the largest range (max - min) of the review-level 2-tuple beta across the
reported systems, ties broken by the larger standard deviation and then by
row position. Everything printed in the example box is read from the runs:
each system's ten labels, its 2-tuple and the fused 2-tuple.

    python -m utils.worked_example london
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .cdm import attach_entities, review_position, scale_for
from .fusion import fuse_models, fuse_runs
from .profiles import load_profile
from .runs import load_runs
from .twotuple import beta_from_y, delta, format_tuple

SCHEME = "LABEL-5"
# The box shows the opening of the review; the full text is in the corpus at
# the row the review id names.
EXCERPT_WORDS = 80
FUSION_LABEL = "Fusion (open-weight)"


def choose(reviews: pd.DataFrame) -> str:
    """Review id with the widest spread of review-level betas across systems."""

    wide = reviews.pivot_table(index="review_id", columns="model", values="beta")
    wide = wide.dropna()
    spread = pd.DataFrame(
        {
            "range": wide.max(axis=1) - wide.min(axis=1),
            "sd": wide.std(axis=1, ddof=0),
            "position": [review_position(r) for r in wide.index],
        },
        index=wide.index,
    )
    spread = spread.sort_values(
        ["range", "sd", "position"], ascending=[False, False, True]
    )
    return str(spread.index[0])


def _tex(text: str) -> str:
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\^{}"),
    ):
        text = text.replace(a, b)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("profile", nargs="?", default="london")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    out, tables_dir = profile.out_dir, profile.out_dir / "tables"
    runs = load_runs(profile)
    runs = attach_entities(runs[runs["scheme"] == SCHEME], profile.data_csv)
    present = profile.model_order
    members = profile.members

    reviews = fuse_runs(runs, "two_tuple")
    review_id = choose(reviews[reviews["model"].isin(present)])
    fused = fuse_models(reviews, "two_tuple", members, FUSION_LABEL)

    scale = scale_for(SCHEME)
    source = pd.read_csv(profile.data_csv)
    position = review_position(review_id)
    row = source.iloc[position]

    records = []
    for model in present:
        r = runs[(runs["model"] == model) & (runs["review_id"] == review_id)]
        r = r.sort_values("run")
        terms = [round(beta_from_y(y, scale)) for y in r["y"]]
        beta = float(np.mean([beta_from_y(y, scale) for y in r["y"]]))
        records.append(
            {
                "review_id": review_id,
                "model": model,
                "runs": len(terms),
                "labels": " ".join(str(t) for t in terms),
                "beta": round(beta, 4),
                "two_tuple": format_tuple(delta(beta, scale), scale),
            }
        )
    fused_beta = float(fused.loc[fused["review_id"] == review_id, "beta"].iloc[0])
    records.append(
        {
            "review_id": review_id,
            "model": FUSION_LABEL,
            "runs": np.nan,
            "labels": "",
            "beta": round(fused_beta, 4),
            "two_tuple": format_tuple(delta(fused_beta, scale), scale),
        }
    )
    table = pd.DataFrame(records)
    table.to_csv(out / "cdm_worked_example.csv", index=False)
    print(table.to_string(index=False))

    words = str(row["REVIEW_BODY"]).split()
    excerpt = _tex(" ".join(words[:EXCERPT_WORDS]))
    if len(words) > EXCERPT_WORDS:
        excerpt += rf" [\ldots{{}} {len(words) - EXCERPT_WORDS} more words]"
    body = [
        rf"\textit{{``{excerpt}''}}",
        "",
        (
            rf"\smallskip\noindent Author's rating: {int(row['REVIEW_RATING_GLOBAL'])}"
            rf" stars. Alternative: {_tex(str(row['ENTITY_NAME']))}."
        ),
        (
            r"Run labels as term indices, 0 = \emph{very negative} to 4 = "
            r"\emph{very positive}."
        ),
        "",
        r"\smallskip\noindent{\footnotesize\begin{tabular}{@{}lll@{}}",
        r"System & Ten runs & 2-tuple \\",
        r"\hline",
    ]
    for rec in records:
        body.append(
            f"{rec['model']} & {rec['labels'] or '--'} & {_tex(rec['two_tuple'])}"
            r" \\"
        )
    body.append(r"\end{tabular}}")
    tex = [
        r"\begin{table}[htbp]",
        r"\examplebox{Worked example: \texttt{" + _tex(review_id) + "}}{%",
        *body,
        "}",
        (
            r"\caption{The review on which the systems disagree most under LABEL-5 "
            r"(largest range of the review-level $\beta$ across systems, chosen by "
            r"\texttt{utils/worked\_example.py}), its ten runs per system, and the "
            r"2-tuples they aggregate to.}"
        ),
        r"\label{tab:cdm_worked_example}",
        r"\end{table}",
        "",
    ]
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "cdm_worked_example.tex").write_text("\n".join(tex))


if __name__ == "__main__":
    main()
