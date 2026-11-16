"""Monte Carlo sensitivity of the simultaneous candidate sets.

    python -m utils.seeds london
    python -m utils.seeds madrid

Recomputes the candidate sets of the main analysis (bootstrap max-t over all
pairwise differences, participants resampled; every system, the fusion and the
reviewers' stars) under other seeds and with more replicas: seeds 1 to 4 at
20,000 replicas, and seeds 42, 1 and 2 at 60,000. The reported analysis is
seed 42 at 20,000 (`cdm_simultaneous_candidates.csv`). Writes
`seed_sensitivity.csv` into the profile's `out_dir`, one row per (replicas,
seed, system): the size of the simultaneous set, of the pairwise set, and the
critical value c.
"""

from __future__ import annotations

import argparse

import pandas as pd

from .cdm import attach_entities
from .cdm_analyze import FUSION_LABEL, SCHEME, star_system
from .fusion import fuse_models, fuse_runs
from .profiles import load_profile
from .runs import load_runs
from .stability import draw_crowd, simultaneous_candidates
from .validity import star_frame

DESIGNS = (
    (20_000, 1),
    (20_000, 2),
    (20_000, 3),
    (20_000, 4),
    (60_000, 42),
    (60_000, 1),
    (60_000, 2),
)


def seed_sensitivity(profile_name: str) -> pd.DataFrame:
    profile = load_profile(profile_name)
    runs = attach_entities(load_runs(profile), profile.data_csv)
    reviews = fuse_runs(runs[runs["scheme"] == SCHEME], "two_tuple")
    with_fusion = pd.concat(
        [reviews, fuse_models(reviews, "two_tuple", profile.members, FUSION_LABEL)],
        ignore_index=True,
    )
    first = with_fusion[with_fusion["model"] == profile.model_order[0]]
    stars = star_system(first, star_frame(profile.data_csv))
    with_stars = pd.concat([with_fusion, stars], ignore_index=True)

    tables = []
    for replicas, seed in DESIGNS:
        draws = draw_crowd(with_fusion, replicas, seed)
        _, sets = simultaneous_candidates(with_stars, draws)
        tables.append(
            pd.DataFrame(
                {
                    "reps": replicas,
                    "seed": seed,
                    "model": sets["model"],
                    "set_size_simult": sets["set_size"],
                    "set_size_pairwise": sets["set_size_pairwise"],
                    "c": sets["c"],
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("profile", choices=["london", "madrid"])
    args = parser.parse_args()

    table = seed_sensitivity(args.profile)
    out = load_profile(args.profile).out_dir / "seed_sensitivity.csv"
    table.to_csv(out, index=False)
    print(
        table.pivot_table(
            index="model",
            columns=["reps", "seed"],
            values="set_size_simult",
            sort=False,
        ).to_string()
    )


if __name__ == "__main__":
    main()
