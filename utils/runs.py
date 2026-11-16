"""Raw archives -> long-format runs and the log of rejected replies.

    python -m utils.runs london

For every model of the profile this extracts `<raw_dir>/<model>/<scheme>.tar.gz`
(`<scheme>/review-<i>/run-<n>.json` at the archive root), parses each reply
with `utils.parsing` and writes, into the profile's `out_dir`:

    runs_<model>.csv      one row per accepted reply:
                          model | scheme | review_id | uid | run | raw | y
    invalid_outputs.csv   one row per rejected reply, with the reason

`y` is the reply normalized to [-1, 1]. `review_id` is the review folder name,
`review-<i>` with `i` the row position in the corpus CSV, and is the unit of
analysis. `uid` (`USER_ID_ENTITY_ID`) is kept for traceability but is not
unique, so nothing groups by it.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections.abc import Iterator

import pandas as pd

from .parsing import LABEL_5_VALUES, LABEL_7_VALUES, classify_label, normalize_score
from .profiles import Profile, load_profile

INVALID_COLUMNS = [
    "model",
    "raw_model_id",
    "experiment",
    "review_dir",
    "run",
    "review_idx",
    "raw",
    "reason",
]


def _run_index(run_name: str) -> tuple[int, str]:
    """Sort key for run-<n>.json: numeric, so run-10 follows run-9."""

    stem = run_name.removesuffix(".json")
    _, _, number = stem.partition("-")
    return (int(number), "") if number.isdigit() else (0, stem)


def iter_records(
    profile: Profile, raw_model: str, scheme_dir: str
) -> Iterator[tuple[str, dict]]:
    """(review folder, run record) of one archive: reviews in name order, runs
    in numeric order."""

    with tarfile.open(profile.archive(raw_model, scheme_dir), "r:gz") as tar:
        members = [
            m for m in tar.getmembers() if m.isfile() and m.name.endswith(".json")
        ]
        members.sort(
            key=lambda m: (m.name.split("/")[-2], _run_index(m.name.split("/")[-1]))
        )
        for member in members:
            extracted = tar.extractfile(member)
            if extracted is not None:
                yield member.name.split("/")[-2], json.load(extracted)


def _reject_reason(scheme_dir: str, raw: str) -> str:
    """`invalid_score`, or which way a label reply failed: `no_label` (no
    category present) or `ambiguous` (two or more distinct categories)."""

    if scheme_dir == "single_score_v2_t1":
        return "invalid_score"
    labels = LABEL_7_VALUES if scheme_dir == "single_label_7c_t1" else LABEL_5_VALUES
    return classify_label(raw, labels)[1]


def build(profile: Profile) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    """Runs table per model (keyed by its CSV name) and the rejected replies."""

    tables: dict[str, pd.DataFrame] = {}
    invalid: list[dict[str, str]] = []
    for entry in profile.models:
        rows = []
        for scheme, (scheme_dir, parser) in profile.schemes.items():
            for review_dir, record in iter_records(profile, entry.raw, scheme_dir):
                raw = record["history"][-1]["content"]
                parsed = parser(raw)
                if scheme == "SCORE":
                    y = None if parsed is None else normalize_score(parsed)
                else:
                    y = parsed[0]
                if y is None:
                    invalid.append(
                        {
                            "model": entry.display,
                            "raw_model_id": entry.raw,
                            "experiment": scheme_dir,
                            "review_dir": review_dir,
                            "run": str(record.get("run_number", "")),
                            "review_idx": str(record.get("review_idx", "")),
                            "raw": raw.strip(),
                            "reason": _reject_reason(scheme_dir, raw.strip()),
                        }
                    )
                    continue
                rows.append(
                    {
                        "model": entry.display,
                        "scheme": scheme,
                        "review_id": review_dir,
                        "uid": record["review_idx"],
                        "run": int(record["run_number"]),
                        "raw": raw.strip(),
                        "y": y,
                    }
                )
        tables[entry.runs_csv] = pd.DataFrame(rows)
    return tables, invalid


def load_runs(profile: Profile) -> pd.DataFrame:
    """Every runs CSV of a profile, in file-name order."""

    paths = sorted(profile.out_dir.glob("runs_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no runs_*.csv in {profile.out_dir}; run "
            f"`python -m utils.runs {profile.name}` first"
        )
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("profile")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    profile.out_dir.mkdir(parents=True, exist_ok=True)
    tables, invalid = build(profile)
    for name, runs in tables.items():
        runs.to_csv(profile.out_dir / name, index=False)
        print(f"  {profile.out_dir / name}  ({len(runs):,} rows)")
    with (profile.out_dir / "invalid_outputs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INVALID_COLUMNS)
        writer.writeheader()
        writer.writerows(invalid)
    print(f"  {profile.out_dir / 'invalid_outputs.csv'}  ({len(invalid)} rejected)")


if __name__ == "__main__":
    main()
