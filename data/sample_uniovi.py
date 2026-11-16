"""Sample the Madrid corpus: 12 alternatives x 50 reviews.

Source: TripAdvisor Restaurant Reviews v2.0.0, Universidad de Oviedo
(Zenodo, doi:10.5281/zenodo.14622324, CC BY 4.0), file `Madrid.zip`.

The protocol is fixed in advance and every step is mechanical:

0. The distributed pickles repeat some rows verbatim (8,608 duplicated review
   ids, 76 duplicated item ids in the Madrid release). Each id is kept once.
1. Reviews in English (`language == "en"`) whose text has at least 20
   whitespace-separated tokens and whose `rating` is a multiple of 10
   (TripAdvisor stores 1-5 stars as 10-50); stars = rating / 10.
2. Restaurants with at least 80 such reviews, restricted to the most populated
   `priceInterval` among them (an empty interval is unrecorded and is not a
   band), so the alternatives compete in one price band.
3. The 12 restaurants with the most such reviews are the alternatives (ties
   broken by restaurant id).
4. 50 reviews per restaurant drawn uniformly at random without replacement,
   seed 20261014, so each alternative keeps its natural star distribution.
5. Rows are ordered by alternative (in step-3 order), then by review id. The
   row position is the review's identity downstream (`review-<i>`), exactly as
   in the London corpus; the order is fixed here and never changed afterwards.

Usage, from the repository root, with `Madrid.zip` of the Zenodo release
unzipped into `data/uniovi_tripadvisor/Madrid`:

    python data/sample_uniovi.py data/uniovi_tripadvisor/Madrid \
        --zip data/uniovi_tripadvisor/Madrid.zip

It rewrites `data/UniOvi-Madrid-600.csv` byte for byte and its manifest.

`--candidates-only` prints the step-3 table and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261014
MIN_TOKENS = 20
MIN_REVIEWS = 80
N_ALTERNATIVES = 12
PER_ALTERNATIVE = 50


def load(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reviews and restaurants as distributed (pickled pandas frames)."""

    reviews = pd.read_pickle(source / "reviews.pkl")
    items = pd.read_pickle(source / "items.pkl")
    # Step 0: exact duplicate rows in the release; one copy per id.
    reviews = reviews.drop_duplicates("reviewId", keep="first")
    items = items.drop_duplicates("itemId", keep="first")
    return reviews, items


def eligible_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Step 1: English, at least MIN_TOKENS tokens, whole-star rating."""

    text = reviews["text"].fillna("").astype(str)
    keep = (
        (reviews["language"] == "en")
        & (text.str.split().str.len() >= MIN_TOKENS)
        & (reviews["rating"] % 10 == 0)
        & reviews["rating"].between(10, 50)
    )
    out = reviews.loc[keep].copy()
    out["stars"] = (out["rating"] // 10).astype(int)
    return out


def candidates(reviews: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Steps 2 and 3: the alternatives, with their descriptive statistics."""

    counts = reviews.groupby("itemId").size().rename("n_en")
    pool = items.set_index("itemId").join(counts, how="inner")
    pool = pool[pool["n_en"] >= MIN_REVIEWS]
    # An empty priceInterval is "not recorded", not a price band.
    priced = pool["priceInterval"].fillna("").str.strip() != ""
    band = pool.loc[priced, "priceInterval"].value_counts().idxmax()
    pool = pool[pool["priceInterval"] == band]
    pool = pool.reset_index().sort_values(["n_en", "itemId"], ascending=[False, True])
    chosen = pool.head(N_ALTERNATIVES).copy()

    rows = []
    for record in chosen.itertuples():
        r = reviews[reviews["itemId"] == record.itemId]
        dist = r["stars"].value_counts().reindex(range(1, 6), fill_value=0)
        rows.append(
            {
                "itemId": record.itemId,
                "name": record.name,
                "priceInterval": band,
                "n_en": int(record.n_en),
                "users": int(r["userId"].nunique()),
                "mean_stars": round(float(r["stars"].mean()), 3),
                **{f"s{k}": int(dist[k]) for k in range(1, 6)},
            }
        )
    return pd.DataFrame(rows)


def draw(reviews: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    """Step 4 and 5: 50 reviews per alternative, in a fixed row order."""

    rng = np.random.default_rng(SEED)
    parts = []
    for record in chosen.itertuples():
        pool = reviews[reviews["itemId"] == record.itemId]
        pool = pool.sort_values("reviewId")
        picked = rng.choice(len(pool), size=PER_ALTERNATIVE, replace=False)
        part = pool.iloc[np.sort(picked)].copy()
        part["ENTITY_NAME"] = record.name
        parts.append(part)
    sample = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(
        {
            "REVIEW_BODY": sample["text"].astype(str).str.strip(),
            "USER_ID": sample["userId"].astype(str),
            "ENTITY_ID": sample["itemId"].astype(str),
            "ENTITY_NAME": sample["ENTITY_NAME"],
            "REVIEW_RATING_GLOBAL": sample["stars"].astype(int),
            "reviewId": sample["reviewId"].astype(str),
            "date": sample["date"].astype(str),
            "priceInterval": chosen["priceInterval"].iloc[0],
        }
    )


def adjacent_margins(chosen: pd.DataFrame) -> pd.DataFrame:
    ordered = chosen.sort_values("mean_stars", ascending=False).reset_index(drop=True)
    gaps = -ordered["mean_stars"].diff()
    ordered["gap_to_previous"] = gaps.round(3)
    return ordered[["name", "mean_stars", "gap_to_previous"]]


def markdown(frame: pd.DataFrame) -> str:
    """A pipe table, without pulling in `tabulate` for one manifest."""

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False)
    ]
    return "\n".join([head, rule, *body])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, help="Unzipped Madrid directory.")
    parser.add_argument("--out", type=Path, default=Path("data/UniOvi-Madrid-600.csv"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/UniOvi-Madrid-600.MANIFEST.md")
    )
    parser.add_argument("--zip", type=Path, default=None, help="Madrid.zip, hashed.")
    parser.add_argument("--candidates-only", action="store_true")
    args = parser.parse_args()

    raw_reviews = pd.read_pickle(args.source / "reviews.pkl")
    raw_items = pd.read_pickle(args.source / "items.pkl")
    reviews, items = load(args.source)
    eligible = eligible_reviews(reviews)
    chosen = candidates(eligible, items)
    pd.set_option("display.width", 200)
    print(chosen.to_string(index=False))
    print()
    print(adjacent_margins(chosen).to_string(index=False))
    if args.candidates_only:
        return

    sample = draw(eligible, chosen)
    sample.to_csv(args.out, index=False)
    repeat_users = sample["USER_ID"].value_counts()
    repeat_users = repeat_users[repeat_users > 1]
    same_alt = sample.groupby(["USER_ID", "ENTITY_ID"]).size()
    same_alt = same_alt[same_alt > 1]

    band = chosen["priceInterval"].iloc[0]
    lines = [
        "# UniOvi Madrid replication corpus (600 reviews)",
        "",
        f"Generated {date.today().isoformat()} by `data/sample_uniovi.py`.",
        "",
        "Source: TripAdvisor Restaurant Reviews v2.0.0, Universidad de Oviedo,",
        "Zenodo doi:10.5281/zenodo.14622324, CC BY 4.0, file `Madrid.zip`.",
        "",
        "## Protocol",
        "",
        "0. Exact duplicate rows of the release dropped: one copy per review id",
        "   and per item id.",
        f"1. English reviews (`language == 'en'`), at least {MIN_TOKENS} tokens,",
        "   rating a multiple of 10 in 10-50; stars = rating / 10.",
        f"2. Restaurants with at least {MIN_REVIEWS} eligible reviews, in the most",
        f"   populated non-empty `priceInterval` among them (`{band}`).",
        f"3. The {N_ALTERNATIVES} with the most eligible reviews (ties by id).",
        f"4. {PER_ALTERNATIVE} reviews each, uniformly without replacement,",
        f"   seed {SEED}.",
        "5. Rows ordered by alternative, then review id; row i is `review-<i>`.",
        "",
        "## Counts",
        "",
        (
            f"- Reviews in the source: {len(raw_reviews):,}"
            f" ({len(raw_reviews) - len(reviews):,} duplicated review ids dropped);"
            f" items: {len(raw_items):,}"
            f" ({len(raw_items) - len(items):,} duplicated item ids dropped)"
        ),
        f"- Eligible after step 1: {len(eligible):,}",
        f"- Sampled: {len(sample)} over {sample['ENTITY_ID'].nunique()} alternatives",
        (
            f"- Distinct users: {sample['USER_ID'].nunique()}; users with more"
            f" than one review in the sample: {len(repeat_users)}"
            f" ({int(repeat_users.sum())} reviews); user-alternative pairs with"
            f" more than one review: {len(same_alt)}"
        ),
        "",
        "## Alternatives (eligible pool, before sampling)",
        "",
        markdown(chosen),
        "",
        "## Sample star distribution",
        "",
        markdown(
            sample.groupby("ENTITY_NAME", sort=False)["REVIEW_RATING_GLOBAL"]
            .agg(["count", "mean"])
            .round(3)
            .reset_index()
        ),
        "",
        "## Hashes (SHA-256)",
        "",
        f"- `{args.out.name}`: `{sha256(args.out)}`",
    ]
    if args.zip and args.zip.is_file():
        lines.append(f"- `{args.zip.name}`: `{sha256(args.zip)}`")
    lines.append("")
    args.manifest.write_text("\n".join(lines))
    print(f"\nwrote {args.out} ({len(sample)} rows) and {args.manifest}")


if __name__ == "__main__":
    main()
