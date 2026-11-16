"""Checks of the aggregation and resampling pipeline.

Pins the conventions that are easy to get wrong and that every reported CDM
number depends on: the rounding rule inside Delta, the domain conversion
between the normalized [-1, 1] values the rest of the pipeline uses and the
term-index domain the 2-tuple model is defined on, the review-to-alternative
join, and what actually separates the aggregation operators.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.cdm import (
    alternative_betas,
    attach_entities,
    expert_betas,
    rank_alternatives,
    review_position,
    scale_for,
)
from utils.fusion import OPERATORS, apply_operator, fuse_models, fuse_runs
from utils.parsing import normalize_label5, normalize_label7
from utils.profiles import load_profile
from utils.runs import load_runs
from utils.stability import (
    build_panels,
    crowd_stability,
    decision_stability,
    discriminance,
    joint_stability,
    mapping_sensitivity,
    operator_scope_contrast,
    ranks_of,
    wilson_interval,
)
from utils.temperature import contrast
from utils.twotuple import (
    SCALE_5,
    SCALE_7,
    TwoTuple,
    beta_from_y,
    delta,
    delta_inv,
    format_tuple,
    two_tuple_mean,
    y_from_beta,
)
from utils.validity import (
    review_agreement,
    star_alternatives,
    star_frame,
)


def _runs(name: str) -> pd.DataFrame:
    profile = load_profile(name)
    return attach_entities(load_runs(profile), profile.data_csv)


def _stars() -> pd.DataFrame:
    return star_frame(load_profile("london").data_csv)


@pytest.fixture(scope="module")
def cdm_runs():
    """The London corpus: six systems, three schemes at T = 1."""

    return _runs("london")


@pytest.fixture(scope="module")
def t0_runs():
    """The London greedy collection: LABEL-5, five local systems."""

    return _runs("london_t0")


@pytest.fixture(scope="module")
def runs(cdm_runs):
    return cdm_runs


# -- the 2-tuple core ------------------------------------------------------


def test_alpha_stays_in_the_half_open_interval():
    """The model defines alpha on [-0.5, 0.5), and the boundary is the point.

    Python's round is half-to-even: at beta = 2.5 it returns 2, which would
    make alpha = +0.5 and put the tuple outside its own domain. Delta rounds
    halves up instead, so the midpoint lands on the upper term.
    """

    for beta in np.arange(0.0, 4.0001, 0.05):
        tuple_ = delta(float(beta), SCALE_5)
        assert -0.5 <= tuple_.alpha < 0.5

    assert delta(2.5, SCALE_5) == TwoTuple(index=3, alpha=-0.5)
    assert delta(1.5, SCALE_5) == TwoTuple(index=2, alpha=-0.5)
    assert round(delta(2.5, SCALE_5).index) != round(2.5)


def test_delta_and_its_inverse_round_trip():
    for beta in np.arange(0.0, 6.0001, 0.1):
        assert delta_inv(delta(float(beta), SCALE_7)) == pytest.approx(float(beta))


def test_two_tuple_rejects_an_out_of_domain_alpha():
    with pytest.raises(ValueError, match=r"\[-0.5, 0.5\)"):
        TwoTuple(index=2, alpha=0.5)


def test_domain_conversion_inverts_the_parsing_normalization():
    """beta_from_y must undo exactly what utils.parsing did to build y."""

    for ordinal in range(SCALE_5.g + 1):
        y = normalize_label5(ordinal)
        assert beta_from_y(y, SCALE_5) == pytest.approx(ordinal)
    for ordinal in range(SCALE_7.g + 1):
        y = normalize_label7(ordinal)
        assert beta_from_y(y, SCALE_7) == pytest.approx(ordinal)

    for y in (-1.0, -0.5, 0.0, 0.25, 1.0):
        assert y_from_beta(beta_from_y(y, SCALE_7), SCALE_7) == pytest.approx(y)


def test_term_set_runs_negative_to_positive():
    """s_0 is the most negative term, matching parsing's ordinal convention."""

    assert SCALE_5.terms[0] == "Very Negative"
    assert SCALE_5.terms[SCALE_5.g] == "Very Positive"
    assert SCALE_7.terms[0] == "Extremely Negative"
    assert format_tuple(two_tuple_mean([3.0, 3.5], SCALE_5), SCALE_5) == (
        "(Positive, +0.25)"
    )


# -- the review -> alternative join ----------------------------------------


def test_reviews_join_to_alternatives_by_position(runs):
    """474 review folders, four alternatives, at the dataset's own counts."""

    assert review_position("review-473") == 473
    assert runs["review_id"].nunique() == 474

    counts = (
        runs[["review_id", "alternative"]]
        .drop_duplicates()["alternative"]
        .value_counts()
        .to_dict()
    )
    assert counts == {
        "The Ivy": 148,
        "The Wolseley": 134,
        "J. Sheekey": 97,
        "Oxo Tower Restaurant, Bar and Brasserie": 95,
    }


def test_a_mismatched_source_file_is_refused(runs, tmp_path):
    """The join is by position, so the wrong CSV silently reassigns reviews.

    attach_entities cross-checks the uid the runner recorded and raises, which
    is the only thing standing between a mixed-up input and a plausible-looking
    ranking of the wrong restaurants.
    """

    data_csv = load_profile("london").data_csv
    shuffled = pd.read_csv(data_csv).iloc[::-1].reset_index(drop=True)
    path = tmp_path / "reversed.csv"
    shuffled.to_csv(path, index=False)

    bare = runs.drop(columns=["user_id", "entity_id", "alternative"])
    with pytest.raises(ValueError, match="collected against a different input"):
        attach_entities(bare, path)


def test_the_two_hierarchies_weight_experts_differently(runs):
    """Three-level averages within expert first, so repeat reviewers count once."""

    reviews = fuse_runs(runs[runs["scheme"] == "LABEL-5"])
    experts = expert_betas(reviews)

    two = alternative_betas(reviews, "two_level").set_index(["model", "alternative"])
    three = alternative_betas(reviews, "three_level").set_index(
        ["model", "alternative"]
    )

    # Fewer experts than reviews: some experts reviewed one restaurant twice.
    assert three["n"].sum() < two["n"].sum()
    assert len(experts) == three["n"].sum()
    # And that changes the collective value, or the hierarchy would be moot.
    assert not np.allclose(two["beta"], three["beta"])


def test_ranks_are_dense_and_ordered(runs):
    reviews = fuse_runs(runs[runs["scheme"] == "LABEL-5"])
    ranked = rank_alternatives(alternative_betas(reviews, "two_level"))
    for _, g in ranked.groupby(["model", "scheme"]):
        assert sorted(g["rank"]) == [1, 2, 3, 4]
        assert list(g["beta"]) == sorted(g["beta"], reverse=True)


# -- what separates the operators ------------------------------------------


def test_two_tuple_aggregation_is_the_mean_on_the_index_domain():
    """Stated plainly because the paper must not overclaim.

    On beta the 2-tuple mean *is* the arithmetic mean. What it adds is that the
    result is not snapped back onto a term.
    """

    betas = [1.0, 2.0, 2.0, 3.0]
    assert OPERATORS["two_tuple"](betas, SCALE_5) == pytest.approx(2.0)
    assert delta_inv(two_tuple_mean(betas, SCALE_5)) == pytest.approx(2.0)


def test_snapping_operators_land_on_terms_and_the_lossless_one_does_not():
    betas = [2.0, 2.0, 3.0, 3.0, 3.0]
    assert OPERATORS["two_tuple"](betas, SCALE_5) == pytest.approx(2.6)
    for operator in ("mean_round", "median_round", "majority"):
        value = OPERATORS[operator](betas, SCALE_5)
        assert value == int(value), operator


def test_majority_breaks_ties_towards_the_lower_term():
    """Ties are common at 10 runs, so the rule cannot depend on run order."""

    assert OPERATORS["majority"]([1.0, 1.0, 3.0, 3.0], SCALE_5) == 1.0
    assert OPERATORS["majority"]([3.0, 3.0, 1.0, 1.0], SCALE_5) == 1.0


def test_snapping_collapses_the_corpus_onto_g_plus_one_values(runs):
    """The measurable cost of snapping, and the reason the study reports it."""

    label5 = runs[runs["scheme"] == "LABEL-5"]
    lossless = discriminance(label5, "two_tuple").set_index("model")
    for operator in ("mean_round", "median_round", "majority"):
        snapped = discriminance(label5, operator).set_index("model")
        for model in snapped.index:
            assert snapped.loc[model, "distinct_values"] == SCALE_5.g + 1
            assert lossless.loc[model, "distinct_values"] > SCALE_5.g + 1


def test_fusion_pools_only_the_requested_models(runs):
    reviews = fuse_runs(runs[runs["scheme"] == "LABEL-5"])
    members = ["Gemma 4 E4B", "Ministral 3 8B"]
    fused = fuse_models(reviews, "two_tuple", members, "Fusion")

    assert set(fused["model"]) == {"Fusion"}
    assert fused["review_id"].nunique() == 474
    assert set(fused["n_models"]) == {len(members)}

    expected = (
        reviews[reviews["model"].isin(members)]
        .groupby("review_id", sort=False)["beta"]
        .mean()
    )
    merged = fused.set_index("review_id")["beta"]
    assert np.allclose(merged.loc[expected.index], expected)


# -- stability -------------------------------------------------------------


def test_decision_stability_is_deterministic_under_a_seed(runs):
    label5 = runs[(runs["scheme"] == "LABEL-5") & (runs["model"] == "GPT-5.2")]
    first = decision_stability(label5, replicas=20, seed=7)
    second = decision_stability(label5, replicas=20, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_discriminance_reports_signal_beside_noise(runs):
    """A ratio alone cannot detect collapse; between_sd is what exposes it."""

    disc = discriminance(runs[runs["scheme"] == "LABEL-5"], "two_tuple", replicas=100)
    assert (disc["between_sd"] > 0).all()
    assert (disc["input_sd"] > 0).all()
    assert (disc["residual_sd"] > 0).all()
    assert ((disc["signal_share"] > 0) & (disc["signal_share"] < 1)).all()


def test_input_noise_is_identical_across_operators_and_residual_noise_is_not(runs):
    """The correction behind Section V-D: which noise term is compared decides
    the answer, and only one of the two is a property of the operator."""

    label5 = runs[runs["scheme"] == "LABEL-5"]
    frames = {
        op: discriminance(label5, op, replicas=100).set_index("model")
        for op in ("two_tuple", "mean_round", "majority")
    }

    lossless = frames["two_tuple"]
    for operator in ("mean_round", "majority"):
        other = frames[operator]
        # Handed the same runs, so identical by construction.
        assert lossless["input_sd"].equals(other["input_sd"])
        # Snapping turns a small continuous perturbation into a whole-term one,
        # so what survives the operator is strictly larger.
        assert (other["residual_sd"] > lossless["residual_sd"]).all()
        assert (other["distinct_values"] < lossless["distinct_values"]).all()


def test_mapping_sensitivity_leaves_score_alone(runs):
    """SCORE outputs do not sit on term indices, so placing terms is undefined."""

    mapping = mapping_sensitivity(runs, "two_tuple")
    assert "SCORE" not in set(mapping["scheme"])
    assert set(mapping["placement"]) == {"uniform", "compressed", "stretched"}
    uniform = mapping[mapping["placement"] == "uniform"]
    assert (uniform["tau_vs_uniform"] == 1.0).all()


def test_scale_for_maps_each_ordinal_scheme_to_its_term_set():
    assert scale_for("LABEL-5") is SCALE_5
    assert scale_for("LABEL-7") is SCALE_7
    # SCORE has no term set of its own; it is expressed on the 5-point grid.
    assert scale_for("SCORE") is SCALE_5


# -- the parsing policy -----------------------------------------------------


def test_extraction_prefers_the_longest_label():
    """`experiments.parsing` matches labels in vocabulary order, so `positive`
    hits inside `very positive` and the intensifier is dropped. Stage 2 here
    matches longest first.
    """

    from utils.parsing import LABEL_5_VALUES, LABEL_7_VALUES, extract_label

    for raw in (
        "The sentiment of this text is **very_positive**.",
        "The sentiment of this text is very positive.",
        "Very Positive",
        "very-positive",
    ):
        assert extract_label(raw, LABEL_7_VALUES) == "very_positive", raw

    assert (
        extract_label("The sentiment is **extremely_positive**.", LABEL_7_VALUES)
        == "extremely_positive"
    )
    assert extract_label("**Sentiment:** **very_positive**", LABEL_5_VALUES) == (
        "very_positive"
    )
    assert extract_label(r"very\_positive", LABEL_5_VALUES) == "very_positive"


def test_extraction_still_rejects_out_of_vocabulary_replies():
    """Unwrapping formatting is not the same as guessing a label."""

    from utils.parsing import LABEL_5_VALUES, extract_label

    for raw in ("excellent", "mixed", "premium", "exciting", "delightful"):
        assert extract_label(raw, LABEL_5_VALUES) is None, raw

    # A word that merely starts with a label is not that label.
    assert extract_label("positively awful", LABEL_5_VALUES) is None


# -- missing outputs --------------------------------------------------------


def test_rejections_are_not_missing_at_random(cdm_runs):
    """Pins the bias that makes the missing-data policy a reported choice.

    Ministral 3 8B's LABEL-5 rejections are concentrated on positive vocabulary
    and land unevenly across the alternatives, so dropping those runs depresses
    some alternatives more than others. If this ever stops holding, the
    sensitivity analysis in the paper needs revisiting rather than deleting.
    """

    from utils.stability import rejection_bias

    bias = rejection_bias(cdm_runs).set_index(["model", "scheme"])
    key = ("Ministral 3 8B", "LABEL-5")
    assert key in bias.index

    row = bias.loc[key]
    assert row["mean_y_affected"] > row["mean_y_rest"], "not skewed positive"
    assert row["max_alt_rate"] > 1.5 * row["min_alt_rate"], "rate is even"


def test_missing_policy_changes_one_system_and_not_the_fusion(cdm_runs):
    """The margin under `available` is not a property of the model.

    It moves by more than an order of magnitude under `listwise` (0.0008 to
    0.0461 since the parser rejects replies naming two categories), which is
    why the paper reports both and declines to read the first as a
    measurement.
    """

    from utils.cdm import alternative_betas, rank_alternatives
    from utils.fusion import fuse_models, fuse_runs
    from utils.stability import apply_missing_policy, missing_data_sensitivity

    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    sens = missing_data_sensitivity(label5).set_index(["model", "policy"])

    available = sens.loc[("Ministral 3 8B", "available"), "margin_min"]
    listwise = sens.loc[("Ministral 3 8B", "listwise"), "margin_min"]
    assert listwise > 10 * available

    # Systems without rejections must be untouched by the policy.
    for model in ("Gemma 4 E4B", "GPT-5.2"):
        assert (
            sens.loc[(model, "available"), "order"]
            == (sens.loc[(model, "listwise"), "order"])
        )

    # The fused ranking survives the choice; that is the paper's claim.
    members = [m for m in cdm_runs["model"].unique() if m != "GPT-5.2"]
    orders = []
    for policy in ("available", "listwise"):
        filtered = apply_missing_policy(label5, policy)
        fused = fuse_models(fuse_runs(filtered), "two_tuple", members, "Fusion")
        ranked = rank_alternatives(alternative_betas(fused, "two_level"))
        orders.append(list(ranked.sort_values("rank")["alternative"]))
    assert orders[0] == orders[1], "fusion is not invariant to the missing-data policy"


# -- the resampling core ----------------------------------------------------


def test_vectorized_operators_agree_with_the_scalar_ones():
    """The resampling analyses use the vectorized path exclusively. If it ever
    parts company with the definitions in `utils.fusion`, every stability and
    discriminance number in the manuscript is wrong and nothing else would
    catch it."""

    rng = np.random.default_rng(7)
    for scale in (SCALE_5, SCALE_7):
        # Integer draws on purpose: they make ties common, which is where the
        # majority operator's tie rule has to be pinned.
        samples = rng.integers(0, scale.g + 1, (400, 10)).astype(float)
        for operator, fn in OPERATORS.items():
            vectorized = apply_operator(samples, operator, scale)
            scalar = np.array([fn(row, scale) for row in samples])
            assert np.allclose(vectorized, scalar), (operator, scale.g)


def test_weight_matrix_reproduces_the_dataframe_aggregation(cdm_runs):
    """Both hierarchies are weighted means, so the resampling core replaces
    them with one matrix multiply. It has to give the same collective values
    and the same ranks as `utils.cdm`, including the tie-breaking."""

    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    for operator in OPERATORS:
        reviews = fuse_runs(label5, operator)
        for hierarchy in ("two_level", "three_level"):
            reference = rank_alternatives(alternative_betas(reviews, hierarchy))
            for panel in build_panels(label5):
                values = panel.full_values(operator) @ panel.weight_matrix(hierarchy).T
                rows = reference[
                    (reference["model"] == panel.model)
                    & (reference["scheme"] == panel.scheme)
                ].set_index("alternative")
                names = panel.names
                assert np.allclose(values, [rows.loc[n, "beta"] for n in names])
                assert np.array_equal(
                    ranks_of(values)[0], [rows.loc[n, "rank"] for n in names]
                )


def test_wilson_interval_stays_inside_the_unit_interval():
    """The normal approximation goes past 1 and collapses to zero width at
    100%, which is exactly where the reported shares sit."""

    lo, hi = wilson_interval(2000, 2000)
    assert 0.0 < lo < 1.0 and hi <= 1.0
    lo, hi = wilson_interval(0, 2000)
    assert lo >= 0.0 and 0.0 < hi < 1.0
    lo, hi = wilson_interval(1000, 2000)
    assert lo < 0.5 < hi


# -- where the operator acts ------------------------------------------------


def test_snapping_at_every_level_dissolves_the_ranking(cdm_runs):
    """Section V-E. An operator whose output is a term cannot express a
    difference smaller than a term, and every difference here is smaller."""

    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    scope = operator_scope_contrast(
        {op: fuse_runs(label5, op) for op in ("two_tuple", "mean_round", "majority")}
    )

    lossless = scope[scope["operator"] == "two_tuple"]
    assert (lossless["distinct_values"] == 4).all()
    assert (lossless["ties"] == 0).all()
    # The 2-tuple operator never snaps, so scope cannot change its result.
    review = lossless[lossless["scope"] == "review"].set_index("model")
    every = lossless[lossless["scope"] == "all_levels"].set_index("model")
    assert review["margin_min"].equals(every["margin_min"])

    for operator in ("mean_round", "majority"):
        collapsed = scope[
            (scope["operator"] == operator) & (scope["scope"] == "all_levels")
        ]
        assert (collapsed["distinct_values"] == 1).all()
        assert (collapsed["margin_min"] == 0).all()
        # ... while the reported pipeline, which averages above the review,
        # still separates all four.
        kept = scope[(scope["operator"] == operator) & (scope["scope"] == "review")]
        assert (kept["distinct_values"] == 4).all()


# -- the two sources of uncertainty -----------------------------------------


def test_crowd_resampling_is_a_different_quantity_from_run_resampling(cdm_runs):
    """Section V-C, the finding that reframes the whole stability claim: the
    crowd moves the ranking further than the models' own inference noise
    does."""

    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    runs_side = decision_stability(label5, "two_tuple", replicas=200).set_index("model")
    crowd_side = crowd_stability(
        fuse_runs(label5, "two_tuple"), replicas=200
    ).set_index("model")

    assert (crowd_side["top1_pct"] < runs_side["top1_pct"]).all()
    assert (crowd_side["top1_pct"] < 90).all()
    # The signed margin has to be able to go negative, or the interval says
    # nothing about whether a pair inverted.
    assert (crowd_side["margin_lo"] < 0).all()
    assert (crowd_side["inverts_pct"] > 0).all()


def test_crowd_resampling_clusters_an_expert_reviews(cdm_runs):
    """Reviews by one person are not independent draws. Drawing them
    separately would understate the uncertainty, so the count of experts, not
    of reviews, is what the bootstrap ranges over."""

    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    reviews = fuse_runs(label5, "two_tuple")
    crowd = crowd_stability(reviews, replicas=50)
    assert (crowd["experts"] == reviews["user_id"].nunique()).all()
    assert crowd["experts"].iloc[0] < reviews["review_id"].nunique()


# -- panel composition ------------------------------------------------------


def test_decision_stability_exports_the_reference_it_is_measured_against(cdm_runs):
    """Top-1 is only readable next to the winner it counts and the lead that
    winner holds; both must be the full-data ranking's own."""

    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    stab = decision_stability(label5, "two_tuple", replicas=20).set_index("model")
    reviews = fuse_runs(label5, "two_tuple")
    for model, g in reviews.groupby("model"):
        betas = g.groupby("alternative", sort=False)["beta"].mean()
        ordered = betas.sort_values(ascending=False)
        row = stab.loc[model]
        assert row["reference_winner"] == ordered.index[0]
        lead = ordered.iloc[0] - ordered.iloc[1]
        assert row["margin_top"] == pytest.approx(lead, abs=1e-5)
        adjacent = [row[f"margin_adj_{i}"] for i in range(1, len(ordered))]
        assert row["margin_min"] == pytest.approx(min(adjacent), abs=1e-5)


def test_mapping_sensitivity_fusion_row_matches_the_fused_ranking(cdm_runs):
    """Under uniform placement the fused row must reproduce the fused system
    that every other table reports."""

    members = [
        "Gemma 4 E4B",
        "Ministral 3 8B",
        "Gemma 4 31B",
        "Ornith 35B",
        "Qwen 3.6 35B",
    ]
    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    mapping = mapping_sensitivity(label5, "two_tuple", "two_level", members, "F")
    fused_rows = mapping[mapping["model"] == "F"].set_index("placement")
    assert set(fused_rows.index) == {"uniform", "compressed", "stretched"}

    reviews = fuse_runs(label5, "two_tuple")
    fused = fuse_models(reviews, "two_tuple", members, "F")
    betas = fused.groupby("alternative", sort=False)["beta"].mean()
    expected = " > ".join(betas.sort_values(ascending=False).index)
    assert fused_rows.loc["uniform", "order"] == expected
    assert fused_rows.loc["uniform", "tau_vs_uniform"] == 1.0


def test_joint_resampling_agrees_with_the_single_source_analyses(cdm_runs):
    """`joint_stability` computes the three conditions with one procedure. Its
    `runs` and `experts` conditions must agree with `decision_stability` and
    `crowd_stability`, which resample the same things with their own code, up
    to Monte-Carlo error; and holding nothing fixed cannot be steadier than
    holding the runs fixed by more than that error."""

    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    label5 = label5[label5["model"].isin(["Gemma 4 E4B", "Gemma 4 31B"])]
    summary, intervals = joint_stability(label5, "two_tuple", 3000, 11)
    joint = summary.set_index(["model", "condition"])

    runs_side = decision_stability(label5, "two_tuple", replicas=3000)
    runs_side = runs_side.set_index("model")
    crowd_side = crowd_stability(fuse_runs(label5, "two_tuple"), replicas=3000)
    crowd_side = crowd_side.set_index("model")
    for model in ("Gemma 4 E4B", "Gemma 4 31B"):
        assert joint.loc[(model, "runs"), "top1_pct"] == pytest.approx(
            runs_side.loc[model, "top1_pct"], abs=3.0
        )
        assert joint.loc[(model, "experts"), "top1_pct"] == pytest.approx(
            crowd_side.loc[model, "top1_pct"], abs=3.0
        )
        assert joint.loc[(model, "both"), "top1_pct"] <= (
            joint.loc[(model, "runs"), "top1_pct"] + 3.0
        )

    # Every interval brackets its full-data value.
    assert (intervals["beta_lo"] <= intervals["beta"] + 1e-9).all()
    assert (intervals["beta"] <= intervals["beta_hi"] + 1e-9).all()

    # Seeded.
    again, _ = joint_stability(label5, "two_tuple", 3000, 11)
    pd.testing.assert_frame_equal(summary, again)


def test_budget_enumerates_every_subset_and_ten_runs_is_the_full_system(cdm_runs):
    """k-subsets are enumerated, not sampled, and the k = 10 row is the system
    the other tables report: same winner, same lead."""

    from utils.budget import budget_curve

    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    members = ["Gemma 4 31B", "Ornith 35B", "Qwen 3.6 35B"]
    systems = {
        "Ornith 35B": label5[label5["model"] == "Ornith 35B"],
        "F": label5[label5["model"].isin(members)],
    }
    curve = budget_curve(
        systems,
        {"Ornith 35B": "J. Sheekey", "F": "J. Sheekey"},
        {m: 0.2 for m in members},
        {"Ornith 35B": ["Ornith 35B"], "F": members},
        budgets=(1, 2, 5, 10),
        replicas=200,
    ).set_index(["system", "k"])

    assert list(curve.loc["F", "subsets"]) == [10, 45, 252, 1]
    assert curve.loc[("F", 10), "cost_s_per_review"] == pytest.approx(6.0)

    fused = fuse_models(fuse_runs(label5, "two_tuple"), "two_tuple", members, "F")
    betas = fused.groupby("alternative")["beta"].mean().sort_values(ascending=False)
    lead = betas.iloc[0] - betas.iloc[1]
    assert curve.loc[("F", 10), "margin_top"] == pytest.approx(lead, abs=1e-5)
    assert curve.loc[("F", 10), "winner_ok_pct"] == 100.0


def test_snapping_caps_review_resolution_but_not_collective_resolution(cdm_runs):
    """A snapping operator leaves at most g+1 review-level values, but the
    levels above the review average them, so the alternatives still come out
    distinct. The paper used to say the opposite."""

    label5 = cdm_runs[
        (cdm_runs["scheme"] == "LABEL-5") & (cdm_runs["model"] == "Gemma 4 31B")
    ]
    for operator in ("majority", "mean_round"):
        row = discriminance(label5, operator, replicas=20).iloc[0]
        assert row["distinct_values"] <= 5
        assert row["distinct_collective"] == label5["alternative"].nunique()


# -- convergent validity against the authors' own ratings -------------------


def test_star_ratings_align_one_to_one_with_the_five_point_term_set():
    """The whole comparison rests on this correspondence: a k-star rating is
    the term at index k-1, so one star is one term and a bias in beta units is
    readable. If the dataset's scale ever changed, every number in the
    agreement table would silently shift."""

    stars = _stars()
    assert set(stars["rating"].unique()) <= set(range(1, 6))
    assert stars["beta_star"].min() == 0.0
    assert stars["beta_star"].max() == float(SCALE_5.g)
    assert (stars["beta_star"] == stars["rating"] - 1).all()
    # And the normalized domain must agree with what parsing produces.
    for rating in range(1, 6):
        row = stars[stars["rating"] == rating].iloc[0]
        assert row["y_star"] == pytest.approx(normalize_label5(rating - 1))


def test_the_pipeline_tracks_the_authors_own_ratings(cdm_runs):
    """The one external check in the study. Not an accuracy claim: it says the
    collective evaluations are made of something the crowd recognises."""

    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    agreement = review_agreement(fuse_runs(label5, "two_tuple"), _stars())
    agreement = agreement[agreement["scheme"] == "LABEL-5"]

    assert (agreement["pearson_r"] > 0.7).all()
    assert (agreement["n"] == 474).all()
    assert (agreement["domain"] == "beta (5-point)").all()
    # Every system reads a review below its author, one of them far below.
    assert (agreement["bias_beta"] < 0).all()
    worst = agreement.loc[agreement["bias_beta"].idxmin()]
    assert worst["model"] == "Ministral 3 8B"


def test_the_star_ranking_does_not_resolve_its_own_top_pair():
    """Section V-G rests on this: the crowd's own ratings separate the top two
    alternatives by about the same margin the fused system does, so the two
    disagreeing about the winner is not a disagreement about anything."""

    ranked = star_alternatives(_stars())
    assert list(ranked["rank"]) == [1, 2, 3, 4]
    gap = ranked["beta"].iloc[0] - ranked["beta"].iloc[1]
    assert 0 < gap < 0.05


# -- sampling temperature ---------------------------------------------------


def test_greedy_decoding_removes_run_variability_but_not_the_collapse(
    cdm_runs, t0_runs
):
    """Section V-H, the result that keeps the paper's premise honest: greedy
    decoding removes almost all run-to-run disagreement and changes neither the
    collapse onto one term nor the margins."""

    sampled = cdm_runs[
        (cdm_runs["scheme"] == "LABEL-5") & (cdm_runs["model"] != "GPT-5.2")
    ]
    shared = sorted(set(sampled["model"]) & set(t0_runs["model"]))
    table = contrast(
        sampled[sampled["model"].isin(shared)],
        t0_runs[t0_runs["model"].isin(shared)],
    ).set_index("model")

    members = table.loc[shared]
    assert (members["within_sd_t0"] < 0.01).all()
    assert (members["within_sd_t0"] < members["within_sd_t1"]).all()
    assert (members["identical_pct_t0"] > 98).all()

    # The decision-level finding survives: the four alternatives still share a
    # single linguistic term almost everywhere.
    assert (table["terms_t1"] == 1).all()
    assert (table["terms_t0"] <= 2).all()
    # And the margins stay in the same range rather than opening up.
    assert table["margin_t0"].max() < 0.1


def test_greedy_decoding_costs_resolution(cdm_runs, t0_runs):
    """The counter-intuitive half of the contrast: when the ten runs agree
    their mean lands back on a term index, so sampling is also what gives the
    per-review aggregate its granularity."""

    sampled = cdm_runs[
        (cdm_runs["scheme"] == "LABEL-5") & (cdm_runs["model"] != "GPT-5.2")
    ]
    shared = sorted(set(sampled["model"]) & set(t0_runs["model"]))
    table = contrast(
        sampled[sampled["model"].isin(shared)],
        t0_runs[t0_runs["model"].isin(shared)],
    )
    assert (table["distinct_t0"] < table["distinct_t1"]).all()


def test_the_temperature_contrast_is_paired(cdm_runs, t0_runs):
    """Rejected outputs do not fall in the same places at the two temperatures,
    and greedy decoding empties whole reviews rather than thinning them. Scored
    on unequal review sets, Ministral's T=1 margin reads 0.00014 instead of
    0.00722 -- a factor of fifty, and the same missing-not-at-random mechanism
    Section V-I documents."""

    sampled = cdm_runs[
        (cdm_runs["scheme"] == "LABEL-5") & (cdm_runs["model"] != "GPT-5.2")
    ]
    shared = sorted(set(sampled["model"]) & set(t0_runs["model"]))
    table = contrast(
        sampled[sampled["model"].isin(shared)],
        t0_runs[t0_runs["model"].isin(shared)],
    ).set_index("model")

    # Every system is scored on one review set at both temperatures.
    for model in shared:
        assert table.loc[model, "reviews_t1"] == table.loc[model, "reviews_t0"]
    # Only Ministral loses anything, and the pairing says how much.
    assert table.loc["Ministral 3 8B", "dropped"] == 8
    assert (table.drop(index="Ministral 3 8B")["dropped"] == 0).all()
    assert table.loc["Ministral 3 8B", "margin_t1"] > 0.005


def test_the_lossless_benefit_belongs_to_the_mean_not_to_the_representation(
    cdm_runs,
):
    """Section IV-D's attribution. `two_tuple` and `continuous_mean` are the
    same arithmetic; every stability, margin, resolution and residual-noise
    figure credited to the 2-tuple operator is produced by a plain continuous
    mean. The 2-tuple model contributes the linguistic rendering, not the
    numbers, and the paper must not claim otherwise."""

    for panel in build_panels(cdm_runs):
        lossless = panel.full_values("two_tuple")
        plain = panel.full_values("continuous_mean")
        assert np.array_equal(lossless, plain)

    rng = np.random.default_rng(0)
    for scale in (SCALE_5, SCALE_7):
        samples = rng.integers(0, scale.g + 1, (200, 10)).astype(float)
        assert np.array_equal(
            apply_operator(samples, "two_tuple", scale),
            apply_operator(samples, "continuous_mean", scale),
        )

    # And it is the rendering that differs: the same beta is a bare number one
    # way and a linguistic pair the other.
    assert format_tuple(delta(3.17, SCALE_5), SCALE_5) == "(Positive, +0.17)"


# -- replication plumbing ----------------------------------------------------


def test_worked_example_is_the_widest_disagreement():
    """The example review is chosen by rule: largest range of review-level
    beta across systems, ties by spread then position."""

    from utils.worked_example import choose

    reviews = pd.DataFrame(
        {
            "review_id": ["review-0", "review-0", "review-1", "review-1"],
            "model": ["A", "B", "A", "B"],
            "beta": [2.0, 2.5, 0.5, 3.5],
        }
    )
    assert choose(reviews) == "review-1"


# -- one Top-1, one abstention rule -----------------------------------------


def _synthetic_reviews(betas: dict[str, list[float]], model: str = "M") -> pd.DataFrame:
    """A review-level frame: one expert per review, alternatives in order."""

    rows = []
    for alternative, values in betas.items():
        for i, beta in enumerate(values):
            rows.append(
                {
                    "model": model,
                    "scheme": "LABEL-5",
                    "review_id": f"{alternative}-{i}",
                    "user_id": f"u-{alternative}-{i}",
                    "alternative": alternative,
                    "beta": beta,
                }
            )
    return pd.DataFrame(rows)


def test_top1_counts_a_tie_for_first_as_kept_in_every_module():
    """`stability` and `budget` once disagreed on a replica that ties the
    winner: one read the replica's winner off frame order, the other asked
    whether anything strictly beat it. Both now use `keeps_winner`."""

    from utils.stability import _replica_summary, keeps_winner

    # Reference: B wins. Replicas: B alone first; A and B tied first (A comes
    # first in frame order); C strictly first.
    values = np.array([[1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [1.0, 1.0, 3.0]])
    reference = ranks_of(np.array([1.0, 2.0, 0.0]))[0]
    kept = keeps_winner(ranks_of(values), 1)
    assert kept.tolist() == [True, True, False]
    # The budget module's former rule, written out: nothing strictly greater.
    assert kept.tolist() == list(values.max(axis=1) <= values[:, 1])
    assert _replica_summary(ranks_of(values), reference)["top1_pct"] == pytest.approx(
        100.0 * 2 / 3, abs=0.1
    )


def test_candidate_set_is_the_winner_alone_when_every_rival_is_separated():
    from utils.stability import candidate_sets, draw_crowd, rival_gaps

    reviews = _synthetic_reviews(
        {"A": [4.0] * 20, "B": [3.0] * 20, "C": [2.0] * 20, "D": [1.0] * 20}
    )
    draws = draw_crowd(reviews, replicas=500, seed=1)
    cands = candidate_sets(rival_gaps(reviews, draws)).iloc[0]
    assert cands["winner"] == "A"
    assert cands["rivals_kept"] == 0
    assert cands["partial_order"] == "A > B > C > D"


def test_an_exact_tie_puts_both_alternatives_in_the_candidate_set():
    from utils.stability import candidate_sets, draw_crowd, rival_gaps

    reviews = _synthetic_reviews(
        {"A": [4.0] * 20, "B": [4.0] * 20, "C": [2.0] * 20, "D": [1.0] * 20}
    )
    draws = draw_crowd(reviews, replicas=500, seed=1)
    gaps = rival_gaps(reviews, draws)
    cands = candidate_sets(gaps).iloc[0]
    assert cands["rivals_kept"] == 1
    assert cands["candidates"] == "B"
    assert (
        gaps[(gaps["kind"] == "winner") & (gaps["lower"] == "B")]["le0_pct"].iloc[0]
        == 100.0
    )


def test_a_non_adjacent_rival_can_be_a_candidate_when_the_adjacent_one_is_not():
    """B is steadily just below A; C is lower on average but its reviews are
    spread so widely that redrawn crowds often lift it above A. Testing only
    adjacent pairs would declare A the winner."""

    from utils.stability import candidate_sets, draw_crowd, rival_gaps

    reviews = _synthetic_reviews(
        {
            "A": [3.0] * 30,
            "B": [2.9] * 30,
            "C": [4.0, 4.0, 0.0, 3.0],
            "D": [0.0] * 30,
        }
    )
    draws = draw_crowd(reviews, replicas=2000, seed=3)
    gaps = rival_gaps(reviews, draws)
    rivals = gaps[gaps["kind"] == "winner"].set_index("lower")
    assert rivals.loc["B", "distinguished"]
    assert not rivals.loc["C", "distinguished"]
    cands = candidate_sets(gaps).iloc[0]
    assert cands["candidates"] == "C"


def test_crowd_figures_share_one_set_of_replicas(cdm_runs):
    """The crowd table, the joint table's crowd row and the candidate sets
    read the same redrawn crowds, so the same system gets the same Top-1 from
    each, exactly, rather than up to Monte-Carlo error."""

    from utils.stability import draw_crowd

    label5 = cdm_runs[cdm_runs["scheme"] == "LABEL-5"]
    label5 = label5[label5["model"].isin(["Gemma 4 E4B", "Gemma 4 31B"])]
    reviews = fuse_runs(label5, "two_tuple")
    draws = draw_crowd(reviews, replicas=1000, seed=5)
    crowd = crowd_stability(reviews, draws=draws).set_index("model")
    joint, _ = joint_stability(label5, "two_tuple", 1000, 5, draws=draws)
    joint = joint[joint["condition"] == "experts"].set_index("model")
    for model in ("Gemma 4 E4B", "Gemma 4 31B"):
        assert crowd.loc[model, "top1_pct"] == joint.loc[model, "top1_pct"]


def test_stratified_draws_keep_the_number_of_clusters_per_alternative():
    from utils.stability import draw_crowd

    reviews = _synthetic_reviews({"A": [1.0] * 7, "B": [2.0] * 5})
    draws = draw_crowd(reviews, replicas=200, seed=2, mode="stratified")
    a = draws.cluster_counts([("A", f"u-A-{i}") for i in range(7)])
    b = draws.cluster_counts([("B", f"u-B-{i}") for i in range(5)])
    assert (a.sum(axis=1) == 7).all()
    assert (b.sum(axis=1) == 5).all()


def test_calls_per_review_count_every_member():
    """One model with k runs makes k calls; a fusion of m members makes m k."""

    from utils.budget import calls_comparison

    budget = pd.DataFrame(
        [
            {
                "system": "A",
                "k": 1,
                "calls_per_review": 1,
                "top1_crowd": 70.0,
                "margin_top": 0.1,
                "winner_ok_pct": 100.0,
                "cost_s_per_review": 0.1,
            },
            {
                "system": "B",
                "k": 2,
                "calls_per_review": 2,
                "top1_crowd": 60.0,
                "margin_top": 0.1,
                "winner_ok_pct": 100.0,
                "cost_s_per_review": 0.2,
            },
            {
                "system": "Fusion (open-weight)",
                "k": 1,
                "calls_per_review": 2,
                "top1_crowd": 65.0,
                "margin_top": 0.1,
                "winner_ok_pct": 100.0,
                "cost_s_per_review": 0.3,
            },
            {
                "system": "A (T=0)",
                "k": 1,
                "calls_per_review": 1,
                "top1_crowd": 99.0,
                "margin_top": 0.1,
                "winner_ok_pct": 0.0,
                "cost_s_per_review": 0.1,
            },
        ]
    )
    calls = calls_comparison(budget).set_index("calls_per_review")
    assert calls.loc[1, "single_system"] == "A"  # greedy rows are kept apart
    assert calls.loc[2, "single_system"] == "B"
    assert calls.loc[2, "fusion_top1"] == 65.0


def test_an_exact_tie_survives_summation_order():
    """Two alternatives with the same mean tie whatever order it was summed in.

    0.1 + 0.2 + 0.3 and 0.3 + 0.2 + 0.1 differ by one ulp; without a tolerance
    the tie would keep the winner in one table and lose it in another.
    """

    from utils.stability import keeps_winner, ranks_of

    a = (0.1 + 0.2 + 0.3) / 3
    b = (0.3 + 0.2 + 0.1) / 3
    assert a != b
    ranks = ranks_of(np.array([[a, b, 0.0]]))
    assert ranks.tolist() == [[1, 1, 3]]
    assert keeps_winner(ranks, 0).all() and keeps_winner(ranks, 1).all()


# -- simultaneous candidate sets (bootstrap max-t) --------------------------


def _clustered(means, sds, clusters=40, seed=0, model="M"):
    """One review per expert, `clusters` experts per alternative."""

    rng = np.random.default_rng(seed)
    return _synthetic_reviews(
        {
            name: list(rng.normal(mean, sd, clusters))
            for name, mean, sd in zip("ABCD", means, sds, strict=True)
        },
        model=model,
    )


@pytest.mark.parametrize("corpus", ["london", "madrid"])
def test_simultaneous_set_contains_the_pairwise_set_on_the_reported_data(corpus):
    """Never narrower than the pairwise rule, on every system and crowd mode.

    A failure here is a finding to report, not a test to relax: it would mean
    strongly asymmetric differences somewhere."""

    path = Path("results") / corpus / "cdm_simultaneous_candidates.csv"
    sets = pd.read_csv(path)
    sets = sets[sets["method"] == "max_t"]
    assert not sets.empty
    assert (sets["set_size"] >= sets["set_size_pairwise"]).all()
    assert sets["pairwise_subset"].all()


def test_simultaneous_set_is_the_winner_alone_when_one_alternative_is_far_ahead():
    from utils.stability import draw_crowd, simultaneous_candidates

    reviews = _clustered([2.5, 2.0, 2.0, 2.0], [0.3, 0.3, 0.3, 0.3], seed=1)
    draws = draw_crowd(reviews, replicas=2000, seed=1)
    _, sets = simultaneous_candidates(reviews, draws)
    assert sets["winner"].iloc[0] == "A"
    assert int(sets["set_size"].iloc[0]) == 1


def test_simultaneous_set_keeps_every_alternative_when_they_are_equal():
    from utils.stability import draw_crowd, simultaneous_candidates

    reviews = _synthetic_reviews({k: [3.0, 2.0, 1.0] * 10 for k in "ABCD"})
    draws = draw_crowd(reviews, replicas=1000, seed=2)
    pairs, sets = simultaneous_candidates(reviews, draws)
    assert int(sets["set_size"].iloc[0]) == 4
    assert not pairs["excluded_by"].any()


def test_simultaneous_intervals_cover_every_true_difference_jointly():
    """Sanity check of joint coverage, unequal variances by a factor of three.

    1,000 repetitions, not 200: at 200 the Monte-Carlo error of the coverage
    (about 1.7 points) is as large as the margin being tested. Measured at
    40, 80 and 160 clusters per alternative the joint coverage is 0.94, a
    little under the nominal 0.95, as bootstrap intervals usually are.
    """

    from utils.stability import draw_crowd, simultaneous_candidates

    means = {"A": 2.4, "B": 2.3, "C": 2.1, "D": 2.0}
    sds = [0.3, 0.6, 0.9, 0.45]
    covered = 0
    repetitions = 1000
    for rep in range(repetitions):
        reviews = _clustered(list(means.values()), sds, seed=10000 + rep)
        draws = draw_crowd(reviews, replicas=2000, seed=rep)
        pairs, _ = simultaneous_candidates(reviews, draws)
        truth = pairs["i"].map(means) - pairs["j"].map(means)
        inside = (pairs["lo_simult"] <= truth) & (truth <= pairs["hi_simult"])
        covered += bool(inside.all())
    assert covered / repetitions >= 0.93


def test_simultaneous_sets_are_deterministic_given_the_draws():
    from utils.stability import draw_crowd, simultaneous_candidates

    reviews = _clustered([2.2, 2.1, 2.0, 1.9], [0.5, 0.5, 0.5, 0.5], seed=7)
    draws = draw_crowd(reviews, replicas=1000, seed=7)
    first = simultaneous_candidates(reviews, draws)
    second = simultaneous_candidates(reviews, draws)
    for a, b in zip(first, second, strict=True):
        assert a.to_csv(index=False) == b.to_csv(index=False)


def test_the_observed_winner_is_always_a_simultaneous_candidate():
    from utils.stability import draw_crowd, rival_gaps, simultaneous_candidates

    for seed in range(5):
        reviews = _clustered([2.2, 2.15, 2.0, 1.8], [0.4, 0.8, 0.5, 0.3], seed=seed)
        draws = draw_crowd(reviews, replicas=1000, seed=seed)
        _, sets = simultaneous_candidates(reviews, draws)
        winner = rival_gaps(reviews, draws).iloc[0]["upper"]
        assert sets["winner"].iloc[0] == winner
        assert winner in sets["candidates"].iloc[0].split("; ")


def test_madrid_simultaneous_sets_do_not_depend_on_the_crowd_mode():
    """The manuscript says the simultaneous sets are unchanged under the
    stratified resampling; that is a claim about the data, pinned here."""

    sets = pd.read_csv("results/madrid/cdm_simultaneous_candidates.csv")
    # The systems; the stars' own set does move (seven to six alternatives).
    sets = sets[(sets["method"] == "max_t") & (sets["model"] != "Stars (reference)")]
    wide = sets.pivot_table(
        index="model", columns="crowd_mode", values="candidates", aggfunc="first"
    )
    assert (wide["experts"] == wide["stratified"]).all()


def test_tiebreak_table_does_not_depend_on_string_hashing():
    """The fused system's reviews were once taken in set order, which changes
    between Python processes; with tied alternatives that reordered them."""

    import subprocess
    import sys

    script = (
        "import pandas as pd\n"
        "from utils.stability import tiebreak_sensitivity\n"
        "rows = []\n"
        "for m in ('A1', 'A2', 'A3'):\n"
        "    for i, alt in enumerate('PQRS' * 6):\n"
        "        for run in range(4):\n"
        "            tied = [0.5, 0.5, 0.0, 0.0][run]\n"
        "            y = tied if alt in 'PQ' else [0.0, 0.5][run % 2]\n"
        "            rows.append(dict(model=m, scheme='LABEL-5', review_id=f'r{i}',\n"
        "                uid=f'u{i}', user_id=f'u{i}', entity_id=alt,\n"
        "                alternative=alt,\n"
        "                run=run, y=y))\n"
        "runs = pd.DataFrame(rows)\n"
        "table = tiebreak_sensitivity(runs, ['A1', 'A2', 'A3'], 'F')\n"
        "print(table.to_csv(index=False))\n"
    )
    outputs = set()
    for seed in ("1", "2", "3"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
            check=True,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1
