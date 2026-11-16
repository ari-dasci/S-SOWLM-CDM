# Small open-weight language models for crowd decision-making

Code, model outputs and results for *Small Open-Weight Language Models for
Crowd Decision-Making from Textual Opinions: Reliability of Linguistic
Aggregation and Output-Level Fusion*.

Two decision problems: **London** (TripR-2020Large, 474 reviews, 4 restaurants;
five local models and GPT-5.2 as hosted reference) and **Madrid** (TripAdvisor
Restaurant Reviews v2.0.0, 600 reviews, 12 restaurants; three local models).
Every model reply is in `raw/`; everything in `results/` is rebuilt from it.

## Install

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"        # Python >= 3.11
```

## Reproduce the tables and figures

```sh
scripts/rebuild.sh              # 12-30 min, no GPU, no model queried
pytest
```

`rebuild.sh` runs, for each corpus `c` in `london`, `madrid`:

| Step | Command | Writes into `results/<c>/` |
|---|---|---|
| Parse the archives | `python -m utils.runs <c>` and `<c>_t0` | `runs_*.csv`, `invalid_outputs.csv` |
| Latency | `python -m utils.latency <c>` | `latency*.csv` |
| Decision analysis | `python -m utils.cdm_analyze <c>` | `cdm_*.csv`, `tables/`, `figures/` |
| Greedy decoding | `python -m utils.temperature <c>` | `tables/cdm_temperature.tex` (CSV in `results/<c>_t0/`) |
| Paired comparisons | `python -m utils.paired <c>` | `cdm_paired.csv`, `cdm_variance_lead.csv`, `cdm_star_*.csv` |
| Protocol tables | `python -m utils.protocol <c>` | `tables/provenance.tex`, `tables/protocol_*.tex` |
| Worked example | `python -m utils.worked_example london` | `tables/cdm_worked_example.tex` |
| Seed sensitivity | `python -m utils.seeds <c>` | `seed_sensitivity.csv` |

Resampling uses seed 42 and 20,000 replicas. `utils.seeds` recomputes the
simultaneous candidate sets with seeds 1-4 at 20,000 replicas and seeds 42, 1
and 2 at 60,000. Any other design:
`python -m utils.cdm_analyze madrid --seed 7 --replicas 60000 --out /tmp/madrid_s7`.

### Where each table and figure comes from

Each `tables/<name>.tex` has a CSV of the same content beside it in
`results/<c>/`. L = `results/london`, M = `results/madrid`.

| Paper | File |
|---|---|
| Table II | `cdm_fusion_vs_best.csv`; Stars rows in `cdm_candidates.csv` (L, M) |
| Table III | `cdm_budget.csv` (L, M) |
| S1 | L `tables/cdm_worked_example.tex` |
| S3, S4 | L `tables/protocol_prompts.tex`, `protocol_decoding.tex` |
| S5, S30 | `tables/cdm_validity.tex` (L, M) |
| S6, S7 | `tables/provenance.tex` (L, M) |
| S8, S31 | `tables/latency.tex` (L, M) |
| S9, S10, S22 | `tables/cdm_alternatives_two_level.tex` (L, M), L `cdm_alternatives_three_level.tex` |
| S11, S37 | `tables/cdm_stars.tex` (L, M) |
| S12 | L `tables/cdm_stability.tex` |
| S13, S32 | `tables/cdm_discriminance.tex` (L, M) |
| S14, S33 | `tables/cdm_operator_scope.tex` (L, M) |
| S15, S16 | `tables/cdm_joint.tex` (L, M) |
| S17 | L `tables/cdm_crowd_stability.tex` |
| S18, S19 | `tables/cdm_candidates.tex` (L, M) |
| S20, S21 | `tables/cdm_simultaneous_pairs.tex` (L, M) |
| S23, S34 | `tables/cdm_mapping.tex` (L, M) |
| S24, S35 | `tables/cdm_missing_data.tex` (L, M) |
| S25 | L `tables/cdm_tiebreak.tex` |
| S26, S36 | `tables/cdm_temperature.tex` (L, M) |
| S27 | L `tables/cdm_fusion_vs_best.tex` |
| S28, S29 | `tables/cdm_budget.tex` (L, M) |
| S38 | `cdm_paired.csv` (L, M) |
| S39 | `cdm_variance_lead.csv` (L, M) |
| S40 | `cdm_star_paired.csv` (L, M), L `cdm_star_gaps.csv` |
| Fig. S1 | L `figures/alternative_profile.pdf` |
| Fig. S2, S3 | `figures/sensitivity.pdf` (L, M) |
| Fig. S4 | L `figures/resolution.pdf` |
| Fig. S5 | L `figures/ranking_stability.pdf` |

Table I, Table S2 and Fig. 1 are descriptive and have no generated source.

## Expected output

`scripts/rebuild.sh` regenerates `results/` from `raw/`; `git status` stays clean
(12-30 min depending on the CPU). With other library versions, a few percentages in
`results/madrid/cdm_joint.csv` and `cdm_rival_gaps.csv` can change in the last digit
(exact ties at the printed precision); tables and figures are unaffected.

## Data and collection environment

`raw/<corpus>/<model>/<scheme>.tar.gz` holds one JSON per run (`model`, `run_number`,
`review_idx`, `latency_s`, `history` with prompt and reply). `results/<c>/runs_*.csv`
has one row per accepted reply (`model`, `scheme`, `review_id`, `uid`, `run`, `raw`, `y`
in [-1, 1]); rejected replies are in `invalid_outputs.csv`. Local models ran with
Ollama (0.31.1 London, 0.40.4 Madrid), Q4_K_M, on one NVIDIA GeForce RTX 5090 (32 GB);
GPT-5.2 through the OpenAI API (`gpt-5.2-2025-12-11`). Tags, digests and runtime
versions are in `raw/<corpus>/model_provenance.csv`.

## Collect new model outputs

Needs [Ollama](https://ollama.com) serving the models of `config/models.yaml`;
prompts and decoding settings are in `config/experiments.yaml`.

```sh
scripts/collect.sh madrid            # or london, london-hosted, madrid-latency
scripts/collect.sh archive madrid    # collected/madrid/ -> raw/madrid/*.tar.gz
RUNS=1 LIMIT=5 scripts/collect.sh madrid   # a small trial
```

`data/sample_uniovi.py` regenerates the Madrid sample from the Zenodo release
(doi:10.5281/zenodo.14622324).

## Layout

```
config/       profiles.yaml (corpora and systems), experiments.yaml (prompts), models.yaml
data/         the two review corpora, Madrid sampling script and manifest
raw/          per-run model outputs, <corpus>/<model>/<scheme>.tar.gz, and runtime snapshots
results/      derived CSVs, LaTeX tables and figures
utils/        analysis: parsing, 2-tuples, aggregation, resampling, tables, figures
scripts/      rebuild.sh, collect.sh
backends/ experiments/ models/ sentiment_prompt_runner.py   the collection runner
tests/        checks of the pipeline and of the shipped results
```

## License

Code under MIT (`LICENSE`). TripR-2020Large and the
TripAdvisor Restaurant Reviews dataset (CC BY 4.0) keep their own terms.
