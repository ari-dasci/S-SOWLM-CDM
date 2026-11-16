#!/usr/bin/env bash
# Collect model outputs and archive them into raw/. Needs Ollama serving the
# models of config/models.yaml (or OPENAI_API_KEY for the hosted reference).
#
#   scripts/collect.sh london           # five local models, sequential requests
#   scripts/collect.sh london-hosted    # GPT-5.2 through the OpenAI API (billed)
#   scripts/collect.sh madrid           # three local models, 4 requests in flight
#   scripts/collect.sh madrid-latency   # 60 reviews, one request at a time
#   scripts/collect.sh archive london|madrid|madrid_latency
#
# Run JSON goes to collected/<corpus>/<model>/<scheme>/review-<i>/run-<n>.json.
# The runner resumes: a run whose JSON exists is skipped. `archive` writes
# raw/<corpus>/<model>/<scheme>.tar.gz and never overwrites an archive.
#
# RUNS and LIMIT shrink a collection for a trial: RUNS=1 LIMIT=5 scripts/collect.sh madrid
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
RUNS=${RUNS:-10}
LIMIT=${LIMIT:+--limit $LIMIT}
SAMPLED="single_label_5c_t1 single_score_v2_t1 single_label_7c_t1"

collect() {  # corpus input backend family size workers experiment [runner args]
    local corpus=$1 input=$2 backend=$3 family=$4 size=$5 workers=$6 experiment=$7
    shift 7
    $PY sentiment_prompt_runner.py run --backend "$backend" --model "$family" \
        --size "$size" -e "$experiment" -r "$RUNS" --workers "$workers" \
        --input "$input" --results-dir "collected/$corpus" $LIMIT "$@"
}

case "${1:-}" in
    london)
        for model in "gemma4 e4b" "ministral 8b" "gemma4 31b" "ornith 35b" "qwen36 35b"; do
            for experiment in $SAMPLED single_label_5c_t0; do
                collect london data/TripR-2020Large_FullReviews.csv ollama $model 1 "$experiment"
            done
        done
        ;;
    london-hosted)
        : "${OPENAI_API_KEY:?export OPENAI_API_KEY first}"
        for experiment in $SAMPLED; do
            collect london data/TripR-2020Large_FullReviews.csv openai gpt 5.2 4 "$experiment"
        done
        ;;
    madrid | madrid-latency)
        corpus=madrid workers=4 experiments="$SAMPLED single_label_5c_t0"
        if [[ $1 == madrid-latency ]]; then
            corpus=madrid_latency workers=1 experiments=$SAMPLED RUNS=1 LIMIT="--limit 60"
        fi
        for model in "gemma4 e4b" "ministral 8b" "qwen36 35b"; do
            # The re-published gemma4:e4b ships a speculative-decoding draft
            # model; it is disabled, as the first collection had none.
            draft=()
            [[ $model == "gemma4 e4b" ]] && draft=(--draft-num-predict 0)
            for experiment in $experiments; do
                collect "$corpus" data/UniOvi-Madrid-600.csv ollama $model "$workers" \
                    "$experiment" "${draft[@]}"
            done
        done
        ;;
    archive)
        corpus=${2:?corpus}
        for model_dir in "collected/$corpus"/*/; do
            dest="raw/$corpus/$(basename "$model_dir" | tr ':' '-')"
            mkdir -p "$dest"
            for scheme_dir in "$model_dir"*/; do
                scheme=$(basename "$scheme_dir")
                if [[ -e "$dest/$scheme.tar.gz" ]]; then
                    echo "keep  $dest/$scheme.tar.gz"
                else
                    tar czf "$dest/$scheme.tar.gz" -C "$model_dir" "$scheme"
                    echo "wrote $dest/$scheme.tar.gz"
                fi
            done
        done
        ;;
    *)
        sed -n '2,13p' "$0" >&2
        exit 2
        ;;
esac
