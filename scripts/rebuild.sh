#!/usr/bin/env bash
# Rebuild every derived file of the paper from the raw archives: runs, tables
# and figures of both corpora. No model is queried; about ten minutes.
#
#   scripts/rebuild.sh
#   PY=.venv/bin/python scripts/rebuild.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

for profile in london london_t0 madrid madrid_t0; do
    echo "== runs: $profile"
    $PY -m utils.runs "$profile"
done

$PY -m utils.latency london
$PY -m utils.latency madrid --raw-dir raw/madrid_latency

for corpus in london madrid; do
    echo "== analysis: $corpus"
    $PY -m utils.cdm_analyze "$corpus"
    $PY -m utils.temperature "$corpus"
    $PY -m utils.paired "$corpus"
    $PY -m utils.protocol "$corpus"
done
$PY -m utils.worked_example london
for corpus in london madrid; do $PY -m utils.seeds "$corpus"; done
