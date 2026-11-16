# Madrid collection (2026-10-14)

Serving: Ollama 0.40.4 (updated that day from 0.31.1, which the London
collections used), NVIDIA GeForce RTX 5090, driver 615.78.08 (London:
610.43.03). systemd drop-in: `OLLAMA_NUM_PARALLEL=8`,
`OLLAMA_CONTEXT_LENGTH=4096`. Runner: `--workers 4` for the collection,
`--workers 1` for the latency re-measurement in `raw/madrid_latency/`.

Digests at collection: `gemma4:e4b` dc35e8d9c606 (a re-published tag; the
London collection of April ran c6eb396dbd59), `ministral-3:8b` 1922accd5827,
`qwen3.6:35b` 07d35212591f (both as in London). `gemma4:e4b` was collected
with `--draft-num-predict 0`: the re-published tag ships a
speculative-decoding draft model.

Incidents, recorded rather than repaired:

- A Ministral 3 8B pilot (100 reviews x 5 runs) run under Ollama 0.31.1
  before the update was discarded; the reported Ministral collection is
  entirely under 0.40.4.
- For about one minute (16:40:12-16:41:04) a `gemma4:e4b` SCORE run executed
  concurrently with the `ministral-3:8b` LABEL-5 T = 1 run. Outputs are
  unaffected; the collection latencies are not used (latency comes from the
  sequential re-measurement).
