# SC baseline results: 64 matched cases per benchmark

Verified 2026-11-18. Five conditions, GSM8K and sanitized MBPP, 640 outputs.
Google diffusiongemma-26B-A4B-it, official JAX implementation; checkpoint
manifest SHA256 `518abf72853c48679c8c412c928ae2cd09a0b61e284ad0bdbd5b7131fd23c48c`.
Fixed seed-4953 shuffled test prefixes, generation seed 0 plus task/batch
index, batch 4, identical prompts/seeds/positions across conditions. Canvas256,
max1024 output tokens, upstream adaptive stopping; caps48 except disabled96.
Retained existing 296 generations via audited import and generated the rest.

| Condition | GSM8K exact match | MBPP single-sample pass@1 |
|---|---:|---:|
| Full SC | 62/64 (96.88%) | 37/64 (57.81%) |
| Disabled, 48 steps | 21/64 (32.81%) | 6/64 (9.38%) |
| Disabled, 96 steps | 34/64 (53.12%) | 8/64 (12.50%) |
| Top-k=8 | 60/64 (93.75%) | 35/64 (54.69%) |
| Threshold=.03 | 61/64 (95.31%) | 33/64 (51.56%) |

Scoring was completed on CPU after generation: matched numeric final-answer
extraction for GSM8K, sanitized public assertions/imports with DGtest-style
code extraction and 10-second timeout for MBPP. Raw outputs remain untouched;
scored outputs live under `cpu_scores`, with scoring protocol/input SHA256.
All 640 grades exist and final groups are complete.

Qualitative example: exact original DGtest codemerge prompt, seed42, batch1,
one 256-token canvas; same five SC conditions. Observed stopping lengths:
Full11, disabled48 reaches cap48 without meeting stopping criterion,
disabled96 meets criterion at91, top-k19, threshold8. The figure follows the
original DGtest heatmap style, with independent step axes and shared 0–4 colors.
These are one example's trajectories, not averages, and stopping is not proof
of correct code. All raw entropy arrays and generated sample text are retained.

This small matched subset supports an important role for SC in capability and
shows that sparse SC preserves much of the observed performance. More steps
partially recover disabled performance. It does not establish statistical
equivalence of sparse and Full SC or reproduce the paper's original test suite.

GPU13965298: COMPLETED03:59:52 on H100 PCIe/ins049, driver550.54.14.
CPU13965303: COMPLETED00:00:12 on ins022. Prior GPU13956052 timed out after3h;
296 valid rows were imported rather than regenerated. The original source was
archived before deploying deferred scoring. Different failed setup attempts
are documented in EXPERIMENT_STATUS.md; no aggregate throughput claim is made.

Local and server result folder: `outputs/sc-benchmark64-deferred-v2`.
Final table: `cpu_scores/benchmark_scores.csv` and `.md`.
Figure: `SC_conditions_entropy.png` and `.pdf`.
Raw trajectories: `example_entropy/*.npz`, texts/config in sibling JSON files.
Validation checked uniqueness, sample/prompt/seed pairing, scoring input hash,
native/Full equality gate, and entropy lengths/flags/array contents, plus visual QA.
