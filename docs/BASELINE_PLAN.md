# SC bottleneck reproduction

## Scope and order

The interim report uses three components: four-condition score evaluation, a
new beta/SC-bottleneck entropy figure, and the user's existing qualitative GIF.
See [prior observations](PRIOR_SC_RESULTS.md). Beta is a user-defined amplitude
coefficient, not the paper's top-k/threshold information restriction. The new
observation sets `S_next = beta * S_next` before next-step `pre_norm` and labels
its merge-intervals prompt as adapted; it does not claim to recover the old run.

First establish the full / zero / top-k=8 / threshold=0.03 baselines. Then use
observed trajectories to select semantic and positional interventions; investigate
successive context updates after those controls are established.

This stage reproduces the bottleneck method from *How Transparent is
DiffusionGemma?*, Section 3.1 and Appendix C. GSM8K and MBPP are our adaptation,
not the paper's original capability suite or a replication of its reported scores.
The paper's threshold is an absolute probability cutoff, not nucleus top-p.

## Method

Operate on temperature-shaped logits only when constructing the SC embeddings
for the following denoising step. Leave that step's sampled canvas and output
logits unchanged. Full uses the upstream embeddings; disabled passes zeros.
For sparse modes, retain the original selected logits and replace the discarded
logits by log(mean(exp(discarded_logits))). This preserves selected probabilities
and total discarded mass, distributing the latter uniformly. Do not force an
argmax when no probability exceeds the threshold. Compute the tail in float32
(float64 for float64 inputs), then retain the original logit dtype for the
upstream encoder; report finite-precision deviations rather than claim exact
real-arithmetic preservation for bfloat16.

`sc_sampler.py` subclasses the original sampler. It retains original prompt
formatting, prefill/KV handling, EOS behavior, adaptive stopping and multi-canvas
generation. No upstream file is edited. Capability runs use the pinned runtime's
generation behavior; they are distinct from the fixed-48-step observation loop.

Before capability results: tiny real-model boundary checks, then equality of
Full SC and the original chat sampler on actual checkpoint output tokens.

## Evaluation

1. Smoke: two training examples per task, all four conditions (16 generations).
2. Pilot: 64 fixed training examples per task, four conditions (512 generations).
3. After inspecting pilot correctness, time and memory, freeze the protocol and
   run complete GSM8K test (1,319) / sanitized MBPP test (257), or fixed test
   subsets of 256 / 128 if the compute budget requires it.

Select with seed 4953; generate with seed 0, one output per example per condition.
All conditions share sample IDs, prompt text and random seeds. Use 256-token
canvases, up to 48 steps per canvas, upstream adaptive stopping, entropy bound
0.1, temperature schedule 0.8 to 0.4, and 1,024 generated tokens. This is a
zero-shot instruction protocol; MBPP includes the provided public assertions.

GSM8K: numeric exact match using the final `####` marker (numeric boxed/final-line
fallback). MBPP: execute the extracted solution and all supplied assertions once,
with a five-second timeout, reporting single-sample pass@1. Check reference
solutions before loading the checkpoint. Generated code runs in a disposable
process with a stripped environment, not a security sandbox; use an isolated
compute allocation. No downloaded task data or generated code is committed.

Reports include completion counts, accuracy, paired improved/regressed counts,
paired accuracy difference and an exploratory paired bootstrap interval. A pilot
or overlapping wide intervals do not establish capability equivalence. Per-example
times include host/API overhead and possible new-shape compilation; they are not
an uninstrumented throughput measurement. True denoising counts are not exposed
by the original adaptive sampler and are not fabricated from outer-loop steps.

## Reproducibility and execution

The downloader pins GCS object generations, verifies available MD5 values and
records SHA256 per file. The runner checks that manifest and file sizes; for
suspected checkpoint corruption rerun the downloader to rehash actual contents.
It records source digest, checkpoint digest, dataset fingerprint, cases/prompts,
settings, software and hardware. Output metadata prevents reuse across changed
sources or settings; a lock blocks concurrent writers. Failed model generation
stops rather than being silently scored as an incorrect answer. A stale lock
after external termination must be removed only after checking the recorded PID
and Slurm job. Results are fsynced after each example for resumability.

From the repository, in the configured `allm` Slurm environment:

```bash
python -m unittest discover -s exp_code/tests -v
python -m exp_code.baselines.verify_tiny
python -m exp_code.baselines.run --data-root /insomnia001/depts/edu/users/yc4953/datasets \
  --output outputs/sc-baseline-v1/pilot --stage pilot --prepare-only
```

`scripts/sc_baseline.sbatch` requests one H100 for at most two hours, runs smoke,
then pilot. Submit it with `afterok` dependencies on successful validation and
checkpoint download. Do not launch full test evaluation automatically from a
pilot. Results, GPU metadata and package snapshots live in ignored `outputs/`.
Checkpoint weights live outside the Git checkout.

`scripts/sc_entropy.sbatch` requests one H100 for at most one hour. It records
six beta values (1, .8, .6, .4, .2, 0), and separately the four paper-baseline
conditions, on one matched prompt/seed and its first canvas. It stores raw NPZ
traces, settings, stopping/cap status and two entropy PNGs. Every panel uses the
same 48-step vertical range and color scale, with blank regions after stopping.
It asserts beta=1 equals full and beta=0 equals disabled. Trace timings include
observation overhead and are not used as speedup claims.
