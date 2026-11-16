# Matched test256 run

Hardware alternative: the two-A6000 route preserves this sample/condition plan
and batch size 4 using the original loader's FSDP sharding. It uses its own output
directory and separate hardware provenance; see GPU_OPTIONS.md. Invocation limits
control resumable chunks only and do not change the planned 256 test cases.

User revision on 2026-11-16 supersedes the earlier 64-per-task pilot and full-test
target for the immediate experiment. Run 256 fixed test cases from GSM8K and
256 fixed sanitized MBPP test cases, five conditions sequentially (2,560 outputs):

| Order | Condition | SC | Maximum steps per canvas |
| --- | --- | --- | --- |
| 1 | full | Original | 48 |
| 2 | disabled_48 | Zero | 48 |
| 3 | disabled_96 | Zero | 96 |
| 4 | top_k | k=8, paper mass-preserving tail | 48 |
| 5 | threshold | p=0.03, paper mass-preserving tail | 48 |

Batch size is 4; one generation per sample, maximum 1,024 output tokens, 256-token
canvas, original adaptive stopping. Steps are caps, not forced iterations after
convergence. Each budget uses the original temperature schedule stretched across
its 48 or 96 steps; the 96-step run is not an identical 48-step prefix followed
by another 48 steps. Selection seed 4953 and fixed ordering/partitioning are
shared across conditions. Batch random keys are deterministic and paired across
conditions, with task/batch offsets recorded per prediction. Stochastic batch
results are not claimed to match batch-size-1 outputs.

Use upstream list-of-prompts `Sampler.sample` with the same dialog formatting as
ChatSampler; load weights once and reuse them through all five conditions. An
original-versus-Full SC batched checkpoint equality gate runs before scores.
Only a short correctness gate is added; there is no separate 64-case pilot.

Store each sample's per-token entropy at every effective denoising step for
every generated canvas. Ordered JAX callbacks transfer only entropy/stop flags,
not full-vocabulary logits. Ignore subsequent dummy evaluations of a member
already frozen by the batch's adaptive loop. Ignore canvases generated only for
other batch members after this sample reached EOS. NPZ files include output
tokens, per-canvas step counts, stopping criterion flags and cap-hit flags.

Score GSM8K numeric exact match and single-sample MBPP pass@1. Summaries show
completion, paired accuracy differences/intervals, effective sample/canvas step
counts and cap hits. These step counts are not GPU kernel counts: frozen batch
members may still be evaluated. Instrumented batch timing includes entropy
callbacks and possible new-shape compilation; equal per-row shares of batch
latency are not per-request latency measurements.

After generation, produce one matched first-canvas entropy heatmap per task,
with a common 96-step axis, common color scale, and explicit blank space after
stopping. All other sample traces remain available for additional figures.

Entry point: `scripts/benchmark_256.sbatch`; results in
`outputs/sc-test256-v2/`. Model download and verification are already complete.
An independent CPU job validates callbacks/native equivalence and checks all
256 selected MBPP references before the benchmark starts. Request one H100 in
the authorized burst partition, at most four hours; this is a wall-time ceiling,
not an ETA. No recurring monitor is enabled. Earlier beta-sweep visualization
is retained as a separate subsequent job, not mixed into benchmark conditions.
