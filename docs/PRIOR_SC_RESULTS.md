# Existing SC scale observations

User-provided pre-experiment figures are retained at
`D:/Personal/Desktop/ALLM Project/figure/`. They are reusable observations, not
new runs, and should be credited as the project's earlier tests. File identities
are in `prior_sc_figures.json`; the original image/GIF is not modified.

## What is visible

`SC_sweep.png`: one example labeled `codemerge`; a 256-position token-entropy
heatmap for SC scale 1, 0.8, 0.6, 0.4, 0.2, and 0. Its displayed stop labels are
12, 10, 11, 12, 12, and 48, respectively. The positive-scale panels show similar
broad entropy patterns; zero SC retains high entropy and reaches the displayed
48-step limit. This is evidence of a convergence difference in this example.
The zero-SC panel has a different vertical range; do not compare apparent panel
heights as if they had equal step scales. A limit hit is right-censored, not a
measured successful convergence time.

`SC_sweep_trace_compare.gif`: 49 display frames comparing the same scale labels.
In the last displayed frame, positive-scale panels show similar merge-function
structure, at local steps 37/39/38/37/37. Zero SC is shown on canvas 1, local step
1, with many unformed positions. The display states that canvases retain local
step indices. Global display frames are not a shared physical step clock.
These counts differ from the PNG and must be reported separately until their
run configurations and raw traces are recovered.

## Interpretation and reuse

- Reuse both figures in a preliminary-results or motivation section, preserving
  their provenance. They support a hypothesis that positive scale changes may
  leave broad generation behavior robust while zero SC slows settling.
- Visual similarity does not establish equal final correctness, identical token
  trajectories, or benchmark-level equivalence. No benchmark scores are visible.
- Scale sweeps change signal amplitude; the paper's top-k/threshold changes its
  information content. The existing figures do not replace those ablations.
- The pinned upstream `SelfConditioning` applies `pre_norm` (RMSNorm) before its
  feed-forward projection. If scaling was applied before this normalization,
  positive scaling can be largely normalized away, up to epsilon/rounding.
  This is a mechanistic hypothesis, not a verified explanation of these figures;
  the old scale insertion point has not yet been recovered.

The user confirms that these are real earlier tests and defines beta as a
multiplicative SC coefficient. The original code may no longer be available.
Reuse the GIF as an archival qualitative illustration; reproduce a fresh entropy
sweep using fully recorded settings and a clearly labeled adapted prompt.

Missing provenance: exact original scaling boundary, prompt text/source,
seed(s), checkpoint/version, sampler/early-stop settings, raw per-step arrays,
and scoring outputs. These are not reconstructed from the image alone.

For new observations, separate final quality from convergence: record per-step
entropy, accepted positions, prediction changes and stopping status. A common
48-step cap gives quality at a fixed budget; a zero-SC cap hit must be flagged.
Any longer-cap follow-up should be a separately labeled controlled experiment.

Preserve these preliminary observations regardless of whether later benchmarks
confirm them. Do not tune evaluation or case selection to force the expected
outcome. Select cases from baseline trajectories before causal interventions.
