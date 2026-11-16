# Interim report evidence

## Research question

What does SC contribute to answer quality and denoising dynamics, and which
parts of its token-space state causally guide later computation?

## Completed preliminary observation

Reuse the user's original beta-sweep GIF as a real qualitative case study.
Beta is the user's multiplier on SC. Positive beta panels show broadly similar
code-generation structure; the zero-SC panel settles much more slowly. The old
PNG also shows positive-beta stop labels around 10-12 versus a 48-step cap hit
for zero SC. The old code/settings are no longer available: present this as
preliminary evidence and distinguish its observations from new controlled runs.
Do not combine the GIF's local step counts with the PNG's stop counts.

## Controlled experiments in progress

- **Scores:** full, disabled, top-k=8, threshold=0.03 on matched GSM8K and MBPP
  examples. Show sample count, exact match / single-sample pass@1, paired
  differences and uncertainty. Start with 64 training examples per task, then
  decide the test-set scope using measured time and memory. This adapts the
  paper's bottleneck experiment to our benchmarks.
- **Entropy:** reproduce the beta sweep (1, .8, .6, .4, .2, 0), plus a separate
  four-condition bottleneck sweep. Use one adapted merge-intervals prompt,
  identical seed and initial canvas, common color/step axes, and record stopped
  versus cap-reached explicitly. Save raw traces and settings with the figure.
  The new prompt is not claimed to be the unrecovered original prompt.
- **Implementation checks:** Appendix C probability preservation, unchanged
  current-step predictions under SC-only changes, Full SC/native-loop equality,
  beta=0/disabled and beta=1/full consistency. Tiny-model correctness checks are
  complete; checkpoint-backed gates and scores are still pending.

## Interpretation to test

The working hypothesis is that information bottlenecks and nonzero beta may
preserve much of the quality/pattern while zero SC slows denoising. Benchmark
scores and convergence measurements remain separate: the old figures alone do
not establish score equivalence. Because the SC branch begins with RMSNorm,
positive-beta robustness may partly reflect normalization; verify rather than
attribute it to redundant semantic information automatically.

## Next experiments

Use baseline trajectories to select cases for semantic candidate swaps and
positional mass shifts with matched controls. Then test repeated context
updates, refreshing prompt conditioning and comparing reuse/reset/regeneration.
These are planned directions, not completed interim findings.
