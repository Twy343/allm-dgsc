# Experiment 2: Positional and Joint Interventions

Status: normalization-preserving token-mass shift and probability center-of-mass
metric implemented; checkpoint-backed cases and controls remain pending.

Shift or anchor a candidate token `v` from position `i` to `i + delta` in the
intermediate token-space state. Examine whether neighboring distributions shift
with it, compensate, or reject the intervention and restore the original
arrangement. Specify the exact normalization-preserving operation during
implementation.

## Controls

- Unmodified rollout.
- Matched random positional interventions.
- Equal-magnitude interventions at irrelevant positions.

## Measurements

- Probability-weighted center of mass of selected token identities.
- Displacement of probability mass across positions.
- Neighboring distribution responses and local edit distance.
- Coordinated shift, compensation, or rejection across denoising steps.

Case location: `datasets/positional_intervention/`.
Shared prerequisites: `docs/PROPOSAL_PLAN.md`.

Implementation: `intervention.py` and `metrics.py`. Use `PositionalShift` as the
`intervention` argument to `exp_code.common.dg_runner.run_fixed_trajectory`.
