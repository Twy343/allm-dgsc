# Experiment 1: Semantic Causal Interventions

Status: core probability-swap hook and downstream log-probability metric
implemented; checkpoint-backed cases and controls remain pending.

Change a selected probability row `P_t[i, :]` by exchanging probability mass
between an observed intermediate token `v` and a counterfactual `v_prime`.
Construct the corresponding next-step SC signal while leaving the visible
canvas at the intervention boundary unchanged. Preserve row normalization.

## Controls

- Unmodified rollout.
- Matched random token swaps.
- Equivalent probability-mass changes at irrelevant nearby positions.

## Measurements

- Log-probability changes for intended counterfactual downstream tokens.
- Probability decreases for the original computation path.
- Final counterfactual consistency.
- Per-step visualizations of probability and visible-canvas changes.

Case location: `datasets/semantic_intervention/`.
Shared prerequisites: `docs/PROPOSAL_PLAN.md`.

Implementation: `intervention.py` and `metrics.py`. Use `SemanticSwap` as the
`intervention` argument to `exp_code.common.dg_runner.run_fixed_trajectory`.
