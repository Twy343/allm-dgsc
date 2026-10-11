# Experiment 1: Semantic Causal Interventions

Status: planned; no experiment code implemented.

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
