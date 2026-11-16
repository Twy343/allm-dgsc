# Proposal Implementation Plan

Source: `DiffusionGemma_Final_Project_Proposal.docx` in this directory.
This document tracks planned work and the first implemented infrastructure.

## Shared Prerequisites

Before the three main experiments:

- Run unmodified DiffusionGemma inference at the pinned source version.
- Reproduce full SC versus disabled SC (`S_t = 0`).
- Evaluate sparse SC using top-k candidates, including `k = 8`, and probability
  thresholding, including `p = 0.03`; preserve retained probabilities and
  uniformly redistribute discarded mass as in the paper's Appendix C.
- Inspect the actual upstream SC computation before defining intervention
  boundaries. Use the proposal's token-space formulation
  `P_t = softmax(L_t / tau_t)` and `S_t = P_t W_E` where applicable.
- Record canvas, logits, token probabilities, SC states or token projections,
  entropy, and sampler keep/renoise decisions at every denoising step.
- Disable or standardize adaptive early stopping for comparable trajectories.
- Record prompt, case ID, seed, checkpoint, tokenizer, temperature schedule,
  sampler settings, intervention step, and intervention positions.
- Shared fixed-step tracing, SC modes, and intervention helpers are implemented
  under `exp_code/common/`; checkpoint-backed validation remains pending.

## Experiment Mapping

| Proposal direction | Code directory | Data directory |
| --- | --- | --- |
| Semantic causal interventions on SC | `exp_code/semantic_intervention/` | `datasets/semantic_intervention/` |
| Positional and joint interventions | `exp_code/positional_intervention/` | `datasets/positional_intervention/` |
| Revision under changing context or state | `exp_code/dynamic_revision/` | `datasets/dynamic_revision/` |

Detailed controls and metrics are listed in each experiment README.

## Planned Sequence

1. Verify baseline inference and expose per-step trajectories.
2. Reproduce SC baseline behavior.
3. Run one semantic intervention with matched controls.
4. Add positional intervention and neighborhood measurements.
5. Add updated-context revision, allowing prompt/KV recomputation.
6. Expand successful case studies to generated variants before making broad
   claims about causal behavior.

## Interpretation and Scope

Interventions should preserve probability-row normalization and avoid arbitrary
extreme vectors. Compare against unmodified rollouts, matched random
interventions, and equal-magnitude irrelevant interventions. Use identical seeds
and standardized sampling settings where appropriate.

Dynamic revision begins with controlled context updates and extends to repeated
updates that change the required answer. Refresh prompt conditioning/KV, reopen
the existing canvas, and compare SC reuse/reset with fresh regeneration. Measure
correctness, required/collateral edits and reconvergence steps/latency before
making efficiency claims.

No training, fine-tuning, large benchmark construction, or efficiency claims are
part of the initial scaffold. Reproduction is an implementation check; the novel
investigation begins with causal interventions on inter-step state.

The current stage-one protocol and execution instructions are in
[BASELINE_PLAN.md](BASELINE_PLAN.md).
