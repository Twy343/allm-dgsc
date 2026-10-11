# Experiment 3: Revision Under Changing Context

Status: planned; no experiment code implemented.

Converge a canvas under initial context `x_0`, change the premise or external state
to `x_1`, and reopen the completed canvas for denoising under the updated context.
Prompt/KV recomputation is allowed. Define SC reuse or reset explicitly during
implementation and record that choice.

Prefer cases where the final conclusion stays the same but local reasoning must
change. Track whether SC reorganizes before or alongside visible edits.

## Controls

- Unmodified continuation under the original context.
- Matched irrelevant context changes when appropriate.
- Full regeneration under the updated context.

## Measurements

- Required-edit recall and unnecessary-edit rate.
- Preservation of unaffected content.
- Final logical correctness.
- Denoising steps to reconvergence.
- Timing of SC changes relative to visible edits.

Case location: `datasets/dynamic_revision/`.
Shared prerequisites: `docs/PROPOSAL_PLAN.md`.
