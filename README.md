# allm-dgsc

Causal Probing and Editable Reasoning in DiffusionGemma.

Research scaffold for the Algorithms in Large Language Models final project.
Scope: inference-time experiments only; no training or fine-tuning.

**Status:** upstream source imported; experiments and datasets are not yet
implemented. There are no experiment runners, generated cases, or results.

## Layout

```text
allm-dgsc/
  dg/                         Original Google DeepMind Gemma repository
    gemma/diffusion/           DiffusionGemma model and sampler implementation
  exp_code/
    semantic_intervention/    Experiment 1: semantic SC interventions
    positional_intervention/  Experiment 2: positional and joint interventions
    dynamic_revision/         Experiment 3: revision after context changes
  datasets/
    semantic_intervention/    Cases for experiment 1 (empty)
    positional_intervention/  Cases for experiment 2 (empty)
    dynamic_revision/         Cases for experiment 3 (empty)
  docs/
    PROPOSAL_PLAN.md           Shared prerequisites and proposal mapping
    UPSTREAM.md                Source URL, pinned commit, and integration notes
    DiffusionGemma_Final_Project_Proposal.docx
```

Each experiment directory contains only a planning README. Dataset directories
contain only Git placeholders. Actual implementation starts after baseline
inference and per-step tracing are verified.

## Experiments

1. **Semantic intervention:** change selected token probabilities passed through
   SC while holding the current visible canvas fixed; measure counterfactual
   downstream consistency.
2. **Positional intervention:** move or anchor candidate token probability mass
   and measure coordinated responses at neighboring positions.
3. **Dynamic revision:** reopen a converged canvas under updated context and
   measure required edits, unnecessary edits, and preservation.

Shared prerequisites include full/disabled/sparse SC baselines, matched controls,
fixed-step trajectories, and logging of canvas, logits, probabilities, SC,
entropy, and keep/renoise decisions. See [the proposal plan](docs/PROPOSAL_PLAN.md).

## Upstream

`dg/` contains the full, unchanged source tree from
[google-deepmind/gemma](https://github.com/google-deepmind/gemma), including its
licenses, tests, examples, and documentation. It is tracked directly in this
repository using a Git subtree, so a normal clone includes all source files.
See [source provenance](docs/UPSTREAM.md) and [the upstream README](dg/README.md).

Model weights are separate and have not been downloaded. Follow upstream setup
instructions when beginning implementation; this scaffold does not yet claim a
working inference environment.

## Data and Outputs

Start with small controlled cases and then generated variants, as proposed.
Dataset requirements are documented in [datasets/README.md](datasets/README.md).
There are no downloaded benchmarks or fabricated experiment results.

Future runtime outputs belong in `outputs/`, which is ignored by Git. Keep model
weights, caches, and large tensors out of the repository.
