# allm-dgsc

Causal Probing and Editable Reasoning in DiffusionGemma.

Research scaffold for the Algorithms in Large Language Models final project.
Scope: inference-time experiments only; no training or fine-tuning.

**Status:** stage-one SC bottleneck implementation and evaluation entry point are
available. Cluster validation and checkpoint acquisition are in progress; no
capability reproduction result is claimed. See [baseline protocol](docs/BASELINE_PLAN.md).

The immediate run is now 256 fixed test cases per task, batch size 4, five
sequential conditions (including zero SC at 48 and 96 steps), with entropy saved
for every sample. See [current protocol](docs/TEST256_PROTOCOL.md). The original
64-case pilot submission was replaced; no recurring monitor is enabled.

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

`exp_code/common/` now provides a fixed-step trajectory runner, compact tracing,
and full/disabled/top-k/threshold SC modes. The three experiment directories
contain their first intervention or evaluation operators. Dataset directories
remain empty placeholders until controlled cases are selected.

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

Model weights are stored separately. `exp_code/baselines/download_checkpoint.py`
downloads the upstream public Orbax checkpoint with pinned object generations
and file integrity verification. Existing cluster environment `allm` is used by
the Slurm scripts; checkpoint-backed generation still needs its correctness gate.

## Data and Outputs

Start with small controlled cases and then generated variants, as proposed.
Dataset requirements are documented in [datasets/README.md](datasets/README.md).
Cluster benchmarks are loaded from the user's external dataset directory;
there are no raw benchmarks or fabricated experiment results committed here.

Future runtime outputs belong in `outputs/`, which is ignored by Git. Keep model
weights, caches, and large tensors out of the repository.

## Development Check

Run the lightweight tests without model weights:

```powershell
python -m unittest discover -s exp_code/tests -v
```

End-to-end runs require the upstream dependencies under `dg/`, an accepted
DiffusionGemma checkpoint license, sufficient accelerator memory, and a locally
available checkpoint. Those environment-specific steps are not yet configured.
