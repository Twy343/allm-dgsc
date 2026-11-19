# allm-dgsc

Causal Probing and Editable Reasoning in DiffusionGemma.

Research scaffold for the Algorithms in Large Language Models final project.
Scope: inference-time experiments only; no training or fine-tuning.

**Status (2026-11-19):** six SC conditions are complete on the same 64 GSM8K
and 64 MBPP cases; see [benchmark results](docs/SC_STRENGTH01_RESULTS.md).
These are controlled subset results, not reproduction of the paper's full suite.

The first research direction now has an archived single-case study:
[four intervention operators, observations and evidence](results/interim_sc_case_study/README.md).
It includes 162 completed perturbation configurations on MBPP277, matched
controls, 31 selected step traces, three figures, and exact execution-source
provenance. The evidence supports case-level conclusions, not population rates.
See [method taxonomy](docs/INTERVENTION_TAXONOMY.md) for what counts as an operator.

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
    semantic_intervention/    Cases and selection metadata for experiment 1
    positional_intervention/  Cases for experiment 2
    dynamic_revision/         Cases for experiment 3
  results/
    interim_sc_case_study/    Curated midterm evidence, figures and provenance
    intervention_taxonomy/    Operator / region / design-axis condition mapping
  docs/
    PROPOSAL_PLAN.md           Shared prerequisites and proposal mapping
    UPSTREAM.md                Source URL, pinned commit, and integration notes
    DiffusionGemma_Final_Project_Proposal.docx
```

`exp_code/common/` now provides a fixed-step trajectory runner, compact tracing,
and full/disabled/top-k/threshold SC modes. The three experiment directories
contain intervention or evaluation operators. The validated resident debugger
used for the archived study is included in its exact execution-source package;
its scope is canvas0. Context/KV refresh and multi-canvas resume are not yet
validated by that debugger.

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
and file integrity verification. The archived study records checkpoint identity,
environment versions, native-prefix checks and deterministic branch replays.

## Data and Outputs

Start with small controlled cases and then generated variants, as proposed.
Dataset requirements are documented in [datasets/README.md](datasets/README.md).
Cluster benchmarks are loaded from an external dataset directory. The curated
case archive includes its four-case input batch, prompts, public tests and real
generated traces for reproducibility; bulk datasets and weights remain external.

Runtime outputs belong in `outputs/`, which is ignored by Git. Selected evidence
is tracked under `results/`; verify its checksums with
`python scripts/archive_interim_results.py --verify`. Keep model weights,
caches, and large tensors out of the repository.

## Development Check

Run the lightweight tests without model weights:

```powershell
python -m unittest discover -s exp_code/tests -v
```

End-to-end runs require the upstream dependencies under `dg/`, an accepted
DiffusionGemma checkpoint license, sufficient accelerator memory, and a locally
available checkpoint. Exact versions for the archived run are recorded in
`results/interim_sc_case_study/provenance/`.
