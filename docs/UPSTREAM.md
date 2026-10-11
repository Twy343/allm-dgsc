# Upstream Source

- Repository: https://github.com/google-deepmind/gemma
- Branch at import: `main`
- Pinned commit: `0f7e7420e84c6789d96ba465608d2a008ecb443d`
- Import date: 2026-10-11
- Local prefix: `dg/`
- Integration: Git subtree with a squashed upstream history
- Source modifications at initialization: none

Google's developer guide links its research integration to this Gemma library:
https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/

## Relevant Entry Points

- `dg/gemma/diffusion/_models.py`: DiffusionGemma model definition.
- `dg/gemma/diffusion/_transformer.py`: diffusion and self-conditioning support.
- `dg/gemma/diffusion/_sampler.py`: per-step outputs and canvas denoising loop.
- `dg/gemma/diffusion/_early_stopping.py`: convergence and stopping behavior.
- `dg/gemma/diffusion/_chat_sampler.py`: conversational sampling interface.

These are implementation references, not experiment entry points. The original
library includes other Gemma models because DiffusionGemma depends on shared
model code.

## Reproducibility

The imported `dg/` tree must match the pinned upstream commit. Model checkpoint,
tokenizer, dependency versions, seeds, and sampling settings still need to be
recorded when runnable experiments are added.

The `upstream` remote points to the original repository; `origin` points to
`Yanzhen-CN/allm-dgsc`. Future upstream updates should be deliberate and recorded
with the new commit ID. Do not automatically update the research baseline.

## Licenses

Upstream license and attribution files are retained in `dg/`, including
`dg/LICENSE`. Consult upstream terms for its code and model checkpoints.
