# What SC is in the pinned DiffusionGemma implementation

Primary evidence is the checked-in author implementation, not an analogy to
autoregressive hidden-state caching. `_sampler.sample_step` shapes raw logits
by the current temperature, samples/renoises the visible canvas, and separately
encodes those shaped logits for the NEXT denoising step.

Let L_t be the shaped logits, p_t=softmax(L_t), E the input embedding table,
and C the visible canvas. `Embedder.encode_logits` computes

    s_t = sqrt(D) * p_t E

with the probabilities cast back to the logits dtype before the embedding
contraction. The next transformer input is

    RMSNorm_post( embed(C_{t+1}) + MLP(RMSNorm_pre(s_t)) ).

The forward pass uses this input, positions and the prefix KV cache to produce
fresh logits for all canvas positions. SC has token content and uncertainty
information compressed into a dense vector, but is not a direct hard token
command. The vocabulary-to-embedding projection is many-to-one: distributions
can differ greatly while their projected vectors differ little.

For a full probability swap between a and b, in exact arithmetic:

    delta(s) = sqrt(D) * (p_a-p_b) * (E_b-E_a).

Its effect depends on embedding geometry, normalization and the learned MLP,
not only the probability moved. RMSNorm can suppress a simple upstream scale
change; the user-defined beta experiment therefore scales the branch AFTER
its MLP. A single-step edit also need not persist: the next model prediction
replaces SC again. The current canvas and unchanged prompt/KV can restore the
original task-consistent interpretation.

## Canvas acceptance is a separate path

The model predicts every position, including positions selected previously.
The sampler sorts position entropies and accepts a prefix for which cumulative
preceding entropy is within the0.1 budget; non-selected positions receive
uniform random vocabulary tokens. All positions are forced selected at the
last allowed step. This is not a permanent per-token commit and is not an
explicit logic/contradiction detector. Adaptive stopping freezes whole samples.

SC uses ALL position distributions, regardless of their current acceptance.
A meaningful candidate can exist in SC while the visible canvas at that site
is still random noise. This is a concrete reason to study SC separately from
visible text and to intervene before and after actual acceptance.

## What the current evidence establishes

The completed64-case/task benchmark shows a large effect of removing SC:
GSM full62/64 vs disabled48 21/64; MBPP37/64 vs6/64. Sparse top-k8 retains60/64
and35/64. These support global functional importance and tolerance of this
bottleneck; they do not prove all useful content resides in a few tokens or
that SC literally performs text-like reasoning.

In GSM480 the one-row SC swap changed later selection/layout/stop time, but
never produced/accepted4 at the targeted physical slot. The semantic result
remained4/2=2. Wording changes moved semantic token alignment, so physical-slot
acceptance delay must not be presented as delayed acceptance of semantic2.

## Next causal questions

1. Does perturbation survive projection? Measure actual SC embedding delta,
   angle and, subsequently, the MLP branch change relative to canvas input.
2. Does it change the next distribution, actual selected content, spatial
   organization, or only stopping? Trace each separately, with matched RNG.
3. Is weak effect a short-pulse recovery? Compare timing, regional coherence,
   intervention strength and prespecified repeated pulses only as justified.
4. Is recovery supplied by SC, canvas or prompt/KV? Compare isolated and joint
   edits. Updating the prompt requires genuine KV refresh, still unimplemented
   in the current debugger; do not infer internal-feature dominance without it.
5. Are controls comparable? Equal entropy/probability mass does not match
   embedding distances. The earlier random token `of` is an approximate
   vocabulary control, not a fully geometry-matched semantic control.

Source locations: dg/gemma/gm/nn/gemma4/_modules.py::Embedder.encode_logits,
dg/gemma/diffusion/_transformer.py::SelfConditioning and DiffusionMixin,
dg/gemma/diffusion/_sampler.py::SampleFromPredictions/sample_step/while carry.


## Paper cross-check (2026-11-18)

Primary paper: https://arxiv.org/html/2606.20560v1 , Sections2.1,3.1,5.2 and
AppendicesA/C. Its top-k/threshold intervention flattens discarded tail mass
uniformly while preserving retained probabilities; it is not renormalized
hard truncation. Local probability.py implements this convention (up to dtype
rounding). The paper's separate sequence-smearing observation motivates
realigning semantic spans before interpreting fixed-position1/2 competition.
Our recent GSM233 layout fork makes that check practically necessary.
These64/task scores adapt the method to GSM8K/MBPP, not the paper's capability
suite. Neither sparse vocabulary retention nor local recovery proves a full
algorithmic account of the model's reasoning.
