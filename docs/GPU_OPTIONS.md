# GPU eligibility and two-A6000 path

On 2026-11-16 a real allocation (13935944, ins091) reported NVIDIA RTX A6000,
49,140 MiB and driver 575.57.08. The existing ALLM JAX environment recognized
the CUDA device and completed an accelerator sum. This establishes the runtime,
not checkpoint capacity or full benchmark correctness. The separately named
R6000 nodes have not yet had their exact physical card model measured.

Limiting the initial submission to H100 was too restrictive. The workload may
use another GPU model if memory and generation correctness pass. Download size
(40.4 GB) is not peak VRAM use; restore must also accommodate the uncompressed
parameter arrays, KV cache and activations. A single A6000 has little margin.

The alternative uses TWO A6000s and the already installed Kauldron
`FSDPSharding` rule through the original Gemma checkpoint loader, preserving
weights/dtypes. Inputs and cache are replicated; large parameter tensors are
partitioned using the upstream rule. No quantization or SC-method rewrite is
introduced. Restore concurrency is bounded at 4 GB to reduce host memory needs.

The model loader reports logical parameter bytes and device memory statistics.
The batched native-versus-Full SC gate and first instrumented batch must pass on
the actual allocation before claiming the checkpoint works on this hardware.
All five conditions run on the same hardware/configuration; timing results from
H100 and A6000 are not merged.

Jobs:

- 13936244: a 30-minute allocation requesting 2 A6000s and 32 GB host RAM; run
  the first formal batch (not a new pilot), including the real checkpoint gate,
  from the exact same 256-per-task sample plan.
- 13936227: the four-hour A6000 continuation, dependent on 13936244 succeeding;
  resumes at the first unwritten batch. Results: `outputs/sc-test256-a6000-v1`.
- 13934635: existing single-H100 fallback is held. If the A6000 run fails, the
  shell releases this fallback. Successful complete A6000 evaluation cancels it.
- 13933040: separate old beta visualization is held while this route is tested.

The short first allocation is used because a four-hour request had a later
backfill window. The scheduler estimated a same-day start for the short request;
estimates can change. No recurring monitor is enabled. Read the logs and confirm
actual first-batch scores/entropy files rather than infer success from RUNNING.
