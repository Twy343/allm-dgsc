# Execution status

Status captured 2026-11-16. Project: ALLM / DiffusionGemma SC only.

## Verified

- SSH key authentication works with `id_ed25519_insomnia`; the initial transient
  connection errors were resolved. No password or Duo bypass was used.
- Local 22 unit tests and Python compilation passed.
- Cluster job 13933039 completed successfully: 22 tests; Matplotlib 3.11.2;
  JAX 0.11.2 CPU tiny real DiffusionGemma model verified SC-only boundaries,
  upstream full-loop equality, beta scaling, 48-step schedule, adaptive trace
  agreement, and beta=0 versus disabled equality.
- Job 13932930 completed: fixed smoke/pilot samples for both datasets and all
  64 selected MBPP reference solutions passed the scorer's test execution.
- Earlier four-mode tiny validation job 13932908 also completed.
- Existing PNG/GIF inspected and indexed; see PRIOR_SC_RESULTS.md.

## Active and queued

| Job | Purpose | State at capture |
| --- | --- | --- |
| 13932907 | Download official GCS checkpoint, pin generations and verify hashes | Running on ins091 |
| 13933051 | Continue a partial download if needed; exit immediately if already verified | Waiting for 13932907 to end |
| 13932936 | Native/Full SC checkpoint gate, smoke, then 64-per-task four-mode pilot | Waiting for verified checkpoint chain |
| 13933040 | New beta and bottleneck first-canvas entropy sweeps | Waiting for verified checkpoint chain |

Download inventory: 33 objects totaling 40,412,959,868 bytes, including tokenizer;
approximately 2.99 TB was free before download. CPU download jobs use no GPU.
Running-job time extension was denied by Slurm, so the queued continuation has
a four-hour limit and range-resumes partial objects rather than restarting them.
GPU jobs each request one H100: score job at most two hours, entropy at most one
hour. These are requested ceilings, not measured usage. No new GPU result is
claimed at capture.

Both GPU dependencies were updated after the latest successful validation.
Successful dependencies may disappear from `scontrol` once satisfied; the
remaining dependency is the checkpoint continuation job. No full test benchmark
was submitted automatically.

## Startup failures resolved

- 13932881 and 13932888 failed immediately because non-login batch shells lacked
  `module`. 13932900 failed because the old anaconda modulefile was unavailable.
- 13932899 was canceled during the activation investigation and replaced by
  successful direct-environment validation. New scripts use the existing
  `envs/allm` executable/library paths and do not require that modulefile.
- No system driver was modified. No SLLM environment or job was touched.

## Where to continue

Remote repository: `/insomnia001/depts/edu/users/yc4953/projects/allm-dgsc`.
Results: `outputs/sc-baseline-v1/{smoke,pilot}` and `outputs/sc-entropy-v1`.
Logs: `/insomnia001/depts/edu/users/yc4953/logs/allm-*.log`.
Checkpoint: `/insomnia001/depts/edu/users/yc4953/models/diffusiongemma-26B-A4B-it`.
Always set `SLURM_CONF=/etc/slurm/slurm.conf` for Slurm queries.

After jobs complete, verify `full-sc-gate.json`, summary completion counts,
checkpoint manifest, generated figures and entropy cap flags before reporting
results. Inspect pilot token truncation and score parsing before expanding the
test set. Preserve the original GIF as earlier evidence, not a new reproduction.
