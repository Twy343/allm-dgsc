# Execution status

Status captured 2026-11-16. Project: ALLM / DiffusionGemma SC only.

## Alternative GPU route

The user requested use of 6000-series GPUs rather than waiting only for H100.
An actual RTX A6000 allocation (13935944) passed a CUDA/JAX computation and
reported 49,140 MiB VRAM. Two-card checkpoint loading/generation is not yet
verified. See GPU_OPTIONS.md. New pending jobs are 13936244 (first formal batch
on two A6000s) and 13936227 (continuation after a successful first batch).
The original H100 benchmark 13934635 and separate beta job 13933040 are held.
The first-batch and continuation scripts release H100 on failure; the complete
A6000 run cancels H100 on success. They do not run the same benchmark concurrently.

## Latest user revision and replacement jobs

The model download and 33-object integrity verification have completed. Manifest
digest: `518abf72853c48679c8c412c928ae2cd09a0b61e284ad0bdbd5b7131fd23c48c`.
The user requests 256 test cases per task and five sequential conditions: Full,
zero SC at 48/96 steps, top-k=8, threshold=.03, with batching and entropy traces.
See TEST256_PROTOCOL.md. This supersedes the earlier pilot/full-test plan.

- Old score/pilot job 13932936 was canceled while pending; it never generated
  benchmark predictions.
- New CPU input/batched-entropy validation 13934626 completed: 25 tests passed,
  tiny native/batched entropy equivalence passed, all 256 selected MBPP reference
  solutions passed, and 256-case manifests for both tasks were saved.
- New batched benchmark: 13934635, one H100, batch size 4, burst partition/QOS,
  validation dependency is satisfied; currently pending for GPU priority.
  Results: `outputs/sc-test256-v2`. No new benchmark output exists yet.
- Existing beta-sweep job 13933040 is moved to burst and chained after 13934635.
- There is no recurring automation. Download/verification job 13933051 completed
  successfully. Older sections below are historical snapshots, not live status.
- Pending GPU wall-time request reduced to four hours after scheduler probes
  showed an earlier backfill opportunity than the initial eight-hour ceiling.

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
| 13932907 | Original slow checkpoint download on ins091 | Canceled after safely retaining partial files |
| 13933051 | Verify completed files / resume any missing bytes | Held during fast transfer; released automatically by transfer script |
| 13932936 | Native/Full SC checkpoint gate, smoke, then 64-per-task four-mode pilot | Waiting for verified checkpoint chain |
| 13933040 | New beta and bottleneck first-canvas entropy sweeps | Waiting for verified checkpoint chain |

Download inventory: 33 objects totaling 40,412,959,868 bytes, including tokenizer;
approximately 2.99 TB was free before download. CPU download jobs use no GPU.
Running-job time extension was denied by Slurm, so the queued continuation has
a four-hour limit and range-resumes partial objects rather than restarting them.
GPU jobs each request one H100: score job at most two hours, entropy at most one
hour. These are requested ceilings, not measured usage. No new GPU result is
claimed at capture.

## Download bottleneck diagnosed and corrected

On 2026-11-16, identical 8 MiB GCS range probes measured approximately 30.8 MB/s
on the login node versus 1.42 MB/s on ins091. The original four-stream transfer
averaged only a few MB/s. Disk capacity was sufficient. Before moving the writer,
the continuation was held and the original download job was canceled and confirmed
gone from the queue. Partial files were retained and resumed with pinned GCS
generations; already complete objects were not retransferred.

`scripts/transfer_checkpoint_login.sh` performs network transfer only under the
system Python and low scheduling priority, then releases job 13933051. CPU hash
verification remains on the compute allocation. Its PID is in
`outputs/setup/login-transfer.pid`; its log is
`/insomnia001/depts/edu/users/yc4953/logs/allm-checkpoint-login-transfer.log`.
This is a one-off recovery script for these job IDs, not a generic submitter.

After moving the transfer, on-disk byte counts rose from 31,181,972,862 to
33,228,100,990 in 10 seconds (approximately 205 MB/s aggregate). This measured
window is not a guaranteed sustained rate. There is no recurring monitor; jobs
continue through their existing Slurm dependency chain.

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
