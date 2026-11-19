# Six-condition results, 64 cases per benchmark

All six conditions complete, verified 2026-11-18. Only 128 new outputs were
generated for the post-MLP branch x.1 group. The prior five conditions were
retained without regeneration or score changes.

| Condition | GSM8K exact match | MBPP single-sample pass@1 |
|---|---:|---:|
| Full SC, 48 steps | 62/64 (96.88%) | 37/64 (57.81%) |
| Disabled, 48 steps | 21/64 (32.81%) | 6/64 (9.38%) |
| Disabled, 96 steps | 34/64 (53.12%) | 8/64 (12.50%) |
| Top-k=8, 48 steps | 60/64 (93.75%) | 35/64 (54.69%) |
| Threshold=.03, 48 steps | 61/64 (95.31%) | 33/64 (51.56%) |
| SC branch x.1, 48 steps | 56/64 (87.50%) | 21/64 (32.81%) |

New intervention: `post_norm(canvas + .1 * ffw(pre_norm(SC)))`, matching
DGtest SCALING.md. It reduces the branch contribution after its own norm/MLP,
so its magnitude change is not canceled by pre_norm. It is not logit scaling
or the earlier pre-normalization beta observation.

The same64 cases/task, prompt hashes, seeds, batch4 index/position, max1024,
canvas256 and baseline checkpoint were checked across all six groups.
The new group reuses the original parameter tree with a module subclass.
Real tiny-model tests passed, followed by real-checkpoint SC-module checks:
beta1 exact identity; beta0 agrees with zero SC; beta.1 agrees with manual
post-MLP scaling and changes the output. Generation itself uses the whole model.

Original codemerge prompt, seed42, max256, one canvas, matched across six:
Full11, disabled48 reaches cap48 without meeting stop rule, disabled96 stop91,
top-k19, threshold8, branchx.1 stop32. Stopping is not proof of code correctness.
Independent observed-step axes and shared 0–4 color scale follow the old figure.

Interpretation: reduced SC still carries useful signal and beats disabled SC
in these samples. It does not retain Full performance: -9.375 percentage points
on GSM8K and -25 points on MBPP, with later stopping in this one example.
Combined with sparse-bottleneck results, this distinguishes selective information
restriction from globally reducing SC's relative contribution. It does not
establish that only high-confidence tokens contain useful information, that the
model ignores amplitude, or statistical equivalence between conditions.

All 128 new outputs were CPU-scored using the same matched numeric extraction,
DGtest-style code cleanup and 10-second MBPP timeout as the old five. No scores
were computed in the new GPU generation loop. GPU13977993 completed01:11:20;
CPU13977994 completed00:00:22, both exit0. GPU released.

Raw new group: outputs/sc-strength01-64-v1.
Combined768 rows/table/PNG/PDF: outputs/sc-sixconditions64-v1.
Both folders downloaded locally. Validation confirmed pairing, input hashes,
all old640 text/score fields unchanged, old5 entropy NPZ byte identity, and
all6 real entropy arrays/flags plus visual inspection of the final figure.
