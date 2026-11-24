# Direction 2: Positional Self-Conditioning Pilot

Five configurations completed validation; one was skipped because its eligibility threshold was not met. Interventions caused transient probability and token-acceptance changes. One half-dose condition retained variable renaming, and both full-dose conditions stopped one denoising step earlier with unchanged final output.

**The pilot did not demonstrate persistent positional following. Runtime and task-accuracy improvements remain untested.**

## Experiment

The intervention moves probability mass for a selected token between self-conditioning (SC) positions 42 and 43, then observes how generation responds.

| Setting | Value |
|---|---|
| Task and seed | MBPP277, seed `100008` |
| Batch | `[mbpp:133, mbpp:464, mbpp:277, mbpp:59]`; focus index `2` |
| Intervention | Quote or return token; after step 2 or 4; half or full transfer |
| Generation | First 256-token canvas; original 48-step maximum and native adaptive stopping |
| Model settings | Unchanged checkpoint, BF16 precision and native sampler |

Each eligible configuration runs five branches from the same restored breakpoint: baseline, no-op, forward, reverse and repeated forward. Each branch requests up to five follow-up steps; native stopping can shorten observation. Missing steps after stopping are left unobserved.

The frozen settings are in [direction2_pilot.json](../configs/direction2_pilot.json). Quote interventions require probabilities of at least 0.01 at both positions; return interventions require at least 0.2 at position 42. The focus must also be active, with finite logits and editable rows.

## Results

Stop counts are completed denoising steps. B, F and R denote baseline, forward and reverse. Final outputs are compared with each configuration's own baseline.

| Configuration | Status | B / F / R stop | Final output |
|---|---|---|---|
| `t2_quote_half` | Validated | 6 / 6 / 7 | Both edited directions match baseline. |
| `t2_return_half` | Validated | 6 / 6 / 7 | Forward retains variable renaming; reverse matches baseline. |
| `t4_quote_half` | Skipped | — | Quote probabilities were below the eligibility threshold. |
| `t4_return_half` | Validated | 6 / 6 / 6 | Both edited directions match baseline. |
| `t2_quote_full` | Validated | 7 / 6 / 6 | Both edited directions match baseline. |
| `t2_return_full` | Validated | 7 / 6 / 6 | Both edited directions match baseline. |

In every completed configuration, the no-op branch matched baseline and the repeated-forward branch matched forward.

The only persistent final-canvas change was in `t2_return_half`, where forward and its repeat changed five token positions: 44, 46, 48, 50 and 57.

```diff
-return {key: value for key, value in d.items() if value >= n}
+return {k: v for k, v in d.items() if v >= n}
```

The anchor token did not remain shifted from position 42 to 43. Across the completed conditions, anchor locations returned toward their session-specific baseline positions. Transient probability, sampled-token and acceptance differences mostly disappeared before the final canvas.

Both full-dose configurations stopped one step earlier in both edited directions, with identical final token IDs and decoded output relative to baseline. No controlled latency or full-batch speedup measurement was performed.

## Validation and limitations

All five completed configurations passed the recorded native/debugger and matched-native MBPP checks, baseline/no-op equality, forward/repeat equality, intervention isolation and final restoration. Subsequent checks revalidated the saved traces, arrays, coverage and dose measurements. These exported artifacts cannot reconstruct the full GPU state.

The five conditions produced 25 branches, but all observations concern **one task and one seed**. Branches and denoising steps are not independent replications.

The main limitations are:

- **Baseline reproducibility:** fresh processes produced different baselines. Even two H100 workers with identical saved step-2 fingerprints later differed in baseline acceptance/output entries. The cause remains unresolved.
- **Hardware and session differences:** half-dose results came from the R6000-labelled run; full-dose results came from H100 runs. Their comparison cannot isolate the effect of dose.
- **Realized dose:** BF16 rounding changed the actual probability transfer, and equal forward/reverse fractions did not imply equal absolute doses. The step-4 return reverse branch produced zero change in the SC embeddings after rounding, so it cannot establish resistance to a substantial edit.
- **Scope:** only the first canvas was evaluated. Full-response MBPP correctness and accuracy were not measured.

## Supporting evidence

The full results ZIP is shared separately. It contains the original detailed report, raw outputs, plots, measurement tables, validation records and recovery history. Those supporting files are outside this GitHub branch; the paths below refer to files **inside the ZIP**.

| Evidence | Archive path |
|---|---|
| Original detailed report | `docs/DIRECTION2_RESULTS.md` |
| Completion and skip records | `outputs/direction2-completion-20261121-2252/completion_manifest.json` |
| Comparisons across processes | `outputs/direction2-inspection-20261122/inspection.json` |
| Final execution and memory validation | `docs/DIRECTION2_RECOVERY_14048218.md` |

Per-condition analysis is under:

```text
outputs/direction2-completion-20261121-2252/terminal_analysis_1795331982965267822/
```

Each completed configuration has a subdirectory containing `measurements.json`, `paired_probabilities.csv`, `token_moments.csv`, `decoded_canvases.json` and plots. The earlier interrupted attempt is preserved separately and excluded from completed results.

The recorded scientific source digest is:

```text
b415c812e7a0914b2a629d4537d11939eb222230af46e17cd36f19f3fdadcd30
```

This identifies the original experiment snapshot, including files omitted from the trimmed GitHub handoff.

## Next step

Check baseline reproducibility across fresh processes on fixed hardware using the same batch and seed. If earlier stopping with identical output repeats, evaluate it with a controlled timing experiment using matched warm-up and full-batch completion.
