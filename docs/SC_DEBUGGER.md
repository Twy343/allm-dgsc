# SC breakpoint console

Purpose: keep one model loaded, pause after denoising step t, fork several
SC interventions from exactly the same state, then export actual continuations.

The first version supports canvas0 only. It is sufficient for the GSM480
example, whose new unmodified response completes in that canvas. It does not
claim to produce a complete answer when a case requires additional canvases.

## Commands

Steps use one-based completed-step counts. Enter `break 5` then `continue`
on SEPARATE lines (semicolons are not parsed). This stops AFTER
step5: its sampled canvas already exists, while its SC has not been consumed
by step6. `swap` changes only that SC, not the canvas or RNG.

```
load gsm8k:480
break 5
continue
save base
inspect 104
continue
export unchanged
restore base
swap 104 236778 236812
step
inspect 104
continue
export swap_2_to_4
restore base
```

For a position already accepted in the unmodified trace:

```
restore base
break 10
continue
save committed
inspect 104
swap 104 236778 236812
step
inspect 104
continue
export after_commit_swap
```

Inspect the actual acceptance field; do not assume a new device or trajectory
has accepted the position at the same step. Fully stopped samples cannot be
reopened by this console; restore a breakpoint before adaptive termination.
`drop NAME` frees a snapshot; at most4 are retained. `quit` releases the GPU.

## State and reproducibility

Weights and the prefix KV cache are retained by one worker. In-memory snapshots
share immutable JAX arrays and preserve the canvas, SC embedding and full latest
SC logits, RNG, step/noise schedule position, stop flags, compact trace prefix,
and edit history. Restore copies mutable host bookkeeping so repeated branches
cannot mutate the saved breakpoint. Token IDs/positions are dynamic inputs to
the swap kernel, allowing reuse of its compilation across different swaps.

Each branch export has steps.csv/tokens.csv/NPZ plus exact edit values and a
partial-trajectory flag. Session metadata records checkpoint/source/device/JAX
identity. Commands are journaled. Snapshots are not disk checkpoints: terminating
the worker or Slurm allocation destroys them. Response canvas logits are held on
the GPU; full model weights are never copied into each snapshot. For an active
research block idle-seconds0 keeps the worker resident until quit or Slurm
expiry. An optional positive idle timeout is available for other sessions.

Additional interventions:

```
mix 104 0.5 236812:1
canvas 104 236812
zero 104
region START ALPHA TOKEN_ID TOKEN_ID ...
text
```

`mix` sets next-step SC to a mixture of its current distribution and a custom
normalized sparse q (ID:weight entries). `canvas` changes only the visible
token. `zero` overrides the targeted SC embedding, retaining logits for
inspection rather than claiming they represent that zero input. Restore before
independent conditions; sequence commands only for intentional joint edits.
All SC commands log the actual embedding delta, and mix logs probability
rounding error. See ITERATIVE_INTERVENTION_PLAN.md for causal comparisons.
`region` applies aligned q templates to a contiguous SC block atomically,
preserving the canvas and other SC rows. Invalid input rolls back the whole
operation. It replaces distributions at fixed positions and never inserts tokens.

`--verify --explore` performs a bounded committed-site probe after checkpoint
validation, records baseline/SC-only/canvas-only/joint effects over at most5
active steps, optionally tests stronger q/zero-SC after a local null, restores
the committed breakpoint, then leaves the model resident at the console.

## Gates

CPU job13992323 exercises a real tiny JAX/Flax diffusion model: native canvas
parity, exact restored no-op continuation, repeated identical swap branches,
unchanged canvas/RNG during SC edits and rejection after a canvas has stopped.
Job13992323 passed all3 tests;13996164 passed the expanded4 tests, and
13996366 passed all5 including bounded exploratory branching/restoration.
These are real tiny-model checks, not full-checkpoint mechanism conclusions.

`python -m exp_code.semantic_intervention.debug_console --verify` additionally
requires actual-checkpoint native first-canvas parity, restored no-op identity
and repeated-swap identity before entering the resident console. The console
starts at the verified GSM480 `base` breakpoint after step5. A CPU pass alone
does not establish that checkpoint prefill and the GPU debugger are validated.


## Saved-logit scope

CSV/NPZ exports contain full-position entropy/selection/canvas plus TOP16 logits,
probabilities and token identities. The original30 traces also have35 watched
candidate probabilities/logits. They do NOT persist every vocabulary logit at
all steps. Current full vocabulary logits are available in resident state and
named snapshots for arbitrary edits; those GPU snapshots expire with process.
For a candidate absent from an exported top16, review reports a probability
upper bound, never an invented zero. Full targeted-row archival needs a future
validated export command; do not describe compact exports as all-logit dumps.

`continue` completes the whole original batch, potentially including slower
other members after the focus sample stops. Bounded `step`/`until` and focus
stop checks are preferable for initial1/3/5-step comparisons.
