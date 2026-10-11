"""Normalization-preserving semantic counterfactual interventions."""

from __future__ import annotations

import dataclasses
from typing import Any

from exp_code.common import probability


def swap_token_probabilities(
    logits: Any,
    *,
    batch_index: int,
    position: int,
    source_token_id: int,
    target_token_id: int,
    fraction: float = 1.0,
) -> Any:
  """Interpolates a probability swap between two tokens in one row.

  `fraction=1` exchanges their probabilities exactly. All other token
  probabilities remain unchanged and the row stays normalized.
  """
  if source_token_id == target_token_id:
    raise ValueError('source and target token IDs must differ')
  if not 0.0 <= fraction <= 1.0:
    raise ValueError('fraction must be in [0, 1]')
  xp = probability.array_module(logits)
  row_logits = logits[batch_index, position, :]
  row_probs = probability.softmax(row_logits)
  source_probability = row_probs[source_token_id]
  target_probability = row_probs[target_token_id]
  new_source = (
      (1.0 - fraction) * source_probability + fraction * target_probability
  )
  new_target = (
      (1.0 - fraction) * target_probability + fraction * source_probability
  )
  if probability.is_jax_array(logits):
    row_probs = row_probs.at[source_token_id].set(new_source)
    row_probs = row_probs.at[target_token_id].set(new_target)
  else:
    row_probs = xp.array(row_probs, copy=True)
    row_probs[source_token_id] = new_source
    row_probs[target_token_id] = new_target
  new_row = probability.probabilities_to_logits(row_probs)
  return probability.replace_row(logits, batch_index, position, new_row)


@dataclasses.dataclass(frozen=True)
class SemanticSwap:
  """A one-step hook compatible with `run_fixed_trajectory`."""

  intervention_step: int
  position: int
  source_token_id: int
  target_token_id: int
  batch_index: int = 0
  fraction: float = 1.0

  def __call__(self, step: int, logits: Any) -> Any:
    if step != self.intervention_step:
      return logits
    return swap_token_probabilities(
        logits,
        batch_index=self.batch_index,
        position=self.position,
        source_token_id=self.source_token_id,
        target_token_id=self.target_token_id,
        fraction=self.fraction,
    )
