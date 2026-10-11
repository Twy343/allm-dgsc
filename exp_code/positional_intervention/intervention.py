"""Normalization-preserving positional interventions."""

from __future__ import annotations

import dataclasses
from typing import Any

from exp_code.common import probability


def _set_token_probability(row_probs: Any, token_id: int, value: Any) -> Any:
  """Changes one token probability and rescales all others proportionally."""
  xp = probability.array_module(row_probs)
  if row_probs.shape[0] < 2:
    raise ValueError('at least two vocabulary entries are required')
  old_value = row_probs[token_id]
  remaining = 1.0 - old_value
  tiny = xp.finfo(row_probs.dtype).tiny
  proportional = row_probs * (1.0 - value) / xp.maximum(remaining, tiny)
  uniform = xp.full_like(row_probs, (1.0 - value) / (row_probs.shape[0] - 1))
  updated = xp.where(remaining > tiny, proportional, uniform)
  if probability.is_jax_array(row_probs):
    return updated.at[token_id].set(value)
  updated = xp.array(updated, copy=True)
  updated[token_id] = value
  return updated


def shift_token_probability(
    logits: Any,
    *,
    batch_index: int,
    source_position: int,
    target_position: int,
    token_id: int,
    fraction: float = 1.0,
) -> Any:
  """Moves feasible probability mass for one token between two positions.

  The source token probability decreases and the target increases by the same
  amount. Other probabilities in each row are rescaled proportionally, so both
  rows remain normalized.
  """
  if source_position == target_position:
    raise ValueError('source and target positions must differ')
  if not 0.0 <= fraction <= 1.0:
    raise ValueError('fraction must be in [0, 1]')
  source_probs = probability.softmax(logits[batch_index, source_position, :])
  target_probs = probability.softmax(logits[batch_index, target_position, :])
  xp = probability.array_module(logits)
  transferable = xp.minimum(
      source_probs[token_id], 1.0 - target_probs[token_id]
  )
  delta = fraction * transferable
  source_probs = _set_token_probability(
      source_probs, token_id, source_probs[token_id] - delta
  )
  target_probs = _set_token_probability(
      target_probs, token_id, target_probs[token_id] + delta
  )
  result = probability.replace_row(
      logits,
      batch_index,
      source_position,
      probability.probabilities_to_logits(source_probs),
  )
  return probability.replace_row(
      result,
      batch_index,
      target_position,
      probability.probabilities_to_logits(target_probs),
  )


@dataclasses.dataclass(frozen=True)
class PositionalShift:
  """A one-step positional hook compatible with `run_fixed_trajectory`."""

  intervention_step: int
  source_position: int
  target_position: int
  token_id: int
  batch_index: int = 0
  fraction: float = 1.0

  def __call__(self, step: int, logits: Any) -> Any:
    if step != self.intervention_step:
      return logits
    return shift_token_probability(
        logits,
        batch_index=self.batch_index,
        source_position=self.source_position,
        target_position=self.target_position,
        token_id=self.token_id,
        fraction=self.fraction,
    )
