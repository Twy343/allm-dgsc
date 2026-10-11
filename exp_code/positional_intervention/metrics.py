"""Metrics for position-sensitive token probability mass."""

from __future__ import annotations

import numpy as np


def probability_center_of_mass(
    probabilities: np.ndarray, *, token_id: int
) -> np.ndarray:
  """Computes a token's probability-weighted position per batch element."""
  values = np.asarray(probabilities)
  if values.ndim != 3:
    raise ValueError('probabilities must have shape [batch, position, vocab]')
  token_mass = values[..., token_id]
  positions = np.arange(values.shape[1], dtype=np.float64)
  total_mass = token_mass.sum(axis=-1)
  weighted = (token_mass * positions[None, :]).sum(axis=-1)
  return np.divide(
      weighted,
      total_mass,
      out=np.full_like(weighted, np.nan, dtype=np.float64),
      where=total_mass > 0,
  )
