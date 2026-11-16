"""Self-conditioning baselines from the project proposal."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal

from exp_code.common import probability


@dataclasses.dataclass(frozen=True)
class SelfConditioningMode:
  """Transforms shaped logits before DiffusionGemma encodes them as SC."""

  kind: Literal['full', 'disabled', 'top_k', 'threshold', 'scaled'] = 'full'
  top_k: int = 8
  min_probability: float = 0.03
  beta: float = 1.0

  def __post_init__(self):
    if not math.isfinite(self.beta) or self.beta < 0:
      raise ValueError('SC beta must be finite and nonnegative')
    if self.kind != 'scaled' and self.beta != 1.0:
      raise ValueError('A nondefault beta requires kind="scaled"')

  def transform_logits(self, logits: Any) -> Any | None:
    """Returns SC logits, or None when self-conditioning is disabled."""
    if self.kind in ('full', 'scaled'):
      return logits
    if self.kind == 'disabled':
      return None
    if self.kind == 'top_k':
      return probability.top_k_logits(logits, self.top_k)
    if self.kind == 'threshold':
      return probability.threshold_logits(logits, self.min_probability)
    raise ValueError(f'Unknown self-conditioning mode: {self.kind}')
