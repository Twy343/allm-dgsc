"""Self-conditioning baselines from the project proposal."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

from exp_code.common import probability


@dataclasses.dataclass(frozen=True)
class SelfConditioningMode:
  """Transforms shaped logits before DiffusionGemma encodes them as SC."""

  kind: Literal['full', 'disabled', 'top_k', 'threshold'] = 'full'
  top_k: int = 8
  min_probability: float = 0.03

  def transform_logits(self, logits: Any) -> Any | None:
    """Returns SC logits, or None when self-conditioning is disabled."""
    if self.kind == 'full':
      return logits
    if self.kind == 'disabled':
      return None
    if self.kind == 'top_k':
      return probability.top_k_logits(logits, self.top_k)
    if self.kind == 'threshold':
      return probability.threshold_logits(logits, self.min_probability)
    raise ValueError(f'Unknown self-conditioning mode: {self.kind}')
