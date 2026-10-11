"""Metrics for semantic counterfactual trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from exp_code.common.probability import log_softmax


def token_log_probability_change(
    *,
    baseline_logits: Any,
    intervention_logits: Any,
    token_ids: np.ndarray,
) -> np.ndarray:
  """Returns intervention minus baseline log-probability for selected tokens.

  `token_ids` must match the logits shape without its final vocabulary axis.
  This supports scoring one expected downstream token at each batch/position.
  """
  baseline = np.asarray(log_softmax(np.asarray(baseline_logits)))
  intervention = np.asarray(log_softmax(np.asarray(intervention_logits)))
  selected = np.asarray(token_ids)
  if baseline.shape != intervention.shape:
    raise ValueError('baseline and intervention logits must have one shape')
  if selected.shape != baseline.shape[:-1]:
    raise ValueError(
        f'token_ids must have shape {baseline.shape[:-1]}, got {selected.shape}'
    )
  indices = selected[..., None]
  baseline_selected = np.take_along_axis(baseline, indices, axis=-1)[..., 0]
  intervention_selected = np.take_along_axis(
      intervention, indices, axis=-1
  )[..., 0]
  return intervention_selected - baseline_selected
