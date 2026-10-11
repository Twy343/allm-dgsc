"""Reopen a completed canvas under recomputed context state."""

from __future__ import annotations

from typing import Any

import numpy as np

from exp_code.common.dg_runner import run_fixed_trajectory
from exp_code.common.sc_modes import SelfConditioningMode
from exp_code.common.trace import TraceConfig
from exp_code.common.trace import TrajectoryTrace


def run_revision_trajectory(
    *,
    sampler: Any,
    params: Any,
    rng: Any,
    completed_canvas: Any,
    updated_context_cache: Any,
    full_attention_mask: Any = None,
    max_denoising_steps: int | None = None,
    initial_sc_embeddings: Any = None,
    sc_mode: SelfConditioningMode = SelfConditioningMode(),
    trace_config: TraceConfig = TraceConfig(),
) -> TrajectoryTrace:
  """Denoises a completed canvas against an updated prompt/KV cache.

  By default SC is reset to zero, making the completed visible canvas the only
  state carried from x_0. Pass the final x_0 SC embeddings explicitly to study
  state reuse as a separate condition.
  """
  canvas = np.asarray(completed_canvas)
  if canvas.ndim != 2:
    raise ValueError('completed_canvas must have shape [batch, position]')
  return run_fixed_trajectory(
      sampler=sampler,
      params=params,
      rng=rng,
      batch_size=canvas.shape[0],
      cache=updated_context_cache,
      full_attention_mask=full_attention_mask,
      canvas_length=canvas.shape[1],
      max_denoising_steps=max_denoising_steps,
      initial_canvas=canvas,
      initial_sc_embeddings=initial_sc_embeddings,
      sc_mode=sc_mode,
      trace_config=trace_config,
  )
