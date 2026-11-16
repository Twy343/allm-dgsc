"""Explicit DiffusionGemma denoising loop with tracing and SC hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from exp_code.common.sc_modes import SelfConditioningMode
from exp_code.common.trace import TraceConfig
from exp_code.common.trace import TrajectoryTrace
from exp_code.common.trace import summarize_step

LogitsIntervention = Callable[[int, Any], Any]


def _encode_sc(sampler: Any, params: Any, logits: Any) -> Any:
  return sampler.model.apply(
      {'params': params},
      logits,
      method=lambda model, values: model.embedder.encode_logits(values),
  )


def run_fixed_trajectory(
    *,
    sampler: Any,
    params: Any,
    rng: Any,
    batch_size: int,
    cache: Any = None,
    full_attention_mask: Any = None,
    canvas_length: int | None = None,
    max_denoising_steps: int | None = None,
    initial_canvas: Any = None,
    initial_sc_embeddings: Any = None,
    sc_mode: SelfConditioningMode = SelfConditioningMode(),
    intervention: LogitsIntervention | None = None,
    trace_config: TraceConfig = TraceConfig(),
    stop_early: bool = False,
) -> TrajectoryTrace:
  """Runs a fixed number of steps and exposes the SC boundary.

  The visible canvas produced at step t is never changed by an intervention.
  Instead, the hook transforms the shaped logits before they are encoded into
  the SC embeddings consumed at step t + 1.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from gemma.diffusion import _sampler  # pylint: disable=g-import-not-at-top

  if canvas_length is None:
    canvas_length = sampler.canvas_length
  if max_denoising_steps is None:
    max_denoising_steps = sampler.max_denoising_steps
  if max_denoising_steps <= 0:
    raise ValueError('max_denoising_steps must be positive')
  if stop_early and batch_size != 1:
    raise ValueError('Adaptive observation currently supports batch_size=1 only')

  initial_canvas_rng, step_rng = jax.random.split(rng)
  if cache is None:
    cache_length = None
    samples_in_cache = None
    positions = jnp.broadcast_to(
        jnp.arange(canvas_length)[None, :], (batch_size, canvas_length)
    )
  else:
    cache_layer = list(cache.values())[0]
    cache_length = cache_layer['k'].shape[1]
    samples_in_cache = cache_layer['end_index']
    if full_attention_mask is None:
      unpadded_offset = samples_in_cache
    else:
      valid_prefix = (
          jnp.arange(cache_length)[None, :] < samples_in_cache[:, None]
      ) & full_attention_mask
      unpadded_offset = jnp.sum(valid_prefix.astype(jnp.int32), axis=-1)
    positions = unpadded_offset[:, None] + jnp.arange(canvas_length)[None, :]

  attention_mask = _sampler._make_global_attention_mask(
      batch_size=batch_size,
      canvas_length=canvas_length,
      cache_length=cache_length,
      num_valid_tokens=samples_in_cache,
      full_attention_mask=full_attention_mask,
  )
  sliding_attention_mask = _sampler._make_block_local_attention_mask(
      batch_size=batch_size,
      canvas_length=canvas_length,
      sliding_window_size=sampler.sliding_window_size,
      cache_length=cache_length,
      num_valid_tokens=samples_in_cache,
  )

  if initial_canvas is None:
    canvas = sampler.diffusion_process.get_initial_sample(
        rng=initial_canvas_rng,
        batch_size=batch_size,
        canvas_length=canvas_length,
        text_vocab_size=sampler.text_vocab_size,
    )
  else:
    canvas = jnp.asarray(initial_canvas)
    expected_shape = (batch_size, canvas_length)
    if canvas.shape != expected_shape:
      raise ValueError(
          f'initial_canvas must have shape {expected_shape}, got {canvas.shape}'
      )
  initial_canvas = np.asarray(canvas)
  embed_dim = sampler.model.config.embed_dim
  if initial_sc_embeddings is None:
    sc_embeddings = jnp.zeros(
        (batch_size, canvas_length, embed_dim), dtype=jnp.bfloat16
    )
  else:
    sc_embeddings = jnp.asarray(initial_sc_embeddings, dtype=jnp.bfloat16)
    expected_shape = (batch_size, canvas_length, embed_dim)
    if sc_embeddings.shape != expected_shape:
      raise ValueError(
          'initial_sc_embeddings must have shape '
          f'{expected_shape}, got {sc_embeddings.shape}'
      )
  steps = []
  stopped_early = False
  noise_proportions = 1.0 - jnp.arange(max_denoising_steps + 1) / max_denoising_steps
  step_function = jax.jit(sampler.sample_step)

  for step in range(max_denoising_steps):
    step_rng, sample_rng = jax.random.split(step_rng)
    current_noise = noise_proportions[step]
    target_noise = noise_proportions[step + 1]
    canvas_before = canvas
    output = step_function(
        canvas=canvas,
        sc_embeddings=sc_embeddings,
        cache=cache,
        positions=positions,
        attention_mask=attention_mask,
        sliding_attention_mask=sliding_attention_mask,
        current_noise_proportion=jnp.full((batch_size,), current_noise),
        target_noise_proportion=jnp.full((batch_size,), target_noise),
        params=params,
        rng=sample_rng,
    )

    sc_logits = sc_mode.transform_logits(output.logits)
    if intervention is not None and sc_logits is not None:
      logits_before_intervention = sc_logits
      sc_logits = intervention(step, sc_logits)
      sc_intervened = sc_logits is not logits_before_intervention
    else:
      sc_intervened = False
    if sc_logits is None:
      next_sc_embeddings = jnp.zeros_like(output.sc_embeddings)
    elif sc_logits is output.logits and intervention is None:
      next_sc_embeddings = output.sc_embeddings
    else:
      next_sc_embeddings = _encode_sc(sampler, params, sc_logits)

    next_sc_embeddings = next_sc_embeddings * sc_mode.beta
    steps.append(
        summarize_step(
            step=step,
            current_noise_proportion=current_noise,
            target_noise_proportion=target_noise,
            canvas_before=canvas_before,
            canvas_after=output.sampled_tokens,
            modified_tokens_mask=output.modified_tokens_mask,
            logits=output.logits,
            sc_embeddings=next_sc_embeddings,
            sc_intervened=sc_intervened,
            config=trace_config,
        )
    )
    canvas = output.sampled_tokens
    sc_embeddings = next_sc_embeddings.astype(jnp.bfloat16)
    if stop_early and bool(jnp.all(sampler.early_stop_fn.should_stop(
        step=jnp.asarray(step), canvas=canvas, previous_canvas=canvas_before,
        logits=output.logits,
    ))):
      stopped_early = True
      break

  return TrajectoryTrace(
      initial_canvas=initial_canvas,
      final_canvas=np.asarray(canvas),
      steps=tuple(steps),
      stopped_early=stopped_early,
      max_denoising_steps=max_denoising_steps,
  )
