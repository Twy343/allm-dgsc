"""Compact serializable denoising trajectory records."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class TraceConfig:
  """Controls trace size; full vocabulary tensors are opt-in."""

  top_k: int = 8
  include_full_logits: bool = False
  include_sc_embeddings: bool = False


@dataclasses.dataclass(frozen=True)
class StepTrace:
  step: int
  current_noise_proportion: float
  target_noise_proportion: float
  canvas_before: np.ndarray
  canvas_after: np.ndarray
  modified_tokens_mask: np.ndarray
  entropy: np.ndarray
  top_token_ids: np.ndarray
  top_token_probabilities: np.ndarray
  sc_intervened: bool
  logits: np.ndarray | None = None
  sc_embeddings: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class TrajectoryTrace:
  initial_canvas: np.ndarray
  final_canvas: np.ndarray
  steps: tuple[StepTrace, ...]
  stopped_early: bool = False
  max_denoising_steps: int | None = None

  def save_npz(self, path: str | Path) -> None:
    """Writes one trajectory to a compressed, non-pickle NumPy archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        'initial_canvas': self.initial_canvas,
        'stopped_early': np.asarray(self.stopped_early),
        'step_cap': np.asarray(self.max_denoising_steps if self.max_denoising_steps is not None else len(self.steps)),
        'final_canvas': self.final_canvas,
        'step_indices': np.asarray([step.step for step in self.steps]),
        'current_noise_proportions': np.asarray(
            [step.current_noise_proportion for step in self.steps]
        ),
        'target_noise_proportions': np.asarray(
            [step.target_noise_proportion for step in self.steps]
        ),
        'sc_intervened': np.asarray(
            [step.sc_intervened for step in self.steps], dtype=bool
        ),
    }
    for field in (
        'canvas_before',
        'canvas_after',
        'modified_tokens_mask',
        'entropy',
        'top_token_ids',
        'top_token_probabilities',
    ):
      payload[field] = np.stack([getattr(step, field) for step in self.steps])
    if self.steps and self.steps[0].logits is not None:
      payload['logits'] = np.stack([step.logits for step in self.steps])
    if self.steps and self.steps[0].sc_embeddings is not None:
      payload['sc_embeddings'] = np.stack(
          [step.sc_embeddings for step in self.steps]
      )
    np.savez_compressed(destination, **payload)


def summarize_step(
    *,
    step: int,
    current_noise_proportion: float,
    target_noise_proportion: float,
    canvas_before: Any,
    canvas_after: Any,
    modified_tokens_mask: Any,
    logits: Any,
    sc_embeddings: Any,
    sc_intervened: bool,
    config: TraceConfig,
) -> StepTrace:
  """Transfers only compact summaries from accelerator to host by default."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  probs = jnp.exp(log_probs)
  entropy = -jnp.sum(probs * log_probs, axis=-1)
  top_probabilities, top_ids = jax.lax.top_k(probs, config.top_k)
  return StepTrace(
      step=step,
      current_noise_proportion=current_noise_proportion,
      target_noise_proportion=target_noise_proportion,
      canvas_before=np.asarray(canvas_before),
      canvas_after=np.asarray(canvas_after),
      modified_tokens_mask=np.asarray(modified_tokens_mask),
      entropy=np.asarray(entropy),
      top_token_ids=np.asarray(top_ids),
      top_token_probabilities=np.asarray(top_probabilities),
      sc_intervened=sc_intervened,
      logits=np.asarray(logits) if config.include_full_logits else None,
      sc_embeddings=(
          np.asarray(sc_embeddings) if config.include_sc_embeddings else None
      ),
  )
