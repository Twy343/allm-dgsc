"""Probability operations shared by the three experiments.

The functions accept NumPy arrays and JAX arrays. JAX is imported lazily so the
small unit-test suite can run without installing the full DiffusionGemma stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def is_jax_array(value: Any) -> bool:
  module = type(value).__module__
  return module.startswith('jax') or module.startswith('jaxlib')


def array_module(value: Any):
  if is_jax_array(value):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

    return jnp
  return np


def replace_row(array: Any, batch_index: int, position: int, row: Any) -> Any:
  """Returns `array` with one [vocab] row replaced."""
  if is_jax_array(array):
    return array.at[batch_index, position, :].set(row)
  result = np.array(array, copy=True)
  result[batch_index, position, :] = row
  return result


def log_softmax(logits: Any, axis: int = -1) -> Any:
  xp = array_module(logits)
  shifted = logits - xp.max(logits, axis=axis, keepdims=True)
  return shifted - xp.log(xp.sum(xp.exp(shifted), axis=axis, keepdims=True))


def softmax(logits: Any, axis: int = -1) -> Any:
  xp = array_module(logits)
  return xp.exp(log_softmax(logits, axis=axis))


def entropy_from_logits(logits: Any) -> Any:
  """Computes categorical entropy along the vocabulary axis."""
  xp = array_module(logits)
  log_probs = log_softmax(logits)
  probs = xp.exp(log_probs)
  return -xp.sum(probs * log_probs, axis=-1)


def top_k_logits(logits: Any, k: int) -> Any:
  """Keeps the k largest logits per row and masks all other entries."""
  if k <= 0 or k > logits.shape[-1]:
    raise ValueError(f'k must be in [1, {logits.shape[-1]}], got {k}')
  xp = array_module(logits)
  indices = xp.argsort(logits, axis=-1)[..., -k:]
  mask = xp.zeros(logits.shape, dtype=bool)
  if is_jax_array(logits):
    import jax  # pylint: disable=g-import-not-at-top

    mask = jax.nn.one_hot(
        indices, logits.shape[-1], dtype=bool
    ).any(axis=-2)
  else:
    np.put_along_axis(mask, indices, True, axis=-1)
  return xp.where(mask, logits, -xp.inf)


def threshold_logits(logits: Any, min_probability: float) -> Any:
  """Masks candidates below a probability threshold and keeps each argmax."""
  if not 0.0 <= min_probability < 1.0:
    raise ValueError('min_probability must be in [0, 1)')
  xp = array_module(logits)
  probs = softmax(logits)
  keep = probs >= min_probability
  argmax = xp.argmax(logits, axis=-1, keepdims=True)
  if is_jax_array(logits):
    import jax  # pylint: disable=g-import-not-at-top

    keep = keep | jax.nn.one_hot(
        argmax[..., 0], logits.shape[-1], dtype=bool
    )
  else:
    np.put_along_axis(keep, argmax, True, axis=-1)
  return xp.where(keep, logits, -xp.inf)


def probabilities_to_logits(probabilities: Any) -> Any:
  """Converts normalized probabilities to stable equivalent logits."""
  xp = array_module(probabilities)
  tiny = xp.finfo(probabilities.dtype).tiny
  return xp.log(xp.clip(probabilities, tiny, 1.0))
