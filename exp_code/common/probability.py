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
  return -xp.sum(probs * xp.where(probs == 0, 0.0, log_probs), axis=-1)


def _flatten_discarded_logits(logits: Any, keep: Any) -> Any:
  """Appendix C: preserve retained probabilities; flatten the discarded tail.

  Compute the tail mass directly (not 1 - retained_mass), avoiding cancellation
  for confident predictions. Float32 is required for bfloat16 model logits.
  An empty retained set becomes uniform; an empty discarded set is unchanged.
  """
  xp = array_module(logits)
  values = xp.asarray(logits, dtype=xp.result_type(logits.dtype, xp.float32))
  maximum = xp.max(values, axis=-1, keepdims=True)
  tail_mass = xp.sum(
      xp.where(keep, 0.0, xp.exp(values - maximum)), axis=-1, keepdims=True
  )
  tail_count = xp.sum(~keep, axis=-1, keepdims=True)
  # log(0) is intentional when all discarded logits were already -inf.
  with np.errstate(divide='ignore'):
    constant = maximum + xp.log(tail_mass / xp.maximum(tail_count, 1))
  # Keep the upstream encoder's precision policy unchanged across conditions.
  return xp.where(keep, values, constant).astype(logits.dtype)


def top_k_logits(logits: Any, k: int) -> Any:
  """Keeps exactly k candidates and uniformly redistributes discarded mass."""
  if k <= 0 or k > logits.shape[-1]:
    raise ValueError(f'k must be in [1, {logits.shape[-1]}], got {k}')
  xp = array_module(logits)
  if is_jax_array(logits):
    import jax  # pylint: disable=g-import-not-at-top

    _, indices = jax.lax.top_k(logits, k)
    flat = indices.reshape((-1, k))
    mask = xp.zeros((flat.shape[0], logits.shape[-1]), dtype=bool)
    mask = mask.at[xp.arange(flat.shape[0])[:, None], flat].set(True)
    mask = mask.reshape(logits.shape)
  else:
    indices = xp.argsort(logits, axis=-1, kind='stable')[..., -k:]
    mask = xp.zeros(logits.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=-1)
  return _flatten_discarded_logits(logits, mask)


def threshold_logits(logits: Any, min_probability: float) -> Any:
  """Algorithm 1: retain p >= cutoff; no forced argmax or renormalization."""
  if not 0.0 <= min_probability <= 1.0:
    raise ValueError('min_probability must be in [0, 1]')
  xp = array_module(logits)
  values = xp.asarray(logits, dtype=xp.result_type(logits.dtype, xp.float32))
  probs = softmax(values)
  keep = probs >= min_probability
  return _flatten_discarded_logits(logits, keep)


def probabilities_to_logits(probabilities: Any) -> Any:
  """Converts normalized probabilities to stable equivalent logits."""
  xp = array_module(probabilities)
  tiny = xp.finfo(probabilities.dtype).tiny
  return xp.log(xp.clip(probabilities, tiny, 1.0))
