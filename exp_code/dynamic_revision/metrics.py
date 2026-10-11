"""Proposal metrics for comparing an updated-context revision."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class RevisionMetrics:
  required_edit_recall: float
  unnecessary_edit_rate: float
  unaffected_preservation: float
  final_token_accuracy: float


def _safe_ratio(numerator: int, denominator: int, empty_value: float) -> float:
  return numerator / denominator if denominator else empty_value


def evaluate_revision(
    *,
    original_tokens: np.ndarray,
    revised_tokens: np.ndarray,
    expected_tokens: np.ndarray,
) -> RevisionMetrics:
  """Scores required edits and preservation against an expected revision."""
  original = np.asarray(original_tokens)
  revised = np.asarray(revised_tokens)
  expected = np.asarray(expected_tokens)
  if original.shape != revised.shape or original.shape != expected.shape:
    raise ValueError(
        'original, revised, and expected tokens must have one shape'
    )

  required = expected != original
  unaffected = ~required
  made_required = required & (revised == expected)
  unnecessary = unaffected & (revised != original)
  correct = revised == expected
  return RevisionMetrics(
      required_edit_recall=_safe_ratio(
          int(made_required.sum()), int(required.sum()), 1.0
      ),
      unnecessary_edit_rate=_safe_ratio(
          int(unnecessary.sum()), int(unaffected.sum()), 0.0
      ),
      unaffected_preservation=_safe_ratio(
          int((unaffected & (revised == original)).sum()),
          int(unaffected.sum()),
          1.0,
      ),
      final_token_accuracy=float(correct.mean()) if correct.size else 1.0,
  )
