"""Check the paper's defining invariants, including numerical edge cases."""
import unittest
import numpy as np

from exp_code.common.probability import softmax, top_k_logits, threshold_logits
from exp_code.common.sc_modes import SelfConditioningMode


class BottleneckTest(unittest.TestCase):
  def test_matches_appendix_c_literal_algorithm(self):
    logits = np.random.default_rng(7).normal(size=(2, 3, 17))
    expected = logits.copy()
    for index in np.ndindex(logits.shape[:-1]):
      row = logits[index]
      keep = softmax(row) >= 0.08
      if (~keep).any():
        expected[index][~keep] = np.log(np.exp(row[~keep]).mean())
    np.testing.assert_allclose(threshold_logits(logits, 0.08), expected)

  def test_no_candidates_is_uniform_not_forced_argmax(self):
    result = threshold_logits(np.array([[0., 1., 2.]]), 0.99)
    np.testing.assert_allclose(softmax(result), np.full((1, 3), 1/3))

  def test_all_candidates_is_identity(self):
    logits = np.array([[0., 1., 2.]])
    np.testing.assert_array_equal(top_k_logits(logits, 3), logits)
    np.testing.assert_array_equal(threshold_logits(logits, 0), logits)

  def test_shift_invariance_and_confident_tail(self):
    logits = np.array([[0., -80., -81., -82.]], dtype=np.float32)
    result = top_k_logits(logits, 1)
    self.assertTrue(np.isfinite(result).all())
    np.testing.assert_allclose(softmax(result)[..., 1:].sum(), softmax(logits)[..., 1:].sum(), rtol=1e-5, atol=0)
    np.testing.assert_allclose(softmax(result), softmax(top_k_logits(logits + 1000, 1)), rtol=1e-4)

  def test_tied_top_k_retains_k_entries(self):
    logits = np.array([[3., 2., 2., 0.]])
    result = top_k_logits(logits, 2)
    self.assertEqual(int((result == logits).sum()), 2)
    np.testing.assert_allclose(softmax(result).sum(), 1)

  def test_full_and_disabled(self):
    logits = np.zeros((1, 2, 4))
    self.assertIs(SelfConditioningMode().transform_logits(logits), logits)
    self.assertIsNone(SelfConditioningMode(kind='disabled').transform_logits(logits))

  def test_beta_is_explicit_and_not_a_temperature(self):
    logits=np.ones((1,2,4))
    self.assertIs(SelfConditioningMode(kind='scaled',beta=0.2).transform_logits(logits),logits)
    for beta in (-1,float('nan'),float('inf')):
      with self.assertRaises(ValueError):
        SelfConditioningMode(kind='scaled',beta=beta)

  def test_invalid_settings(self):
    for k in (0, 5):
      with self.assertRaises(ValueError):
        top_k_logits(np.zeros((1, 4)), k)
    with self.assertRaises(ValueError):
      threshold_logits(np.zeros((1, 4)), 1.1)


if __name__ == '__main__':
  unittest.main()
