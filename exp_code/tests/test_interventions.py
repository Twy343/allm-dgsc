import unittest

import numpy as np

from exp_code.common.probability import softmax
from exp_code.common.probability import threshold_logits
from exp_code.common.probability import top_k_logits
from exp_code.dynamic_revision.metrics import evaluate_revision
from exp_code.positional_intervention import intervention as positional
from exp_code.positional_intervention.metrics import probability_center_of_mass
from exp_code.semantic_intervention.intervention import swap_token_probabilities
from exp_code.semantic_intervention.metrics import token_log_probability_change


class InterventionTest(unittest.TestCase):

  def test_semantic_full_swap_keeps_all_other_logits_and_extreme_tail(self):
    logits=np.array([[[8.,2.,-1000.,-2000.],[1.,4.,-3.,-7.]]],dtype=np.float32)
    original=logits.copy()
    result=swap_token_probabilities(logits,batch_index=0,position=0,source_token_id=0,target_token_id=1)
    np.testing.assert_array_equal(logits,original)
    np.testing.assert_array_equal(result[0,0],original[0,0,[1,0,2,3]])
    np.testing.assert_array_equal(result[0,1],original[0,1])
    self.assertEqual(result.dtype,original.dtype)

  def test_zero_fraction_is_a_true_identity(self):
    logits=np.array([[[1.,2.,-1000.]]],dtype=np.float32)
    self.assertIs(swap_token_probabilities(logits,batch_index=0,position=0,
        source_token_id=0,target_token_id=1,fraction=0),logits)

  def test_semantic_swap_preserves_probability_row(self):
    logits = np.array([[[2.0, 1.0, 0.0, -1.0]]])
    before = softmax(logits)[0, 0]
    result = swap_token_probabilities(
        logits,
        batch_index=0,
        position=0,
        source_token_id=0,
        target_token_id=2,
    )
    after = softmax(result)[0, 0]
    self.assertAlmostEqual(float(after.sum()), 1.0)
    self.assertAlmostEqual(float(after[0]), float(before[2]))
    self.assertAlmostEqual(float(after[2]), float(before[0]))
    self.assertAlmostEqual(float(after[1]), float(before[1]))

  def test_positional_shift_preserves_rows_and_moves_mass(self):
    logits = np.array([[[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    before = softmax(logits)[0]
    result = positional.shift_token_probability(
        logits,
        batch_index=0,
        source_position=0,
        target_position=1,
        token_id=0,
        fraction=0.5,
    )
    after = softmax(result)[0]
    np.testing.assert_allclose(after.sum(axis=-1), np.ones(2))
    self.assertLess(after[0, 0], before[0, 0])
    self.assertGreater(after[1, 0], before[1, 0])
    self.assertAlmostEqual(
        float(before[0, 0] - after[0, 0]),
        float(after[1, 0] - before[1, 0]),
    )

  def test_sparse_modes_preserve_retained_probability_and_tail_mass(self):
    logits = np.array([[[4.0, 3.0, 1.0, 0.0]]])
    original = softmax(logits)
    top_k = top_k_logits(logits, 2)
    thresholded = threshold_logits(logits, 0.2)
    for result in (top_k, thresholded):
      probs = softmax(result)
      np.testing.assert_allclose(probs[..., :2], original[..., :2])
      np.testing.assert_allclose(probs[..., 2:], original[..., 2:].sum() / 2)
      np.testing.assert_allclose(probs.sum(axis=-1), 1)

  def test_revision_metrics(self):
    metrics = evaluate_revision(
        original_tokens=np.array([1, 2, 3, 4]),
        revised_tokens=np.array([1, 8, 7, 4]),
        expected_tokens=np.array([1, 8, 3, 4]),
    )
    self.assertEqual(metrics.required_edit_recall, 1.0)
    self.assertAlmostEqual(metrics.unnecessary_edit_rate, 1 / 3)
    self.assertAlmostEqual(metrics.unaffected_preservation, 2 / 3)
    self.assertEqual(metrics.final_token_accuracy, 0.75)

  def test_probability_center_of_mass(self):
    probabilities = np.zeros((1, 4, 2))
    probabilities[0, :, 0] = [0.0, 0.25, 0.5, 0.25]
    probabilities[0, :, 1] = 1.0 - probabilities[0, :, 0]
    result = probability_center_of_mass(probabilities, token_id=0)
    self.assertAlmostEqual(float(result[0]), 2.0)

  def test_token_log_probability_change(self):
    baseline = np.array([[[2.0, 0.0], [0.0, 2.0]]])
    intervention = np.array([[[3.0, 0.0], [0.0, 1.0]]])
    changes = token_log_probability_change(
        baseline_logits=baseline,
        intervention_logits=intervention,
        token_ids=np.array([[0, 1]]),
    )
    self.assertGreater(changes[0, 0], 0.0)
    self.assertLess(changes[0, 1], 0.0)


if __name__ == '__main__':
  unittest.main()
