"""CPU scoring must distinguish a real induced error from unsupported syntax."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_filter_operator import function_tests
from summarize_filter_conditions import binding_analysis


class FilterReviewTests(unittest.TestCase):
    def test_valid_rename_and_threshold_repair_pass(self):
        for text in (
            'def dict_filter(d, n):\n    return {k: v for k, v in d.items() if v >= n}',
            'def dict_filter(d, threshold):\n    return {k: n for k, n in d.items() if n >= threshold}',
        ):
            self.assertEqual(function_tests(text)['tests_passed'], 3)

    def test_shadowed_threshold_is_a_behavioral_failure(self):
        result = function_tests('def dict_filter(d, n):\n    return {k: n for k, n in d.items() if n >= n}')
        self.assertEqual(result['function_only_status'], 'tested')
        self.assertEqual(result['tests_passed'], 0)

    def test_noise_is_not_mislabeled_as_evaluator_unsupported(self):
        result = function_tests('def dict_filter(d, n):\n    return {k: unknown for k, v in d.items() if v >= n}')
        self.assertEqual(result['tests_passed'], 0)
        self.assertTrue(result['function_only_status'].startswith('runtime_failure'))
        result = function_tests('def dict_filter(d, n):\n    return {k: v for k, v in d.items() if! >= n}')
        self.assertEqual(result['tests_passed'], 0)
        self.assertTrue(result['function_only_status'].startswith('syntax_failure'))

    def test_unrelated_calls_are_not_executed(self):
        result = function_tests('def dict_filter(d, n):\n    return print("must not execute")')
        self.assertEqual(result['tests_passed'], '')
        self.assertTrue(result['function_only_status'].startswith('unsupported'))

    def test_binding_check_allows_safe_dictionary_shadow(self):
        text = 'def dict_filter(d, n):\n    return {key: d for key, d in d.items() if d >= n}'
        self.assertTrue(binding_analysis(text)['structurally_correct'])

    def test_binding_check_rejects_collision_and_missing_reference(self):
        collision = 'def dict_filter(d, n):\n    return {key: n for key, n in d.items() if n >= n}'
        missing = 'def dict_filter(d, threshold):\n    return {key: value for key, value in d.items() if value >= n}'
        self.assertTrue(binding_analysis(collision)['threshold_shadowed'])
        self.assertFalse(binding_analysis(collision)['structurally_correct'])
        self.assertFalse(binding_analysis(missing)['structurally_correct'])


if __name__ == '__main__':
    unittest.main()
