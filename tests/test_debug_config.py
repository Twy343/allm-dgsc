import copy
import json
from pathlib import Path
import unittest

from exp_code.common.debug_config import resolve_batch


class DebugConfigTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'configs' / 'debugger_run.json'
        self.config = json.loads(path.read_text(encoding='utf-8'))

    def test_compact_plan_retains_original_mbpp_batch_identity(self):
        batch, focus, seed = resolve_batch(self.config, 'mbpp:277')
        self.assertEqual([c['id'] for c in batch],
                         ['mbpp:133', 'mbpp:464', 'mbpp:277', 'mbpp:59'])
        self.assertEqual((focus, seed), (2, 100008))
        reordered = copy.deepcopy(self.config)
        reordered['plans'] = dict(reversed(list(reordered['plans'].items())))
        self.assertEqual(resolve_batch(reordered, 'mbpp:277'), (batch, focus, seed))

    def test_gate_case_identity(self):
        _, focus, seed = resolve_batch(self.config, 'gsm8k:480')
        self.assertEqual((focus, seed), (3, 0))

    def test_legacy_plan_seed_formula_and_incomplete_batch(self):
        config = {'seed': 7, 'plans': {'gsm8k': {'cases': [
            {'id': f'gsm8k:{i}'} for i in range(8)]}}}
        _, focus, seed = resolve_batch(config, 'gsm8k:6')
        self.assertEqual((focus, seed), (2, 8))
        config['plans']['gsm8k']['cases'].pop()
        with self.assertRaisesRegex(ValueError, 'complete original'):
            resolve_batch(config, 'gsm8k:6')
