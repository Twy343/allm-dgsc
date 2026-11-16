import unittest
from exp_code.baselines.data import make_case
from exp_code.baselines.scoring import score_gsm8k, score_mbpp, extract_code


class ScoringTest(unittest.TestCase):
  def test_numeric_equivalence(self):
    self.assertTrue(score_gsm8k('Work\n#### 1,200.00', 'Work\n#### 1200')['passed'])
    self.assertTrue(score_gsm8k(r'Answer: \boxed{-3.5}', '#### -3.50')['passed'])

  def test_never_grade_an_intermediate_number(self):
    self.assertFalse(score_gsm8k('The input is 12.\nI cannot solve it.', '#### 12')['passed'])
    self.assertFalse(score_gsm8k('#### 9', '#### 12')['passed'])

  def test_reference_code_and_incorrect_code(self):
    tests = ['assert twice(3) == 6']
    self.assertTrue(score_mbpp('```python\ndef twice(x): return 2*x\n```', tests)['passed'])
    self.assertFalse(score_mbpp('def twice(x): return x', tests)['passed'])

  def test_code_timeout(self):
    result = score_mbpp('while True: pass', ['assert True'], timeout=0.15)
    self.assertEqual(result['status'], 'timeout')

  def test_select_final_code(self):
    text = '<think>```python\nbad()\n```</think>\n```python\ndef f(): return 2\n```'
    self.assertEqual(extract_code(text), 'def f(): return 2')

  def test_mbpp_prompt_excludes_reference(self):
    case = make_case('mbpp', dict(task_id=5, prompt='Add two numbers.',
                      code='SECRET_REFERENCE', test_list=['assert f(1,2)==3']), 0)
    self.assertNotIn('SECRET_REFERENCE', case['prompt'])
    self.assertEqual(case['id'], 'mbpp:5')


if __name__ == '__main__':
  unittest.main()
