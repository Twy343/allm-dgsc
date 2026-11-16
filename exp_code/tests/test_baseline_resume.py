import json
from pathlib import Path
import tempfile
import unittest

from exp_code.baselines.run import run_lock, summarize


class ResumeTest(unittest.TestCase):
  def test_exclusive_lock_released_on_error(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp)
      with self.assertRaises(ValueError):
        with run_lock(output):
          with self.assertRaises(FileExistsError):
            with run_lock(output):
              pass
          raise ValueError('Interrupted')
      self.assertFalse((output/'.run.lock').exists())

  def test_summary_reports_partial_pairs_and_rejects_duplicates(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp)
      (output/'run.json').write_text(json.dumps(dict(
          sha256='test', plans={'gsm8k':{'cases':[{'id':'a'},{'id':'b'}]}})))
      rows = [dict(run_sha256='test',task='gsm8k',mode=mode,case_id='a',
                   score={'passed':passed},generation_seconds=1)
              for mode,passed in [('full',False),('top_k',True)]]
      file = output/'predictions.jsonl'
      file.write_text('\n'.join(json.dumps(row) for row in rows))
      result=summarize(output)
      self.assertFalse(result['gsm8k/top_k']['complete'])
      self.assertEqual(result['gsm8k/top_k']['paired_vs_full']['delta_accuracy'],1)
      file.write_text('\n'.join(json.dumps(row) for row in rows+[rows[0]]))
      with self.assertRaises(ValueError):
        summarize(output)


if __name__ == '__main__':
  unittest.main()
