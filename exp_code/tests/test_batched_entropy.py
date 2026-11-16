import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from exp_code.baselines.data import prepare
from exp_code.baselines.batched_run import CONDITIONS,batches
from exp_code.baselines.entropy_recorder import EntropyRecorder


class BatchedEntropyTest(unittest.TestCase):
  def test_each_test_set_selects_256_deterministically(self):
    class Source:
      _fingerprint='fake-test-fingerprint'
      def __len__(self): return 257
      def __getitem__(self,i):
        return dict(question=str(i),answer='#### 1',prompt=str(i),
                    task_id=i,code='pass',test_list=['assert True'])
    loader=SimpleNamespace(load_from_disk=lambda _: {'test':Source()})
    with patch.dict('sys.modules',{'datasets':loader}):
      for task in ('gsm8k','mbpp'):
        a=prepare(task,'fake','test256')
        b=prepare(task,'fake','test256')
        self.assertEqual(a,b)
        self.assertEqual(a['split'],'test')
        self.assertEqual(len(a['cases']),256)
        self.assertEqual(len({c['id'] for c in a['cases']}),256)

  def test_condition_order_and_batch_coverage(self):
    self.assertEqual(list(CONDITIONS),['full','disabled_48','disabled_96','top_k','threshold'])
    self.assertEqual(CONDITIONS['disabled_96']['steps'],96)
    chunks=list(batches(list(range(256)),4))
    self.assertEqual(len(chunks),64)
    self.assertEqual([i for _,items in chunks for i in items],list(range(256)))

  def test_freezing_and_finished_canvas_are_not_counted_twice(self):
    recorder=EntropyRecorder(4,canvas_length=3)
    recorder.record(0,np.array([[1,2,3],[4,5,6]]),np.array([True,False]))
    recorder.record(1,np.ones((2,3))*9,np.array([False,False]))
    recorder.record(2,np.ones((2,3))*8,np.array([False,True]))
    recorder.record(0,np.ones((2,3)),np.array([True,True]))
    a=recorder.sample(0,np.array([2,99,0,0,0,0]),(99,))
    self.assertEqual(a['steps'].tolist(),[1])
    self.assertTrue(np.isnan(a['entropy'][0,1:]).all())
    b=recorder.sample(1,np.array([2,2,2,2,99,0]),(99,))
    self.assertEqual(b['steps'].tolist(),[3,1])
    self.assertEqual(b['cap_reached'].tolist(),[False,False])


if __name__=='__main__':
  unittest.main()
