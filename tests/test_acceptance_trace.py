"""Real JAX kernel parity and batch-freezing checks for trace instrumentation."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from gemma.diffusion import _sampler, _early_stopping
from exp_code.common.acceptance_trace import TracedPredictions, AcceptanceRecorder


class AcceptanceTests(unittest.TestCase):
  def test_sc_swap_changes_only_next_sc_at_requested_step_and_canvas(self):
    from unittest import mock
    from exp_code.common.sc_sampler import SCDiffusionSampler
    from exp_code.common.trace_sc_sampler import ObservedDiffusionSampler
    logits=jnp.array([[[4.,2.,0.,-1.],[1.,2.,3.,4.]]])
    class Encoder:
      def apply(self,variables,values,method):return jax.nn.softmax(values)
    original=_sampler.SampleStepOutput(sampled_tokens=jnp.array([[0,3]]),
        logits=logits,sc_embeddings=jax.nn.softmax(logits),
        modified_tokens_mask=jnp.ones((1,2),dtype=bool))
    sampler=ObservedDiffusionSampler(model=Encoder(),end_tokens=(),forbidden_tokens=(),
        sampling=None,cache_length=10,special_tokens=None,canvas_length=2,
        max_denoising_steps=4,text_vocab_size=4,swap=(1,0,5,0,0,1,True))
    with mock.patch.object(SCDiffusionSampler,'sample_step',return_value=original):
      for noise,offset,expected_swap in ((.75,5,True),(1.,5,False),(.75,7,False)):
        out=jax.jit(lambda n,p:sampler.sample_step(current_noise_proportion=n,
            positions=p,params={}))(jnp.array([noise]),jnp.array([[offset,offset+1]]))
        np.testing.assert_array_equal(out.sampled_tokens,original.sampled_tokens)
        np.testing.assert_array_equal(out.logits,original.logits)
        expected=logits.at[0,0,0].set(2.).at[0,0,1].set(4.) if expected_swap else logits
        np.testing.assert_allclose(out.sc_embeddings,jax.nn.softmax(expected),atol=1e-7)

  def test_watched_candidates_use_actual_logits(self):
    recorder=AcceptanceRecorder(1,())
    logits=jnp.array([[[4.,2.,-1.,-3.],[0.,1.,3.,4.]]])
    sampler=TracedPredictions(text_vocab_size=4,top_k=2,watched_token_ids=(1,3),
        callback=recorder.record,early_stop_fn=_early_stopping.TokenStabilityEarlyStop())
    jax.block_until_ready(jax.jit(sampler)(rng=jax.random.PRNGKey(2),
        denoiser_logits=logits,canvas=jnp.zeros((1,2),dtype=jnp.int32),
        current_noise_proportion=jnp.ones(1),target_noise_proportion=jnp.zeros(1)))
    jax.effects_barrier()
    np.testing.assert_array_equal(recorder.events[0]['watched_logits'],np.array(logits)[...,[1,3]])

  def test_kernel_identity_and_actual_mask(self):
    for final in (False,True):
      for seed in (0,17):
        logits=jax.random.normal(jax.random.PRNGKey(seed),(2,12,16))*5
        canvas=jnp.zeros((2,12),dtype=jnp.int32)
        kwargs=dict(rng=jax.random.PRNGKey(seed+1),denoiser_logits=logits,
            canvas=canvas,current_noise_proportion=jnp.ones(2),
            target_noise_proportion=jnp.full(2,0. if final else .5))
        records=[]
        upstream=_sampler.SampleFromPredictions(text_vocab_size=16)
        traced=TracedPredictions(text_vocab_size=16,callback=lambda *xs: records.append(xs),
            early_stop_fn=_early_stopping.TokenStabilityEarlyStop())
        actual=jax.jit(traced)(**kwargs)
        np.testing.assert_array_equal(actual,jax.jit(upstream)(**kwargs))
        jax.effects_barrier()
        event=records[0]; accepted=np.array(event[5]);sampled=np.array(event[4])
        np.testing.assert_array_equal(np.array(actual)[accepted],sampled[accepted])
        if final:self.assertTrue(accepted.all())
        else:
          p=np.array(jax.nn.softmax(logits));lp=np.array(jax.nn.log_softmax(logits))
          ent=-(p*lp).sum(-1);order=np.argsort(ent,axis=-1)
          sorted_e=np.take_along_axis(ent,order,-1)
          mask=np.cumsum(sorted_e,-1)-sorted_e<=.1
          expected=np.zeros_like(accepted)
          np.put_along_axis(expected,order,mask,-1)
          np.testing.assert_array_equal(accepted,expected)

  def test_freezing_eos_reopening_and_csv(self):
    recorder=AcceptanceRecorder(2,end_tokens=(9,))
    before=np.array([[1,2],[3,4]])
    # Accepted can be unchanged; renoised can by chance also be unchanged.
    def record(step,accepted,after,stop):
      recorder.record(step,np.ones(2),np.ones(2)*.5,before,after,
          np.array(accepted),after,np.ones((2,2)),np.ones((2,2,2),int),
          np.ones((2,2,2)),np.ones((2,2,2)),np.array(stop))
    record(0,[[True,False],[True,False]],np.array([[1,9],[3,4]]),[True,False])
    record(1,[[False,True],[False,True]],before,[False,True])
    record(0,[[True,True],[True,True]],before,[True,True])
    self.assertEqual([e['active'].tolist() for e in recorder.events],
                     [[True,True],[False,True],[False,True]])
    class Tokenizer:
      def decode(self,ids):return str(ids[0])
    with tempfile.TemporaryDirectory() as folder:
      first=recorder.export(folder,0,'mbpp:1',0,Tokenizer())
      second=recorder.export(folder,1,'mbpp:2',0,Tokenizer())
      self.assertEqual(first['actual_total_steps'],1)
      self.assertEqual(second['actual_total_steps'],3)
      self.assertEqual(second['steps_per_canvas'],'[2, 1]')
      self.assertTrue((Path(folder)/'mbpp-2_tokens.csv').read_bytes().startswith(b'\xef\xbb\xbf'))


if __name__=='__main__':unittest.main()
