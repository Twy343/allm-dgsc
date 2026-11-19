import unittest
import jax
import jax.numpy as jnp
import numpy as np
from gemma.diffusion import _models,_sampler,_early_stopping
from gemma.gm.nn.gemma4 import _config,_modules
from exp_code.common.canvas_debugger import CanvasDebugger


class DebuggerTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cfg=_config.TransformerConfig(num_embed=32,embed_dim=8,num_heads=2,num_kv_heads=1,
        head_dim=4,hidden_dim=16,attention_types=[_modules.AttentionType.GLOBAL],
        kv_cache_sharing_config=None,use_post_attn_norm=True,use_post_ffw_norm=True,
        final_logit_softcap=None,global_rope_proportion=1.)
    model=_models.DiffusionGemma_26B_A4B(config=cfg)
    variables=model.init(jax.random.PRNGKey(7),tokens=jnp.ones((1,3),dtype=jnp.int32),
        sc_embeddings=jnp.zeros((1,3,8)),method=model.call_with_self_conditioning)
    cls.params=variables['params']
    cls.loop=_sampler.DiffusionSampler(model=model,end_tokens=(),forbidden_tokens=(),
        sampling=None,cache_length=8,special_tokens=None,canvas_length=3,
        max_denoising_steps=4,text_vocab_size=32)

  def engine(self,sampler=None):
    return CanvasDebugger(sampler=sampler or self.loop,params=self.params,
        rng=jax.random.PRNGKey(11),batch_size=1)

  def test_native_canvas_and_restored_continuation_match(self):
    engine=self.engine();engine.until(2);engine.save('break')
    first=np.array(engine.run().canvas)
    records=[e['after'].copy() for e in engine.recorder.events]
    engine.restore('break');again=np.array(engine.run().canvas)
    np.testing.assert_array_equal(first,again)
    for actual,expected in zip(engine.recorder.events,records):
      np.testing.assert_array_equal(actual['after'],expected)
    native=self.loop.sample_next_canvas(canvas_length=3,max_denoising_steps=4,
        batch_size=1,cache=None,params=self.params,rng=jax.random.PRNGKey(11))
    np.testing.assert_array_equal(first,native)

  def test_swap_keeps_canvas_rng_and_restore_repeats_branch(self):
    engine=self.engine();engine.until(2);engine.save('break')
    original=engine.state
    result=engine.swap(position=1,source=0,target=1)
    np.testing.assert_array_equal(engine.state.canvas,original.canvas)
    np.testing.assert_array_equal(engine.state.rng,original.rng)
    np.testing.assert_array_equal(engine.state.done,original.done)
    np.testing.assert_array_equal(engine.state.sc_logits[0,1,:2],original.sc_logits[0,1,[1,0]])
    self.assertEqual(result['first_affected_step_number'],3)
    expected=np.array(engine.run().canvas)
    engine.restore('break');engine.swap(position=1,source=0,target=1)
    np.testing.assert_array_equal(engine.run().canvas,expected)
    engine.restore('break');np.testing.assert_array_equal(engine.state.sc_logits,original.sc_logits)
    self.assertEqual(engine.edits,[])

  def test_stopped_and_initial_sc_edits_are_rejected(self):
    engine=self.engine()
    with self.assertRaises(ValueError):engine.swap(position=0,source=0,target=1)
    engine.run()
    with self.assertRaises(ValueError):engine.swap(position=0,source=0,target=1)

  def test_custom_distribution_canvas_and_zero_controls_are_restorable(self):
    engine=self.engine();engine.until(2);engine.save('base');original=engine.state
    engine.mix(position=1,weights={0:1,1:3},alpha=0)
    self.assertIs(engine.state,original)
    engine.mix(position=1,weights={0:1,1:3},alpha=1)
    probs=np.asarray(jax.nn.softmax(engine.state.sc_logits[0,1].astype(jnp.float32)))
    np.testing.assert_allclose(probs[:2],[.25,.75],atol=.003)
    self.assertAlmostEqual(float(probs.sum()),1.,places=6)
    np.testing.assert_array_equal(engine.state.sc_logits[0,0],original.sc_logits[0,0])
    np.testing.assert_array_equal(engine.state.sc_embeddings[0,0],original.sc_embeddings[0,0])
    np.testing.assert_array_equal(engine.state.canvas,original.canvas)
    np.testing.assert_array_equal(engine.state.rng,original.rng)
    engine.restore('base');engine.change_canvas(position=1,token=7)
    np.testing.assert_array_equal(engine.state.sc_embeddings,original.sc_embeddings)
    np.testing.assert_array_equal(engine.state.sc_logits,original.sc_logits)
    self.assertEqual(int(engine.state.canvas[0,1]),7)
    engine.zero_sc(position=0);engine.mix(position=1,weights={0:1},alpha=.5)
    np.testing.assert_array_equal(engine.state.sc_embeddings[0,0],np.zeros(8))
    branch=np.array(engine.run().canvas)
    engine.restore('base');engine.change_canvas(position=1,token=7)
    engine.zero_sc(position=0);engine.mix(position=1,weights={0:1},alpha=.5)
    np.testing.assert_array_equal(engine.run().canvas,branch)
    engine.restore('base');np.testing.assert_array_equal(engine.state.canvas,original.canvas)
    np.testing.assert_array_equal(engine.state.sc_embeddings,original.sc_embeddings)

  def test_bounded_exploration_restores_the_same_committed_state(self):
    import tempfile
    from pathlib import Path
    from exp_code.semantic_intervention.iterative_probe import explore_committed
    engine=self.engine();engine.until(1);original=engine.state
    pos=int(np.flatnonzero(engine.recorder.events[-1]['accepted'][0])[0])
    source=int(original.canvas[0,pos]);target=(source+1)%32
    class Tokenizer:
      def decode(self,ids):return str(ids[0])
    with tempfile.TemporaryDirectory() as folder:
      report=explore_committed(engine,focus=0,position=pos,source=source,target=target,
          case_id='tiny:1',seed=11,tokenizer=Tokenizer(),output=Path(folder)/'probe')
      self.assertEqual(report['status'],'exploration_complete')
      self.assertTrue((Path(folder)/'probe/observations.csv').exists())
      np.testing.assert_array_equal(engine.state.canvas,original.canvas)
      np.testing.assert_array_equal(engine.state.rng,original.rng)


  def test_region_edits_are_atomic_and_preserve_canvas_and_other_rows(self):
    engine=self.engine();engine.until(1);engine.save('base');original=engine.state
    engine.region(start=0,tokens=[1,2],alpha=1.)
    np.testing.assert_array_equal(engine.state.canvas,original.canvas)
    np.testing.assert_array_equal(engine.state.rng,original.rng)
    np.testing.assert_array_equal(engine.state.sc_embeddings[0,2],original.sc_embeddings[0,2])
    for position,target in enumerate([1,2]):
      p=np.array(jax.nn.softmax(engine.state.sc_logits[0,position].astype(jnp.float32)))
      self.assertEqual(float(p[target]),1.)
    branch=np.array(engine.run().canvas)
    engine.restore('base');engine.region(start=0,tokens=[1,2],alpha=1.)
    np.testing.assert_array_equal(engine.run().canvas,branch)
    engine.restore('base')
    with self.assertRaises(ValueError):engine.region(start=0,tokens=[1,99])
    self.assertIs(engine.state,original);self.assertEqual(engine.edits,[])

  def test_region_success_requires_a_whole_accepted_span_and_tracks_movement(self):
    from exp_code.semantic_intervention.region_probe import accepted_occurrences
    tokens=np.array([0,8,2,4,0]);mask=np.array([False,True,True,False,False])
    self.assertEqual(accepted_occurrences(tokens,mask,[8,2,4]),[])
    mask[3]=True
    self.assertEqual(accepted_occurrences(tokens,mask,[8,2,4]),[1])


  def test_real_tokenizer_region_templates_align_to_saved_reference(self):
    import json
    from pathlib import Path
    model=Path('/insomnia001/depts/edu/users/yc4953/models/diffusiongemma-26B-A4B-it/tokenizer.model')
    data=Path('outputs/full-sc-steptrace-v1/samples.json')
    if not model.exists() or not data.exists():self.skipTest('Real tokenizer/reference fixture not present')
    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=str(model))
    source=sp.EncodeAsIds('4 \\div 2 = 2');target=sp.EncodeAsIds('8 \\div 2 = 4')
    row=next(r for r in json.loads(data.read_text()) if r['case_id']=='gsm8k:480')
    ids=json.loads(row['output_tokens'])[:256]
    matches=[i for i in range(len(ids)-len(source)+1) if ids[i:i+len(source)]==source]
    self.assertEqual(len(source),len(target));self.assertEqual(len(matches),1)
    print('REAL_REGION_TEMPLATE_PREFLIGHT',matches[0],len(source),source,target)


  def test_typed_key_restore_and_swap_preserve_random_stream(self):
    engine=CanvasDebugger(sampler=self.loop,params=self.params,
        rng=jax.random.key(11),batch_size=1)
    engine.until(2);engine.save('typed')
    original=np.array(jax.random.key_data(engine.state.rng))
    engine.swap(position=1,source=0,target=1)
    np.testing.assert_array_equal(jax.random.key_data(engine.state.rng),original)
    first=np.array(engine.run().canvas)
    engine.restore('typed');engine.swap(position=1,source=0,target=1)
    np.testing.assert_array_equal(engine.run().canvas,first)


  def test_failed_gate_retains_the_same_model_for_controlled_retry(self):
    import tempfile
    import types
    from pathlib import Path
    from unittest import mock
    from exp_code.semantic_intervention import debug_console
    chat=object();result=('engine','initial',0,11)
    retry=mock.Mock(return_value=result)
    with tempfile.TemporaryDirectory() as folder:
      with (
          mock.patch.object(debug_console,'verify_checkpoint',side_effect=TypeError('typed key fixture')),
          mock.patch('builtins.input',return_value='retry'),
          mock.patch('importlib.reload',return_value=types.SimpleNamespace(verify_checkpoint=retry)),
      ):
        actual=debug_console.resident_checkpoint_gate(chat,{},Path(folder))
      self.assertEqual(actual,result)
      self.assertIs(retry.call_args.args[0],chat)
      self.assertTrue((Path(folder)/'gate_failure_1.json').exists())


  def test_exploration_command_failure_keeps_model_and_restores_breakpoint(self):
    import tempfile
    from pathlib import Path
    from unittest import mock
    from exp_code.semantic_intervention.debug_console import Console
    model=object()
    with tempfile.TemporaryDirectory() as folder:
      console=Console(model,{},Path(folder))
      console.engine=mock.Mock();console.engine.snapshots={'base':object()}
      console.do_explore=mock.Mock(side_effect=RuntimeError('probe fixture failure'))
      self.assertFalse(console.onecmd('explore'))
      self.assertIs(console.chat,model)
      console.engine.restore.assert_called_once_with('base')
      self.assertTrue((Path(folder)/'command_errors.jsonl').exists())


if __name__=='__main__':unittest.main()
