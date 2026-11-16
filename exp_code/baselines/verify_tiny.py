"""Actual upstream tiny model: SC boundary and full-loop equality, no weights.

This is a correctness gate, not a capability reproduction result.
"""
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
from gemma.diffusion import _models, _sampler, _transformer, _early_stopping
from gemma.gm.nn.gemma4 import _config, _modules
from gemma.gm.text import _sampling

from exp_code.common.sc_modes import SelfConditioningMode
from exp_code.common.sc_sampler import SCDiffusionSampler
from exp_code.common.probability import softmax, top_k_logits, threshold_logits
from exp_code.common.dg_runner import run_fixed_trajectory
from exp_code.baselines.entropy_recorder import EntropyRecorder


def main():
  config = _config.TransformerConfig(
      num_embed=32, embed_dim=8, num_heads=2, num_kv_heads=1,
      head_dim=4, hidden_dim=16,
      attention_types=[_modules.AttentionType.GLOBAL],
      kv_cache_sharing_config=None, use_post_attn_norm=True,
      use_post_ffw_norm=True, final_logit_softcap=None,
      global_rope_proportion=1.0,
  )
  model = _models.DiffusionGemma_26B_A4B(
      config=config,
      self_conditioning_config=_transformer.SelfConditioningConfig(
          features=8, hidden_dim=16),
  )
  settings = dict(
      model=model, end_tokens=(99,), forbidden_tokens=None,
      sampling=_sampling.Greedy(), cache_length=16, special_tokens=None,
      sample_from_predictions=_sampler.SampleFromPredictions(text_vocab_size=32),
      canvas_length=4, max_denoising_steps=4, text_vocab_size=32,
  )
  native = _sampler.DiffusionSampler(**settings)
  params = model.init(
      rngs=jax.random.PRNGKey(0), tokens=jnp.ones((1,4), jnp.int32),
      sc_embeddings=jnp.ones((1,4,8), jnp.bfloat16),
      attention_mask=jnp.ones((1,4,4), jnp.bool_),
      method=model.call_with_self_conditioning,
  )['params']
  generate = dict(canvas_length=4, max_denoising_steps=4, batch_size=1,
                  cache=None, params=params, rng=jax.random.PRNGKey(42))
  expected = native.sample_next_canvas(**generate)
  full = SCDiffusionSampler(**settings)
  np.testing.assert_array_equal(expected, full.sample_next_canvas(**generate))
  step = dict(
      canvas=jnp.ones((1,4), jnp.int32),
      sc_embeddings=jnp.zeros((1,4,8), jnp.bfloat16), cache=None,
      positions=jnp.arange(4)[None,:],
      attention_mask=jnp.ones((1,4,4), jnp.bool_),
      current_noise_proportion=jnp.ones((1,)),
      target_noise_proportion=jnp.zeros((1,)), params=params,
      rng=jax.random.PRNGKey(1),
  )
  original = native.sample_step(**step)
  for kind in ('full', 'disabled', 'top_k', 'threshold'):
    sampler = SCDiffusionSampler(**settings, sc_mode=SelfConditioningMode(kind=kind))
    actual = sampler.sample_step(**step)
    # A change to the SC boundary cannot change this step's visible prediction.
    np.testing.assert_array_equal(actual.sampled_tokens, original.sampled_tokens)
    np.testing.assert_array_equal(actual.logits, original.logits)
    transformed = sampler.sc_mode.transform_logits(original.logits)
    if transformed is None:
      np.testing.assert_array_equal(actual.sc_embeddings, 0)
    else:
      embeddings = model.apply({'params':params}, transformed,
          method=lambda m,x:m.embedder.encode_logits(x))
      np.testing.assert_allclose(np.asarray(actual.sc_embeddings, dtype=float),
                                 np.asarray(embeddings, dtype=float))
    result = sampler.sample_next_canvas(**generate)
    assert result.shape == (1,4)
  values = np.random.default_rng(3).normal(size=(2,3,32)).astype(np.float32)
  for fn, arg in ((top_k_logits,8),(threshold_logits,0.03)):
    np.testing.assert_allclose(softmax(fn(values,arg)),
        np.asarray(softmax(jax.jit(lambda x:fn(x,arg))(jnp.asarray(values)))),
        atol=2e-7, rtol=2e-5)
  # Entire loop must match original when retaining every vocabulary entry.
  identity = dataclasses.replace(full, sc_mode=SelfConditioningMode(kind='top_k',top_k=32))
  np.testing.assert_array_equal(expected, identity.sample_next_canvas(**generate))
  for beta in (0., 0.2, 1.):
    scaled = dataclasses.replace(full, sc_mode=SelfConditioningMode(kind='scaled',beta=beta))
    actual = scaled.sample_step(**step)
    np.testing.assert_array_equal(actual.sampled_tokens,original.sampled_tokens)
    np.testing.assert_allclose(np.asarray(actual.sc_embeddings,dtype=float),
        np.asarray(original.sc_embeddings * beta,dtype=float))
  # Real 48-step schedule and adaptive stopping must match the native loop.
  for stopper in (_early_stopping.NoEarlyStop(),
                  _early_stopping.EntropyEarlyStop(entropy_threshold=100)):
    observer = dataclasses.replace(native, early_stop_fn=stopper)
    kwargs = dict(generate, max_denoising_steps=48)
    reference = observer.sample_next_canvas(**kwargs)
    trace = run_fixed_trajectory(sampler=observer,stop_early=True,**{
        k:v for k,v in kwargs.items() if k!='canvas_length'})
    np.testing.assert_array_equal(reference,trace.final_canvas)
    expected_steps=48 if isinstance(stopper,_early_stopping.NoEarlyStop) else 1
    assert len(trace.steps)==expected_steps
  zeros = run_fixed_trajectory(sampler=native, sc_mode=SelfConditioningMode(kind='scaled',beta=0),
      **{k:v for k,v in generate.items() if k!='canvas_length'})
  disabled = run_fixed_trajectory(sampler=native, sc_mode=SelfConditioningMode(kind='disabled'),
      **{k:v for k,v in generate.items() if k!='canvas_length'})
  np.testing.assert_array_equal(zeros.final_canvas,disabled.final_canvas)
  for kind in ('full','disabled','top_k','threshold'):
    recorder=EntropyRecorder(4,canvas_length=4)
    plain=SCDiffusionSampler(**settings,sc_mode=SelfConditioningMode(kind=kind))
    observed=dataclasses.replace(plain,entropy_callback=recorder.record)
    batch=dict(generate,batch_size=2)
    expected=plain.sample_next_canvas(**batch)
    result=observed.sample_next_canvas(**batch)
    jax.effects_barrier()
    np.testing.assert_array_equal(expected,result)
    for i in range(2):
      payload=recorder.sample(i,np.asarray(result[i]),(99,))
      assert payload['steps'].tolist()==[4]
      assert payload['entropy'].shape==(1,4,4)
      assert np.isfinite(payload['entropy']).all()
  recorder=EntropyRecorder(4,canvas_length=4)
  observed=dataclasses.replace(full,entropy_callback=recorder.record,
      early_stop_fn=_early_stopping.EntropyEarlyStop(entropy_threshold=100))
  result=observed.sample_next_canvas(**dict(generate,batch_size=2))
  jax.effects_barrier()
  assert recorder.sample(0,np.asarray(result[0]),(99,))['steps'].tolist()==[1]
  print('BATCHED_ENTROPY_NATIVE_EQUIVALENCE_VERIFIED',flush=True)
  print('TINY_SC_BOUNDARY_AND_LOOP_VERIFIED', jax.__version__, jax.devices(), flush=True)


if __name__ == '__main__':
  main()
