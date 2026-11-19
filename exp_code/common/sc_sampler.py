"""SC-only ablations composed with the unchanged upstream generation loop.

The upstream step samples the current canvas from unmodified logits first.
Only the embeddings passed to the NEXT step are changed. Upstream adaptive
stopping, prompt prefill, EOS handling and multi-canvas generation are retained.
"""
from __future__ import annotations

import dataclasses
import functools

import jax.numpy as jnp
import jax
from typing import Any
from gemma.diffusion import _chat_sampler, _sampler

from exp_code.common.sc_modes import SelfConditioningMode


def _init_fields(instance):
  return {f.name: getattr(instance, f.name)
          for f in dataclasses.fields(instance) if f.init}


@dataclasses.dataclass(frozen=True, kw_only=True)
class SCDiffusionSampler(_sampler.DiffusionSampler):
  sc_mode: SelfConditioningMode = SelfConditioningMode()
  entropy_callback: Any = dataclasses.field(default=None, repr=False)

  def sample_step(self, **kwargs):
    output = super().sample_step(**kwargs)
    if self.entropy_callback is not None:
      log_probs = jax.nn.log_softmax(output.logits.astype(jnp.float32))
      probs = jnp.exp(log_probs)
      entropy = -jnp.sum(probs*jnp.where(probs==0,0.,log_probs),axis=-1)
      step = jnp.rint((1-kwargs['current_noise_proportion'][0])*self.max_denoising_steps).astype(jnp.int32)
      stop = self.early_stop_fn.should_stop(step=step,
          canvas=output.sampled_tokens,previous_canvas=kwargs['canvas'],logits=output.logits)
      jax.debug.callback(self.entropy_callback,step,entropy,stop,ordered=True)
    logits = self.sc_mode.transform_logits(output.logits)
    if logits is output.logits:
      if self.sc_mode.beta == 1.0:
        return output
      return output.replace(sc_embeddings=output.sc_embeddings * self.sc_mode.beta)
    if logits is None:
      embeddings = jnp.zeros_like(output.sc_embeddings)
    else:
      embeddings = self.model.apply(
          {'params': kwargs['params']}, logits,
          method=lambda model, values: model.embedder.encode_logits(values),
      )
    return output.replace(sc_embeddings=embeddings)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SCSampler(_chat_sampler.Sampler):
  sc_mode: SelfConditioningMode = SelfConditioningMode()
  entropy_callback: Any = dataclasses.field(default=None, repr=False)

  def _initialize_sampler_loop(self, sampling):
    original = super()._initialize_sampler_loop(sampling)
    return SCDiffusionSampler(**_init_fields(original), sc_mode=self.sc_mode,
                              entropy_callback=self.entropy_callback)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class SCChatSampler(_chat_sampler.ChatSampler):
  sc_mode: SelfConditioningMode = SelfConditioningMode()

  @functools.cached_property
  def sampler(self):
    return SCSampler(**_init_fields(super().sampler), sc_mode=self.sc_mode)


def sampler_for_condition(condition,*,model,**kwargs):
  """Construct SC bottleneck or post-MLP branch-strength interventions."""
  if condition['kind']=='branch_scaled':
    from exp_code.common.sc_strength import scaled_model
    model=scaled_model(model,float(condition['beta']))
    mode=SelfConditioningMode(kind='full')
  else:
    mode=SelfConditioningMode(kind=condition['kind'])
  return SCSampler(model=model,sc_mode=mode,**kwargs)
