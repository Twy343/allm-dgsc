"""Full-SC observation and one-site SC-only swaps, retaining upstream loops."""
import dataclasses
import jax
import jax.numpy as jnp
from exp_code.common.sc_sampler import SCSampler, SCDiffusionSampler, _init_fields
from exp_code.semantic_intervention.intervention import swap_token_probabilities


@dataclasses.dataclass(frozen=True,kw_only=True)
class ObservedDiffusionSampler(SCDiffusionSampler):
  context_callback: object = None
  swap: tuple = ()  # step, batch, canvas offset, position, source ID, target ID, enabled
  row_callback: object = None

  def sample_step(self,**kwargs):
    out=super().sample_step(**kwargs)
    step=jnp.rint((1-kwargs['current_noise_proportion'][0])*self.max_denoising_steps).astype(jnp.int32)
    offsets=kwargs['positions'][:,0]
    if self.context_callback is not None:
      jax.debug.callback(self.context_callback,step,offsets,ordered=True)
    if not self.swap:return out
    target_step,batch,offset,pos,source,target,enabled=self.swap
    active=(step==target_step)&(offsets[batch]==offset)&enabled
    def modify(logits):
      altered=swap_token_probabilities(logits,batch_index=batch,position=pos,
          source_token_id=source,target_token_id=target)
      return self.model.apply({'params':kwargs['params']},altered,
          method=lambda model,x:model.embedder.encode_logits(x))
    if self.row_callback is not None:
      jax.debug.callback(self.row_callback,step,offsets[batch],
          out.logits[batch,pos].astype(jnp.float32),active,ordered=True)
    embeddings=jax.lax.cond(active,modify,lambda _:out.sc_embeddings,out.logits)
    return out.replace(sc_embeddings=embeddings)


@dataclasses.dataclass(frozen=True,kw_only=True)
class ObservedSampler(SCSampler):
  context_callback: object = None
  swap: tuple = ()
  row_callback: object = None

  def _initialize_sampler_loop(self,sampling):
    base=super()._initialize_sampler_loop(sampling)
    return ObservedDiffusionSampler(**_init_fields(base),context_callback=self.context_callback,
        swap=self.swap,row_callback=self.row_callback)
