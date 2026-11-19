"""Read-only native SC-carry observation for the first three denoising steps."""
import dataclasses
import jax
import jax.numpy as jnp
from exp_code.common.sc_sampler import SCSampler,SCDiffusionSampler,_init_fields


@dataclasses.dataclass(frozen=True,kw_only=True)
class CarryLoop(SCDiffusionSampler):
  sc_callback: object = None

  def sample_step(self,**kwargs):
    step=jnp.rint((1-kwargs['current_noise_proportion'][0])*self.max_denoising_steps).astype(jnp.int32)
    if self.sc_callback is not None:
      jax.lax.cond(step<3,
          lambda _:jax.debug.callback(self.sc_callback,step,kwargs['sc_embeddings'],ordered=True),
          lambda _:None,operand=None)
    return super().sample_step(**kwargs)


@dataclasses.dataclass(frozen=True,kw_only=True)
class CarryChat(SCSampler):
  sc_callback: object = None

  def _initialize_sampler_loop(self,sampling):
    base=super()._initialize_sampler_loop(sampling)
    return CarryLoop(**_init_fields(base),sc_callback=self.sc_callback)
