"""Scale the SC branch AFTER pre-norm/MLP and BEFORE residual addition.

Matches DGtest SCALING.md: post_norm(canvas_embedding + beta * SC_branch).
Scaling the incoming SC embedding instead can be undone by pre_norm.
The upstream checkpoint parameter names/tree are unchanged.
"""
import dataclasses
import math
from gemma.diffusion import _transformer as upstream


class BranchScaledSelfConditioning(upstream.SelfConditioning):
  beta: float = 1.0

  def __call__(self,*,canvas_embeddings,self_conditioning_signal):
    normed=self.pre_norm(self_conditioning_signal)
    branch=self.ffw(normed)
    return self.post_norm(canvas_embeddings + self.beta*branch)


@dataclasses.dataclass(frozen=True,kw_only=True)
class BranchScaledConfig(upstream.SelfConditioningConfig):
  beta: float = 1.0

  def make(self):
    if not math.isfinite(self.beta) or self.beta<0:
      raise ValueError('SC branch scale must be finite and nonnegative')
    return BranchScaledSelfConditioning(features=self.features,hidden_dim=self.hidden_dim,beta=self.beta)


def scaled_model(model,beta):
  cfg=model.self_conditioning_config
  if cfg is None:
    cfg=upstream.SelfConditioningConfig(features=model.config.embed_dim,hidden_dim=model.config.hidden_dim)
  return model.clone(self_conditioning_config=BranchScaledConfig(
      features=cfg.features,hidden_dim=cfg.hidden_dim,beta=beta))
