"""Breakpoint/restore over a real canvas; snapshots never contain model weights.

Steps and RNG splitting follow the pinned upstream DiffusionSampler. Edits
occur after sampling step t; SC and canvas changes are logged separately.
"""
import dataclasses
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from gemma.diffusion import _sampler
from exp_code.common.acceptance_trace import AcceptanceRecorder,TracedPredictions


@dataclasses.dataclass(frozen=True)
class DebugState:
  step: int
  canvas: Any
  sc_embeddings: Any
  rng: Any
  done: Any
  sc_logits: Any = None


class CanvasDebugger:
  """One canvas, arbitrary repeatable branches; no reset of the noise schedule."""
  def __init__(self,*,sampler,params,rng,batch_size,cache=None,
               full_attention_mask=None,max_snapshots=4):
    self.sampler=sampler;self.params=params;self.cache=cache
    self.batch_size=batch_size;self.max_snapshots=max_snapshots
    self.snapshots={};self.edits=[]
    length=sampler.canvas_length
    if cache is None:
      cache_length=None;samples=None
      self.positions=jnp.broadcast_to(jnp.arange(length),(batch_size,length))
    else:
      layer=next(iter(cache.values()));cache_length=layer['k'].shape[1]
      samples=layer['end_index']
      offset=samples
      if full_attention_mask is not None:
        valid=(jnp.arange(cache_length)[None,:]<samples[:,None])&full_attention_mask
        offset=jnp.sum(valid.astype(jnp.int32),axis=-1)
      self.positions=offset[:,None]+jnp.arange(length)[None,:]
    self.attention=_sampler._make_global_attention_mask(batch_size=batch_size,
        canvas_length=length,cache_length=cache_length,num_valid_tokens=samples,
        full_attention_mask=full_attention_mask)
    self.sliding=_sampler._make_block_local_attention_mask(batch_size=batch_size,
        canvas_length=length,sliding_window_size=sampler.sliding_window_size,
        cache_length=cache_length,num_valid_tokens=samples)
    initial_rng,step_rng=jax.random.split(rng)
    canvas=sampler.diffusion_process.get_initial_sample(rng=initial_rng,
        batch_size=batch_size,canvas_length=length,text_vocab_size=sampler.text_vocab_size)
    self.state=DebugState(0,canvas,jnp.zeros((batch_size,length,sampler.model.config.embed_dim),
        dtype=jnp.bfloat16),step_rng,jnp.zeros(batch_size,dtype=bool))
    self.recorder=AcceptanceRecorder(batch_size,sampler.end_tokens,sampler.max_denoising_steps)
    instrumented=dataclasses.replace(sampler,sample_from_predictions=TracedPredictions(
        entropy_bound=sampler.sample_from_predictions.entropy_bound,
        text_vocab_size=sampler.text_vocab_size,max_steps=sampler.max_denoising_steps,
        top_k=min(16,sampler.text_vocab_size),callback=self.recorder.record,
        early_stop_fn=sampler.early_stop_fn))
    def sample_with_carry_cast(**kwargs):
      out=instrumented.sample_step(**kwargs)
      # Match the upstream while-body boundary. Returning a float32 SC tensor
      # first and casting it in a separate eager dispatch can change the dot
      # lowering/rounding relative to the native fused bfloat16 carry.
      print('SC_CARRY_DTYPE_BOUNDARY',out.sc_embeddings.dtype,'->',kwargs['sc_embeddings'].dtype,flush=True)
      return out.replace(sc_embeddings=out.sc_embeddings.astype(kwargs['sc_embeddings'].dtype))
    self._sample_step=jax.jit(sample_with_carry_cast)
    self._captured_logits=None
    def capture_logits(values):self._captured_logits=np.array(values,copy=True)
    def advance_native(params,cache,full_mask,carry,stop_at):
      # Keep the original while-carry and derive attention inside the same
      # compiled region. Returning a separate sample_step changes the GPU
      # compilation context of the nonzero-SC forward pass.
      if cache is None:
        cache_size=None;valid=None
        positions=jnp.broadcast_to(jnp.arange(length),(batch_size,length))
      else:
        layer=next(iter(cache.values()));cache_size=layer['k'].shape[1]
        valid=layer['end_index'];offset=valid
        if full_mask is not None:
          prefix=(jnp.arange(cache_size)[None,:]<valid[:,None])&full_mask
          offset=jnp.sum(prefix.astype(jnp.int32),axis=-1)
        positions=offset[:,None]+jnp.arange(length)[None,:]
      attention=_sampler._make_global_attention_mask(batch_size=batch_size,
          canvas_length=length,cache_length=cache_size,num_valid_tokens=valid,
          full_attention_mask=full_mask)
      sliding=_sampler._make_block_local_attention_mask(batch_size=batch_size,
          canvas_length=length,cache_length=cache_size,num_valid_tokens=valid,
          sliding_window_size=sampler.sliding_window_size)
      noises=1.-jnp.arange(sampler.max_denoising_steps+1)/sampler.max_denoising_steps
      def cond(c):
        return (~jnp.all(c.done))&(c.step<stop_at)&(c.step<sampler.max_denoising_steps)
      def body(c):
        next_rng,sample_rng=jax.random.split(c.rng)
        out=instrumented.sample_step(canvas=c.canvas,sc_embeddings=c.sc_embeddings,
            cache=cache,positions=positions,attention_mask=attention,
            sliding_attention_mask=sliding,params=params,rng=sample_rng,
            current_noise_proportion=jnp.full((batch_size,),noises[c.step]),
            target_noise_proportion=jnp.full((batch_size,),noises[c.step+1]))
        done=c.done|sampler.early_stop_fn.should_stop(step=c.step,
            canvas=out.sampled_tokens,previous_canvas=c.canvas,logits=out.logits)
        boundary=(c.step+1>=stop_at)|jnp.all(done)|(c.step+1>=sampler.max_denoising_steps)
        jax.lax.cond(boundary,
            lambda _:jax.debug.callback(capture_logits,out.logits,ordered=True),
            lambda _:None,operand=None)
        return _sampler._WhileLoopCarry(step=c.step+1,
            canvas=jnp.where(c.done[:,None],c.canvas,out.sampled_tokens),
            sc_embeddings=out.sc_embeddings.astype(c.sc_embeddings.dtype),
            rng=next_rng,done=done)
      return jax.lax.while_loop(cond,body,carry)
    self._advance_native=jax.jit(advance_native)
    self.full_attention_mask=full_attention_mask
    # Upstream constructs this schedule INSIDE its jitted canvas loop. Build
    # it in the same compiled context; eager divide/subtract can round to a
    # different float32 value and change temperature-shaped BF16 predictions.
    self.noises=jax.jit(lambda:1.-jnp.arange(sampler.max_denoising_steps+1)/sampler.max_denoising_steps)()
    # Parameter arrays are dynamic inputs, never captured as JIT constants.
    def swap_and_encode(params,logits,batch,position,source,target):
      a=logits[batch,position,source];b=logits[batch,position,target]
      changed=logits.at[batch,position,source].set(b).at[batch,position,target].set(a)
      encoded=sampler.model.apply({'params':params},changed,
          method=lambda model,x:model.embedder.encode_logits(x))
      return changed,encoded.astype(jnp.bfloat16)
    self._swap_and_encode=jax.jit(swap_and_encode)
    def mix_and_encode(params,logits,batch,position,q,alpha):
      log_p=jax.nn.log_softmax(logits[batch,position].astype(jnp.float32))
      log_q=jnp.where(q>0,jnp.log(q),-jnp.inf)
      row=jnp.logaddexp(log_p+jnp.log1p(-alpha),log_q+jnp.log(alpha))
      changed=logits.at[batch,position].set(row.astype(logits.dtype))
      encoded=sampler.model.apply({'params':params},changed,
          method=lambda model,x:model.embedder.encode_logits(x))
      return changed,encoded.astype(jnp.bfloat16)
    self._mix_and_encode=jax.jit(mix_and_encode)

  def _editable(self,batch,position):
    if not 0<=batch<self.batch_size or not 0<=position<self.sampler.canvas_length:
      raise ValueError('Invalid batch or position')
    if self.state.sc_logits is None:raise ValueError('Run a step before editing its next-step inputs')
    if self.finished or bool(self.state.done[batch]):
      raise ValueError('Sample already stopped; restore an earlier breakpoint')

  def _embedding_effect(self,before,after):
    a=np.asarray(before,dtype=np.float32);b=np.asarray(after,dtype=np.float32)
    norm=float(np.linalg.norm(a));change=float(np.linalg.norm(b-a))
    return dict(sc_embedding_delta_l2=change,
        sc_embedding_relative_delta=change/max(norm,1e-12),
        sc_embedding_cosine=float(np.dot(a,b)/(norm*np.linalg.norm(b)))
            if norm and np.linalg.norm(b) else None)

  @property
  def finished(self):
    return self.state.step>=self.sampler.max_denoising_steps or bool(jnp.all(self.state.done))

  def step(self):
    if self.finished:raise ValueError('Canvas already stopped; restore an earlier breakpoint')
    s=self.state
    carry=_sampler._WhileLoopCarry(step=jnp.int32(s.step),canvas=s.canvas,
        sc_embeddings=s.sc_embeddings,rng=s.rng,done=s.done)
    self._captured_logits=None
    out=self._advance_native(self.params,self.cache,self.full_attention_mask,carry,jnp.int32(s.step+1))
    jax.block_until_ready(out.canvas);jax.effects_barrier()
    self.recorder.record_context(s.step,np.asarray(self.positions[:,0]))
    if self._captured_logits is None:raise RuntimeError('Missing breakpoint logits')
    self.state=DebugState(int(out.step),out.canvas,out.sc_embeddings,out.rng,out.done,
        jnp.asarray(self._captured_logits))
    return self.state

  def until(self,completed_steps):
    if completed_steps<self.state.step:raise ValueError('Use restore to move backwards')
    if completed_steps>self.sampler.max_denoising_steps:raise ValueError('Step exceeds original cap')
    while self.state.step<completed_steps and not self.finished:self.step()
    return self.state

  def run(self):
    while not self.finished:self.step()
    return self.state

  def save(self,name):
    if name not in self.snapshots and len(self.snapshots)>=self.max_snapshots:
      raise ValueError('Snapshot limit reached; delete an unused snapshot first')
    r=self.recorder
    self.snapshots[name]=(self.state,tuple(r.events),r.finished.copy(),r.done.copy(),
        tuple(r.final_canvas),r.canvas_index,r.previous_step,tuple(self.edits))

  def restore(self,name):
    state,events,finished,done,final,index,previous,edits=self.snapshots[name]
    self.state=state;self.edits=list(edits)
    r=self.recorder;r.events=list(events);r.finished=finished.copy();r.done=done.copy()
    r.final_canvas=list(final);r.canvas_index=index;r.previous_step=previous
    return state

  def swap(self,*,position,source,target,batch=0):
    s=self.state
    self._editable(batch,position)
    if source==target or not (0<=source<self.sampler.text_vocab_size and 0<=target<self.sampler.text_vocab_size):
      raise ValueError('Invalid or identical token IDs')
    before=np.asarray(s.sc_logits[batch,position,jnp.array([source,target])])
    if not np.isfinite(before).all():raise ValueError('Cannot swap forbidden/nonfinite logits')
    logits,encoded=self._swap_and_encode(self.params,s.sc_logits,
        jnp.int32(batch),jnp.int32(position),jnp.int32(source),jnp.int32(target))
    # Preserve other SC rows bitwise, including any prior zero-SC override.
    embeddings=s.sc_embeddings.at[batch,position].set(encoded[batch,position])
    self.state=dataclasses.replace(s,sc_logits=logits,sc_embeddings=embeddings)
    record=dict(kind='sc_swap',after_step_number=s.step,after_step_index=s.step-1,
        first_affected_step_number=s.step+1,batch=batch,position=position,
        source_token_id=source,target_token_id=target,
        source_logit_before=float(before[0]),target_logit_before=float(before[1]),
        source_logit_after=float(before[1]),target_logit_after=float(before[0]),
        **self._embedding_effect(s.sc_embeddings[batch,position],embeddings[batch,position]))
    self.edits.append(record)
    return record

  def mix(self,*,position,weights,alpha,batch=0):
    """q is a sparse custom distribution; next SC uses (1-alpha)*p + alpha*q."""
    self._editable(batch,position)
    if not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('alpha must be in [0,1]')
    q=np.zeros(self.sampler.text_vocab_size,dtype=np.float32)
    for token,weight in weights.items():
      if not isinstance(token,int) or not 0<=token<len(q) or not np.isfinite(weight) or weight<0:
        raise ValueError('Invalid token or distribution weight')
      if weight and not np.isfinite(float(self.state.sc_logits[batch,position,token])):
        raise ValueError('Distribution includes a forbidden token')
      q[token]=weight
    if not np.isfinite(q.sum()) or q.sum()<=0:raise ValueError('Distribution must have positive finite mass')
    q/=q.sum();s=self.state
    before=jax.nn.softmax(s.sc_logits[batch,position].astype(jnp.float32))
    if alpha:
      logits,encoded=self._mix_and_encode(self.params,s.sc_logits,jnp.int32(batch),
          jnp.int32(position),jnp.asarray(q),jnp.float32(alpha))
      embeddings=s.sc_embeddings.at[batch,position].set(encoded[batch,position])
      self.state=dataclasses.replace(s,sc_logits=logits,sc_embeddings=embeddings)
    else:embeddings=s.sc_embeddings
    after=jax.nn.softmax(self.state.sc_logits[batch,position].astype(jnp.float32))
    requested=(1-alpha)*before+alpha*jnp.asarray(q)
    record=dict(kind='sc_distribution_mix',after_step_number=s.step,
        first_affected_step_number=s.step+1,batch=batch,position=position,alpha=alpha,
        q={int(i):float(q[i]) for i in np.flatnonzero(q)},
        requested_vs_stored_probability_l1=float(jnp.sum(jnp.abs(after-requested))),
        probability_note='Actual float32 softmax of stored logits; dtype rounding is measured',
        **self._embedding_effect(s.sc_embeddings[batch,position],embeddings[batch,position]))
    self.edits.append(record);return record

  def change_canvas(self,*,position,token,batch=0):
    self._editable(batch,position)
    if not 0<=token<self.sampler.text_vocab_size:raise ValueError('Invalid token ID')
    s=self.state;before=int(s.canvas[batch,position])
    self.state=dataclasses.replace(s,canvas=s.canvas.at[batch,position].set(token))
    record=dict(kind='canvas_token',after_step_number=s.step,
        first_affected_step_number=s.step+1,batch=batch,position=position,
        token_before=before,token_after=token,sc_unchanged=True)
    self.edits.append(record);return record

  def zero_sc(self,*,position,batch=0):
    self._editable(batch,position);s=self.state
    changed=s.sc_embeddings.at[batch,position].set(0)
    self.state=dataclasses.replace(s,sc_embeddings=changed)
    record=dict(kind='sc_embedding_zero',after_step_number=s.step,
        first_affected_step_number=s.step+1,batch=batch,position=position,
        retained_logits_are_not_actual_zero_embedding=True,
        **self._embedding_effect(s.sc_embeddings[batch,position],changed[batch,position]))
    self.edits.append(record);return record

  def region(self,*,start,tokens,alpha=1.,batch=0):
    """Aligned per-position q templates, no insertion or canvas changes."""
    if not tokens or start<0 or start+len(tokens)>self.sampler.canvas_length:
      raise ValueError('Invalid nonempty region')
    if not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('alpha must be in [0,1]')
    for offset,token in enumerate(tokens):
      self._editable(batch,start+offset)
      if not isinstance(token,int) or not 0<=token<self.sampler.text_vocab_size:
        raise ValueError('Invalid region token ID')
      if not np.isfinite(float(self.state.sc_logits[batch,start+offset,token])):
        raise ValueError('Region contains a forbidden candidate')
    before=self.state;edit_count=len(self.edits)
    try:
      for offset,token in enumerate(tokens):
        self.mix(position=start+offset,weights={token:1.},alpha=alpha,batch=batch)
      record=dict(kind='sc_region',after_step_number=before.step,
          first_affected_step_number=before.step+1,batch=batch,position=start,
          length=len(tokens),target_token_ids=list(tokens),alpha=alpha,
          aligned_template=True,canvas_unchanged=True,
          per_row_embedding_delta=[e['sc_embedding_delta_l2'] for e in self.edits[edit_count:]])
      self.edits.append(record);return record
    except Exception:
      self.state=before;del self.edits[edit_count:]
      raise

  def inspect(self,position,*,batch=0,k=8):
    if not 0<=batch<self.batch_size or not 0<=position<self.sampler.canvas_length:
      raise ValueError('Invalid batch or position')
    if self.state.sc_logits is None:raise ValueError('No prediction yet')
    row=self.state.sc_logits[batch,position].astype(jnp.float32)
    probabilities=jax.nn.softmax(row)
    values,ids=jax.lax.top_k(row,min(k,len(row)))
    return dict(completed_steps=self.state.step,position=position,batch=batch,
        latest_pending_edit=next((e['kind'] for e in reversed(self.edits)
            if e.get('after_step_number')==self.state.step and e['batch']==batch
            and e['position']==position and e['kind']!='canvas_token'),'model_prediction'),
        canvas_token_id=int(self.state.canvas[batch,position]),
        active_at_last_step=bool(self.recorder.events[-1]['active'][batch]),
        accepted_at_last_step=bool(self.recorder.events[-1]['accepted'][batch,position]),
        stopped=bool(self.state.done[batch]),token_ids=np.asarray(ids).tolist(),
        logits=np.asarray(values).tolist(),probabilities=np.asarray(probabilities[ids]).tolist(),
        note='SC logits for the next step; pending swaps are included')
