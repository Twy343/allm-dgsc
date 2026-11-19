"""Instrument the actual upstream selection/renoising decision, not token changes.

Selection kernel adapted verbatim from gemma.diffusion._sampler
(Copyright 2026 DeepMind Technologies Limited, Apache-2.0).
"""
import csv
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from gemma.diffusion import _sampler


@dataclasses.dataclass(frozen=True)
class TracedPredictions(_sampler.SampleFromPredictions):
  callback: object = None
  early_stop_fn: object = None
  max_steps: int = 48
  top_k: int = 8
  watched_token_ids: tuple = ()

  def __call__(self, *, rng, denoiser_logits, canvas,
               current_noise_proportion, target_noise_proportion):
    categorical_rng, noise_rng = jax.random.split(rng)
    denoiser_tokens = jax.random.categorical(
        categorical_rng, denoiser_logits.astype(jnp.float32))
    log_probs = jax.nn.log_softmax(denoiser_logits.astype(jnp.float32))
    probs = jnp.exp(log_probs)
    safe_log_probs = jnp.where(probs == 0, 0.0, log_probs)
    entropy = -jnp.sum(safe_log_probs * probs, axis=-1)
    sorted_index = jnp.argsort(entropy, axis=-1)
    sorted_entropy = jnp.take_along_axis(entropy, sorted_index, axis=-1)
    accumulated = jnp.cumsum(sorted_entropy, axis=-1)
    sorted_mask = accumulated - sorted_entropy <= self.entropy_bound
    accepted = jnp.zeros_like(sorted_index, dtype=jnp.bool_).at[
        jnp.arange(canvas.shape[0])[:, None], sorted_index].set(sorted_mask)
    accepted = accepted | (target_noise_proportion <= 0.0)[:, None]
    random_tokens = jax.random.randint(noise_rng, shape=canvas.shape,
                                      minval=0, maxval=self.text_vocab_size)
    output = jnp.where(accepted, denoiser_tokens, random_tokens)
    if self.callback is not None:
      step = jnp.rint((1-current_noise_proportion[0])*self.max_steps).astype(jnp.int32)
      stop = self.early_stop_fn.should_stop(
          step=step, canvas=output, previous_canvas=canvas, logits=denoiser_logits)
      top_logits, top_ids = jax.lax.top_k(denoiser_logits, self.top_k)
      top_probs = jnp.take_along_axis(probs, top_ids, axis=-1)
      jax.debug.callback(self.callback, step, current_noise_proportion,
          target_noise_proportion, canvas, denoiser_tokens, accepted, output,
          entropy, top_ids, top_probs, top_logits.astype(jnp.float32), stop,
          ordered=True)
      if self.watched_token_ids:
        ids = jnp.array(self.watched_token_ids)
        jax.debug.callback(self.callback.__self__.record_watched,
            denoiser_logits[...,ids].astype(jnp.float32), probs[...,ids], ordered=True)
    return output


def write_csv(path, rows):
  if not rows:
    raise ValueError('Refusing an empty trace CSV')
  with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


class AcceptanceRecorder:
  """Keep active steps only; upstream freezes stopped members of each batch.

  A selected token may be reopened next step. EOS is checked on the finished
  canvas, exactly as upstream, not on intermediate noisy canvases.
  """
  def __init__(self, batch_size, end_tokens, max_steps=48, progress_label=''):
    self.batch_size = batch_size
    self.end_tokens = tuple(end_tokens)
    self.max_steps = max_steps
    self.events = []
    self.finished = np.zeros(batch_size, bool)
    self.done = np.zeros(batch_size, bool)
    self.final_canvas = [None]*batch_size
    self.canvas_index = -1
    self.previous_step = None
    self.progress_label = progress_label

  def record_context(self, step, offsets):
    if int(step)!=self.events[-1]['step']:raise ValueError('Trace context mismatch')
    self.events[-1]['position_offsets']=np.array(offsets,copy=True)

  def record_watched(self, logits, probabilities):
    self.events[-1]['watched_logits']=np.array(logits,copy=True)
    self.events[-1]['watched_probabilities']=np.array(probabilities,copy=True)

  def record(self, step, current, target, before, sampled, accepted, after,
             entropy, top_ids, top_probs, top_logits, stop):
    step = int(step)
    if step == 0:
      if self.canvas_index >= 0:
        self._finish_canvas()
      self.canvas_index += 1
      self.done = self.finished.copy()
      self.final_canvas = [None]*self.batch_size
    elif self.previous_step != step-1:
      raise ValueError('Nonsequential trace callbacks')
    active = ~self.done
    values = [np.array(x, copy=True) for x in (
        before, sampled, accepted, after, entropy, top_ids, top_probs, top_logits)]
    self.events.append(dict(canvas=self.canvas_index, step=step, active=active.copy(),
        current=np.array(current), target=np.array(target), stop=np.array(stop),
        before=values[0], sampled=values[1], accepted=values[2], after=values[3],
        entropy=values[4], top_ids=values[5], top_probs=values[6], top_logits=values[7]))
    for i in np.flatnonzero(active):
      self.final_canvas[i] = values[3][i]
    self.done |= np.asarray(stop, bool)
    self.previous_step = step
    if self.progress_label and (step%4==0 or self.done.all()):
      print('DENOISE_PROGRESS',self.progress_label,'canvas',self.canvas_index,
          'step',step+1,'active',int(active.sum()),'accepted',
          [int(values[2][i].sum()) for i in np.flatnonzero(active)],flush=True)

  def _finish_canvas(self):
    for i, tokens in enumerate(self.final_canvas):
      if tokens is not None and np.isin(tokens, self.end_tokens).any():
        self.finished[i] = True

  def export(self, folder, batch_position, case_id, seed, tokenizer):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    events = [e for e in self.events if e['active'][batch_position]]
    if not events:
      raise ValueError('No active trace for '+case_id)
    piece_cache = {}
    def piece(token):
      token = int(token)
      if token not in piece_cache:
        # Isolated token decoding is separate from SentencePiece spelling.
        piece_cache[token] = tokenizer.decode([token])
      return piece_cache[token]
    steps = []; tokens = []; previous = None; canvas = -1
    final_lengths={}
    for event in events:
      ids=event['after'][batch_position]
      ends=np.flatnonzero(np.isin(ids,self.end_tokens))
      final_lengths[event['canvas']]=int(ends[0])+1 if len(ends) else len(ids)
    for event in events:
      i = batch_position
      if canvas != event['canvas']:
        canvas = event['canvas']; previous = np.zeros_like(event['accepted'][i])
      accepted = event['accepted'][i]
      common = dict(case_id=case_id, seed=seed, canvas_index=canvas,
          step_index=event['step'], step_number=event['step']+1)
      steps.append(dict(**common, accepted_count=int(accepted.sum()),
          renoised_count=int((~accepted).sum()),
          accepted_positions=json.dumps(np.flatnonzero(accepted).tolist()),
          renoised_positions=json.dumps(np.flatnonzero(~accepted).tolist()),
          newly_accepted_count=int((accepted & ~previous).sum()),
          reopened_count=int((~accepted & previous).sum()),
          changed_count=int((event['before'][i]!=event['after'][i]).sum()),
          mean_entropy=float(event['entropy'][i].mean()),
          current_noise=float(event['current'][i]), target_noise=float(event['target'][i]),
          adaptive_stop_met=bool(event['stop'][i]),
          forced_accept_at_cap=bool(event['target'][i]<=0)))
      for pos in range(len(accepted)):
        before = int(event['before'][i,pos]); after = int(event['after'][i,pos])
        sampled = int(event['sampled'][i,pos]); ids = event['top_ids'][i,pos].tolist()
        tokens.append(dict(**common, canvas_position=pos,
            response_position=canvas*len(accepted)+pos,
            in_final_response=pos<final_lengths[canvas],
            model_position=int(event.get('position_offsets',np.zeros(self.batch_size))[i])+pos,
            input_token_id=before, input_token=piece(before),
            denoiser_token_id=sampled, denoiser_token=piece(sampled),
            output_token_id=after, output_token=piece(after),
            accepted=bool(accepted[pos]), renoised=bool(~accepted[pos]),
            changed=before!=after, reopened=bool(previous[pos] and not accepted[pos]),
            entropy=float(event['entropy'][i,pos]),
            top_token_ids=json.dumps(ids),
            top_tokens=json.dumps([piece(t) for t in ids],ensure_ascii=False),
            top_probabilities=json.dumps(event['top_probs'][i,pos].tolist()),
            top_shaped_logits=json.dumps(event['top_logits'][i,pos].tolist())))
        if 'watched_logits' in event:
          tokens[-1]['watched_logits']=json.dumps(event['watched_logits'][i,pos].tolist())
          tokens[-1]['watched_probabilities']=json.dumps(event['watched_probabilities'][i,pos].tolist())
      previous = accepted
    name = case_id.replace(':','-')
    write_csv(folder/f'{name}_steps.csv', steps)
    write_csv(folder/f'{name}_tokens.csv', tokens)
    np.savez_compressed(folder/f'{name}.npz', **{
        key: np.stack([e[key][batch_position] for e in events])
        for key in ('before','sampled','accepted','after','entropy','top_ids',
                    'top_probs','top_logits')},
        canvas_indices=np.array([e['canvas'] for e in events]),
        step_indices=np.array([e['step'] for e in events]))
    if 'watched_logits' in events[0]:
      np.savez_compressed(folder/f'{name}_watched.npz',
          logits=np.stack([e['watched_logits'][batch_position] for e in events]),
          probabilities=np.stack([e['watched_probabilities'][batch_position] for e in events]))
    last_by_canvas = {s['canvas_index']:s for s in steps}
    return dict(case_id=case_id, seed=seed, actual_total_steps=len(steps),
        canvases=len(last_by_canvas), steps_per_canvas=json.dumps([
            s['step_number'] for s in last_by_canvas.values()]),
        stop_reasons=json.dumps(['adaptive' if s['adaptive_stop_met'] else 'step_cap'
            for s in last_by_canvas.values()]),
        steps_csv=f'{name}_steps.csv', tokens_csv=f'{name}_tokens.csv')
