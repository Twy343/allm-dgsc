"""One reproducible first-canvas observation, separate from capability scores."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
from pathlib import Path
import time

from exp_code.baselines.run import checkpoint_identity, run_lock, source_identity, write_json
from exp_code.common.sc_modes import SelfConditioningMode

PROMPT = '''Write a Python function merge_intervals(intervals). Normalize reversed
endpoint pairs, sort by start, and merge overlapping or touching intervals.
Return the merged list, and return [] for an empty input. Explain briefly and
provide the complete function.'''


def plot(traces, destination):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import numpy as np
  from matplotlib.colors import Normalize

  figure, axes = plt.subplots(1,len(traces),figsize=(3.2*len(traces),3.4),sharey=True,
                              layout='constrained',squeeze=False)
  norm=Normalize(0,4)
  for axis,(name,trace) in zip(axes[0],traces.items()):
    values=np.stack([step.entropy[0] for step in trace.steps])
    # Same 48-step y range for every condition; blank regions mean stopped.
    image=axis.imshow(values,origin='lower',aspect='auto',norm=norm,cmap='viridis',
                      extent=(-0.5,255.5,-0.5,len(values)-0.5))
    axis.set_ylim(-0.5,47.5)
    axis.set_xlim(-0.5,255.5)
    flag='stop' if trace.stopped_early else 'cap reached'
    axis.set_title(f'{name}\n{flag}: {len(values)}')
    axis.set_xlabel('Token position')
    axis.set_facecolor('#eeeeee')
  axes[0,0].set_ylabel('Denoising step (zero based)')
  figure.colorbar(image,ax=list(axes[0]),label='Token entropy (nats; color clipped at 4)')
  figure.suptitle('Adapted merge-intervals case, first 256-token canvas, seed 0')
  figure.savefig(destination,dpi=180)
  plt.close(figure)


def main():
  import dialog
  import jax
  import numpy as np
  from gemma import gm, diffusion
  from gemma.gm.text import _prefill
  from exp_code.common.dg_runner import run_fixed_trajectory

  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  if not any(d.platform=='gpu' for d in jax.devices()):
    raise RuntimeError('Allocate a GPU before checkpoint-backed observation')
  args.output.mkdir(parents=True,exist_ok=True)
  with run_lock(args.output):
    if (args.output/'summary.json').exists():
      raise FileExistsError('Observation already finished; use a new directory to rerun')
    identity=checkpoint_identity(args.checkpoint)
    model=diffusion.DiffusionGemma_26B_A4B()
    params=gm.ckpts.load_params(str(args.checkpoint),text_only=True)
    tokenizer=gm.text.Gemma4Tokenizer(path=str(args.checkpoint/'tokenizer.model'))
    high=diffusion.Sampler(model=model,params=params,tokenizer=tokenizer,
                           cache_length=4096,max_out_length=256)
    formatted=dialog.Conversation(dialog.User(PROMPT)).as_text(format=tokenizer.FORMAT)
    inputs=high._get_inputs(prompt=formatted,images=None,add_bos=True,
                            has_batch_dim=False,sharding=None)
    state=_prefill.prefill(model=model,params=params,input=inputs,last_state=None,
        cache_length=4096,pad_length=high.pad_length,rng=jax.random.PRNGKey(0),
        sharding=None,max_out_length=256)
    native=high._initialize_sampler_loop(high.sampling)
    _,rng=jax.random.split(state.rng)
    modes={f'beta={beta:g}':SelfConditioningMode(kind='scaled',beta=beta)
           for beta in (1.,0.8,0.6,0.4,0.2,0.)}
    modes.update(full=SelfConditioningMode(),disabled=SelfConditioningMode(kind='disabled'),
                 top_k_8=SelfConditioningMode(kind='top_k'),
                 threshold_003=SelfConditioningMode(kind='threshold'))
    write_json(args.output/'protocol.json',dict(
        case_id='codemerge-adapted-v1',prompt=PROMPT,formatted_prompt=formatted,
        prompt_sha256=hashlib.sha256(formatted.encode()).hexdigest(),seed=0,
        checkpoint_sha256=identity,source_sha256=source_identity(),
        beta_site='SC embedding before next-step self_conditioner.pre_norm',
        first_canvas_only=True,cap=48,adaptive=True,
        device_kind=[d.device_kind for d in jax.devices()],
        modes={name:dataclasses.asdict(mode) for name,mode in modes.items()},
        prior_prompt_recovered=False,
    ))
    traces={}
    summary={}
    for name,mode in modes.items():
      start=time.perf_counter()
      trace=run_fixed_trajectory(sampler=native,params=params,rng=rng,batch_size=1,
          cache=state.cache,full_attention_mask=state.full_attention_mask,
          sc_mode=mode,stop_early=True)
      trace.save_npz(args.output/f'{name}.npz')
      traces[name]=trace
      summary[name]=dict(steps=len(trace.steps),early_stop=trace.stopped_early,
          cap_reached=not trace.stopped_early,seconds_with_trace=time.perf_counter()-start,
          mean_entropy=[float(np.mean(s.entropy)) for s in trace.steps],
          text=tokenizer.decode(trace.final_canvas[0]))
      print(name,summary[name]['steps'],'early_stop',trace.stopped_early,flush=True)
    np.testing.assert_array_equal(traces['beta=1'].final_canvas,traces['full'].final_canvas)
    np.testing.assert_array_equal(traces['beta=0'].final_canvas,traces['disabled'].final_canvas)
    plot({n:t for n,t in traces.items() if n.startswith('beta=')},args.output/'beta_entropy.png')
    plot({n:t for n,t in traces.items() if not n.startswith('beta=')},args.output/'bottleneck_entropy.png')
    write_json(args.output/'summary.json',summary)


if __name__=='__main__':
  main()
