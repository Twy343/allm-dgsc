"""Load once, measure batches 4/8/16, then run all five matched conditions."""
import argparse
import gc
from pathlib import Path
import time

from exp_code.baselines.batched_run import batches,evaluate
from exp_code.baselines.data import prepare
from exp_code.baselines.entropy_recorder import EntropyRecorder
from exp_code.baselines.run import checkpoint_identity,run_lock,write_json


def main():
  import dialog
  import jax
  import numpy as np
  from exp_code.baselines.model_loading import load_model
  from exp_code.common.sc_sampler import SCSampler

  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-root',type=Path,required=True)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  args.output.mkdir(parents=True,exist_ok=True)
  with run_lock(args.output):
    checkpoint_sha=checkpoint_identity(args.checkpoint)
    plans={task:prepare(task,args.data_root/path,'test256') for task,path in
           [('gsm8k',Path('gsm8k/main')),('mbpp',Path('mbpp/sanitized'))]}
    write_json(args.output/'samples.json',plans)
    print('SINGLE_WEIGHT_LOAD_START',flush=True)
    loaded=load_model(args.checkpoint,1)
    model,params,tokenizer,sampling_sharding=loaded
    profile_cases=plans['gsm8k']['cases'][:16]
    profile={}
    for size in (4,8,16):
      recorder=EntropyRecorder(48)
      sampler=SCSampler(model=model,params=params,tokenizer=tokenizer,
          cache_length=4096,max_out_length=1024,canvas_length=256,
          max_denoising_steps=48,entropy_callback=recorder.record)
      def generate(cases,seed):
        recorder.reset()
        prompts=[dialog.Conversation(dialog.User(c['prompt'])).as_text(format=tokenizer.FORMAT)
                 for c in cases]
        start=time.perf_counter()
        output=sampler.sample(prompts,rng=seed,max_new_tokens=1024,
                              return_state=True,sharding=sampling_sharding)
        jax.block_until_ready(output.tokens)
        jax.effects_barrier()
        seconds=time.perf_counter()-start
        count=int(np.count_nonzero(np.asarray(output.tokens)!=int(tokenizer.special_tokens.PAD)))
        return seconds,count
      print('PROFILE_BATCH_START',size,flush=True)
      try:
        warm,_=generate(profile_cases[:size],0)
        seconds,tokens=0.,0
        for index,cases in batches(profile_cases,size):
          elapsed,count=generate(cases,index)
          seconds+=elapsed
          tokens+=count
        profile[str(size)]=dict(status='ok',warmup_seconds=warm,
            measured_examples=16,measured_seconds=seconds,examples_per_second=16/seconds,
            output_nonpad_tokens=tokens,tokens_per_second=tokens/seconds,
            memory=[d.memory_stats() for d in jax.devices()])
        print('PROFILE_BATCH_RESULT',size,profile[str(size)],flush=True)
      except Exception as error:
        message=str(error)
        if not any(s in message.lower() for s in ('out of memory','resource_exhausted')):
          raise
        profile[str(size)]=dict(status='oom',error=message[:2000])
        print('PROFILE_BATCH_OOM',size,flush=True)
        break
      finally:
        del sampler,recorder
        gc.collect()
        write_json(args.output/'batch_profile.json',dict(
            checkpoint_sha256=checkpoint_sha,case_ids=[c['id'] for c in profile_cases],
            profiling='Full SC with entropy, same 16 prompts; warmup excluded',
            note='Stochastic batching can change output lengths; these are tuning measurements',
            batches=profile))
    candidates=[int(b) for b,r in profile.items() if r['status']=='ok']
    if not candidates:
      raise RuntimeError('No tested batch size fits')
    selected=max(candidates,key=lambda b:profile[str(b)]['examples_per_second'])
    write_json(args.output/'selected_batch.json',dict(batch_size=selected,profile=profile))
    print('SELECTED_BATCH',selected,'REUSING_SAME_WEIGHTS_FOR_ALL_CONDITIONS',flush=True)
    run_args=argparse.Namespace(output=args.output,checkpoint=args.checkpoint,
        batch_size=selected,model_shards=1,seed=0,max_new_tokens=1024,
        max_batches=None,max_seconds=None)
    evaluate(run_args,plans,loaded_model=loaded)


if __name__=='__main__':
  main()
