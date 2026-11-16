"""256 held-out cases per task, five conditions, batched with entropy traces."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from exp_code.baselines.data import digest,prepare
from exp_code.baselines.entropy_recorder import EntropyRecorder
from exp_code.baselines.run import checkpoint_identity,run_lock,score,source_identity,summarize,write_json
from exp_code.baselines.scoring import score_mbpp

CONDITIONS={
    'full':dict(kind='full',steps=48),
    'disabled_48':dict(kind='disabled',steps=48),
    'disabled_96':dict(kind='disabled',steps=96),
    'top_k':dict(kind='top_k',steps=48),
    'threshold':dict(kind='threshold',steps=48),
}


def batches(cases,size):
  for start in range(0,len(cases),size):
    yield start//size,cases[start:start+size]


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-root',type=Path,required=True)
  parser.add_argument('--checkpoint',type=Path)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--batch-size',type=int,choices=(1,2,4,8),default=4)
  parser.add_argument('--seed',type=int,default=0)
  parser.add_argument('--max-new-tokens',type=int,default=1024)
  parser.add_argument('--prepare-only',action='store_true')
  args=parser.parse_args()
  if args.max_new_tokens not in (256,512,768,1024):
    parser.error('max-new-tokens must be 256/512/768/1024')
  if not args.prepare_only and args.checkpoint is None:
    parser.error('--checkpoint is required')
  args.output.mkdir(parents=True,exist_ok=True)
  with run_lock(args.output):
    plans={task:prepare(task,args.data_root/path,'test256') for task,path in
           [('gsm8k',Path('gsm8k/main')),('mbpp',Path('mbpp/sanitized'))]}
    manifest=args.output/'samples.json'
    if manifest.exists() and json.loads(manifest.read_text())!=plans:
      raise ValueError('Changed fixed samples; choose another output directory')
    write_json(manifest,plans)
    if args.prepare_only:
      for case in plans['mbpp']['cases']:
        if not score_mbpp(case['reference'],case['tests'],case['test_imports'])['passed']:
          raise ValueError('Reference solution fails: '+case['id'])
      print('TEST256_PREPARED_AND_REFERENCES_VERIFIED',flush=True)
      return
    evaluate(args,plans)


def evaluate(args,plans):
  import dialog
  import jax
  import numpy as np
  from gemma import gm,diffusion
  from exp_code.common.sc_modes import SelfConditioningMode
  from exp_code.common.sc_sampler import SCSampler

  devices=jax.devices()
  if not any(d.platform=='gpu' for d in devices):
    raise RuntimeError('A GPU allocation is required')
  metadata=dict(protocol='test256-batched-entropy-v2',plans=plans,conditions=CONDITIONS,
      source_sha256=source_identity(),checkpoint_sha256=checkpoint_identity(args.checkpoint),
      checkpoint=str(args.checkpoint),batch_size=args.batch_size,seed=args.seed,
      rng_policy='seed + task_index*100000 + fixed_batch_index; identical across conditions',
      max_new_tokens=args.max_new_tokens,canvas_length=256,cache_length=4096,
      stopping='upstream adaptive, per-canvas caps 48 or 96',temperature=[.8,.4],
      sparse_tail='uniform mass preserving',top_k=8,threshold=.03,
      device_kind=[d.device_kind for d in devices],
      versions={n:importlib.metadata.version(n) for n in ('jax','flax','gemma','datasets')},
      entropy='all generated canvases, per token/step, ignore frozen members after stop',
      timing='instrumented batch latency; per-example generation_seconds is equal share')
  metadata['sha256']=digest(metadata)
  run_file=args.output/'run.json'
  if run_file.exists():
    if json.loads(run_file.read_text())['sha256']!=metadata['sha256']:
      raise ValueError('Changed run configuration/source; choose another output directory')
  else:
    write_json(run_file,dict(metadata,created_utc=datetime.now(timezone.utc).isoformat()))
  predictions=args.output/'predictions.jsonl'
  done=set()
  if predictions.exists():
    summarize(args.output)
    for line in predictions.read_text(encoding='utf-8').splitlines():
      row=json.loads(line)
      if not (args.output/row['entropy_file']).is_file():
        raise ValueError('Missing entropy for recorded prediction')
      done.add((row['task'],row['mode'],row['case_id']))
  print('CHECKPOINT_LOAD_START',flush=True)
  started=time.perf_counter()
  model=diffusion.DiffusionGemma_26B_A4B()
  params=gm.ckpts.load_params(str(args.checkpoint),text_only=True)
  tokenizer=gm.text.Gemma4Tokenizer(path=str(args.checkpoint/'tokenizer.model'))
  print('CHECKPOINT_LOADED',time.perf_counter()-started,[str(d) for d in devices],flush=True)
  common=dict(model=model,params=params,tokenizer=tokenizer,
              cache_length=4096,max_out_length=args.max_new_tokens,canvas_length=256)
  def prompts(cases):
    return [dialog.Conversation(dialog.User(c['prompt'])).as_text(format=tokenizer.FORMAT)
            for c in cases]
  gate_cases=plans['gsm8k']['cases'][:args.batch_size]
  gate_prompts=prompts(gate_cases)
  print('NATIVE_BATCH_GATE_START',args.batch_size,flush=True)
  native=diffusion.Sampler(**common,max_denoising_steps=48)
  expected=native.sample(gate_prompts,rng=args.seed,max_new_tokens=args.max_new_tokens,return_state=True)
  full=SCSampler(**common,max_denoising_steps=48)
  actual=full.sample(gate_prompts,rng=args.seed,max_new_tokens=args.max_new_tokens,return_state=True)
  np.testing.assert_array_equal(expected.tokens,actual.tokens)
  if expected.text!=actual.text:
    raise AssertionError('Full SC/native batch text differs')
  gate_tokens=np.asarray(expected.tokens)
  write_json(args.output/'full-sc-gate.json',dict(passed=True,batch_size=args.batch_size,
      run_sha256=metadata['sha256'],case_ids=[c['id'] for c in gate_cases]))
  del expected,actual,native,full
  for name,condition in CONDITIONS.items():
    recorder=EntropyRecorder(condition['steps'])
    sampler=SCSampler(**common,max_denoising_steps=condition['steps'],
        sc_mode=SelfConditioningMode(kind=condition['kind']),entropy_callback=recorder.record)
    end_tokens=sampler._initialize_sampler_loop(sampler.sampling).end_tokens
    print('CONDITION_START',name,condition,'batch_size',args.batch_size,flush=True)
    for task_index,(task,plan) in enumerate(plans.items()):
      for batch_index,cases in batches(plan['cases'],args.batch_size):
        if all((task,name,c['id']) in done for c in cases):
          continue
        recorder.reset()
        batch_seed=args.seed+task_index*100000+batch_index
        start=time.perf_counter()
        output=sampler.sample(prompts(cases),rng=batch_seed,
            max_new_tokens=args.max_new_tokens,return_state=True)
        jax.block_until_ready(output.tokens)
        jax.effects_barrier()
        elapsed=time.perf_counter()-start
        tokens=np.asarray(output.tokens)
        if name=='full' and task=='gsm8k' and batch_index==0:
          np.testing.assert_array_equal(gate_tokens,tokens)
          write_json(args.output/'full-sc-gate.json',dict(passed=True,
              entropy_logging_equivalent=True,batch_size=args.batch_size,
              run_sha256=metadata['sha256'],case_ids=[c['id'] for c in gate_cases]))
        if len(output.text)!=len(cases) or not recorder.canvases:
          raise AssertionError('Missing batched output or entropy callbacks')
        for i,(case,text) in enumerate(zip(cases,output.text)):
          key=task,name,case['id']
          if key in done:
            continue
          relative=Path('entropy')/name/task/f'{case["id"].replace(":","-")}.npz'
          trace_info=recorder.save_sample(args.output/relative,i,tokens[i],end_tokens)
          row=dict(run_sha256=metadata['sha256'],task=task,mode=name,case_id=case['id'],
              seed=batch_seed,batch_index=batch_index,batch_position=i,batch_size=len(cases),
              prompt_sha256=hashlib.sha256(case['prompt'].encode()).hexdigest(),
              text=text,score=score(task,text,case),generation_seconds=elapsed/len(cases),
              batch_seconds=elapsed,entropy_file=relative.as_posix(),**trace_info,
              slurm_job=os.environ.get('SLURM_JOB_ID'),device_kind=metadata['device_kind'])
          with predictions.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
          done.add(key)
        print('BATCH_COMPLETE',name,task,batch_index,'n',len(cases),'seconds',elapsed,flush=True)
        summarize(args.output)
    print('CONDITION_COMPLETE',name,flush=True)
    del sampler
  final=summarize(args.output)
  if not all(v['complete'] for v in final.values()):
    raise AssertionError('Incomplete benchmark')
  print('BENCHMARK_COMPLETE',len(done),'predictions',flush=True)


if __name__=='__main__':
  main()
