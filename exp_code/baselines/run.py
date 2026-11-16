"""Prepare and run matched SC baselines; refuse mixed configurations on resume."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

from exp_code.baselines.data import digest, prepare
from exp_code.baselines.scoring import score_gsm8k, score_mbpp

ROOT = Path(__file__).resolve().parents[2]
MODES = ('full', 'disabled', 'top_k', 'threshold')


def write_json(path, value):
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
  temporary.replace(path)


@contextmanager
def run_lock(output):
  lock = output / '.run.lock'
  with lock.open('x') as stream:
    stream.write(str(os.getpid()))
  try:
    yield
  finally:
    lock.unlink()


def source_identity():
  paths = [p for directory in ('exp_code', 'dg/gemma')
           for p in (ROOT / directory).rglob('*.py')]
  return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(paths)})


def checkpoint_identity(path):
  manifest = json.loads((path / 'verified-manifest.json').read_text())
  if digest(manifest['files']) != manifest['sha256']:
    raise ValueError('Checkpoint manifest digest mismatch')
  # The downloader computed file hashes; here ensure no file went missing.
  prefix = 'checkpoints/diffusiongemma-26B-A4B-it/'
  for item in manifest['files']:
    name = item['name']
    relative = name[len(prefix):] if name.startswith(prefix) else 'tokenizer.model'
    file = path / relative
    if not file.resolve().is_relative_to(path.resolve()):
      raise ValueError('Invalid checkpoint manifest path')
    if not file.is_file() or file.stat().st_size != item['size']:
      raise ValueError(f'Checkpoint file missing or wrong size: {relative}')
  return manifest['sha256']


def score(task, text, case):
  if task == 'gsm8k':
    return score_gsm8k(text, case['reference'])
  return score_mbpp(text, case['tests'], case['test_imports'])


def summarize(output):
  metadata = json.loads((output / 'run.json').read_text())
  plans = metadata['plans']
  groups = defaultdict(list)
  for line in (output / 'predictions.jsonl').read_text(encoding='utf-8').splitlines():
    row = json.loads(line)
    if row['run_sha256'] != metadata['sha256']:
      raise ValueError('Foreign prediction in result file')
    groups[(row['task'], row['mode'])].append(row)
  summary = {}
  for task in plans:
    expected = {c['id'] for c in plans[task]['cases']}
    for mode in MODES:
      rows = groups[(task, mode)]
      ids = [r['case_id'] for r in rows]
      if len(set(ids)) != len(ids) or not set(ids) <= expected:
        raise ValueError('Duplicate or foreign case')
      summary[f'{task}/{mode}'] = dict(
          completed=len(rows), expected=len(expected), complete=set(ids)==expected,
          accuracy=sum(r['score']['passed'] for r in rows)/len(rows) if rows else None,
          generation_seconds=sum(r['generation_seconds'] for r in rows),
          metric='exact_match' if task=='gsm8k' else 'pass@1_single_sample',
      )
    baseline = {r['case_id']: int(r['score']['passed']) for r in groups[(task,'full')]}
    for mode in MODES[1:]:
      variant = {r['case_id']: int(r['score']['passed']) for r in groups[(task,mode)]}
      shared = baseline.keys() & variant.keys()
      differences = [variant[i]-baseline[i] for i in sorted(shared)]
      if differences:
        import numpy as np
        sample = np.asarray(differences, dtype=float)
        boot = np.random.default_rng(4953).choice(sample, size=(2000,len(sample))).mean(axis=1)
        interval = np.quantile(boot,[0.025,0.975]).tolist()
      else:
        interval = None
      summary[f'{task}/{mode}']['paired_vs_full'] = dict(
          n=len(shared), delta_accuracy=sum(differences)/len(shared) if shared else None,
          improved=sum(d==1 for d in differences), regressed=sum(d==-1 for d in differences),
          complete=len(shared)==len(expected),
          bootstrap_95_interval=interval,
      )
  write_json(output / 'summary.json', summary)
  return summary


def run(args, plans):
  import jax
  import numpy as np
  from gemma import gm
  from gemma import diffusion
  from exp_code.common.sc_modes import SelfConditioningMode
  from exp_code.common.sc_sampler import SCChatSampler

  if not any(device.platform == 'gpu' for device in jax.devices()):
    raise RuntimeError('Capability evaluation requires a GPU allocation')
  checkpoint_sha = checkpoint_identity(args.checkpoint)
  metadata = dict(
      protocol='sc-bottleneck-v1', plans=plans, source_sha256=source_identity(),
      checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
      seed=args.seed, max_new_tokens=args.max_new_tokens,
      cache_length=4096, canvas_length=256, max_denoising_steps=48,
      early_stopping='upstream_default_adaptive', temperature=[0.8,0.4],
      top_k=8, threshold=0.03, tail='uniform_mass_preserving',
      devices=[str(d) for d in jax.devices()],
      device_kind=[d.device_kind for d in jax.devices()],
      slurm_job=os.environ.get('SLURM_JOB_ID'),
      hostname=os.uname().nodename,
      versions={n:importlib.metadata.version(n) for n in ('jax','flax','gemma','datasets')},
  )
  # Hardware, job ID and hostname are provenance, not resume compatibility keys.
  compatible = {k:v for k,v in metadata.items() if k not in ('devices','slurm_job','hostname')}
  metadata['sha256'] = digest(compatible)
  run_file = args.output / 'run.json'
  if run_file.exists():
    previous = json.loads(run_file.read_text())
    if previous['sha256'] != metadata['sha256']:
      raise ValueError('Run configuration/source/checkpoint changed; use a new output directory')
  else:
    metadata['created_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(run_file, metadata)
  log_file = args.output / 'predictions.jsonl'
  done = set()
  if log_file.exists():
    summarize(args.output)  # Detect duplicates/foreign records BEFORE generating.
    for line in log_file.read_text(encoding='utf-8').splitlines():
      row = json.loads(line)
      done.add((row['task'], row['mode'], row['case_id']))
  # Verify grader/reference compatibility before spending GPU time.
  for case in plans['mbpp']['cases']:
    reference_score = score_mbpp(case['reference'], case['tests'], case['test_imports'])
    if not reference_score['passed']:
      raise ValueError(f'MBPP reference failed preflight: {case["id"]}: {reference_score}')
  model = diffusion.DiffusionGemma_26B_A4B()
  params = gm.ckpts.load_params(str(args.checkpoint), text_only=True)
  tokenizer = gm.text.Gemma4Tokenizer(path=str(args.checkpoint / 'tokenizer.model'))
  common = dict(model=model, params=params, tokenizer=tokenizer,
                max_out_length=args.max_new_tokens, cache_length=4096,
                canvas_length=256, max_denoising_steps=48,
                multi_turn=False, print_stream=False)
  # Checkpoint-backed gate; no capability rows are written unless it passes.
  prompt = plans['gsm8k']['cases'][0]['prompt']
  native = diffusion.ChatSampler(**common)
  full = SCChatSampler(**common)
  native_text = native.chat(prompt, rng=args.seed, max_new_tokens=args.max_new_tokens)
  full_text = full.chat(prompt, rng=args.seed, max_new_tokens=args.max_new_tokens)
  np.testing.assert_array_equal(native.last_state.predicted_tokens, full.last_state.predicted_tokens)
  if native_text != full_text:
    raise AssertionError('Full SC differs from original upstream chat generation')
  write_json(args.output / 'full-sc-gate.json', dict(passed=True,
             run_sha256=metadata['sha256'], case_id=plans['gsm8k']['cases'][0]['id'],
             output_sha256=hashlib.sha256(full_text.encode()).hexdigest()))
  del native
  for mode in MODES:
    chat = full if mode=='full' else SCChatSampler(**common, sc_mode=SelfConditioningMode(kind=mode))
    # Separate compilation/warmup from per-example timings. These are end-to-end
    # latencies, not an uninstrumented throughput benchmark.
    warmup_start = time.perf_counter()
    chat.chat(prompt, rng=args.seed, max_new_tokens=args.max_new_tokens)
    jax.block_until_ready(chat.last_state.predicted_tokens)
    print('WARMUP', mode, time.perf_counter()-warmup_start, flush=True)
    for task, plan in plans.items():
      for case in plan['cases']:
        key = task, mode, case['id']
        if key in done:
          continue
        start = time.perf_counter()
        text = chat.chat(case['prompt'], rng=args.seed, max_new_tokens=args.max_new_tokens)
        jax.block_until_ready(chat.last_state.predicted_tokens)
        elapsed = time.perf_counter()-start
        # Execute MBPP only after timing generation.
        result = dict(run_sha256=metadata['sha256'], task=task, mode=mode,
                      case_id=case['id'], seed=args.seed,
                      prompt_sha256=hashlib.sha256(case['prompt'].encode()).hexdigest(),
                      text=text, score=score(task,text,case), generation_seconds=elapsed,
                      slurm_job=os.environ.get('SLURM_JOB_ID'), devices=metadata['devices'])
        with log_file.open('a',encoding='utf-8') as stream:
          stream.write(json.dumps(result,ensure_ascii=False)+'\n')
          stream.flush()
          os.fsync(stream.fileno())
        done.add(key)
        print(task,mode,case['id'],result['score'],flush=True)
    summarize(args.output)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-root', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--stage', choices=('smoke','pilot','subset','full'), default='pilot')
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--prepare-only', action='store_true')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--max-new-tokens', type=int, default=1024)
  args = parser.parse_args()
  if args.max_new_tokens <= 0 or args.max_new_tokens > 2048:
    parser.error('max-new-tokens must be in 1..2048 for the 4096-token cache')
  if not args.prepare_only and args.checkpoint is None:
    parser.error('--checkpoint is required for generation')
  args.output.mkdir(parents=True,exist_ok=True)
  with run_lock(args.output):
    plans = {task:prepare(task, args.data_root/path, args.stage)
             for task,path in [('gsm8k',Path('gsm8k/main')),('mbpp',Path('mbpp/sanitized'))]}
    plan_file=args.output/'samples.json'
    if plan_file.exists() and json.loads(plan_file.read_text()) != plans:
      raise ValueError('Sample manifest changed; use a new output directory')
    write_json(plan_file,plans)
    if args.prepare_only:
      print('PREPARED_ONLY', {t:len(p['cases']) for t,p in plans.items()}, flush=True)
      return
    run(args, plans)


if __name__ == '__main__':
  main()
