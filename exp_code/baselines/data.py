"""Fixed, auditable samples shared across SC conditions."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def digest(value):
  return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':')).encode()).hexdigest()


def make_case(task, row, index):
  if task == 'gsm8k':
    prompt = (row['question'] + '\nSolve step by step. End with '
              '"#### <numeric answer>" on a separate line.')
    return dict(id=f'gsm8k:{index}', source_index=index, prompt=prompt,
                reference=row['answer'])
  problem = row.get('prompt', row.get('text'))
  if not problem:
    raise ValueError('MBPP record has no prompt/text')
  tests = row['test_list']
  prompt = ('Write a complete Python solution to the following task. Return '
            'the solution in a single Python code block.\n' + problem
            + '\nThe solution must pass:\n' + '\n'.join(tests))
  return dict(id=f'mbpp:{row["task_id"]}', source_index=index, prompt=prompt,
              reference=row['code'], tests=tests,
              test_imports=row.get('test_imports', []))


def prepare(task, dataset_dir, stage='pilot', selection_seed=4953):
  from datasets import load_from_disk

  dataset = load_from_disk(str(dataset_dir))
  split = 'train' if stage in ('smoke', 'pilot') else 'test'
  source = dataset[split]
  limit = {'smoke': 2, 'pilot': 64, 'subset': 256 if task == 'gsm8k' else 128,
           'full': len(source)}[stage]
  if len(source) < limit:
    raise ValueError(f'{task}/{split}: need {limit}, got {len(source)}')
  indices = list(range(len(source)))
  random.Random(selection_seed).shuffle(indices)
  indices = indices[:limit]
  cases = [make_case(task, source[i], i) for i in indices]
  if len({c['id'] for c in cases}) != len(cases):
    raise ValueError('Duplicate case IDs')
  manifest = dict(task=task, stage=stage, split=split,
                  dataset_dir=str(Path(dataset_dir).resolve()),
                  dataset_fingerprint=source._fingerprint,
                  selection_seed=selection_seed, cases=cases)
  manifest['sha256'] = digest(manifest)
  return manifest
