"""Freeze smoke/pilot samples and check the MBPP scorer on references."""
import argparse
from pathlib import Path
from exp_code.baselines.data import prepare
from exp_code.baselines.run import write_json
from exp_code.baselines.scoring import score_mbpp


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--data-root', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args=parser.parse_args()
  for stage in ('smoke','pilot'):
    plans={task:prepare(task,args.data_root/path,stage)
           for task,path in [('gsm8k',Path('gsm8k/main')),('mbpp',Path('mbpp/sanitized'))]}
    destination=args.output/stage
    destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'samples.json',plans)
    if stage=='pilot':
      for case in plans['mbpp']['cases']:
        result=score_mbpp(case['reference'],case['tests'],case['test_imports'])
        if not result['passed']:
          raise ValueError(f'Reference failed: {case["id"]}: {result}')
    print('INPUTS_VERIFIED',stage,{t:len(p['cases']) for t,p in plans.items()},flush=True)


if __name__=='__main__':
  main()
