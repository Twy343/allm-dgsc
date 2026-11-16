"""Plot matched first-canvas entropy from the saved benchmark traces."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.colors import Normalize
  parser=argparse.ArgumentParser()
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  metadata=json.loads((args.output/'run.json').read_text())
  modes=list(metadata['conditions'])
  rows=[json.loads(s) for s in (args.output/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
  by_key={(r['task'],r['mode'],r['case_id']):r for r in rows}
  for task,plan in metadata['plans'].items():
    case_id=plan['cases'][0]['id']
    figure,axes=plt.subplots(1,len(modes),figsize=(3.3*len(modes),4),sharey=True,layout='constrained')
    for axis,mode in zip(axes,modes):
      row=by_key[(task,mode,case_id)]
      with np.load(args.output/row['entropy_file'],allow_pickle=False) as trace:
        values=trace['entropy'][0]
        steps=int(trace['steps'][0])
        flag='cap' if trace['cap_reached'][0] else 'stop'
      image=axis.imshow(np.ma.masked_invalid(values),origin='lower',aspect='auto',
                        cmap='viridis',norm=Normalize(0,4),extent=(-.5,255.5,-.5,len(values)-.5))
      axis.set_ylim(-.5,95.5)
      axis.set_facecolor('#eeeeee')
      axis.set_title(f'{mode}\n{flag}: {steps}')
      axis.set_xlabel('Token position')
    axes[0].set_ylabel('Denoising step (zero based)')
    figure.colorbar(image,ax=list(axes),label='Entropy (nats, color clipped at 4)')
    seed=by_key[(task,modes[0],case_id)]['seed']
    figure.suptitle(f'{task} {case_id}: matched first canvas, seed {seed}')
    figure.savefig(args.output/f'{task}_entropy.png',dpi=180)
    plt.close(figure)


if __name__=='__main__':
  main()
