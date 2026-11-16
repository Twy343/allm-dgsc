"""Ordered, compact host entropy records; never transfer vocabulary logits."""
from pathlib import Path
import numpy as np


class EntropyRecorder:
  def __init__(self, step_cap, canvas_length=256):
    self.step_cap=step_cap
    self.canvas_length=canvas_length
    self.reset()

  def reset(self):
    self.canvases=[]

  def record(self, step, entropy, stopped):
    step=int(step)
    if step==0:
      self.canvases.append([])
    if not self.canvases or step!=len(self.canvases[-1]):
      raise ValueError('Out-of-order entropy callback')
    self.canvases[-1].append((np.array(entropy,copy=True),np.array(stopped,copy=True)))

  def sample(self, index, output_tokens, end_tokens):
    tokens=np.asarray(output_tokens)
    stops=np.flatnonzero(np.isin(tokens,end_tokens))
    count=(int(stops[0])//self.canvas_length+1) if len(stops) else len(self.canvases)
    if count>len(self.canvases):
      raise ValueError('Missing canvas entropy')
    values=np.full((count,self.step_cap,self.canvas_length),np.nan,dtype=np.float32)
    lengths=[]
    criteria=[]
    for canvas,records in enumerate(self.canvases[:count]):
      hit=False
      length=0
      for step,(entropy,stopped) in enumerate(records):
        values[canvas,step]=entropy[index]
        length=step+1
        if bool(stopped[index]):
          hit=True
          break  # Ignore frozen-batch members' subsequent dummy computation.
      lengths.append(length)
      criteria.append(hit)
    return dict(entropy=values,steps=np.asarray(lengths,dtype=np.int32),
                stop_criterion_met=np.asarray(criteria,dtype=bool),
                cap_reached=np.asarray(lengths)==self.step_cap,
                output_tokens=tokens,step_cap=np.asarray(self.step_cap))

  def save_sample(self,path,index,output_tokens,end_tokens):
    payload=self.sample(index,output_tokens,end_tokens)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**payload)
    return dict(steps=payload['steps'].tolist(),
                cap_reached=payload['cap_reached'].tolist(),
                stop_criterion_met=payload['stop_criterion_met'].tolist())
