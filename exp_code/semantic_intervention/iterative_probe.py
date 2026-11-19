"""Bounded first investigation from a committed-site breakpoint, then restore.

Hypothesis is based on the earlier GSM480 pre-acceptance swap. This probes the
same site after actual acceptance. It is exploratory, not held-out evidence.
"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from exp_code.common.acceptance_trace import write_csv


def explore_committed(engine,*,focus,position,source,target,case_id,seed,tokenizer,output):
  output=Path(output)
  if output.exists():raise ValueError('Exploration output exists; choose a new name')
  output.mkdir(parents=True)
  while not engine.finished and not bool(engine.state.done[focus]):
    last=engine.recorder.events[-1] if engine.recorder.events else None
    if (last is not None and last['active'][focus]
        and bool(last['accepted'][focus,position])
        and int(engine.state.canvas[focus,position])==source):break
    engine.step()
  if engine.finished or bool(engine.state.done[focus]):
    (output/'selection_blocked.json').write_text(json.dumps(dict(
        reason='No still-active committed source position found',case_id=case_id,
        position=position,source=source),indent=2)+'\n',encoding='utf-8')
    return dict(status='selection_blocked')
  engine.save('committed')
  starting_step=engine.state.step
  controls={};rows=[];decisions=[]

  def run_branch(name,operation):
    engine.restore('committed')
    if operation is not None:operation()
    branch=[]
    for horizon in range(1,6):
      if engine.finished or bool(engine.state.done[focus]):break
      engine.step()
      event=engine.recorder.events[-1]
      logits=engine.state.sc_logits[focus,position].astype(jnp.float32)
      lp=jax.nn.log_softmax(logits);p=jnp.exp(lp)
      mask=event['accepted'][focus].copy()
      row=dict(case_id=case_id,condition=name,break_after_step=starting_step,
          horizon=horizon,step_number=engine.state.step,position=position,
          source_probability=float(p[source]),target_probability=float(p[target]),
          entropy=float(-jnp.sum(p*jnp.where(p==0,0.,lp))),
          accepted=bool(mask[position]),
          canvas_token_id=int(engine.state.canvas[focus,position]),
          canvas_token=tokenizer.decode([int(engine.state.canvas[focus,position])]),
          previously_committed_target_reopened=bool(horizon==1 and not mask[position]),
          paired_baseline_available=False,target_probability_delta='',
          acceptance_mask_changed_positions='',
          note='Local continuation only; no final-answer claim')
      if name=='baseline':controls[horizon]=(row.copy(),mask)
      elif horizon in controls:
        reference,base_mask=controls[horizon]
        row.update(paired_baseline_available=True,
            target_probability_delta=row['target_probability']-reference['target_probability'],
            acceptance_mask_changed_positions=int((mask!=base_mask).sum()))
      rows.append(row);branch.append(row)
      print('EXPLORATION_STEP',name,'+',horizon,
          'P(target)',row['target_probability'],'accepted',row['accepted'],flush=True)
    summary=engine.recorder.export(output/name,focus,case_id,seed,tokenizer)
    summary.update(edits=engine.edits,scope='canvas0, at most5 active steps after committed breakpoint',
                   focus_stopped=bool(engine.state.done[focus]))
    (output/name/'branch.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_csv(output/'observations.csv',rows)
    return branch

  try:
    run_branch('baseline',None)
    sc=run_branch('sc_only',lambda:engine.swap(position=position,source=source,target=target,batch=focus))
    run_branch('canvas_only',lambda:engine.change_canvas(position=position,token=target,batch=focus))
    def joint():
      engine.swap(position=position,source=source,target=target,batch=focus)
      engine.change_canvas(position=position,token=target,batch=focus)
    run_branch('sc_and_canvas',joint)
    local_effect=any(r['paired_baseline_available'] and
        (abs(r['target_probability_delta'])>1e-4 or r['acceptance_mask_changed_positions']>0) for r in sc)
    if not local_effect:
      decisions.append('No paired SC-swap effect above exploratory numerical threshold; test stronger q and zero-SC delivery/source control.')
      run_branch('sc_onehot_target',lambda:engine.mix(position=position,weights={target:1.},alpha=1.,batch=focus))
      run_branch('sc_zero',lambda:engine.zero_sc(position=position,batch=focus))
    else:
      decisions.append('SC swap affects local distributions/selection; inspect recovery and spatial spread before stronger or sustained perturbations.')
    report=dict(status='exploration_complete',case_id=case_id,position=position,
        breakpoint_after_step=starting_step,source=source,target=target,
        sc_local_effect=local_effect,decisions=decisions,
        limit='N=1 exploratory, <=5 active steps/branch; no hidden-feature dominance conclusion',
        controls='baseline, SC-only, canvas-only, concordant joint; fixed prompt/KV/RNG')
    if case_id=='gsm8k:480' and hasattr(engine,'reference_canvas_tokens'):
      from exp_code.semantic_intervention.region_probe import explore_region
      engine.restore('committed')
      report['region_probe']=explore_region(engine,focus=focus,case_id=case_id,seed=seed,
          tokenizer=tokenizer,output=output/'regions')
    (output/'decision.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
  finally:
    engine.restore('committed')
