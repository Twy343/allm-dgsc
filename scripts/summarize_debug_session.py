"""Offline, CPU-only review of resident debugger exports (no model required).

Exports at different horizons share a prefix and are NOT independent runs.
Probability columns are empty if the token was outside saved top16; the
top16 minimum is then an upper bound, not evidence of zero probability.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    review = root / 'review'
    review.mkdir(exist_ok=True)
    summaries, steps, targets = [], [], []
    sites = {'gsm8k:480': (104, 236778, 236812),
             'gsm8k:331': (168, 236778, 236812),
             'gsm8k:233': (146, 236770, 236778),
             'mbpp:277': (58, 6867, 655)}
    for metadata in sorted(root.rglob('branch.json')):
        folder = metadata.parent
        meta = json.loads(metadata.read_text(encoding='utf-8'))
        case = meta['case_id']
        label = str(folder.relative_to(root)).replace('\\', '/')
        arrays = next(folder.glob('*.npz'))
        with np.load(arrays) as compressed:
            data = {key: compressed[key] for key in compressed.files}
        with (folder / meta['steps_csv']).open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        edits = meta.get('edits', [])
        first_edit = min((e['after_step_number'] for e in edits), default=None)
        final = records[-1]
        # Do not use legacy stop_reasons: a paused prefix was labelled step_cap.
        actual_cap = final['forced_accept_at_cap'] == 'True'
        focus_stopped = final['adaptive_stop_met'] == 'True'
        ending = 'forced_cap' if actual_cap else 'adaptive_stop' if focus_stopped else 'paused'
        summaries.append(dict(branch=label, case_id=case, seed=meta['seed'],
            saved_steps=len(records), edit_count=len(edits), edit_after_step=first_edit,
            ending=ending, batch_partial=meta.get('partial'),
            accepted_last=final['accepted_count'], mean_entropy_last=final['mean_entropy'],
            decoded_canvas=meta.get('decoded_canvas', ''),
            independent_replicate=False))
        position, source, alternative = sites.get(case, (None, None, None))
        for index, record in enumerate(records):
            common = dict(record)
            common.update(branch=label, edit_after_step=first_edit)
            steps.append(common)
            if position is None:
                continue
            ids = data['top_ids'][index, position]
            probs = data['top_probs'][index, position]
            def prob(token):
                hits = np.flatnonzero(ids == token)
                return float(probs[hits[0]]) if len(hits) else ''
            targets.append(dict(branch=label, case_id=case,
                step_number=record['step_number'], position=position,
                edit_after_step=first_edit,
                source_id=source, alternative_id=alternative,
                p_source=prob(source), p_alternative=prob(alternative),
                absent_token_probability_upper_bound=float(probs.min()),
                predicted_id=int(data['sampled'][index, position]),
                output_id=int(data['after'][index, position]),
                accepted=bool(data['accepted'][index, position]),
                entropy=float(data['entropy'][index, position]),
                physical_position_is_not_semantic_alignment=True))
    write_csv(review / 'branches.csv', summaries)
    write_csv(review / 'steps.csv', steps)
    write_csv(review / 'target_positions.csv', targets)
    result = dict(export_count=len(summaries), cases=sorted({r['case_id'] for r in summaries}),
                  note='Horizon exports overlap. Canvas0 only; no full-answer correctness inferred.')
    (review / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    summarize(parser.parse_args().session)
