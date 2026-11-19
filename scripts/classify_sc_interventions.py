"""Classify mathematical edit operators separately from experimental settings.

Read existing condition summaries only. Do not alter raw generations, count
positions as methods, or count a joint edit as a fifth primitive operator.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


OPERATORS = {
    'sc_swap': 'O1_candidate_logit_swap',
    'sc_distribution_mix': 'O2_SC_distribution_edit',
    'sc_embedding_zero': 'O3_SC_ablation',
    'canvas_token': 'O4_canvas_substitution',
}


def geometry(positions):
    p = sorted(set(positions))
    if not p:
        return 'none'
    if len(p) == 1:
        return 'single_site'
    if p == list(range(256)):
        return 'full_canvas'
    if len(p) == 255:
        return 'all_but_one_site'
    if p == list(range(p[0], p[-1]+1)):
        return 'contiguous_region'
    return 'multiple_separated_sites'


def main(source, output):
    with source.open(encoding='utf-8-sig', newline='') as stream:
        conditions = list(csv.DictReader(stream))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    coverage = Counter()
    channels = Counter()
    baseline = incomplete = completed = 0
    for condition in conditions:
        edits = json.loads(condition['operations_json'])
        unknown = {e['kind'] for e in edits} - set(OPERATORS)
        if unknown:
            raise ValueError(f'Unclassified operators: {unknown}')
        families = sorted({OPERATORS[e['kind']] for e in edits})
        sc = [e for e in edits if e['kind'] != 'canvas_token']
        canvas = [e for e in edits if e['kind'] == 'canvas_token']
        channel = 'joint' if sc and canvas else 'SC_only' if sc else 'canvas_only' if canvas else 'baseline'
        done = condition['focus_adaptive_stop'] == 'True' or condition['forced_cap'] == 'True'
        if not edits:
            baseline += int(done)
        elif done:
            completed += 1
            coverage.update(families)
            channels[channel] += 1
        else:
            incomplete += 1
        alphas = sorted({e['alpha'] for e in sc if 'alpha' in e})
        subtypes = []
        if any(e.get('alpha') == 1 for e in sc):
            subtypes.append('distribution_replacement')
        if any(0 < e.get('alpha', -1) < 1 for e in sc):
            subtypes.append('distribution_interpolation')
        times = sorted({e['after_step_number'] for e in edits})
        rows.append(dict(
            condition_id=condition['condition_sha256'][:12],
            representative=condition['representative'],
            status='stopped' if done else 'partial_prefix',
            operator_families=';'.join(families),
            distribution_subtypes=';'.join(subtypes),
            channel_design=channel,
            SC_scope=geometry([e['position'] for e in sc]),
            canvas_scope=geometry([e['position'] for e in canvas]),
            SC_positions=json.dumps(sorted({e['position'] for e in sc})),
            canvas_positions=json.dumps(sorted({e['position'] for e in canvas})),
            zeroed_positions=json.dumps(sorted({e['position'] for e in sc if e['kind']=='sc_embedding_zero'})),
            edit_after_steps=json.dumps(times),
            temporal_design='repeated_pulse' if len(times)>1 else 'single_pulse' if times else 'baseline',
            mixture_alphas=json.dumps(alphas),
            method_count_is_not_condition_count=True))
    with (output/'condition_method_map.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = dict(
        study='MBPP277 fixed prompt, original batch4, seed100008',
        input_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        raw_archive_sha256='1820393c9b8bc04f46fe274c1c0eb3db193c59e3ba0a68d69a2f46e07fe1efcc',
        primitive_operator_families=len(OPERATORS),
        completed_perturbation_configurations=completed,
        completed_baselines=baseline,
        partial_prefix_sequences=incomplete,
        unique_operation_sequences=len(rows),
        completed_configurations_by_channel=dict(channels),
        primitive_coverage_counts_overlapping=dict(coverage),
        note='Coverage counts overlap for combined operators. Positions, timing, strength, scope and duration are factors, not new methods.')
    assert len(rows) == completed+baseline+incomplete
    (output/'classification_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    main(a.source, a.output)
