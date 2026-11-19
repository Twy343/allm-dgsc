"""Deduplicate horizon exports/replays into actual MBPP277 intervention conditions.

An exact repeat is a reproducibility check, not a new independent observation.
One prompt/seed/batch is studied. No population success rate is estimated.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
from review_filter_operator import function_tests
from summarize_debug_session import write_csv

FIELDS = ('kind', 'after_step_number', 'batch', 'position', 'alpha', 'q',
          'source_token_id', 'target_token_id', 'token_after')


def binding_analysis(text):
    result = dict(threshold_parameter='', loop_value='', predicate_left='',
                  predicate_right='', threshold_shadowed='', structurally_correct='')
    match = re.search(r'def dict_filter\([^\n]*\):\n'
                      r'(?:[ \t]+[^\n]*\n|[ \t]*\n)*?[ \t]+return[^\n]*', text)
    if not match:
        return result
    try:
        function = ast.parse(match.group()).body[0]
        dictionary, threshold = [arg.arg for arg in function.args.args]
        expression = next(node.value for node in function.body if isinstance(node, ast.Return))
        generator = expression.generators[0]
        key, value = [n.id for n in generator.target.elts]
        comparison = generator.ifs[0]
        lhs, rhs = comparison.left.id, comparison.comparators[0].id
        source = generator.iter
        shape = (isinstance(expression, ast.DictComp) and len(expression.generators) == 1
                 and len(generator.ifs) == 1 and len(comparison.ops) == 1
                 and isinstance(comparison.ops[0], ast.GtE)
                 and isinstance(source, ast.Call) and isinstance(source.func, ast.Attribute)
                 and source.func.attr == 'items' and source.func.value.id == dictionary
                 and not source.args and not source.keywords
                 and expression.key.id == key and expression.value.id == value
                 and lhs == value and rhs == threshold
                 and dictionary != threshold
                 and threshold not in (key, value) and key != value)
        result.update(threshold_parameter=threshold, loop_value=value,
                      predicate_left=lhs, predicate_right=rhs,
                      threshold_shadowed=threshold in (key, value), structurally_correct=shape)
    except (SyntaxError, AttributeError, IndexError, TypeError, ValueError, StopIteration):
        result['structurally_correct'] = False
    return result


def summarize(root):
    groups = {}
    exports = 0
    for file in sorted(root.rglob('branch.json')):
        meta = json.loads(file.read_text(encoding='utf-8'))
        if meta['case_id'] != 'mbpp:277':
            continue
        exports += 1
        edits = [{key: edit[key] for key in FIELDS if key in edit}
                 for edit in meta['edits'] if edit['kind'] != 'sc_region']
        # Region edits already contain their actual per-row mix operations.
        payload = dict(case_id=meta['case_id'], seed=meta['seed'], edits=edits)
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        identity = hashlib.sha256(encoded.encode()).hexdigest()
        with (file.parent / meta['steps_csv']).open(encoding='utf-8-sig', newline='') as stream:
            steps = list(csv.DictReader(stream))
        current = dict(meta=meta, file=file, steps=steps, payload=payload)
        group = groups.setdefault(identity, dict(labels=[], representative=current))
        group['labels'].append(file.parent.name)
        if len(steps) > len(group['representative']['steps']):
            group['representative'] = current
    rows = []
    for identity, group in groups.items():
        item = group['representative']
        meta, last = item['meta'], item['steps'][-1]
        tests = function_tests(meta['decoded_canvas'])
        rows.append(dict(condition_sha256=identity,
            representative=item['file'].parent.name,
            export_count=len(group['labels']), export_labels=json.dumps(group['labels']),
            after_first_edit=min((e['after_step_number'] for e in meta['edits']), default=''),
            edit_count=len(item['payload']['edits']),
            observed_focus_steps=len(item['steps']),
            focus_adaptive_stop=last['adaptive_stop_met'],
            forced_cap=last['forced_accept_at_cap'],
            final_unaccepted=256-int(last['accepted_count']),
            final_reopened=last['reopened_count'],
            mean_entropy_last=last['mean_entropy'],
            **tests, **binding_analysis(meta['decoded_canvas']),
            operations_json=json.dumps(item['payload']['edits'], sort_keys=True),
            decoded_canvas=meta['decoded_canvas']))
    review = root / 'review'
    review.mkdir(exist_ok=True)
    write_csv(review / 'filter_conditions.csv', rows)
    result = dict(case_id='mbpp:277', seed=100008, original_batch_size=4,
        export_count=exports, unique_operation_sequences=len(rows),
        independent_prompts=1, independent_seeds=1,
        adaptive_stopped_conditions=sum(r['focus_adaptive_stop'] == 'True' for r in rows),
        paused_conditions=sum(r['focus_adaptive_stop'] != 'True' and r['forced_cap'] != 'True' for r in rows),
        note='Descriptive single-case factorial study; horizons and exact replays are deduplicated. No population inference.')
    (review / 'filter_study_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    summarize(parser.parse_args().session)
