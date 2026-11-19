"""Align actual MBPP277 comparator spans; test the extracted function on CPU.

This is function-only evaluation of a canvas0 prefix, not a full-response MBPP
benchmark score. A narrow AST allowlist prevents executing unrelated model text.
"""
import argparse
import ast
import csv
import json
from pathlib import Path
import re

from summarize_debug_session import write_csv


PATTERN = re.compile(r'\breturn\s*\{[^{}\n]*\bif\s+[A-Za-z_]\w*\s*'
                     r'(?P<op>>=|<=|==|!=|<|>)\s*[A-Za-z_]\w*\s*\}')


def _result(status, passed, details=None):
    return dict(function_only_status=status, tests_passed=passed, tests_total=3,
                test_details=json.dumps(details) if details is not None else '')


def function_tests(text):
    match = re.search(r'def dict_filter\([^\n]*\):\n'
                      r'(?:[ \t]+[^\n]*\n|[ \t]*\n)*?[ \t]+return[^\n]*', text)
    if not match:
        return _result('unavailable', '')
    try:
        tree = ast.parse(match.group())
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Expr,
                   ast.Constant, ast.Return, ast.DictComp, ast.comprehension,
                   ast.Call, ast.Attribute, ast.Name, ast.Load, ast.Store,
                   ast.Tuple, ast.Compare, ast.GtE, ast.Lt, ast.Gt, ast.LtE,
                   ast.Eq, ast.NotEq)
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ValueError('Unsupported AST ' + type(node).__name__)
            if isinstance(node, ast.Name) and node.id.startswith('__'):
                raise ValueError('Unsupported name')
            if isinstance(node, ast.Attribute) and node.attr != 'items':
                raise ValueError('Unsupported attribute')
            if isinstance(node, ast.Call) and not (
                    isinstance(node.func, ast.Attribute) and node.func.attr == 'items'
                    and not node.args and not node.keywords):
                raise ValueError('Unsupported call')
        namespace = {'__builtins__': {}}
        exec(compile(tree, '<isolated generated filter>', 'exec'), namespace)
        example = {'Cierra Vega': 175, 'Alden Cantrell': 180,
                   'Kierra Gentry': 165, 'Pierre Cox': 190}
        results, errors, details = [], [], []
        for threshold in (170, 180, 190):
            expected = {k: v for k, v in example.items() if v >= threshold}
            try:
                actual = namespace['dict_filter'](dict(example), threshold)
                results.append(actual == expected)
                actual_repr = repr(actual)
            except Exception as error:
                results.append(False)
                errors.append(type(error).__name__ + ': ' + str(error))
                actual_repr = errors[-1]
            details.append(dict(threshold=threshold, input_repr=repr(example),
                                expected_repr=repr(expected), actual_repr=actual_repr,
                                passed=results[-1]))
        status = 'tested' if not errors else 'runtime_failure: ' + ' | '.join(errors)
        return _result(status, sum(results), details)
    except SyntaxError as error:
        return _result('syntax_failure: ' + str(error), 0)
    except (ValueError, TypeError, NameError) as error:
        return _result('unsupported: ' + str(error), '')


def main(root):
    spans, scores = [], []
    for file in sorted(root.rglob('branch.json')):
        meta = json.loads(file.read_text(encoding='utf-8'))
        if meta['case_id'] != 'mbpp:277':
            continue
        branch = str(file.parent.relative_to(root))
        scores.append(dict(branch=branch, **function_tests(meta['decoded_canvas'])))
        grouped = {}
        with (file.parent / meta['tokens_csv']).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                grouped.setdefault(int(row['step_number']), []).append(row)
        for step, rows in grouped.items():
            for source, token_key in [('visible', 'output_token'), ('candidate', 'denoiser_token')]:
                text = ''.join(row[token_key] for row in rows)
                offsets = []
                cursor = 0
                for row in rows:
                    offsets.append((cursor, cursor + len(row[token_key]), row))
                    cursor += len(row[token_key])
                for match in PATTERN.finditer(text):
                    start, end = match.span('op')
                    operator_rows = [r for a, b, r in offsets if a < end and b > start]
                    expression_rows = [r for a, b, r in offsets if a < match.end() and b > match.start()]
                    spans.append(dict(branch=branch, step_number=step, source=source,
                        operator=match.group('op'), expression=match.group(),
                        operator_positions=json.dumps([int(r['canvas_position']) for r in operator_rows]),
                        operator_accepted=all(r['accepted'] == 'True' for r in operator_rows),
                        expression_fully_accepted=all(r['accepted'] == 'True' for r in expression_rows)))
    review = root / 'review'
    review.mkdir(exist_ok=True)
    write_csv(review / 'filter_semantic_spans.csv', spans)
    write_csv(review / 'filter_function_tests.csv', scores)
    print(json.dumps(dict(exports=len(scores), aligned_spans=len(spans),
                          function_tests='Only extracted allowlisted function; not complete benchmark scoring.')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    main(parser.parse_args().session)
