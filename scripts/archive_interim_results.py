"""Build a small, checksummed midterm evidence archive from saved exports.

No inference, scoring changes, or modifications to the original exports.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


SELECTION = {
    'mbpp277_early_baseline_h5': 'baseline',
    'mbpp277_swapcontent_accepted4_swap_return_v_h5': 'O1_single_site_repair',
    'mbpp277_swapcontent_accepted4_swap_all3_v_h5': 'O1_coherent_rename',
    'mbpp277_early_swap_h5': 'O1_operator_repair',
    'mbpp277_rename_early_coherent_sc_v_h5': 'O2_unedited_site_propagation',
    'mbpp277_rename_early_single_sc_v_h5': 'O2_single_site_control',
    'mbpp277_rename_follow_all3_original_h5': 'O2_original_content_control',
    'mbpp277_rename_accepted4_coherent_sc_v_h3': 'O2_alpha_100',
    'mbpp277_rename_accepted4_region_kv_h3': 'O2_return_region_template',
    'mbpp277_stopedge_a065_h5': 'O2_alpha_065',
    'mbpp277_stopedge_a070_h5': 'O2_alpha_070_stop_failure',
    'mbpp277_control_accepted4_v_a075_h5': 'O2_alpha_075_stop_failure',
    'mbpp277_stopedge_a080_h5': 'O2_alpha_080_recovery',
    'mbpp277_early_zero_h5': 'O3_local_zero',
    'mbpp277_isolation_all_zero_finished': 'O3_global_zero_pulse',
    'mbpp277_isolation_only_original_finished': 'O3_target_only_preserved',
    'mbpp277_collision_accepted4_canvas_h5': 'O4_canvas_only_repair',
    'mbpp277_alias_accepted4_n_shadow_unsafe_sc_h5': 'O2_SC_only_binding_control',
    'mbpp277_alias_accepted4_n_shadow_unsafe_joint_h5': 'O2_O4_wrong_binding_retained',
    'mbpp277_collision_accepted4_escape_sc_h5': 'O2_O4_consistent_binding_repair',
    'mbpp277_conflict_sc_v_canvas_n_h5': 'O2_O4_conflicting_channels_A',
    'mbpp277_conflict_sc_n_canvas_v_h5': 'O2_O4_conflicting_channels_B',
    'mbpp277_lateoperator_le_canvas_h5': 'O4_operator_control',
    'mbpp277_lateoperator_le_sc_h5': 'O2_operator_control',
    'mbpp277_lateoperator_le_joint_h5': 'O2_O4_wrong_le_retained',
    'mbpp277_lateoperator_gt_joint_h5': 'O2_O4_gt_accepted_then_repaired',
    'mbpp277_lateoperator_eq_joint_h5': 'O2_O4_wrong_eq_retained',
    'mbpp277_stopdelay_v_a075_h5': 'comment_control_noise_recovery',
    'mbpp277_stopdelay_n_joint_h5': 'comment_control_wrong_binding_persists',
    'mbpp277_operatorconfirm_le_joint_comment_h5': 'comment_control_le_recovery',
    'mbpp277_operatorconfirm_eq_joint_comment_h5': 'comment_control_eq_persists',
}
ROI = {14, 35, 63, *range(42, 61)}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    for folder in ['tables', 'figures', 'provenance', 'evidence']:
        (output / folder).mkdir(exist_ok=True)
    review = source / 'review'
    for name in ['filter_conditions.csv', 'case277_interventions.csv',
                 'case277_operations.csv', 'late_rename_dose.csv',
                 'operator_trajectories.csv', 'rename_propagation.csv']:
        shutil.copyfile(review / name, output / 'tables' / name)
    for stem in ['operator_recovery', 'rename_propagation', 'late_rename_dose']:
        for suffix in ['png', 'pdf']:
            shutil.copyfile(review / f'{stem}.{suffix}', output / 'figures' / f'{stem}.{suffix}')
    for name in ['trace_validation.json', 'export_audit.json', 'exact_replay_checks.json',
                 'filter_study_summary.json', 'environment_versions.json',
                 'checkpoint_verified_manifest.json', 'source-files.sha256',
                 'execution-source.tar.gz', 'stopdelay_matching.json']:
        shutil.copyfile(review / name, output / 'provenance' / name)
    shutil.copyfile(source / 'study_manifest.json', output / 'provenance' / 'study_manifest.json')
    conditions = read_csv(review / 'filter_conditions.csv')
    index = {label: row for row in conditions for label in json.loads(row['export_labels'])}
    selected = []
    for label, role in SELECTION.items():
        row = index[label]
        branch_dir = source / label
        dest = output / 'evidence' / label
        dest.mkdir(exist_ok=True)
        branch = json.loads((branch_dir / 'branch.json').read_text(encoding='utf-8'))
        shutil.copyfile(branch_dir / 'branch.json', dest / 'branch.json')
        shutil.copyfile(branch_dir / branch['steps_csv'], dest / 'steps.csv')
        token_rows = [r for r in read_csv(branch_dir / branch['tokens_csv'])
                      if int(r['canvas_position']) in ROI]
        write_csv(dest / 'tokens_selected_positions.csv', token_rows)
        selected.append(dict(branch=label, role=role,
                             condition_sha256=row['condition_sha256'],
                             after_step=row['after_first_edit'],
                             stopped=row['focus_adaptive_stop'],
                             actual_total_steps=branch['actual_total_steps'],
                             function_tests=f"{row['tests_passed']}/{row['tests_total']}",
                             operations_json=row['operations_json']))
    write_csv(output / 'selected_evidence.csv', selected)
    provenance = dict(
        archived_date='2026-11-19', raw_session=source.name,
        raw_archive='sc-debug-session-14004156-phase1.tar.gz',
        raw_archive_sha256='1820393c9b8bc04f46fe274c1c0eb3db193c59e3ba0a68d69a2f46e07fe1efcc',
        selected_branches=len(selected), selected_positions=sorted(ROI),
        scope='One MBPP prompt, seed 100008, original batch 4 / focus 2, canvas0 only.',
        copies='branch.json and steps.csv are byte-for-byte copies; token CSVs are explicitly filtered.',
        logits='Top16 per saved position, not full vocabulary histories.',
        source='execution-source.tar.gz contains the exact resident-session source, not an assertion that current HEAD was executed.',
        scoring='Extracted function only, three public tests. Not full-response MBPP benchmark scores.',
    )
    (output / 'provenance' / 'archive.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    files = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (output / 'SHA256SUMS').write_text(
        ''.join(f'{sha(p)}  {p.relative_to(output).as_posix()}\n' for p in files),
        encoding='utf-8', newline='\n')
    print(json.dumps({'files': len(files), 'selected_branches': len(selected),
                      'bytes': sum(p.stat().st_size for p in files)}))


def verify(output):
    from review_filter_operator import function_tests

    count = 0
    for line in (output / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        assert sha(output / relative) == expected, relative
        count += 1
    assert len(read_csv(output / 'tables' / 'filter_conditions.csv')) == 166
    for row in read_csv(output / 'selected_evidence.csv'):
        folder = output / 'evidence' / row['branch']
        branch = json.loads((folder / 'branch.json').read_text(encoding='utf-8'))
        score = function_tests(branch['decoded_canvas'])
        assert f"{score['tests_passed']}/{score['tests_total']}" == row['function_tests'], row['branch']
        steps = read_csv(folder / 'steps.csv')
        assert len(steps) == int(row['actual_total_steps']), row['branch']
        for token in read_csv(folder / 'tokens_selected_positions.csv'):
            step = steps[int(token['step_number']) - 1]
            accepted = int(token['canvas_position']) in json.loads(step['accepted_positions'])
            assert accepted == (token['accepted'] == 'True'), row['branch']
    print(f'Verified {count} checksums, 166 indexed conditions, selected step/token masks and 31 function scores.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/sc-debug-session-14004156'))
    parser.add_argument('--output', type=Path, default=Path('results/interim_sc_case_study'))
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        build(args.source, args.output)
