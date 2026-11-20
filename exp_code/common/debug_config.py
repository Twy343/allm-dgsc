"""Resolve original batch identity from full or compact debugger plans."""


def resolve_batch(config, case_id):
    task = case_id.split(':')[0]
    plan = config['plans'][task]
    cases = plan['cases']
    at = next((i for i, case in enumerate(cases) if case['id'] == case_id), None)
    if at is None:
        raise ValueError('Unknown case ID')
    start = (at // 4) * 4
    batch = cases[start:start + 4]
    if len(batch) != 4:
        raise ValueError('Debugger requires the complete original four-case batch')
    task_offset = plan.get('seed_offset', list(config['plans']).index(task) * 100000)
    batch_index = plan.get('batch_index_offset', 0) + start // 4
    return batch, at % 4, config['seed'] + task_offset + batch_index
