"""Two explicit schema projections; raw patches never appear in actor_view."""
from .records import (VERSION, ContractError, Record, Environment, Provenance,
                      Snapshots, TestPlan, ids, keys, patch_sha, task_data, text)

REB_KEYS = {'instance_id', 'repo', 'base_commit', 'problem_statement', 'patch',
            'test_patch', 'FAIL_TO_PASS', 'PASS_TO_PASS', 'image_name', 'language',
            'license', 'created_at'}
SMITH_KEYS = {'instance_id', 'repo', 'problem_statement', 'patch', 'FAIL_TO_PASS',
              'PASS_TO_PASS', 'image_name'}


def _adapt(row, *, provider, snapshots: Snapshots, environment: Environment,
           test_plan: TestPlan, provenance: Provenance, family_id):
    """Caller must explicitly project full upstream rows into supported columns."""
    required = SMITH_KEYS if provider == 'swe-smith-py' else REB_KEYS - {'created_at'}
    keys(row, SMITH_KEYS if provider == 'swe-smith-py' else REB_KEYS, required)
    for field in required - {'FAIL_TO_PASS', 'PASS_TO_PASS'}:
        text(row[field], field)
    if provider == 'swe-rebench-v2':
        if row['language'].lower() != 'python':
            raise ContractError('first_batch_python_only')
        if row['base_commit'] != snapshots.base_commit:
            raise ContractError('source_base_mismatch')
    snap, env, plan, prov = snapshots.data(), environment.data(), test_plan.data(), provenance.data()
    if row['image_name'] != env['image_name']:
        raise ContractError('source_image_mismatch')
    if set(ids(row['FAIL_TO_PASS'], 'f2p_ids')) != set(plan['f2p_ids']) or set(ids(row['PASS_TO_PASS'], 'p2p_ids')) != set(plan['p2p_ids']):
        raise ContractError('source_test_selection_mismatch')
    if 'created_at' in row and type(row['created_at']) not in (str, int):
        raise ContractError('unsupported_source_date')
    role, operation = ('inject_into_clean', 'reverse_injection') if provider == 'swe-smith-py' else ('repair_from_broken', 'apply_repair')
    result = Record.seal('task', dict(
        schema=VERSION, provider=provider, source_id=row['instance_id'],
        task_id=provider + '/' + row['instance_id'], family_id=text(family_id, 'family_id'),
        repo=row['repo'], problem_statement=row['problem_statement'],
        source_patch={'role': role, 'sha256': patch_sha(row['patch'])},
        test_fixture_sha256=patch_sha(row['test_patch']) if provider == 'swe-rebench-v2' else None,
        reference_operation=operation, snapshots=snap, environment=env,
        test_plan=plan, provenance=prov, training_release='blocked',
    ))
    task_data(result)
    return result


def adapt_rebench(row, **context):
    return _adapt(row, provider='swe-rebench-v2', **context)


def adapt_smith(row, **context):
    return _adapt(row, provider='swe-smith-py', **context)
