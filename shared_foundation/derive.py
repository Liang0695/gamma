"""Versioned signal prototypes. No batches, ACL proofs or real reward execution."""
from .records import ContractError, PURPOSE_TREE, Record, request_data, task_data, validate_fact

REWARD_POLICY = 'terminal-binary/0.1-prototype'
SFT_POLICY = 'sft-eligibility/0.1-prototype'


def _result(request, payload):
    req = request.data('request')
    if payload['run_status'] != 'completed':
        return None, payload['cause']
    statuses = {t['test_id']: t['status'] for t in payload['test_outcomes']}
    if any(node not in statuses for node in req['selected_test_ids']):
        return None, 'declared_test_not_covered'
    if any(status not in {'PASS', 'FAIL'} for status in statuses.values()):
        return None, 'non_semantic_test_status'
    if payload['cause'] is not None:
        return None, payload['cause']
    if any(status == 'FAIL' and node not in req['selected_test_ids']
           for node, status in statuses.items()):
        # No baseline evidence to decide whether this is an unrelated old failure
        # or a new regression. Never promote it to an unconditional success.
        return None, 'unselected_test_failure_requires_baseline'
    return int(all(statuses[node] == 'PASS' for node in req['selected_test_ids'])), None


def derive_reward(task, request, fact, *, policy_id=REWARD_POLICY):
    if policy_id != REWARD_POLICY:
        raise ContractError('unsupported_reward_policy')
    obj = validate_fact(task, request, fact)
    if obj['purpose'] != 'candidate.evaluate':
        raise ContractError('candidate_fact_required')
    value, cause = _result(request, obj)
    return Record.seal('reward', dict(
        policy_id=policy_id, task_sha256=task.sha256, request_sha256=request.sha256,
        fact_sha256=fact.sha256, reward=value, cause=cause,
        prototype=True, publishable=False,
    ))


def derive_sft_eligibility(task, request, fact, *, policy_id=SFT_POLICY):
    if policy_id != SFT_POLICY:
        raise ContractError('unsupported_sft_policy')
    result = derive_reward(task, request, fact).data('reward')
    return Record.seal('sft_eligibility', dict(
        policy_id=policy_id, task_sha256=task.sha256, fact_sha256=fact.sha256,
        eligible=result['reward'] == 1, cause=result['cause'],
        prototype=True, publishable=False, gemma_batch_generated=False,
    ))


def qualify(task, request_fact_pairs):
    """Check five legacy-compatible segments, two distinct observations each."""
    task_data(task)
    grouped = {p: [] for p in PURPOSE_TREE if p.startswith('qualification.')}
    seen = set()
    references = []
    for request, fact in request_fact_pairs:
        req = request_data(task, request)
        obj = validate_fact(task, request, fact)
        if req['purpose'] not in grouped:
            raise ContractError('qualification_fact_required')
        if obj['run_id'] in seen:
            raise ContractError('duplicate_run_id')
        seen.add(obj['run_id'])
        grouped[req['purpose']].append((request, obj))
        references.append(fact.sha256)
    causes = []
    for purpose, runs in grouped.items():
        if len(runs) != 2:
            causes.append(purpose + ':two_runs_required')
            continue
        signatures = []
        for request, obj in runs:
            _, cause = _result(request, obj)
            if cause:
                causes.append(purpose + ':' + cause)
            statuses = {t['test_id']: t['status'] for t in obj['test_outcomes']}
            signatures.append(tuple(sorted(statuses.items())))
            selected = request.data('request')['selected_test_ids']
            desired = 'FAIL' if purpose == 'qualification.broken_f2p' else 'PASS'
            if any(statuses.get(node) != desired for node in selected):
                causes.append(purpose + ':contrast_not_established')
        if signatures[0] != signatures[1]:
            causes.append(purpose + ':unstable_results')
    return Record.seal('qualification', dict(
        task_sha256=task.sha256, facts_sha256=sorted(references),
        eligible=not causes, causes=sorted(set(causes)),
        policy_id='qualification/0.1-prototype', prototype=True, publishable=False,
    ))
