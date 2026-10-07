"""Strict immutable JSON records; hashes prove content/binding, not authenticity."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping

VERSION = 'shared-foundation/0.1'
PURPOSE_TREE = {
    'qualification.broken_f2p': 'broken',
    'qualification.reference_f2p': 'reference',
    'qualification.clean_f2p': 'clean',
    'qualification.reference_p2p': 'reference',
    'qualification.clean_p2p': 'clean',
    'candidate.evaluate': 'broken',
}
STATUSES = {'PASS', 'FAIL', 'SKIP', 'ERROR', 'NOT_RUN'}
RUN_STATUSES = {'completed', 'invalid_env', 'timeout', 'inconclusive', 'not_run'}


class ContractError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def text(value, name):
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ContractError('invalid_' + name)
    return value


def digest(value, name, width=64):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{%d}' % width, value):
        raise ContractError('invalid_' + name)
    return value


def keys(value, allowed, required=()):
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ContractError('object_required')
    if set(value) - set(allowed):
        raise ContractError('unknown_fields')
    if set(required) - set(value):
        raise ContractError('missing_fields')


def canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ContractError('non_json_value') from error


def sha(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def patch_sha(content):
    return hashlib.sha256(text(content, 'patch').encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class Record:
    kind: str
    json_text: str
    sha256: str

    @classmethod
    def seal(cls, kind, payload):
        encoded = canonical(payload)
        return cls(kind, encoded, hashlib.sha256(encoded.encode('utf-8')).hexdigest())

    def data(self, kind=None):
        if kind is not None and self.kind != kind:
            raise ContractError('record_kind_mismatch')
        digest(self.sha256, 'record_hash')
        if hashlib.sha256(self.json_text.encode('utf-8')).hexdigest() != self.sha256:
            raise ContractError('record_hash_mismatch')
        payload = json.loads(self.json_text)
        if canonical(payload) != self.json_text:
            raise ContractError('record_not_canonical')
        return payload  # Fresh copy; edits cannot mutate the sealed record.


@dataclass(frozen=True)
class Snapshots:
    base_commit: str
    clean: str
    broken: str
    reference: str

    def data(self):
        digest(self.base_commit, 'base_commit', 40)
        for name in ('clean', 'broken', 'reference'):
            digest(getattr(self, name), name + '_tree')
        if self.broken == self.reference:
            raise ContractError('broken_reference_identical')
        return dict(base_commit=self.base_commit, clean=self.clean,
                    broken=self.broken, reference=self.reference)


@dataclass(frozen=True)
class Environment:
    env_id: str
    image_name: str
    image_digest: str
    lock_sha256: str

    def data(self):
        return dict(env_id=text(self.env_id, 'env_id'),
                    image_name=text(self.image_name, 'image_name'),
                    image_digest=digest(self.image_digest, 'image_digest'),
                    lock_sha256=digest(self.lock_sha256, 'env_lock'))


def ids(values, name):
    if not isinstance(values, (tuple, list)) or not values:
        raise ContractError('invalid_' + name)
    result = [text(v, name) for v in values]
    if any('\n' in v or '\r' in v for v in result) or len(set(result)) != len(result):
        raise ContractError('invalid_' + name)
    return result


def argv_groups(groups):
    if not isinstance(groups, (tuple, list)) or not groups:
        raise ContractError('argv_groups_required')
    result = []
    for argv in groups:
        if not isinstance(argv, (tuple, list)) or not argv:
            raise ContractError('argv_array_required')
        result.append([text(v, 'argv_item') for v in argv])
    return result


@dataclass(frozen=True)
class TestPlan:
    f2p_ids: tuple[str, ...]
    p2p_ids: tuple[str, ...]
    f2p_argv: tuple[tuple[str, ...], ...]
    p2p_argv: tuple[tuple[str, ...], ...]
    parser_sha256: str

    def data(self):
        f2p, p2p = ids(self.f2p_ids, 'f2p_ids'), ids(self.p2p_ids, 'p2p_ids')
        if set(f2p) & set(p2p):
            raise ContractError('test_sets_overlap')
        return dict(f2p_ids=f2p, p2p_ids=p2p,
                    f2p_argv=argv_groups(self.f2p_argv),
                    p2p_argv=argv_groups(self.p2p_argv),
                    parser_sha256=digest(self.parser_sha256, 'parser_hash'))


@dataclass(frozen=True)
class Provenance:
    dataset_revision: str
    input_kind: str = 'unapproved_external'
    approval_ref: str | None = None

    def data(self):
        digest(self.dataset_revision, 'dataset_revision', 40)
        if self.input_kind not in {'synthetic_fixture', 'unapproved_external'}:
            raise ContractError('unsupported_input_kind')
        if self.approval_ref is not None:
            text(self.approval_ref, 'approval_ref')
        return dict(dataset_revision=self.dataset_revision,
                    input_kind=self.input_kind, approval_ref=self.approval_ref)


TASK_KEYS = {'schema', 'provider', 'source_id', 'task_id', 'family_id', 'repo',
             'problem_statement', 'source_patch', 'test_fixture_sha256',
             'reference_operation', 'snapshots', 'environment', 'test_plan',
             'provenance', 'training_release'}


def task_data(task):
    payload = task.data('task')
    keys(payload, TASK_KEYS, TASK_KEYS)
    if payload['schema'] != VERSION or payload['training_release'] != 'blocked':
        raise ContractError('unsupported_task_version_or_release')
    roles = {'swe-rebench-v2': ('repair_from_broken', 'apply_repair'),
             'swe-smith-py': ('inject_into_clean', 'reverse_injection')}
    provider = payload['provider']
    if provider not in roles:
        raise ContractError('unsupported_provider')
    keys(payload['source_patch'], {'role', 'sha256'}, {'role', 'sha256'})
    role, operation = roles[provider]
    if (payload['source_patch']['role'], payload['reference_operation']) != (role, operation):
        raise ContractError('patch_direction_mismatch')
    digest(payload['source_patch']['sha256'], 'source_patch_hash')
    if payload['test_fixture_sha256'] is not None:
        digest(payload['test_fixture_sha256'], 'test_fixture_hash')
    for field in ('source_id', 'task_id', 'family_id', 'repo', 'problem_statement'):
        text(payload[field], field)
    if payload['task_id'] != provider + '/' + payload['source_id']:
        raise ContractError('task_source_identity_mismatch')
    snap = payload['snapshots']
    keys(snap, {'base_commit', 'clean', 'broken', 'reference'}, {'base_commit', 'clean', 'broken', 'reference'})
    Snapshots(**snap).data()
    env = payload['environment']
    keys(env, {'env_id', 'image_name', 'image_digest', 'lock_sha256'}, {'env_id', 'image_name', 'image_digest', 'lock_sha256'})
    Environment(**env).data()
    plan = payload['test_plan']
    plan_keys = {'f2p_ids', 'p2p_ids', 'f2p_argv', 'p2p_argv', 'parser_sha256'}
    keys(plan, plan_keys, plan_keys)
    TestPlan(**plan).data()
    prov = payload['provenance']
    keys(prov, {'dataset_revision', 'input_kind', 'approval_ref'}, {'dataset_revision', 'input_kind', 'approval_ref'})
    Provenance(**prov).data()
    return payload


def actor_view(task):
    payload = task_data(task)
    return {k: payload[k] for k in ('task_id', 'family_id', 'repo', 'problem_statement')}


def publication_guard(task):
    payload = task_data(task)
    if payload['provenance']['input_kind'] == 'synthetic_fixture':
        raise ContractError('synthetic_publication_forbidden')
    if payload['provenance']['approval_ref'] is None:
        raise ContractError('unapproved_publication_forbidden')
    # A string approval_ref is not a signed authorization. No production publisher.
    raise ContractError('release_authority_unconnected')


def prepare_request(task, purpose, *, actor_id, executor_id, candidate=None):
    payload = task_data(task)
    if purpose not in PURPOSE_TREE:
        raise ContractError('unknown_run_purpose')
    tree_name = PURPOSE_TREE[purpose]
    bound_candidate = None
    if purpose == 'candidate.evaluate':
        if candidate is None:
            raise ContractError('candidate_required')
        fields = {'role', 'task_id', 'actor_id', 'patch', 'patch_sha256', 'base_commit', 'base_tree_sha256'}
        keys(candidate, fields, fields)
        if candidate['role'] != 'candidate':
            raise ContractError('candidate_role_required')
        if candidate['task_id'] != payload['task_id'] or candidate['actor_id'] != actor_id:
            raise ContractError('candidate_identity_mismatch')
        if candidate['base_commit'] != payload['snapshots']['base_commit'] or candidate['base_tree_sha256'] != payload['snapshots']['broken']:
            raise ContractError('candidate_base_mismatch')
        if digest(candidate['patch_sha256'], 'candidate_hash') != patch_sha(candidate['patch']):
            raise ContractError('candidate_hash_mismatch')
        bound_candidate = dict(candidate)
    elif candidate is not None:
        raise ContractError('qualification_cannot_accept_candidate')
    plan = payload['test_plan']
    if purpose.endswith('_p2p'):
        selected, commands = plan['p2p_ids'], plan['p2p_argv']
    elif purpose.startswith('qualification.'):
        selected, commands = plan['f2p_ids'], plan['f2p_argv']
    else:
        selected = plan['f2p_ids'] + plan['p2p_ids']
        commands = plan['f2p_argv'] + plan['p2p_argv']
    return Record.seal('request', dict(
        schema=VERSION, task_sha256=task.sha256, task_id=payload['task_id'],
        purpose=purpose, actor_id=text(actor_id, 'actor_id'),
        executor_id=text(executor_id, 'executor_id'),
        input_tree_sha256=payload['snapshots'][tree_name],
        env_sha256=sha(payload['environment']), test_plan_sha256=sha(plan),
        selected_test_ids=selected, command_argv=commands, candidate=bound_candidate,
    ))


def request_data(task, request):
    obj = request.data('request')
    expected = prepare_request(task, obj.get('purpose'), actor_id=obj.get('actor_id'),
                               executor_id=obj.get('executor_id'), candidate=obj.get('candidate'))
    if request.sha256 != expected.sha256:
        raise ContractError('request_binding_mismatch')
    return obj


FACT_KEYS = {'schema', 'run_id', 'task_sha256', 'request_sha256', 'task_id',
             'purpose', 'actor_id', 'executor_id', 'input_tree_sha256',
             'env_sha256', 'test_plan_sha256', 'candidate_sha256', 'run_status',
             'cause', 'test_outcomes', 'stdout_sha256', 'stderr_sha256', 'duration_ms'}


def load_fact(payload):
    keys(payload, FACT_KEYS, FACT_KEYS)
    if payload['schema'] != VERSION or payload['purpose'] not in PURPOSE_TREE:
        raise ContractError('unsupported_fact_version_or_purpose')
    for field in ('task_sha256', 'request_sha256', 'input_tree_sha256', 'env_sha256',
                  'test_plan_sha256', 'stdout_sha256', 'stderr_sha256'):
        digest(payload[field], field)
    if payload['candidate_sha256'] is not None:
        digest(payload['candidate_sha256'], 'candidate_hash')
    for field in ('run_id', 'task_id', 'actor_id', 'executor_id'):
        text(payload[field], field)
    if payload['run_status'] not in RUN_STATUSES:
        raise ContractError('unknown_run_status')
    cause = payload['cause']
    if cause is not None:
        text(cause, 'cause')
    if payload['run_status'] != 'completed' and cause is None:
        raise ContractError('na_cause_required')
    duration = payload['duration_ms']
    if type(duration) is not int or duration < 0:
        raise ContractError('invalid_duration')
    if not isinstance(payload['test_outcomes'], list):
        raise ContractError('test_outcomes_array_required')
    observed = set()
    for outcome in payload['test_outcomes']:
        keys(outcome, {'test_id', 'status'}, {'test_id', 'status'})
        node = text(outcome['test_id'], 'test_id')
        if node in observed:
            raise ContractError('duplicate_test_id')
        observed.add(node)
        if outcome['status'] not in STATUSES:
            raise ContractError('unknown_test_status')
    return Record.seal('fact', payload)


def record_fact(task, request, *, run_id, run_status, test_outcomes, cause=None,
                stdout_sha256, stderr_sha256, duration_ms):
    """Package caller-supplied observations. Does NOT execute or certify a run."""
    obj = request_data(task, request)
    candidate = obj['candidate']
    return load_fact(dict(
        schema=VERSION, run_id=run_id, task_sha256=task.sha256,
        request_sha256=request.sha256, task_id=obj['task_id'], purpose=obj['purpose'],
        actor_id=obj['actor_id'], executor_id=obj['executor_id'],
        input_tree_sha256=obj['input_tree_sha256'], env_sha256=obj['env_sha256'],
        test_plan_sha256=obj['test_plan_sha256'],
        candidate_sha256=candidate['patch_sha256'] if candidate else None,
        run_status=run_status, cause=cause, test_outcomes=test_outcomes,
        stdout_sha256=stdout_sha256, stderr_sha256=stderr_sha256,
        duration_ms=duration_ms,
    ))


def validate_fact(task, request, fact):
    req = request_data(task, request)
    obj = fact.data('fact')
    load_fact(obj)
    expected = dict(task_sha256=task.sha256, request_sha256=request.sha256,
                    **{k: req[k] for k in ('task_id', 'purpose', 'actor_id', 'executor_id',
                                         'input_tree_sha256', 'env_sha256', 'test_plan_sha256')})
    expected['candidate_sha256'] = req['candidate']['patch_sha256'] if req['candidate'] else None
    if any(obj[k] != v for k, v in expected.items()):
        raise ContractError('fact_binding_mismatch')
    return obj
