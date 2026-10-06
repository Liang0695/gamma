"""Real local execution of exact self-authored fixtures; NOT an untrusted sandbox."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from . import local_parser
from .local_fixtures import (ACTOR, EXECUTOR, FIXTURE_VERSION, NODES, file_digest,
                             files, patch, tree_digest)
from .owned_process import OwnedProcess
from .records import (ContractError, Record, keys, record_fact, request_data, sha, task_data)

EXECUTION_VERSION = 'trusted-local-execution/0.1'


def _no_links(path):
    if path.is_symlink() or (path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400):
        raise ContractError('local_path_link_forbidden')


def inside(path, root):
    root = Path(root).absolute()
    path = Path(path).absolute()
    try:
        path.relative_to(root)
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ContractError('local_path_outside_root') from error
    current = path
    while True:
        _no_links(current)
        if current == root:
            break
        current = current.parent
    return path


def _snapshot(directory, limit):
    expected = files(limit)
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != set(expected):
        raise ContractError('trusted_snapshot_files_mismatch')
    observed = {}
    for name, content in expected.items():
        path = directory / name
        _no_links(path)
        data = path.read_bytes()
        if data != content:
            raise ContractError('trusted_snapshot_content_mismatch')
        observed[name] = data
    return observed


@dataclass(frozen=True)
class LocalExecution:
    fact: Record
    capture: Record
    log_directory: Path


class LocalSyntheticRunner:
    """Explicit roots, fixed code/patch grammar, fresh workspace, actual byte logs."""
    def __init__(self, *, allowed_root, snapshot_root, work_root, log_root,
                 timeout_seconds=2.0, max_log_bytes=65536):
        self.root = Path(allowed_root).absolute()
        if not self.root.is_dir():
            raise ContractError('explicit_local_root_required')
        _no_links(self.root)
        self.snapshots = inside(snapshot_root, self.root)
        self.work = inside(work_root, self.root)
        self.logs = inside(log_root, self.root)
        roots = (self.snapshots.resolve(), self.work.resolve(), self.logs.resolve())
        if any(a == b or a in b.parents or b in a.parents
               for i, a in enumerate(roots) for b in roots[i + 1:]):
            raise ContractError('local_roots_overlap')
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 5:
            raise ContractError('invalid_local_timeout')
        if type(max_log_bytes) is not int or not 64 <= max_log_bytes <= 1_000_000:
            raise ContractError('invalid_local_log_limit')
        self.timeout, self.max_log_bytes = float(timeout_seconds), max_log_bytes
        self.work.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.started_processes = 0
        self._captures = {}

    def _preflight(self, task, request):
        obj, req = task_data(task), request_data(task, request)
        provenance = obj['provenance']
        if provenance['input_kind'] != 'synthetic_fixture':
            raise ContractError('real_source_execution_forbidden')
        if (obj['source_id'] != 'trusted-local-001' or obj['repo'] != 'synthetic.local/trusted-tiny'
            or obj['family_id'] != 'synthetic.local/trusted-tiny-root'
            or obj['problem_statement'] != 'Self-authored trusted synthetic execution fixture only.'
            or provenance['dataset_revision'] != sha({'fixture': FIXTURE_VERSION})[:40]):
            raise ContractError('unregistered_fixture')
        if req['actor_id'] != ACTOR or req['executor_id'] != EXECUTOR:
            raise ContractError('local_execution_identity_mismatch')
        expected_env = dict(env_id=FIXTURE_VERSION, image_name='local/trusted-standardlib:0.1',
                            image_digest=sha({'fixture': FIXTURE_VERSION}),
                            lock_sha256=sha({'python': sys.version, 'fixture': FIXTURE_VERSION}))
        if obj['environment'] != expected_env:
            raise ContractError('unregistered_local_environment')
        if obj['test_plan']['parser_sha256'] != file_digest(Path(local_parser.__file__).read_bytes()):
            raise ContractError('local_parser_hash_mismatch')
        source_patch = patch(3, 2, role='inject_into_clean') if obj['provider'] == 'swe-smith-py' else patch()
        if obj['source_patch']['sha256'] != file_digest(source_patch.encode()):
            raise ContractError('unregistered_source_patch')
        for state, limit in (('clean', 3), ('broken', 2), ('reference', 3)):
            directory = inside(self.snapshots / state, self.snapshots)
            if tree_digest(_snapshot(directory, limit)) != obj['snapshots'][state]:
                raise ContractError('snapshot_binding_mismatch')
        if obj['snapshots']['base_commit'] != tree_digest(files(3))[:40]:
            raise ContractError('local_base_revision_mismatch')
        for argv in req['command_argv']:
            if len(argv) != 6 or argv[:5] != [sys.executable, '-I', '-S', '-B', 'checks.py'] or argv[5] not in NODES:
                raise ContractError('untrusted_local_argv')
        if any(node not in NODES for node in req['selected_test_ids']):
            raise ContractError('unregistered_local_test_id')
        candidate = req['candidate']
        limit = 2 if req['purpose'].startswith('candidate') or 'broken' in req['purpose'] else 3
        if candidate:
            try:
                change = json.loads(candidate['patch'])
            except ValueError as error:
                raise ContractError('untrusted_local_patch') from error
            fields = {'format', 'role', 'file', 'before', 'after'}
            keys(change, fields, fields)
            if change['file'] != 'tiny.py':
                raise ContractError('local_patch_path_forbidden')
            if change['format'] != FIXTURE_VERSION or change['role'] != 'repair_from_broken' or change['before'] != 2:
                raise ContractError('local_patch_direction_mismatch')
            if type(change['before']) is not int or type(change['after']) is not int or change['after'] not in (0, 2, 3):
                raise ContractError('untrusted_local_patch')
            limit = change['after']
        return obj, req, limit

    def execute(self, task, request):
        obj, req, limit = self._preflight(task, request)  # No process before this passes.
        run_id = uuid.uuid4().hex
        workspace = inside(self.work / run_id, self.work)
        log_dir = inside(self.logs / run_id, self.logs)
        workspace.mkdir(exist_ok=False)
        try:
            log_dir.mkdir(exist_ok=False)
            before_limit = 2 if req['purpose'].startswith('candidate') or 'broken' in req['purpose'] else 3
            for name, data in files(before_limit).items():
                (workspace / name).write_bytes(data)
            actual_base = tree_digest(_snapshot(workspace, before_limit))
            if actual_base != req['input_tree_sha256']:
                raise ContractError('materialized_base_mismatch')
            if req['candidate']:
                # Restricted self-authored patch is a numeric edit, never arbitrary code.
                (workspace / 'tiny.py').write_bytes(files(limit)['tiny.py'])
            actual_patched = tree_digest(_snapshot(workspace, limit))
        except BaseException:
            inside(workspace, self.work)
            shutil.rmtree(workspace)
            raise
        started = time.monotonic()
        commands, outcomes, causes = [], {}, []
        aggregate_stdout, aggregate_stderr = bytearray(), bytearray()
        cleanup_complete = True
        try:
            for index, argv in enumerate(req['command_argv']):
                gate = workspace / ('.owned_gate_%d' % index)
                stdout_file, stderr_file = log_dir / ('%d.stdout.bin' % index), log_dir / ('%d.stderr.bin' % index)
                owner = None
                timed_out = False
                exit_code = None
                command_started = time.monotonic()
                env = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP') if k in os.environ}
                env['SF_EXEC_GATE'] = str(gate)
                try:
                    with stdout_file.open('xb') as stdout, stderr_file.open('xb') as stderr:
                        owner = OwnedProcess(argv, cwd=workspace, env=env, stdout=stdout, stderr=stderr, gate=gate)
                        self.started_processes += 1
                        try:
                            exit_code = owner.wait(self.timeout)
                        except subprocess.TimeoutExpired:
                            timed_out = True
                        finally:
                            owner.stop(workspace)
                            exit_code = owner.proc.returncode
                except (OSError, subprocess.SubprocessError) as error:
                    causes.append('process_control_error:' + type(error).__name__)
                raw_stdout, raw_stderr = stdout_file.read_bytes(), stderr_file.read_bytes()
                aggregate_stdout.extend(raw_stdout)
                aggregate_stderr.extend(raw_stderr)
                parsed, parse_cause = local_parser.parse_log(raw_stdout, max_bytes=self.max_log_bytes)
                if len(raw_stderr) > self.max_log_bytes:
                    parse_cause = 'log_limit_exceeded'
                if timed_out:
                    causes.append('command_timeout')
                elif parse_cause:
                    causes.append(parse_cause)
                for item in parsed:
                    if item['test_id'] in outcomes:
                        causes.append('duplicate_command_test_id')
                    outcomes[item['test_id']] = item['status']
                if any(item['status'] == 'ERROR' for item in parsed):
                    causes.append('trusted_test_error')
                if any(item['status'] == 'SKIP' for item in parsed):
                    causes.append('trusted_test_skip')
                if not parse_cause and not timed_out:
                    expected_exit = 2 if any(t['status'] == 'ERROR' for t in parsed) else 1 if any(t['status'] == 'FAIL' for t in parsed) else 0
                    if exit_code != expected_exit:
                        causes.append('command_exit_log_mismatch')
                stopped = bool(owner and owner.proc.poll() is not None and owner.children_stopped and owner.job_empty)
                cleanup_complete &= stopped
                if not stopped:
                    causes.append('owned_process_cleanup_failed')
                commands.append(dict(argv=list(argv), cwd=str(workspace), exit_code=exit_code,
                                     duration_ms=int((time.monotonic() - command_started) * 1000),
                                     timed_out=timed_out, registered_pids=owner.registered if owner else [],
                                     processes_stopped=stopped, stdout_file=stdout_file.name,
                                     stderr_file=stderr_file.name, stdout_sha256=file_digest(raw_stdout),
                                     stderr_sha256=file_digest(raw_stderr), parser_cause=parse_cause))
                if timed_out or not stopped:
                    break
            for node in req['selected_test_ids']:
                if node not in outcomes:
                    outcomes[node] = 'NOT_RUN'
                    causes.append('declared_test_not_covered')
        finally:
            # Recursive removal only of this registered UUID workspace, after path checks.
            inside(workspace, self.work)
            shutil.rmtree(workspace)
        all_stdout, all_stderr = bytes(aggregate_stdout), bytes(aggregate_stderr)
        (log_dir / 'stdout.bin').write_bytes(all_stdout)
        (log_dir / 'stderr.bin').write_bytes(all_stderr)
        fixture_unchanged = all(tree_digest(_snapshot(self.snapshots / state, value)) == obj['snapshots'][state]
                                for state, value in (('clean', 3), ('broken', 2), ('reference', 3)))
        if not fixture_unchanged or workspace.exists():
            causes.append('local_filesystem_cleanup_failed')
        cause = ';'.join(sorted(set(causes))) or None
        status = 'timeout' if 'command_timeout' in causes else 'invalid_env' if any(c.startswith('process_control_error') or c == 'owned_process_cleanup_failed' for c in causes) else 'inconclusive' if causes else 'completed'
        fact = record_fact(task, request, run_id=run_id, run_status=status, cause=cause,
                           test_outcomes=[dict(test_id=node, status=state) for node, state in sorted(outcomes.items())],
                           stdout_sha256=file_digest(all_stdout), stderr_sha256=file_digest(all_stderr),
                           duration_ms=int((time.monotonic() - started) * 1000))
        capture = Record.seal('local_execution', dict(
            schema=EXECUTION_VERSION, parser_version=local_parser.PARSER_VERSION,
            run_id=run_id, snapshot_root=str(self.snapshots), actual_base_tree=actual_base,
            actual_patched_tree=actual_patched, task_sha256=task.sha256, request_sha256=request.sha256,
            fact_sha256=fact.sha256, commands=commands, workspace_removed=not workspace.exists(),
            fixture_unchanged=fixture_unchanged, processes_stopped=cleanup_complete,
            trusted_synthetic_only=True, untrusted_code_isolated=False, publishable=False,
        ))
        (log_dir / 'fact.json').write_text(fact.json_text, encoding='utf-8')
        (log_dir / 'capture.json').write_text(capture.json_text, encoding='utf-8')
        self._captures[run_id] = capture.sha256
        return LocalExecution(fact, capture, log_dir)

    def verify_logs(self, result):
        payload = result.capture.data('local_execution')
        if self._captures.get(payload['run_id']) != result.capture.sha256:
            raise ContractError('local_capture_not_emitted')
        folder = inside(result.log_directory, self.logs)
        if folder != self.logs / payload['run_id']:
            raise ContractError('local_log_directory_mismatch')
        fact = result.fact.data('fact')
        if payload['fact_sha256'] != result.fact.sha256:
            raise ContractError('local_capture_fact_mismatch')
        if file_digest((folder / 'fact.json').read_bytes()) != result.fact.sha256:
            raise ContractError('local_fact_file_hash_mismatch')
        if file_digest((folder / 'capture.json').read_bytes()) != result.capture.sha256:
            raise ContractError('local_capture_file_hash_mismatch')
        for stream in ('stdout', 'stderr'):
            if file_digest((folder / (stream + '.bin')).read_bytes()) != fact[stream + '_sha256']:
                raise ContractError('raw_log_hash_mismatch')
        for command in payload['commands']:
            for stream in ('stdout', 'stderr'):
                path = inside(folder / command[stream + '_file'], folder)
                if file_digest(path.read_bytes()) != command[stream + '_sha256']:
                    raise ContractError('raw_command_log_hash_mismatch')
        return True

    def derive_reward(self, task, request, result):
        """Execution-aware entry: refuses hand-made captures or changed raw logs."""
        from .derive import derive_reward
        self.verify_logs(result)
        return derive_reward(task, request, result.fact)

    def derive_sft_eligibility(self, task, request, result):
        from .derive import derive_sft_eligibility
        self.verify_logs(result)
        return derive_sft_eligibility(task, request, result.fact)
