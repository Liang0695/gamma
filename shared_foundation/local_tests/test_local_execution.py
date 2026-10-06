"""Opt-in production runner cases: self-authored code/logs, no external tasks."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from shared_foundation import ContractError, Record, prepare_request, qualify
from shared_foundation.records import PURPOSE_TREE, patch_sha
from shared_foundation.local_fixtures import ACTOR, EXECUTOR, files, fixture_task, install_fixture, patch, tree_digest
from shared_foundation.local_runner import LocalExecution, LocalSyntheticRunner

PACKAGE = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = None  # Set by run_local_checks for persisted real raw logs.


class LocalExecutionTests(unittest.TestCase):
    def setUp(self):
        work_parent = PACKAGE / '_local_work'
        work_parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='owned-case-', dir=work_parent)
        self.root = Path(self.temp.name)
        self.snapshots = install_fixture(self.root / 'snapshots')
        self.results = []

    def tearDown(self):
        if ARCHIVE_ROOT is not None:
            target = Path(ARCHIVE_ROOT) / self._testMethodName
            target.mkdir(parents=True, exist_ok=True)
            for result in self.results:
                shutil.copytree(result.log_directory, target / result.capture.data()['run_id'])
        self.temp.cleanup()  # Only this TemporaryDirectory's own paths.

    def runner(self, **options):
        return LocalSyntheticRunner(allowed_root=self.root, snapshot_root=self.snapshots,
                                    work_root=self.root / 'work', log_root=self.root / 'logs', **options)

    def request(self, task, content=None, purpose='candidate.evaluate'):
        candidate = None
        if purpose == 'candidate.evaluate':
            content = content or patch()
            obj = task.data()
            candidate = dict(role='candidate', task_id=obj['task_id'], actor_id=ACTOR,
                             patch=content, patch_sha256=patch_sha(content),
                             base_commit=obj['snapshots']['base_commit'],
                             base_tree_sha256=obj['snapshots']['broken'])
        return prepare_request(task, purpose, actor_id=ACTOR, executor_id=EXECUTOR, candidate=candidate)

    def execute(self, runner, task, request):
        result = runner.execute(task, request)
        self.results.append(result)
        self.assertTrue(runner.verify_logs(result))
        capture = result.capture.data()
        self.assertTrue(capture['workspace_removed'])
        self.assertTrue(capture['fixture_unchanged'])
        self.assertTrue(capture['processes_stopped'])
        self.assertFalse(capture['untrusted_code_isolated'])
        self.assertFalse(capture['publishable'])
        return result

    def test_actual_candidate_logs_parser_and_immutable_derivations(self):
        runner, task = self.runner(), fixture_task()
        request = self.request(task)
        result = self.execute(runner, task, request)
        before = {p.name: p.read_bytes() for p in result.log_directory.iterdir()}
        reward = runner.derive_reward(task, request, result).data()
        eligibility = runner.derive_sft_eligibility(task, request, result).data()
        self.assertEqual(reward['reward'], 1)
        self.assertTrue(eligibility['eligible'])
        self.assertFalse(eligibility['gemma_batch_generated'])
        self.assertIn(b'SFTEST', (result.log_directory / 'stdout.bin').read_bytes())
        self.assertEqual(before, {p.name: p.read_bytes() for p in result.log_directory.iterdir()})
        self.assertEqual(result.capture.data()['actual_base_tree'], tree_digest(files(2)))
        self.assertEqual(result.capture.data()['actual_patched_tree'], tree_digest(files(3)))
        commands = result.capture.data()['commands']
        self.assertEqual(commands[0]['argv'][-1], 'upper edge')
        self.assertEqual([c['exit_code'] for c in commands], [0, 0])

    def test_real_assertion_failure_is_zero_not_environment_na(self):
        runner, task = self.runner(), fixture_task()
        request = self.request(task, patch(after=0))
        result = self.execute(runner, task, request)
        self.assertEqual(result.fact.data()['run_status'], 'completed')
        self.assertEqual(runner.derive_reward(task, request, result).data()['reward'], 0)
        self.assertEqual(result.capture.data()['commands'][0]['exit_code'], 1)

    def test_wrong_patch_direction_never_executes(self):
        runner, task = self.runner(), fixture_task()
        request = self.request(task, patch(3, 2, role='inject_into_clean'))
        with self.assertRaisesRegex(ContractError, 'local_patch_direction_mismatch'):
            runner.execute(task, request)
        self.assertEqual(runner.started_processes, 0)
        self.assertEqual(list((self.root / 'work').iterdir()), [])

    def test_missing_candidate_cannot_execute_or_use_reference(self):
        runner, task = self.runner(), fixture_task()
        payload = self.request(task).data()
        payload['candidate'] = None
        with self.assertRaisesRegex(ContractError, 'candidate_required'):
            runner.execute(task, Record.seal('request', payload))
        self.assertEqual(runner.started_processes, 0)

    def test_p2p_not_executed_is_na_with_real_empty_log(self):
        runner = self.runner()
        task = fixture_task(p2p_commands=('empty suite',))
        request = self.request(task)
        result = self.execute(runner, task, request)
        reward = runner.derive_reward(task, request, result).data()
        self.assertIsNone(reward['reward'])
        self.assertIn('declared_test_not_covered', reward['cause'])
        self.assertIn(b'"count": 0', (result.log_directory / '1.stdout.bin').read_bytes())
        self.assertIn({'test_id': 'lower edge', 'status': 'NOT_RUN'}, result.fact.data()['test_outcomes'])

    def test_error_skip_and_empty_suite_do_not_turn_into_zero(self):
        runner = self.runner()
        for node, cause in (('error case', 'trusted_test_error'), ('skip case', 'trusted_test_skip'),
                            ('empty suite', 'empty_test_suite')):
            with self.subTest(node=node):
                task = fixture_task(f2p=(node,))
                request = self.request(task)
                result = self.execute(runner, task, request)
                reward = runner.derive_reward(task, request, result).data()
                self.assertIsNone(reward['reward'])
                self.assertIn(cause, reward['cause'])

    def test_missing_end_marker_is_log_truncation_not_pass(self):
        runner, task = self.runner(), fixture_task(f2p=('missing end',))
        request = self.request(task)
        result = self.execute(runner, task, request)
        self.assertIn(b'"status": "PASS"', (result.log_directory / '0.stdout.bin').read_bytes())
        reward = runner.derive_reward(task, request, result).data()
        self.assertIsNone(reward['reward'])
        self.assertIn('log_incomplete', reward['cause'])

    def test_over_limit_log_is_preserved_in_full_and_na(self):
        runner, task = self.runner(max_log_bytes=256), fixture_task(f2p=('large log',))
        request = self.request(task)
        result = self.execute(runner, task, request)
        self.assertGreater(len((result.log_directory / '0.stdout.bin').read_bytes()), 5000)
        reward = runner.derive_reward(task, request, result).data()
        self.assertIsNone(reward['reward'])
        self.assertIn('log_limit_exceeded', reward['cause'])

    def test_timeout_stops_the_owned_process(self):
        runner, task = self.runner(timeout_seconds=0.2), fixture_task(f2p=('sleep case',))
        request = self.request(task)
        result = self.execute(runner, task, request)
        reward = runner.derive_reward(task, request, result).data()
        self.assertIsNone(reward['reward'])
        self.assertEqual(result.fact.data()['run_status'], 'timeout')
        self.assertTrue(result.capture.data()['commands'][0]['timed_out'])
        self.assertEqual(len(result.capture.data()['commands'][0]['registered_pids']), 1)

    def test_timeout_stops_registered_child_as_well(self):
        runner, task = self.runner(timeout_seconds=0.7), fixture_task(f2p=('spawn case',))
        request = self.request(task)
        result = self.execute(runner, task, request)
        command = result.capture.data()['commands'][0]
        self.assertTrue(command['timed_out'])
        self.assertEqual(len(command['registered_pids']), 2)
        self.assertTrue(command['processes_stopped'])
        self.assertIn(b'OWNED_CHILD', (result.log_directory / 'stdout.bin').read_bytes())

    def test_candidates_never_reuse_residual_workspace(self):
        runner, task = self.runner(), fixture_task(f2p=('residual check',))
        request = self.request(task)
        first = self.execute(runner, task, request)
        second = self.execute(runner, task, request)
        self.assertEqual(runner.derive_reward(task, request, first).data()['reward'], 1)
        self.assertEqual(runner.derive_reward(task, request, second).data()['reward'], 1)
        self.assertNotEqual(first.capture.data()['commands'][0]['cwd'], second.capture.data()['commands'][0]['cwd'])
        self.assertEqual(first.capture.data()['actual_base_tree'], second.capture.data()['actual_base_tree'])
        self.assertEqual(list((self.root / 'work').iterdir()), [])

    def test_path_escape_patch_and_root_are_rejected_before_execution(self):
        with self.assertRaisesRegex(ContractError, 'local_path_outside_root'):
            LocalSyntheticRunner(allowed_root=self.root, snapshot_root=self.snapshots,
                                 work_root=self.root.parent / 'outside-case-root', log_root=self.root / 'logs')
        runner, task = self.runner(), fixture_task()
        with self.assertRaisesRegex(ContractError, 'local_patch_path_forbidden'):
            runner.execute(task, self.request(task, patch(file='../tiny.py')))
        self.assertEqual(runner.started_processes, 0)

    def test_changed_snapshot_and_non_numeric_code_patch_are_not_executed(self):
        runner, task = self.runner(), fixture_task()
        malicious = json.loads(patch())
        malicious['after'] = "__import__('os').system('external')"
        with self.assertRaisesRegex(ContractError, 'untrusted_local_patch'):
            runner.execute(task, self.request(task, json.dumps(malicious)))
        (self.snapshots / 'broken' / 'tiny.py').write_bytes(b'not the trusted program')
        with self.assertRaisesRegex(ContractError, 'trusted_snapshot_content_mismatch'):
            runner.execute(task, self.request(task))
        self.assertEqual(runner.started_processes, 0)

    def test_real_sources_and_arbitrary_argv_rejected(self):
        runner = self.runner()
        task = fixture_task(input_kind='unapproved_external')
        with self.assertRaisesRegex(ContractError, 'real_source_execution_forbidden'):
            runner.execute(task, self.request(task))
        task = fixture_task()
        payload = task.data()
        payload['test_plan']['f2p_argv'] = [['external-program', '-c', 'print("pass")']]
        altered = Record.seal('task', payload)
        with self.assertRaisesRegex(ContractError, 'untrusted_local_argv'):
            runner.execute(altered, self.request(altered))
        self.assertEqual(runner.started_processes, 0)

    def test_callers_cannot_supply_pass_outcomes_to_executor(self):
        runner, task = self.runner(), fixture_task()
        with self.assertRaises(TypeError):
            runner.execute(task, self.request(task), test_outcomes=[{'test_id': 'upper edge', 'status': 'PASS'}])
        self.assertEqual(runner.started_processes, 0)

    def test_log_tamper_and_handmade_capture_rejected(self):
        runner, task = self.runner(), fixture_task()
        request = self.request(task)
        result = self.execute(runner, task, request)
        payload = result.capture.data()
        payload['publishable'] = True
        forged = LocalExecution(result.fact, Record.seal('local_execution', payload), result.log_directory)
        with self.assertRaisesRegex(ContractError, 'local_capture_not_emitted'):
            runner.derive_reward(task, request, forged)
        with (result.log_directory / 'stdout.bin').open('ab') as output:
            output.write(b'fake trailing PASS')
        with self.assertRaisesRegex(ContractError, 'raw_log_hash_mismatch'):
            runner.derive_reward(task, request, result)

    def test_identical_logs_from_other_run_cannot_replace_run_identity(self):
        runner, task = self.runner(), fixture_task()
        request = self.request(task)
        first = self.execute(runner, task, request)
        second = self.execute(runner, task, request)
        self.assertEqual(first.fact.data()['stdout_sha256'], second.fact.data()['stdout_sha256'])
        swapped = LocalExecution(first.fact, first.capture, second.log_directory)
        with self.assertRaisesRegex(ContractError, 'local_log_directory_mismatch'):
            runner.derive_reward(task, request, swapped)

    def test_real_qualification_segments_both_source_directions(self):
        runner = self.runner()
        for provider in ('rebench', 'smith'):
            task, pairs = fixture_task(provider=provider), []
            for purpose in PURPOSE_TREE:
                if not purpose.startswith('qualification.'):
                    continue
                request = self.request(task, purpose=purpose)
                for _ in range(2):
                    result = self.execute(runner, task, request)
                    pairs.append((request, result.fact))
            self.assertTrue(qualify(task, pairs).data()['eligible'])
            self.assertFalse(qualify(task, pairs).data()['publishable'])
