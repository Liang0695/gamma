from dataclasses import replace
import unittest

from shared_foundation import (
    ContractError, Environment, Provenance, Record, Snapshots, TestPlan,
    actor_view, adapt_rebench, adapt_smith, derive_reward, derive_sft_eligibility,
    load_fact, prepare_request, publication_guard, qualify, record_fact,
)
from shared_foundation.records import PURPOSE_TREE, patch_sha, sha, task_data


def h(value):
    return sha({'synthetic': value})


class SyntheticFixture(unittest.TestCase):
    """Hand-authored observations, not a running task or actual oracle."""
    def setUp(self):
        self.snap = Snapshots('a' * 40, h('clean'), h('broken'), h('reference'))
        self.env = Environment('synthetic-env', 'synthetic/image:fixture', h('image'), h('lock'))
        self.plan = TestPlan(
            ('tests/test space.py::test_target',), ('tests/test space.py::test_regression',),
            (('python', '-m', 'pytest', 'tests/test space.py::test_target'),),
            (('python', '-m', 'pytest', 'tests/test space.py::test_regression'),), h('parser'),
        )
        self.context = dict(snapshots=self.snap, environment=self.env, test_plan=self.plan,
                            provenance=Provenance('b' * 40, 'synthetic_fixture'),
                            family_id='synthetic/family-root')
        self.smith = dict(instance_id='synthetic-1', repo='synthetic/repo',
                          problem_statement='Synthetic fixture, not executed',
                          patch='diff --git a/f.py b/f.py\n-expected\n+broken\n',
                          FAIL_TO_PASS=list(self.plan.f2p_ids),
                          PASS_TO_PASS=list(self.plan.p2p_ids), image_name=self.env.image_name)
        self.rebench = dict(self.smith, base_commit=self.snap.base_commit, language='Python',
                            license='MIT', test_patch='synthetic test diff', created_at='2025-01-01')
        self.task = adapt_rebench(self.rebench, **self.context)

    def candidate(self, task=None):
        task = task or self.task
        content = 'diff --git a/f.py b/f.py\n-broken\n+fixed\n'
        return dict(role='candidate', task_id=task.data()['task_id'], actor_id='fixture-actor',
                    patch=content, patch_sha256=patch_sha(content),
                    base_commit=self.snap.base_commit, base_tree_sha256=self.snap.broken)

    def request(self, task=None, candidate=None, purpose='candidate.evaluate'):
        task = task or self.task
        if purpose == 'candidate.evaluate' and candidate is None:
            candidate = self.candidate(task)
        return prepare_request(task, purpose, actor_id='fixture-actor',
                               executor_id='fixture-executor', candidate=candidate)

    def fact(self, request=None, task=None, status='completed', outcomes=None, cause=None, run='fixture-run'):
        task = task or self.task
        request = request or self.request(task)
        if outcomes is None:
            outcomes = [dict(test_id=n, status='PASS') for n in request.data()['selected_test_ids']]
        return record_fact(task, request, run_id=run, run_status=status, test_outcomes=outcomes,
                           cause=cause, stdout_sha256=h('stdout'), stderr_sha256=h('stderr'), duration_ms=1)

    def rejects(self, code, callable_, *args, **kwargs):
        with self.assertRaises(ContractError) as caught:
            callable_(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)


class AdapterTests(SyntheticFixture):
    def test_both_patch_directions_and_three_trees(self):
        smith = adapt_smith(self.smith, **self.context).data()
        self.assertEqual(smith['source_patch']['role'], 'inject_into_clean')
        self.assertEqual(smith['reference_operation'], 'reverse_injection')
        self.assertEqual(smith['snapshots'], self.snap.data())
        self.assertEqual(self.task.data()['source_patch']['role'], 'repair_from_broken')

    def test_inverted_direction_is_rejected_on_production_path(self):
        for task in (self.task, adapt_smith(self.smith, **self.context)):
            payload = task.data()
            payload['reference_operation'] = 'reverse_injection' if payload['provider'] == 'swe-rebench-v2' else 'apply_repair'
            self.rejects('patch_direction_mismatch', task_data, Record.seal('task', payload))

    def test_unknown_source_fields_are_not_silently_dropped(self):
        self.rejects('unknown_fields', adapt_rebench, dict(self.rebench, oracle_hint='answer'), **self.context)

    def test_unknown_nested_task_fields_are_rejected(self):
        payload = self.task.data()
        payload['snapshots']['approved'] = True
        self.rejects('unknown_fields', task_data, Record.seal('task', payload))

    def test_source_base_and_image_must_match(self):
        self.rejects('source_base_mismatch', adapt_rebench, dict(self.rebench, base_commit='c' * 40), **self.context)
        self.rejects('source_image_mismatch', adapt_smith, dict(self.smith, image_name='other/image'), **self.context)

    def test_boolean_date_is_not_an_integer_timestamp(self):
        self.rejects('unsupported_source_date', adapt_rebench, dict(self.rebench, created_at=True), **self.context)

    def test_node_ids_are_not_argv(self):
        payload = self.task.data()
        self.assertEqual(payload['test_plan']['f2p_argv'][0][-1], 'tests/test space.py::test_target')
        invalid = replace(self.plan, p2p_argv=self.plan.p2p_ids)
        self.rejects('argv_array_required', invalid.data)

    def test_source_test_ids_must_match_plan(self):
        self.rejects('source_test_selection_mismatch', adapt_smith, dict(self.smith, PASS_TO_PASS=['other']), **self.context)

    def test_empty_duplicate_and_overlapping_test_lists(self):
        for changed, code in ((replace(self.plan, f2p_ids=()), 'invalid_f2p_ids'),
                              (replace(self.plan, f2p_ids=('same', 'same')), 'invalid_f2p_ids'),
                              (replace(self.plan, p2p_ids=self.plan.f2p_ids), 'test_sets_overlap')):
            self.rejects(code, changed.data)

    def test_snapshots_require_hashes_and_distinct_broken_reference(self):
        self.rejects('invalid_broken_tree', replace(self.snap, broken='latest').data)
        self.rejects('broken_reference_identical', replace(self.snap, broken=self.snap.reference).data)

    def test_public_projection_contains_no_patch_or_oracle(self):
        view = actor_view(self.task)
        self.assertEqual(set(view), {'task_id', 'family_id', 'repo', 'problem_statement'})
        self.assertNotIn('patch', view)
        self.assertNotIn('test_plan', view)

    def test_unapproved_and_synthetic_inputs_never_publish(self):
        self.rejects('synthetic_publication_forbidden', publication_guard, self.task)
        ext = adapt_rebench(self.rebench, **dict(self.context, provenance=Provenance('b' * 40)))
        self.rejects('unapproved_publication_forbidden', publication_guard, ext)
        claimed = adapt_rebench(self.rebench, **dict(self.context, provenance=Provenance('b' * 40, approval_ref='self-claim')))
        self.rejects('release_authority_unconnected', publication_guard, claimed)


class RequestFactTests(SyntheticFixture):
    def test_no_candidate_cannot_fall_back_to_reference(self):
        self.assertIsNotNone(self.task.data()['source_patch']['sha256'])
        self.rejects('candidate_required', prepare_request, self.task, 'candidate.evaluate',
                     actor_id='fixture-actor', executor_id='fixture-executor')

    def test_empty_candidate_is_rejected(self):
        candidate = self.candidate()
        candidate['patch'] = ''
        self.rejects('invalid_patch', self.request, candidate=candidate)

    def test_bad_hash_and_base_are_rejected(self):
        for field, value, code in (('patch_sha256', h('wrong'), 'candidate_hash_mismatch'),
                                   ('base_commit', 'd' * 40, 'candidate_base_mismatch'),
                                   ('base_tree_sha256', h('wrong'), 'candidate_base_mismatch')):
            candidate = self.candidate()
            candidate[field] = value
            self.rejects(code, self.request, candidate=candidate)

    def test_wrong_actor_task_and_role(self):
        for field, value, code in (('actor_id', 'other', 'candidate_identity_mismatch'),
                                   ('task_id', 'other', 'candidate_identity_mismatch'),
                                   ('role', 'reference', 'candidate_role_required')):
            candidate = self.candidate()
            candidate[field] = value
            self.rejects(code, self.request, candidate=candidate)

    def test_explicit_candidate_equal_to_reference_is_not_auto_rejected(self):
        # A solver may independently produce the same correct patch; presence matters.
        candidate = self.candidate()
        candidate['patch'] = self.rebench['patch']
        candidate['patch_sha256'] = patch_sha(candidate['patch'])
        self.assertEqual(self.request(candidate=candidate).data()['candidate']['role'], 'candidate')

    def test_qualification_and_candidate_modes_are_distinct(self):
        self.rejects('qualification_cannot_accept_candidate', self.request,
                     purpose='qualification.reference_f2p', candidate=self.candidate())
        reference = self.request(purpose='qualification.reference_f2p')
        self.rejects('candidate_fact_required', derive_reward, self.task, reference, self.fact(reference))

    def test_fact_identity_hash_and_request_binding(self):
        request = self.request()
        fact = self.fact(request)
        self.rejects('record_hash_mismatch', derive_reward, self.task, request, replace(fact, sha256=h('bad')))
        for field in ('actor_id', 'executor_id', 'task_id', 'input_tree_sha256', 'request_sha256', 'candidate_sha256'):
            payload = fact.data()
            payload[field] = 'other' if field.endswith('_id') else h('other')
            forged = load_fact(payload)
            self.rejects('fact_binding_mismatch', derive_reward, self.task, request, forged)

    def test_fact_unknown_fields_and_statuses(self):
        payload = self.fact().data()
        self.rejects('unknown_fields', load_fact, dict(payload, passed=True))
        payload['test_outcomes'][0]['approved'] = True
        self.rejects('unknown_fields', load_fact, payload)
        payload = self.fact().data()
        payload['test_outcomes'][0]['status'] = 'WARNING'
        self.rejects('unknown_test_status', load_fact, payload)

    def test_duplicate_nodes_and_invalid_duration(self):
        payload = self.fact().data()
        payload['test_outcomes'].append(payload['test_outcomes'][0])
        self.rejects('duplicate_test_id', load_fact, payload)
        for duration in (True, -1, 1.0):
            payload = self.fact().data()
            payload['duration_ms'] = duration
            self.rejects('invalid_duration', load_fact, payload)


class DerivationTests(SyntheticFixture):
    def test_success_and_semantic_failure_positive_paths(self):
        request = self.request()
        passed = self.fact(request)
        self.assertEqual(derive_reward(self.task, request, passed).data()['reward'], 1)
        self.assertTrue(derive_sft_eligibility(self.task, request, passed).data()['eligible'])
        failed = self.fact(request, outcomes=[dict(test_id=n, status='FAIL' if n in self.plan.f2p_ids else 'PASS') for n in request.data()['selected_test_ids']])
        self.assertEqual(derive_reward(self.task, request, failed).data()['reward'], 0)
        self.assertFalse(derive_sft_eligibility(self.task, request, failed).data()['eligible'])

    def test_p2p_not_covered_is_na_not_zero(self):
        request = self.request()
        fact = self.fact(request, outcomes=[dict(test_id=self.plan.f2p_ids[0], status='PASS')])
        result = derive_reward(self.task, request, fact).data()
        self.assertIsNone(result['reward'])
        self.assertEqual(result['cause'], 'declared_test_not_covered')

    def test_regression_is_semantic_zero(self):
        request = self.request()
        fact = self.fact(request, outcomes=[dict(test_id=n, status='FAIL' if n in self.plan.p2p_ids else 'PASS') for n in request.data()['selected_test_ids']])
        self.assertEqual(derive_reward(self.task, request, fact).data()['reward'], 0)

    def test_na_reasons_survive_derivation(self):
        request = self.request()
        for status in ('invalid_env', 'timeout', 'inconclusive', 'not_run'):
            fact = self.fact(request, status=status, cause='synthetic_' + status, outcomes=[])
            result = derive_reward(self.task, request, fact).data()
            self.assertIsNone(result['reward'])
            self.assertEqual(result['cause'], 'synthetic_' + status)

    def test_skip_error_and_unknown_node_evidence_is_na(self):
        request = self.request()
        for status in ('SKIP', 'ERROR', 'NOT_RUN'):
            fact = self.fact(request, outcomes=[dict(test_id=n, status=status) for n in request.data()['selected_test_ids']])
            self.assertIsNone(derive_reward(self.task, request, fact).data()['reward'])

    def test_extra_passed_node_does_not_make_exact_list_false_negative(self):
        request = self.request()
        outcomes = [dict(test_id=n, status='PASS') for n in request.data()['selected_test_ids']]
        outcomes.append(dict(test_id='extra::test', status='PASS'))
        self.assertEqual(derive_reward(self.task, request, self.fact(request, outcomes=outcomes)).data()['reward'], 1)

    def test_unselected_failure_cannot_be_promoted_to_success(self):
        request = self.request()
        outcomes = [dict(test_id=n, status='PASS') for n in request.data()['selected_test_ids']]
        outcomes.append(dict(test_id='extra::test', status='FAIL'))
        result = derive_reward(self.task, request, self.fact(request, outcomes=outcomes)).data()
        self.assertIsNone(result['reward'])
        self.assertEqual(result['cause'], 'unselected_test_failure_requires_baseline')

    def test_completed_run_with_uncertain_cause_does_not_get_reward(self):
        request = self.request()
        result = derive_reward(self.task, request, self.fact(request, cause='collection_unconfirmed')).data()
        self.assertIsNone(result['reward'])
        self.assertEqual(result['cause'], 'collection_unconfirmed')

    def test_derived_views_never_mutate_fact_or_publish_batches(self):
        request = self.request()
        fact = self.fact(request)
        original = (fact.json_text, fact.sha256)
        result = derive_reward(self.task, request, fact)
        sft = derive_sft_eligibility(self.task, request, fact)
        leaked_copy = fact.data()
        leaked_copy['test_outcomes'][0]['status'] = 'FAIL'
        self.assertEqual((fact.json_text, fact.sha256), original)
        self.assertEqual(fact.data()['test_outcomes'][0]['status'], 'PASS')
        self.assertNotEqual(result.sha256, sft.sha256)
        self.assertEqual(result.data()['fact_sha256'], fact.sha256)
        self.assertFalse(sft.data()['gemma_batch_generated'])
        self.assertFalse(result.data()['publishable'])

    def test_unknown_policy_rejected(self):
        request = self.request()
        fact = self.fact(request)
        self.rejects('unsupported_reward_policy', derive_reward, self.task, request, fact, policy_id='latest')
        self.rejects('unsupported_sft_policy', derive_sft_eligibility, self.task, request, fact, policy_id='latest')


class QualificationTests(SyntheticFixture):
    def pairs(self):
        pairs = []
        for purpose in PURPOSE_TREE:
            if not purpose.startswith('qualification.'):
                continue
            request = self.request(purpose=purpose)
            status = 'FAIL' if purpose == 'qualification.broken_f2p' else 'PASS'
            for repeat in range(2):
                outcomes = [dict(test_id=n, status=status) for n in request.data()['selected_test_ids']]
                pairs.append((request, self.fact(request, outcomes=outcomes, run=purpose + str(repeat))))
        return pairs

    def test_ten_observations_establish_fixture_contrast_only(self):
        result = qualify(self.task, self.pairs()).data()
        self.assertTrue(result['eligible'])
        self.assertEqual(len(result['facts_sha256']), 10)
        self.assertTrue(result['prototype'])
        self.assertFalse(result['publishable'])

    def test_missing_second_observation_is_not_qualified(self):
        self.assertFalse(qualify(self.task, self.pairs()[:-1]).data()['eligible'])

    def test_duplicate_run_cannot_count_as_independent_repeat(self):
        pairs = self.pairs()
        pairs[-1] = pairs[-2]
        self.rejects('duplicate_run_id', qualify, self.task, pairs)

    def test_unstable_reference_and_na_do_not_pass(self):
        pairs = self.pairs()
        request, _ = pairs[-1]
        pairs[-1] = (request, self.fact(request, outcomes=[dict(test_id=n, status='FAIL') for n in request.data()['selected_test_ids']], run='changed'))
        result = qualify(self.task, pairs).data()
        self.assertFalse(result['eligible'])
        self.assertTrue(any('unstable' in cause for cause in result['causes']))

    def test_candidate_fact_cannot_be_qualification(self):
        request = self.request()
        self.rejects('qualification_fact_required', qualify, self.task, [(request, self.fact(request))])


if __name__ == '__main__':
    unittest.main()
