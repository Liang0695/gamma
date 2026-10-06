import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from shared_foundation.native_artifact import (build_artifact, load_artifact, save_artifact, validate_artifact)
from shared_foundation.native_e0 import E0_SHA, adapt_e0
from shared_foundation.native_messages import map_native, validate_mapping
from shared_foundation.native_render import OfficialNativeRenderer, labels_from_offsets
from shared_foundation.records import ContractError, Record, canonical, sha

RENDERER = None
E0_TYPES = None
ARTIFACT_ROOT = None


def example(*, call=True):
    steps = [
        dict(step_id='sys', role='system', content='Trusted synthetic messages only.'),
        dict(step_id='u0', role='user', content='检查中文文件及引号 "quote"。'),
        dict(step_id='history', role='assistant', content='same target text', reasoning='HISTORY REASONING',
             tool_calls=[dict(call_id='read-1', name='read_file', arguments={'path': '目录/引号".py'})]),
        dict(step_id='result', role='tool', content='文件包含 LIMIT = 2\n"quoted"',
             tool_result=dict(call_id='read-1', name='read_file')),
        dict(step_id='u1', role='user', content='现在仅修复这一处。'),
        dict(step_id='target', role='assistant', content='same target text', reasoning='TARGET REASONING'),
    ]
    if call:
        steps[-1]['tool_calls'] = [dict(call_id='edit-1', name='apply_edit', arguments={
            'path': '目录/引号".py', 'text': '你好，"双引号"\nLIMIT = 3', 'nested': {'ok': True, 'n': 3}})]
    tools = [dict(type='function', function=dict(name=name, description='Synthetic fixture tool.',
        parameters=dict(type='object', properties={'path': {'type': 'string'}}, required=['path'])))
        for name in ('read_file', 'apply_edit')]
    provenance = dict(input_kind='synthetic_fixture', task_id='synthetic-native-001',
        family_id='synthetic-native-family', source_sha256=sha({'synthetic_source': 1}),
        fact_sha256=sha({'synthetic_fact_reference': 1}), source_version='synthetic-native-source/0.1',
        fact_version='synthetic-fact-reference/0.1')
    return steps, tools, provenance


def mapping(call=True):
    steps, tools, provenance = example(call=call)
    return map_native(steps, target_step_id='target', tools=tools, provenance=provenance)


PLAN = dict(steps=1, lr=0.0001, seq_len=1024, lora_rank=16, lora_alpha=32, seed=7)


class MappingChecks(unittest.TestCase):
    def rejected(self, fn, code):
        with self.assertRaises(ContractError) as cm:
            fn()
        self.assertEqual(cm.exception.code, code)

    def test_structured_mapping_keeps_unicode_quotes_and_explicit_target(self):
        obj = validate_mapping(mapping())
        self.assertEqual(obj['target_index'], 5)
        self.assertIsInstance(obj['messages'][-1]['tool_calls'][0]['function']['arguments'], dict)
        self.assertEqual(obj['messages'][3]['tool_call_id'], 'read-1')
        self.assertFalse(obj['publishable'])

    def test_duplicate_target_and_missing_target_rejected(self):
        steps, tools, p = example()
        steps[2]['step_id'] = 'target'
        self.rejected(lambda: map_native(steps, target_step_id='target', tools=tools, provenance=p), 'native_ambiguous_step_id')
        steps[2]['step_id'] = 'history'
        self.rejected(lambda: map_native(steps, target_step_id='absent', tools=tools, provenance=p), 'native_target_missing')

    def test_empty_target_and_nonassistant_target_rejected(self):
        steps, tools, p = example(call=False)
        steps[-1]['content'] = '   '
        self.rejected(lambda: map_native(steps, target_step_id='target', tools=tools, provenance=p), 'native_empty_action')
        steps[-1]['content'] = 'body'
        self.rejected(lambda: map_native(steps, target_step_id='u1', tools=tools, provenance=p), 'native_target_must_be_assistant')

    def test_post_target_feedback_cannot_become_input(self):
        steps, tools, p = example()
        steps.append(dict(step_id='future', role='tool', content='Future answer',
                          tool_result=dict(call_id='edit-1', name='apply_edit')))
        self.rejected(lambda: map_native(steps, target_step_id='target', tools=tools, provenance=p), 'native_post_target_input_forbidden')

    def test_wrong_result_binding_and_string_arguments_rejected(self):
        steps, tools, p = example()
        steps[3]['tool_result']['name'] = 'apply_edit'
        self.rejected(lambda: map_native(steps, target_step_id='target', tools=tools, provenance=p), 'native_result_binding_invalid')
        steps, tools, p = example()
        steps[-1]['tool_calls'][0]['arguments'] = '{}'
        self.rejected(lambda: map_native(steps, target_step_id='target', tools=tools, provenance=p), 'native_call_binding_invalid')

    def test_reserved_control_tokens_and_unknown_fields_rejected(self):
        steps, tools, p = example()
        steps[-1]['content'] = '<turn|>'
        self.rejected(lambda: map_native(steps, target_step_id='target', tools=tools, provenance=p), 'native_control_token_injection')
        steps[-1]['content'] = 'body'
        steps[-1]['loss_eligible'] = True
        self.rejected(lambda: map_native(steps, target_step_id='target', tools=tools, provenance=p), 'unknown_fields')

    def test_resealed_mapping_version_or_projection_cannot_bypass_rebuild(self):
        obj = mapping().data()
        obj['schema'] = 'native-message-map/999'
        self.rejected(lambda: validate_mapping(Record.seal('native_mapping', obj)), 'native_mapping_binding_or_version_mismatch')
        obj = mapping().data()
        obj['messages'][-1]['content'] = 'injected'
        self.rejected(lambda: validate_mapping(Record.seal('native_mapping', obj)), 'native_mapping_binding_or_version_mismatch')

    def test_empty_overlap_crossing_and_missing_offsets_rejected(self):
        self.rejected(lambda: labels_from_offsets([1], [(0, 1)], [], 1), 'native_empty_action_span')
        self.rejected(lambda: labels_from_offsets([1], [(0, 2)], [dict(start=1,end=2,channel='action')], 2), 'native_ambiguous_token_span')
        self.rejected(lambda: labels_from_offsets([1], [(0, 0)], [dict(start=0,end=1,channel='action')], 1), 'native_precise_offsets_required')

    def test_missing_official_materials_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.rejected(lambda: OfficialNativeRenderer(material_root=directory), 'official_material_missing_or_linked')

    def test_material_hash_tamper_and_shim_renderer_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'chat_template.jinja').write_bytes(b'fake template')
            self.rejected(lambda: OfficialNativeRenderer(material_root=directory), 'official_material_hash_mismatch')
        self.rejected(lambda: build_artifact(mapping(), renderer=object(), plan=PLAN), 'native_official_renderer_required')


class OfficialChecks(MappingChecks):
    """Only executes when a real official renderer was loaded by the explicit runner."""
    # Do not inherit duplicate generic test methods in the suite.
    def setUp(self):
        if RENDERER is None:
            self.skipTest('official material validation not_run; no shim fallback')

    def artifact(self, call=True, **kwargs):
        return build_artifact(mapping(call), renderer=RENDERER, plan=PLAN, **kwargs)

    def test_canonical_trace_bytes_ids_tool_result_and_unicode(self):
        original = mapping()
        rendered = self.artifact().data()['render']
        self.assertEqual(rendered['canonical_text'], RENDERER.canonical(original))
        self.assertEqual(rendered['input_ids'], RENDERER.backend.encode(rendered['canonical_text'], add_special_tokens=False).ids)
        self.assertTrue(rendered['canonical_byte_equal'])
        self.assertTrue(rendered['canonical_token_ids_equal'])
        self.assertIn('<|tool_response>response:read_file', rendered['canonical_text'])
        self.assertIn('目录/引号".py', rendered['canonical_text'])
        self.assertNotIn('\ue000', rendered['canonical_text'])

    def test_history_and_target_with_identical_body_do_not_share_labels(self):
        r = self.artifact(call=False).data()['render']
        text = r['canonical_text']
        first, second = text.index('same target text'), text.rindex('same target text')
        for label, (start, end) in zip(r['labels'], r['offsets']):
            if start < first + len('same target text') and end > first:
                self.assertEqual(label, -100)
            if second <= start and end <= second + len('same target text'):
                self.assertNotEqual(label, -100)
        self.assertEqual(r['supervised_tokens'], sum(label != -100 for label in r['labels']))
        self.assertGreater(r['supervised_tokens'], 0)

    def test_same_model_turn_after_tool_result_still_isolates_target(self):
        steps, tools, p = example(call=False)
        del steps[4]  # No user separator: official template continues the same model turn.
        m = map_native(steps, target_step_id='target', tools=tools, provenance=p)
        artifact = build_artifact(m, renderer=RENDERER, plan=PLAN)
        r = artifact.data()['render']
        text = r['canonical_text']
        first, final = text.index('same target text'), text.rindex('same target text')
        self.assertLess(first, final)
        self.assertEqual(text.count('<|turn>model\n'), 1)
        for label, (start, end) in zip(r['labels'], r['offsets']):
            if start < first + len('same target text') and end > first:
                self.assertEqual(label, -100)
            if final <= start and end <= final + len('same target text'):
                self.assertNotEqual(label, -100)
        self.assertTrue(r['canonical_byte_equal'] and r['canonical_token_ids_equal'])
        if ARTIFACT_ROOT:
            ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
            save_artifact(ARTIFACT_ROOT / (artifact.sha256 + '.json'), artifact, renderer=RENDERER)

    def test_reasoning_is_context_and_thinking_kwargs_have_real_behavior(self):
        off = self.artifact(enable_thinking=False, preserve_thinking=False).data()['render']
        on = self.artifact(enable_thinking=True, preserve_thinking=True).data()['render']
        self.assertEqual(on['kwargs']['enable_thinking'], True)
        self.assertNotIn('<|think|>', off['canonical_text'])
        self.assertIn('<|think|>', on['canonical_text'])
        self.assertNotIn('HISTORY REASONING', off['canonical_text'])
        self.assertIn('HISTORY REASONING', on['canonical_text'])
        pos = on['canonical_text'].index('TARGET REASONING')
        for label, (begin, end) in zip(on['labels'], on['offsets']):
            if begin < pos + len('TARGET REASONING') and end > pos:
                self.assertEqual(label, -100)
        self.assertIn('TARGET REASONING', off['canonical_text'])  # Actual template semantics, not shim's on/off rule.
        if ARTIFACT_ROOT:
            ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
            artifact = self.artifact(enable_thinking=True, preserve_thinking=True)
            save_artifact(ARTIFACT_ROOT / (artifact.sha256 + '.json'), artifact, renderer=RENDERER)

    def test_real_offsets_reject_empty_or_partial_token_action_spans(self):
        encoded = RENDERER.backend.encode('uninterrupted 中文动作', add_special_tokens=False)
        self.rejected(lambda: labels_from_offsets(encoded.ids, encoded.offsets, [], len('uninterrupted 中文动作')), 'native_empty_action_span')
        start, end = next(pair for pair in encoded.offsets if pair[1] - pair[0] > 1)
        region = [dict(start=start+1, end=end, channel='action')]
        self.rejected(lambda: labels_from_offsets(encoded.ids, encoded.offsets, region, len('uninterrupted 中文动作')), 'native_ambiguous_token_span')

    def test_real_terminal_tokens_are_supervised(self):
        call = self.artifact().data()['render']
        plain = self.artifact(call=False).data()['render']
        self.assertIn(RENDERER.special_ids['<tool_call|>'], call['labels'])
        self.assertIn(RENDERER.special_ids['<|tool_response>'], call['labels'])
        self.assertIn(RENDERER.special_ids['<turn|>'], plain['labels'])
        self.assertNotIn(RENDERER.special_ids['<bos>'], call['labels'])

    def test_no_silent_truncation(self):
        self.rejected(lambda: build_artifact(mapping(), renderer=RENDERER, plan=dict(PLAN, seq_len=2)), 'native_truncation_forbidden')
        RENDERER.backend.enable_truncation(max_length=2)
        try:
            self.rejected(lambda: self.artifact(), 'native_backend_padding_or_truncation_forbidden')
        finally:
            RENDERER.backend.no_truncation()

    def test_roundtrip_stores_actual_arrays_and_complete_plan(self):
        artifact = self.artifact()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'batch-plan.json'
            save_artifact(path, artifact, renderer=RENDERER)
            loaded = load_artifact(path, renderer=RENDERER, expected_sha256=artifact.sha256)
            self.assertEqual(loaded, artifact)
            self.assertEqual(loaded.data()['plan']['batches'][0]['labels'], loaded.data()['render']['labels'])
            self.assertFalse(loaded.data()['plan']['training_authorized'])
            self.assertEqual(loaded.data()['mix_eligibility'], 'not_assessed')
        if ARTIFACT_ROOT:
            ARTIFACT_ROOT.mkdir(exist_ok=True, parents=True)
            save_artifact(ARTIFACT_ROOT / (artifact.sha256 + '.json'), artifact, renderer=RENDERER)

    def test_file_hash_tamper_and_resealed_label_tamper_rejected(self):
        artifact = self.artifact()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'batch.json'
            save_artifact(path, artifact, renderer=RENDERER)
            envelope = json.loads(path.read_bytes())
            envelope['payload']['batch']['labels'][0] = 2
            path.write_bytes(canonical(envelope).encode())
            self.rejected(lambda: load_artifact(path, renderer=RENDERER, expected_sha256=artifact.sha256), 'native_artifact_hash_mismatch')
        obj = artifact.data()
        obj['batch']['labels'][0] = 2
        self.rejected(lambda: validate_artifact(Record.seal('native_artifact', obj), renderer=RENDERER), 'native_artifact_replay_mismatch')

    def test_version_template_mask_and_count_tampering_rejected(self):
        artifact = self.artifact()
        for section, field, value in [('render', 'mask_version', 'bad'), ('render', 'supervised_tokens', 999),
                                      ('batch', 'supervised_tokens', 999), ('render', 'render_version', 'bad')]:
            obj = artifact.data()
            obj[section][field] = value
            self.rejected(lambda: validate_artifact(Record.seal('native_artifact', obj), renderer=RENDERER), 'native_artifact_replay_mismatch')
        obj = artifact.data()
        obj['material_lock']['files']['chat_template.jinja']['sha256'] = '0' * 64
        self.rejected(lambda: validate_artifact(Record.seal('native_artifact', obj), renderer=RENDERER), 'native_artifact_material_lock_mismatch')
        obj = artifact.data()
        obj['schema'] = 'bad'
        self.rejected(lambda: validate_artifact(Record.seal('native_artifact', obj), renderer=RENDERER), 'native_artifact_version_or_publication_mismatch')

    def test_unapproved_input_stays_unpublishable_and_refs_preserved(self):
        steps, tools, p = example()
        p['input_kind'] = 'unapproved_external'
        m = map_native(steps, target_step_id='target', provenance=p, tools=tools)
        before = m.sha256
        artifact = build_artifact(m, renderer=RENDERER, plan=PLAN)
        self.assertEqual(m.sha256, before)
        self.assertEqual(artifact.data()['provenance'], p)
        self.assertFalse(artifact.data()['publishable'])

    def test_e0_types_optional_actual_count_and_wrong_pin_rejected(self):
        if E0_TYPES is None:
            self.skipTest('explicit pinned E0 types not supplied')
        batch_type, plan_type = E0_TYPES
        artifact = self.artifact()
        plan = adapt_e0(artifact, renderer=RENDERER, e0_sha=E0_SHA, TrainBatch=batch_type, TrainRunPlan=plan_type)
        self.assertIsInstance(plan, plan_type)
        self.assertIsInstance(plan.batches[0], batch_type)
        self.assertEqual(plan.batches[0].supervised_tokens, artifact.data()['batch']['supervised_tokens'])
        self.assertEqual(plan.batches[0].labels, artifact.data()['batch']['labels'])
        self.rejected(lambda: adapt_e0(artifact, renderer=RENDERER, e0_sha='0'*40,
                                      TrainBatch=batch_type, TrainRunPlan=plan_type), 'e0_revision_mismatch')


# Inherit only the assertion helper; generic mapping checks belong to one suite.
for _name in list(MappingChecks.__dict__):
    if _name.startswith('test_') and _name not in OfficialChecks.__dict__:
        setattr(OfficialChecks, _name, None)
