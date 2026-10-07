"""Actual fixed E0 objects, spoof rejection and independent constructed-value checks."""
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from shared_foundation.native_artifact import build_artifact
from shared_foundation.native_e0 import E0_SHA, E0_SOURCE_LF_SHA, adapt_e0, load_e0_module
from shared_foundation.records import ContractError
from shared_foundation.native_tests import test_native as fixture

E0_ROOT = None


class E0BindingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if fixture.RENDERER is None or fixture.E0_MODULE is None or E0_ROOT is None:
            raise unittest.SkipTest('explicit official materials and fixed E0 module required')
        cls.artifact = build_artifact(fixture.mapping(), renderer=fixture.RENDERER, plan=fixture.PLAN)

    def rejected(self, fn, code):
        with self.assertRaises(ContractError) as cm:
            fn()
        self.assertEqual(cm.exception.code, code)

    def adapt(self, module=None, **kwargs):
        module = fixture.E0_MODULE if module is None else module
        return adapt_e0(self.artifact, renderer=fixture.RENDERER, e0_sha=E0_SHA,
                        e0_module=module, **kwargs)

    def test_normal_bound_actual_types_full_arrays_and_every_plan_field(self):
        module = fixture.E0_MODULE
        result = self.adapt(TrainBatch=module.TrainBatch, TrainRunPlan=module.TrainRunPlan)
        expected = self.artifact.data()
        self.assertIs(type(result), module.TrainRunPlan)
        self.assertIs(type(result.batches[0]), module.TrainBatch)
        self.assertEqual(result.batches[0].input_ids, expected['batch']['input_ids'])
        self.assertEqual(result.batches[0].labels, expected['batch']['labels'])
        self.assertEqual(result.batches[0].supervised_tokens, sum(x != -100 for x in result.batches[0].labels))
        for name, value in expected['plan']['config'].items():
            self.assertEqual(getattr(result, name), value)
            self.assertIs(type(getattr(result, name)), type(value))

    def test_reviewed_name_and_module_spoofs_rejected_before_constructor(self):
        called = []
        class FakeBatch:
            def __init__(self, input_ids, labels):
                called.append('batch')
                self.input_ids = [0] * len(input_ids)
                self.labels = [0] * len(labels)
                self.supervised_tokens = sum(x != -100 for x in labels)
        class FakePlan:
            def __init__(self, **kw):
                called.append('plan')
                self.__dict__.update(kw)
            def assert_runnable(self): pass
        FakeBatch.__name__, FakePlan.__name__ = 'TrainBatch', 'TrainRunPlan'
        FakeBatch.__module__ = FakePlan.__module__ = fixture.E0_MODULE.__name__
        self.rejected(lambda: self.adapt(TrainBatch=FakeBatch, TrainRunPlan=FakePlan), 'e0_type_identity_mismatch')
        self.assertEqual(called, [])

    def test_unregistered_module_with_copied_attributes_rejected(self):
        fake = ModuleType(fixture.E0_MODULE.__name__)
        fake.__dict__.update(fixture.E0_MODULE.__dict__)
        self.rejected(lambda: self.adapt(fake), 'e0_verified_module_required')

    def test_module_type_attribute_replacement_rejected(self):
        module = fixture.E0_MODULE
        with patch.object(module, 'TrainBatch', type('TrainBatch', (), {})):
            self.rejected(lambda: self.adapt(module), 'e0_module_binding_mismatch')

    def test_actual_types_from_different_snapshot_not_same_binding(self):
        other = load_e0_module(e0_root=E0_ROOT, e0_sha=E0_SHA)
        self.rejected(lambda: self.adapt(TrainBatch=other.TrainBatch, TrainRunPlan=other.TrainRunPlan), 'e0_type_identity_mismatch')

    def test_legacy_type_only_call_has_no_module_authority(self):
        self.rejected(lambda: adapt_e0(self.artifact, renderer=fixture.RENDERER, e0_sha=E0_SHA,
            TrainBatch=fixture.E0_MODULE.TrainBatch, TrainRunPlan=fixture.E0_MODULE.TrainRunPlan), 'e0_verified_module_required')

    def test_changed_dependency_bytes_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in E0_SOURCE_LF_SHA:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((Path(E0_ROOT) / name).read_bytes())
            (root / 'v3/common/canonical.py').write_bytes(b'not the pinned dependency')
            self.rejected(lambda: load_e0_module(e0_root=root, e0_sha=E0_SHA), 'e0_source_bytes_mismatch')

    def test_returned_arrays_rewritten_to_zero_rejected(self):
        cls = fixture.E0_MODULE.TrainBatch
        original = cls.__init__
        for field in ('input_ids', 'labels'):
            with self.subTest(field=field):
                def corrupted(instance, input_ids, labels):
                    original(instance, input_ids, labels)
                    setattr(instance, field, [0] * len(getattr(instance, field)))
                with patch.object(cls, '__init__', corrupted):
                    self.rejected(lambda: self.adapt(), 'e0_batch_arrays_mismatch')

    def test_missing_truncated_or_nonlist_array_rejected(self):
        cls = fixture.E0_MODULE.TrainBatch
        original = cls.__init__
        for mode in ('missing', 'short', 'tuple'):
            with self.subTest(mode=mode):
                def corrupted(instance, input_ids, labels):
                    original(instance, input_ids, labels)
                    if mode == 'missing': del instance.labels
                    elif mode == 'short': instance.labels = instance.labels[:-1]
                    else: instance.labels = tuple(instance.labels)
                with patch.object(cls, '__init__', corrupted):
                    self.rejected(lambda: self.adapt(), 'e0_batch_arrays_mismatch')

    def test_reported_count_mismatch_rejected_with_correct_arrays(self):
        cls = fixture.E0_MODULE.TrainBatch
        original = cls.__init__
        for wrong in (0, 999, True):
            with self.subTest(wrong=wrong):
                def corrupted(instance, input_ids, labels):
                    original(instance, input_ids, labels)
                    instance.supervised_tokens = wrong
                with patch.object(cls, '__init__', corrupted):
                    self.rejected(lambda: self.adapt(), 'e0_batch_count_mismatch')

    def test_every_plan_config_field_mismatch_rejected(self):
        cls = fixture.E0_MODULE.TrainRunPlan
        original = cls.__init__
        for field in fixture.PLAN:
            with self.subTest(field=field):
                def corrupted(instance, **kwargs):
                    original(instance, **kwargs)
                    setattr(instance, field, getattr(instance, field) + 1)
                with patch.object(cls, '__init__', corrupted):
                    self.rejected(lambda: self.adapt(), 'e0_plan_config_mismatch')

    def test_plan_numeric_type_mismatch_rejected(self):
        cls = fixture.E0_MODULE.TrainRunPlan
        original = cls.__init__
        def corrupted(instance, **kwargs):
            original(instance, **kwargs)
            instance.steps = True  # True == 1 must not bypass exact value/type comparison.
        with patch.object(cls, '__init__', corrupted):
            self.rejected(lambda: self.adapt(), 'e0_plan_config_mismatch')

    def test_plan_missing_or_replaced_batch_rejected(self):
        cls = fixture.E0_MODULE.TrainRunPlan
        original = cls.__init__
        for mode in ('empty', 'tuple', 'replacement'):
            with self.subTest(mode=mode):
                def corrupted(instance, **kwargs):
                    original(instance, **kwargs)
                    if mode == 'empty': instance.batches = []
                    elif mode == 'tuple': instance.batches = tuple(instance.batches)
                    else:
                        expected = self.artifact.data()['batch']
                        instance.batches = [fixture.E0_MODULE.TrainBatch(expected['input_ids'], expected['labels'])]
                with patch.object(cls, '__init__', corrupted):
                    self.rejected(lambda: self.adapt(), 'e0_plan_batches_mismatch')

    def test_plan_constructor_mutating_batch_array_rejected(self):
        cls = fixture.E0_MODULE.TrainRunPlan
        original = cls.__init__
        def corrupted(instance, **kwargs):
            original(instance, **kwargs)
            instance.batches[0].labels[0] = 0
        with patch.object(cls, '__init__', corrupted):
            self.rejected(lambda: self.adapt(), 'e0_batch_arrays_mismatch')

    def test_plan_assertion_mutation_rechecked_before_return(self):
        cls = fixture.E0_MODULE.TrainRunPlan
        original = cls.assert_runnable
        def corrupted(instance):
            original(instance)
            instance.seed += 1
        with patch.object(cls, 'assert_runnable', corrupted):
            self.rejected(lambda: self.adapt(), 'e0_plan_config_mismatch')
