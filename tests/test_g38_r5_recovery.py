"""CPU fixtures for r4 independent findings. No production approval is issued."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import unittest
import uuid
from unittest.mock import patch

from tests._tmp import temp_dir_outside_repo
from v3.train import runner, input_binding, v6_approval as approval
from v3.train.engineering_check import release_before_resume, _plan
from v3.common.errors import PolicyViolation, MissingInput

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/v3/evidence/kaggle-38-g9-evidence'
LOCK = ROOT / 'v3/locks/official-interface.json'
LAYOUT = {name: name for name in input_binding.REQUIRED_INPUT_FILES}


def binding():
    return input_binding.bind_model_inputs(root=str(SOURCE), interface_path=str(LOCK), layout=LAYOUT)


def backend(root, device='cpu'):
    bound = binding()
    model = Path(root) / 'model'
    model.mkdir(exist_ok=True)
    shutil.copyfile(SOURCE / 'config.json', model / 'config.json')
    bound['approved_local_model_dir'] = str(model.resolve())
    return runner.TorchPeftBackend(model_id=bound['repo_id'], revision=bound['revision'],
        target_modules=['q_proj'], device=device, model_inputs=bound, local_model_dir=str(model))


class Model:
    def to(self, *_): return self
    def named_parameters(self): return []
    def parameters(self): return []


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        # Historical CPU object-binding fixture; real environment checks have
        # separate positive/negative tests in test_g38_publication_environment.
        fixture = patch('v3.train.runtime_environment.verify_deployed_environment',
                        return_value={'kind':'CPU-fixture'})
        fixture.start()
        self.addCleanup(fixture.stop)
    def test_live_external_reference_blocks_reload_and_resume(self):
        with temp_dir_outside_repo('recovery_') as root:
            obj = backend(root)
            old = Model()
            obj.model = old
            obj._optimizer = types.SimpleNamespace(model=old)
            obj._torch = types.SimpleNamespace(bfloat16='fixture')
            obj.adapter_params_digest = lambda: 'same'
            carrier = Path(root) / runner.adapter_contract.adapter_dir_relative_path('v3_policy')
            carrier.mkdir(parents=True)
            (carrier / runner.ADAPTER_FILE).write_bytes(b'fixture')
            auto = types.SimpleNamespace(from_pretrained=unittest.mock.Mock(return_value=Model()))
            peft = types.ModuleType('peft')
            peft.PeftModel = types.SimpleNamespace(from_pretrained=lambda *_, **kwargs: Model())
            with patch.object(obj, '_import_stack', return_value=(obj._torch,None,None,None,auto,None)), \
                 patch.object(obj, '_training_load_options', return_value={}), \
                 patch.object(runner.adapter_contract, 'assert_official_adapter_carrier'), \
                 patch.dict(sys.modules, {'peft': peft}):
                for _ in range(2):
                    with self.assertRaisesRegex(PolicyViolation, 'old_base_still_alive'):
                        obj.load_adapter(root)
                auto.from_pretrained.assert_not_called()
                with self.assertRaisesRegex(PolicyViolation, 'old_base_still_alive'):
                    release_before_resume(backend_impl=obj, profiler_backend=obj, training={})
                del old
                result = obj.load_adapter(root)
                self.assertEqual(auto.from_pretrained.call_count, 1)
                self.assertFalse(result['old_base_alive_during_fresh_load'])
                self.assertTrue(auto.from_pretrained.call_args.kwargs['local_files_only'])

    def test_resume_drops_training_model_reference_before_observing(self):
        with temp_dir_outside_repo('release_') as root:
            obj = backend(root)
            obj.model = Model()
            training = {'model': obj.model}
            result = release_before_resume(backend_impl=obj, profiler_backend=obj, training=training)
            self.assertTrue(result['only_one_base_resident'])
            self.assertEqual(training, {})

    def test_weight_directory_config_and_path_are_both_checked_before_load(self):
        with temp_dir_outside_repo('source_') as root:
            obj = backend(root)
            self.assertEqual(obj._offline_load_dir(), str((Path(root)/'model').resolve()))
            other = Path(root)/'other'; other.mkdir()
            shutil.copyfile(SOURCE/'config.json', other/'config.json')
            obj.local_model_dir = str(other)
            with self.assertRaisesRegex(PolicyViolation, 'local_model_source_unapproved'):
                obj._offline_load_dir()
            obj.model_inputs['approved_local_model_dir'] = str(other)
            (other/'config.json').write_text('{"model_type":"different"}')
            auto = unittest.mock.Mock()
            with patch.object(obj, '_import_stack', auto):
                with self.assertRaisesRegex(PolicyViolation, 'local_model_config_mismatch'):
                    obj.prepare(_plan(1))
                auto.assert_not_called()

    def test_cuda_requirement_survives_device_change_and_missing_api(self):
        with temp_dir_outside_repo('cuda_') as root:
            obj = backend(root, 'cuda:0')
            obj.device = 'cpu'
            obj.cuda_rng = types.SimpleNamespace(is_available=lambda: False)
            with self.assertRaisesRegex(MissingInput, 'cuda_rng_capture_failed'):
                obj._cuda_rng_states()
            with self.assertRaisesRegex(MissingInput, 'cuda_rng_missing_in_checkpoint'):
                obj._restore_cuda_rng_states({'available':False,'devices':{}})
            cpu = backend(root)
            self.assertFalse(cpu._cuda_rng_states()['available'])
            with self.assertRaisesRegex(MissingInput, 'cuda_rng_unavailable_for_restore'):
                cpu._restore_cuda_rng_states({'available':True,'cuda_in_use':True,
                    'training_devices':['cuda:0'],'device_count':1,'devices':{}})

    def test_measured_objects_and_published_anchor_are_required(self):
        with temp_dir_outside_repo('approval_'+uuid.uuid4().hex+'_') as root:
            root = Path(root)
            # A clean, local Git fixture measures an actual HEAD, not the old E0.
            (root/'v3/locks').mkdir(parents=True); (root/'tools').mkdir()
            (root/'docs/v3/evidence').mkdir(parents=True)
            shutil.copyfile(ROOT/'docs/v3/evidence/kaggle-38-train-stack-manifest.json',
                            root/'docs/v3/evidence/kaggle-38-train-stack-manifest.json')
            (root/'v3/code.py').write_text('fixture = True\n')
            for name in ('train.lock.json','official-interface.json'):
                shutil.copyfile(ROOT/'v3/locks'/name, root/'v3/locks'/name)
            for args in (['init'], ['add','.'], ['-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-m','CPU fixture']):
                subprocess.run(['git','-C',str(root),*args], check=True, capture_output=True)
            options = approval.runtime_options(steps=2, checkpoint_every=1, stop_at_step=1,
                budget_seconds=30, requested_gpu_hours=0.01, model_id=binding()['repo_id'],
                model_revision=binding()['revision'], target_modules=['q_proj'])
            plan = _plan(2)
            context = approval.measure_runtime_context(plan=plan, input_binding=binding(),
                local_model_dir=str(SOURCE), options=options, repo_root=root)
            self.assertNotEqual(context['target_sha'], approval.LEGACY_E0_FIXTURE_COMMIT)
            with temp_dir_outside_repo('record_') as records:
                evidence = Path(records)/'evidence'; evidence.write_bytes(b'CPU fixture only')
                record = Path(records)/'approval.json'
                payload = dict(approval_id='CPU-FIXTURE', target_sha=context['target_sha'],
                    gpu_hours_approved=0.01, verified_by='fixture', verified_utc='fixture',
                    evidence_sha256=approval.sha256_file(evidence), runtime_context=context)
                record.write_text(json.dumps(payload), encoding='utf-8')
                anchor = approval.sha256_file(record)
                kwargs = dict(approval_path=str(record), expected_sha256=anchor,
                    evidence_ref_path=str(evidence), requested_gpu_hours=0.01, runtime_context=context)
                with self.assertRaisesRegex(PolicyViolation, 'approval_not_trusted'):
                    approval.require_approval(**kwargs)
                # Explicit injected deployment fixture; never a CLI approval.
                result = approval.require_approval(**kwargs, published_anchor=anchor)
                self.assertEqual(result['target_sha'], context['target_sha'])
                for key in ('code_sha256','data_sha256','source_sha256','config_sha256','deps_sha256'):
                    changed = copy.deepcopy(context); changed['hashes'][key] = '0'*64
                    with self.subTest(key=key), self.assertRaisesRegex(PolicyViolation, 'approval_runtime_mismatch'):
                        approval.require_approval(**{**kwargs,'runtime_context':changed},published_anchor=anchor)
                plan.batches[0].labels[-1] += 1
                changed = approval.measure_runtime_context(plan=plan,input_binding=binding(),
                    local_model_dir=str(SOURCE),options=options,repo_root=root)
                self.assertNotEqual(context['hashes']['data_sha256'],changed['hashes']['data_sha256'])
                with self.assertRaisesRegex(PolicyViolation, 'approval_runtime_mismatch'):
                    approval.require_approval(**{**kwargs,'runtime_context':changed},published_anchor=anchor)
                with self.assertRaisesRegex(PolicyViolation, 'approval_not_trusted'):
                    approval.require_approval(**{**kwargs,'requested_gpu_hours':1},published_anchor=anchor)
            (root/'v3/code.py').write_text('fixture = False\n')
            with self.assertRaisesRegex(PolicyViolation, 'runtime_code_unbound'):
                approval.measure_runtime_context(plan=plan,input_binding=binding(),
                    local_model_dir=str(SOURCE),options=options,repo_root=root)


if __name__ == '__main__':
    unittest.main()
