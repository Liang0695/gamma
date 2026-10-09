"""Actual runner/lifecycle/CPU AdamW, with a loader matching PEFT defaults."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch
import torch._dynamo  # Initialize lazy optimizer imports before patch.dict(sys.modules).

from tests._tmp import temp_dir_outside_repo
from v3.train import runner
from v3.train.lifecycle import TrainingLifecycle, LifecyclePolicy
from v3.common.errors import PolicyViolation


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(17)
        self.base_layer=torch.nn.Linear(4,4,bias=False)
        self.lora_A=torch.nn.Linear(4,2,bias=False)
        self.lora_B=torch.nn.Linear(2,4,bias=False)
        self.base_layer.requires_grad_(False)
    def forward(self,input_ids,labels):
        x=torch.nn.functional.one_hot(input_ids.reshape(-1)%4,4).float()
        logits=self.base_layer(x)+self.lora_B(self.lora_A(x))
        loss=torch.nn.functional.cross_entropy(logits,labels.reshape(-1)%4)
        return types.SimpleNamespace(loss=loss)


class CPUBackend(runner.TorchPeftBackend):
    def __init__(self):
        super().__init__(model_id='fixture/model',revision='fixed',target_modules=['q_proj','o_proj'],
            model_inputs={'repo_id':'fixture/model','revision':'fixed'},device='cpu')
        self.saved=None;self.events=[];self.life=None
    def prepare(self,plan):
        self.model=TinyAdapter();self._torch=torch;self.lr=plan.lr
        self.sampler_order_sha256=runner._sampler_order_sha256(plan)
        self.load_count=1
        self.parameter_ownership=self.parameter_ownership_snapshot()
        return {'unverified_claims':['CPU loader fixture'],'parameter_ownership':self.parameter_ownership}
    def save_adapter(self,dest_dir,*,adapter_name=None):
        self.saved={n:p.detach().clone() for n,p in self.model.named_parameters()}
        carrier=Path(dest_dir)/runner.adapter_contract.adapter_dir_relative_path(adapter_name or 'v3_policy')
        carrier.mkdir(parents=True,exist_ok=True);(carrier/runner.ADAPTER_FILE).write_bytes(b'CPU fixture')
        self.events.append('adapter_saved')
        return {'adapter_only':True,'contains_base_weights':False}
    def release_base(self,reason):
        if self.life is not None:
            assert self.life.checkpoints, 'final checkpoint must exist BEFORE release'
            state=self.life.store.load(self.life.checkpoints[-1]['checkpoint_dir'])
            assert state.optimizer['state'], 'trained AdamW moments must be saved'
            self.events.append('checkpoint_verified_before_release')
        return super().release_base(reason)
    def _import_stack(self):
        return (torch,None,None,None,types.SimpleNamespace(from_pretrained=lambda *a,**k: TinyAdapter()),None)


class ReloadTests(unittest.TestCase):
    def plan(self,steps=1):
        return runner.TrainRunPlan(steps=steps,lr=0.01,seq_len=4,lora_rank=2,lora_alpha=4,
                                  batches=[runner.TrainBatch([1,2,3,0],[1,2,3,0])])
    def loader(self,backend,calls):
        def load(base,path,is_trainable=False,**kwargs):
            calls.append(is_trainable)
            for name,p in base.named_parameters():
                with torch.no_grad():p.copy_(backend.saved[name])
                p.requires_grad_(is_trainable and any(x in ('lora_A','lora_B') for x in name.split('.')))
            return base
        module=types.ModuleType('peft');module.PeftModel=types.SimpleNamespace(from_pretrained=load)
        return module

    def test_digest_uses_adapter_identity_and_tensor_values_not_gradient_flags_or_sum(self):
        backend=CPUBackend();backend.prepare(self.plan())
        before=backend.adapter_params_digest();base=backend.base_params_digest()
        backend.model.requires_grad_(False)
        self.assertEqual(before,backend.adapter_params_digest())
        self.assertEqual(base,backend.base_params_digest())
        with torch.no_grad():
            weight=backend.model.lora_A.weight
            weight[0,0]+=0.125;weight[0,1]-=0.125
        self.assertNotEqual(before,backend.adapter_params_digest())

    def test_production_training_finalizes_before_reload_and_restores_moments_to_new_params(self):
        with temp_dir_outside_repo('reload_lifecycle_') as root:
            backend=CPUBackend();calls=[]
            hashes={k:'a'*64 for k in ('source_sha256','data_sha256','config_sha256','code_sha256','deps_sha256')}
            life=TrainingLifecycle(store_root=str(Path(root)/'ckpt'),hashes=hashes,
                                   policy=LifecyclePolicy(checkpoint_every=0))
            backend.life=life
            with patch.dict(sys.modules,{'peft':self.loader(backend,calls)}), \
                 patch.object(runner.adapter_contract,'assert_official_adapter_carrier'):
                result=runner.run_training(backend,self.plan(),str(Path(root)/'adapter'),lifecycle=life)
                self.assertTrue(result['executed'])
                self.assertEqual(calls,[True])
                self.assertTrue(result['reloaded']['training_state_restored']['optimizer_state_restored'])
                backend.load_adapter(str(Path(root)/'adapter'),is_trainable=False)
                self.assertTrue(all(not p.requires_grad for p in backend.model.parameters()))
                self.assertIsNone(backend._optimizer)
                state=life.store.load(life.checkpoints[-1]['checkpoint_dir'])
                self.assertEqual(state.global_step,1)
                self.assertTrue(state.optimizer['state'])
                self.assertIn('checkpoint_verified_before_release',backend.events)
                frozen_digest=backend.adapter_params_digest()
                backend.load_adapter(str(Path(root)/'adapter'),is_trainable=True)
                self.assertEqual(calls,[True,False,True])
                self.assertEqual(backend.adapter_params_digest(),frozen_digest)
                restored=backend.import_training_state(state)
                self.assertTrue(restored['optimizer_state_restored'])
                params=[p for p in backend.model.parameters() if p.requires_grad]
                optimizer_params=[p for g in backend._optimizer.param_groups for p in g['params']]
                self.assertEqual({id(p) for p in params},{id(p) for p in optimizer_params})
                self.assertTrue(all(p in backend._optimizer.state for p in params))
                self.assertTrue(all(float(s['step'])==1 for s in backend._optimizer.state.values()))
                self.assertTrue(any(torch.count_nonzero(s['exp_avg']).item()>0 for s in backend._optimizer.state.values()))
                trained=backend.train_steps(self.plan(steps=1))
                self.assertEqual(trained['steps_executed'],1)
                self.assertTrue(all(float(s['step'])==2 for s in backend._optimizer.state.values()))
                self.assertEqual(backend.global_step,2)
                self.assertNotEqual(backend.adapter_params_digest(),frozen_digest)
                self.assertFalse(backend.model.base_layer.weight.requires_grad)
                # The actual lifecycle.begin path starts from a fresh model,
                # not from our adapter loader. Checkpoint bytes must restore it.
                fresh=CPUBackend();fresh.prepare(self.plan())
                self.assertNotEqual(fresh.adapter_params_digest(),frozen_digest)
                resumed_life=TrainingLifecycle(store_root=str(Path(root)/'resume'),hashes=hashes,
                                              resume_from=state.checkpoint_dir)
                resumed=resumed_life.begin(fresh)
                self.assertTrue(resumed['backend_import']['checkpoint_adapter_restored'])
                self.assertEqual(fresh.adapter_params_digest(),frozen_digest)
                self.assertTrue(all(float(s['step'])==1 for s in fresh._optimizer.state.values()))

    def test_optional_state_roundtrip_and_hash_protection(self):
        import json
        from v3.train import checkpoint
        from v3.common.errors import IntegrityError
        with temp_dir_outside_repo('optional_state_') as root:
            backend=CPUBackend();backend.prepare(self.plan());backend.ensure_scheduler(self.plan())
            backend._scaler=torch.amp.GradScaler('cpu',init_scale=256)
            backend.train_steps(self.plan())
            hashes={k:'a'*64 for k in ('source_sha256','data_sha256','config_sha256','code_sha256','deps_sha256')}
            life=TrainingLifecycle(store_root=root,hashes=hashes)
            record=life.save_checkpoint(backend,step=1,reason='CPU optional-state control')
            state=life.store.load(record['checkpoint_dir'])
            self.assertEqual(state.scheduler,backend._scheduler.state_dict())
            self.assertEqual(state.scaler,backend._scaler.state_dict())
            fresh=CPUBackend();fresh.prepare(self.plan());fresh.ensure_scheduler(self.plan())
            fresh._scaler=torch.amp.GradScaler('cpu',init_scale=1)
            fresh.import_training_state(state)
            self.assertEqual(fresh._scheduler.state_dict(),backend._scheduler.state_dict())
            self.assertEqual(fresh._scaler.get_scale(),256)
            path=Path(record['checkpoint_dir'])/checkpoint.OPTIONAL_STATE_NAME
            path.write_text(json.dumps({'scheduler':None,'scaler':None}),encoding='utf-8')
            with self.assertRaises(IntegrityError):life.store.load(record['checkpoint_dir'])

    def test_suppressed_final_save_cannot_release_trained_optimizer(self):
        with temp_dir_outside_repo('no_final_save_') as root:
            backend=CPUBackend()
            hashes={k:'a'*64 for k in ('source_sha256','data_sha256','config_sha256','code_sha256','deps_sha256')}
            life=TrainingLifecycle(store_root=str(Path(root)/'ckpt'),hashes=hashes,
                policy=LifecyclePolicy(checkpoint_every=0,stop_at_step=1),save_on_stop=False)
            with self.assertRaisesRegex(PolicyViolation,'reload_requires_final_checkpoint'):
                runner.run_training(backend,self.plan(),str(Path(root)/'adapter'),lifecycle=life)
            self.assertTrue(backend._optimizer.state)
            self.assertIsNotNone(backend.model)


if __name__=='__main__':unittest.main()
