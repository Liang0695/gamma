"""Actual PEFT + production run_training on a CPU-only tiny model loader."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dest-dir',required=True)
    dest=Path(parser.parse_args().dest_dir).resolve()
    if dest==ROOT or ROOT in dest.parents or (dest.exists() and any(dest.iterdir())):
        raise ValueError('require a fresh directory outside source')
    for name,version in (('peft','0.21.0'),('transformers','5.17.0'),('torch','2.10.0')):
        if metadata.version(name)!=version:raise ValueError('wrong version: '+name)
    import torch,peft
    from transformers import PretrainedConfig
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    from tests.test_g38_lora_scope import ScopeFixture
    from v3.train import runner,targets
    from v3.train.lifecycle import TrainingLifecycle,LifecyclePolicy

    def build():
        torch.manual_seed(38)
        return ScopeFixture(clippable=lambda:Gemma4ClippableLinear(Gemma4VisionConfig(use_clipped_linears=True),4,4),
                            config=PretrainedConfig(vocab_size=16,tie_word_embeddings=False))
    class Backend(runner.TorchPeftBackend):
        requires_gpu=False
        def __init__(self):
            super().__init__(model_id='CPU/fixture',revision='tiny',target_modules=['q_proj','o_proj'],device='cpu',
                model_inputs={'repo_id':'CPU/fixture','revision':'tiny'})
        def prepare(self,plan):
            base=build()
            for p in base.parameters():p.requires_grad_(False)
            scope=targets.resolve_peft_scope(base,self.target_modules,torch.nn.Linear)
            config=peft.LoraConfig(r=2,lora_alpha=4,lora_dropout=0,bias='none',task_type='CAUSAL_LM',target_modules=scope['target_regex'])
            self.model=peft.get_peft_model(base,config);targets.verify_peft_scope(base,scope)
            self._torch=torch;self.lr=plan.lr;self.load_count=1
            self.sampler_order_sha256=runner._sampler_order_sha256(plan)
            self.parameter_ownership=self.parameter_ownership_snapshot()
            return {'parameter_ownership':self.parameter_ownership,'unverified_claims':['tiny CPU loader, not Gemma weights']}
        def _import_stack(self):
            return (torch,peft,peft.LoraConfig,peft.get_peft_model,
                    types.SimpleNamespace(from_pretrained=lambda *a,**kw:build()),None)
    plan=runner.TrainRunPlan(steps=1,lr=0.01,seq_len=4,lora_rank=2,lora_alpha=4,
                            batches=[runner.TrainBatch([1,2,3,4],[1,2,3,4])])
    hashes={k:'a'*64 for k in ('source_sha256','data_sha256','config_sha256','code_sha256','deps_sha256')}
    life=TrainingLifecycle(store_root=str(dest/'checkpoints'),hashes=hashes,policy=LifecyclePolicy(checkpoint_every=0))
    backend=Backend()
    report=runner.run_training(backend,plan,str(dest/'adapter'),lifecycle=life)
    checkpoint=life.checkpoints[-1]['checkpoint_dir']
    digest=backend.adapter_params_digest()
    assert backend._optimizer.state and report['reloaded']['training_state_restored']['optimizer_state_restored']
    active={id(p) for p in backend.model.parameters() if p.requires_grad}
    assert {id(p) for group in backend._optimizer.param_groups for p in group['params']}==active
    assert all(float(state['step'])==1 for state in backend._optimizer.state.values())
    backend.load_adapter(str(dest/'adapter'),is_trainable=False)
    assert backend.adapter_params_digest()==digest
    assert all(not p.requires_grad for p in backend.model.parameters())
    # Release the first fixture before constructing the resume fixture, too.
    backend.release_base('CPU fresh-resume control');backend.assert_base_released()
    fresh=Backend();fresh.prepare(plan)
    resumed=TrainingLifecycle(store_root=str(dest/'resumed'),hashes=hashes,resume_from=checkpoint)
    restored=resumed.begin(fresh)
    assert restored['backend_import']['checkpoint_adapter_restored']
    assert fresh.adapter_params_digest()==digest
    step=fresh.train_steps(plan)
    assert step['steps_executed']==1 and fresh.global_step==2
    assert all(float(state['step'])==2 for state in fresh._optimizer.state.values())
    result={'kind':'actual PEFT CPU production lifecycle smoke','run_training_executed':report['executed'],
        'final_checkpoint_before_release':True,'frozen_reload_digest_equal':True,
        'adapter_and_moments_restored':True,'new_parameter_ownership_verified':True,
        'resumed_global_step':fresh.global_step,'loaded_Gemma_weights':False,'gpu_used':False}
    (dest/'lifecycle-smoke.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
