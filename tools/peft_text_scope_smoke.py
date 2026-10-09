"""Real PEFT CPU smoke on a tiny model; no hub calls, pretrained weights or GPU."""
import argparse
import json
from importlib import metadata
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest-dir',required=True)
    args=parser.parse_args()
    dest=Path(args.dest_dir).resolve()
    if dest==ROOT or ROOT in dest.parents:raise ValueError('output must be outside source checkout')
    for name,version in (('peft','0.21.0'),('transformers','5.17.0'),('torch','2.10.0')):
        if metadata.version(name)!=version:raise ValueError('wrong installed version: '+name)
    import torch
    from peft import LoraConfig,PeftModel,get_peft_model
    from transformers import PretrainedConfig
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    from tests.test_g38_lora_scope import ScopeFixture
    from v3.train.targets import resolve_peft_scope,verify_peft_scope

    def build():
        torch.manual_seed(38)
        model=ScopeFixture(clippable=lambda:Gemma4ClippableLinear(Gemma4VisionConfig(use_clipped_linears=True),4,4),
                           config=PretrainedConfig(vocab_size=16,tie_word_embeddings=False))
        for p in model.parameters():p.requires_grad_(False)
        return model
    def config(target):
        return LoraConfig(r=2,lora_alpha=4,lora_dropout=0,bias='none',task_type='CAUSAL_LM',target_modules=target)
    # Exact old failure, using actual installed PEFT and Transformers wrapper.
    old_error=None
    try:get_peft_model(build(),config(['q_proj','o_proj']))
    except ValueError as exc:old_error=str(exc)
    assert old_error and 'Gemma4ClippableLinear' in old_error and 'not supported' in old_error
    base=build();scope=resolve_peft_scope(base,['q_proj','o_proj'],torch.nn.Linear)
    frozen={n:p.detach().clone() for n,p in base.named_parameters()}
    model=get_peft_model(base,config(scope['target_regex']))
    mounted=verify_peft_scope(base,scope)
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.01)
    ids=torch.tensor([[1,2,3,4]])
    loss=model(input_ids=ids,labels=ids).loss
    loss.backward();optimizer.step();optimizer.zero_grad()
    assert torch.isfinite(loss)
    assert any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if p.requires_grad)
    for name,old in frozen.items():
        prefix,_,leaf=name.rpartition('.')
        current=dict(base.named_parameters()).get(name)
        if current is None:current=dict(base.named_parameters())[prefix+'.base_layer.'+leaf]
        assert torch.equal(old,current),name
    model.eval();expected=model(input_ids=ids).logits.detach()
    if dest.exists() and any(dest.iterdir()):raise ValueError('output directory must be fresh/empty')
    dest.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(str(dest),save_embedding_layers=False)
    saved=json.loads((dest/'adapter_config.json').read_text())
    assert saved['target_modules']==scope['target_regex']
    reloaded=PeftModel.from_pretrained(build(),str(dest),is_trainable=False)
    reloaded.eval()
    assert torch.allclose(expected,reloaded(input_ids=ids).logits,atol=1e-6,rtol=1e-5)
    result={'kind':'tiny CPU actual-PEFT smoke','old_suffix_failure':old_error,
        'target_regex':scope['target_regex'],'mounted_count':len(mounted),
        'excluded_suffix_matches':scope['excluded_suffix_matches'],
        'forward_backward_optimizer':True,'base_and_towers_unchanged':True,
        'adapter_changed':True,'save_reload_equal':True,'loss':float(loss.detach()),
        'loaded_pretrained_weights':False,'gpu_used':False}
    (dest/'smoke-result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
