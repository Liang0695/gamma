"""CPU proof with real CT packed weights and the HF post-load hook; no Gemma weights."""
import json
from importlib import metadata

def main():
    for name,version in [('torch','2.10.0'),('transformers','5.17.0'),('peft','0.21.0'),('compressed-tensors','0.15.0.1')]:
        assert metadata.version(name)==version,(name,metadata.version(name))
    import torch
    from transformers import CompressedTensorsConfig
    from transformers.quantizers.auto import AutoHfQuantizer
    from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization import apply_quantization_config
    from peft import LoraConfig,get_peft_model
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.q_proj=torch.nn.Linear(32,16,bias=False)
        def forward(self,x):return self.q_proj(x)
    def packed():
        torch.manual_seed(38);model=Tiny()
        cfg=CompressedTensorsConfig(config_groups={'g':{'targets':['Linear'],'weights':{
            'num_bits':4,'type':'int','strategy':'group','group_size':32,'symmetric':True}}},
            format='pack-quantized',quantization_status='frozen')
        apply_quantization_config(model,cfg.quantization_config,run_compressed=False)
        with torch.no_grad():
            model.q_proj.weight_scale.fill_(0.02)
            if getattr(model.q_proj,'weight_zero_point',None) is not None:model.q_proj.weight_zero_point.zero_()
        compressor=ModelCompressor.from_compression_config(cfg);compressor.compress_model(model)
        assert not hasattr(model.q_proj,'weight')
        assert hasattr(model.q_proj,'weight_packed')
        return model,cfg
    def lora():return LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj'],bias='none')
    model,cfg=packed();old_error=None
    try:get_peft_model(model,lora())
    except AttributeError as exc:old_error=str(exc)
    assert old_error and 'weight' in old_error,old_error
    model,cfg=packed()
    merged=AutoHfQuantizer.merge_quantization_configs(cfg,CompressedTensorsConfig(dequantize=True,use_optimized_inference=False))
    assert merged.dequantize and merged.is_quantization_compressed
    quantizer=CompressedTensorsHfQuantizer(merged)
    quantizer._process_model_after_weight_loading(model)
    assert isinstance(model.q_proj.weight,torch.nn.Parameter) and model.q_proj.weight.is_floating_point()
    base=model.q_proj.weight.detach().clone()
    adapted=get_peft_model(model,lora())
    params=[p for p in adapted.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=0.01)
    before=[p.detach().clone() for p in params]
    loss=adapted(torch.ones(2,32)).square().mean();loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(a,b) for a,b in zip(before,params))
    assert torch.equal(base,model.q_proj.base_layer.weight)
    print(json.dumps({'kind':'real CT/HF/PEFT CPU packed-weight smoke','old_weight_missing_error':old_error,
        'dequantize':merged.dequantize,'weight_present_before_peft':True,'lora_step_passed':True,
        'base_unchanged':True,'Gemma_weights_loaded':False,'GPU_used':False},indent=2))

if __name__=='__main__':main()
