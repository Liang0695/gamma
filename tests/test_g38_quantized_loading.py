import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch
from tests._tmp import temp_dir_outside_repo
from tests.test_g38_lora_scope import ScopeFixture
from v3.train import runner,targets
from v3.common.errors import PolicyViolation

class QuantizedLoadingTests(unittest.TestCase):
    def test_linear_without_weight_is_rejected_before_peft(self):
        model=ScopeFixture();layer=model.model.language_model.layers[0].self_attn.q_proj
        del layer.weight
        layer.register_buffer('weight_packed',torch.zeros(1,dtype=torch.int32))
        self.assertIsInstance(layer,torch.nn.Linear)
        with self.assertRaisesRegex(PolicyViolation,'lora_text_weight_unavailable'):
            targets.resolve_peft_scope(model,['q_proj','o_proj'],torch.nn.Linear)
        for weight in (torch.empty(4,4,device='meta'),torch.zeros(4,4,dtype=torch.int32)):
            layer.weight=torch.nn.Parameter(weight,requires_grad=False)
            with self.assertRaisesRegex(PolicyViolation,'lora_text_weight_unavailable'):
                targets.resolve_peft_scope(model,['q_proj','o_proj'],torch.nn.Linear)
    def test_loading_attributes_are_explicit_and_config_bytes_unchanged(self):
        with temp_dir_outside_repo('ct_load_') as root:
            config=Path(root)/'config.json'
            config.write_text(json.dumps({'quantization_config':{'quant_method':'compressed-tensors','format':'pack-quantized'}}))
            before=config.read_bytes()
            class Config:
                def __init__(self,**kw):self.kw=kw;self.__dict__.update(kw)
            fake=types.ModuleType('transformers');fake.CompressedTensorsConfig=Config
            backend=runner.TorchPeftBackend(model_id='fixture',revision='fixed',target_modules=['q_proj','o_proj'],device='cuda:0')
            with patch.dict(sys.modules,{'transformers':fake}):options=backend._training_load_options(root)
            self.assertEqual(options['quantization_config'].kw,{'dequantize':True,'use_optimized_inference':False})
            self.assertEqual(options['device_map'],{'':'cuda:0'})
            self.assertEqual(config.read_bytes(),before)

if __name__=='__main__':unittest.main()
