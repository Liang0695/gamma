"""Small CPU model preserving the text/vision target-name collision."""
import re
import types
import unittest
import torch
from v3.train.targets import resolve_peft_scope, verify_peft_scope, TARGET_REGEX
from v3.common.errors import MissingInput, PolicyViolation


class ClippableFixture(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(4,4,bias=False)
    def forward(self,x):return self.linear(x.clamp(-1,1)).clamp(-1,1)


class Attention(torch.nn.Module):
    def __init__(self,factory):
        super().__init__();self.q_proj=factory();self.o_proj=factory()
    def forward(self,x):return x+0.01*self.o_proj(self.q_proj(x))


class Block(torch.nn.Module):
    def __init__(self,factory):
        super().__init__();self.self_attn=Attention(factory)


class ScopeFixture(torch.nn.Module):
    def __init__(self,clippable=ClippableFixture,layers=60,config=None):
        super().__init__()
        self.model=torch.nn.Module()
        self.model.vision_tower=Block(clippable)
        self.model.audio_tower=Block(clippable)
        self.model.language_model=torch.nn.Module()
        self.model.language_model.layers=torch.nn.ModuleList(
            [Block(lambda:torch.nn.Linear(4,4,bias=False)) for _ in range(layers)])
        self.model.language_model.embed_tokens=torch.nn.Embedding(16,4)
        self.lm_head=torch.nn.Linear(4,16,bias=False)
        self.config=config
    def get_input_embeddings(self):return self.model.language_model.embed_tokens
    def prepare_inputs_for_generation(self,input_ids,**kwargs):return {'input_ids':input_ids,**kwargs}
    def forward(self,input_ids=None,inputs_embeds=None,labels=None,**kwargs):
        x=self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.model.language_model.layers:x=layer.self_attn(x)
        logits=self.lm_head(x)
        loss=None if labels is None else torch.nn.functional.cross_entropy(logits.reshape(-1,16),labels.reshape(-1))
        return types.SimpleNamespace(loss=loss,logits=logits)


class ScopeTests(unittest.TestCase):
    def test_suffix_collision_is_excluded_by_actual_peft_regex(self):
        model=ScopeFixture();modules=dict(model.named_modules())
        suffix=[n for n in modules if n.endswith(('.q_proj','.o_proj'))]
        self.assertEqual(len(suffix),124)
        self.assertTrue(any(isinstance(modules[n],ClippableFixture) for n in suffix))
        scope=resolve_peft_scope(model,['q_proj','o_proj'],torch.nn.Linear)
        self.assertEqual(len(scope['matched']),120)
        self.assertEqual(len(scope['excluded_suffix_matches']),4)
        self.assertEqual([n for n in modules if re.fullmatch(scope['target_regex'],n)],
                         [n for n in modules if n in scope['matched']])
        self.assertTrue(all(isinstance(modules[n],torch.nn.Linear) for n in scope['matched']))
        self.assertIsInstance(model.model.vision_tower.self_attn.q_proj,ClippableFixture)

    def test_missing_text_layer_and_extra_layer_are_rejected(self):
        for layers in (59,61):
            with self.subTest(layers=layers),self.assertRaises(MissingInput):
                resolve_peft_scope(ScopeFixture(layers=layers),['q_proj','o_proj'],torch.nn.Linear)

    def test_wrapper_inside_text_is_not_silently_unwrapped(self):
        model=ScopeFixture();model.model.language_model.layers[0].self_attn.q_proj=ClippableFixture()
        with self.assertRaisesRegex(PolicyViolation,'lora_text_target_unsupported'):
            resolve_peft_scope(model,['q_proj','o_proj'],torch.nn.Linear)

    def test_scope_request_cannot_expand_and_partial_mount_is_rejected(self):
        model=ScopeFixture()
        with self.assertRaises(PolicyViolation):resolve_peft_scope(model,['q_proj','o_proj','k_proj'],torch.nn.Linear)
        scope=resolve_peft_scope(model,['q_proj','o_proj'],torch.nn.Linear)
        with self.assertRaisesRegex(PolicyViolation,'lora_mounted_scope_mismatch'):
            verify_peft_scope(model,scope)


if __name__=='__main__':unittest.main()
