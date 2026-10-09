# r10: preserve the approved text-tower scope at PEFT injection

Job 88918 used r9 Git `55de96d49556a738d49fd52eb1e0e67c9b602c57` and reached an A100. It loaded weights, then failed before any optimizer step because PEFT was asked to adapt a `Gemma4ClippableLinear`. No checkpoint or adapter was expected to exist after that failure.

The repository already defines `TARGET_REGEX = ^model\.language_model\.layers\.\d+\.self_attn\.(q_proj|o_proj)$`, 60 text layers / 120 modules, with vision/projectors frozen. The execution bug was passing only the derived suffix list `q_proj,o_proj` into PEFT: suffix matching also selected vision/audio modules. Unwrapping those towers would make an unintended target injectable rather than restore the agreed scope.

Fixed-version primary sources:

- [Transformers v5.17.0 Gemma4 implementation](https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/models/gemma4/modeling_gemma4.py): vision/audio use the clippable wrapper; text attention q/o use ordinary Linear. The pinned model config declares 60 text layers and Gemma4ForConditionalGeneration.
- [PEFT v0.21.0 matching implementation](https://raw.githubusercontent.com/huggingface/peft/v0.21.0/src/peft/tuners/tuners_utils.py): a list permits suffix matches; a string is a regular-expression target.

## Minimal correction

Before injection, resolve the existing full text regex against actual named modules and require exactly the intended 120 paths and supported Linear types. Keep the logical request restricted to q/o. Pass the full regex to LoraConfig; do not unwrap or modify vision/audio wrappers. After PEFT injection, verify exactly those paths are mounted and no parameters outside their LoRA A/B matrices are trainable. Record the resolved target names, types, excluded suffix collisions and actual mounted names in preparation evidence.

The saved PEFT adapter configuration keeps the full regex, so `PeftModel.from_pretrained` uses the same scope during reload. This does not change serving or adapter-carrier contracts. If text targets have a different architecture or are themselves unsupported wrappers, fail with a specific error instead of silently changing architecture or widening targets.

## Verification

```text
python -m unittest tests.test_g38_lora_scope -v
python run_tests.py
```

Local CPU tests construct all 60 tiny text layers plus vision/audio clippable fixtures. The old suffix rule selects 124 modules including four unsupported wrappers; the full regex selects only 120 supported text targets. Missing/extra layers, unsupported text wrappers, scope expansion and partial injection are rejected. These tests validate resolution/guards, not an installed PEFT execution.

The author environment has no PEFT/Transformers installation and does not install a new training stack. The included target-side CPU smoke uses **the existing** PEFT 0.21.0 / Transformers 5.17.0 / torch2.10.0, actual Gemma4ClippableLinear objects with width4, and a tiny 60-layer text fixture. It reproduces the old PEFT failure, injects the corrected regex, checks frozen base/towers, performs one CPU optimizer step, saves and reloads the adapter and compares outputs. No model download or Gemma weights are used.

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/scc/pb24511961/v3/envs/py3119/bin/python -I /ABS/R10_SOURCE/tools/peft_text_scope_smoke.py --dest-dir /ABS/OUTSIDE/fresh-r10-cpu-smoke
```

The smoke fails on missing/wrong packages rather than reporting a skipped success. Output must be outside the source tree and fresh. This script is for Liang's CPU verification, not a new GPU job or production approval. Its actual PEFT result remains pending until the target operator returns evidence.

## Execution/accounting decision

The one-submission approval `KAGGLE-38-r9-first-gpu-20261009-01` is consumed by 88918 and cannot be reused. Controller post-run data reports one allocated GPU and RunTime114s (~0.0317 GPU-h), but sacct has no matching rows and no RUNNING collector snapshot was captured. Retain the conservative 0.25 GPU-h debit; do not finalize/refund the remainder based on empty accounting. Preserve the runtime observation separately and leave the existing ledger finalization rules unchanged.

Independent code review and the actual tiny PEFT smoke precede any new code intake/context/approval. No automatic resubmission, new GPU allowance, dependency upgrade or repeated weight hash is authorized here.
