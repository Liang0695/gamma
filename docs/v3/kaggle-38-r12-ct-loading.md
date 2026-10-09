# r12: materialize compressed-tensors weights before PEFT

Job89019 reached A100 but PEFT failed on an nn.Linear without weight. The raw log says **Compressing model**, then Loading weights; it does not establish that weights were dequantized before injection. Class identity alone was insufficient evidence of PEFT readiness. The text-only120 target scope remains correct and is retained.

Fixed-version primary sources:

- [Transformers5.17.0 CompressedTensorsConfig](https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/utils/quantization_config.py) defaults dequantize=False; dequantize=True requests load-time restoration for fine-tuning.
- [The fixed quantizer](https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/quantizers/quantizer_compressed_tensors.py) calls compressor.decompress_model after loading only when that flag is set.
- [The fixed config merge](https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/quantizers/auto.py) preserves the checkpoint scheme and overrides only loading attributes from the supplied CompressedTensorsConfig.

The author also inspected compressed-tensors0.15.0.1 source from the exact SHA-locked194260-byte wheel, without installing it. Its compressor removes dense weight while compressed and restores it on decompression; the automatic hook waits for forward, too late for PEFT construction.

## Change

Both initial load and controlled reload pass CompressedTensorsConfig(dequantize=True,use_optimized_inference=False), retaining the original on-disk config and quantization scheme. Explicit device_map maps the entire model to the selected device, avoiding CPU expansion of the full BF16 base under the16GiB host limit. Loading policy is recorded in evidence. Text targets must expose real, non-meta floating weights of the declared Linear shape before PEFT; reload checks the same condition.

No fabricated weight attribute, custom PEFT dispatcher, parameter-dimension guess, target widening, on-disk model mutation or dependency upgrade is used. This materializes BF16 weights on GPU and changes residency versus packed execution; actual80GB A100 peak/RSS still must be measured within the unchanged resource ceiling. Do not claim this fixes every possible runtime issue before the next authorized execution.

## Validation

```text
python -m unittest tests.test_g38_quantized_loading -v
python run_tests.py
```

Local tests cover missing/meta/integer weights and explicit load attributes without modifying config bytes. The author environment does not contain PEFT/Transformers/CT, so native compressed execution is not claimed locally.

Liang's existing installed environment can run the small CPU proof:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/scc/pb24511961/v3/envs/py3119/bin/python -I /ABS/R12/tools/ct_peft_smoke.py
```

The script uses actual CT4-bit group32 compression on one32×16 Linear, reproduces PEFT's missing-weight error before forward, then runs the actual HF post-load dequantization hook and a LoRA optimizer step. It does not download/read Gemma weights or use GPU. Native result remains pending until returned by the operator. The independently accepted plain-text scope and production checkpoint tests are not repeated as a substitute for this proof.

Both88918 and89019 consumed their one-submission approvals. Preserve conservative0.25 each (0.50 total); no refund/finalization inferred from missing accounting. No GPU retry is authorized by this candidate. After focused review/native CPU proof, bind new code/context and issue a new bounded approval, maintaining the00:30 deadline and existing budgets.
