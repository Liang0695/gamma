# r11: production reload and final checkpoint lifecycle

Baseline r10 ZIP: `b429d568e10e0aa78cb945917ca743bace7dee4f40e58b8a00643d5ca2f0cdbd`.
Independent findings: `01a12026-74f1-7898-a5a0-750d7cfdc609`.

Both supplied counterexamples reproduce on r10: inference-mode PEFT reload returns an empty trainability-filtered digest; forcing a trainable loader then exposes optimizer=None at late finalize.

The fixes follow the production call chain:

- Adapter digests select LoRA A/B tensor identities, independent of requires_grad. Names, shapes and normalized float32 byte hashes are included (balanced value changes no longer collide by sum). Base digests exclude adapters regardless of their freeze state.
- `load_adapter` explicitly requests its trainability mode, verifies adapter/base gradient flags and preserves the old-model weak-reference refusal before any new base load. Inference is the default standalone integrity mode; the lifecycle path explicitly requests trainable reload.
- `run_training` writes the final complete checkpoint before destructive release/reload. A missing checkpoint at the final step refuses release rather than using an older snapshot. After controlled reload, it restores the saved state onto the new parameters, so the subsequent engineering profile has a usable optimizer and trainable adapter.
- The required CPU resume test exposed adjacent serialization gaps: optional scheduler/scaler state was accepted but never written/read, and JSON optimizer state IDs remained strings. Optional state is now hashed in the checkpoint manifest and returned on load; IDs are converted back to integer parameter IDs before PyTorch remapping. Wired scheduler/scaler instances receive their saved state; an unwired required component still rejects rather than silently skipping it.
- Actual checkpoint adapter bytes are returned by CheckpointStore and restored into the new LoRA tensors after identity/shape/hash checks. A fresh lifecycle.begin therefore restores weights as well as moments/RNG/cursor. Base weights are not included or copied.
- Reload counts and parameter ownership are refreshed. Engineering-profile evidence compares against the restored model/optimizer ownership and retains the original training ownership separately; it no longer treats stale IDs as proof of reuse of the original base.

PEFT's default inference behavior is documented in the fixed [0.21.0 implementation](https://raw.githubusercontent.com/huggingface/peft/v0.21.0/src/peft/peft_model.py). No PEFT/Transformers install or version change was made locally.

## CPU verification

```text
python -m unittest tests.test_g38_reload_checkpoint -v
python run_tests.py
```

Tests use actual runner/lifecycle/checkpoint code, real CPU PyTorch and AdamW. Only model preparation/carrier writing and the PEFT loader are fixture substitutes. They exercise one step → adapter save → final checkpoint → controlled reload → moments restored on new parameter objects → another step; also inference reload, fresh lifecycle.begin, optional-state integrity and refusal before destructive release if final saving is suppressed.

The target-side `tools/peft_lifecycle_smoke.py` additionally runs **actual installed PEFT** through production `run_training`, `save_adapter`, `load_adapter`, checkpoint and resume methods using a small CPU model factory. It does not load Gemma weights, access Hub or use GPU:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/scc/pb24511961/v3/envs/py3119/bin/python -I /ABS/R11_SOURCE/tools/peft_lifecycle_smoke.py --dest-dir /ABS/OUTSIDE/fresh-r11-lifecycle
```

That actual-PEFT result remains pending until Liang returns it. The earlier r10 text-scope smoke is a separate result and does not substitute for this production lifecycle check. Preserve its completed evidence; do not repeat unchanged preparation.

Text scope, installed locks, published-source checks, import isolation and v6 originals remain unchanged. Independent review and target CPU validation precede new intake/context/approval. Job88918's approval remains consumed; conservative0.25 GPU-h debit and no automatic retry remain in force.
