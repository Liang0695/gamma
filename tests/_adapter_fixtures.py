"""G8 测试用的**合成** adapter 导出目录与 v2 清单。

只构造字节级产物（`adapter_model.safetensors` + `adapter_config.json`），
**不 import torch/peft、不加载任何权重**。它证明的是"生产/消费共用一份契约且与字节对账"，
不证明真实 adapter 可加载。
"""

from __future__ import annotations

import json
import os

from v3.submit import adapter_contract as ac
from v3.train import adapter_export_contract as contract

#: 合成权重占位字节（不是真的 safetensors，仅用于哈希对账）。
FAKE_WEIGHT_BYTES = b"SYNTHETIC-NOT-A-REAL-SAFETENSORS-PAYLOAD"

FIXTURE_HASHES = {
    "source_sha256": "1" * 64,
    "data_sha256": "2" * 64,
    "config_sha256": "3" * 64,
    "code_sha256": "4" * 64,
    "deps_sha256": "5" * 64,
}


def config_payload(adapter_name: str) -> dict:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj", "v_proj"],
        "_synthetic_fixture": True,
        "_declared_adapter_name": adapter_name,
    }


def make_official_export_dir(root: str, adapter_name: str = "v3_policy") -> str:
    """在 `root` 下造一个符合官方 PEFT 目录契约的导出目录，返回其路径。"""
    directory = os.path.join(root, *ac.adapter_dir_relative_path(adapter_name).split("/"))
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, ac.ADAPTER_WEIGHTS_FILENAME), "wb") as handle:
        handle.write(FAKE_WEIGHT_BYTES)
    with open(
        os.path.join(directory, ac.ADAPTER_CONFIG_FILENAME), "w", encoding="utf-8"
    ) as handle:
        json.dump(config_payload(adapter_name), handle, ensure_ascii=False, sort_keys=True)
    return root


def make_export_manifest(
    export_dir: str,
    adapter_name: str = "v3_policy",
    *,
    training_hashes: dict | None = None,
) -> dict:
    """用**生产入口**产出一份 v2 清单（测试必须走生产者，不手拼字段）。"""
    return contract.build_export_manifest(
        export_dir,
        adapter_name=adapter_name,
        rank=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules_regex=".*(q_proj|v_proj)$",
        matched_module_count=8,
        base_repo_id="google/gemma-4-31B-it-qat-w4a16-ct",
        base_revision="52f3f65bc7a02d555763bc923bd1d9094898219d",
        vllm_mapper_revision="vllm-fixture-0.0.1",
        training_hashes=training_hashes or dict(FIXTURE_HASHES),
    )


def legacy_self_report_manifest(adapter_name: str = "v3_policy") -> dict:
    """E0 旧口径的"手拼清单"：正是 G8 要关掉的通过方式（用于负例）。"""
    return {
        **{key: "a" * 64 for key in FIXTURE_HASHES},
        "adapter_only": True,
        "files": [ac.carrier_weights_relative_path(adapter_name)],
        "load_ok": True,
        "params_changed": True,
        "fixture_pass": 20,
        "fixture_total": 20,
    }
