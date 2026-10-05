"""训练配置（fail-closed）。KAGGLE-20 §5 的配置草案 + KAGGLE-19 §3 的训练窗。

硬性约束：
- 未锁定字段、浮动版本、自动 latest → 加载即阻断；
- 无效 adapter 静默过关、GPU 自动开工 → 显式拒绝（见 `assert_start_allowed`）；
- lr 只在 {2e-5, 5e-5, 1e-4} 里选，最多追加 2 个预注册候选，不做全网格。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping

from ..common.errors import MissingInput, PolicyViolation, reject_unpinned
from .targets import FIRST_ROUND_SEQ_LEN, TARGET_REGEX

#: 设计稿给定的首轮超参（不是"已找到最优值"）。
DEFAULT_CONFIG: dict = {
    "config_version": "v3-train-config/1",
    "precision": "bf16",
    "quantization": {
        "mode": "ct_dequantize",
        "load_in_4bit": False,
        "nf4_double_quant": False,
        "compute_dtype": "bf16",
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "use_dora": False,
        "use_rslora": False,
        "modules_to_save": [],
        "target_modules_regex": TARGET_REGEX,
        "expected_module_count": 120,
    },
    "optim": {
        "lr": 5e-5,
        "lr_candidates": [2e-5, 5e-5, 1e-4],
        "max_lr_candidates_after_first": 2,
        "optimizer": "adamw",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "warmup_ratio": 0.10,
        "min_warmup_steps": 1,
        "lr_scheduler": "cosine",
        "adam_moments_dtype": "fp32",
        "lora_params_dtype": "fp32",
    },
    "batching": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "seq_len": FIRST_ROUND_SEQ_LEN,
        "packing": False,
        "dataloader_num_workers": 0,
        "streaming": True,
    },
    "runtime": {
        "gradient_checkpointing": True,
        "use_cache": False,
        "cpu_offload": False,
        "fsdp": False,
        "deepspeed": False,
        "torch_compile": False,
        "parallel_generation_service": False,
        "attention_backend": "sdpa",
        "single_device_placement": True,
    },
    "budget": {
        "max_optimizer_updates": 32,
        "max_train_hours": 2.0,
        "job_hours": 4.0,
        "stop_new_work_at_hours": 3.75,
        "checkpoint_every_updates": 10,
        "checkpoint_every_minutes": 15,
        "keep_last_checkpoints": 2,
        "min_steps_for_claim": 16,
        "min_input_tokens_for_claim": 200000,
    },
    "seed": 17,
    "freeze": [
        "vision_tower",
        "multi_modal_projector",
        "embed_tokens",
        "lm_head",
        "layernorms",
    ],
}


@dataclass
class TrainingConfig:
    payload: dict = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_CONFIG)))

    @classmethod
    def default(cls) -> "TrainingConfig":
        return cls()

    @classmethod
    def from_file(cls, path: str) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        merged.update(loaded)
        return cls(merged)

    # ---- 访问器 ----

    @property
    def seq_len(self) -> int:
        return int(self.payload["batching"]["seq_len"])

    @property
    def lora(self) -> dict:
        return dict(self.payload["lora"])

    @property
    def budget(self) -> dict:
        return dict(self.payload["budget"])

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self.payload))

    def config_sha256(self) -> str:
        from ..common.canonical import sha256_json

        return sha256_json(self.payload)

    # ---- 校验 ----

    def validate(self, base_revision: str | None = None) -> list[str]:
        problems: list[str] = []
        if self.payload["precision"] != "bf16":
            problems.append("首轮 precision 必须是 bf16（NF4 只在登记后作为后备）")
        lora = self.payload["lora"]
        if lora["r"] != 16:
            problems.append("首轮 rank 必须是 16（其余 rank 属候选，不是首轮）")
        if lora["lora_alpha"] != 32:
            problems.append("alpha 必须是 32")
        if abs(float(lora["lora_dropout"]) - 0.05) > 1e-9:
            problems.append("dropout 必须是 0.05")
        if lora["bias"] != "none":
            problems.append("bias 必须是 none")
        if lora["use_dora"] or lora["use_rslora"]:
            problems.append("只允许普通 LoRA，禁止 DoRA/rsLoRA")
        if lora["modules_to_save"]:
            problems.append("modules_to_save 必须为空")
        optim = self.payload["optim"]
        if float(optim["lr"]) not in [float(x) for x in optim["lr_candidates"]]:
            problems.append("lr 必须在预注册候选集合内")
        if int(optim["max_lr_candidates_after_first"]) > 2:
            problems.append("首轮后最多追加 2 个预注册 lr 候选，不做全网格")
        batching = self.payload["batching"]
        if batching["per_device_train_batch_size"] != 1:
            problems.append("batch 必须是 1")
        if batching["gradient_accumulation_steps"] != 8:
            problems.append("accum 必须是 8")
        if batching["packing"]:
            problems.append("packing 必须关闭")
        if batching["dataloader_num_workers"] != 0:
            problems.append("workers 必须是 0")
        runtime = self.payload["runtime"]
        for flag in ("cpu_offload", "fsdp", "deepspeed", "torch_compile", "parallel_generation_service"):
            if runtime[flag]:
                problems.append("%s 必须关闭" % flag)
        if not runtime["gradient_checkpointing"]:
            problems.append("必须开启梯度 checkpointing")
        if runtime["use_cache"]:
            problems.append("use_cache 必须为 False")
        budget = self.payload["budget"]
        if budget["max_optimizer_updates"] > 32:
            problems.append("最多 32 次 optimizer update")
        if float(budget["max_train_hours"]) > 2.0:
            problems.append("训练最多 2 小时")
        if self.seq_len not in (1024, 2048, 4096):
            problems.append("seq_len 必须是 1024/2048/4096 之一（首轮 2048）")
        if base_revision is not None:
            try:
                reject_unpinned(base_revision, "base_revision")
            except Exception as exc:  # UnverifiedLock
                problems.append(str(exc))
        return problems

    def assert_valid(self, base_revision: str | None = None) -> None:
        problems = self.validate(base_revision)
        if problems:
            raise PolicyViolation(
                "train_config_invalid", "训练配置不合法（%d 项）" % len(problems), problems=problems
            )


def assert_start_allowed(
    *,
    config: TrainingConfig,
    train_lock_summary: Mapping,
    gates: Mapping,
    authorization: Mapping,
) -> dict:
    """开训前的 fail-closed 闸门：依赖锁、门槛、授权三者缺一不可。

    `gates` 至少要有 `t1_engineering_pass`（T1 门槛）与 `memory_plan_pass`。
    `authorization` 至少要有 `gpu_hours_released` 与 `stage`。
    """
    problems: list[str] = []
    if not train_lock_summary.get("verified"):
        problems.append("训练依赖锁 verified=false")
    for key in ("t1_engineering_pass", "memory_plan_pass", "export_manifest_pass"):
        if not gates.get(key):
            problems.append("门槛未通过：%s" % key)
    hours = float(authorization.get("gpu_hours_released") or 0)
    if hours <= 0:
        problems.append("未被释放任何 GPU 预算（本编码任务不独自申请作业）")
    if not authorization.get("operator"):
        problems.append("未指定 107 作业操作者（所有 107 操作仅 Liang 运行时）")
    if not authorization.get("v2_conflict_checked"):
        problems.append("未确认与 V2 排期无冲突（单账号单 GPU 必须串行）")
    if problems:
        raise PolicyViolation(
            "start_not_allowed", "开训闸门未通过（%d 项）" % len(problems), problems=problems
        )
    return {
        "allowed": True,
        "gpu_hours_released": hours,
        "stage": authorization.get("stage"),
        "operator": authorization.get("operator"),
    }


def lr_schedule_pairs(config: TrainingConfig) -> dict:
    from ..common.canonical import sha256_json

    return {
        "lr": config.payload["optim"]["lr"],
        "warmup_ratio": config.payload["optim"]["warmup_ratio"],
        "scheduler": config.payload["optim"]["lr_scheduler"],
        "config_sha256": sha256_json(config.payload),
    }
