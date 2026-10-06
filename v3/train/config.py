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

from ..common.errors import MissingInput, PolicyViolation, require_field, reject_unpinned
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


def _deep_merge(base: Mapping, override: Mapping) -> dict:
    """深合并：`dict` 递归合并，list / 标量整体覆盖（返回新对象，不改入参）。

    浅合并（原来的 `merged.update(loaded)`）会让 `{"lora": {"r": 16}}` 把整个
    `lora` 默认字典替换掉，随后 `validate()` 抛裸 `KeyError`。
    """
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = json.loads(json.dumps(value))
    return merged


def _unknown_keys(payload: Mapping, template: Mapping, prefix: str = "") -> list[str]:
    """列出不在 `DEFAULT_CONFIG` 里的配置键（fail-closed，不得静默忽略）。"""
    unknown: list[str] = []
    if not isinstance(payload, Mapping):
        return unknown
    for key, value in payload.items():
        path = "%s.%s" % (prefix, key) if prefix else str(key)
        if key not in template:
            unknown.append(path)
            continue
        if isinstance(template[key], Mapping) and isinstance(value, Mapping):
            unknown.extend(_unknown_keys(value, template[key], path))
    return unknown


def _as_int(value, where: str, problems: list[str]):
    """整数取值；非法时记入 problems 并返回 None（**不抛裸 TypeError/ValueError**）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        problems.append("%s 不是整数：%r" % (where, value))
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        problems.append("%s 不是整数：%r" % (where, value))
        return None


def _as_float(value, where: str, problems: list[str]):
    """浮点取值；非法时记入 problems 并返回 None（**不抛裸 TypeError/ValueError**）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        problems.append("%s 不是数值：%r" % (where, value))
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        problems.append("%s 不是数值：%r" % (where, value))
        return None


@dataclass
class TrainingConfig:
    payload: dict = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_CONFIG)))

    @classmethod
    def default(cls) -> "TrainingConfig":
        return cls()

    @classmethod
    def from_file(cls, path: str) -> "TrainingConfig":
        """从 JSON 文件加载训练配置：**允许部分配置，缺失项一律取 DEFAULT_CONFIG 默认值**。

        - 合并方式是**深合并** `_deep_merge(DEFAULT_CONFIG, loaded)`：dict 递归合并，
          list / 标量整体覆盖。因此 `{"lora": {"r": 16}}` 只覆盖 rank，`lora_alpha`
          等兄弟键保留默认值（浅合并会丢掉整个 `lora` 字典 → 裸 `KeyError`）。
        - 文件不是 JSON 对象 → `MissingInput("config_not_object")`；
          JSON 语法错误 → `MissingInput("config_not_json")`；
        - 出现 `DEFAULT_CONFIG` 未登记的键 → fail-closed：
          `PolicyViolation("unknown_config_key", ...)`，不静默忽略。

        返回 `TrainingConfig`；结构性问题抛 `v3/common/errors.py` 的 fail-closed 异常。
        """
        with open(path, "r", encoding="utf-8") as handle:
            try:
                loaded = json.load(handle)
            except json.JSONDecodeError as exc:
                raise MissingInput(
                    "config_not_json", "配置文件不是合法 JSON：%s（%s）" % (path, exc)
                )
        if not isinstance(loaded, Mapping):
            raise MissingInput(
                "config_not_object", "配置文件必须是 JSON 对象：%s" % path
            )
        unknown = _unknown_keys(loaded, DEFAULT_CONFIG)
        if unknown:
            raise PolicyViolation(
                "unknown_config_key",
                "配置含未登记键（不得静默忽略）：%s" % ", ".join(sorted(unknown)),
                unknown=sorted(unknown),
                path=path,
            )
        return cls(_deep_merge(DEFAULT_CONFIG, loaded))

    # ---- 访问器 ----

    def _section(self, key: str) -> dict:
        """取一个必须是对象的配置段；缺失/类型不对 → MissingInput。"""
        payload = self.payload if isinstance(self.payload, Mapping) else {}
        return require_field(payload, key, "config")

    @property
    def seq_len(self) -> int:
        batching = self._section("batching")
        value = require_field(batching, "seq_len", "config.batching")
        try:
            return int(value)
        except (TypeError, ValueError):
            raise MissingInput(
                "config_field_not_integer", "config.batching.seq_len 不是整数：%r" % (value,)
            )

    @property
    def lora(self) -> dict:
        return dict(self._section("lora"))

    @property
    def budget(self) -> dict:
        return dict(self._section("budget"))

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self.payload))

    def config_sha256(self) -> str:
        from ..common.canonical import sha256_json

        return sha256_json(self.payload)

    # ---- 校验 ----

    def validate(self, base_revision: str | None = None) -> list[str]:
        """静态校验，返回问题列表（空列表 = 通过）。

        取字段一律走 `require_field(...)`：缺字段 → `MissingInput`（fail-closed），
        数值字段走 `_as_int` / `_as_float`（非法值记入 problems）——**任何路径都不抛裸
        `KeyError` / `TypeError` / `ValueError`**。未登记的配置键也会被列为问题。
        """
        problems: list[str] = []
        payload = self.payload if isinstance(self.payload, Mapping) else {}
        for key in _unknown_keys(payload, DEFAULT_CONFIG):
            problems.append("未登记的配置键 %r：不得静默忽略" % key)

        precision = require_field(payload, "precision", "config")
        if precision != "bf16":
            problems.append("首轮 precision 必须是 bf16（NF4 只在登记后作为后备）")

        lora = require_field(payload, "lora", "config")
        rank = _as_int(require_field(lora, "r", "config.lora"), "config.lora.r", problems)
        if rank is not None and rank != 16:
            problems.append("首轮 rank 必须是 16（其余 rank 属候选，不是首轮）")
        alpha = _as_int(
            require_field(lora, "lora_alpha", "config.lora"), "config.lora.lora_alpha", problems
        )
        if alpha is not None and alpha != 32:
            problems.append("alpha 必须是 32")
        dropout = _as_float(
            require_field(lora, "lora_dropout", "config.lora"),
            "config.lora.lora_dropout",
            problems,
        )
        if dropout is not None and abs(dropout - 0.05) > 1e-9:
            problems.append("dropout 必须是 0.05")
        if require_field(lora, "bias", "config.lora") != "none":
            problems.append("bias 必须是 none")
        if require_field(lora, "use_dora", "config.lora") or require_field(
            lora, "use_rslora", "config.lora"
        ):
            problems.append("只允许普通 LoRA，禁止 DoRA/rsLoRA")
        if require_field(lora, "modules_to_save", "config.lora"):
            problems.append("modules_to_save 必须为空")

        optim = require_field(payload, "optim", "config")
        lr = _as_float(require_field(optim, "lr", "config.optim"), "config.optim.lr", problems)
        candidates_raw = require_field(optim, "lr_candidates", "config.optim")
        if not isinstance(candidates_raw, (list, tuple)) or not candidates_raw:
            problems.append("config.optim.lr_candidates 必须是非空列表")
        else:
            candidates = [
                _as_float(item, "config.optim.lr_candidates", problems) for item in candidates_raw
            ]
            if lr is not None and lr not in [item for item in candidates if item is not None]:
                problems.append("lr 必须在预注册候选集合内")
        max_candidates = _as_int(
            require_field(optim, "max_lr_candidates_after_first", "config.optim"),
            "config.optim.max_lr_candidates_after_first",
            problems,
        )
        if max_candidates is not None and max_candidates > 2:
            problems.append("首轮后最多追加 2 个预注册 lr 候选，不做全网格")

        batching = require_field(payload, "batching", "config")
        per_device = _as_int(
            require_field(batching, "per_device_train_batch_size", "config.batching"),
            "config.batching.per_device_train_batch_size",
            problems,
        )
        if per_device is not None and per_device != 1:
            problems.append("batch 必须是 1")
        accum = _as_int(
            require_field(batching, "gradient_accumulation_steps", "config.batching"),
            "config.batching.gradient_accumulation_steps",
            problems,
        )
        if accum is not None and accum != 8:
            problems.append("accum 必须是 8")
        if require_field(batching, "packing", "config.batching"):
            problems.append("packing 必须关闭")
        workers = _as_int(
            require_field(batching, "dataloader_num_workers", "config.batching"),
            "config.batching.dataloader_num_workers",
            problems,
        )
        if workers is not None and workers != 0:
            problems.append("workers 必须是 0")

        runtime = require_field(payload, "runtime", "config")
        for flag in ("cpu_offload", "fsdp", "deepspeed", "torch_compile", "parallel_generation_service"):
            if require_field(runtime, flag, "config.runtime"):
                problems.append("%s 必须关闭" % flag)
        if not require_field(runtime, "gradient_checkpointing", "config.runtime"):
            problems.append("必须开启梯度 checkpointing")
        if require_field(runtime, "use_cache", "config.runtime"):
            problems.append("use_cache 必须为 False")

        budget = require_field(payload, "budget", "config")
        max_updates = _as_int(
            require_field(budget, "max_optimizer_updates", "config.budget"),
            "config.budget.max_optimizer_updates",
            problems,
        )
        if max_updates is not None and max_updates > 32:
            problems.append("最多 32 次 optimizer update")
        max_hours = _as_float(
            require_field(budget, "max_train_hours", "config.budget"),
            "config.budget.max_train_hours",
            problems,
        )
        if max_hours is not None and max_hours > 2.0:
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

    optim = config._section("optim")
    return {
        "lr": require_field(optim, "lr", "config.optim"),
        "warmup_ratio": require_field(optim, "warmup_ratio", "config.optim"),
        "scheduler": require_field(optim, "lr_scheduler", "config.optim"),
        "config_sha256": sha256_json(config.payload),
    }
