"""LoRA 目标模块与资源账（KAGGLE-20 §3/§5，KAGGLE-19 §3/§6 T1）。

- 目标正则：`^model\\.language_model\\.layers\\.\\d+\\.self_attn\\.(q_proj|o_proj)$`，预计 **120** 个；
- r16/alpha32/dropout0.05/bias none，普通 LoRA（非 DoRA/rsLoRA），modules_to_save 为空；
- 冻结：视觉塔/投影、embedding、lm_head、norm；
- 预计 28,672,000 可训练参数，BF16 adapter ≈54.69MiB（算术估计，不是实测）；
- 内存硬门槛：seq2048 smoke 峰值 >72GiB 或 RSS>12GiB 即不得扩大规模。

**加载成功不等于有效**：本模块只做静态规划与账目，真实 shape/显存必须由 GPU 侧实测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..common.errors import MissingInput, PolicyViolation

#: 核心几何（KAGGLE-20 §3 已核对）。
GEOMETRY = {
    "num_hidden_layers": 60,
    "hidden_size": 5376,
    "intermediate_size": 21504,
    "vocab_size": 262144,
    "local_layers": 50,
    "local_head_dim": 256,
    "local_q_heads": 32,
    "global_layers": 10,
    "global_head_dim": 512,
    "global_q_heads": 32,
    "sliding_window": 1024,
}

TARGET_REGEX = r"^model\.language_model\.layers\.\d+\.self_attn\.(q_proj|o_proj)$"

#: 冻结模块（正则前缀）。
FROZEN_PATTERNS = (
    r"^model\.vision_tower\.",
    r"^model\.multi_modal_projector\.",
    r"^model\.language_model\.embed_tokens(\.|$)",
    r"^lm_head(\.|$)",
    r".*_layernorm.*",
    r".*\.norm$",
    r".*\.input_layernorm$",
    r".*\.post_attention_layernorm$",
)

#: 内存硬门槛（KAGGLE-20 §5）。
PEAK_GPU_LIMIT_GIB = 72.0
HOST_RSS_LIMIT_GIB = 12.0
HOST_RESERVE_GIB = 3.0

FIRST_ROUND_SEQ_LEN = 2048
FALLBACK_SEQ_LEN = 1024


def expected_module_count() -> int:
    return GEOMETRY["num_hidden_layers"] * 2


def project_dim(layer_index: int, projection: str) -> int:
    """q_proj 的输出维度（local 8192 / global 16384）或 o_proj 的输入维度。"""
    is_local = layer_index < GEOMETRY["local_layers"]
    head_dim = GEOMETRY["local_head_dim"] if is_local else GEOMETRY["global_head_dim"]
    q_heads = GEOMETRY["local_q_heads"] if is_local else GEOMETRY["global_q_heads"]
    return head_dim * q_heads


def lora_param_count(rank: int = 16, include_mlp: bool = False) -> int:
    """r16 参数量：16 × [50×2×(5376+8192) + 10×2×(5376+16384)] = 28,672,000。"""
    hidden = GEOMETRY["hidden_size"]
    total = 0
    for layer in range(GEOMETRY["num_hidden_layers"]):
        dim = project_dim(layer, "q_proj")
        for _ in ("q_proj", "o_proj"):
            total += rank * (hidden + dim)
    if include_mlp:
        # 扩展候选：文本 MLP（gate/up/down）rank16 额外 77,414,400 参数。
        hidden, mlp = GEOMETRY["hidden_size"], GEOMETRY["intermediate_size"]
        total += GEOMETRY["num_hidden_layers"] * rank * (hidden + mlp) * 2
        total += GEOMETRY["num_hidden_layers"] * rank * (mlp + hidden)
    return total


def adapter_bytes(rank: int = 16, dtype: str = "bf16") -> int:
    bytes_per = {"bf16": 2, "fp32": 4}[dtype]
    return lora_param_count(rank) * bytes_per


def adapter_train_state_gib(rank: int = 16) -> float:
    """参数+梯度+Adam 两矩 ≈ 16 bytes/参数。"""
    return lora_param_count(rank) * 16 / float(1 << 30)


def is_frozen(name: str) -> bool:
    return any(re.match(pattern, name) for pattern in FROZEN_PATTERNS)


@dataclass
class ModulePlan:
    """目标模块清单的核对结果。"""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    matched: list[str] = field(default_factory=list)
    unmatched_expected: list[str] = field(default_factory=list)
    extra_candidates: list[str] = field(default_factory=list)
    explained_mapping: Mapping | None = None

    @classmethod
    def from_names(cls, names: Iterable[str], rank: int = 16) -> "ModulePlan":
        pattern = re.compile(TARGET_REGEX)
        plan = cls(rank=rank)
        for name in names:
            if pattern.match(name):
                plan.matched.append(name)
            elif "self_attn" in name and name.endswith("_proj"):
                # 例如 vLLM 融合后的 qkv_proj：必须显式解释映射，不能直接当成 q/o。
                plan.extra_candidates.append(name)
        plan.matched.sort()
        plan.extra_candidates.sort()
        return plan

    def expected_names(self) -> list[str]:
        names = []
        for layer in range(GEOMETRY["num_hidden_layers"]):
            for projection in ("q_proj", "o_proj"):
                names.append(
                    "model.language_model.layers.%d.self_attn.%s" % (layer, projection)
                )
        return names

    def validate(self) -> None:
        """120 模块**或已解释的正确映射**，二者必居其一，否则阻断。"""
        expected = self.expected_names()
        missing = sorted(set(expected) - set(self.matched))
        # 先查「非预期的 q/o 命名」（例如融合后的 qkv_proj）：它比缺模块更可疑。
        if self.extra_candidates and not self.explained_mapping:
            raise PolicyViolation(
                "unexplained_extra_modules",
                "出现非预期 self_attn q/o 命名（例如融合后的 qkv_proj），必须先解释映射",
                extra=self.extra_candidates[:10],
            )
        if missing and not self.explained_mapping:
            raise MissingInput(
                "target_modules_incomplete",
                "只匹配到 %d/%d 个目标模块，且未提供已解释映射"
                % (len(self.matched), len(expected)),
                missing=missing[:10],
            )
        if missing and self.explained_mapping:
            reason = str(self.explained_mapping.get("reason", "")).strip()
            evidence = self.explained_mapping.get("evidence")
            if not reason or not evidence:
                raise MissingInput(
                    "mapping_not_explained",
                    "missing 模块的映射解释必须含 reason 与 evidence",
                )

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "target_regex": TARGET_REGEX,
            "matched_count": len(self.matched),
            "expected_count": expected_module_count(),
            "trainable_params": lora_param_count(self.rank),
            "adapter_bytes_bf16": adapter_bytes(self.rank, "bf16"),
            "adapter_train_state_gib": round(adapter_train_state_gib(self.rank), 4),
            "explained_mapping": dict(self.explained_mapping) if self.explained_mapping else None,
            "note": "参数量是算术估计；真实 shape 必须由锁定版本导出后才允许训练。",
        }


@dataclass
class MemoryPlan:
    peak_gpu_gib: float
    host_rss_gib: float
    seq_len: int = FIRST_ROUND_SEQ_LEN
    available_gpu_gib: float | None = None
    downgrade_used: bool = False

    def effective_gpu_limit(self) -> float:
        """未知可用显存时取 min(72GiB, 实测空闲 − 4GiB)。"""
        if self.available_gpu_gib is None:
            return PEAK_GPU_LIMIT_GIB
        return min(PEAK_GPU_LIMIT_GIB, max(0.0, self.available_gpu_gib - 4.0))

    def evaluate(self) -> dict:
        limit = self.effective_gpu_limit()
        rss_limit = HOST_RSS_LIMIT_GIB
        gpu_ok = self.peak_gpu_gib <= limit
        rss_ok = self.host_rss_gib <= rss_limit
        if gpu_ok and rss_ok:
            return {
                "status": "pass",
                "seq_len": self.seq_len,
                "gpu_limit_gib": limit,
                "host_rss_limit_gib": rss_limit,
            }
        if not self.downgrade_used and self.seq_len > FALLBACK_SEQ_LEN:
            return {
                "status": "retry_seq1024",
                "reason": "首次超门槛：按登记路线降到 seq1024 重测一次",
                "gpu_limit_gib": limit,
                "host_rss_limit_gib": rss_limit,
            }
        return {
            "status": "stop",
            "reason": "门槛仍不过：停止 BF16 路线并按登记路线转 QLoRA，不循环重启试错",
            "gpu_limit_gib": limit,
            "host_rss_limit_gib": rss_limit,
        }


def dense_reference_bytes(params: int = 31_273_088_876) -> int:
    """冻结 dense 参考：BF16 权重字节数（禁止在 CPU 构造完整 state_dict）。"""
    return params * 2


def disk_budget_gib(*, needs_bf16_intermediate: bool = False) -> dict:
    """磁盘规划：80GiB 仅模型流程起点，不是全流水线保证。"""
    base = 80
    if needs_bf16_intermediate:
        base += 63
    return {
        "minimum_gib": base,
        "note": "先核查权重/环境/最大快照/日志 + 15% 余量，再按实际文件 manifest 调整。",
    }


def assert_no_auto_gpu(flags: Mapping) -> None:
    """禁止 GPU 自动开工：必须有显式 allow_gpu 且门槛已通过。"""
    if not flags.get("allow_gpu"):
        raise PolicyViolation(
            "gpu_not_authorized",
            "未显式授权 GPU：本编码任务不独自申请作业，GPU 执行交 Liang 统筹",
        )
    if not flags.get("gates_passed"):
        raise PolicyViolation(
            "gates_not_passed", "验收门槛未通过前不得启动 GPU 执行"
        )
    if flags.get("budget_gpu_hours", 0) <= 0:
        raise PolicyViolation("no_gpu_budget", "未被释放任何 GPU 预算")
