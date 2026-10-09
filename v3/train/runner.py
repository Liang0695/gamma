"""可执行训练后端：模型加载 / LoRA 挂载 / 前向 / 反向 / 优化器步进 / 保存重载。

为什么有这个文件（KAGGLE-26 / Q0 报告 §7）：

旧交付里 `entry.start()` 在通过全部闸门后**无条件**抛出一个"未实现/不执行"的阻断
（该错误码已从仓库彻底移除），最后一条语句就是抛出，不存在走到训练代码的路径；
全仓 `v3/` 里 `import torch` / `get_peft_model` / `.backward()` / `optimizer.step()`
命中数为 0。因此当时"不存在可执行的训练入口"，T1 无法开工。Mika 明确这条属于 E0 原范围：
**实现**与 **GPU 执行**要分开 —— 实现必须在交付物里，可执行、可验证；GPU 实耗仍交
Liang 统筹链，且必须先过 `entry.start()` 的全部闸门。

两个后端：

- `SyntheticBackend`：纯标准库的**微型 LoRA 训练循环**（真前向、真反向手推梯度、
  真优化器步进、真保存重载）。它在 CPU 上完整跑通，用来证明"训练循环本身可用"，
  并作为 `t1_engineering_pass` 闸门的可复现证据。它**不是**模型训练，
  不产生任何可用于比赛的成绩。
- `TorchPeftBackend`：真实 `transformers` + `peft` 路径（BF16 base + rank16 LoRA）。
  延迟导入，缺锁定依赖即 `Blocked`；本机无 GPU、无锁定软件，因此本轮**未实测**，
  这一点在各处如实标注，不冒充已验证。

两条路径共用同一个 `run_training()` 编排：prepare → train → adapter-only 保存 →
重载校验 → 出具证据。任何一步失败都 fail-closed，不静默降级。
"""

from __future__ import annotations

import base64
import gc
import hashlib
import inspect
import json
import math
import os
import random
import weakref
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..common.canonical import sha256_json
from ..common.errors import Blocked, IntegrityError, MissingInput, PolicyViolation
from ..submit import adapter_contract
from .streaming import assert_full_state_dict_guard

#: 真实训练后端的适配器权重文件名 —— **官方 PEFT 载体**的 basename
#: （KAGGLE-27 整改①：不是 `adapter.safetensors`）。
ADAPTER_FILE = adapter_contract.ADAPTER_WEIGHTS_FILENAME
#: 导出时的 adapter 名（= 提交 YAML 里 `adapter:` 声明的名字）。
#: 它不是模型 pin 的替身：模型/revision 仍必须来自锁定参数（见 `resolve_backend_pins`）。
DEFAULT_ADAPTER_NAME = adapter_contract.DEFAULT_ADAPTER_NAME
#: 合成后端的适配器文件名。**不是**提交载体，故意不叫 .safetensors，
#: 避免被任何提交校验器误当官方适配器。
SYNTHETIC_ADAPTER_FILE = "adapter.synthetic.json"
#: 证据文件名。
EVIDENCE_FILE = "training_evidence.json"

#: 梯度范数下限：低于该值视为"没有真的发生反向"，不得当作训练成功。
MIN_GRAD_NORM = 1e-12


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _params_digest(values: Sequence[float]) -> str:
    return sha256_json([round(float(value), 12) for value in values])


def _sampler_order_sha256(plan: "TrainRunPlan") -> str:
    """批次顺序的规范摘要（G14：数据游标必须与 RNG 一起恢复，不可只恢复其一）。

    摘要只覆盖**采样顺序**（`plan.batches` 的次序与每批的被监督位置），**不**覆盖总步数：
    续训作业的 `steps` 通常与首片不同（首片 4h、续片再 4h），若把步数算进摘要，
    同一份数据在续训时会被误判成"数据换了"而拒绝恢复 —— 那会把续训堵死。
    总进度由 checkpoint 里的 `cursor.position` / `global_step` 表达。
    """
    order = [
        {
            "index": index,
            "input_ids": list(batch.input_ids),
            "supervised_positions": [
                position for position, label in enumerate(batch.labels) if label != -100
            ],
        }
        for index, batch in enumerate(plan.batches)
    ]
    return sha256_json({"batch_count": len(order), "sampler_order": order})


def python_rng_state(state) -> dict:
    """`random.Random.getstate()` → 可 JSON 化的结构（两个后端共用同一编码）。"""
    return {
        "encoding": "random.Random.getstate",
        "version": int(state[0]),
        "internal": [int(item) for item in state[1]],
        "gauss_next": state[2],
    }


def python_rng_restore(payload: Mapping):
    """`python_rng_state()` 的逆运算，返回可直接喂给 `Random.setstate()` 的元组。"""
    try:
        return (
            int(payload["version"]),
            tuple(int(item) for item in payload["internal"]),
            payload.get("gauss_next"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MissingInput(
            "rng_python_state_invalid", "Python RNG 状态结构不对：%s" % exc
        ) from exc


def plan_from_json(path: str) -> "TrainRunPlan":
    """从 JSON 读一份**真实批次**训练计划（G7b 的 `--plan-json`）。

    这是"真实测量"与"合成 fixture"的分界线：计划里的 `batches` 必须由执行方按真实
    数据给出，入口**不会**自己编造训练数据。缺字段一律 fail-closed。
    """
    if not os.path.exists(path):
        raise MissingInput("training_plan_missing", "训练计划文件不存在：%s" % path)
    with open(path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise MissingInput("training_plan_invalid", "训练计划不是 JSON 对象：%s" % path)
    raw_batches = payload.get("batches")
    if not isinstance(raw_batches, list) or not raw_batches:
        raise MissingInput(
            "training_plan_invalid",
            "训练计划缺 batches：入口不会自行编造训练数据（%s）" % path,
        )
    batches = []
    for index, item in enumerate(raw_batches):
        if not isinstance(item, Mapping):
            raise MissingInput("training_plan_invalid", "batches[%d] 不是对象" % index)
        try:
            batches.append(
                TrainBatch(input_ids=list(item["input_ids"]), labels=list(item["labels"]))
            )
        except KeyError as exc:
            raise MissingInput(
                "training_plan_invalid", "batches[%d] 缺字段：%s" % (index, exc)
            ) from exc
    for field_name in ("steps", "lr", "seq_len", "lora_rank", "lora_alpha"):
        if payload.get(field_name) in (None, ""):
            raise MissingInput(
                "training_plan_invalid", "训练计划缺 %s（%s）" % (field_name, path)
            )
    plan = TrainRunPlan(
        steps=int(payload["steps"]),
        lr=float(payload["lr"]),
        seq_len=int(payload["seq_len"]),
        lora_rank=int(payload["lora_rank"]),
        lora_alpha=int(payload["lora_alpha"]),
        batches=batches,
        seed=int(payload.get("seed") or 20261005),
        adapter_name=str(payload.get("adapter_name") or DEFAULT_ADAPTER_NAME),
    )
    plan.assert_runnable()
    return plan


@dataclass
class TrainBatch:
    """一个训练批次：`input_ids` 与 `labels`（-100 为忽略位，与 masks 口径一致）。"""

    input_ids: list[int]
    labels: list[int]
    supervised_tokens: int = 0

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise MissingInput(
                "batch_shape_mismatch",
                "input_ids 与 labels 长度不一致（%d vs %d）"
                % (len(self.input_ids), len(self.labels)),
            )
        if not self.supervised_tokens:
            self.supervised_tokens = sum(1 for value in self.labels if value != -100)
        if self.supervised_tokens <= 0:
            raise PolicyViolation(
                "batch_without_supervised_tokens",
                "批次里没有任何被监督 token，不得当作训练样本",
            )

    def to_dict(self) -> dict:
        return {
            "length": len(self.input_ids),
            "supervised_tokens": self.supervised_tokens,
            "input_ids_sha256": sha256_json(list(self.input_ids)),
        }


@dataclass
class TrainRunPlan:
    """一次训练运行的输入契约。"""

    steps: int
    lr: float
    seq_len: int
    lora_rank: int
    lora_alpha: int
    batches: list[TrainBatch] = field(default_factory=list)
    seed: int = 20261005
    #: 导出时的官方 PEFT adapter 目录名（必须与提交 YAML 的 `adapter:` 一致）。
    adapter_name: str = DEFAULT_ADAPTER_NAME

    def assert_runnable(self) -> None:
        if self.steps <= 0:
            raise MissingInput("no_training_steps", "训练步数为 0，不得启动")
        if not self.batches:
            raise MissingInput("no_training_data", "没有训练批次，不得启动")
        if self.lr <= 0:
            raise PolicyViolation("non_positive_lr", "学习率必须为正")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise PolicyViolation("bad_lora_geometry", "LoRA rank/alpha 必须为正")
        # adapter 名必须能构成合法官方 PEFT 目录（空/带分隔符/`..` 一律拒）。
        adapter_contract.adapter_dir_relative_path(self.adapter_name)


class TrainBackend:
    """训练后端接口。实现必须自己声明 `requires_gpu` 与 `verified`。"""

    name = "base"
    requires_gpu = False
    verified = False

    def prepare(self, plan: TrainRunPlan) -> dict:
        raise NotImplementedError

    def train_steps(self, plan: TrainRunPlan) -> dict:
        raise NotImplementedError

    def save_adapter(self, dest_dir: str, *, adapter_name: str | None = None) -> dict:
        raise NotImplementedError

    def load_adapter(self, src_dir: str, *, adapter_name: str | None = None) -> dict:
        raise NotImplementedError

    def adapter_params_digest(self) -> str:
        raise NotImplementedError

    def base_params_digest(self) -> str:
        raise NotImplementedError

    def observe_full_state_dict(self) -> dict | None:
        """产出一次"完整 state_dict 是否整体驻留过"的**实测观测**（🟡-8）。

        返回 `None` 表示"本后端没有能力提供观测" → `run_training` 会 fail-closed
        （`full_state_dict_observation_missing`），不会静默跳过门槛。

        观测必须是**遍历出来的**事实，不是自述结论。返回体约定：

            {"probe": str, "method": str,
             "peak_full_state_dict_bytes": int,   # 曾经同时驻留的整份 state_dict 峰值
             "declared_weight_bytes": int}        # 全部权重的声明字节数
        """
        return None


class SyntheticBackend(TrainBackend):
    """纯标准库微型 LoRA 训练循环（CPU 可跑，用来证明循环本身可用）。

    几何：私有词表 `vocab`、模型维 `dim`、LoRA rank `rank`。

    - 冻结基座：`base_embed`（vocab×dim）与 `base_out`（vocab×vocab）；
    - 可训练 adapter：`lora_a`（rank×dim）、`lora_b`（vocab×rank）；
    - 前向：`logits(x) = base_out[x] + lora_b @ (lora_a @ base_embed[x])`；
    - 损失：被监督位置上的交叉熵均值（`labels == -100` 的位置不参与，和 mask 口径一致）；
    - 反向：手推 `dlogits`，再解析求 `lora_a` / `lora_b` 的梯度，基座梯度**不计算**；
    - 优化器：带动量的 SGD，只更新 adapter 参数。

    诚实边界：这是**玩具模型**，只能证明"前向/反向/步进/保存重载这条链路是活的"，
    不能证明任何真实训练收益，也不产生可提交产物。
    """

    name = "synthetic-cpu-loop"
    requires_gpu = False
    verified = False

    def __init__(self, *, vocab: int = 32, dim: int = 8, seed: int = 7) -> None:
        self.vocab = int(vocab)
        self.dim = int(dim)
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self.base_embed = [[self._rng.uniform(-1, 1) for _ in range(self.dim)] for _ in range(self.vocab)]
        self.base_out = [[self._rng.uniform(-0.1, 0.1) for _ in range(self.vocab)] for _ in range(self.vocab)]
        self.lora_a: list[list[float]] = []
        self.lora_b: list[list[float]] = []
        self._velocity_a: list[list[float]] = []
        self._velocity_b: list[list[float]] = []
        self.prepared = False
        self.rank = 0
        #: G14：续训/游标状态（由 `import_training_state` / `prepare` 初始化）。
        self.global_step = 0
        self.start_step = 0
        self.cursor_position = 0
        self.consumed_input_tokens = 0
        self.consumed_supervised_tokens = 0
        self.sampler_order_sha256 = ""

    # ---------------------------------------------------------------- 前向
    def _lora_update(self, token: int) -> list[float]:
        """B @ (A @ e_token)：返回长度 vocab 的增量 logits。"""
        embed = self.base_embed[token]
        latent = [
            sum(self.lora_a[row][col] * embed[col] for col in range(self.dim))
            for row in range(self.rank)
        ]
        return [
            sum(self.lora_b[row][col] * latent[col] for col in range(self.rank))
            for row in range(self.vocab)
        ]

    def forward(self, input_ids: Sequence[int]) -> list[list[float]]:
        logits = []
        for token in input_ids:
            base = list(self.base_out[token])
            delta = self._lora_update(token)
            logits.append([base[index] + delta[index] for index in range(self.vocab)])
        return logits

    @staticmethod
    def _softmax(row: Sequence[float]) -> list[float]:
        top = max(row)
        exps = [math.exp(value - top) for value in row]
        total = sum(exps)
        return [value / total for value in exps]

    def _loss_and_grads(self, batch: TrainBatch) -> tuple[float, list[list[float]], list[list[float]], int]:
        logits = self.forward(batch.input_ids)
        grad_a = [[0.0] * self.dim for _ in range(self.rank)]
        grad_b = [[0.0] * self.rank for _ in range(self.vocab)]
        total_loss = 0.0
        counted = 0
        for position, (token, label) in enumerate(zip(batch.input_ids, batch.labels)):
            if label == -100:
                continue
            probs = self._softmax(logits[position])
            total_loss += -math.log(max(probs[label], 1e-12))
            counted += 1
            dlogits = list(probs)
            dlogits[label] -= 1.0
            embed = self.base_embed[token]
            latent = [
                sum(self.lora_a[row][col] * embed[col] for col in range(self.dim))
                for row in range(self.rank)
            ]
            for row in range(self.vocab):
                if dlogits[row] == 0.0:
                    continue
                for col in range(self.rank):
                    grad_b[row][col] += dlogits[row] * latent[col]
            for row in range(self.rank):
                factor = sum(self.lora_b[out][row] * dlogits[out] for out in range(self.vocab))
                for col in range(self.dim):
                    grad_a[row][col] += factor * embed[col]
        if counted == 0:
            raise PolicyViolation("batch_without_supervised_tokens", "批次没有可监督位置")
        scale = 1.0 / counted
        return (
            total_loss * scale,
            [[value * scale for value in row] for row in grad_a],
            [[value * scale for value in row] for row in grad_b],
            counted,
        )

    # ---------------------------------------------------------------- 接口
    def prepare(self, plan: TrainRunPlan) -> dict:
        plan.assert_runnable()
        # token id 越界必须 fail-closed，而不是在 forward 里抛裸 IndexError。
        hashed = {
            token
            for batch in plan.batches
            for token, label in zip(batch.input_ids, batch.labels)
            if label != -100
        }
        out_of_range = sorted(token for token in hashed if not 0 <= token < self.vocab)
        if out_of_range:
            raise PolicyViolation(
                "token_id_out_of_range",
                "批次里的被监督 token id 超出合成词表范围 [0,%d)" % self.vocab,
                vocab=self.vocab,
                tokens=out_of_range[:10],
            )
        labels_out_of_range = sorted(
            {
                label
                for batch in plan.batches
                for label in batch.labels
                if label != -100 and not 0 <= label < self.vocab
            }
        )
        if labels_out_of_range:
            raise PolicyViolation(
                "label_id_out_of_range",
                "批次里的监督目标 id 超出合成词表范围 [0,%d)" % self.vocab,
                vocab=self.vocab,
                labels=labels_out_of_range[:10],
            )
        self.rank = int(plan.lora_rank)
        self._rng = random.Random(self.seed + plan.lora_rank)
        self.lora_a = [
            [self._rng.uniform(-0.01, 0.01) for _ in range(self.dim)] for _ in range(self.rank)
        ]
        self.lora_b = [[0.0] * self.rank for _ in range(self.vocab)]
        self._velocity_a = [[0.0] * self.dim for _ in range(self.rank)]
        self._velocity_b = [[0.0] * self.rank for _ in range(self.vocab)]
        # G14：游标/计数随 plan 初始化；续训时由 import_training_state 覆盖（不重置）。
        self.sampler_order_sha256 = _sampler_order_sha256(plan)
        if not getattr(self, "resumed_state_applied", False):
            self.global_step = 0
            self.start_step = 0
            self.cursor_position = 0
            self.consumed_input_tokens = 0
            self.consumed_supervised_tokens = 0
        self.prepared = True
        trainable = self.rank * self.dim + self.vocab * self.rank
        return {
            "backend": self.name,
            "requires_gpu": False,
            "vocab": self.vocab,
            "dim": self.dim,
            "lora_rank": self.rank,
            "lora_alpha": plan.lora_alpha,
            "trainable_params": trainable,
            "frozen_params": self.vocab * self.dim + self.vocab * self.vocab,
            "base_frozen_enforced": True,
            "note": "合成玩具模型：只验证训练链路，不产生可提交产物、不代表任何训练收益",
        }

    def train_steps(self, plan: TrainRunPlan, on_step=None) -> dict:
        """跑 `plan.steps` 步；`on_step(info)` 返回非空字符串即**立刻停止**并如实回报。

        续训时从 `self.start_step` 起步（见 :meth:`import_training_state`），
        所以 `global_step` 与数据游标都是连续的，不是重新从头开始。
        """
        if not self.prepared:
            raise Blocked("backend_not_prepared", "train_steps 在 prepare 之前被调用")
        momentum = 0.9
        base_before = (self.adapter_params_digest(), self.base_params_digest())
        losses: list[float] = []
        grad_norms: list[float] = []
        supervised_tokens = 0
        start_step = int(self.start_step)
        stop_reason: str | None = None
        executed = 0
        for offset in range(plan.steps):
            step = start_step + offset
            batch = plan.batches[step % len(plan.batches)]
            loss, grad_a, grad_b, counted = self._loss_and_grads(batch)
            if not math.isfinite(loss):
                raise IntegrityError("non_finite_loss", "第 %d 步 loss 非有限值" % step)
            norm = math.sqrt(
                sum(value * value for row in grad_a for value in row)
                + sum(value * value for row in grad_b for value in row)
            )
            if norm <= MIN_GRAD_NORM:
                raise IntegrityError(
                    "zero_gradient",
                    "第 %d 步梯度范数为 0：反向没有真的发生，不得当作训练成功" % step,
                )
            # 优化器步进：只更新 adapter（带动量 SGD），基座不参与。
            for row in range(self.rank):
                for col in range(self.dim):
                    self._velocity_a[row][col] = (
                        momentum * self._velocity_a[row][col] + grad_a[row][col]
                    )
                    self.lora_a[row][col] -= plan.lr * self._velocity_a[row][col]
            for row in range(self.vocab):
                for col in range(self.rank):
                    self._velocity_b[row][col] = (
                        momentum * self._velocity_b[row][col] + grad_b[row][col]
                    )
                    self.lora_b[row][col] -= plan.lr * self._velocity_b[row][col]
            losses.append(loss)
            grad_norms.append(norm)
            supervised_tokens += counted
            executed += 1
            self.cursor_position = step + 1
            self.consumed_input_tokens += sum(1 for item in batch.input_ids)
            self.consumed_supervised_tokens += counted
            self.global_step = step + 1
            if on_step is not None:
                reason = on_step(
                    {
                        "step": step + 1,
                        "loss": loss,
                        "grad_norm": norm,
                        "optimizer_step_count": executed,
                        "supervised_tokens_seen": supervised_tokens,
                    }
                )
                if reason:
                    stop_reason = str(reason)
                    break
        adapter_after, base_after = self.adapter_params_digest(), self.base_params_digest()
        if losses:
            summary = {
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_decreased": losses[-1] < losses[0],
                "losses": losses,
                "grad_norms": grad_norms,
                "grad_norm_min": min(grad_norms),
                "grad_norm_nonzero": all(value > MIN_GRAD_NORM for value in grad_norms),
            }
        else:
            summary = {
                "loss_first": None,
                "loss_last": None,
                "loss_decreased": False,
                "losses": losses,
                "grad_norms": grad_norms,
                "grad_norm_min": None,
                "grad_norm_nonzero": False,
            }
        return {
            "backend": self.name,
            "steps": plan.steps,
            "steps_executed": executed,
            "global_step": int(self.global_step),
            "start_step": start_step,
            "stopped": stop_reason is not None,
            "stop_reason": stop_reason,
            "supervised_tokens_seen": supervised_tokens,
            "optimizer": "sgd-momentum(0.9)，只作用于 adapter 参数",
            "optimizer_step_count": executed,
            "adapter_params_before": base_before[0],
            "adapter_params_after": adapter_after,
            "adapter_params_changed": adapter_after != base_before[0],
            "base_params_before": base_before[1],
            "base_params_after": base_after,
            "base_params_frozen": base_after == base_before[1],
            **summary,
        }

    def adapter_params_digest(self) -> str:
        flat = [value for row in self.lora_a for value in row]
        flat += [value for row in self.lora_b for value in row]
        return _params_digest(flat)

    def base_params_digest(self) -> str:
        flat = [value for row in self.base_embed for value in row]
        flat += [value for row in self.base_out for value in row]
        return _params_digest(flat)

    def observe_full_state_dict(self) -> dict:
        """逐参数驻留账（🟡-8）：本后端的循环里**没有任何**整份参数容器。

        这是遍历出来的结构事实，不是自述结论：`forward` / `_loss_and_grads` /
        优化器步进都按行/按列取用，从没把 `base_* + lora_*` 拼进同一个列表。
        因此"整份 state_dict 峰值"为 0，而"权重总量"按 float64 字节如实累加。
        """
        rows = 0
        elements = 0
        for matrix in (self.base_embed, self.base_out, self.lora_a, self.lora_b):
            for row in matrix:
                rows += 1
                elements += len(row)
        bytes_per_element = 8  # Python float（float64）
        return {
            "probe": "synthetic-parameter-residency",
            "method": "逐参数遍历：从不在同一容器里持有全部参数",
            "peak_full_state_dict_bytes": 0,
            "declared_weight_bytes": elements * bytes_per_element,
            "rows_visited": rows,
        }

    def export_adapter_bytes(self) -> bytes:
        """导出当前 adapter 的字节表示（G14：中途 checkpoint 里要存一份 adapter 副本）。

        与 :meth:`save_adapter` **同源**：同一份 payload、同一份编码，
        所以"checkpoint 里的副本"和"正式保存的 adapter"哈希必然一致。
        """
        return self._adapter_payload_bytes()

    def _adapter_payload_bytes(self) -> bytes:
        payload = {
            "format": "synthetic-lora-adapter/1",
            "is_official_submission_carrier": False,
            "lora_rank": self.rank,
            "dim": self.dim,
            "vocab": self.vocab,
            "lora_a": self.lora_a,
            "lora_b": self.lora_b,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def save_adapter(self, dest_dir: str, *, adapter_name: str | None = None) -> dict:
        # 合成后端**不是**提交载体，故意不产出 .safetensors，也不建官方 PEFT 目录：
        # `adapter_name` 参数只为接口一致而接受，不参与落盘。
        os.makedirs(dest_dir, exist_ok=True)
        raw = self._adapter_payload_bytes()
        path = os.path.join(dest_dir, SYNTHETIC_ADAPTER_FILE)
        with open(path, "wb") as handle:
            handle.write(raw)
        return {
            "path": path,
            "filename": SYNTHETIC_ADAPTER_FILE,
            "bytes": len(raw),
            "sha256": _sha256_bytes(raw),
            "adapter_only": True,
            "contains_base_weights": False,
            "is_official_submission_carrier": False,
        }

    # ---- G14：训练状态导出 / 导入（checkpoint 用） ----

    def export_training_state(self) -> dict:
        """导出可 JSON 化的完整续训状态：optimizer / scheduler / scaler / RNG / 游标。

        - optimizer：带动量 SGD 的速度项（`_velocity_*`）；
        - scheduler：合成后端的计划里没有 scheduler，**如实报 None**，不编造；
        - scaler：合成后端没有 AMP，**如实报 None**；
        - RNG：Python 的 `random.Random.getstate()`（numpy / torch 本后端不使用，
          明确写成 `not-used-by-backend` 而不是填假值）。
        """
        state = self._rng.getstate()
        rng = {
            "python_state": {
                "encoding": "random.Random.getstate",
                "version": int(state[0]),
                "internal": [int(item) for item in state[1]],
                "gauss_next": state[2],
            },
            "numpy_state": "not-used-by-backend:synthetic-cpu-loop",
            "torch_state": "not-used-by-backend:synthetic-cpu-loop",
            "rng_scope": "python-random-only",
        }
        cursor = {
            "sampler_order_sha256": self.sampler_order_sha256,
            "position": int(self.cursor_position),
            "global_step": int(self.global_step),
        }
        return {
            "optimizer": {
                "kind": "sgd-momentum",
                "momentum": 0.9,
                "velocity_a": self._velocity_a,
                "velocity_b": self._velocity_b,
            },
            "scheduler": None,
            "scaler": None,
            "rng": rng,
            "cursor": cursor,
            "consumed_input_tokens": int(self.consumed_input_tokens),
            "consumed_supervised_tokens": int(self.consumed_supervised_tokens),
            "accum_boundary": 1,
            "global_step": int(self.global_step),
            "rng_scope": "python-random-only",
            "tensor_encoding": "none（纯标准库后端不含张量）",
        }

    def import_training_state(self, state) -> dict:
        """把 checkpoint 里的状态灌回后端；游标与 RNG **一起**恢复，不可只恢复其一。"""
        optimizer = state.optimizer or {}
        rng = state.rng or {}
        cursor = state.cursor or {}
        python_state = rng.get("python_state")
        if not isinstance(python_state, Mapping):
            raise MissingInput(
                "rng_python_state_missing", "checkpoint 里没有可用的 Python RNG 状态"
            )
        if not cursor.get("sampler_order_sha256") or cursor.get("position") is None:
            raise MissingInput("cursor_incomplete", "checkpoint 里没有可用的数据游标")
        try:
            restored = (
                int(python_state["version"]),
                tuple(int(item) for item in python_state["internal"]),
                python_state.get("gauss_next"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MissingInput(
                "rng_python_state_invalid", "Python RNG 状态结构不对：%s" % exc
            ) from exc
        if self.sampler_order_sha256 and cursor["sampler_order_sha256"] != self.sampler_order_sha256:
            raise IntegrityError(
                "sampler_order_mismatch",
                "checkpoint 的批次顺序摘要与当前 plan 不一致：不得续训（数据换了）",
                expected=self.sampler_order_sha256,
                actual=cursor["sampler_order_sha256"],
            )
        self._rng.setstate(restored)
        self._velocity_a = [[float(value) for value in row] for row in optimizer.get("velocity_a", [])]
        self._velocity_b = [[float(value) for value in row] for row in optimizer.get("velocity_b", [])]
        self.global_step = int(state.global_step)
        self.start_step = int(state.global_step)
        self.cursor_position = int(cursor["position"])
        self.consumed_input_tokens = int(state.consumed_input_tokens)
        self.consumed_supervised_tokens = int(state.consumed_supervised_tokens)
        self.resumed_state_applied = True
        return {
            "applied": True,
            "global_step": self.global_step,
            "cursor_position": self.cursor_position,
            "sampler_order_sha256": cursor["sampler_order_sha256"],
            "optimizer_velocity_rows": [len(self._velocity_a), len(self._velocity_b)],
            "rng_restored": "python-random",
            "rng_not_used_by_backend": ["numpy", "torch"],
        }

    def measure_phases(self, plan: TrainRunPlan, probe) -> dict:
        """G7b：把一个训练步拆成 前向 / 反向 / 优化器步进，逐阶段交给探针测量。

        只跑**一个**步（内存峰值关心的是驻留，不是步数）；跑完把这一步算进游标与计数，
        因此测量本身不会让状态账目对不上。
        """
        if not self.prepared:
            raise Blocked("backend_not_prepared", "measure_phases 在 prepare 之前被调用")
        step = int(self.start_step)
        batch = plan.batches[step % len(plan.batches)]
        with probe.phase("forward"):
            loss, grad_a, grad_b, counted = self._loss_and_grads(batch)
        with probe.phase("backward"):
            norm = math.sqrt(
                sum(value * value for row in grad_a for value in row)
                + sum(value * value for row in grad_b for value in row)
            )
        with probe.phase("optimizer_step"):
            momentum = 0.9
            for row in range(self.rank):
                for col in range(self.dim):
                    self._velocity_a[row][col] = (
                        momentum * self._velocity_a[row][col] + grad_a[row][col]
                    )
                    self.lora_a[row][col] -= plan.lr * self._velocity_a[row][col]
            for row in range(self.vocab):
                for col in range(self.rank):
                    self._velocity_b[row][col] = (
                        momentum * self._velocity_b[row][col] + grad_b[row][col]
                    )
                    self.lora_b[row][col] -= plan.lr * self._velocity_b[row][col]
        self.global_step = step + 1
        self.cursor_position = step + 1
        self.consumed_input_tokens += sum(1 for item in batch.input_ids)
        self.consumed_supervised_tokens += counted
        return {
            "phases_measured": ["forward", "backward", "optimizer_step"],
            "loss": loss,
            "grad_norm": norm,
            "global_step": self.global_step,
            "note": "合成后端的分解测量：证明测量入口覆盖三个阶段，不代表真实模型内存。",
        }


    def load_adapter(self, src_dir: str, *, adapter_name: str | None = None) -> dict:
        path = os.path.join(src_dir, SYNTHETIC_ADAPTER_FILE)
        if not os.path.exists(path):
            raise MissingInput("adapter_file_missing", "找不到合成 adapter：%s" % path)
        with open(path, "rb") as handle:
            raw = handle.read()
        payload = json.loads(raw.decode("utf-8"))
        before = self.adapter_params_digest()
        self.rank = int(payload["lora_rank"])
        self.lora_a = [[float(value) for value in row] for row in payload["lora_a"]]
        self.lora_b = [[float(value) for value in row] for row in payload["lora_b"]]
        self.prepared = True
        after = self.adapter_params_digest()
        if after != before:
            raise IntegrityError(
                "adapter_reload_mismatch",
                "重载后的 adapter 参数摘要与保存前不一致",
                before=before,
                after=after,
            )
        return {
            "path": path,
            "sha256": _sha256_bytes(raw),
            "format": payload.get("format"),
            "reloaded_adapter_params": after,
            "matches_saved_adapter": True,
        }


class TorchPeftBackend(TrainBackend):
    """真实 `transformers` + `peft` 路径：BF16 基座 + rank16 LoRA（GPU 执行面）。

    **本轮未实测**：本机没有 GPU、没有锁定版本的 transformers/peft/torch，
    也没有下载授权，因此这条路径只做到"代码完整、缺依赖即 fail-closed"。
    不得把它当作已验证的 GPU 训练能力；GPU 实耗仍交 Liang 统筹链。

    真实执行时的关键动作（每一步都在代码里，不是注释里的承诺）：

    1. `AutoModelForCausalLM.from_pretrained(..., revision=<锁定 revision>)`
       —— revision 必传，禁止 latest；
    2. 基座全部 `requires_grad_(False)`，再 `get_peft_model()` 只解冻 LoRA；
    3. `loss.backward()` 真反向，`optimizer.step()` 真步进，并采样梯度范数；
    4. `save_pretrained()` 只落 adapter 权重，导出前校验没有完整 state_dict。
    """

    name = "torch-peft-bf16-lora"
    requires_gpu = True
    verified = False

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        target_modules: Sequence[str],
        allow_download: bool = False,
        device: str = "cuda",
        model_inputs: Mapping | None = None,
        local_load_authorized: bool = False,
        local_model_dir: str | None = None,
    ) -> None:
        if not revision or str(revision).strip().lower() in ("latest", "main", "head", ""):
            raise PolicyViolation(
                "unpinned_model_revision",
                "模型 revision 必须精确锁定，禁止 latest/main/head",
                revision=revision,
            )
        self.model_id = model_id
        self.revision = revision
        self.target_modules = list(target_modules)
        self.allow_download = bool(allow_download)
        self.device = device
        #: KAGGLE-38 r3 · C/D：**被绑定的输入**（配置/tokenizer/template）与是否授权本地加载。
        #: 两者必须分开：`allow_download` 只是"允许联网取权重"的网络许可，
        #: 它**不能**充当"可以在本机把这个 job 跑起来"的工程作业批准。
        self.model_inputs = dict(model_inputs) if model_inputs else None
        self.local_load_authorized = bool(local_load_authorized)
        #: r4 · 缺陷 3/4：真实加载**只**从固定本地目录离线进行。
        #: `local_model_dir` 是获批快照落位；缺它或缺文件即拒，**不回退联网**。
        self.local_model_dir = local_model_dir
        self.model = None
        self.tokenizer = None
        self._torch = None
        #: 加载次数（缺陷 B：真实后端第二次 prepare 会把 31B 基座再加载一遍）。
        self.load_count = 0
        #: 参数归属：证明内存测量测的是**训练用过的同一组**模型/optimizer 参数。
        self.parameter_ownership: dict | None = None
        #: r4 · 缺陷 1：最近一次 `release_base()` 的释放证据（弱引用存活情况 + 释放清单）。
        self.last_release_evidence: dict | None = None
        self._released_model_refs = []
        self.training_devices = {str(device)}
        # ---- G14：续训状态与步数账目（与 SyntheticBackend 同名同义） ----
        # 旧版 `train_steps(plan)` 没有 `on_step`、也不回报 `global_step` /
        # `steps_executed` / `stopped`，所以 `run_training(lifecycle=...)` 在中途
        # checkpoint 与停止处理的接线处直接 fail-closed。
        self.global_step = 0
        self.start_step = 0
        self.cursor_position = 0
        self.consumed_input_tokens = 0
        self.consumed_supervised_tokens = 0
        self.sampler_order_sha256: str | None = None
        self.accum_boundary = 1
        self.resumed_state_applied = False
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        #: 整份 state_dict 材料化计数器。本实现恒为 0：所有取权路径都逐参数/逐 adapter
        #: 张量取用，从不调用 `model.state_dict()`（那会一次性物化基座，硬规则禁止）。
        self._full_state_dict_materializations = 0
        #: r4 · 缺陷 3：绑定存在时，构造阶段就拒绝"核验官方却加载另一个 revision"。
        #: 独立审查的复现正是"只构造、不 prepare"就打印出不一致身份，所以闸门必须
        #: 在构造处就生效，而不是等到加载时才拦。
        if self.model_inputs is not None:
            self.assert_model_identity()

    @staticmethod
    def _import_stack():
        """延迟导入并 assert 锁定依赖：缺一即 Blocked，不用未锁定版本顶替。"""
        from .runtime_environment import verify_deployed_environment
        verify_deployed_environment()
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - 环境相关
            raise Blocked("torch_unavailable", "缺少锁定版本 torch：%s" % exc) from exc
        try:
            import peft  # type: ignore
            from peft import LoraConfig, get_peft_model  # type: ignore
        except Exception as exc:  # pragma: no cover - 环境相关
            raise Blocked("peft_unavailable", "缺少锁定版本 peft：%s" % exc) from exc
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - 环境相关
            raise Blocked("transformers_unavailable", "缺少锁定版本 transformers：%s" % exc) from exc
        return torch, peft, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer

    def assert_model_identity(self) -> dict:
        """缺陷 3：模型 repo/revision 必须与**被绑定的输入**同一身份。

        独立审查实测：绑定三份官方小文件后，`TorchPeftBackend(model_id='arbitrary/unapproved',
        revision='arbitrary-revision')` 仍被接受——即"核验了官方 tokenizer/template，
        却加载另一个模型 revision"。这里把两者焊在一起：绑定的 `repo_id`/`revision`
        就是模型加载**唯一**允许的身份。
        """
        if not self.model_inputs:
            raise PolicyViolation(
                "model_inputs_unbound",
                "真实模式必须先绑定被核验的输入（配置/tokenizer/template）再加载模型",
            )
        declared_repo = self.model_inputs.get("repo_id")
        declared_revision = self.model_inputs.get("revision")
        if declared_repo != self.model_id or declared_revision != self.revision:
            raise PolicyViolation(
                "model_identity_mismatch",
                "模型身份与已绑定/已批准的输入不一致：后端 (%s@%s) vs 绑定 (%s@%s)"
                % (self.model_id, self.revision, declared_repo, declared_revision),
                backend_repo=self.model_id,
                backend_revision=self.revision,
                bound_repo=declared_repo,
                bound_revision=declared_revision,
            )
        return {
            "repo_id": declared_repo,
            "revision": declared_revision,
            "identity_bound": True,
            "binding_sha256": self.model_inputs.get("binding_sha256"),
        }

    def _offline_load_dir(self) -> str:
        """缺陷 4：真实加载**只**从固定本地目录离线进行，缺资源即拒（不回退联网）。

        旧实现把 `--allow-download` 当成 GPU 本地加载的先决条件：本地权重明明已在，
        `allow_download=False` 也会抛 `model_download_not_authorized`；而一旦置真，
        两次 `from_pretrained` 又都没有 `local_files_only=True`，可能意外请求 Hub。
        下载许可只作用于**预取阶段**，不参与本地加载判定。
        """
        if not self.local_model_dir:
            raise MissingInput(
                "local_model_dir_missing",
                "真实加载必须给出固定的本地权重目录（--local-model-dir）："
                "缺本地资源即拒绝，不允许联网回退",
            )
        resolved = os.path.realpath(str(self.local_model_dir))
        if not os.path.isdir(resolved):
            raise MissingInput(
                "local_model_dir_missing",
                "本地权重目录不存在或不是目录：%s" % resolved,
                path=resolved,
            )
        identity = self.assert_model_identity()
        expected = (self.model_inputs.get("consumed_inputs", {}).get("config.json", {})
                    .get("sha256"))
        config = os.path.join(resolved, "config.json")
        if not expected or not os.path.isfile(config):
            raise MissingInput("local_model_config_missing", "本地模型配置或官方绑定缺失")
        with open(config, "rb") as handle:
            measured = hashlib.sha256(handle.read()).hexdigest()
        if measured != expected:
            raise PolicyViolation("local_model_config_mismatch", "模型目录自身配置与官方 pin 不符")
        approved = self.model_inputs.get("approved_local_model_dir")
        if not approved or resolved != os.path.realpath(approved):
            raise PolicyViolation("local_model_source_unapproved", "模型规范路径未与批准记录绑定")
        self.model_load_args = {"source": resolved, "config_sha256": measured,
                                "local_files_only": True, **identity}
        return resolved

    def release_base(self, reason: str) -> dict:
        """缺陷 1：受控释放旧基座/optimizer 及全部持有引用，并留下可核对的释放证据。

        独立审查实测：`load_adapter()` 里 `self.model` 在 `from_pretrained` 执行时
        仍是旧模型；工程检查的恢复探针同样在旧基座被持有时再训练一份——**至少**在
        重载瞬间双基座同时驻留，违反 16GB 主机约束。

        这里用**弱引用**证明释放：返回时旧对象若已无可达强引用，弱引用即失效。
        （不声称真机未发生 OOM；真机峰值待首片实测。）
        """
        released: list[str] = []
        evidence: dict = {"reason": reason}
        old_model = self.model
        if old_model is not None:
            try:
                ref = weakref.ref(old_model)
            except TypeError as exc:
                raise PolicyViolation("base_release_unobservable", "旧模型不支持弱引用，无法证明释放") from exc
            self._released_model_refs.append(ref)
            self.model = None
            released.append("model")
            if ref is not None:
                evidence["old_base_ref_alive_after_release"] = ref() is not None
        else:
            evidence["old_base_ref_alive_after_release"] = False
        for attr, label in (
            ("_optimizer", "optimizer"),
            ("_scheduler", "scheduler"),
            ("_scaler", "scaler"),
        ):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
                released.append(label)
        # tokenizer 也一并放下：它同样持有整份词表。
        if self.tokenizer is not None:
            self.tokenizer = None
            released.append("tokenizer")
        del old_model
        gc.collect()
        evidence["old_base_ref_alive_after_release"] = any(
            ref() is not None for ref in self._released_model_refs
        )
        evidence["released"] = released
        evidence["model_is_none"] = self.model is None
        evidence["optimizer_is_none"] = self._optimizer is None
        self.last_release_evidence = evidence
        return evidence

    def assert_base_released(self):
        gc.collect()
        alive = any(ref() is not None for ref in self._released_model_refs)
        if self.last_release_evidence is not None:
            self.last_release_evidence["old_base_ref_alive_after_release"] = alive
        if alive:
            raise PolicyViolation("old_base_still_alive", "旧基座仍被持有，拒绝加载新基座")
        return {"old_base_alive": alive, "only_one_base_resident": not alive}

    def prepare(self, plan: TrainRunPlan) -> dict:
        plan.assert_runnable()
        # 缺陷 B：**明确准备一次**。第二次 prepare 会把 31B 基座再加载一遍，
        # 让"复用同一实例只加载一次"的声明失效，并可能让内存测量测到另一组参数。
        if self.model is not None or self.load_count:
            raise PolicyViolation(
                "backend_already_prepared",
                "该后端实例已经 prepare 过（load_count=%d）：不得重复加载基座；"
                "需要新实例时请显式构造新对象，恢复路径用受控重新实例化" % self.load_count,
                load_count=self.load_count,
            )
        # ---- 缺陷 3：模型身份必须与已绑定/已批准的输入同一 ----
        identity = self.assert_model_identity()
        # ---- 缺陷 4：下载许可**不**参与本地加载判定 ----
        # `--allow-download` 只作用于预取阶段；它既不能放开本地加载，也不能阻止它。
        if self.device.startswith("cuda") and not self.local_load_authorized:
            raise PolicyViolation(
                "local_load_not_authorized",
                "真实（cuda）加载需要工程作业批准 --local-load-authorized；"
                "--allow-download 只授权联网预取，不构成加载批准",
            )
        # ---- 缺陷 4：只从固定本地路径离线加载，缺资源即拒，绝不联网回退 ----
        # 资源检查放在 `_import_stack()` **之前**：缺本地权重是"环境资源"结论，
        # 不应该先被"缺某个库"挡住（两者的处置完全不同）。
        if self.local_model_dir:
            load_source = self._offline_load_dir()
            load_revision = None
            load_mode = "local_offline"
        else:
            if self.device.startswith("cuda"):
                raise MissingInput(
                    "local_model_dir_missing",
                    "cuda 真实加载必须给出 --local-model-dir；缺本地资源即拒绝，不允许联网回退",
                )
            load_source = self.model_id
            load_revision = self.revision
            load_mode = "cpu_fixture_offline"
        torch, _peft, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer = (
            self._import_stack()
        )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise Blocked("cuda_unavailable", "指定 cuda 设备但 torch.cuda.is_available() 为 False")
        self._torch = torch
        # ---- KAGGLE-38 r3 · C：消费者实际读哪份配置/tokenizer/template ----
        tokenizer_source = self._resolve_tokenizer_source()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source["tokenizer_root"],
            revision=None if tokenizer_source["is_load_view"] else self.revision,
            local_files_only=True,
        )
        self.load_count += 1
        base = AutoModelForCausalLM.from_pretrained(
            load_source,
            revision=load_revision,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.load_mode = load_mode
        self.model_identity = identity
        self.model_load_args = {
            "source": str(load_source),
            "revision": load_revision,
            "local_files_only": True,
            "load_mode": load_mode,
            "model_id": self.model_id,
            "declared_revision": self.revision,
        }
        frozen = 0
        for parameter in base.parameters():
            parameter.requires_grad_(False)
            frozen += 1
        from .targets import resolve_peft_scope, verify_peft_scope
        target_scope = resolve_peft_scope(base, self.target_modules, torch.nn.Linear)
        lora_config = LoraConfig(
            r=int(plan.lora_rank),
            lora_alpha=int(plan.lora_alpha),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_scope['target_regex'],
        )
        self.model = get_peft_model(base, lora_config)
        mounted_targets = verify_peft_scope(base, target_scope)
        self.model.to(self.device)
        matched = sorted(
            {
                name.split(".")[-1]
                for name, module in self.model.named_modules()
                if name.endswith("lora_A")
            }
        )
        trainable = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise PolicyViolation("no_trainable_adapter_params", "LoRA 挂载后没有任何可训练参数")
        if not matched:
            raise PolicyViolation("lora_not_mounted", "没有匹配到任何 LoRA target 模块")
        # G14：批次顺序摘要 —— import_training_state 会据此拒绝"数据换了的续训"。
        self.sampler_order_sha256 = _sampler_order_sha256(plan)
        self.lr = float(plan.lr)
        self.parameter_ownership = self.parameter_ownership_snapshot()
        return {
            "backend": self.name,
            "requires_gpu": True,
            "verified": False,
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "dtype": "bfloat16",
            "frozen_base_parameter_tensors": frozen,
            "lora_target_modules": self.target_modules,
            "lora_target_scope": target_scope,
            "lora_mounted_targets": mounted_targets,
            "lora_matched_suffixes": matched,
            "trainable_parameter_tensors": len(trainable),
            "base_frozen_enforced": True,
            "sampler_order_sha256": self.sampler_order_sha256,
            "global_step_start": int(self.global_step),
            "load_count": self.load_count,
            "parameter_ownership": self.parameter_ownership,
            "tokenizer_source": tokenizer_source,
            "consumed_inputs": dict(self.model_inputs.get("consumed_inputs") or {})
            if self.model_inputs
            else {},
            "weights_sha256": "hash_pending",
            "unverified_claims": [
                "本路径未在真实 GPU 上执行过（无 GPU / 无锁定软件）",
                "真实显存峰值、TP4、长上下文行为未测",
                "权重（model.safetensors）的 SHA 明确 hash_pending，未纳入输入绑定",
            ],
        }

    def parameter_ownership_snapshot(self) -> dict:
        """参数归属指纹：证明内存测量与训练用的是**同一组**模型/optimizer 参数。"""
        if self.model is None:
            raise Blocked("backend_not_prepared", "参数归属需要先 prepare")
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        return {
            "model_object_id": id(self.model),
            "optimizer_object_id": id(self._optimizer) if self._optimizer is not None else None,
            "parameter_count": len(list(self.model.parameters())),
            "trainable_parameter_count": len(trainable),
            "parameter_ids_sha256": sha256_json(
                sorted(str(id(parameter)) for parameter in self.model.parameters())
            ),
            "trainable_parameter_ids_sha256": sha256_json(
                sorted(str(id(parameter)) for parameter in trainable)
            ),
        }

    def is_prepared(self) -> bool:
        return self.model is not None

    def assert_prepared(self) -> dict:
        """复用确认：`profile(..., prepare=False)` 用它证明没有偷偷再加载一次。"""
        if self.model is None:
            raise PolicyViolation(
                "backend_not_prepared", "声称复用已准备实例，但该实例其实还没 prepare"
            )
        return {
            "reused": True,
            "load_count": self.load_count,
            "parameter_ownership": self.parameter_ownership,
        }

    def _resolve_tokenizer_source(self) -> dict:
        """决定分词器/模板**实际从哪儿读**（缺陷 C 的落点）。

        - 有被绑定的加载视图 → 指向视图目录（视图里的字节已逐个核对过 pin）；
          此时**不允许**再让 `from_pretrained` 走 hub revision 去取另一份；
        - 没有绑定 → 真实后端一律 fail-closed（`model_inputs_unbound`），
          绝不隐式读现场被改过的 `prep/model`。
        """
        binding = self.model_inputs or {}
        view_dir = binding.get("load_view_dir")
        if view_dir and os.path.isdir(str(view_dir)):
            return {
                "kind": "controlled-load-view",
                "is_load_view": True,
                "tokenizer_root": str(view_dir),
                "tokenizer_config_path": os.path.join(str(view_dir), "tokenizer_config.json"),
                "tokenizer_data_path": os.path.join(str(view_dir), "tokenizer.json"),
                "chat_template_path": os.path.join(str(view_dir), "chat_template.jinja"),
                "binding_sha256": binding.get("binding_sha256"),
                "view_manifest_sha256": binding.get("view_manifest_sha256"),
                "consumed_inputs": dict(binding.get("consumed_inputs") or {}),
                "template_override": binding.get("template_override"),
            }
        raise PolicyViolation(
            "model_inputs_unbound",
            "真实后端没有绑定加载输入（配置/tokenizer/template）：不得一边核验官方原件、"
            "一边让 from_pretrained 隐式读现场可能被改过的工作副本",
            model_id=self.model_id,
            revision=self.revision,
        )

    def train_steps(self, plan: TrainRunPlan, on_step=None) -> dict:
        """真前向 / 真反向 / 真优化器步进，并按 G14 约定支持逐步回调。

        `on_step(info)` 返回非空字符串时在**一步边界**停下，并如实回报
        `stopped` / `stop_reason`；`self.global_step`、数据游标与已消费 token 同步推进，
        因此中途 checkpoint 与续训的账目连续，不会出现"停了却不知道停在哪"。
        """
        if self.model is None:
            raise Blocked("backend_not_prepared", "train_steps 在 prepare 之前被调用")
        torch = self._torch
        self.model.train()
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = self._ensure_optimizer(trainable)
        base_before = (self.adapter_params_digest(), self.base_params_digest())
        losses: list[float] = []
        grad_norms: list[float] = []
        supervised_tokens = 0
        start_step = int(self.start_step)
        stop_reason: str | None = None
        executed = 0
        for offset in range(int(plan.steps)):
            step = start_step + offset
            batch = plan.batches[step % len(plan.batches)]
            input_ids = torch.tensor([batch.input_ids], device=self.device)
            labels = torch.tensor([batch.labels], device=self.device)
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if not bool(torch.isfinite(loss)):
                raise IntegrityError("non_finite_loss", "第 %d 步 loss 非有限值" % step)
            loss.backward()  # 真反向传播
            grad_norm = 0.0
            for parameter in trainable:
                if parameter.grad is not None:
                    grad_norm += float(parameter.grad.detach().float().pow(2).sum().item())
            grad_norm = math.sqrt(grad_norm)
            if grad_norm <= MIN_GRAD_NORM:
                raise IntegrityError(
                    "zero_gradient",
                    "第 %d 步梯度范数为 0：反向没有真的发生" % step,
                )
            optimizer.step()  # 真优化器步进
            optimizer.zero_grad(set_to_none=True)
            if self._scheduler is not None:
                self._scheduler.step()
            value = float(loss.detach().float().item())
            losses.append(value)
            grad_norms.append(grad_norm)
            supervised_tokens += batch.supervised_tokens
            executed += 1
            self.cursor_position = step + 1
            self.consumed_input_tokens += sum(1 for _item in batch.input_ids)
            self.consumed_supervised_tokens += batch.supervised_tokens
            self.global_step = step + 1
            if on_step is not None:
                reason = on_step(
                    {
                        "step": step + 1,
                        "global_step": int(self.global_step),
                        "loss": value,
                        "grad_norm": grad_norm,
                        "optimizer_step_count": executed,
                        "supervised_tokens_seen": supervised_tokens,
                    }
                )
                if reason:
                    stop_reason = str(reason)
                    break
        base_after = self.base_params_digest()
        adapter_after = self.adapter_params_digest()
        if losses:
            summary = {
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_decreased": losses[-1] < losses[0],
                "losses": losses,
                "grad_norms": grad_norms,
                "grad_norm_min": min(grad_norms),
                "grad_norm_nonzero": all(value > MIN_GRAD_NORM for value in grad_norms),
            }
        else:
            summary = {
                "loss_first": None,
                "loss_last": None,
                "loss_decreased": False,
                "losses": losses,
                "grad_norms": grad_norms,
                "grad_norm_min": None,
                "grad_norm_nonzero": False,
            }
        return {
            "backend": self.name,
            "steps": plan.steps,
            "steps_executed": executed,
            "global_step": int(self.global_step),
            "start_step": start_step,
            "stopped": stop_reason is not None,
            "stop_reason": stop_reason,
            "supervised_tokens_seen": supervised_tokens,
            "optimizer": "torch.optim.AdamW（只作用于 requires_grad 的 adapter 参数）",
            "optimizer_step_count": executed,
            "adapter_params_before": base_before[0],
            "adapter_params_after": adapter_after,
            "adapter_params_changed": adapter_after != base_before[0],
            "base_params_before": base_before[1],
            "base_params_after": base_after,
            "base_params_frozen": base_after == base_before[1],
            **summary,
        }

    def _ensure_optimizer(self, trainable):
        """惰性构造优化器并**复用**：续训时状态才回灌得进去（每次新建会丢掉动量）。

        `lr` 来自 `prepare(plan)`，因此本方法在没有 plan 的恢复路径上也能用。
        """
        if self._optimizer is None:
            torch = self._torch
            if torch is None:
                raise Blocked("backend_not_prepared", "优化器需要先 prepare（torch 未加载）")
            self._optimizer = torch.optim.AdamW(trainable, lr=float(getattr(self, "lr", 1e-4)))
        return self._optimizer

    def ensure_scheduler(self, plan: TrainRunPlan, *, total_steps: int | None = None) -> dict:
        """可选：挂一个常数学习率调度器（真实执行面才需要显式调用）。

        本机不执行 —— 只在真实 GPU 作业里由监督器调用；未调用时
        `export_training_state()` 的 `scheduler` 如实为 `None`，不编造。
        """
        if self.model is None:
            raise Blocked("backend_not_prepared", "scheduler 需要先 prepare")
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = self._ensure_optimizer(trainable)
        torch = self._torch
        total = int(total_steps or plan.steps)
        self._scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=total
        )
        return {"scheduler": "ConstantLR", "total_iters": total}

    def _parameter_digest(self, *, trainable: bool) -> str:
        if self.model is None:
            raise Blocked("backend_not_prepared", "adapter 摘要需要先 prepare")
        torch = self._torch
        chunks = []
        for name, parameter in sorted(self.model.named_parameters()):
            # Adapter identity survives inference-mode reload (all tensors frozen).
            is_adapter = any(part in ("lora_A", "lora_B") for part in name.split("."))
            if is_adapter != trainable:
                continue
            chunks.append(name)
            if is_adapter:
                value = parameter.detach().to('cpu').float().contiguous()
                chunks.append({'shape':list(value.shape),
                               'float32_sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()})
            else:
                chunks.append(str(float(parameter.detach().float().sum().item())))
        if not chunks:
            return ""
        return sha256_json(chunks)

    def adapter_params_digest(self) -> str:
        return self._parameter_digest(trainable=True)

    def base_params_digest(self) -> str:
        return self._parameter_digest(trainable=False)

    def observe_full_state_dict(self) -> dict:
        """真实后端的内存纪律观测（🟡-8，**未在 GPU 上实测**）。

        探针真的遍历一次模型参数，但**逐 tensor 单次**累加，从不调用
        `model.state_dict()`（那份会一次性物化全部权重 —— 58GiB 级别，正是硬规则禁止的）。
        `peak_full_state_dict_bytes` 是"本实现里整份驻留的次数 × 权重总量"：
        计数来自 `_full_state_dict_materializations`，任何将来新增的整份材料化都必须
        显式登记，否则计数不动、门槛就会带上"未见材料化"的实测依据。

        诚实边界：**它证明的是代码路径没有整份材料化，不是 RSS/显存实测**。
        真实显存峰值仍需在 107 的作业里测（本机无 GPU，未验证）。
        """
        if self.model is None:
            raise Blocked("backend_not_prepared", "内存纪律观测需要先 prepare")
        total = 0
        largest = 0
        for _name, parameter in self.model.named_parameters():
            numel = int(parameter.numel())
            element_size = int(getattr(parameter, "element_size", lambda: 2)())
            size = numel * element_size
            total += size
            largest = max(largest, size)
        materializations = int(getattr(self, "_full_state_dict_materializations", 0))
        return {
            "probe": "torch-parameter-traversal",
            "method": (
                "逐 tensor 单次遍历 named_parameters()；从不调用 model.state_dict()"
                "（整份材料化计数器 = %d）" % materializations
            ),
            "peak_full_state_dict_bytes": materializations * total,
            "declared_weight_bytes": total,
            "largest_tensor_bytes": largest,
            "full_state_dict_materializations": materializations,
        }

    def save_adapter(self, dest_dir: str, *, adapter_name: str | None = None) -> dict:
        """导出到**官方 PEFT 载体目录**：`<dest_dir>/adapters/<name>/`。

        KAGGLE-27 整改①：官方口径是 `adapters/<adapter_name>/adapter_model.safetensors`
        加同目录 `adapter_config.json`（README §3.4）。旧版把 `save_pretrained` 直接
        写进 `dest_dir` 并去找 `adapter.safetensors` —— 两者都不对：
        PEFT 真实写出的 basename 是 `adapter_model.safetensors`，所以旧检查
        **必定失败**；即便改名，缺 config 时官方 `discover_adapters()` 仍会把
        adapter 名退化成本文件名 stem，声明名解析不到（`AdapterNotFoundError`）。
        """
        if self.model is None:
            raise Blocked("backend_not_prepared", "save_adapter 需要先 prepare")
        name = str(adapter_name or DEFAULT_ADAPTER_NAME).strip()
        peft_dir = os.path.join(dest_dir, *adapter_contract.adapter_dir_relative_path(name).split("/"))
        os.makedirs(peft_dir, exist_ok=True)
        self.model.save_pretrained(peft_dir, safe_serialization=True)
        files = sorted(os.listdir(peft_dir))
        if ADAPTER_FILE not in files:
            raise IntegrityError(
                "adapter_safetensors_missing",
                "save_pretrained 后没有 %s；官方提交载体只接受 .safetensors" % ADAPTER_FILE,
                files=files,
                adapter_dir=peft_dir,
            )
        # 强制 config 存在且可解析 —— 不靠"PEFT 应该会写"来假定。
        config = adapter_contract.assert_adapter_config_present(peft_dir)
        weights = os.path.join(peft_dir, ADAPTER_FILE)
        with open(weights, "rb") as handle:
            raw = handle.read()
        carrier = adapter_contract.assert_official_adapter_carrier(
            dest_dir, declared_adapter_name=name
        )
        return {
            "path": peft_dir,
            "carrier_root": dest_dir,
            "files": files,
            "filename": ADAPTER_FILE,
            "relative": adapter_contract.carrier_weights_relative_path(name),
            "adapter_name": name,
            "adapter_config": config,
            "bytes": len(raw),
            "sha256": _sha256_bytes(raw),
            "adapter_only": True,
            "contains_base_weights": False,
            "is_official_submission_carrier": True,
            "carrier_contract": carrier,
        }

    def load_adapter(self, src_dir: str, *, adapter_name: str | None = None,
                     is_trainable: bool = False) -> dict:
        """Integrity reload is inference-only by default; resumption is explicit."""
        module = self._import_stack()
        _torch, _peft, _LoraConfig, _get_peft_model, AutoModelForCausalLM, _AutoTokenizer = module
        from peft import PeftModel  # type: ignore

        name = str(adapter_name or DEFAULT_ADAPTER_NAME).strip()
        peft_dir = os.path.join(src_dir, *adapter_contract.adapter_dir_relative_path(name).split("/"))
        weights = os.path.join(peft_dir, ADAPTER_FILE)
        if not os.path.exists(weights):
            raise MissingInput("adapter_file_missing", "找不到 adapter：%s" % weights)
        adapter_contract.assert_official_adapter_carrier(src_dir, declared_adapter_name=name)
        before = self.adapter_params_digest()
        # ---- 缺陷 1：**先释放旧基座**再加载新基座，避免重载瞬间双基座同时驻留 ----
        # 旧实现把 `self.model` 留到 `from_pretrained` 之后才换掉，于是新基座加载
        # 全程旧基座仍然可达（独立审查实测 `old_base_live_at_fresh_load [True]`）。
        release_evidence = self.release_base("adapter_reload")
        observed = self.assert_base_released()
        identity = self.assert_model_identity()
        if self.local_model_dir:
            load_source = self._offline_load_dir()
            load_revision = None
        else:
            if str(self.device).startswith("cuda"):
                raise MissingInput("local_model_dir_missing", "CUDA 重载必须使用批准的本地目录")
            load_source = self.model_id
            load_revision = self.revision
        fresh_base = AutoModelForCausalLM.from_pretrained(
            load_source,
            revision=load_revision,
            torch_dtype=self._torch.bfloat16,
            local_files_only=True,
        )
        self.load_count += 1
        reloaded = PeftModel.from_pretrained(fresh_base, peft_dir, is_trainable=is_trainable)
        reloaded.to(self.device)
        self.model = reloaded
        for parameter_name, parameter in self.model.named_parameters():
            adapter_parameter = any(part in ('lora_A','lora_B') for part in parameter_name.split('.'))
            if bool(parameter.requires_grad) != bool(is_trainable and adapter_parameter):
                raise PolicyViolation('adapter_reload_trainability_mismatch',
                                      'Reload changed the requested adapter/base freezing scope',
                                      parameter=parameter_name)
        self.parameter_ownership = self.parameter_ownership_snapshot()
        after = self.adapter_params_digest()
        if after != before:
            raise IntegrityError(
                "adapter_reload_mismatch",
                "重载后的 adapter 参数摘要与保存前不一致",
                before=before,
                after=after,
            )
        return {
            "path": peft_dir,
            "adapter_name": name,
            "reloaded_adapter_params": after,
            "matches_saved_adapter": True,
            "is_trainable": bool(is_trainable),
            "optimizer_restored": False,
            "base_reloaded_from_scratch": True,
            # 缺陷 1 证据：释放发生在 `from_pretrained` **之前**，且旧基座弱引用已失效。
            "old_base_release_evidence": release_evidence,
            "old_base_alive_during_fresh_load": observed["old_base_alive"],
            "model_load_args": getattr(self, "model_load_args", {}),
            "model_identity": identity,
            "local_files_only": True,
        }

    # ---- G14：adapter 字节导出 / 训练状态导出导入 / 逐阶段测量 ----

    def _adapter_tensors(self) -> dict:
        """只取 adapter（`requires_grad`）张量。

        **不调用** `model.state_dict()` —— 那份会一次性物化整个基座（硬规则禁止），
        所以这里的取用方式是逐参数 `named_parameters()` 过滤，与
        `observe_full_state_dict()` 的材料化计数器口径一致（本路径恒为 0 次）。
        """
        if self.model is None:
            raise Blocked("backend_not_prepared", "adapter 张量需要先 prepare")
        tensors: dict = {}
        for name, parameter in sorted(self.model.named_parameters()):
            if not bool(parameter.requires_grad):
                continue
            tensors[name] = parameter.detach().to("cpu").float().contiguous()
        if not tensors:
            raise PolicyViolation(
                "no_trainable_adapter_params", "没有可导出的 adapter 参数"
            )
        return tensors

    def export_adapter_bytes(self) -> bytes:
        """导出 adapter 的**字节表示**（G14：中途 checkpoint 里要放一份 adapter 副本）。

        与 :meth:`save_adapter` 同源：两者都只取 `requires_grad` 的参数。编码选
        float32 + base64 的自描述 JSON，不依赖 `safetensors` 版本，跨机器可逐字节复算。
        """
        tensors = self._adapter_tensors()
        payload = {
            "format": "torch-peft-adapter-bytes/1",
            "adapter_only": True,
            "contains_base_weights": False,
            "is_official_submission_carrier": False,
            "tensor_encoding": "float32-little-endian-base64",
            "tensors": {
                name: {
                    "shape": [int(item) for item in tuple(value.shape)],
                    "dtype": "float32",
                    "sha256": _sha256_bytes(value.numpy().tobytes()),
                    "data_base64": base64.b64encode(value.numpy().tobytes()).decode("ascii"),
                }
                for name, value in tensors.items()
            },
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def _cuda_rng_api(self):
        """CUDA RNG 读数接口（缺陷 E）。可注入：测试用替身证明 CPU 端接线正确。

        真实执行面就是 `torch.cuda`；本机没卡时返回 None，状态如实标 unavailable ——
        但**接线必须真的存在**：只存 `torch.get_rng_state()`（CPU）会漏掉实际参与
        dropout / 数据增强的**设备** RNG，续训因此不可复现。
        """
        injected = getattr(self, "cuda_rng", None)
        if injected is not None:
            return injected
        if self._torch is None:
            return None
        return getattr(self._torch, "cuda", None)

    def _cuda_device_in_use(self, cuda) -> bool:
        """本后端是否**真的**在 CUDA 设备上跑（决定 RNG 取证能不能降级）。"""
        self.training_devices.add(str(self.device))
        return any(device.startswith("cuda") for device in self.training_devices)

    def _cuda_rng_states(self) -> dict:
        """逐 CUDA 设备的 RNG 状态。

        缺陷 5（独立审查实测）：注入 `is_available=True` / `device_count=1` /
        `get_rng_state()` 抛错时，旧实现把异常降级成
        `{'available': False, 'reason': 'unavailable:RuntimeError', 'devices': {}}` ——
        真 CUDA 作业可以在**随机状态不完整**的情况下继续保存，并显示 checkpoint/恢复成功。

        现在的口径：

        - 真实 CUDA 设备参与训练（`device` 是 cuda 且设备可用）时，任何采集失败
          **fail-closed**：`MissingInput("cuda_rng_capture_failed")`，不产出半份状态；
        - 没有 CUDA 设备的 CPU fixture 如实标 `available: False` 并给出 reason
          （这不是"失败"，是明确 not-applicable）。
        """
        cuda = self._cuda_rng_api()
        in_use = self._cuda_device_in_use(cuda)
        if in_use:
            try:
                if cuda is None or not cuda.is_available() or cuda.device_count() <= 0:
                    raise RuntimeError("CUDA unavailable")
            except Exception as exc:
                raise MissingInput("cuda_rng_capture_failed", "已选择 CUDA，但 RNG API 不可用") from exc
        if cuda is None or not bool(getattr(cuda, "is_available", lambda: False)()):
            return {
                "available": False,
                "reason": "not-applicable:no-cuda-device",
                "devices": {},
                "cuda_in_use": False,
            }
        try:
            count = int(cuda.device_count())
        except Exception as exc:  # noqa: BLE001 - 设备相关
            if in_use:
                raise MissingInput(
                    "cuda_rng_capture_failed",
                    "真实 CUDA 参与训练但无法读取设备数（%s）：拒绝保存不完整随机状态"
                    % type(exc).__name__,
                    device=self.device,
                    error=type(exc).__name__,
                ) from exc
            return {
                "available": False,
                "reason": "not-applicable:cuda-query-failed:%s" % type(exc).__name__,
                "devices": {},
                "cuda_in_use": False,
            }
        if count <= 0:
            return {
                "available": False,
                "reason": "not-applicable:device-count-zero",
                "devices": {},
                "cuda_in_use": False,
            }
        if not in_use:
            # 设备存在但本次不在 cuda 上训练：不必采，也不能冒充"已采集"。
            return {
                "available": False,
                "reason": "not-applicable:device-not-in-use",
                "devices": {},
                "cuda_in_use": False,
            }
        try:
            states = {}
            for index in range(count):
                raw = cuda.get_rng_state(index)
                states[str(index)] = {
                    "encoding": "torch.cuda.get_rng_state",
                    "device": index,
                    "data_base64": base64.b64encode(
                        bytes(raw.detach().cpu().numpy().tobytes())
                    ).decode("ascii"),
                }
        except Exception as exc:  # noqa: BLE001 - 设备相关
            raise MissingInput(
                "cuda_rng_capture_failed",
                "真实 CUDA 参与训练但逐设备 RNG 采集失败（%s）：拒绝保存不完整随机状态，"
                "不允许降级为 available=false 后继续" % type(exc).__name__,
                device=self.device,
                error=type(exc).__name__,
                devices_requested=count,
            ) from exc
        if len(states) != count:
            raise MissingInput(
                "cuda_rng_capture_incomplete",
                "真实 CUDA 参与训练但只采到 %d/%d 个设备的 RNG：拒绝保存不完整随机状态"
                % (len(states), count),
                device=self.device,
                captured=sorted(states),
                device_count=count,
            )
        return {
            "available": True,
            "training_devices": sorted(self.training_devices),
            "device_count": count,
            "devices": states,
            "cuda_in_use": True,
        }

    def _restore_cuda_rng_states(self, payload) -> list:
        """回灌逐设备 RNG 状态；返回真正恢复的设备列表。

        缺陷 5：真实 CUDA 参与训练时，checkpoint 缺设备状态必须**拒绝**，
        不能静默跳过（那会让恢复轨迹不等价却显示成功）。
        """
        cuda = self._cuda_rng_api()
        if isinstance(payload, Mapping):
            self.training_devices.update(payload.get("training_devices") or [])
            if payload.get("cuda_in_use") or payload.get("available"):
                self.training_devices.add("cuda")
        in_use = self._cuda_device_in_use(cuda)
        if not isinstance(payload, Mapping) or not payload.get("available"):
            if in_use:
                raise MissingInput(
                    "cuda_rng_missing_in_checkpoint",
                    "真实 CUDA 参与训练但 checkpoint 里没有设备 RNG 状态"
                    "（reason=%r）：拒绝只恢复一半的随机轨迹"
                    % ((payload or {}).get("reason") if isinstance(payload, Mapping) else payload),
                    device=self.device,
                )
            return []
        try:
            available = cuda is not None and cuda.is_available()
        except Exception as exc:
            raise MissingInput("cuda_rng_unavailable_for_restore", "CUDA API 查询失败") from exc
        if not available:
            raise MissingInput(
                "cuda_rng_unavailable_for_restore",
                "checkpoint 带 CUDA RNG 状态，但当前没有可用的 CUDA 接口：拒绝只恢复一半",
            )
        import numpy  # noqa: PLC0415

        restored = []
        states = payload.get("devices") or {}
        count = int(payload.get("device_count", 0))
        if count <= 0 or set(states) != {str(i) for i in range(count)} or cuda.device_count() < count:
            raise MissingInput("cuda_rng_missing_in_checkpoint", "CUDA 设备状态缺失或设备不匹配")
        for key, item in sorted((payload.get("devices") or {}).items()):
            if not isinstance(item, Mapping):
                raise MissingInput(
                    "cuda_rng_state_invalid", "CUDA RNG 状态条目不是对象：device=%s" % key
                )
            raw = numpy.frombuffer(
                base64.b64decode(str(item["data_base64"])), dtype=numpy.uint8
            ).copy()
            index = int(item.get("device", key))
            cuda.set_rng_state(self._torch.from_numpy(raw), index)
            restored.append(index)
        if in_use and not restored:
            raise MissingInput(
                "cuda_rng_missing_in_checkpoint",
                "真实 CUDA 参与训练但 checkpoint 的设备 RNG 状态为空：拒绝恢复",
                device=self.device,
            )
        return restored

    def _rng_state(self) -> dict:
        """四份 RNG 状态（torch-CPU / torch-CUDA / numpy / python）；取不到的**如实**标 unavailable。"""
        rng: dict = {
            "training_devices": sorted(self.training_devices | {str(self.device)}),
            "python_state": python_rng_state(random.getstate()),
            "numpy_state": "unavailable:no-numpy",
            "torch_state": "unavailable:no-torch",
            "torch_cuda_states": {"available": False, "reason": "unavailable:no-torch", "devices": {}},
            "rng_scope": "torch-cpu+torch-cuda+numpy+python",
            "device_rng_scope": str(self.device),
        }
        try:
            import numpy  # noqa: PLC0415

            state = numpy.random.get_state()
            rng["numpy_state"] = {
                "encoding": "numpy.random.get_state",
                "legacy": str(state[0]),
                "keys_base64": base64.b64encode(state[1].astype("<u4").tobytes()).decode("ascii"),
                "pos": int(state[2]),
                "has_gauss": int(state[3]),
                "cached_gaussian": float(state[4]),
            }
        except Exception as exc:  # noqa: BLE001 - 环境相关
            rng["numpy_state"] = "unavailable:%s" % type(exc).__name__
        if self._torch is not None:
            raw = self._torch.get_rng_state()
            rng["torch_state"] = {
                "encoding": "torch.get_rng_state",
                "data_base64": base64.b64encode(
                    bytes(raw.detach().cpu().numpy().tobytes())
                ).decode("ascii"),
            }
        rng["torch_cuda_states"] = self._cuda_rng_states()
        return rng

    def export_training_state(self) -> dict:
        """导出可 JSON 化的完整续训状态：optimizer / scheduler / scaler / RNG / 游标。

        - `optimizer`：`AdamW.state_dict()`（含 `step`、`exp_avg`、`exp_avg_sq`），
          张量经 `lifecycle.jsonify_tensors` 编成 base64；
        - `scheduler` / `scaler`：**没有挂就如实写 `None`**，并用
          `*_status = "not-applicable:..."` 明确"不是漏存"；挂了就必须完整导出，
          `import_training_state` 对"checkpoint 有状态但本后端没挂"的错配直接拒绝；
        - `rng`：torch-CPU / torch-**CUDA 逐设备** / numpy / python 四份一起给；
        - `cursor`：批次顺序摘要 + 位置 + global_step（`import_training_state` 强制核对）。
        """
        if self.model is None:
            raise Blocked("backend_not_prepared", "导出训练状态需要先 prepare")
        from .lifecycle import jsonify_tensors  # noqa: PLC0415 - 避免模块级循环导入

        optimizer_state = (
            jsonify_tensors(self._optimizer.state_dict(), where="optimizer")
            if self._optimizer is not None
            else None
        )
        scheduler_state = (
            jsonify_tensors(self._scheduler.state_dict(), where="scheduler")
            if self._scheduler is not None
            else None
        )
        scaler_state = (
            jsonify_tensors(self._scaler.state_dict(), where="scaler")
            if self._scaler is not None
            else None
        )
        rng = self._rng_state()
        cuda_states = rng.get("torch_cuda_states") or {}
        return {
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "scheduler_status": (
                "exported"
                if scheduler_state is not None
                else "not-applicable:no-scheduler-wired"
            ),
            "scaler": scaler_state,
            "scaler_status": (
                "exported" if scaler_state is not None else "not-applicable:no-amp-scaler"
            ),
            "rng": rng,
            "cursor": {
                "sampler_order_sha256": self.sampler_order_sha256,
                "position": int(self.cursor_position),
                "global_step": int(self.global_step),
            },
            "consumed_input_tokens": int(self.consumed_input_tokens),
            "consumed_supervised_tokens": int(self.consumed_supervised_tokens),
            "accum_boundary": int(self.accum_boundary),
            "global_step": int(self.global_step),
            "rng_scope": "torch-cpu+torch-cuda+numpy+python",
            "device_rng_scope": str(self.device),
            "cuda_rng_available": bool(cuda_states.get("available")),
            "tensor_encoding": "base64-json（lifecycle.jsonify_tensors）",
        }

    def import_training_state(self, state) -> dict:
        """把 checkpoint 里的状态灌回后端；游标与 RNG **一起**恢复，不可只恢复其一。

        负例（都必须拒绝，不得静默继续）：optimizer 状态缺失、游标缺失、
        批次顺序摘要与当前 plan 不一致（说明换了数据）、RNG 结构不对。
        """
        if self.model is None:
            raise Blocked("backend_not_prepared", "导入训练状态需要先 prepare")
        from .lifecycle import unjsonify_tensors  # noqa: PLC0415

        rng = state.rng or {}
        self.training_devices.update(rng.get("training_devices") or [])
        self.training_devices.add(str(rng.get("device_rng_scope", self.device)))
        cursor = state.cursor or {}
        if not cursor.get("sampler_order_sha256") or cursor.get("position") is None:
            raise MissingInput("cursor_incomplete", "checkpoint 里没有可用的数据游标")
        if (
            self.sampler_order_sha256
            and cursor["sampler_order_sha256"] != self.sampler_order_sha256
        ):
            raise IntegrityError(
                "sampler_order_mismatch",
                "checkpoint 的批次顺序摘要与当前 plan 不一致：不得续训（数据换了）",
                expected=self.sampler_order_sha256,
                actual=cursor["sampler_order_sha256"],
            )
        if state.optimizer is None:
            raise MissingInput(
                "optimizer_state_missing",
                "checkpoint 里没有 optimizer 状态：续训不得只恢复 RNG 与游标",
            )
        adapter_restored = False
        adapter_bytes = getattr(state, 'adapter_bytes', None)
        if getattr(state, 'checkpoint_dir', None):
            if adapter_bytes is None or _sha256_bytes(adapter_bytes) != state.adapter_sha256:
                raise IntegrityError('checkpoint_adapter_missing', 'Missing or changed checkpoint adapter bytes')
            payload = json.loads(adapter_bytes.decode('utf-8'))
            if payload.get('format') != 'torch-peft-adapter-bytes/1' or payload.get('contains_base_weights'):
                raise IntegrityError('checkpoint_adapter_format', 'Checkpoint is not a torch adapter-only payload')
            params = {name:p for name,p in self.model.named_parameters()
                      if any(part in ('lora_A','lora_B') for part in name.split('.'))}
            if set(payload.get('tensors', {})) != set(params):
                raise IntegrityError('checkpoint_adapter_keys', 'Checkpoint adapter identity differs from model')
            import numpy
            with self._torch.no_grad():
                for name,param in params.items():
                    item = payload['tensors'][name]
                    raw = base64.b64decode(item['data_base64'], validate=True)
                    if list(param.shape) != item['shape'] or _sha256_bytes(raw) != item['sha256']:
                        raise IntegrityError('checkpoint_adapter_tensor', 'Checkpoint tensor changed: '+name)
                    array = numpy.frombuffer(raw, dtype='<f4').copy().reshape(item['shape'])
                    param.copy_(self._torch.from_numpy(array).to(device=param.device, dtype=param.dtype))
            adapter_restored = True
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        optimizer = self._ensure_optimizer(trainable)
        torch = self._torch
        optimizer_state = unjsonify_tensors(state.optimizer, decoder=_torch_decoder(torch))
        # JSON object keys are strings, but AdamW maps moments by integer IDs.
        # Leaving "0" as a string silently detaches its moments from new params.
        moments = optimizer_state.get('state', {})
        try:
            ids = {int(key):value for key,value in moments.items()}
        except (ValueError, TypeError) as exc:
            raise IntegrityError('optimizer_state_invalid', 'Invalid optimizer parameter IDs') from exc
        if len(ids) != len(moments):
            raise IntegrityError('optimizer_state_invalid', 'Duplicate normalized optimizer parameter IDs')
        optimizer_state['state'] = ids
        optimizer.load_state_dict(optimizer_state)
        if state.scheduler is not None:
            if self._scheduler is None:
                raise MissingInput(
                    "scheduler_not_wired",
                    "checkpoint 带 scheduler 状态但本后端没有挂 scheduler：拒绝只恢复一半",
                )
            self._scheduler.load_state_dict(
                unjsonify_tensors(state.scheduler, decoder=_torch_decoder(torch))
            )
        if state.scaler is not None:
            if self._scaler is None:
                raise MissingInput(
                    "scaler_not_wired",
                    "checkpoint 带 scaler 状态但本后端没有 AMP scaler：拒绝只恢复一半",
                )
            self._scaler.load_state_dict(unjsonify_tensors(state.scaler, decoder=_torch_decoder(torch)))
        # RNG：三份分别恢复；有状态就恢复，标 unavailable 的如实跳过（不编造）。
        python_state = rng.get("python_state")
        if not isinstance(python_state, Mapping):
            raise MissingInput(
                "rng_python_state_missing", "checkpoint 里没有可用的 Python RNG 状态"
            )
        random.setstate(python_rng_restore(python_state))
        restored_rng = ["python"]
        numpy_state = rng.get("numpy_state")
        if isinstance(numpy_state, Mapping):
            import numpy  # noqa: PLC0415

            keys = numpy.frombuffer(
                base64.b64decode(str(numpy_state["keys_base64"])), dtype="<u4"
            ).astype(numpy.uint32)
            numpy.random.set_state(
                (
                    str(numpy_state.get("legacy") or "MT19937"),
                    keys,
                    int(numpy_state["pos"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
            restored_rng.append("numpy")
        torch_state = rng.get("torch_state")
        if isinstance(torch_state, Mapping):
            import numpy  # noqa: PLC0415

            raw = numpy.frombuffer(
                base64.b64decode(str(torch_state["data_base64"])), dtype=numpy.uint8
            ).copy()
            torch.set_rng_state(torch.from_numpy(raw))
            restored_rng.append("torch")
        # 缺陷 E：CUDA 逐设备 RNG 必须一起恢复 —— 只恢复 CPU 状态会让
        # dropout / 数据增强在续训时从头抽另一条随机流，续训不可复现。
        cuda_restored = self._restore_cuda_rng_states(rng.get("torch_cuda_states"))
        restored_rng.extend("cuda:%d" % index for index in cuda_restored)
        self.global_step = int(state.global_step)
        self.start_step = int(state.global_step)
        self.cursor_position = int(cursor["position"])
        self.consumed_input_tokens = int(state.consumed_input_tokens)
        self.consumed_supervised_tokens = int(state.consumed_supervised_tokens)
        self.accum_boundary = int(state.accum_boundary)
        self.resumed_state_applied = True
        self.parameter_ownership = self.parameter_ownership_snapshot()
        cuda_block = rng.get("torch_cuda_states") or {}
        return {
            "applied": True,
            "global_step": self.global_step,
            "cursor_position": self.cursor_position,
            "sampler_order_sha256": cursor["sampler_order_sha256"],
            "optimizer_state_restored": True,
            "checkpoint_adapter_restored": adapter_restored,
            "parameter_ownership": self.parameter_ownership,
            "scheduler_state_restored": bool(state.scheduler is not None),
            "scheduler_status": getattr(state, "scheduler_status", None),
            "scaler_status": getattr(state, "scaler_status", None),
            "rng_restored": restored_rng,
            "rng_not_available_in_checkpoint": [
                name
                for name in ("numpy", "torch")
                if not isinstance(rng.get("%s_state" % name), Mapping)
            ],
            "cuda_rng_available_in_checkpoint": bool(cuda_block.get("available")),
            "cuda_rng_restored_devices": cuda_restored,
            "cuda_rng_note": (
                "本 checkpoint 记录了逐设备 CUDA RNG 状态"
                if cuda_block.get("available")
                else "本 checkpoint 没有 CUDA RNG 状态（%s）：若真实执行在 CUDA 上，"
                "这是一条必须补齐的缺口" % cuda_block.get("reason")
            ),
        }

    def measure_phases(self, plan: TrainRunPlan, probe) -> dict:
        """G7b：把一个真实训练步拆成 前向 / 反向 / 优化器步进，逐阶段交给探针测量。

        只跑**一个**步（内存峰值关心驻留而非步数）；跑完把这一步算进游标与计数，
        测量本身不会让状态账目对不上。
        """
        if self.model is None:
            raise Blocked("backend_not_prepared", "measure_phases 在 prepare 之前被调用")
        torch = self._torch
        self.model.train()
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        optimizer = self._ensure_optimizer(trainable)
        step = int(self.start_step)
        batch = plan.batches[step % len(plan.batches)]
        input_ids = torch.tensor([batch.input_ids], device=self.device)
        labels = torch.tensor([batch.labels], device=self.device)
        with probe.phase("forward"):
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        if not bool(torch.isfinite(loss)):
            raise IntegrityError("non_finite_loss", "测量步 loss 非有限值")
        with probe.phase("backward"):
            loss.backward()
            grad_norm = 0.0
            for parameter in trainable:
                if parameter.grad is not None:
                    grad_norm += float(parameter.grad.detach().float().pow(2).sum().item())
            grad_norm = math.sqrt(grad_norm)
        if grad_norm <= MIN_GRAD_NORM:
            raise IntegrityError("zero_gradient", "测量步梯度范数为 0：反向没有真的发生")
        with probe.phase("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.global_step = step + 1
        self.cursor_position = step + 1
        self.consumed_input_tokens += sum(1 for _item in batch.input_ids)
        self.consumed_supervised_tokens += batch.supervised_tokens
        return {
            "phases_measured": ["forward", "backward", "optimizer_step"],
            "loss": float(loss.detach().float().item()),
            "grad_norm": grad_norm,
            "global_step": int(self.global_step),
            "note": "真实后端的分解测量：三个阶段的读数都来自**模型真的执行**。",
        }


def _torch_decoder(torch):
    """`lifecycle.unjsonify_tensors` 用的张量解码器（缺 torch 就 fail-closed）。"""
    if torch is None:
        raise Blocked("torch_unavailable_for_state_decode", "解码张量状态需要 torch")
    import numpy  # noqa: PLC0415

    def decode(payload: Mapping):
        raw = base64.b64decode(str(payload["data_base64"]))
        array = numpy.frombuffer(
            raw, dtype=numpy.dtype(str(payload.get("numpy_dtype") or "float32"))
        )
        shape = tuple(int(item) for item in payload.get("shape") or ())
        if shape:
            array = array.reshape(shape)
        return torch.from_numpy(array.copy())

    return decode


def _train_steps_supports_on_step(backend: TrainBackend) -> bool:
    """后端是否支持 `train_steps(plan, on_step=...)`（旧后端只有 `(plan)`）。"""
    try:
        parameters = inspect.signature(backend.train_steps).parameters
    except (TypeError, ValueError):  # pragma: no cover - 内建函数等
        return False
    if "on_step" in parameters:
        return True
    return any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())


def run_training(
    backend: TrainBackend,
    plan: TrainRunPlan,
    dest_dir: str,
    *,
    extra_hashes: Mapping | None = None,
    full_state_dict_observation: Mapping | None = None,
    weight_manifest: Mapping | None = None,
    lifecycle=None,
    stop=None,
) -> dict:
    """编排一次完整训练：prepare → train → **内存纪律门槛** → adapter-only 保存 →
    重载校验 → 出证据。

    任一步 fail-closed；返回体里 `executed=True` 只在**整条链路真的跑完**时出现。

    🟡-8：`assert_no_full_state_dict` 曾经只在 tests 里被调用（Y9 的第三处没有生产
    接线）。现在保存 adapter **之前**必经 `assert_full_state_dict_guard(...)`：
    观测优先取调用方传入的 `full_state_dict_observation`，否则问后端要
    （`backend.observe_full_state_dict()`）；两者都拿不到就 `PolicyViolation`，
    **不会**跳过门槛、也不会用自述结论顶替观测。

    G14（本轮新增）：传入 `lifecycle`（:class:`v3.train.lifecycle.TrainingLifecycle`）时，
    训练循环被接上 checkpoint 保存/恢复与停止处理：

    - `resume_from` 时先 `checkpoint.resume()` 校验环境哈希，再把 optimizer /
      scheduler / scaler / RNG / 游标 / global_step 灌回后端；
    - 每步回调按节奏落中途 checkpoint；收到停止请求（信号 / 截止 / 步数上限）时
      **在一步边界停下**，落一份 `stop:<reason>` checkpoint，然后照常保存 adapter
      并**重载校验** —— 停止不等于跳过完整性检查；
    - `base_params_frozen`（基座冻结）在所有路径上都必须成立；`grad_norm_nonzero`
      只有在**至少完成一步优化器步进**时才作为硬条件（0 步就停不算"零梯度失败"，
      但会在报告里如实写出 `zero_step_stop=True`）。
    """
    plan.assert_runnable()
    if stop is None and lifecycle is not None:
        stop = getattr(lifecycle, "stop", None)
    if stop is not None and lifecycle is not None:
        stop.install()
    prepared = backend.prepare(plan)
    lifecycle_resume = lifecycle.begin(backend) if lifecycle is not None else None

    on_step = lifecycle.after_step if lifecycle is not None else None
    if on_step is not None and not _train_steps_supports_on_step(backend):
        raise PolicyViolation(
            "backend_lacks_step_hook",
            "后端 %s 的 train_steps 不接受 on_step：无法接线中途 checkpoint/停止，"
            "不得假装已经接上" % type(backend).__name__,
        )
    if on_step is not None:
        trained = backend.train_steps(plan, on_step=lambda info: on_step(backend, info))
    else:
        trained = backend.train_steps(plan)

    stopped = bool(trained.get("stopped"))
    optimizer_steps = int(trained.get("optimizer_step_count") or 0)
    integrity_notes: list[str] = []
    if not trained.get("grad_norm_nonzero"):
        if optimizer_steps > 0:
            raise IntegrityError("zero_gradient", "训练过程中出现过零梯度，不得当作成功")
        integrity_notes.append(
            "停止发生在任何优化器步进之前（optimizer_step_count=0）："
            "本条不构成零梯度证据，也不构成训练成功证据"
        )
    if not trained.get("base_params_frozen", False):
        raise PolicyViolation(
            "base_params_changed",
            "基座参数发生了变化：adapter-only 训练被破坏",
        )
    observation = full_state_dict_observation
    if observation is None:
        observation = backend.observe_full_state_dict()
    memory_guard = assert_full_state_dict_guard(observation, weight_manifest=weight_manifest)
    adapter_name = str(getattr(plan, "adapter_name", "") or DEFAULT_ADAPTER_NAME).strip()
    saved = backend.save_adapter(dest_dir, adapter_name=adapter_name)
    if not saved.get("adapter_only") or saved.get("contains_base_weights"):
        raise PolicyViolation(
            "adapter_payload_not_adapter_only",
            "保存产物不是纯 adapter，禁止把完整权重当 adapter 交付",
        )
    final_step = int(
        trained.get("global_step")
        or getattr(backend, "global_step", 0)
        or optimizer_steps
    )
    lifecycle_report = (
        lifecycle.finalize(backend, step=final_step, trained=trained)
        if lifecycle is not None
        else None
    )
    # Persist moments/RNG/cursor while the trained model and optimizer still
    # exist. Reload is a destructive, single-base integrity check, not a resume.
    restore_after_reload = isinstance(backend, TorchPeftBackend) and lifecycle_report is not None
    if restore_after_reload:
        checkpoints = lifecycle_report['checkpoints']
        if not checkpoints or checkpoints[-1]['step'] != final_step:
            raise PolicyViolation('reload_requires_final_checkpoint',
                                  'Refuse to release trained state without a checkpoint at the final step')
        reloaded = backend.load_adapter(dest_dir, adapter_name=adapter_name, is_trainable=True)
        latest = lifecycle_report['checkpoints'][-1]['checkpoint_dir']
        reloaded['training_state_restored'] = backend.import_training_state(lifecycle.store.load(latest))
    else:
        reloaded = backend.load_adapter(dest_dir, adapter_name=adapter_name)
    report = {
        "backend": backend.name,
        "requires_gpu": backend.requires_gpu,
        "verified_on_this_machine": backend.verified,
        "executed": True,
        "stopped": stopped,
        "stop_reason": trained.get("stop_reason"),
        "zero_step_stop": bool(stopped and optimizer_steps == 0),
        "integrity_notes": integrity_notes,
        "prepared": prepared,
        "trained": trained,
        "memory_guard": memory_guard,
        "saved": saved,
        "reloaded": reloaded,
        "dest_dir": os.path.abspath(dest_dir),
        "lifecycle": lifecycle_report,
        "resume": lifecycle_resume,
        "evidence_hashes": dict(extra_hashes or {}),
        "unverified_claims": list(prepared.get("unverified_claims") or []),
    }
    if lifecycle_report is not None:
        report["unverified_claims"] = list(report["unverified_claims"]) + [
            "checkpoint 的写盘/哈希校验已通过，但**未**在真实 GPU 上验证过恢复训练",
        ]
    os.makedirs(dest_dir, exist_ok=True)
    evidence_path = os.path.join(dest_dir, EVIDENCE_FILE)
    with open(evidence_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    report["evidence_path"] = evidence_path
    if stop is not None and lifecycle is not None:
        stop.uninstall()
    return report


def build_smoke_batches(*, count: int = 4, seq_len: int = 12, vocab_cap: int = 24) -> list[TrainBatch]:
    """确定性的合成批次（无 gold、无真实数据、无网络）。"""
    batches = []
    for index in range(count):
        tokens = [((index * 5 + position * 3) % vocab_cap) + 2 for position in range(seq_len)]
        labels = list(tokens)
        for position in range(0, seq_len, 4):
            labels[position] = -100  # 模拟 system/padding 位的忽略
        batches.append(TrainBatch(input_ids=tokens, labels=labels))
    return batches


def self_check_cpu(*, steps: int = 40, workdir: str | None = None, keep_artifacts: bool = False) -> dict:
    """CPU 训练循环自检：真跑一遍前向/反向/步进/保存/重载，返回证据。

    这是 `entry.start()` 里 `t1_engineering_pass` 闸门的**测量来源**，
    也是 CLI `--smoke` 的实现。它证明"训练循环是活的"，**不**证明任何训练收益。

    产物默认写在仓库内的 `v3/train/_smoke_run`（沙箱只允许写工作区），
    跑完即清理；`keep_artifacts=True` 时保留，供人工复核。
    """
    plan = TrainRunPlan(
        steps=steps,
        lr=0.3,
        seq_len=12,
        lora_rank=4,
        lora_alpha=8,
        batches=build_smoke_batches(seq_len=12),
    )
    backend = SyntheticBackend(vocab=32, dim=8)
    dest = workdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke_run")
    report = run_training(backend, plan, dest)
    trained = report["trained"]
    checks = {
        "forward_backward_ran": bool(trained["steps"]) and trained["grad_norm_nonzero"],
        "optimizer_stepped": trained["optimizer_step_count"] == steps,
        "loss_finite": math.isfinite(trained["loss_last"]),
        "base_frozen": bool(trained["base_params_frozen"]),
        "adapter_changed": bool(trained["adapter_params_changed"]),
        "adapter_only_saved": bool(report["saved"]["adapter_only"]),
        "reload_matches": bool(report["reloaded"]["matches_saved_adapter"]),
    }
    # 注意：**不**把 `loss_decreased` 放进通过条件。loss 是否下降取决于步数与
    # 学习率，不是"训练循环正确"的不变量；把它当门槛会造出一个会骗人的绿灯。
    # 它仍如实出现在报告里，供人判断。
    result = {
        "stage": "cpu-training-loop-self-check",
        "backend": backend.name,
        "is_real_model_training": False,
        "steps": steps,
        "checks": checks,
        "all_passed": all(checks.values()),
        "loss_first": trained["loss_first"],
        "loss_last": trained["loss_last"],
        "loss_decreased_observation_only": bool(trained["loss_decreased"]),
        "grad_norm_min": trained["grad_norm_min"],
        "supervised_tokens_seen": trained["supervised_tokens_seen"],
        "optimizer_step_count": trained["optimizer_step_count"],
        "adapter_sha256": report["saved"]["sha256"],
        "saved_filename": report["saved"]["filename"],
        # 🟡-8：保存前真的过了完整 state_dict 驻留门槛，这里回带实测观测（不是通过条件）。
        "full_state_dict_guard": report["memory_guard"],
        "note": (
            "合成玩具模型上的自检：只证明前向/反向/优化器步进/保存重载这条链路可用，"
            "不证明真实模型可训练，也不产生任何可提交产物。"
            "loss_decreased 只作观察值，不作为通过条件。"
        ),
    }
    if not keep_artifacts:
        import shutil

        shutil.rmtree(dest, ignore_errors=True)
        result["artifacts_kept"] = False
    else:
        result["artifacts_kept"] = True
        result["artifacts_dir"] = dest
    return result
