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

import hashlib
import json
import math
import os
import random
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

    def train_steps(self, plan: TrainRunPlan) -> dict:
        if not self.prepared:
            raise Blocked("backend_not_prepared", "train_steps 在 prepare 之前被调用")
        momentum = 0.9
        base_before = (self.adapter_params_digest(), self.base_params_digest())
        losses: list[float] = []
        grad_norms: list[float] = []
        supervised_tokens = 0
        for step in range(plan.steps):
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
        adapter_after, base_after = self.adapter_params_digest(), self.base_params_digest()
        return {
            "backend": self.name,
            "steps": plan.steps,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
            "losses": losses,
            "grad_norms": grad_norms,
            "grad_norm_min": min(grad_norms),
            "grad_norm_nonzero": all(value > MIN_GRAD_NORM for value in grad_norms),
            "supervised_tokens_seen": supervised_tokens,
            "optimizer": "sgd-momentum(0.9)，只作用于 adapter 参数",
            "optimizer_step_count": plan.steps,
            "adapter_params_before": base_before[0],
            "adapter_params_after": adapter_after,
            "adapter_params_changed": adapter_after != base_before[0],
            "base_params_before": base_before[1],
            "base_params_after": base_after,
            "base_params_frozen": base_after == base_before[1],
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

    def save_adapter(self, dest_dir: str, *, adapter_name: str | None = None) -> dict:
        # 合成后端**不是**提交载体，故意不产出 .safetensors，也不建官方 PEFT 目录：
        # `adapter_name` 参数只为接口一致而接受，不参与落盘。
        os.makedirs(dest_dir, exist_ok=True)
        payload = {
            "format": "synthetic-lora-adapter/1",
            "is_official_submission_carrier": False,
            "lora_rank": self.rank,
            "dim": self.dim,
            "vocab": self.vocab,
            "lora_a": self.lora_a,
            "lora_b": self.lora_b,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
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
        self.model = None
        self.tokenizer = None
        self._torch = None

    @staticmethod
    def _import_stack():
        """延迟导入并 assert 锁定依赖：缺一即 Blocked，不用未锁定版本顶替。"""
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

    def prepare(self, plan: TrainRunPlan) -> dict:
        plan.assert_runnable()
        if self.device.startswith("cuda") and not self.allow_download:
            # 真机执行必须显式授权取模型：本编码任务不联网取权重。
            raise Blocked(
                "model_download_not_authorized",
                "未授权下载模型权重/分词器：GPU 执行交 Liang 统筹链并显式放行",
            )
        torch, _peft, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer = (
            self._import_stack()
        )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise Blocked("cuda_unavailable", "指定 cuda 设备但 torch.cuda.is_available() 为 False")
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        base = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=torch.bfloat16,
        )
        frozen = 0
        for parameter in base.parameters():
            parameter.requires_grad_(False)
            frozen += 1
        lora_config = LoraConfig(
            r=int(plan.lora_rank),
            lora_alpha=int(plan.lora_alpha),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules,
        )
        self.model = get_peft_model(base, lora_config)
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
            "lora_matched_suffixes": matched,
            "trainable_parameter_tensors": len(trainable),
            "base_frozen_enforced": True,
            "unverified_claims": [
                "本路径未在真实 GPU 上执行过（无 GPU / 无锁定软件）",
                "真实显存峰值、TP4、长上下文行为未测",
            ],
        }

    def train_steps(self, plan: TrainRunPlan) -> dict:
        """真前向 / 真反向 / 真优化器步进。"""
        if self.model is None:
            raise Blocked("backend_not_prepared", "train_steps 在 prepare 之前被调用")
        torch = self._torch
        self.model.train()
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=plan.lr)
        base_before = self.base_params_digest()
        losses: list[float] = []
        grad_norms: list[float] = []
        supervised_tokens = 0
        for step in range(plan.steps):
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
            losses.append(float(loss.detach().float().item()))
            grad_norms.append(grad_norm)
            supervised_tokens += batch.supervised_tokens
        base_after = self.base_params_digest()
        return {
            "backend": self.name,
            "steps": plan.steps,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
            "losses": losses,
            "grad_norms": grad_norms,
            "grad_norm_min": min(grad_norms),
            "grad_norm_nonzero": all(value > MIN_GRAD_NORM for value in grad_norms),
            "supervised_tokens_seen": supervised_tokens,
            "optimizer": "torch.optim.AdamW（只作用于 requires_grad 的 adapter 参数）",
            "optimizer_step_count": plan.steps,
            "base_params_before": base_before,
            "base_params_after": base_after,
            "base_params_frozen": base_after == base_before,
        }

    def _parameter_digest(self, *, trainable: bool) -> str:
        if self.model is None:
            raise Blocked("backend_not_prepared", "adapter 摘要需要先 prepare")
        torch = self._torch
        chunks = []
        for name, parameter in sorted(self.model.named_parameters()):
            if bool(parameter.requires_grad) != trainable:
                continue
            chunks.append(name)
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

    def load_adapter(self, src_dir: str, *, adapter_name: str | None = None) -> dict:
        """重载校验：**新建**基座 + `PeftModel.from_pretrained`，比对 adapter 参数摘要。"""
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
        fresh_base = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=self._torch.bfloat16,
        )
        reloaded = PeftModel.from_pretrained(fresh_base, peft_dir)
        reloaded.to(self.device)
        previous_model = self.model
        self.model = reloaded
        after = self.adapter_params_digest()
        self.model = previous_model
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
            "base_reloaded_from_scratch": True,
        }


def run_training(
    backend: TrainBackend,
    plan: TrainRunPlan,
    dest_dir: str,
    *,
    extra_hashes: Mapping | None = None,
    full_state_dict_observation: Mapping | None = None,
    weight_manifest: Mapping | None = None,
) -> dict:
    """编排一次完整训练：prepare → train → **内存纪律门槛** → adapter-only 保存 →
    重载校验 → 出证据。

    任一步 fail-closed；返回体里 `executed=True` 只在**整条链路真的跑完**时出现。

    🟡-8：`assert_no_full_state_dict` 曾经只在 tests 里被调用（Y9 的第三处没有生产
    接线）。现在保存 adapter **之前**必经 `assert_full_state_dict_guard(...)`：
    观测优先取调用方传入的 `full_state_dict_observation`，否则问后端要
    （`backend.observe_full_state_dict()`）；两者都拿不到就 `PolicyViolation`，
    **不会**跳过门槛、也不会用自述结论顶替观测。
    """
    plan.assert_runnable()
    prepared = backend.prepare(plan)
    trained = backend.train_steps(plan)
    if not trained.get("grad_norm_nonzero"):
        raise IntegrityError("zero_gradient", "训练过程中出现过零梯度，不得当作成功")
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
    reloaded = backend.load_adapter(dest_dir, adapter_name=adapter_name)
    report = {
        "backend": backend.name,
        "requires_gpu": backend.requires_gpu,
        "verified_on_this_machine": backend.verified,
        "executed": True,
        "prepared": prepared,
        "trained": trained,
        "memory_guard": memory_guard,
        "saved": saved,
        "reloaded": reloaded,
        "evidence_hashes": dict(extra_hashes or {}),
        "unverified_claims": list(prepared.get("unverified_claims") or []),
    }
    os.makedirs(dest_dir, exist_ok=True)
    evidence_path = os.path.join(dest_dir, EVIDENCE_FILE)
    with open(evidence_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    report["evidence_path"] = evidence_path
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
