"""G7b：**限定合成工程检查**入口（与正式 train 分开授权）。

## 为什么单独一个入口

KAGGLE-32 v6 `GAPS-AND-OWNERS.md` §G7b 要求：

- 有一个受控的**真实测量**入口，覆盖前向 + 反向 + 优化器步进下的主机 RSS 与显存峰值；
- 工程检查与正式 train **分开授权**，不能靠手工置 true 或伪造 profile 过门；
- 本地只跑 fixture，**不声称**真实 GPU 验证。

所以：

- `v3/train/entry.py --start` 仍是唯一正式开训入口，门槛一条不放宽；
- 本模块是**工程检查**入口：默认后端是 CPU 合成 toy model，
  产物全部落在 `--dest-dir`（仓库外），报告里恒有
  `is_real_training=false` / `is_real_gpu=false`（除非真的跑在 CUDA 上并测到显存）。

## 一次工程检查做什么（全部有界、可停）

1. 20 项 mask fixture 全跑；
2. 合成 LoRA 训练循环跑 N 步（前向/反向/优化器步进），带
   **中途 checkpoint + 停止请求 + 截止时间**；
3. `profile.measure_memory_profile()` 逐阶段测主机峰值 RSS（显存测不到就
   如实写 `unavailable`，不写 `peak_gpu_gib`）；
4. 官方 PEFT 载体契约 + `v3-adapter-export/2` 清单**生产→消费**往返；
5. 从最近的 checkpoint **恢复**再跑一小段，证明恢复接线（哈希不一致必须拒绝）；
6. 产出证据包 JSON：命令、实际耗时、退出码、停止原因、剩余预算、产物哈希链。

**停止点**：`--time-budget-seconds`（monotonic 硬截止）+ `--stop-at-step`（步数上限）。
两者都在**一步边界**生效：先落 checkpoint，再照常保存 adapter 并重载校验，
所以"到点停机"不会跳过完整性检查。

## 诚实边界

- `--backend synthetic`（默认）：产物是**合成 fixture**，报告里
  `adapter_export.artifacts_are_synthetic=true`，绝不可当提交载体；
- `--backend torch-peft`：**真实后端路径，入口已接通** —— 按锁定 pin 构造
  `runner.TorchPeftBackend`，用 `--plan-json` 提供的**真实批次**跑前向/反向/优化器步进，
  并用真实 `measure_phases` 取主机峰值与显存峰值。缺锁定依赖（本机没有
  transformers/peft）或未授权下载时，它以 `Blocked`（`peft_unavailable` /
  `model_download_not_authorized` …）如实 fail-closed —— 是**跑到那一步才拒**，
  不是 CLI 层拒绝。缺 `--plan-json` 是 `MissingInput("training_plan_missing")`。

## 单一总截止（本轮整改）

所有阶段 —— mask fixture、训练、内存测量、adapter 导出、恢复探针、负例 —— **共用同一个
`StopRequest`**（同一个 `deadline_monotonic`）。恢复探针不再新建一个"没有截止时间"的
停止请求；每一阶段开始前先 `poll()`，超时即以
`Blocked("engineering_check_deadline_exceeded")` 收尾，并写出 `budget_ledger` 逐段账目。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Mapping

from ..common.canonical import sha256_json
from ..common.errors import Blocked, FailClosed, MissingInput, PolicyViolation
from . import adapter_export_contract as export_contract
from . import destdir
from . import input_binding as input_binding_mod
from . import profile as profile_mod
from . import runner
from . import supervised_check
from . import v6_approval
from .checkpoint import COMPLETE_MARKER, CheckpointStore
from .fixtures import run_all as run_fixture_suite
from .lifecycle import (
    STOP_DEADLINE,
    LifecyclePolicy,
    StopRequest,
    TrainingLifecycle,
    verify_checkpoint_roundtrip,
)

#: 后端选择（与正式 train 分开授权）。
BACKEND_SYNTHETIC = "synthetic"
BACKEND_TORCH_PEFT = "torch-peft"
BACKENDS = (BACKEND_SYNTHETIC, BACKEND_TORCH_PEFT)

#: 默认接口锁路径（仓库内）。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def default_interface_path() -> str:
    """默认的 `official-interface.json` 路径。"""
    return os.path.join(_REPO_ROOT, "v3", "locks", "official-interface.json")

#: 合成检查默认步数（有界，避免把工程检查跑成实验）。
DEFAULT_STEPS = 12
#: 合成检查默认 checkpoint 间隔。
DEFAULT_CHECKPOINT_EVERY = 4

#: 合成 fixture 的哈希值（**不是**真实环境哈希；报告里明确标注 synthetic）。
SYNTHETIC_HASHES = {
    "source_sha256": "0" * 64,
    "data_sha256": "1" * 64,
    "config_sha256": "2" * 64,
    "code_sha256": "3" * 64,
    "deps_sha256": "4" * 64,
}

#: 合成权重占位字节：**故意**不是合法 safetensors，避免被误当真实载体。
SYNTHETIC_WEIGHT_BYTES = b"SYNTHETIC-FIXTURE-NOT-A-REAL-SAFETENSORS-PAYLOAD"


class CheckpointNotUsed(Exception):
    """内部信号：本轮没有落到任何 checkpoint（不应发生，属于接线失败）。"""


def build_synthetic_export(export_dir: str, adapter_name: str) -> dict:
    """造一个符合官方 PEFT 目录契约的**合成**导出目录（仅供契约往返验证）。"""
    relative_dir = export_contract.adapter_contract.adapter_dir_relative_path(adapter_name)
    directory = os.path.join(export_dir, *relative_dir.split("/"))
    os.makedirs(directory, exist_ok=True)
    weights = os.path.join(directory, export_contract.adapter_contract.ADAPTER_WEIGHTS_FILENAME)
    config = os.path.join(directory, export_contract.adapter_contract.ADAPTER_CONFIG_FILENAME)
    with open(weights, "wb") as handle:
        handle.write(SYNTHETIC_WEIGHT_BYTES)
    payload = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj", "o_proj"],
        "_synthetic_fixture": True,
        "_not_a_submission_carrier": True,
        "_declared_adapter_name": adapter_name,
    }
    with open(config, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
    return {
        "export_dir": export_dir,
        "adapter_name": adapter_name,
        "weights_path": weights,
        "config_path": config,
        "artifacts_are_synthetic": True,
        "note": "合成占位字节，**不是**可加载的 adapter，也不是提交载体。",
    }


def _plan(steps: int, *, seq_len: int = 12, adapter_name: str = "v3_policy"):
    return runner.TrainRunPlan(
        steps=int(steps),
        lr=0.3,
        seq_len=int(seq_len),
        lora_rank=4,
        lora_alpha=8,
        batches=runner.build_smoke_batches(seq_len=seq_len),
        adapter_name=adapter_name,
    )


class BudgetLedger:
    """逐段账目 + **独立硬截止**（KAGGLE-38 r2 的洞在 r3 修掉）。

    r2 的实现是 `reason = stop.poll(); if reason == STOP_DEADLINE: raise`。
    这个判定有个洞：`StopRequest.request()` 是**首次生效**的，一旦业务原因
    （`step_limit` / `signal`）先置位，`poll()` 就永远返回那个业务原因、**再也不看时钟**。
    Mika 的合成反例：`deadline=10`、先置 `step_limit`、`time.monotonic()=11`，
    `check("adapter_export")` 没有抛错，而 `remaining = -1 s`。

    现在的判定与业务停止原因**解耦**：直接问 `stop.hard_deadline_exceeded()`（看时钟）。
    另外 `record()` 会检测**阶段内**越界（过程中才过期），把它记进
    `deadline_overrun_sections`，让报告与 verdict 都不能装作没发生。
    """

    def __init__(self, stop: StopRequest, started: float) -> None:
        self.stop = stop
        self.started = float(started)
        self.sections: list[dict] = []
        self.overruns: list[dict] = []

    def remaining(self) -> float | None:
        return self.stop.remaining_seconds()

    def check(self, section: str) -> None:
        """阶段**开始前**的硬截止闸门（不受业务停止原因遮蔽）。"""
        self.stop.poll()  # 让业务原因先登记（幂等，不覆盖）
        if self.stop.hard_deadline_exceeded():
            # 若截止还没被登记，补记一次（首次生效，不会覆盖业务原因）。
            self.stop.request(
                STOP_DEADLINE, remaining_seconds=self.stop.remaining_seconds()
            )
            raise Blocked(
                "engineering_check_deadline_exceeded",
                "总截止已到（%s 阶段开始前，剩余 %.3f s）：停止后续阶段，"
                "不越过 --time-budget-seconds；本节判定独立于 step_limit/signal 等业务停止原因"
                % (section, self.stop.remaining_seconds() or 0.0),
                section=section,
                budget_seconds=self._budget,
                remaining_seconds=self.stop.remaining_seconds(),
                stop_reason=self.stop.reason,
                deadline_source=self.deadline_source,
            )

    #: 只用于报错上下文。
    _budget: float | None = None
    #: 这个 deadline 是谁给的：`--time-budget-seconds` / 外层监督器 epoch / 两者取早。
    deadline_source: str = "time-budget-seconds"

    def section(self, name: str):
        return _LedgerSection(self, str(name))

    def record(self, name: str, before: float | None, t0: float) -> dict:
        after = self.remaining()
        entry = {
            "section": name,
            "remaining_seconds_before": before,
            "remaining_seconds_after": after,
            "elapsed_seconds": round(time.monotonic() - t0, 4),
            "deadline_monotonic": self.stop.deadline_monotonic,
            "deadline_source": self.deadline_source,
        }
        # 阶段**内**越界（开始时有时间、结束时没了）：如实记，不算通过。
        if after is not None and after <= 0.0 and (before is None or before > 0.0):
            entry["deadline_exceeded_within_section"] = True
            self.overruns.append(dict(entry))
        self.sections.append(entry)
        return entry


class _LedgerSection:
    def __init__(self, ledger: BudgetLedger, name: str) -> None:
        self.ledger = ledger
        self.name = name
        self._before: float | None = None
        self._t0 = 0.0

    def __enter__(self):
        self.ledger.check(self.name)
        self._before = self.ledger.remaining()
        self._t0 = time.monotonic()
        return self.ledger

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.ledger.record(self.name, self._before, self._t0)
        return False


def resolve_backend_choice(
    backend: str,
    *,
    model_id: str | None,
    model_revision: str | None,
    target_modules,
    allow_download: bool,
    device: str,
) -> str:
    """校验后端名（**不做**"有没有依赖"的判断：那必须在真实执行路径上才炸）。"""
    value = str(backend or BACKEND_SYNTHETIC).strip().lower()
    if value not in BACKENDS:
        raise PolicyViolation(
            "engineering_check_unknown_backend",
            "未知后端 %r：只支持 %s" % (backend, list(BACKENDS)),
        )
    if value == BACKEND_TORCH_PEFT:
        problems = []
        if not model_id:
            problems.append("model_id")
        if not model_revision:
            problems.append("model_revision")
        if not list(target_modules or ()):
            problems.append("target_modules")
        if problems:
            raise MissingInput(
                "engineering_check_backend_pins_missing",
                "torch-peft 后端缺后端 pin：%s（不得用 latest 或编造 target 列表顶替）"
                % ", ".join(problems),
                missing=problems,
            )
    return value


def build_plan(
    backend: str,
    *,
    steps: int,
    adapter_name: str,
    plan_json: str | None,
    plan=None,
):
    """按后端选训练计划：合成后端造 fixture；真实后端**必须**给真实批次。"""
    if plan is not None:
        return plan
    if backend == BACKEND_TORCH_PEFT:
        if not plan_json:
            raise MissingInput(
                "training_plan_missing",
                "torch-peft 真实检查必须给 --plan-json（真实批次）：入口不会自行编造训练数据",
            )
        return runner.plan_from_json(plan_json)
    return _plan(steps, adapter_name=adapter_name)


def run_engineering_check(
    *,
    dest_dir: str,
    steps: int = DEFAULT_STEPS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    stop_at_step: int | None = None,
    time_budget_seconds: float | None = None,
    adapter_name: str = "v3_policy",
    hashes: dict | None = None,
    resume_probe_steps: int = 3,
    probe_cuda: bool = False,
    cuda_probe_mib: int = 64,
    extra_note: str | None = None,
    backend: str = BACKEND_SYNTHETIC,
    plan_json: str | None = None,
    plan=None,
    model_id: str | None = None,
    model_revision: str | None = None,
    target_modules=None,
    allow_download: bool = False,
    local_load_authorized: bool = False,
    device: str = "cuda",
    deadline_epoch: float | None = None,
    model_inputs_root: str | None = None,
    model_inputs_layout=None,
    interface_path: str | None = None,
    decoy_input_roots=None,
    template_override: str | None = None,
    template_override_authorized: bool = False,
    template_override_reason: str | None = None,
    approval_path: str | None = None,
    approval_expected_sha256: str | None = None,
    approval_evidence_ref: str | None = None,
    requested_gpu_hours: float | None = None,
    local_model_dir: str | None = None,
    supervised_deadline: Mapping | None = None,
) -> dict:
    """跑一次完整的工程检查并返回证据包（不抛异常时整条链路都真的跑过）。"""
    started = time.monotonic()
    backend = resolve_backend_choice(
        backend,
        model_id=model_id,
        model_revision=model_revision,
        target_modules=target_modules,
        allow_download=allow_download,
        device=device,
    )
    is_real_backend = backend == BACKEND_TORCH_PEFT
    resolution = destdir.resolve_dest_dir(dest_dir, must_be_outside_repo=True)
    root = resolution["resolved"]
    export_dir = os.path.join(root, "fixture_adapter_export")
    checkpoint_root = os.path.join(root, "checkpoints")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(checkpoint_root, exist_ok=True)

    # ---- 缺陷 D：真实模式的准入与合成模式**分开** ----
    # `allow_download` 只是"允许联网取权重"的网络许可；它**不能**当作
    # "可以在本机把这个作业跑起来"的工程作业批准。两者必须分别授权。
    if is_real_backend:
        if hashes is None:
            raise MissingInput(
                "real_mode_requires_real_hashes",
                "真实后端（torch-peft）必须用 --hashes 传入真实的环境/数据/配置/代码/依赖哈希："
                "不得回退到 SYNTHETIC_HASHES —— 那会让 checkpoint 的同源判定建立在假哈希上",
            )
        if not local_load_authorized:
            raise PolicyViolation(
                "local_load_not_authorized",
                "真实后端未获**本地加载**授权（--local-load-authorized）："
                "--allow-download 只授权联网取权重，不构成工程作业批准",
            )
        if time_budget_seconds is None and supervised_deadline is None:
            raise MissingInput(
                "real_mode_requires_time_budget",
                "真实后端必须显式给出 --time-budget-seconds（或受监督模式下的外层截止文件）："
                "真实作业不得无界运行",
            )
        if stop_at_step is None:
            raise MissingInput(
                "real_mode_requires_stop_at_step",
                "真实后端必须显式给出 --stop-at-step：首片必须有确定的停机点",
            )
        if not model_inputs_root:
            raise MissingInput(
                "model_inputs_root_missing",
                "真实后端必须给出 --model-inputs-root（被核验的官方配置/tokenizer/template 所在目录）："
                "不得让 from_pretrained 隐式读现场可能被改过的工作副本",
            )
        # ---- 缺陷 2：真实哈希与"本地加载许可"都**不能自报**，必须绑 v6 批准 ----
        # 独立审查实测：五个哈希全填字符串 "1" 被接受；`--local-load-authorized`
        # 只是一个调用方自己按的布尔。现在改为消费 KAGGLE-32 v6 的批准记录
        # （shard_guard 契约：字段齐全不算批准，必须带**外部锚**）。
        if not v6_approval.deployment_approval_anchor():
            raise PolicyViolation("approval_not_trusted", "部署方尚未接入已发布批准锚")
        if supervised_deadline is None:
            raise PolicyViolation("approval_not_trusted", "真实检查必须由 v6 监督器启动")
        from .deployment import read_publication
        expiry = float(read_publication()['approval']['not_after_epoch'])
        if float(supervised_deadline['deadline_epoch']) > expiry:
            raise PolicyViolation('approval_not_trusted', 'Child cutoff exceeds publication expiry')
        if not local_model_dir:
            raise MissingInput(
                "local_model_dir_missing",
                "真实后端必须给出固定的本地权重目录 --local-model-dir："
                "只允许离线本地加载，缺资源即拒绝，不允许联网回退",
            )

    # ---- 输入绑定（缺陷 C）：消费者实际读哪份 config / tokenizer / template ----
    input_binding: dict | None = None
    load_view: dict | None = None
    template_override_record: dict | None = None
    if model_inputs_root:
        input_binding = input_binding_mod.bind_model_inputs(
            root=model_inputs_root,
            interface_path=interface_path or default_interface_path(),
            layout=model_inputs_layout,
            allowed_roots=[model_inputs_root, root],
            decoy_roots=decoy_input_roots,
        )
        template_override_record = input_binding_mod.assert_no_template_override(
            override_template=template_override,
            override_authorized=bool(template_override_authorized),
            override_reason=template_override_reason,
            binding=input_binding,
        )
        load_view = input_binding_mod.assemble_load_view(
            input_binding, os.path.join(root, "load_view")
        )
        input_binding = dict(input_binding)
        input_binding["load_view_dir"] = load_view["view_dir"]
        input_binding["view_manifest_sha256"] = load_view["manifest_sha256"]
        input_binding["template_override"] = template_override_record

    # ---- 总截止：预算截止 与 外层监督器 epoch 截止 **取早** ----
    budget_deadline = (
        None if time_budget_seconds is None else started + float(time_budget_seconds)
    )
    outer_deadline = None
    if deadline_epoch is not None:
        outer_deadline = started + max(0.0, float(deadline_epoch) - time.time())
    # ---- r5（独审阻断 6）：v6 监督器签发的截止不可被放宽 ----
    # 受监督模式下，真正生效的截止由外层文件给定；任何比它更晚的请求都**显式拒绝**
    # （不是静默夹紧 —— 夹紧会让"谁给的截止"变得不可审计）。
    if supervised_deadline is not None:
        widening = supervised_check.reject_widening(
            supervised=supervised_deadline,
            requested_budget_seconds=time_budget_seconds,
            requested_deadline_epoch=deadline_epoch,
            now_epoch=time.time(),
            now_monotonic=time.monotonic(),
        )
    else:
        widening = {"ok": True, "widened": False, "details": None}
    supervised_deadline_monotonic = (
        None if supervised_deadline is None else float(supervised_deadline["deadline_monotonic"])
    )
    deadlines = [
        item
        for item in (budget_deadline, outer_deadline, supervised_deadline_monotonic)
        if item is not None
    ]
    effective_deadline = min(deadlines) if deadlines else None
    if supervised_deadline_monotonic is not None and (
        effective_deadline == supervised_deadline_monotonic
    ):
        deadline_source = "v6-supervisor"
    elif budget_deadline is not None and outer_deadline is not None:
        deadline_source = (
            "outer-supervisor" if outer_deadline < budget_deadline else "time-budget-seconds"
        )
    elif outer_deadline is not None:
        deadline_source = "outer-supervisor"
    else:
        deadline_source = "time-budget-seconds"

    stop = StopRequest(
        deadline_monotonic=effective_deadline,
        deadline_epoch=(
            None if effective_deadline is None else time.time() + (effective_deadline - started)
        ),
    )
    stop.install()
    ledger = BudgetLedger(stop, started)
    # 受监督模式下没有 `--time-budget-seconds`（截止只由外层签发），账目里的预算
    # 就取签发文件里的那个数，避免"有截止、没预算"的自相矛盾账目。
    ledger._budget = (
        time_budget_seconds
        if time_budget_seconds is not None
        else (None if supervised_deadline is None else supervised_deadline["budget_seconds"])
    )
    ledger.deadline_source = deadline_source
    used_hashes = dict(hashes or SYNTHETIC_HASHES)
    hashes_are_synthetic = hashes is None
    plan = build_plan(
        backend,
        steps=int(steps),
        adapter_name=adapter_name,
        plan_json=plan_json,
        plan=plan,
    )

    if is_real_backend:
        options = v6_approval.runtime_options(
            steps=steps, checkpoint_every=checkpoint_every, stop_at_step=stop_at_step,
            budget_seconds=supervised_deadline["approved_budget_seconds"],
            requested_gpu_hours=requested_gpu_hours, model_id=model_id,
            model_revision=model_revision, target_modules=target_modules,
            device=device, adapter_name=adapter_name)
        actual_context = v6_approval.measure_runtime_context(
            plan=plan, input_binding=input_binding, local_model_dir=local_model_dir,
            options=options)
        approval_summary = v6_approval.require_approval(
            approval_path=approval_path, expected_sha256=approval_expected_sha256,
            requested_gpu_hours=requested_gpu_hours, evidence_ref_path=approval_evidence_ref,
            runtime_context=actual_context,
            published_anchor=v6_approval.deployment_approval_anchor())
        if (dict(hashes) != actual_context["hashes"] or
                supervised_deadline.get("approval_context_sha256") != sha256_json(actual_context)):
            raise PolicyViolation("approval_runtime_mismatch", "子进程实际运行对象与外层批准不符")
        used_hashes = actual_context["hashes"]
        input_binding["approved_local_model_dir"] = actual_context["model"]["local_model_dir"]

    report: dict = {
        "stage": "engineering-check",
        "kind": "engineering-check" if is_real_backend else "synthetic-engineering-check",
        "backend": backend,
        "is_real_training": False,
        "is_real_gpu": False,
        "started_unix": time.time(),
        "dest_dir_resolution": resolution,
        "adapter_name": adapter_name,
        "plan_source": (
            "plan-json（真实批次）"
            if is_real_backend
            else ("外部注入 plan" if plan_json is None and plan is not None else "合成 fixture")
        ),
        "training_plan": {
            "steps": int(getattr(plan, "steps", steps)),
            "seq_len": int(getattr(plan, "seq_len", 0)),
            "batch_count": len(getattr(plan, "batches", []) or []),
            "lora_rank": int(getattr(plan, "lora_rank", 0)),
            "lora_alpha": int(getattr(plan, "lora_alpha", 0)),
            "batches_sha256": sha256_json(
                [batch.to_dict() for batch in (getattr(plan, "batches", []) or [])]
            ),
        },
        "budget": {
            "time_budget_seconds": time_budget_seconds,
            "stop_at_step": stop_at_step,
            "steps": int(steps),
            "checkpoint_every": int(checkpoint_every),
            "deadline_monotonic": stop.deadline_monotonic,
            "deadline_epoch": stop.deadline_epoch,
            "deadline_source": deadline_source,
            "outer_deadline_epoch": deadline_epoch,
            "single_deadline_for_all_sections": True,
            "hard_deadline_independent_of_stop_reason": True,
            # r5：受监督模式由外层 v6 监督器签发截止，子进程只能收紧不能放宽。
            "supervised": (
                None
                if supervised_deadline is None
                else {
                    "active": True,
                    "source": supervised_deadline.get("source"),
                    "budget_seconds": supervised_deadline.get("budget_seconds"),
                    "deadline_monotonic": supervised_deadline.get("deadline_monotonic"),
                    "file_sha256": supervised_deadline.get("file_sha256"),
                    "anchor_verified": supervised_deadline.get("verified_against_anchor"),
                    "widening_check": widening,
                    "effective_deadline_is_supervised": deadline_source == "v6-supervisor",
                }
            ),
            "supervised_active": supervised_deadline is not None,
        },
        "input_binding": {
            "bound": input_binding is not None,
            "root": (input_binding or {}).get("root"),
            "revision": (input_binding or {}).get("revision"),
            "binding_sha256": (input_binding or {}).get("binding_sha256"),
            "consumed_inputs": (input_binding or {}).get("consumed_inputs") or {},
            "shadowed_files": (input_binding or {}).get("shadowed_files") or [],
            "load_view_dir": (input_binding or {}).get("load_view_dir"),
            "load_view_manifest_sha256": (input_binding or {}).get("view_manifest_sha256"),
            "template_override": template_override_record,
            "weights_sha256": "hash_pending",
            "note": (
                "消费者实际读的字节 = 上面 consumed_inputs 里逐文件列出的 sha256；"
                "权重不在绑定范围（hash_pending）。"
                if input_binding
                else "本次没有绑定加载输入（合成模式可用 fixture；真实模式必须绑定）"
            ),
        },
        "authorization": {
            "approval": approval_summary if is_real_backend else None,
            "runtime_environment": actual_context['runtime_environment'] if is_real_backend else None,
            "allow_download": bool(allow_download),
            "local_load_authorized": bool(local_load_authorized),
            "download_permission_note": (
                "--allow-download 只是联网取权重的网络许可；"
                "本地加载许可是独立的 --local-load-authorized"
            ),
        },
        "hashes": {
            "values": used_hashes,
            "are_synthetic_fixture": hashes_are_synthetic,
            "note": (
                "合成 fixture 哈希：只用于验证 checkpoint 的哈希一致性判定；"
                "真实作业必须用 --hashes 传入真实哈希文件"
            )
            if hashes_are_synthetic
            else "来自 --hashes 文件",
        },
        "notes": [],
    }
    if extra_note:
        report["notes"].append(str(extra_note))

    # 1) mask fixture
    with ledger.section("mask_fixtures"):
        fixtures = run_fixture_suite()
    report["mask_fixtures"] = {
        "count": fixtures.get("fixture_count"),
        "passed": fixtures.get("passed"),
        "all_passed": bool(fixtures.get("all_passed")),
        "renderer": fixtures.get("renderer"),
    }
    report["fixture_pass"] = int(fixtures.get("passed") or 0)
    report["fixture_total"] = int(fixtures.get("fixture_count") or 0)
    if not fixtures.get("all_passed"):
        report["verdict"] = "fail"
        report["failure"] = "mask fixture 未全过"
        return _finalize(report, root, stop, started, ledger)

    backend_kwargs = dict(
        adapter_name=adapter_name,
        model_id=model_id,
        model_revision=model_revision,
        target_modules=target_modules,
        allow_download=allow_download,
        device=device,
        model_inputs=input_binding,
        local_load_authorized=local_load_authorized,
        local_model_dir=local_model_dir,
    )
    backend_impl = _make_backend(backend, **backend_kwargs)
    lifecycle = TrainingLifecycle(
        store_root=checkpoint_root,
        hashes=used_hashes,
        policy=LifecyclePolicy(
            checkpoint_every=int(checkpoint_every), stop_at_step=stop_at_step
        ),
        stop=stop,
    )

    # 2) 训练（带中途 checkpoint 与停止）—— 合成后端是 fixture，torch-peft 是真的
    with ledger.section("train"):
        training = runner.run_training(
            backend_impl,
            plan,
            os.path.join(root, "adapter_out"),
            extra_hashes=used_hashes,
            lifecycle=lifecycle,
            stop=stop,
        )
    report["training"] = training
    report["stopped"] = bool(training.get("stopped"))
    report["stop_reason"] = training.get("stop_reason")
    report["stop"] = stop.to_dict()
    report["checkpoint_count"] = (training.get("lifecycle") or {}).get("checkpoint_count")
    if is_real_backend:
        report["is_real_training"] = bool(training.get("executed"))
        report["is_real_gpu"] = str(getattr(backend_impl, "device", "")).startswith("cuda")

    # 3) 内存 profile（主机真实测量；显存测不到就是 unavailable）
    #    缺陷 B：真实后端**复用同一个后端实例**，而且这里**不再** prepare 第二次 ——
    #    否则 31B 基座会被加载两遍，模型/optimizer 实例错配、驻留翻倍，
    #    "只加载一次"的声明也就不成立。prepare=False 时用前一次的 prepared 结果，
    #    并断言测的是同一组参数（parameter_ownership）。
    with ledger.section("memory_profile"):
        if is_real_backend:
            profiler_backend = backend_impl
            reused_prepared_report = training.get("prepared")
            memory_profile = profile_mod.measure_memory_profile(
                profiler_backend,
                plan,
                seq_len=plan.seq_len,
                prepare=False,
                prepared_report=reused_prepared_report,
                note="真实后端工程检查：复用已加载实例，主机峰值与显存峰值来自同一次加载的模型。",
            )
        else:
            profiler_backend = _make_backend(BACKEND_SYNTHETIC, **backend_kwargs)
            memory_profile = profile_mod.measure_memory_profile(
                profiler_backend,
                _plan(steps=1, adapter_name=adapter_name),
                seq_len=plan.seq_len,
                note="合成工程检查：主机峰值真实测量；显存仅在真实 CUDA 环境可用时才有数字。",
            )
    report["memory_profile"] = memory_profile
    report["memory_profile_payload"] = profile_mod.profile_to_memory_plan_payload(memory_profile)
    report["memory_profile_provenance"] = profile_mod.check_profile_provenance(memory_profile)
    report["phases_covered"] = [item["phase"] for item in memory_profile.get("phases") or []]
    if is_real_backend:
        # 参数归属反例判据：测量用的必须是训练用过的**同一组**参数。
        training_ownership = (training.get("prepared") or {}).get("parameter_ownership")
        measurement_ownership = memory_profile.get("backend_parameter_ownership")
        report["memory_profile_same_instance"] = {
            "reused_prepared_backend": bool(memory_profile.get("reused_prepared_backend")),
            "backend_load_count": int(memory_profile.get("backend_load_count") or 0),
            "training_parameter_ownership": training_ownership,
            "measurement_parameter_ownership": measurement_ownership,
            "same_parameter_ids": bool(training_ownership)
            and training_ownership == measurement_ownership,
            "loaded_exactly_once": int(memory_profile.get("backend_load_count") or 0) == 1,
        }
    report["cuda_probe"] = (
        profile_mod.probe_cuda_reader(mib=int(cuda_probe_mib)) if probe_cuda else {"skipped": True}
    )

    # 4) 官方载体 + v2 导出清单：生产 → 消费往返
    #    真实后端：对**真的落盘 adapter** 的目录出清单（artifacts_are_synthetic=false）；
    #    合成后端：对 fixture 目录出清单（明确标 synthetic）。
    with ledger.section("adapter_export"):
        synthetic_export = None
        if is_real_backend:
            export_root = os.path.join(root, "adapter_out")
            artifacts_are_synthetic = False
        else:
            synthetic_export = build_synthetic_export(export_dir, adapter_name)
            export_root = export_dir
            artifacts_are_synthetic = True
        manifest = export_contract.build_export_manifest(
            export_root,
            adapter_name=adapter_name,
            rank=plan.lora_rank,
            lora_alpha=plan.lora_alpha,
            lora_dropout=0.0,
            target_modules_regex=".*(q_proj|o_proj)$",
            matched_module_count=len(
                ((training.get("prepared") or {}).get("lora_matched_suffixes") or [])
            )
            or 2,
            base_repo_id=model_id or "google/gemma-4-31B-it-qat-w4a16-ct",
            base_revision=model_revision or "52f3f65bc7a02d555763bc923bd1d9094898219d",
            # 分阶段校验：训练阶段**允许** mapper revision 未知（它只能在评分宿主观测），
            # 清单里如实写 null + serving_pins_verified=false，serving 阶段另行单独判。
            vllm_mapper_revision=None,
            stage=export_contract.STAGE_TRAINING,
            serving_pin_verified=False,
            training_hashes=used_hashes,
            known_ranking=(
                {"synthetic_fixture": True, "not_a_submission_carrier": True}
                if artifacts_are_synthetic
                else {"engineering_check": True, "not_a_submission_carrier": True}
            ),
        )
        validation = export_contract.validate_export_manifest(
            manifest, export_dir=export_root, stage=export_contract.STAGE_TRAINING
        )
    manifest_path = os.path.join(root, "adapter_export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    report["adapter_export"] = {
        "synthetic_export": synthetic_export,
        "manifest_path": manifest_path,
        "export_root": export_root,
        "format": manifest["format"],
        "file_count": manifest["file_count"],
        "artifact_digest": manifest["artifact_digest"],
        "validation": validation,
        "artifacts_are_synthetic": artifacts_are_synthetic,
        "serving_pins_verified": validation.get("serving_pins_verified"),
    }

    # 5) 恢复接线：从最近 checkpoint 起继续（**同一个总截止**，不再新建无截止的 StopRequest）
    with ledger.section("resume_probe"):
        store = CheckpointStore(root=checkpoint_root)
        complete = store._complete_steps()
        if not complete:
            raise CheckpointNotUsed("工程检查没有落到任何 checkpoint：G14 接线失败")
        last = complete[-1]
        roundtrip = verify_checkpoint_roundtrip(checkpoint_root, last, used_hashes)
        # ---- 缺陷 1：进入恢复探针前**受控释放**旧基座/optimizer/全部持有引用 ----
        # 独立审查实测：旧实现保留 `backend_impl`、`training`、`profiler_backend`
        # 的引用，随后又用 `resumed_backend` 完整训练——于是双基座同时驻留。
        # 这里先释放，并留下可用弱引用核对的释放证据，再构造新后端。
        release_evidence = release_before_resume(
            backend_impl=backend_impl,
            profiler_backend=locals().get("profiler_backend"),
            training=training,
        )
        # 恢复探针用**同一种后端的一个新实例**：它必须从 checkpoint 起步，
        # 而不是接着主训练那份内存里的状态跑（否则等于没测恢复）。
        resumed_backend = _make_backend(backend, **backend_kwargs)
        resume_lifecycle = TrainingLifecycle(
            store_root=checkpoint_root,
            hashes=used_hashes,
            policy=LifecyclePolicy(checkpoint_every=0),
            resume_from=store.checkpoint_dir(last),
            # 关键：共享同一个 deadline，恢复探针也在总预算之内。
            stop=StopRequest(
                deadline_monotonic=stop.deadline_monotonic,
                deadline_epoch=stop.deadline_epoch,
            ),
        )
        resume_plan = (
            build_plan(
                backend,
                steps=int(resume_probe_steps),
                adapter_name=adapter_name,
                plan_json=plan_json,
            )
            if not is_real_backend
            else runner.TrainRunPlan(
                steps=int(resume_probe_steps),
                lr=plan.lr,
                seq_len=plan.seq_len,
                lora_rank=plan.lora_rank,
                lora_alpha=plan.lora_alpha,
                batches=plan.batches,
                seed=plan.seed,
                adapter_name=plan.adapter_name,
            )
        )
        resume_report = runner.run_training(
            resumed_backend,
            resume_plan,
            os.path.join(root, "adapter_out_resumed"),
            extra_hashes=used_hashes,
            lifecycle=resume_lifecycle,
            stop=resume_lifecycle.stop,
        )
    report["resume"] = {
        "roundtrip": roundtrip,
        "resume_report": (resume_report.get("lifecycle") or {}).get("resume"),
        "steps_executed_after_resume": (resume_report.get("trained") or {}).get("steps_executed"),
        "global_step_after_resume": (resume_report.get("trained") or {}).get("global_step"),
        "adapter_reload_matches": bool((resume_report.get("reloaded") or {}).get("matches_saved_adapter")),
        "deadline_shared_with_main_stop": True,
    }
    report["resume_probe_steps"] = int(resume_probe_steps)

    # 6) 负例：环境哈希不一致必须拒绝恢复
    with ledger.section("negative_hash_mismatch"):
        wrong = dict(used_hashes)
        wrong["data_sha256"] = "f" * 64
        try:
            verify_checkpoint_roundtrip(checkpoint_root, last, wrong)
            report["negative_hash_mismatch_rejected"] = False
            report["notes"].append("警告：哈希不一致未被拒绝（这是接线缺陷）")
        except FailClosed as exc:
            report["negative_hash_mismatch_rejected"] = True
            report["negative_hash_mismatch_code"] = exc.code

    # 硬截止：逐段检测**阶段内**越界（KAGGLE-38 r3 缺陷 A）。
    report["deadline_overrun_sections"] = [
        item["section"] for item in (getattr(ledger, "overruns", None) or [])
    ]
    report["verdict"] = "pass" if _all_checks_ok(report) else "fail"
    if report["deadline_overrun_sections"]:
        report["failure"] = "有阶段在过程中越过总截止：%s" % report["deadline_overrun_sections"]
    return _finalize(report, root, stop, started, ledger)


def release_before_resume(*, backend_impl, profiler_backend, training) -> dict:
    """缺陷 1：恢复探针开始前**受控释放**旧基座/optimizer/全部持有引用。

    独立审查（评论 01a11aa9…）实测：旧实现在构造并训练 `resumed_backend` 时，
    `backend_impl`、`training`、`profiler_backend` 都还持着主训练那份基座——至少在
    重载瞬间双基座同时驻留，违反 16GB 主机约束。

    本函数只做三件事，且都留下可核对的证据：

    1. 对每个后端调用 `release_base()`（弱引用证明旧对象已不可达）；
    2. 从 `training` 报告里摘掉**对象型**引用（保留数据字段，报告不受影响）；
    3. 返回一份 `release_evidence`，供报告与复核者核对"此刻只有一个基座"。

    不声称真机峰值已在本地测得——真机内存峰值仍待首片实测。
    """
    evidence: dict = {"released_backends": [], "dropped_training_refs": [], "deduped": []}
    seen: set[int] = set()
    for label, backend in (("backend_impl", backend_impl), ("profiler_backend", profiler_backend)):
        if backend is None:
            continue
        if id(backend) in seen:
            evidence["deduped"].append(label)
            continue
        seen.add(id(backend))
        if not hasattr(backend, "release_base"):
            evidence["released_backends"].append({"name": label, "release_supported": False})
            continue
        info = backend.release_base("before_resume_probe")
        evidence["released_backends"].append({"name": label, "release_supported": True, **info})
    if isinstance(training, dict):
        for key in list(training):
            value = training[key]
            # 只摘"能自己放基座/本身就是模型"的引用；纯数据字段一律保留。
            drop = hasattr(value, "release_base") or hasattr(value, "named_parameters")
            if drop:
                info = (
                    value.release_base("training_report_ref")
                    if hasattr(value, "release_base")
                    else {}
                )
                training.pop(key, None)
                evidence["dropped_training_refs"].append({"key": key, **info})
    # Drop the loop's last local strong reference before observing old models.
    value = None
    observations = []
    for label, backend in (("backend_impl", backend_impl), ("profiler_backend", profiler_backend)):
        if hasattr(backend, "assert_base_released"):
            observed = backend.assert_base_released()
            observations.append(observed)
            for entry in evidence["released_backends"]:
                if entry["name"] == label:
                    entry["old_base_ref_alive_after_release"] = observed["old_base_alive"]
    evidence["only_one_base_resident"] = all(
        item["only_one_base_resident"] for item in observations
    )
    evidence["observations"] = observations
    evidence["note"] = (
        "释放发生在 resumed_backend 构造**之前**；旧基座是否已不可达由 release_base() "
        "里的弱引用判定给出。真机内存峰值待首片实测，本地不声称已发生 OOM。"
    )
    return evidence


def _make_backend(
    backend: str,
    *,
    adapter_name: str,
    model_id: str | None,
    model_revision: str | None,
    target_modules,
    allow_download: bool,
    device: str,
    model_inputs=None,
    local_load_authorized: bool = False,
    local_model_dir: str | None = None,
):
    """构造后端。**不在构造处判依赖是否存在** —— 缺依赖必须由 prepare() 如实 `Blocked`。"""
    if backend == BACKEND_TORCH_PEFT:
        return runner.TorchPeftBackend(
            model_id=str(model_id),
            revision=str(model_revision),
            target_modules=list(target_modules or ()),
            allow_download=bool(allow_download),
            device=str(device),
            model_inputs=model_inputs,
            local_load_authorized=bool(local_load_authorized),
            local_model_dir=local_model_dir,
        )
    return runner.SyntheticBackend(vocab=32, dim=8)


def _all_checks_ok(report: dict) -> bool:
    fixtures = report.get("mask_fixtures") or {}
    training = report.get("training") or {}
    trained = training.get("trained") or {}
    resume = report.get("resume") or {}
    per_section = all(
        [
            bool(fixtures.get("all_passed")),
            bool(training.get("executed")),
            bool((training.get("saved") or {}).get("adapter_only")),
            bool((training.get("reloaded") or {}).get("matches_saved_adapter")),
            bool(trained.get("base_params_frozen")),
            int(report.get("checkpoint_count") or 0) >= 1,
            bool(resume.get("adapter_reload_matches")),
            bool(report.get("negative_hash_mismatch_rejected")),
            bool((report.get("adapter_export") or {}).get("validation", {}).get("ok")),
        ]
    )
    # 硬截止：任何一段在过程中越过总截止，都不能算通过（KAGGLE-38 r3 缺陷 A）。
    no_overrun = not report.get("deadline_overrun_sections")
    return bool(per_section and no_overrun)


def _finalize(report: dict, root: str, stop: StopRequest, started: float, ledger=None) -> dict:
    elapsed = time.monotonic() - started
    remaining = None
    if stop.deadline_monotonic is not None:
        remaining = round(float(stop.deadline_monotonic) - time.monotonic(), 3)
    report["stop"] = stop.to_dict()
    report["elapsed_seconds"] = round(elapsed, 4)
    report["remaining_budget_seconds"] = remaining
    report["budget_ledger"] = list(getattr(ledger, "sections", []) or [])
    report["deadline_overruns"] = list(getattr(ledger, "overruns", None) or [])
    report["single_deadline_enforced"] = True
    report["hard_deadline_enforced"] = True
    report["finished_unix"] = time.time()
    report["exit_code_expected"] = 0 if report.get("verdict") == "pass" else 1
    report["unverified_claims"] = [
        "获批配置下的真实显存峰值：合成后端（--backend synthetic）的 memory_profile_payload "
        "恒为 null；要真实读数必须用 --backend torch-peft + --plan-json 在获批 GPU 上跑",
        "真实模型加载/前向/反向（合成后端是 toy 模型，不是 gemma-4-31B）",
        "checkpoint 在真实 GPU 上恢复训练（只证明了写盘/哈希/导入接线）",
        "官方 compiler 对真实 adapter 的接受度（本模块只做静态载体契约）",
        "serving 阶段：vllm_mapper_revision 未验证，清单只通过训练阶段校验",
        "权重 model.safetensors 的 SHA 明确 hash_pending，未纳入输入绑定",
    ]
    report["honest_scope"] = (
        "这是**合成工程检查**：证明入口存在、接线正确、能停机/保存/重载、能出证据；"
        "不证明任何训练收益，也不构成训练资格。"
    )
    # 证据哈希链：对除本字段外的内容做规范摘要，便于事后核对未被改动。
    report["evidence_sha256"] = sha256_json(
        {key: value for key, value in report.items() if key != "evidence_sha256"}
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V3 合成工程检查（G7b）：不训练真实模型、不占 GPU、产物落在仓库外"
    )
    parser.add_argument("--dest-dir", required=True, help="产物目录：绝对路径且必须在仓库外")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--stop-at-step", type=int, default=None)
    parser.add_argument("--time-budget-seconds", type=float, default=None)
    parser.add_argument("--adapter-name", default="v3_policy")
    parser.add_argument("--hashes", default=None, help="5 项真实哈希的 JSON（真实作业必给）")
    parser.add_argument("--resume-probe-steps", type=int, default=3)
    parser.add_argument(
        "--probe-cuda",
        action="store_true",
        help="附带一次受控极小显存探针（分配→加一→释放），只验证显存读数路径可用",
    )
    parser.add_argument("--cuda-probe-mib", type=int, default=64)
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default=BACKEND_SYNTHETIC,
        help=(
            "synthetic（默认，CPU toy 模型）或 torch-peft（真实后端：需要 --plan-json "
            "给出的真实批次，以及锁定依赖/GPU；缺依赖时在 prepare 处 Blocked）"
        ),
    )
    parser.add_argument(
        "--plan-json",
        default=None,
        help="真实批次训练计划（torch-peft 必给；见 runner.plan_from_json 的字段约定）",
    )
    parser.add_argument("--model-id", default=None, help="torch-peft 的模型 repo（禁 latest）")
    parser.add_argument("--model-revision", default=None, help="torch-peft 的固定 revision")
    parser.add_argument(
        "--target-modules",
        default=None,
        help="torch-peft 的 LoRA target 模块，逗号分隔，例如 q_proj,o_proj",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "授权**联网取权重/分词器**（网络许可）。注意：它**不**等于"
            "可以在本机把作业跑起来的工程批准 —— 后者是独立的 --local-load-authorized"
        ),
    )
    parser.add_argument(
        "--local-load-authorized",
        action="store_true",
        help="授权**本地加载并执行**（真实作业的工程批准；与 --allow-download 分开）",
    )
    parser.add_argument(
        "--deadline-epoch",
        type=float,
        default=None,
        help=(
            "外层监督器（KAGGLE-32 v6 shard_supervisor）的总截止（Unix epoch，秒）。"
            "给了就与 --time-budget-seconds **取早**，并在报告里记 deadline_source。"
            "也可用环境变量 V3_DEADLINE_EPOCH。"
        ),
    )
    parser.add_argument(
        "--model-inputs-root",
        default=None,
        help=(
            "被核验的官方配置/tokenizer/template 所在目录（真实模式必给）："
            "本入口会逐字节核对 pin 并装配受控加载视图，绝不让 from_pretrained 隐式读工作副本"
        ),
    )
    parser.add_argument(
        "--model-inputs-layout",
        default=None,
        help="布局 JSON（文件名 → root 下的相对路径），默认用现场快照布局 official/+shared/",
    )
    parser.add_argument("--interface", default=None, help="official-interface.json 路径")
    parser.add_argument(
        "--approval",
        default=None,
        help=(
            "KAGGLE-32 v6 批准记录（真实模式**必需**）：自报 --hashes 与布尔开关都不能"
            "替代批准。契约复用 v6 shard_guard.py，要求带外部锚。"
        ),
    )
    parser.add_argument(
        "--approval-expected-sha256",
        default=None,
        help=(
            "批准记录的**外部锚**：外部公布的该文件 SHA-256。没有它就一律拒绝 ——"
            "`verified_by` 是自报字段，谁都能填，不构成信任。"
        ),
    )
    parser.add_argument(
        "--approval-evidence-ref",
        default=None,
        help="被批准的证据对象路径（批准记录声明了 evidence_sha256 时必须给出，供实算比对）",
    )
    parser.add_argument(
        "--requested-gpu-hours",
        type=float,
        default=None,
        help="本次申请使用的 GPU-h，用于与批准额度比对（超过即拒）",
    )
    parser.add_argument(
        "--local-model-dir",
        default=None,
        help=(
            "固定的本地权重目录（真实模式**必需**）：只允许离线本地加载"
            "（local_files_only=True），缺本地资源即拒绝，不联网回退"
        ),
    )
    parser.add_argument(
        "--decoy-input-root",
        action="append",
        default=None,
        help=(
            "只做取证：另一个可能放着同名文件但被改过的目录（例如现场 prep/model）。"
            "命中的差异会如实记进 input_binding.shadowed_files，不会让它被加载"
        ),
    )
    parser.add_argument(
        "--template-override-file",
        default=None,
        help="显式覆盖 chat template 的文件；默认拒绝，必须同时给 --allow-template-override 与理由",
    )
    parser.add_argument("--allow-template-override", action="store_true")
    parser.add_argument("--template-override-reason", default=None)
    parser.add_argument("--device", default="cuda", help="torch-peft 设备（默认 cuda）")
    # ---- r5（独审阻断 6）：受监督模式 -------------------------------------------------
    parser.add_argument(
        "--supervised",
        action="store_true",
        help="受 v6 监督器管辖：截止只能来自 --deadline-file，且不接受任何放宽",
    )
    parser.add_argument(
        "--deadline-file",
        default=None,
        help="外层监督器签发的截止文件（受监督模式必需）",
    )
    parser.add_argument(
        "--deadline-file-sha256",
        default=None,
        help="截止文件的自锚 SHA-256（受监督模式必需）：文件被换掉即拒绝",
    )
    parser.add_argument("--out", default=None, help="证据包输出路径（默认写在 dest-dir 下）")
    args = parser.parse_args(argv)

    hashes = None
    if args.hashes:
        from .lifecycle import hashes_from_json

        hashes = hashes_from_json(args.hashes)
    target_modules = (
        [item.strip() for item in str(args.target_modules).split(",") if item.strip()]
        if args.target_modules
        else None
    )
    layout = None
    if args.model_inputs_layout:
        with open(args.model_inputs_layout, "r", encoding="utf-8-sig") as handle:
            layout = json.load(handle)
    template_override = None
    if args.template_override_file:
        with open(args.template_override_file, "r", encoding="utf-8") as handle:
            template_override = handle.read()
    deadline_epoch = args.deadline_epoch
    if deadline_epoch is None and os.environ.get("V3_DEADLINE_EPOCH"):
        try:
            deadline_epoch = float(os.environ["V3_DEADLINE_EPOCH"])
        except ValueError:
            raise MissingInput(
                "outer_deadline_invalid",
                "环境变量 V3_DEADLINE_EPOCH 不是数字：%r" % os.environ.get("V3_DEADLINE_EPOCH"),
            )
    # ---- r5（独审阻断 6）：受监督模式 -------------------------------------------------
    # 真正的硬截止由外层 v6 监督器签发（文件 + 自锚），子进程只读不改：
    #   * 缺文件 / 缺自锚 / 哈希不符 → 直接拒绝；
    #   * 环境里出现被剥掉的 DEADLINE/BUDGET 通道 → 判为重新注入并拒绝；
    #   * CLI 请求比签发截止更晚 → deadline_widening_rejected（显式失败，不静默夹紧）。
    supervised_deadline = None
    try:
        if args.supervised:
            supervised_deadline = supervised_check.read_deadline_file(
                args.deadline_file or os.environ.get(supervised_check.CUTOFF_FILE_ENV),
                args.deadline_file_sha256
                or os.environ.get(supervised_check.CUTOFF_SHA_ENV),
            )
    except FailClosed as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
        return exc.exit_code
    try:
        report = run_engineering_check(
            dest_dir=args.dest_dir,
            steps=int(args.steps),
            checkpoint_every=int(args.checkpoint_every),
            stop_at_step=args.stop_at_step,
            time_budget_seconds=args.time_budget_seconds,
            adapter_name=args.adapter_name,
            hashes=hashes,
            resume_probe_steps=int(args.resume_probe_steps),
            probe_cuda=bool(args.probe_cuda),
            cuda_probe_mib=int(args.cuda_probe_mib),
            backend=args.backend,
            plan_json=args.plan_json,
            model_id=args.model_id,
            model_revision=args.model_revision,
            target_modules=target_modules,
            allow_download=bool(args.allow_download),
            local_load_authorized=bool(args.local_load_authorized),
            device=args.device,
            deadline_epoch=deadline_epoch,
            model_inputs_root=args.model_inputs_root,
            model_inputs_layout=layout,
            interface_path=args.interface,
            decoy_input_roots=args.decoy_input_root,
            template_override=template_override,
            template_override_authorized=bool(args.allow_template_override),
            template_override_reason=args.template_override_reason,
            approval_path=args.approval,
            approval_expected_sha256=args.approval_expected_sha256,
            approval_evidence_ref=args.approval_evidence_ref,
            requested_gpu_hours=args.requested_gpu_hours,
            local_model_dir=args.local_model_dir,
            supervised_deadline=supervised_deadline,
        )
    except FailClosed as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
        return exc.exit_code
    out_path = args.out or os.path.join(
        report["dest_dir_resolution"]["resolved"], "engineering_check_evidence.json"
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    summary = {
        "verdict": report.get("verdict"),
        "backend": report.get("backend"),
        "is_real_training": report.get("is_real_training"),
        "is_real_gpu": report.get("is_real_gpu"),
        "evidence_path": out_path,
        "evidence_sha256": report.get("evidence_sha256"),
        "stopped": report.get("stopped"),
        "stop_reason": report.get("stop_reason"),
        "checkpoint_count": report.get("checkpoint_count"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "remaining_budget_seconds": report.get("remaining_budget_seconds"),
        "deadline_source": (report.get("budget") or {}).get("deadline_source"),
        "deadline_overrun_sections": report.get("deadline_overrun_sections"),
        "budget_ledger_sections": [item["section"] for item in report.get("budget_ledger") or []],
        "input_binding": {
            "bound": (report.get("input_binding") or {}).get("bound"),
            "binding_sha256": (report.get("input_binding") or {}).get("binding_sha256"),
            "load_view_dir": (report.get("input_binding") or {}).get("load_view_dir"),
        },
        "consumed_inputs": (report.get("input_binding") or {}).get("consumed_inputs"),
        "memory_profile_payload": report.get("memory_profile_payload"),
        "gpu_measurement_status": (report.get("memory_profile") or {}).get("gpu_measurement_status"),
        "memory_profile_same_instance": report.get("memory_profile_same_instance"),
        "cuda_probe": report.get("cuda_probe"),
        "unverified_claims": report.get("unverified_claims"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("verdict") == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
