"""HF `v3_policy` 训练入口（fail-closed，禁止 GPU 自动开工）。

职责边界（KAGGLE-26 / Q0 报告 §7 与 Mika 待审处置的明确要求）：

- **实现**在本仓库里：真实模型加载 / LoRA 挂载 / 前向 / 反向 / 优化器步进 /
  adapter-only 保存与重载，全部在 `v3/train/runner.py`，CPU 上可跑自检；
- **执行**仍受闸门约束：依赖锁 verified、三项门槛**实测**通过、操作者与预算明确，
  才允许 `start()` 真正开训；GPU 实耗交 Liang 统筹链，本编码任务不独自申请作业。

与旧版的差别：旧 `start()` 在通过全部闸门后**无条件**抛出一个"未实现/不执行"的阻断
（该错误码已从仓库中彻底移除，见 `tests/test_q0_train_entry.py` 的全仓检查），
等于不存在训练路径；三个 `gates` 也是硬编码常量 `{"t1_engineering_pass": False, ...}`，
不是测量结果。现在：

1. `gates` 由 :func:`measure_gates` **实测**得出（fixture 全过 + CPU 训练循环自检、
   真实内存实测、导出 manifest 校验），每项都带证据与"为什么不通过"的原因；
2. 通过后 `start()` 会真的调用 `runner.run_training()` 并返回运行报告，
   不再有"无条件抛出"的路径。

命令：
    python -m v3.train.entry --preflight          # 静态检查 + 闸门实测状态
    python -m v3.train.entry --smoke              # 只跑 CPU 训练循环自检
    python -m v3.train.entry --start --operator Liang --gpu-hours 6 --stage T1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Sequence

from ..common.errors import Blocked, FailClosed, MissingInput, PolicyViolation, UnverifiedLock
from ..submit import adapter_contract
from ..t0.deps import DependencyLock, assert_distinct_lock_channels, load_pair
from ..t0.official import OfficialInterface
from . import adapter_export_contract, destdir, profile as profile_mod, runner
from .checkpoint import REQUIRED_HASH_KEYS, assert_adapter_valid_frozen_spec
from .config import TrainingConfig, assert_start_allowed
from .fixtures import run_all as run_fixture_suite
from .lifecycle import STOP_EXTERNAL, LifecyclePolicy, StopRequest, TrainingLifecycle
from .targets import (
    FIRST_ROUND_SEQ_LEN,
    MemoryPlan,
    ModulePlan,
    adapter_train_state_gib,
    assert_no_auto_gpu,
    dense_reference_bytes,
    disk_budget_gib,
    expected_module_count,
    lora_param_count,
    target_module_names,
    GEOMETRY,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOCKS_DIR = os.path.join(REPO_ROOT, "v3", "locks")

#: 三个开训闸门的名字（顺序即报告顺序）。
GATE_NAMES = ("t1_engineering_pass", "memory_plan_pass", "export_manifest_pass")

#: 真实后端必须从锁里取到的 pin（缺一即拒绝，**没有**硬编码回退）。
REQUIRED_BACKEND_PINS = ("model_repo_id", "model_revision")


def resolve_backend_pins(
    *,
    interface_path: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    target_modules: Sequence[str] | None = None,
    config: TrainingConfig | None = None,
) -> dict:
    """解析真实后端的 `(model_id, revision, target_modules)`：**锁优先，缺 pin 必拒绝**。

    旧实现（🟡-3）三处都错：

    1. `authorization.get("model_id")` —— `authorization` 只有 4 个键（gpu_hours_released /
       operator / stage / v2_conflict_checked），这个取值**恒为 None**；
    2. 于是落到硬编码 `"google/gemma-4-1b-it"` —— **不是**锁定模型
       `google/gemma-4-31B-it-qat-w4a16-ct`；
    3. `DEFAULT_CONFIG["lora"]["target_suffixes"]` —— 这个键**根本不存在**，
       `.get(..., [])` 静默返回空清单，即使 revision 修好也会 `lora_not_mounted`。

    现在：显式参数优先；否则从 `official-interface.json` 的 pins 取
    `model_repo_id` / `model_revision`（`OfficialInterface.pin()` 对 `verified != true`
    的 pin 抛 `UnverifiedLock`）；`target_modules` 由目标正则机械导出。
    取不到任何一项都抛 fail-closed 异常，**不猜、不回退**。
    """
    config = config or TrainingConfig.default()
    #: 明确不是 pin 的取值（`latest`/空/None 一律拒绝，绝不静默换成别的 revision）。
    unpinned = ("", "latest", "main", "head", "none", "null", "unknown")
    source = "explicit-argument"
    resolved_id: str | None = None
    resolved_revision: str | None = None
    if model_id is not None:
        text = str(model_id).strip()
        if not text or text.lower() in unpinned:
            raise MissingInput(
                "model_id_pin_missing",
                "显式给出的 model_id 为空或无意义：不得回退到别处取模型",
                model_id=model_id,
            )
        resolved_id = text
    if model_revision is not None:
        text = str(model_revision).strip()
        if text.lower() in unpinned:
            raise UnverifiedLock(
                "model_revision_pin_missing",
                "显式给出的 revision 不是 pin（%r）：显式传入就必须是精确 revision" % (model_revision,),
                revision=model_revision,
            )
        resolved_revision = text

    if resolved_id is None or resolved_revision is None:
        path = interface_path or os.path.join(LOCKS_DIR, "official-interface.json")
        if not os.path.exists(path):
            raise MissingInput(
                "official_interface_missing",
                "找不到官方接口锁，无法取 model_repo_id / model_revision pin：%s" % path,
                path=path,
            )
        interface = OfficialInterface.from_file(path)
        if resolved_id is None or resolved_revision is None:
            source = "official-interface.json:pins"
        if resolved_id is None:
            resolved_id = interface.pin("model_repo_id")
        if resolved_revision is None:
            # `pin()` 先校验 verified == true；再核对原始 value 确实是字符串 pin，
            # 否则 `str(None)` 会变成 "None" 悄悄通过。
            resolved_revision = interface.pin("model_revision")
            raw = interface.pin_entry("model_revision").get("value")
            if not isinstance(raw, str) or raw.strip().lower() in unpinned:
                raise UnverifiedLock(
                    "model_revision_pin_missing",
                    "official-interface 的 model_revision pin 不是有效 revision：%r" % (raw,),
                    revision=raw,
                )
    if not resolved_id:
        raise MissingInput(
            "model_id_pin_missing",
            "取不到锁定的 model_repo_id：不得回退到硬编码模型",
        )
    lora = config.lora
    modules = [str(item) for item in (target_modules or [])]
    if not modules:
        # 从配置里的目标正则导出；无正则时退回冻结常量 TARGET_REGEX。
        regex = str(lora.get("target_modules_regex") or "").strip()
        modules = target_module_names(regex) if regex else target_module_names()
    if not modules:
        raise MissingInput(
            "target_modules_missing",
            "取不到 target_modules：不得用空清单构造 LoRA（会静默变成 lora_not_mounted）",
        )
    return {
        "model_id": resolved_id,
        "model_revision": str(resolved_revision),
        "target_modules": sorted(modules),
        "source": source,
    }


def measure_gates(
    *,
    fixture_report: dict | None = None,
    memory_profile: dict | None = None,
    export_manifest: dict | None = None,
    export_dir: str | None = None,
    engineering_evidence: dict | None = None,
    run_cpu_self_check: bool = True,
) -> dict:
    """**实测**三项开训闸门，返回 `{"gates": {...}, "evidence": {...}}`。

    每一项都给出布尔结果 + 依据 + 未通过原因，不做任何"默认 True"的假设。

    - `t1_engineering_pass`：20 项 mask fixture 全过 **且** CPU 训练循环自检
      （前向/反向/优化器步进/保存重载）全过；
    - `memory_plan_pass`：必须有**真实测量**的显存/RSS 数字，且测量来源可核
      （G7b：`v3.train.profile.check_profile_provenance`）。手打的
      `{"peak_gpu_gib": ..., "host_rss_gib": ...}` 不再能过门；
    - `export_manifest_pass`：必须给出 `v3-adapter-export/2` 契约的导出清单**并绑定
      实际产物目录**（`export_dir`），由消费侧逐文件重算哈希对账（G8：
      `v3.train.adapter_export_contract`）。清单里自报的
      `load_ok` / `params_changed` / `fixture_pass` / `fixture_total` 一律拒绝。
    """
    fixtures = fixture_report if fixture_report is not None else run_fixture_suite()
    self_check = runner.self_check_cpu() if run_cpu_self_check else {"all_passed": False, "skipped": True}

    memory = None
    memory_provenance: dict | None = None
    memory_reason = "没有提供实测内存 profile：未实测不得当作通过（设计常量只是上限）"
    if isinstance(memory_profile, dict) and memory_profile:
        memory_provenance = profile_mod.check_profile_provenance(memory_profile)
        if not memory_provenance["ok"]:
            memory = {
                "measured": True,
                "pass": False,
                "status": "blocked",
                "problems": list(memory_provenance["problems"]),
            }
            memory_reason = "内存 profile 未通过来源校验：%s" % memory_provenance["problems"][0]
        else:
            try:
                # Q0 Y9-2：走 evaluate_or_block —— 缺实测字段 / 实测不过门槛都 fail-closed，
                # 不再是"只取 effective_gpu_limit() 常量"。
                memory = MemoryPlan(
                    peak_gpu_gib=float(memory_profile.get("peak_gpu_gib") or 0.0),
                    host_rss_gib=float(memory_profile.get("host_rss_gib") or 0.0),
                    seq_len=int(memory_profile.get("seq_len", FIRST_ROUND_SEQ_LEN)),
                    available_gpu_gib=memory_profile.get("available_gpu_gib"),
                    downgrade_used=bool(memory_profile.get("downgrade_used", False)),
                ).evaluate_or_block(memory_profile)
                memory["source"] = memory_profile.get("source", "caller-provided-measurement")
                memory["provenance"] = memory_provenance
            except FailClosed as exc:
                memory = {"measured": True, "pass": False, "status": "blocked", "error": exc.to_dict()}
                memory_reason = "实测内存判定 fail-closed：%s（%s）" % (exc.code, exc.message)

    # —— G8：导出清单必须过版本化契约，并且与**实际产物字节**对账 ——
    manifest_problems: list[str] = adapter_export_contract.collect_export_manifest_problems(
        export_manifest,
        export_dir=export_dir,
        require_artifacts=True,
    )
    adapter_validation: dict | None = None
    if not manifest_problems and export_manifest:
        try:
            adapter_validation = adapter_export_contract.validate_export_manifest(
                export_manifest, export_dir=export_dir
            )
        except FailClosed as exc:  # pragma: no cover - collect 已先跑过一遍
            manifest_problems.append("导出清单校验未过：%s（%s）" % (exc.code, exc.message))

    # —— 工程检查证据：只做**交叉核对**，不当作通过依据 ——
    cross_checks: list[str] = []
    cross_failures: list[str] = []
    engineering: dict | None = None
    if isinstance(engineering_evidence, dict) and engineering_evidence:
        engineering = {
            "stage": engineering_evidence.get("stage"),
            "backend": engineering_evidence.get("backend"),
            "is_real_gpu": bool(engineering_evidence.get("is_real_gpu", False)),
            "is_real_training": bool(engineering_evidence.get("is_real_training", False)),
        }
        declared_pass = engineering_evidence.get("fixture_pass")
        declared_total = engineering_evidence.get("fixture_total")
        if declared_pass is not None or declared_total is not None:
            recomputed_pass = int(fixtures.get("passed") or 0)
            recomputed_total = int(fixtures.get("fixture_count") or 0)
            if int(declared_pass or 0) != recomputed_pass or int(declared_total or 0) != recomputed_total:
                cross_failures.append(
                    "工程检查证据自报 fixture %s/%s，本入口重算为 %s/%s：不一致"
                    % (declared_pass, declared_total, recomputed_pass, recomputed_total)
                )
            else:
                cross_checks.append("fixture 计数与重算一致（%d/%d）" % (recomputed_pass, recomputed_total))
                # 计数一致时才走冻结规格的生产入口（20/20 是硬要求）。
                try:
                    adapter_validation = assert_adapter_valid_frozen_spec(
                        dict(export_manifest or {}),
                        load_ok=bool(engineering_evidence.get("load_ok", False)),
                        params_changed=bool(engineering_evidence.get("params_changed", False)),
                        fixture_pass=int(declared_pass or 0),
                        fixture_total=int(declared_total or 0),
                    )
                    cross_checks.append("冻结规格 20/20 校验通过（计数来自重算一致的证据）")
                except FailClosed as exc:
                    cross_failures.append(
                        "adapter 冻结规格校验未过：%s（%s）" % (exc.code, exc.message)
                    )
    if cross_failures:
        manifest_problems.extend(cross_failures)

    gates = {
        "t1_engineering_pass": bool(fixtures.get("all_passed")) and bool(self_check.get("all_passed")),
        "memory_plan_pass": bool(memory and memory.get("status") == "pass"),
        "export_manifest_pass": not manifest_problems,
    }
    evidence = {
        "mask_fixtures": {
            "count": fixtures.get("fixture_count"),
            "passed": fixtures.get("passed"),
            "all_passed": bool(fixtures.get("all_passed")),
            "renderer": fixtures.get("renderer"),
        },
        "cpu_training_self_check": self_check,
        "memory_plan": memory,
        "memory_plan_reason": memory_reason if not gates["memory_plan_pass"] else None,
        "memory_profile_provenance": memory_provenance,
        "export_manifest": {
            "provided": bool(export_manifest),
            "export_dir": export_dir,
            "contract": adapter_export_contract.PRODUCTION_FORMAT,
            "problems": manifest_problems,
            "adapter_validation": adapter_validation,
            "engineering_evidence": engineering,
            "cross_checks": cross_checks,
        },
    }
    return {"gates": gates, "evidence": evidence}


def _lock_summaries(
    train_lock_path: str, serving_lock_path: str
) -> tuple[dict, dict, dict, list[dict]]:
    """读两份依赖锁并做**通道隔离**判定（Q0 Y9：接入 `deps.load_pair`）。

    `load_pair` 内部会调用 `assert_channels_isolated`（内容级）与
    `assert_distinct_lock_channels`（归一化路径 / 文件字节 SHA / 包版本映射级），
    因此"训练锁与推理锁被换成同一份"会在 preflight 阶段就阻断，而不是只在静态审查里被提。
    返回 `(train_summary, serving_summary, isolation, blockers)`。
    """
    blockers: list[dict] = []
    summaries: dict[str, dict] = {}
    isolation: dict = {"checked": False}
    try:
        train_lock, serving_lock = load_pair(train_lock_path, serving_lock_path)
        isolation = {"checked": True, **assert_distinct_lock_channels(train_lock, serving_lock)}
    except FailClosed as exc:
        # 两侧锁被换成同一份（或被隔离规则判为同一份）时在此阻断。
        blockers.append({"code": "lock_channels_not_isolated", "detail": exc.to_dict()})
        isolation = {"checked": True, "distinct": False, "error": exc.to_dict()}
    for label, path in (("train", train_lock_path), ("serving", serving_lock_path)):
        try:
            lock = DependencyLock.from_file(path)
            problems = lock.validate()
            if problems:
                blockers.append({"code": "%s_lock_invalid" % label, "problems": problems})
            summary = lock.summary()
            if not summary["verified"]:
                blockers.append(
                    {
                        "code": "%s_lock_unverified" % label,
                        "message": "%s 依赖锁 verified=false：禁止自动取 latest" % label,
                    }
                )
            summaries[label] = summary
        except FailClosed as exc:
            blockers.append({"code": "%s_lock_load_failed" % label, "detail": exc.to_dict()})
    return summaries.get("train", {}), summaries.get("serving", {}), isolation, blockers


def preflight(
    train_lock_path: str | None = None,
    serving_lock_path: str | None = None,
    interface_path: str | None = None,
    config_path: str | None = None,
) -> dict:
    """收集全部阻断项与闸门实测状态，不抛异常（供 CLI 与文档引用）。"""
    train_lock_path = train_lock_path or os.path.join(LOCKS_DIR, "train.lock.json")
    serving_lock_path = serving_lock_path or os.path.join(LOCKS_DIR, "serving.lock.json")
    interface_path = interface_path or os.path.join(LOCKS_DIR, "official-interface.json")

    blockers: list[dict] = []
    notes: list[str] = []

    config = TrainingConfig.from_file(config_path) if config_path else TrainingConfig.default()
    config_problems = config.validate()
    if config_problems:
        blockers.append({"code": "train_config_invalid", "problems": config_problems})

    train_summary, serving_summary, isolation, lock_blockers = _lock_summaries(
        train_lock_path, serving_lock_path
    )
    blockers.extend(lock_blockers)

    interface_summary: dict = {}
    try:
        interface = OfficialInterface.from_file(interface_path)
        interface.assert_registry_complete()
        interface_summary = {
            "tool_count": len(interface.tools()),
            "interface_sha256": interface.interface_sha256(),
            "verified_pins": interface.verified_pins(),
        }
        unverified_pins = [k for k, v in interface_summary["verified_pins"].items() if not v]
        if unverified_pins:
            blockers.append(
                {
                    "code": "interface_pins_unverified",
                    "pins": sorted(unverified_pins),
                    "message": "未验证的 pin 不得当作已确认事实使用",
                }
            )
    except FailClosed as exc:
        blockers.append({"code": "official_interface_failed", "detail": exc.to_dict()})

    plan = ModulePlan.from_names([])
    if len(plan.matched) != expected_module_count():
        notes.append(
            "未提供真实模块清单：target 模块数按设计值 %d 计，实际必须导出后才允许训练"
            % expected_module_count()
        )

    fixture_report = run_fixture_suite()
    if not fixture_report["all_passed"]:
        blockers.append(
            {
                "code": "mask_fixtures_failed",
                "failed": [
                    item["fixture_id"] for item in fixture_report["results"] if item["status"] != "pass"
                ],
            }
        )

    gate_state = measure_gates(fixture_report=fixture_report)
    # 闸门未过**不是**新的 blocker code（阻断项集合保持稳定，便于与既有证据逐项比对）；
    # 它体现在 `gates` 段与 `can_start_training_now` 上。
    for name, passed in gate_state["gates"].items():
        if not passed:
            notes.append("开训闸门 %s 未通过（见 gates.evidence）" % name)

    # Q0 Y9：内存门槛必须走 MemoryPlan.evaluate()，而不是只取 effective_gpu_limit() 常量。
    memory_upper_bound = MemoryPlan(peak_gpu_gib=0.0, host_rss_gib=0.0).effective_gpu_limit()
    memory_unmeasured = MemoryPlan(peak_gpu_gib=0.0, host_rss_gib=0.0).evaluate()
    memory_unmeasured["measured"] = False
    memory_unmeasured["note"] = (
        "这是**未实测**时的对照：0 GiB 占用当然过门槛，因此该结果不构成任何内存合格结论；"
        "真正的门槛由实测 profile 经 measure_gates() 判定"
    )

    report = {
        "stage": "T0-preflight",
        "config": {
            "seq_len": config.seq_len,
            "lora_rank": config.lora["r"],
            "optimizer_updates_cap": config.budget["max_optimizer_updates"],
            "config_sha256": config.config_sha256(),
        },
        "locks": {"train": train_summary, "serving": serving_summary, "channels_isolated": isolation},
        "official_interface": interface_summary,
        "target_plan": {
            "expected_modules": expected_module_count(),
            "trainable_params": lora_param_count(config.lora["r"]),
            "adapter_train_state_gib": round(adapter_train_state_gib(config.lora["r"]), 4),
            "dense_reference_bytes": dense_reference_bytes(),
            "geometry_layers": GEOMETRY["num_hidden_layers"],
            "first_round_seq_len": FIRST_ROUND_SEQ_LEN,
            "gpu_peak_limit_gib": memory_upper_bound,
            "memory_plan_upper_bound_only": True,
            "memory_plan_unmeasured_baseline": memory_unmeasured,
            "disk": disk_budget_gib(),
        },
        "mask_fixtures": {
            "count": fixture_report["fixture_count"],
            "passed": fixture_report["passed"],
            "renderer": fixture_report["renderer"],
            "unverified_claims": fixture_report["unverified_claims"],
        },
        "gates": gate_state["gates"],
        "gates_evidence": gate_state["evidence"],
        "training_entry": {
            "implemented": True,
            "module": "v3/train/runner.py",
            "backends": ["synthetic-cpu-loop（CPU 自检）", "torch-peft-bf16-lora（GPU 执行面，未实测）"],
            "real_gpu_executed_on_this_machine": False,
            "note": "训练实现已在仓库内且 CPU 自检可复跑；GPU 实耗交 Liang 统筹链并先过闸门",
        },
        "blockers": blockers,
        "notes": notes,
        "can_start_training_now": False,
        "unverified_claims": [
            "实际训练收益（未跑真实 GPU）",
            "TP4 / 长上下文行为",
            "官方模板渲染逐字节一致",
            "vLLM 实际 mapper 版本与加载回归",
            "真实模块 shape 与显存峰值",
            "torch/peft 后端在真实 GPU 上的端到端行为（本机无 GPU、无锁定软件）",
        ],
    }
    return report


def start(
    *,
    operator: str | None,
    gpu_hours: float,
    stage: str | None,
    v2_conflict_checked: bool = False,
    model_id: str | None = None,
    model_revision: str | None = None,
    target_modules: Sequence[str] | None = None,
    train_lock_path: str | None = None,
    serving_lock_path: str | None = None,
    interface_path: str | None = None,
    config_path: str | None = None,
    memory_profile: dict | None = None,
    export_manifest: dict | None = None,
    backend=None,
    plan=None,
    dest_dir: str | None = None,
    export_dir: str | None = None,
    engineering_evidence: dict | None = None,
    checkpoint_root: str | None = None,
    hashes: dict | None = None,
    resume_from: str | None = None,
    interrupted_mid_accum: bool = False,
    checkpoint_every: int = 0,
    stop_at_step: int | None = None,
    time_budget_seconds: float | None = None,
    stop: StopRequest | None = None,
    allow_download: bool = False,
) -> dict:
    """真正的开训入口：闸门全过且授权明确时**真的执行训练**，否则 fail-closed。

    阻断顺序刻意先查授权与门槛，再查软件锁：报告出来的是"为什么不能开训"，
    而不是一个来自最底层的 ImportError。

    与旧版的差别：通过全部检查后**不再无条件抛出**。`backend` 省略时使用真实
    `TorchPeftBackend`（缺锁定依赖/GPU 时会以环境原因 `Blocked`）；闸门或授权不满足时
    仍然抛 `PolicyViolation("start_not_allowed")` —— fail-closed 保持不变。

    🟡-3：`model_id` / `model_revision` / `target_modules` 不再有硬编码回退 ——
    未显式给出时一律走 :func:`resolve_backend_pins`（来自 official-interface 锁的
    `model_repo_id` / `model_revision` pin，`verified != true` 即 `UnverifiedLock`；
    `target_modules` 由目标正则机械导出）。缺 pin 在**构造后端之前**就拒绝。

    G13：`dest_dir` **必须显式给出**，且经 :func:`v3.train.destdir.resolve_dest_dir`
    校验为"绝对路径 + 仓库外 + 可写"。产物路径（evidence / 保存 / 重载）随后由
    :func:`v3.train.destdir.verify_paths_consistent` 核对同源。

    G14：给出 `checkpoint_root` + `hashes` 时接线 checkpoint 保存/恢复与停止处理；
    `resume_from` 指定要恢复的 checkpoint 目录，`interrupted_mid_accum` 触发回滚，
    `stop` 提供信号/截止时间停止请求（省略时按 `time_budget_seconds` 新建一个）。
    """
    train_lock_path = train_lock_path or os.path.join(LOCKS_DIR, "train.lock.json")
    serving_lock_path = serving_lock_path or os.path.join(LOCKS_DIR, "serving.lock.json")
    config = TrainingConfig.from_file(config_path) if config_path else TrainingConfig.default()
    config.assert_valid()
    lock = DependencyLock.from_file(train_lock_path)

    # 先把 pin 解析出来：缺 pin 必须在这里就停下，而不是等到构造后端才炸。
    pins = resolve_backend_pins(
        interface_path=interface_path,
        model_id=model_id,
        model_revision=model_revision,
        target_modules=target_modules,
        config=config,
    )

    gate_state = measure_gates(
        memory_profile=memory_profile,
        export_manifest=export_manifest,
        export_dir=export_dir,
        engineering_evidence=engineering_evidence,
    )
    gates = gate_state["gates"]
    authorization = {
        "gpu_hours_released": gpu_hours,
        "operator": operator,
        "stage": stage,
        "v2_conflict_checked": v2_conflict_checked,
        "model_id": pins["model_id"],
        "model_revision": pins["model_revision"],
        "target_modules": list(pins["target_modules"]),
        "pins_source": pins["source"],
    }

    problems: list[str] = []
    try:
        assert_no_auto_gpu(
            {
                "allow_gpu": bool(operator and gpu_hours > 0),
                "gates_passed": all(gates.values()),
                "budget_gpu_hours": gpu_hours,
            }
        )
    except PolicyViolation as exc:
        problems.append(exc.message)
    try:
        assert_start_allowed(
            config=config, train_lock_summary=lock.summary(), gates=gates, authorization=authorization
        )
    except PolicyViolation as exc:
        problems.extend(exc.context.get("problems") or [exc.message])

    if problems:
        raise PolicyViolation(
            "start_not_allowed",
            "开训闸门未通过（%d 项）：GPU 执行交 Liang 统筹且门槛验收后才启动" % len(problems),
            problems=problems,
            gates=gates,
            gate_evidence=gate_state["evidence"],
        )

    if plan is None:
        raise MissingInput(
            "training_plan_missing",
            "闸门已过但未提供训练计划（批次/lr/steps）：入口不会自行编造训练数据",
            gates=gates,
        )
    if backend is None:
        # 参数全部来自 resolve_backend_pins：无硬编码模型、无编造的 target 清单。
        backend = runner.TorchPeftBackend(
            model_id=pins["model_id"],
            revision=pins["model_revision"],
            target_modules=pins["target_modules"],
            allow_download=allow_download,
        )

    # —— G13：产物目录必须显式给出、且在仓库外 ——
    resolution = destdir.resolve_dest_dir(dest_dir, must_be_outside_repo=True)

    # —— G14：停止请求与 checkpoint 生命周期 ——
    if stop is None:
        stop = StopRequest(
            deadline_monotonic=(
                None if time_budget_seconds is None
                else time.monotonic() + float(time_budget_seconds)
            )
        )
    lifecycle = None
    if checkpoint_root or resume_from:
        if not checkpoint_root:
            raise MissingInput(
                "checkpoint_root_missing",
                "给了 resume_from 却没有 checkpoint_root：无法定位 checkpoint 存储根",
            )
        lifecycle = TrainingLifecycle(
            store_root=checkpoint_root,
            hashes=dict(hashes or {}),
            policy=LifecyclePolicy(
                checkpoint_every=int(checkpoint_every), stop_at_step=stop_at_step
            ),
            resume_from=resume_from,
            interrupted_mid_accum=bool(interrupted_mid_accum),
            stop=stop,
        )

    report = runner.run_training(
        backend,
        plan,
        resolution["resolved"],
        lifecycle=lifecycle,
        stop=stop,
    )
    # 输出 / 保存 / 重载必须落在同一个已解析的 dest_dir 之下。
    report["dest_dir_resolution"] = resolution
    report["path_consistency"] = destdir.verify_paths_consistent(
        resolution,
        [
            ("adapter_saved", (report.get("saved") or {}).get("path")),
            ("adapter_reloaded", (report.get("reloaded") or {}).get("path")),
            ("evidence", report.get("evidence_path")),
        ],
    )
    report["gates"] = gates
    report["gate_evidence"] = gate_state["evidence"]
    report["authorization"] = authorization
    report["backend_pins"] = pins
    report["operator"] = operator
    report["stage"] = stage
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V3 v3_policy 训练入口（fail-closed）")
    parser.add_argument("--preflight", action="store_true", help="只做静态检查并打印阻断项")
    parser.add_argument("--smoke", action="store_true", help="只跑 CPU 训练循环自检")
    parser.add_argument("--start", action="store_true", help="尝试开训（闸门未过仍会阻断）")
    parser.add_argument("--operator", default=None)
    parser.add_argument("--gpu-hours", type=float, default=0.0)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--v2-conflict-checked", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--memory-profile", default=None, help="实测内存 profile 的 JSON 路径")
    parser.add_argument("--export-manifest", default=None, help="导出 manifest 的 JSON 路径")
    # G8：清单必须与**实际产物目录**对账（无此参数即 fail-closed）。
    parser.add_argument("--export-dir", default=None, help="adapter 导出产物目录（仓库外绝对路径）")
    parser.add_argument(
        "--engineering-evidence", default=None, help="工程检查证据 JSON（只做交叉核对，不作为通过依据）"
    )
    # G13：产物目录必须显式给出。
    parser.add_argument(
        "--dest-dir",
        default=None,
        help="训练产物目录：绝对路径且必须在仓库外（无默认值，缺省即拒绝开训）",
    )
    # G14：checkpoint 保存 / 恢复 / 停止。
    parser.add_argument("--checkpoint-root", default=None, help="checkpoint 存储根目录（仓库外）")
    parser.add_argument("--hashes", default=None, help="5 项环境/数据哈希的 JSON 路径（启用 checkpoint 必需）")
    parser.add_argument("--resume-from", default=None, help="要恢复的 checkpoint 目录")
    parser.add_argument(
        "--interrupted-mid-accum",
        action="store_true",
        help="上次中断发生在 accum 中途：恢复时回滚到上一份完整 optimizer 步",
    )
    parser.add_argument("--checkpoint-every", type=int, default=0, help="每 N 个 optimizer 步落一份 checkpoint")
    parser.add_argument("--stop-at-step", type=int, default=None, help="到达该步后立刻停止并保存")
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="硬截止（monotonic 计时）：到点即停止、保存、重载，用于同一总截止内停机",
    )
    # 🟡-3：显式 pin 可选；不给时一律从 official-interface 锁取（无硬编码回退）。
    parser.add_argument("--interface", default=None, help="official-interface.json 路径")
    parser.add_argument("--model-id", default=None, help="显式覆盖锁定的 model_repo_id")
    parser.add_argument("--model-revision", default=None, help="显式覆盖锁定的 model_revision")
    parser.add_argument(
        "--target-modules",
        default=None,
        help="逗号分隔的 target_modules（例如 q_proj,o_proj）；不给则从目标正则导出",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        report = runner.self_check_cpu()
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["all_passed"] else 1

    if args.start:
        memory_profile = None
        if args.memory_profile:
            with open(args.memory_profile, "r", encoding="utf-8") as handle:
                memory_profile = json.load(handle)
        export_manifest = None
        if args.export_manifest:
            with open(args.export_manifest, "r", encoding="utf-8") as handle:
                export_manifest = json.load(handle)
        engineering_evidence = None
        if args.engineering_evidence:
            with open(args.engineering_evidence, "r", encoding="utf-8") as handle:
                engineering_evidence = json.load(handle)
        hashes = None
        if args.hashes:
            from .lifecycle import hashes_from_json

            hashes = hashes_from_json(args.hashes)
        try:
            start(
                operator=args.operator,
                gpu_hours=args.gpu_hours,
                stage=args.stage,
                v2_conflict_checked=args.v2_conflict_checked,
                model_id=args.model_id,
                model_revision=args.model_revision,
                target_modules=(
                    [item.strip() for item in args.target_modules.split(",") if item.strip()]
                    if args.target_modules
                    else None
                ),
                interface_path=args.interface,
                config_path=args.config,
                memory_profile=memory_profile,
                export_manifest=export_manifest,
                export_dir=args.export_dir,
                engineering_evidence=engineering_evidence,
                dest_dir=args.dest_dir,
                checkpoint_root=args.checkpoint_root,
                hashes=hashes,
                resume_from=args.resume_from,
                interrupted_mid_accum=bool(args.interrupted_mid_accum),
                checkpoint_every=int(args.checkpoint_every),
                stop_at_step=args.stop_at_step,
                time_budget_seconds=args.time_budget_seconds,
            )
        except FailClosed as exc:
            print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
            return exc.exit_code
        return 0

    report = preflight(config_path=args.config, interface_path=args.interface)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not report["blockers"] else 5


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
