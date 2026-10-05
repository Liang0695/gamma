"""HF `v3_policy` 训练入口（fail-closed，禁止 GPU 自动开工）。

本模块**不负责**在本机跑 GPU：它做的是
1. 静态 preflight（依赖锁、配置、目标模块、内存账、mask fixture），
2. 把"为什么现在不能开训"变成可复现的阻断报告，
3. 只有在依赖锁 verified、门槛全过、预算与操作者都明确时才允许 `start()` —— 而
   `start()` 在缺少锁定软件时会如实抛 `Blocked`，不会静默降级成 latest。

命令：
    python -m v3.train.entry --preflight
    python -m v3.train.entry --start --operator Liang --gpu-hours 6 --stage T1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from ..common.errors import Blocked, FailClosed, PolicyViolation
from ..t0.deps import DependencyLock
from ..t0.official import OfficialInterface
from .config import DEFAULT_CONFIG, TrainingConfig, assert_start_allowed
from .fixtures import run_all as run_fixture_suite
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
    GEOMETRY,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOCKS_DIR = os.path.join(REPO_ROOT, "v3", "locks")


def preflight(
    train_lock_path: str | None = None,
    serving_lock_path: str | None = None,
    interface_path: str | None = None,
    config_path: str | None = None,
) -> dict:
    """收集全部阻断项，不抛异常（供 CLI 与文档引用）。"""
    train_lock_path = train_lock_path or os.path.join(LOCKS_DIR, "train.lock.json")
    serving_lock_path = serving_lock_path or os.path.join(LOCKS_DIR, "serving.lock.json")
    interface_path = interface_path or os.path.join(LOCKS_DIR, "official-interface.json")

    blockers: list[dict] = []
    notes: list[str] = []

    config = TrainingConfig.from_file(config_path) if config_path else TrainingConfig.default()
    config_problems = config.validate()
    if config_problems:
        blockers.append({"code": "train_config_invalid", "problems": config_problems})

    train_summary = {}
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
            if label == "train":
                train_summary = summary
        except FailClosed as exc:
            blockers.append({"code": "%s_lock_load_failed" % label, "detail": exc.to_dict()})

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

    gpu_limit = MemoryPlan(peak_gpu_gib=0.0, host_rss_gib=0.0).effective_gpu_limit()
    report = {
        "stage": "T0-preflight",
        "config": {
            "seq_len": config.seq_len,
            "lora_rank": config.lora["r"],
            "optimizer_updates_cap": config.budget["max_optimizer_updates"],
            "config_sha256": config.config_sha256(),
        },
        "locks": {"train": train_summary},
        "official_interface": interface_summary,
        "target_plan": {
            "expected_modules": expected_module_count(),
            "trainable_params": lora_param_count(config.lora["r"]),
            "adapter_train_state_gib": round(adapter_train_state_gib(config.lora["r"]), 4),
            "dense_reference_bytes": dense_reference_bytes(),
            "geometry_layers": GEOMETRY["num_hidden_layers"],
            "first_round_seq_len": FIRST_ROUND_SEQ_LEN,
            "gpu_peak_limit_gib": gpu_limit,
            "disk": disk_budget_gib(),
        },
        "mask_fixtures": {
            "count": fixture_report["fixture_count"],
            "passed": fixture_report["passed"],
            "renderer": fixture_report["renderer"],
            "unverified_claims": fixture_report["unverified_claims"],
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
        ],
    }
    return report


def start(
    *,
    operator: str | None,
    gpu_hours: float,
    stage: str | None,
    v2_conflict_checked: bool = False,
    train_lock_path: str | None = None,
    config_path: str | None = None,
) -> dict:
    """真正的开训入口：本任务下必然阻断（软件锁未验证 + 门槛未过 + 未获 GPU 授权）。

    阻断顺序刻意先查授权与门槛，再查软件锁：这样报告出来的是"为什么不能开训"，
    而不是一个来自最底层的 ImportError。
    """
    train_lock_path = train_lock_path or os.path.join(LOCKS_DIR, "train.lock.json")
    config = TrainingConfig.from_file(config_path) if config_path else TrainingConfig.default()
    config.assert_valid()
    lock = DependencyLock.from_file(train_lock_path)

    gates = {"t1_engineering_pass": False, "memory_plan_pass": False, "export_manifest_pass": False}
    authorization = {
        "gpu_hours_released": gpu_hours,
        "operator": operator,
        "stage": stage,
        "v2_conflict_checked": v2_conflict_checked,
    }

    problems: list[str] = []
    try:
        assert_no_auto_gpu(
            {"allow_gpu": bool(operator and gpu_hours > 0), "gates_passed": all(gates.values()), "budget_gpu_hours": gpu_hours}
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
        )
    raise Blocked(
        "gpu_execution_not_implemented",
        "本编码任务不携带 GPU/锁定训练软件，训练执行交 Liang 统筹并在门槛验收后启动",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V3 v3_policy 训练入口（fail-closed）")
    parser.add_argument("--preflight", action="store_true", help="只做静态检查并打印阻断项")
    parser.add_argument("--start", action="store_true", help="尝试开训（当前必然阻断）")
    parser.add_argument("--operator", default=None)
    parser.add_argument("--gpu-hours", type=float, default=0.0)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--v2-conflict-checked", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    if args.start:
        try:
            start(
                operator=args.operator,
                gpu_hours=args.gpu_hours,
                stage=args.stage,
                v2_conflict_checked=args.v2_conflict_checked,
                config_path=args.config,
            )
        except FailClosed as exc:
            print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
            return exc.exit_code
        return 0

    report = preflight(config_path=args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not report["blockers"] else 5


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
