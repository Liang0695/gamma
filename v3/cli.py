"""V3 数据/训练流水线 CLI（KAGGLE-21 §7 的契约实现，本轮为 fail-closed 骨架）。

    python -m v3.cli ingest --source-lock locks/source.lock.json --out out/source_manifest.json
    python -m v3.cli split --registry data/registry.json --denylist locks/v2-denylist.json
    python -m v3.cli validate-env --manifest data/env.lock.json
    python -m v3.cli verify --question-set data/p0-eight.json --repeat 2 --fresh
    python -m v3.cli export --records data/records.json --template-lock locks/template.lock.json --out out/export.json
    python -m v3.cli audit --release data/release.json
    python -m v3.cli rollout --train-only --budget locks/budget.json
    python -m v3.cli exp1 --corpus data/corpus.json --output-dir out/
    python -m v3.cli train-preflight

每步都做到「输入 hash → 输出 manifest，失败非零退出且写状态」，
并且**不自动扩大预算、不补缺题、不启动第二个 GPU 作业**。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Mapping, Sequence

from .common.canonical import sha256_json, write_canonical
from .common.errors import (
    Blocked,
    FailClosed,
    IntegrityError,
    MissingInput,
    PolicyViolation,
    UnverifiedLock,
)
from .data.dedup import (
    assert_no_split_leak,
    dedup_report,
    deterministically_select,
    selection_manifest,
)
from .data.exporter import build_export
from .data.faces import FACE_ENV_LOCK, validate_face, validate_release
from .data.oracle import SubprocessRunner, validate_question_set
from .data.source_lock import adapt, assert_train_only, ingest_manifest
from .exp.exp1 import build_synthetic_corpus, load_corpus, run_exp1
from .t0.deps import DependencyLock
from .train.template import CANONICAL_TEMPLATE_SHA256

EXIT_OK = 0
EXIT_BLOCKED = 5


def _read_json(path: str | None, what: str, required: bool = True) -> dict:
    if not path:
        if required:
            raise MissingInput("missing_argument", "缺少必需的 %s" % what)
        return {}
    if not os.path.exists(path):
        raise MissingInput("file_not_found", "%s 不存在：%s" % (what, path), path=path)
    # utf-8-sig：本仓库写出的 JSON 一律无 BOM，但同伴用 PowerShell 导出时会带 BOM，
    # 直接拒收会让交接莫名其妙地失败。
    with open(path, "r", encoding="utf-8-sig") as handle:
        text = handle.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PolicyViolation(
            "invalid_json", "%s 不是合法 JSON：%s" % (what, exc), path=path
        )


def _write(path: str | None, payload: Mapping) -> str | None:
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    digest = write_canonical(path, payload)
    return digest


def _finalize(payload: Mapping) -> tuple[dict, str]:
    """先对**不含自身 hash** 的正文求 hash，再把 hash 写进 manifest（外部账口径）。"""
    digest = sha256_json(payload)
    final = dict(payload)
    final["manifest_sha256"] = digest
    return final, digest


def _count_tokens(messages) -> int:
    return sum(len(str(message.get("content", "")).split()) for message in messages) + len(messages)


# ---------------------------------------------------------------- ingest


def cmd_ingest(args) -> int:
    """校验来源锁：固定 revision + 许可 approved，逐条不留空。

    支持两种形状：本仓库的 `{"sources": [...]}` 与 D0（KAGGLE-23）的
    `{"repos": {...}}`（见 v3/data/source_lock.py）。归一后统一校验；不通过即非零退出。

    `--train-only` 生成训练侧视图：全量记录都做许可/revision 判定，
    只有 `split_role="train"` 的记录进入 `sources`（其余列在 `excluded_non_train`）。
    """
    lock = _read_json(args.source_lock, "--source-lock")
    manifest = ingest_manifest(lock, train_only=bool(args.train_only))
    manifest, digest = _finalize(manifest)
    _write(args.out, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------- split


def cmd_split(args) -> int:
    """家族/时间隔离 + 去重 + 确定性选择；V2 排除清单相交即阻断。"""
    registry = _read_json(args.registry, "--registry")
    records = registry.get("tasks")
    if not records:
        raise MissingInput("empty_registry", "registry 没有 tasks")
    denylist_payload = _read_json(args.denylist, "--denylist", required=False)
    denylist = list(denylist_payload.get("families") or []) + list(denylist_payload.get("repos") or [])
    assert_no_split_leak(records, denylist)

    report = dedup_report(records)
    if report["status"] != "clean" and not args.allow_review:
        raise PolicyViolation(
            "dedup_review_required",
            "检测到重复/近似重复，需人工审查后重跑（阈值已预注册）",
            report=report,
        )
    quota = registry.get("quota") or {"*": 1}
    selected = deterministically_select(records, quota, salt=args.salt)
    manifest = {
        "stage": "split",
        "salt": args.salt,
        "dedup": report,
        "selection": selection_manifest(selected, salt=args.salt),
        "registry_sha256": sha256_json(registry),
        "denylist_sha256": sha256_json(denylist_payload),
    }
    manifest, digest = _finalize(manifest)
    _write(args.out, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------- validate-env


def cmd_validate_env(args) -> int:
    manifest = _read_json(args.manifest, "--manifest")
    validate_face(FACE_ENV_LOCK, manifest)
    summary = {
        "stage": "validate-env",
        "env_id_sha256": sha256_json(manifest),
        "cpu": manifest.get("cpu"),
        "ram_bytes": manifest.get("ram_bytes"),
        "concurrency": 1,
        "note": "不是仅 pip freeze：os/arch、容器 digest、wheel hash、测试选集与 timeout 都必须齐备。",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------- verify


def cmd_verify(args) -> int:
    question_set = _read_json(args.question_set, "--question-set")
    questions = question_set.get("questions")
    if not questions:
        raise MissingInput("empty_question_set", "题目集为空")
    if not args.fresh:
        raise PolicyViolation("verify_requires_fresh", "verify 必须带 --fresh（干净 workspace）")
    if args.repeat < 2:
        raise PolicyViolation("verify_requires_two_runs", "每题需要至少两次复验，--repeat 2")

    runner = SubprocessRunner(timeout_seconds=question_set.get("timeout_seconds", 300))
    report = validate_question_set(questions, lambda question: runner)
    report["stage"] = "verify"
    report["repeat"] = args.repeat
    report["question_set_sha256"] = sha256_json(question_set)
    digest = _write(args.out, report)
    report["manifest_sha256"] = digest
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK if report["qualified"] else EXIT_BLOCKED


# ---------------------------------------------------------------- export


def cmd_export(args) -> int:
    payload = _read_json(args.records, "--records")
    records = payload.get("records")
    if not records:
        raise MissingInput("empty_records", "records 为空")
    template_lock = _read_json(args.template_lock, "--template-lock", required=False)
    if not template_lock.get("template_sha256"):
        raise UnverifiedLock(
            "template_lock_missing",
            "缺少模板锁定（template_sha256）：不得用未锁定模板导出训练视图",
        )
    # Q0 🔵-7：旧实现只要求 template_sha256 **非空**，既不与 canonical SHA 比对、
    # 也不要求 verified=true —— 等于"填个字符串就能导出"。现在两条都强制。
    if not template_lock.get("verified"):
        raise UnverifiedLock(
            "template_lock_unverified",
            "模板锁 verified != true：未核实的模板不得用于导出训练视图",
            template_lock=template_lock,
        )
    if template_lock["template_sha256"] != CANONICAL_TEMPLATE_SHA256:
        raise IntegrityError(
            "template_sha_mismatch",
            "模板锁里的 template_sha256 与 canonical 值不一致",
            expected=CANONICAL_TEMPLATE_SHA256,
            actual=template_lock["template_sha256"],
        )
    mask_config = _read_json(args.mask_config, "--mask-config", required=False)
    export = build_export(
        records,
        _count_tokens,
        window_tokens=int(args.window_tokens),
        export_config={"template_lock": template_lock, "mask_config": mask_config},
    )
    export["export_manifest"]["stage"] = "export"
    digest = _write(args.out, export["export_manifest"])
    export["export_manifest"]["manifest_sha256"] = digest
    print(json.dumps(export["export_manifest"], ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------- audit


def cmd_audit(args) -> int:
    release = _read_json(args.release, "--release")
    summary = validate_release(release)
    summary["stage"] = "audit"
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------- rollout


def cmd_rollout(args) -> int:
    """轨迹生成：本任务不具备模型与预算，如实阻断，不自动扩预算。"""
    if not args.train_only:
        raise PolicyViolation("rollout_train_only", "rollout 只允许对 train 生成（--train-only）")
    budget = _read_json(args.budget, "--budget")
    if budget.get("gpu_hours_released", 0) <= 0:
        raise PolicyViolation(
            "no_gpu_budget", "未被释放 GPU 预算：不自动扩大预算、不启动第二个 GPU 作业"
        )
    raise Blocked(
        "rollout_requires_teacher_model",
        "轨迹生成需要许可的教师模型与目标平台环境，交 107 运行统筹执行",
    )


# ---------------------------------------------------------------- exp1 / preflight


def cmd_exp1(args) -> int:
    corpus = load_corpus(args.corpus) if args.corpus else build_synthetic_corpus()
    report = run_exp1(corpus, output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


def cmd_train_preflight(args) -> int:
    from .train.entry import preflight

    report = preflight(config_path=args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK if not report["blockers"] else EXIT_BLOCKED


def cmd_deps(args) -> int:
    train = DependencyLock.from_file(args.train_lock).summary()
    serving = DependencyLock.from_file(args.serving_lock).summary()
    print(json.dumps({"train": train, "serving": serving}, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK if (train["verified"] and serving["verified"]) else EXIT_BLOCKED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="v3", description="V3 数据/训练流水线（fail-closed）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="校验来源锁（支持 D0 版本化 manifest）")
    p.add_argument("--source-lock", required=True)
    p.add_argument("--train-only", action="store_true", help="拒绝非 train 角色的来源")
    p.add_argument("--out")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("split", help="家族/时间去重与确定性选择")
    p.add_argument("--registry", required=True)
    p.add_argument("--denylist")
    p.add_argument("--salt", default="KAGGLE-21-V3-data-v1")
    p.add_argument("--allow-review", action="store_true", help="重复项已人工审查后放行")
    p.add_argument("--out")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("validate-env", help="校验 env.lock")
    p.add_argument("--manifest", required=True)
    p.set_defaults(func=cmd_validate_env)

    p = sub.add_parser("verify", help="broken/reference/P2P 对照复验")
    p.add_argument("--question-set", required=True)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("export", help="导出训练视图")
    p.add_argument("--records", required=True)
    p.add_argument("--template-lock", required=True)
    # Q0 🔵-7：设计稿 KAGGLE-21 §7 的 export 契约里有 --mask-config，旧实现漏了。
    p.add_argument("--mask-config", default=None, help="mask 配置的 JSON 路径（设计稿 §7 要求）")
    p.add_argument("--window-tokens", type=int, default=2048)
    p.add_argument("--out")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("audit", help="发布前审计")
    p.add_argument("--release", required=True)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("rollout", help="轨迹生成（需教师模型与预算）")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--budget", required=True)
    p.set_defaults(func=cmd_rollout)

    p = sub.add_parser("exp1", help="CPU EXP-1")
    p.add_argument("--corpus")
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_exp1)

    p = sub.add_parser("train-preflight", help="训练静态 preflight")
    p.add_argument("--config")
    p.set_defaults(func=cmd_train_preflight)

    p = sub.add_parser("deps", help="打印两侧依赖锁状态")
    p.add_argument("--train-lock", default=os.path.join(os.path.dirname(__file__), "locks", "train.lock.json"))
    p.add_argument("--serving-lock", default=os.path.join(os.path.dirname(__file__), "locks", "serving.lock.json"))
    p.set_defaults(func=cmd_deps)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except FailClosed as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
