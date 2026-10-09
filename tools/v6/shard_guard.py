#!/usr/bin/env python3
"""KAGGLE-32 v4 ② 无副作用守卫（纯 Python，DRAFT）

把 v3 里靠 bash `case` / `grep` 做的三类守卫**集中到一个纯 Python、无副作用**
的入口，原因有三：

1. **可离线证明**（Mika 第 1 项）：本机没有任何 POSIX shell，bash 里的预算算术
   无法被 fixture 测试。挪到 Python 后可以真跑断言。
2. **严格类型校验**（Mika 第 2 项）：bash 的字符白名单会放过 `.`、`1..2`；
   Python 用 `float()` + `math.isfinite()` 能真正拒绝。
3. **无副作用地先校验路径**（Mika 第 3 项）：v3 先 `mkdir`/开日志、再判
   OUT_ROOT 是否在仓库内 —— 错误配置已经污染仓库才退出。本脚本**只读**，
   不创建任何文件，必须在任何写入之前调用。

**它不写任何文件。** 成功时把一份"计划" JSON 打到 stdout，调用方据此设置
`main_timeout` 等；失败时打印结构化原因到 stdout 并以非零退出。

退出码
------
0   通过
40  参数/路径非法（绝对路径、仓库外、stage 白名单）
41  预算非法（非数字 / 非有限 / <= 0）
42  预算超出授权（总预算 > GPU-h 余额；总预算 > 3h45m 停止线；> 平台上限）
43  授权证据非法（解析失败 / 缺字段 / 空值 / 类型错 / 哈希不符 / 目标 SHA 不符 / 范围不符）
44  授权来源未被外部锚定（缺 `--approval-expected-sha256`）

关于"批准来源如何核实"（Mika 第 2 项）
-------------------------------------
本脚本**不能**用证据文件里的 `verified_by` 自证批准真实性 —— 那是自报字段，
任何人都能填。可信链必须是**外部锚**：

    Mika 在评论里公布该批准文件的 SHA256  →  提交方把它抄进
    `--approval-expected-sha256`  →  本脚本校验
    `sha256(实际文件) == 期望值`

没有外部锚（这个参数）就一律拒绝（exit 44），**不认为"文件存在"或"字段齐全"
构成批准**。这条是本脚本对 Mika 要求的直接落实。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

# ---------------------------------------------------------------- 硬常量

#: 107 `qos_stu_default` 单作业上限（秒）
JOB_CAP_SECONDS = 4 * 3600
#: KAGGLE-28 规定的"停止新工作"线：3h45m
STOP_LINE_SECONDS = 3 * 3600 + 45 * 60
#: 主命令被 TERM 后到 KILL 的宽限
KILL_AFTER_SECONDS = 30
#: 准备阶段硬上限（Mika：准备也必须被整体截止覆盖，不能只扣预留却无上限）
PREP_CAP_SECONDS = 120
#: 哈希与收尾硬上限
WRAPUP_CAP_SECONDS = 180
#: 主命令至少要有这么久，否则不值得启动
MIN_MAIN_SECONDS = 60
#: 固定版本
PIN_COMMIT = "128d9b98b05ddf128c2e77b599e078de65675b8a"
#: stage 白名单（防路径穿越；`..` 与 `/` 一律不在字符集里）
STAGE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

#: 每个模式**只**允许它支持的 stage（Mika 第 3 项：限制 stage 为本模式支持的值）。
#: 这也堵住了"拿 CPU 入口的 stage 去跑 GPU 预算"之类的错配。
STAGE_ALLOWED_BY_MODE = {
    "gpu": ("start", "T1"),
    "cpu": ("preflight", "smoke"),
}

APPROVAL_REQUIRED_FIELDS = (
    "approval_id",
    "target_sha",
    "gpu_hours_approved",
    "verified_by",
    "verified_utc",
    "evidence_sha256",
)


def emit(status: str, code: int, message: str, **extra) -> int:
    payload = {"schema": "v3-shard-guard/1", "status": status, "code": code, "message": message}
    payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return code


# ---------------------------------------------------------------- 严格解析


def strict_positive_number(text, name: str):
    """严格解析**有限正数**。拒绝 `.`、`1..2`、`nan`、`inf`、空、带空白、布尔样式。"""
    if text is None:
        raise ValueError("%s 缺失" % name)
    raw = str(text)
    if raw != raw.strip() or raw == "":
        raise ValueError("%s 含空白或为空：%r" % (name, raw))
    if raw.startswith("+") or raw.lower().startswith(("nan", "inf", "-inf")):
        raise ValueError("%s 不是普通有限正数：%r" % (name, raw))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError("%s 不是数字：%r" % (name, raw))
    if not math.isfinite(value):
        raise ValueError("%s 不是有限数：%r" % (name, raw))
    if value <= 0:
        raise ValueError("%s 必须 > 0：%r" % (name, raw))
    return value


def strict_positive_int(text, name: str):
    value = strict_positive_number(text, name)
    if value != int(value):
        raise ValueError("%s 必须是整数秒：%r" % (name, text))
    return int(value)


def check_stage(stage: str, mode: str = None):
    if not STAGE_RE.match(stage or ""):
        raise ValueError(
            "stage 非法（只允许 [A-Za-z0-9_.-]，1-32 字符）：%r —— 该值会被拼进输出路径" % (stage,)
        )
    if ".." in stage:
        raise ValueError("stage 含 '..'，拒绝：%r" % (stage,))
    if mode is not None:
        allowed = STAGE_ALLOWED_BY_MODE.get(mode, ())
        if stage not in allowed:
            raise ValueError(
                "stage %r 不属于 %s 模式支持的值 %s" % (stage, mode, list(allowed))
            )


def check_abs_outside_repo(raw_path: str, name: str, repo_real: str):
    if not raw_path:
        raise ValueError("%s 为空" % name)
    if not os.path.isabs(raw_path):
        raise ValueError("%s 必须是绝对路径：%r" % (name, raw_path))
    # normpath 不需要目录存在；realpath 会解析已存在的部分
    normalized = os.path.normpath(raw_path)
    if ".." in normalized.split(os.sep):
        raise ValueError("%s 含 '..'，拒绝：%r" % (name, raw_path))
    candidate = os.path.realpath(normalized)
    repo_prefix = repo_real.rstrip(os.sep) + os.sep
    if candidate == repo_real or candidate.startswith(repo_prefix):
        raise ValueError("%s 落在仓库内（%s）：拒绝，避免污染工作树" % (name, candidate))
    return candidate


def canonical_not_in_repo(raw_path: str, name: str, repo_real: str):
    """**仅词法**校验（normpath），不做 realpath。

    ⚠️ **不要用它做写入前的路径校验。** Mika 第 3 项指出：词法检查识破不了
    符号链接 / junction —— 一个词法上在仓库外的路径，若其某个**已存在**的父级
    是指向仓库的链接，`mkdir` 仍会写进仓库。`os.path.realpath` 能解析
    "不存在的路径里已存在的那部分父级"，所以"目录尚不存在"**不是**省略它的理由。

    保留本函数只作对照与测试演示；生产路径校验一律走 `check_abs_outside_repo`。
    """
    if not raw_path or not os.path.isabs(raw_path):
        raise ValueError("%s 必须是绝对路径：%r" % (name, raw_path))
    normalized = os.path.normpath(raw_path)
    if ".." in normalized.split(os.sep):
        raise ValueError("%s 含 '..'，拒绝：%r" % (name, raw_path))
    repo_prefix = repo_real.rstrip(os.sep) + os.sep
    if normalized == repo_real or normalized.startswith(repo_prefix):
        raise ValueError("%s 落在仓库内（%s）：拒绝" % (name, normalized))
    return normalized


# ---------------------------------------------------------------- 授权证据


def load_approval(path: str):
    """严格 JSON 解析。**不做字符串搜索**（v3 用 grep 找字段名，空值也能通过）。"""
    if not os.path.isfile(path):
        raise ValueError("批准证据文件不存在：%s" % path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError("批准证据不是合法 JSON：%s" % exc)
    except OSError as exc:
        raise ValueError("批准证据不可读：%s" % exc)
    if not isinstance(payload, dict):
        raise ValueError("批准证据根节点必须是 JSON 对象，实际是 %s" % type(payload).__name__)
    return payload


def validate_approval(payload: dict, *, pin_commit: str, gpu_hours: float,
                      committed_sha256: str, actual_file_sha256: str,
                      evidence_ref_sha256=None):
    """逐字段严格校验。返回校验通过后的批准摘要。

    关于 `evidence_sha256` 的**语义**（v4 修正）：
    它指的是**被批准的那个证据对象**的哈希，**不是**批准文件自身的哈希。
    批准文件自身的完整性由 `--approval-expected-sha256`（外部锚）保证。
    v4 初版把两者混为一谈，导致一个自指哈希永远不可能自洽 —— 已修正。
    所以：声明了 `evidence_sha256` 就必须同时给出 `--approval-evidence-ref`
    指向那个证据对象，脚本会实算它并与声明比对。
    """
    problems = []

    for field in APPROVAL_REQUIRED_FIELDS:
        if field not in payload:
            problems.append("缺字段 %s" % field)

    def nonempty_str(field):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append("%s 必须是非空字符串，实际 %r" % (field, value))
            return None
        return value.strip()

    approval_id = nonempty_str("approval_id")
    verified_by = nonempty_str("verified_by")
    verified_utc = nonempty_str("verified_utc")
    target_sha = nonempty_str("target_sha")
    declared_evidence_sha = nonempty_str("evidence_sha256")

    # 目标 SHA 必须就是本次固定版本
    if target_sha is not None and target_sha != pin_commit:
        problems.append("target_sha=%s 与本次固定版本 %s 不符" % (target_sha, pin_commit))

    # evidence_sha256：64 位小写 hex，且必须与**被批准证据对象**的实算哈希一致
    if declared_evidence_sha is not None:
        if not HEX64_RE.match(declared_evidence_sha):
            problems.append("evidence_sha256 不是 64 位小写 hex：%r" % declared_evidence_sha)
        elif evidence_ref_sha256 is None:
            problems.append(
                "声明了 evidence_sha256 却没有给出 --approval-evidence-ref："
                "无法实测被批准证据对象的哈希"
            )
        elif declared_evidence_sha != evidence_ref_sha256:
            problems.append(
                "evidence_sha256 与被批准证据对象实际哈希不符：声明 %s / 实际 %s"
                % (declared_evidence_sha, evidence_ref_sha256)
            )

    # gpu_hours_approved：有限正数（不是"看起来像数字"）
    approved_hours = None
    try:
        approved_hours = strict_positive_number(payload.get("gpu_hours_approved"), "gpu_hours_approved")
    except ValueError as exc:
        problems.append(str(exc))

    if approved_hours is not None and gpu_hours > approved_hours:
        problems.append(
            "本次 GPU-h=%.6g 超出批准范围 %.6g" % (gpu_hours, approved_hours)
        )

    # 外部锚：本次调用的期望哈希必须与**批准文件**的实际哈希一致
    if not HEX64_RE.match(committed_sha256 or ""):
        problems.append("--approval-expected-sha256 不是 64 位小写 hex：%r" % committed_sha256)
    elif committed_sha256 != actual_file_sha256:
        problems.append(
            "批准文件哈希与外部锚不符：锚 %s / 实际 %s（外部锚应来自 Mika 评论里公布的哈希）"
            % (committed_sha256, actual_file_sha256)
        )

    if problems:
        raise ValueError("; ".join(problems))

    return {
        "approval_id": approval_id,
        "verified_by": verified_by,
        "verified_utc": verified_utc,
        "target_sha": target_sha,
        "gpu_hours_approved": approved_hours,
        "evidence_sha256": declared_evidence_sha,
        "evidence_ref_sha256": evidence_ref_sha256,
        "approval_file_sha256": actual_file_sha256,
        "external_anchor_sha256": committed_sha256,
    }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- 预算规划


def plan_budgets(*, total_budget, gpu_hours, already_elapsed=0, mode="gpu"):
    """统一约束：批准余额 ∩ 阶段上限 ∩ 3h45m 停止线 ∩ TERM/KILL/收尾预留。

    返回 `(main_timeout, limits_dict)` 或抛 ValueError。
    """
    limits = {
        "total_budget_seconds": total_budget,
        "gpu_hours_seconds": int(gpu_hours * 3600) if gpu_hours is not None else None,
        "stage_cap_seconds": JOB_CAP_SECONDS,
        "stop_line_seconds": STOP_LINE_SECONDS,
        "kill_after_seconds": KILL_AFTER_SECONDS,
        "prep_cap_seconds": PREP_CAP_SECONDS,
        "wrapup_cap_seconds": WRAPUP_CAP_SECONDS,
        "already_elapsed_seconds": already_elapsed,
    }

    # ① 总预算不得越过 3h45m 停止线与平台上限
    if total_budget > STOP_LINE_SECONDS:
        raise ValueError(
            "总预算 %ds 超过 3h45m 停止线 %ds" % (total_budget, STOP_LINE_SECONDS)
        )
    if total_budget > JOB_CAP_SECONDS:
        raise ValueError("总预算 %ds 超过平台单作业上限 %ds" % (total_budget, JOB_CAP_SECONDS))

    # ② 总预算不得越过授权余额（Mika 的反例：14400s vs 0.1 GPU-h=360s）
    if gpu_hours is not None:
        authorized = gpu_hours * 3600
        if total_budget > authorized:
            raise ValueError(
                "总预算 %ds 超过授权余额 %ds（= %.6g GPU-h × 3600）；"
                "预算不能超过已批准的 GPU 小时" % (total_budget, int(authorized), gpu_hours)
            )

    # ③ 准备 / TERM / 收尾都必须被整体截止覆盖，且有各自硬上限
    fixed = PREP_CAP_SECONDS + KILL_AFTER_SECONDS + WRAPUP_CAP_SECONDS
    if total_budget < fixed + MIN_MAIN_SECONDS:
        raise ValueError(
            "总预算 %ds 不足以覆盖 准备%d + kill-after%d + 收尾%d + 主命令至少%d"
            % (total_budget, PREP_CAP_SECONDS, KILL_AFTER_SECONDS, WRAPUP_CAP_SECONDS, MIN_MAIN_SECONDS)
        )

    # ④ 主命令上限 = min(所有上限) − 已用 − 固定预留
    ceilings = [
        total_budget,
        STOP_LINE_SECONDS,
        JOB_CAP_SECONDS,
    ]
    if gpu_hours is not None:
        ceilings.append(int(gpu_hours * 3600))
    main_timeout = min(ceilings) - already_elapsed - KILL_AFTER_SECONDS - WRAPUP_CAP_SECONDS
    limits["binding_ceiling_seconds"] = min(ceilings)

    if main_timeout < MIN_MAIN_SECONDS:
        raise ValueError(
            "主命令可用时间 %ds 低于下限 %ds（已用 %ds，固定预留 %ds）"
            % (main_timeout, MIN_MAIN_SECONDS, already_elapsed, KILL_AFTER_SECONDS + WRAPUP_CAP_SECONDS)
        )
    return main_timeout, limits


# ---------------------------------------------------------------- CLI


def build_parser():
    parser = argparse.ArgumentParser(description="KAGGLE-32 v4 无副作用守卫（只读，不写文件）")
    parser.add_argument("--mode", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--ledger-dir", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--total-budget-seconds", required=True)
    parser.add_argument("--gpu-hours", default=None, help="本次申请的 GPU 小时（start 必填）")
    parser.add_argument("--already-elapsed", type=int, default=0)
    parser.add_argument("--approval-evidence", default=None)
    parser.add_argument("--approval-expected-sha256", default=None,
                        help="外部锚：Mika 评论里公布的**批准文件** SHA256")
    parser.add_argument("--approval-evidence-ref", default=None,
                        help="被批准的那个证据对象；其哈希须等于批准文件里的 evidence_sha256")
    parser.add_argument("--pin-commit", default=PIN_COMMIT)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # ---- 纯只读：先校验、后写入（调用方在写入前调用本脚本）----
    if not os.path.isdir(args.repo):
        return emit("rejected", 40, "仓库目录不存在：%s" % args.repo)
    repo_real = os.path.realpath(args.repo)

    try:
        check_stage(args.stage, args.mode)
        # 用 realpath 解析**实际**路径。`os.path.realpath` 也会解析"不存在的路径里
        # 已存在的那部分父级"，所以"目录尚不存在"**不是**省略它的理由
        # （v4 初版错用了只做词法 normpath 的 canonical_not_in_repo，
        #  符号链接/junction 指向仓库内时可被绕过）。
        # 返回的是**校验后的规范路径**，调用方必须用它去建目录，而不是用原始入参。
        out_root = check_abs_outside_repo(args.out_root, "V3_OUT_ROOT", repo_real)
        ledger_dir = check_abs_outside_repo(args.ledger_dir, "V3_LEDGER_DIR", repo_real)
    except ValueError as exc:
        return emit("rejected", 40, str(exc))

    try:
        total_budget = strict_positive_int(args.total_budget_seconds, "V3_TOTAL_BUDGET_SECONDS")
    except ValueError as exc:
        return emit("rejected", 41, str(exc))

    gpu_hours = None
    if args.mode == "gpu":
        if args.gpu_hours is None:
            return emit("rejected", 41, "GPU 模式必须给出 --gpu-hours")
        try:
            gpu_hours = strict_positive_number(args.gpu_hours, "V3_GPU_HOURS")
        except ValueError as exc:
            return emit("rejected", 41, str(exc))

    # ---- 授权证据 ----
    # Mika 第 1 项：v4 初版把整套授权检查写在 `if args.approval_evidence:` 里，
    # 于是**省略全部 approval 参数**就能 exit 0（实测主命令预算 3390s 被直接放行）。
    # GPU 模式必须强制要求三项齐全，**不能**靠下游（E0 缺 --dest-dir）的独立阻断兜底。
    if args.mode == "gpu":
        missing = [
            name for name, value in (
                ("--approval-evidence", args.approval_evidence),
                ("--approval-expected-sha256", args.approval_expected_sha256),
                ("--approval-evidence-ref", args.approval_evidence_ref),
            ) if not value
        ]
        if missing:
            return emit(
                "rejected", 44,
                "GPU 模式必须同时提供批准文件、外部锚与证据对象；缺 %s。"
                "没有已核批准不得进入 GPU 预算规划。" % ", ".join(missing),
            )

    approval = None
    if args.approval_evidence:
        if not args.approval_expected_sha256:
            return emit(
                "rejected", 44,
                "给了 --approval-evidence 却没有 --approval-expected-sha256："
                "批准来源必须由**外部锚**核实（Mika 评论里公布的哈希），"
                "不能靠文件自报的 verified_by 自证",
            )
        try:
            actual_sha = sha256_file(args.approval_evidence)
        except OSError as exc:
            return emit("rejected", 43, "批准证据不可读：%s" % exc)
        evidence_ref_sha = None
        if args.approval_evidence_ref:
            try:
                evidence_ref_sha = sha256_file(args.approval_evidence_ref)
            except OSError as exc:
                return emit("rejected", 43, "被批准证据对象不可读：%s" % exc)
        try:
            payload = load_approval(args.approval_evidence)
            approval = validate_approval(
                payload,
                pin_commit=args.pin_commit,
                gpu_hours=gpu_hours if gpu_hours is not None else 0.0,
                committed_sha256=args.approval_expected_sha256,
                actual_file_sha256=actual_sha,
                evidence_ref_sha256=evidence_ref_sha,
            )
        except ValueError as exc:
            return emit("rejected", 43, str(exc))

    # ---- 预算规划 ----
    try:
        main_timeout, limits = plan_budgets(
            total_budget=total_budget,
            gpu_hours=gpu_hours,
            already_elapsed=max(0, args.already_elapsed),
            mode=args.mode,
        )
    except ValueError as exc:
        return emit("rejected", 42, str(exc))

    # GPU 模式的成功结果**必须**带已核 approval；调用方（sbatch）也应再查一次。
    if args.mode == "gpu" and not (approval or {}).get("approval_id"):
        return emit("rejected", 44, "内部不一致：GPU 模式必须产出已核 approval，实际为空")

    return emit(
        "ok", 0, "guards passed",
        mode=args.mode,
        stage=args.stage,
        out_root=out_root,
        ledger_dir=ledger_dir,
        repo_real=repo_real,
        main_timeout_seconds=main_timeout,
        # 供 shard_supervisor.py 使用的整体预算：同一个 min(上限)，
        # 由监督器自己按 prep/main/wrapup 分配（prep 与 wrapup 也受其控制）。
        supervisor_total_budget_seconds=limits["binding_ceiling_seconds"],
        supervisor_prep_cap_seconds=PREP_CAP_SECONDS,
        supervisor_wrapup_cap_seconds=WRAPUP_CAP_SECONDS,
        supervisor_kill_after_seconds=KILL_AFTER_SECONDS,
        limits=limits,
        approval=approval,
        approval_required=(args.mode == "gpu"),
        approval_verified=bool((approval or {}).get("approval_id")),
        note=(
            "本脚本只读、不创建任何文件；调用方必须在本脚本返回 0 之后才做 mkdir/日志重定向，"
            "并且**必须使用本输出里的 out_root / ledger_dir 这两个已校验的规范路径**"
            "（它们经过 os.path.realpath，能识破经符号链接/junction 指向仓库内的路径）。"
            "已验证会在写入前非零退出的情况：路径落在仓库内（含经链接）、stage 非法或不属于本模式、"
            "缺批准、预算超授权、批准哈希不符。"
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
