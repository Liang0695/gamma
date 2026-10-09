"""真实模式的**批准绑定**：复用 KAGGLE-32 v6 `shard_guard.py` 的批准契约。

独立审查（KAGGLE-38 评论 01a11aa9…，阻断 2）实测：五个哈希全填 `"1"` 被接受，
`--local-load-authorized` 只是一个调用方自己按的布尔开关——没有任何东西核对
「谁批准的、批准了什么、批准是否还成立」。也就是说真实模式的入口门槛是**自报**的。

Mika 的整改要求（KAGGLE-38 顶部第 2 项）：**复用 32 v6 批准/guard/supervisor 契约，
禁止另造同名授权框架。** 本模块因此**不是**新框架，而是 v6 契约在训练入口上的
**同一个契约**：字段名、正则、外部锚规则逐条对齐

    KAGGLE-32 v6 包 `shard_guard.py`  sha256 f459321fde54e0fa660eba7d63d3f8a1fc031afacffe4fc58fc7512f4bb46845
    （APPROVAL_REQUIRED_FIELDS / HEX64_RE / strict_positive_number / 外部锚校验）

核心口径与 v6 完全一致，逐字照搬：

1. **字段齐全不算批准**。`load_approval()` 严格解析 JSON，不做字符串搜索
   （旧做法用 grep 找字段名，空值也能过）。
2. **必须有外部锚**。批准文件自身的哈希要由**外部**公布（Mika 在评论里给），
   调用方把锚抄进 `--approval-expected-sha256`，这里实算文件哈希与锚比对。
   **没有外部锚就一律拒绝**——`verified_by` 是自报字段，谁都能填，不构成信任。
3. **目标版本必须就是获批版本**：读取实际干净 Git HEAD，重算代码、计划（含 labels）、
   输入、锁文件与运行选项；整体必须等于批准记录的 runtime_context。
4. **批准额度是有限正数**，本次用量不得超过。

本模块只做**校验**，不写任何东西、不读受保护输入；被拒绝时不返回部分结果。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from ..common.canonical import sha256_json

from ..common.errors import IntegrityError, MissingInput, PolicyViolation

# Historical fixture identifier only; never used to authorize a new runtime.
LEGACY_E0_FIXTURE_COMMIT = "128d9b98b05ddf128c2e77b599e078de65675b8a"

#: 与 v6 `shard_guard.py::HEX64_RE` 同口径：只接受 64 位**小写** hex。
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

#: 与 v6 `shard_guard.py::APPROVAL_REQUIRED_FIELDS` 逐字段同序。
APPROVAL_REQUIRED_FIELDS = (
    "approval_id",
    "target_sha",
    "gpu_hours_approved",
    "verified_by",
    "verified_utc",
    "evidence_sha256",
)

#: v6 `shard_guard.py` 的 SHA-256：本模块声明的复用对象，供复核者核对口径来源。
V6_SHARD_GUARD_SHA256 = (
    "f459321fde54e0fa660eba7d63d3f8a1fc031afacffe4fc58fc7512f4bb46845"
)


def strict_positive_number(text, name: str) -> float:
    """严格解析**有限正数**（与 v6 同名函数同行为）。

    拒绝 `.`、`1..2`、`nan`、`inf`、空、带空白、`+1` 这些「看起来像数字」的输入。
    """
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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_approval(path: str) -> dict:
    """严格 JSON 解析批准记录。不做字符串搜索。"""
    if not path:
        raise MissingInput(
            "approval_record_missing",
            "真实模式必须给出 v6 批准记录（--approval）：自报哈希与布尔开关都不能替代批准",
        )
    if not os.path.isfile(path):
        raise MissingInput(
            "approval_record_missing", "批准记录文件不存在：%s" % path, path=path
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise IntegrityError(
            "approval_record_invalid_json", "批准记录不是合法 JSON：%s" % exc, path=path
        )
    except OSError as exc:
        raise MissingInput(
            "approval_record_unreadable", "批准记录不可读：%s" % exc, path=path
        )
    if not isinstance(payload, dict):
        raise IntegrityError(
            "approval_record_not_object",
            "批准记录根节点必须是 JSON 对象，实际是 %s" % type(payload).__name__,
            path=path,
        )
    return payload


def validate_approval(
    payload: dict,
    *,
    committed_sha256: str,
    actual_file_sha256: str,
    pin_commit: str,
    requested_gpu_hours: float | None = None,
    evidence_ref_sha256: str | None = None,
) -> dict:
    """逐字段严格校验批准记录，返回批准摘要；任何问题都 fail-closed。

    与 v6 `shard_guard.validate_approval` 同口径，差别只在错误类型走本仓库的
    `PolicyViolation` / `IntegrityError` 分层（v6 是独立脚本，用 `ValueError`）。
    """
    problems: list[str] = []

    for field in APPROVAL_REQUIRED_FIELDS:
        if field not in payload:
            problems.append("缺字段 %s" % field)

    def nonempty_str(field):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append("%s 必须是非空字符串，实际 %r" % (field, value))
            return None
        return value.strip()

    nonempty_str("approval_id")
    nonempty_str("verified_by")
    nonempty_str("verified_utc")
    target_sha = nonempty_str("target_sha")
    declared_evidence_sha = nonempty_str("evidence_sha256")

    if target_sha is not None and target_sha != pin_commit:
        problems.append(
            "target_sha=%s 与本轮固定版本 %s 不符" % (target_sha, pin_commit)
        )

    if declared_evidence_sha is not None:
        if not HEX64_RE.match(declared_evidence_sha):
            problems.append("evidence_sha256 不是 64 位小写 hex：%r" % declared_evidence_sha)
        elif evidence_ref_sha256 is None:
            problems.append(
                "声明了 evidence_sha256 却没有给出被批准证据对象：无法实测其哈希"
            )
        elif declared_evidence_sha != evidence_ref_sha256:
            problems.append(
                "evidence_sha256 与被批准证据对象实际哈希不符：声明 %s / 实际 %s"
                % (declared_evidence_sha, evidence_ref_sha256)
            )

    approved_hours = None
    try:
        approved_hours = strict_positive_number(
            payload.get("gpu_hours_approved"), "gpu_hours_approved"
        )
    except ValueError as exc:
        problems.append(str(exc))

    if (
        approved_hours is not None
        and requested_gpu_hours is not None
        and float(requested_gpu_hours) > approved_hours
    ):
        problems.append(
            "本次 GPU-h=%.6g 超出批准范围 %.6g" % (requested_gpu_hours, approved_hours)
        )

    # ---- 外部锚（v6 的核心）：没有锚就不是批准 ----
    if not HEX64_RE.match(committed_sha256 or ""):
        problems.append(
            "缺少有效的外部锚 --approval-expected-sha256（应为 64 位小写 hex）：%r"
            % (committed_sha256,)
        )
    elif committed_sha256 != actual_file_sha256:
        problems.append(
            "批准记录哈希与外部锚不符：锚 %s / 实际 %s（锚应来自外部公布的哈希）"
            % (committed_sha256, actual_file_sha256)
        )

    if problems:
        raise PolicyViolation(
            "approval_not_trusted",
            "v6 批准记录未通过校验，真实模式拒绝启动：%s" % "；".join(problems),
            problems=problems,
            pin_commit=pin_commit,
            external_anchor_present=bool(HEX64_RE.match(committed_sha256 or "")),
            contract_source="KAGGLE-32 v6 shard_guard.py sha256=%s" % V6_SHARD_GUARD_SHA256,
        )

    return {
        "approval_id": payload["approval_id"],
        "target_sha": target_sha,
        "gpu_hours_approved": approved_hours,
        "verified_by": payload["verified_by"],
        "verified_utc": payload["verified_utc"],
        "evidence_sha256": declared_evidence_sha,
        "evidence_ref_sha256": evidence_ref_sha256,
        "approval_file_sha256": actual_file_sha256,
        "external_anchor_sha256": committed_sha256,
        "contract_source": "KAGGLE-32 v6 shard_guard.py sha256=%s"
        % V6_SHARD_GUARD_SHA256,
    }


def require_approval(
    *,
    approval_path: str | None,
    expected_sha256: str | None,
    requested_gpu_hours: float | None = None,
    evidence_ref_path: str | None = None,
    runtime_context: dict | None = None,
    published_anchor: str | None = None,
) -> dict:
    """真实模式入口的唯一批准闸门：读记录 → 逐字段校验 → 返回摘要。

    任一环节不成立即抛 fail-closed 异常；**不返回部分结果**，也不落任何文件。
    """
    if not published_anchor or published_anchor != expected_sha256 or runtime_context is None:
        raise PolicyViolation("approval_not_trusted",
                              "缺少部署方发布的批准锚或实际运行对象；调用方自算锚不构成批准")
    payload = load_approval(approval_path or "")
    actual = sha256_file(approval_path)
    if expected_sha256 is not None and not HEX64_RE.match(str(expected_sha256)):
        raise PolicyViolation(
            "approval_expected_sha256_malformed",
            "外部锚不是 64 位小写 hex：%r" % (expected_sha256,),
            external_anchor=str(expected_sha256),
        )
    evidence_ref_sha256 = None
    if evidence_ref_path:
        if not os.path.isfile(evidence_ref_path):
            raise MissingInput(
                "approval_evidence_ref_missing",
                "被批准证据对象不存在：%s" % evidence_ref_path,
                path=evidence_ref_path,
            )
        evidence_ref_sha256 = sha256_file(evidence_ref_path)
    result = validate_approval(
        payload,
        committed_sha256=str(expected_sha256 or ""),
        actual_file_sha256=actual,
        pin_commit=runtime_context["target_sha"],
        requested_gpu_hours=requested_gpu_hours,
        evidence_ref_sha256=evidence_ref_sha256,
    )
    if payload.get("runtime_context") != runtime_context:
        raise PolicyViolation("approval_runtime_mismatch", "实际代码/计划/输入/依赖/路径/预算与批准不符")
    result["runtime_context"] = runtime_context
    return result


def deployment_approval_anchor():
    """Only the fixed operator-installed publication can supply the anchor."""
    from .deployment import read_publication
    return read_publication()['anchor']


def runtime_options(*, steps, checkpoint_every, stop_at_step, budget_seconds,
                    requested_gpu_hours, model_id, model_revision, target_modules,
                    device="cuda", adapter_name="v3_policy"):
    return {"steps": int(steps), "checkpoint_every": int(checkpoint_every),
            "stop_at_step": stop_at_step, "budget_seconds": float(budget_seconds),
            "requested_gpu_hours": requested_gpu_hours, "model_id": model_id,
            "model_revision": model_revision, "target_modules": list(target_modules or []),
            "device": device, "adapter_name": adapter_name}


def measure_runtime_context(*, plan, input_binding, local_model_dir, options,
                            repo_root=None, environment_policy=None):
    """Read actual consumer objects. No caller-supplied hashes are trusted here."""
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    try:
        target = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", target):
            raise ValueError("invalid Git revision")
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                    "--untracked-files=all"], text=True).strip():
            raise ValueError("dirty deployment tree")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise PolicyViolation("runtime_code_unbound", "生产运行需干净且已入库的实际 Git 版本") from exc
    code = {}
    for directory in ("v3", "tools"):
        for path in sorted((root / directory).rglob("*.py")):
            code[path.relative_to(root).as_posix()] = sha256_file(str(path))
    deps = {name: sha256_file(str(root / "v3" / "locks" / name))
            for name in ("train.lock.json", "official-interface.json")}
    from .runtime_environment import verify_deployed_environment, probe_environment
    manifest = root / 'docs/v3/evidence/kaggle-38-train-stack-manifest.json'
    # Read-only prepublication measurement may supply a policy. Production
    # parent/child never accept this argument from CLI and re-read publication.
    environment = (verify_deployed_environment() if environment_policy is None else
                   probe_environment(environment_policy, manifest))
    deps['train_stack_manifest'] = sha256_file(str(manifest))
    inputs = {}
    for name, item in sorted(input_binding["consumed_inputs"].items()):
        actual = sha256_file(item["path"])
        if actual != item["sha256"]:
            raise PolicyViolation("runtime_input_changed", "已绑定输入被更改")
        inputs[name] = {"path": os.path.realpath(item["path"]), "sha256": actual}
    plan_data = asdict(plan)  # includes labels; TrainBatch.to_dict() omits them
    context = {
        "target_sha": target, "code_files": code, "dependency_files": deps,
        "runtime_environment": environment,
        "model": {"repo_id": input_binding["repo_id"], "revision": input_binding["revision"],
                  "local_model_dir": os.path.realpath(local_model_dir), "inputs": inputs},
        "plan": plan_data, "options": dict(options),
    }
    context["hashes"] = {
        "source_sha256": sha256_json(context["model"]),
        "data_sha256": sha256_json(plan_data["batches"]),
        "config_sha256": sha256_json({"plan": plan_data, "options": dict(options)}),
        "code_sha256": sha256_json(code),
        "deps_sha256": sha256_json({'files':deps, 'environment':environment}),
    }
    return context
