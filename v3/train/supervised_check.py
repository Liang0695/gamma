"""KAGGLE-38 r5：v6 监督器**适配层**（独审阻断 6 的落点）。

## 为什么需要它

独审阻断 6 的原文：

    外层监督截止仍只靠可选的 CLI/环境变量自报，未见与 v6 监督器的可信接线。
    不传 --deadline-epoch/V3_DEADLINE_EPOCH，真实模式仍只按调用方
    --time-budget-seconds 运行；BudgetLedger 在阶段开始和结束检查，
    不能中断卡住的加载或保存。

也就是说：**账目字段不等于硬截止**。真正的硬截止必须是"外层进程能在子进程卡住时把它杀掉"
的能力，而不是子进程自己在报告里写一句"我按截止停了"。

## 本模块做什么（r4 报告 §二 列出的三条，逐条对应）

1. **把 `run_engineering_check` 作为子进程交给 v6 监督器**，由外层传入并强制
   `execution_end = min(批准预算, 外层截止)`。外层在这里只做编排，**不另造框架**：
   直接复用 KAGGLE-32 v6 的 `shard_supervisor.py` / `shard_job.py` / `shard_guard.py`
   （按 SHA-256 固定，改一个字节就拒绝）。
2. **剥掉可由子进程扩时的入口**：启动子进程前从继承环境里删掉
   `V3_DEADLINE_EPOCH` / `V3_TOTAL_BUDGET_SECONDS` / `V3_TIME_BUDGET_SECONDS`
   等一切 `V3_*` 的 DEADLINE/BUDGET 键（逐个记进证据），并把授权截止写成
   **父进程签发的文件**（带 SHA-256 自锚），子进程只能读、不能改；子进程若拿到
   更晚的 CLI/环境截止，一律 `deadline_widening_rejected`。
3. **CPU 正反例**（`tests/test_g38_supervised_check.py`）：卡住的子进程被 TERM/KILL、
   批准上下文被篡改被拒、截止缺失/被放宽被拒。全部不占 GPU、不加载模型。

## 复用对象（SHA-256 固定，逐字节核对）

```
shard_supervisor.py  a3f96572a14cf98965c739d85bd5ecee318ace912fd0339453036a877e6420e5
shard_job.py         63ee2531757e3a34d70660ca257171dcd3b23f893cbddd8e0137c68215f9cd33
shard_guard.py       f459321fde54e0fa660eba7d63d3f8a1fc031afacffe4fc58fc7512f4bb46845
stage_scripts.json   7c1a7b6009ee2e82f61820060d53a92fc91cbf1e33e2f6013c63eb85e9e1bfd2
deadline_wrapper.py  5c515706ad2e1f52bafc6e7d10c1b39343c56a908f72b09940b8355139726f2c
```

出处：KAGGLE-32 v6 交付包（附件 `01a118ad-75ee-7f37-9363-c8dd4bdbe9bb`，
整包 SHA-256 `0151b35ad92c36985d1632d37b9b06e9513c64e3afd4c4c8d26f889dfd1676c9`）。

## 诚实边界

- 本机（Windows）实测的是 **`--mode plan`** 的 CPU 编排；**真实 GPU 档（`--mode gpu`）**
  在本机**未执行**（没有批准记录、没有训练栈）。两条档位共用同一段编排代码，
  差别只在预算来源（批准的 GPU-h vs 调用方给的 CPU 预算）。
- v6 自述：POSIX 进程组回收**未在 Windows 上验证**；本模块原样转述，不夸大为已验证。
- 本模块**不**放宽任何训练闸门，也不产生训练资格。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Mapping

from ..common.canonical import sha256_json
from ..common.errors import FailClosed, IntegrityError, MissingInput, PolicyViolation

#: 必须存在且逐字节核对的 v6 文件（名 → SHA-256）。
V6_REQUIRED_FILES = {
    "shard_supervisor.py": "a3f96572a14cf98965c739d85bd5ecee318ace912fd0339453036a877e6420e5",
    "shard_job.py": "63ee2531757e3a34d70660ca257171dcd3b23f893cbddd8e0137c68215f9cd33",
    "shard_guard.py": "f459321fde54e0fa660eba7d63d3f8a1fc031afacffe4fc58fc7512f4bb46845",
    "stage_scripts.json": "7c1a7b6009ee2e82f61820060d53a92fc91cbf1e33e2f6013c63eb85e9e1bfd2",
}

#: 可选但"给了就必须核对"的 v6 文件（口径参考物）。
V6_OPTIONAL_FILES = {
    "deadline_wrapper.py": "5c515706ad2e1f52bafc6e7d10c1b39343c56a908f72b09940b8355139726f2c",
}

#: v6 交付包整包 SHA-256（出处锚）。
V6_PACKAGE_SHA256 = "0151b35ad92c36985d1632d37b9b06e9513c64e3afd4c4c8d26f889dfd1676c9"

#: 交付包内自带的 v6 副本位置（相对仓库根）。
DEFAULT_V6_RELATIVE = os.path.join("tools", "v6")

#: 允许从子进程环境里**剥掉**的键（前缀规则）：任何 DEADLINE/BUDGET 通道。
STRIPPED_ENV_PATTERNS = ("DEADLINE", "BUDGET", "KILL_AFTER", "TIME_LIMIT")

#: 父进程签发的截止文件名。
DEADLINE_FILENAME = "supervised_deadline.json"

#: 子进程据此判断"我在监督器下运行"。
SUPERVISED_FLAG_ENV = "V3_SUPERVISED"
#: 父进程把"授权截止文件在哪、它的 SHA-256 是多少"告诉子进程的两个**只读**通道。
#: 刻意**不**含 `DEADLINE` / `BUDGET` 字样：那两个词属于"可被用来扩时的通道"，
#: 会被 `sanitize_child_env()` 剥掉，也会被 `reject_widening()` 判为重新注入。
CUTOFF_FILE_ENV = "V3_SUPERVISED_CUTOFF_FILE"
CUTOFF_SHA_ENV = "V3_SUPERVISED_CUTOFF_SHA256"
#: 允许出现在受监督子进程环境里的 V3_* 键（其余 DEADLINE/BUDGET 键一律判为注入）。
ALLOWED_SUPERVISED_ENV = (SUPERVISED_FLAG_ENV, CUTOFF_FILE_ENV, CUTOFF_SHA_ENV)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ ① 复用前先核对 v6 包


def verify_v6_package(directory: str) -> dict:
    """逐字节核对 v6 监督器包；缺文件或哈希不符一律拒绝。

    这是"复用"与"另造一个长得像的框架"之间的分界线：只要有人改了监督器的一个字节，
    这里就 `v6_package_tampered`，而不是继续拿它当可信外层。
    """
    if not directory or not os.path.isdir(directory):
        raise MissingInput(
            "v6_supervisor_missing",
            "找不到 KAGGLE-32 v6 监督器目录（--v6-dir 或 V3_V6_DIR）："
            "本入口不自行实现监督器，必须复用 v6 契约",
            directory=directory,
            expected=dict(V6_REQUIRED_FILES),
        )
    verified: dict[str, str] = {}
    problems: list[str] = []
    for name, expected in sorted(V6_REQUIRED_FILES.items()):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            problems.append("缺少 %s" % name)
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append("%s 哈希不符：期望 %s 实际 %s" % (name, expected, actual))
            continue
        verified[name] = actual
    optional: dict[str, str] = {}
    for name, expected in sorted(V6_OPTIONAL_FILES.items()):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            optional[name] = "absent"
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append("%s（可选）哈希不符：期望 %s 实际 %s" % (name, expected, actual))
            continue
        optional[name] = actual
    if problems:
        raise IntegrityError(
            "v6_package_tampered",
            "v6 监督器包未通过逐字节核对：%s" % "；".join(problems),
            problems=problems,
            directory=os.path.abspath(directory),
            expected=dict(V6_REQUIRED_FILES),
        )
    return {
        "directory": os.path.abspath(directory),
        "verified_files": verified,
        "optional_files": optional,
        "package_sha256_anchor": V6_PACKAGE_SHA256,
        "note": "逐字节核对通过：本入口复用的是 v6 原件，不是同名的本地改写版",
    }


def default_v6_dir(repo_root: str | None = None) -> str:
    root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, DEFAULT_V6_RELATIVE)


# ------------------------------------------------------------------ ② 预算与截止（不可放宽）


def strict_positive(text, name: str) -> float:
    """严格解析有限正数（与 v6 `shard_guard.strict_positive_number` 同口径）。"""
    raw = None if text is None else str(text)
    if raw is None or raw != raw.strip() or raw == "":
        raise ValueError("%s 缺失或含空白：%r" % (name, raw))
    if raw.startswith("+") or raw.lower().startswith(("nan", "inf", "-inf")):
        raise ValueError("%s 不是普通有限正数：%r" % (name, raw))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError("%s 不是数字：%r" % (name, raw))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("%s 必须是有限正数：%r" % (name, raw))
    return value


def compute_budget(
    *,
    approved_budget_seconds: float | None = None,
    outer_deadline_epoch: float | None = None,
    now_epoch: float | None = None,
    now_monotonic: float | None = None,
) -> dict:
    """`min(批准预算, 外层截止剩余)` —— **只能收紧，不能放宽**。

    两者都缺 → 拒绝（真实模式不得无界运行；受监督模式必须有外层截止）。
    """
    moment_epoch = float(now_epoch if now_epoch is not None else time.time())
    moment_monotonic = float(now_monotonic if now_monotonic is not None else time.monotonic())
    candidates: list[dict] = []
    if approved_budget_seconds is not None:
        candidates.append(
            {
                "source": "approved-budget",
                "seconds": float(approved_budget_seconds),
            }
        )
    if outer_deadline_epoch is not None:
        candidates.append(
            {
                "source": "outer-deadline",
                "seconds": float(outer_deadline_epoch) - moment_epoch,
            }
        )
    if not candidates:
        raise MissingInput(
            "supervised_budget_missing",
            "受监督模式必须给出批准预算或外层截止之一：不得无界运行",
        )
    for item in candidates:
        if not math.isfinite(item["seconds"]) or item["seconds"] <= 0:
            raise PolicyViolation(
                "supervised_budget_invalid",
                "预算来源 %s 不是有限正数：%r" % (item["source"], item["seconds"]),
            )
    chosen = min(candidates, key=lambda item: item["seconds"])
    return {
        "budget_seconds": float(chosen["seconds"]),
        "binding_source": chosen["source"],
        "candidates": candidates,
        "started_monotonic": moment_monotonic,
        "deadline_monotonic": moment_monotonic + float(chosen["seconds"]),
        "deadline_epoch": moment_epoch + float(chosen["seconds"]),
        "never_widens": True,
        "note": "只取更早者：外层截止与本入口预算都不可能被对方放宽",
    }


def phase_caps(budget_seconds: float) -> dict:
    """按总预算切出 prep / wrapup / kill 的**内部预留**（都是预算内的切分，不是加时）。

    默认值来自 v6（prep 120s / wrapup 180s / kill 30s），在短预算下按比例收紧，
    否则 `setup_hard_end` 会算成负数、监督器直接跳过 setup。
    """
    budget = float(budget_seconds)
    return {
        "prep_cap": min(120.0, max(0.5, budget * 0.15)),
        "wrapup_cap": min(180.0, max(0.5, budget * 0.25)),
        "kill_after": min(30.0, max(0.2, budget * 0.05)),
    }


def sanitize_child_env(env: Mapping | None = None) -> dict:
    """剥掉一切可由子进程扩时的 V3_* 截止/预算通道，并如实记录剥了哪些。"""
    source = dict(os.environ if env is None else env)
    stripped: dict[str, str] = {}
    kept = dict(source)
    for key in list(kept.keys()):
        upper = key.upper()
        if upper.startswith('PYTHON'):
            stripped[key] = kept.pop(key)
            continue
        if not upper.startswith("V3_"):
            continue
        if any(pattern in upper for pattern in STRIPPED_ENV_PATTERNS):
            stripped[key] = kept.pop(key)
    kept.pop("V3_TOTAL_BUDGET_SECONDS", None)
    # v6's unmodified setup/audit workers use an absolute trusted script path.
    # They inherit this fixed value, not the caller's user-site preferences.
    kept['PYTHONNOUSERSITE'] = '1'
    return {
        "env": kept,
        "stripped": stripped,
        "stripped_keys": sorted(stripped),
        "note": (
            "子进程拿不到任何 V3_* 截止/预算环境变量：唯一的授权截止来自父进程签发的"
            "截止文件（带 SHA-256 自锚）"
        ),
    }


# ------------------------------------------------------------------ 截止文件（父签发）


def write_deadline_file(path: str, payload: Mapping) -> dict:
    """父进程签发截止文件，返回含自锚 SHA-256 的记录。"""
    body = dict(payload)
    body.pop("file_sha256", None)
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    digest = sha256_file(path)
    record = dict(body)
    record["file_sha256"] = digest
    record["path"] = os.path.abspath(path)
    return record


def read_deadline_file(path: str | None, expected_sha256: str | None) -> dict:
    """子进程读截止文件：**先核自锚**，再取字段；任何不一致都拒绝。"""
    if not path:
        raise MissingInput(
            "supervised_deadline_missing",
            "受监督模式必须给出 --deadline-file（由外层监督器签发）："
            "缺了它就只剩子进程自报的预算，独审阻断 6 正是这一点",
        )
    if not os.path.isfile(path):
        raise MissingInput(
            "supervised_deadline_missing", "截止文件不存在：%s" % path, path=path
        )
    actual = sha256_file(path)
    if not expected_sha256:
        raise MissingInput(
            "supervised_deadline_anchor_missing",
            "读截止文件必须同时给出它的 SHA-256 自锚（--deadline-file-sha256）："
            "否则文件被换成更晚的截止也无从发现",
        )
    if str(expected_sha256) != actual:
        raise IntegrityError(
            "supervised_deadline_tampered",
            "截止文件哈希与自锚不符（被改过或换过）：锚 %s / 实际 %s"
            % (expected_sha256, actual),
            expected=str(expected_sha256),
            actual=actual,
            path=os.path.abspath(path),
        )
    with open(path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise IntegrityError("supervised_deadline_invalid", "截止文件根节点必须是对象")
    for field in ("deadline_monotonic", "deadline_epoch", "budget_seconds", "source"):
        if field not in payload:
            raise IntegrityError(
                "supervised_deadline_invalid", "截止文件缺字段 %s" % field, payload=dict(payload)
            )
    try:
        for field in ("deadline_monotonic", "deadline_epoch", "budget_seconds"):
            value = float(payload[field])
            if not math.isfinite(value):
                raise ValueError(field)
    except (TypeError, ValueError):
        raise IntegrityError(
            "supervised_deadline_invalid",
            "截止文件里的时间/预算字段不是有限数：%r" % dict(payload),
        )
    return {
        "path": os.path.abspath(path),
        "file_sha256": actual,
        "source": str(payload["source"]),
        "budget_seconds": float(payload["budget_seconds"]),
        "deadline_monotonic": float(payload["deadline_monotonic"]),
        "deadline_epoch": float(payload["deadline_epoch"]),
        "approval_context_sha256": payload.get("approval_context_sha256"),
        "approved_budget_seconds": payload.get("approved_budget_seconds"),
        "verified_against_anchor": True,
    }


def reject_widening(
    *,
    supervised: Mapping,
    requested_budget_seconds: float | None,
    requested_deadline_epoch: float | None,
    now_epoch: float | None = None,
    now_monotonic: float | None = None,
    env: Mapping | None = None,
) -> dict:
    """拒绝任何"把截止往后推"的请求（不是静默夹紧，而是显式失败）。

    三条判据：

    1. CLI/环境的预算会得出比受监督截止**更晚**的时刻 → `deadline_widening_rejected`；
    2. 受监督子进程环境里出现 `V3_*DEADLINE*`/`V3_*BUDGET*` → 有人重新注入了扩时通道
       → `deadline_widening_env_present`；
    3. 不晚于受监督截止的请求：允许（它只是更严），并如实记 `requested_tighter`。
    """
    moment_epoch = float(now_epoch if now_epoch is not None else time.time())
    moment_monotonic = float(now_monotonic if now_monotonic is not None else time.monotonic())
    supervised_deadline = float(supervised["deadline_monotonic"])
    supervised_budget = float(supervised["budget_seconds"])
    details: dict = {
        "supervised_deadline_monotonic": supervised_deadline,
        "supervised_budget_seconds": supervised_budget,
        "supervised_source": supervised.get("source"),
    }
    source_env = dict(os.environ if env is None else env)
    injected = sorted(
        key
        for key in source_env
        if key.upper().startswith("V3_")
        and key not in ALLOWED_SUPERVISED_ENV
        and any(pattern in key.upper() for pattern in STRIPPED_ENV_PATTERNS)
    )
    if injected:
        raise PolicyViolation(
            "deadline_widening_env_present",
            "受监督子进程的环境里仍存在截止/预算通道 %s：外层必须剥掉它们，"
            "否则子进程可以自行把截止推后" % injected,
            injected_keys=injected,
            detail=details,
        )
    requested_deadline = None
    requested_source = None
    if requested_budget_seconds is not None:
        requested_deadline = moment_monotonic + float(requested_budget_seconds)
        requested_source = "time-budget-seconds"
    if requested_deadline_epoch is not None:
        candidate = moment_monotonic + max(0.0, float(requested_deadline_epoch) - moment_epoch)
        if requested_deadline is None or candidate < requested_deadline:
            requested_deadline = candidate
            requested_source = "deadline-epoch"
    if requested_deadline is None:
        details["requested"] = None
        return {"ok": True, "widened": False, "details": details}
    details["requested_deadline_monotonic"] = requested_deadline
    details["requested_source"] = requested_source
    if requested_deadline > supervised_deadline:
        raise PolicyViolation(
            "deadline_widening_rejected",
            "请求的截止比外层监督器签发的截止更晚（%.3f > %.3f，来源 %s）："
            "受监督模式不接受任何放宽，外层截止不可被子进程改写"
            % (requested_deadline, supervised_deadline, requested_source),
            requested_deadline_monotonic=requested_deadline,
            supervised_deadline_monotonic=supervised_deadline,
            requested_source=requested_source,
            detail=details,
        )
    details["requested_tighter"] = True
    return {"ok": True, "widened": False, "requested_tighter": True, "details": details}


# ------------------------------------------------------------------ 计划与编排


def isolated_module_command(module):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return [sys.executable, '-I', '-X', 'utf8', '-B', os.path.join(root, 'tools', 'isolated_train_entry.py'), module]


def build_supervised_plan(
    *,
    v6_dir: str,
    main_command: list[str],
    run_dir: str,
    fixture_stuck_seconds: float | None = None,
) -> dict:
    """三段式计划：prep（核对遍历态）+ main（真正的工程检查）+ wrapup（哈希收尾）。

    - **prep**：复用 v6 `shard_job.py` 的 `hash`? 不 —— prep 在这一层做的是
      "确认子进程拿到的截止文件与自锚一致"，由本模块的 `--emit-prep` 动作完成；
    - **main**：`python -m v3.train.engineering_check ...`（真实子进程）；
    - **wrapup**：直接复用 v6 `shard_job.py hash <run_dir>`，失败即整体非零。
    - `fixture_stuck_seconds`：**只给 CPU 反例用**——把 main 换成一个卡住的进程，
      用来验证外层真的能 TERM/KILL 它。
    """
    if fixture_stuck_seconds is not None:
        main = [
            sys.executable,
            "-I",
            "-X", "utf8",
            "-B",
            "-c",
            "import time,sys;print('stuck-main-started',flush=True);time.sleep(%.3f)" % float(fixture_stuck_seconds),
        ]
    else:
        main = list(main_command)
    return {
        "phases": [
            {
                "name": "prep",
                "kind": "prep",
                "command": isolated_module_command('v3.train.supervised_check') + ['--emit-prep'],
            },
            {"name": "main", "kind": "main", "command": main},
            {
                "name": "wrapup",
                "kind": "wrapup",
                "command": [sys.executable, "-I", "-X", "utf8", "-B", os.path.join(v6_dir, "shard_job.py"), "hash", run_dir],
            },
        ],
        "note": "三段俱在：wrapup 不可省略（v6 validate_plan 强制），且全部在同一个总截止内",
    }


def parse_supervisor_output(text: str) -> dict:
    """解析监督器的 JSONL 事件流（非 JSON 行如实收集，不静默丢弃）。"""
    records: list[dict] = []
    noise: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError:
            noise.append(stripped)
            continue
        if isinstance(item, dict):
            records.append(item)
        else:
            noise.append(stripped)
    events: dict[str, list[dict]] = {}
    for item in records:
        events.setdefault(str(item.get("event")), []).append(item)
    return {
        "records": records,
        "events": {key: len(value) for key, value in sorted(events.items())},
        "by_event": events,
        "non_json_lines": noise,
        "job_result": (events.get("job_result") or [{}])[-1],
    }


def _build_engineering_child_command(
    *,
    dest_dir: str,
    deadline_path: str,
    deadline_sha256: str,
    budget_seconds: float,
    args,
) -> list[str]:
    """构造 main 阶段的子命令（`python -m v3.train.engineering_check ...`）。

    **刻意不传 `--time-budget-seconds`**：受监督模式下唯一的预算来源是外层签发的
    截止文件。若同时传一个数值，子进程实测到的时刻总比父进程晚一点点
    （进程启动开销），那点差值就会被 `deadline_widening_rejected` 正确地抓住 ——
    与其传一个必然"略宽"的数，不如让文件成为唯一权威。
    """
    command = isolated_module_command('v3.train.engineering_check') + [
        "--supervised",
        "--dest-dir",
        os.path.abspath(dest_dir),
        "--deadline-file",
        os.path.abspath(deadline_path),
        "--deadline-file-sha256",
        deadline_sha256,
        "--steps",
        str(int(args.steps)),
        "--checkpoint-every",
        str(int(args.checkpoint_every)),
    ]
    if args.stop_at_step is not None:
        command += ["--stop-at-step", str(int(args.stop_at_step))]
    if args.hashes:
        command += ["--hashes", os.path.abspath(args.hashes)]
    if args.backend:
        command += ["--backend", str(args.backend)]
    if args.plan_json:
        command += ["--plan-json", os.path.abspath(args.plan_json)]
    if args.probe_cuda:
        command += ["--probe-cuda"]
    for flag, value in (
        ("--model-id", args.model_id),
        ("--model-revision", args.model_revision),
        ("--target-modules", args.target_modules),
        ("--model-inputs-root", args.model_inputs_root),
        ("--local-model-dir", args.local_model_dir),
        ("--approval", args.approval),
        ("--approval-expected-sha256", args.approval_expected_sha256),
        ("--approval-evidence-ref", args.approval_evidence_ref),
        ("--interface", args.interface),
        ("--out", os.path.join(os.path.dirname(os.path.abspath(deadline_path)), "child_evidence.json")),
    ):
        if value:
            command += [flag, str(value)]
    if args.local_load_authorized:
        command += ["--local-load-authorized"]
    if args.allow_download:
        command += ["--allow-download"]
    if args.requested_gpu_hours is not None:
        command += ["--requested-gpu-hours", str(float(args.requested_gpu_hours))]
    return command


def run_supervised_check(
    *,
    v6_dir: str,
    dest_dir: str,
    run_root: str,
    steps: int = 8,
    checkpoint_every: int = 4,
    stop_at_step: int | None = 6,
    approved_budget_seconds: float | None = None,
    outer_deadline_epoch: float | None = None,
    approval_context: Mapping | None = None,
    fixture_stuck_seconds: float | None = None,
    fixture_phases: list | None = None,
    dry_run: bool = False,
    extra_child_args: Mapping | None = None,
    env: Mapping | None = None,
) -> dict:
    """跑一次受监督的工程检查（外层编排 + 单一总截止 + 强制收尾）。

    返回监督器账目 + 本层证据；`dry_run=True` 时只回命令与预算，不启动任何进程。
    """
    package = verify_v6_package(v6_dir)
    overrides = dict(extra_child_args or {})
    if overrides.get("backend") == "torch-peft":
        from . import v6_approval, input_binding, runner
        from .deployment import read_publication
        from .lifecycle import hashes_from_json
        anchor = v6_approval.deployment_approval_anchor()
        if not anchor:
            raise PolicyViolation("approval_not_trusted", "部署方尚未接入已发布批准锚")
        published = read_publication()['approval']
        approved_end = float(published['not_after_epoch'])
        if outer_deadline_epoch is None:
            outer_deadline_epoch = approved_end
        elif float(outer_deadline_epoch) > approved_end:
            raise PolicyViolation('approval_not_trusted', 'Outer cutoff exceeds publication expiry')
        if fixture_phases is not None or fixture_stuck_seconds is not None:
            raise PolicyViolation("production_fixture_forbidden", "真实入口不能替换监督阶段")
        if approved_budget_seconds is None or overrides.get("requested_gpu_hours") is None:
            raise MissingInput("approval_budget_missing", "真实入口必须给出批准预算和 GPU-h")
        hours = v6_approval.strict_positive_number(overrides["requested_gpu_hours"], "GPU-h")
        if float(approved_budget_seconds) > hours * 3600:
            raise PolicyViolation("approval_budget_exceeded", "墙钟预算超出单卡 GPU-h 申请")
        binding = input_binding.bind_model_inputs(
            root=overrides.get("model_inputs_root"),
            interface_path=overrides.get("interface") or os.path.join(
                os.path.dirname(__file__), "..", "locks", "official-interface.json"))
        modules = [item.strip() for item in (overrides.get("target_modules") or "").split(",") if item.strip()]
        options = v6_approval.runtime_options(
            steps=steps, checkpoint_every=checkpoint_every, stop_at_step=stop_at_step,
            budget_seconds=approved_budget_seconds, requested_gpu_hours=hours,
            model_id=overrides.get("model_id"), model_revision=overrides.get("model_revision"),
            target_modules=modules)
        approval_context = v6_approval.measure_runtime_context(
            plan=runner.plan_from_json(overrides.get("plan_json")), input_binding=binding,
            local_model_dir=overrides.get("local_model_dir"), options=options)
        v6_approval.require_approval(
            approval_path=overrides.get("approval"), expected_sha256=overrides.get("approval_expected_sha256"),
            evidence_ref_path=overrides.get("approval_evidence_ref"), requested_gpu_hours=hours,
            runtime_context=approval_context, published_anchor=anchor)
        if hashes_from_json(overrides.get("hashes")) != approval_context["hashes"]:
            raise PolicyViolation("approval_runtime_mismatch", "外层哈希与实际运行对象不符")
    started_monotonic = time.monotonic()
    started_epoch = time.time()
    budget = compute_budget(
        approved_budget_seconds=approved_budget_seconds,
        outer_deadline_epoch=outer_deadline_epoch,
        now_epoch=started_epoch,
        now_monotonic=started_monotonic,
    )
    caps = phase_caps(budget["budget_seconds"])
    sanitized = sanitize_child_env(env)

    run_id = "local-%s" % time.strftime("%Y%m%dT%H%M%S", time.gmtime(started_epoch))
    run_dir = os.path.join(os.path.abspath(run_root), run_id)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=False)
    deadline_path = os.path.join(run_dir, DEADLINE_FILENAME)
    deadline_record = write_deadline_file(
        deadline_path,
        {
            "source": "v6-supervisor-adapter",
            "budget_seconds": budget["budget_seconds"],
            "approved_budget_seconds": approved_budget_seconds,
            "binding_source": budget["binding_source"],
            "deadline_monotonic": budget["deadline_monotonic"],
            "deadline_epoch": budget["deadline_epoch"],
            "approval_context_sha256": sha256_json(dict(approval_context or {})),
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_epoch)),
            "never_widens": True,
        },
    )
    child_env = dict(sanitized["env"])
    # v6 sends UTF-8 JSON over stdin; Python's Windows locale default is not UTF-8.
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env[SUPERVISED_FLAG_ENV] = "1"
    child_env[CUTOFF_FILE_ENV] = deadline_path
    child_env[CUTOFF_SHA_ENV] = deadline_record["file_sha256"]

    overrides = dict(extra_child_args or {})
    args = _Args(
        steps=steps,
        checkpoint_every=checkpoint_every,
        stop_at_step=stop_at_step,
        hashes=overrides.get("hashes"),
        backend=overrides.get("backend", "synthetic"),
        plan_json=overrides.get("plan_json"),
        probe_cuda=bool(overrides.get("probe_cuda", False)),
        model_id=overrides.get("model_id"),
        model_revision=overrides.get("model_revision"),
        target_modules=overrides.get("target_modules"),
        model_inputs_root=overrides.get("model_inputs_root"),
        local_model_dir=overrides.get("local_model_dir"),
        approval=overrides.get("approval"),
        approval_expected_sha256=overrides.get("approval_expected_sha256"),
        approval_evidence_ref=overrides.get("approval_evidence_ref"),
        interface=overrides.get("interface"),
        local_load_authorized=bool(overrides.get("local_load_authorized", False)),
        allow_download=bool(overrides.get("allow_download", False)),
        requested_gpu_hours=overrides.get("requested_gpu_hours"),
    )
    plan = build_supervised_plan(
        v6_dir=package["directory"],
        main_command=_build_engineering_child_command(
            dest_dir=dest_dir,
            deadline_path=deadline_path,
            deadline_sha256=deadline_record["file_sha256"],
            budget_seconds=budget["budget_seconds"],
            args=args,
        ),
        run_dir=run_dir,
        fixture_stuck_seconds=fixture_stuck_seconds,
    )
    if fixture_phases is not None:
        # **仅 CPU fixture 用**：替换三段命令（例如换成不触碰磁盘的打印命令），
        # 用来在"监督器子进程不允许访问工作区"的沙箱里仍然验证编排与硬截止。
        plan = {"phases": [dict(item) for item in fixture_phases], "note": "fixture-phases"}
    plan_path = os.path.join(run_dir, "plan.json")
    with open(plan_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)

    ledger_path = os.path.join(run_dir, "ledger.jsonl")
    supervisor = os.path.join(package["directory"], "shard_supervisor.py")
    command = [
        sys.executable,
        "-I",
        "-X", "utf8",
        "-B",
        supervisor,
        "--plan",
        plan_path,
        "--run-dir",
        run_dir,
        "--ledger",
        ledger_path,
        "--total-budget",
        "%.6f" % budget["budget_seconds"],
        "--started-monotonic",
        "%.6f" % started_monotonic,
        "--prep-cap",
        "%.6f" % caps["prep_cap"],
        "--wrapup-cap",
        "%.6f" % caps["wrapup_cap"],
        "--kill-after",
        "%.6f" % caps["kill_after"],
    ]
    base = {
        "approved_runtime_context": dict(approval_context or {}),
        "stage": "supervised-engineering-check",
        "v6_package": package,
        "budget": budget,
        "phase_caps": caps,
        "deadline_file": deadline_record,
        "child_env_stripped_keys": sanitized["stripped_keys"],
        "child_env_note": sanitized["note"],
        "run_dir": run_dir,
        "plan_path": plan_path,
        "ledger_path": ledger_path,
        "supervisor_command": command,
        "is_real_gpu": False,
        "note": "本层只做编排：真正的硬截止能力来自 v6 监督器对子进程的 TERM/KILL",
    }
    if dry_run:
        base["dry_run"] = True
        base["verdict"] = "planned"
        return base

    finished = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        cwd=os.path.abspath(os.path.join(package["directory"], "..", "..")),
    )
    parsed = parse_supervisor_output(finished.stdout)
    job = parsed["job_result"] or {}
    overall_exit_code = int(job.get("overall_exit_code", int(finished.returncode)))
    terminate_events = parsed["by_event"].get("terminate_sent") or []
    kill_events = parsed["by_event"].get("kill_settled") or []
    phases = [
        item for item in parsed["records"] if item.get("event") == "phase_end"
    ]
    report = dict(base)
    report.update(
        {
            "supervisor_exit_code": int(finished.returncode),
            "main_exit_code": job.get("main_exit_code"),
            "wrapup_exit_code": job.get("wrapup_exit_code"),
            "overall_exit_code": overall_exit_code,
            "overall_status": job.get("overall_status"),
            "budget_respected": job.get("budget_respected"),
            "ledger_written": job.get("ledger_written"),
            "within_observation_tolerance": job.get("within_observation_tolerance"),
            "termination": {
                "terminate_signals_sent": len(terminate_events),
                "kill_settled_events": len(kill_events),
                "escalated_to_kill": [item.get("escalated_to_kill") for item in kill_events],
                "phases_terminated": sorted(
                    {str(item.get("phase")) for item in terminate_events}
                ),
                "detail": kill_events,
            },
            "phases": [
                {
                    "phase": item.get("phase"),
                    "kind": item.get("kind"),
                    "exit_code": item.get("exit_code"),
                    "stop_reason": item.get("stop_reason"),
                    "child_settled": item.get("child_settled"),
                    "elapsed_seconds": item.get("elapsed_seconds"),
                }
                for item in phases
            ],
            "events": parsed["events"],
            "non_json_lines": parsed["non_json_lines"],
            "supervisor_stderr_tail": finished.stderr[-2000:],
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 4),
            "verdict": "pass" if overall_exit_code == 0 else "fail",
        }
    )
    report["evidence_sha256"] = sha256_json(
        {key: value for key, value in report.items() if key != "evidence_sha256"}
    )
    return report


class _Args:
    """给 `_build_engineering_child_command` 用的轻量参数袋（CLI 与 API 共用）。"""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


# ------------------------------------------------------------------ CLI


def _emit_prep() -> int:
    """prep 阶段动作：确认子进程拿得到的截止文件与自锚一致（只读）。"""
    path = os.environ.get(CUTOFF_FILE_ENV)
    anchor = os.environ.get(CUTOFF_SHA_ENV)
    record = read_deadline_file(path, anchor)
    print(
        json.dumps(
            {"event": "prep_deadline_verified", "deadline_file": record},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="KAGGLE-38 r5：把工程检查交给 KAGGLE-32 v6 监督器（单一总截止）"
    )
    parser.add_argument("--emit-prep", action="store_true", help="仅 prep 动作（内部使用）")
    parser.add_argument("--v6-dir", default=os.environ.get("V3_V6_DIR") or default_v6_dir())
    parser.add_argument("--dest-dir", default=None, help="工程检查产物目录（仓库外绝对路径）")
    parser.add_argument("--run-root", default=None, help="监督器运行目录根（仓库外绝对路径）")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--stop-at-step", type=int, default=6)
    parser.add_argument("--approved-budget-seconds", type=float, default=None)
    parser.add_argument("--outer-deadline-epoch", type=float, default=None)
    parser.add_argument("--hashes", default=None)
    parser.add_argument("--backend", default="synthetic")
    parser.add_argument("--plan-json", default=None)
    parser.add_argument("--probe-cuda", action="store_true")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--target-modules", default=None)
    parser.add_argument("--model-inputs-root", default=None)
    parser.add_argument("--local-model-dir", default=None)
    parser.add_argument("--local-load-authorized", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--approval", default=None)
    parser.add_argument("--approval-expected-sha256", default=None)
    parser.add_argument("--approval-evidence-ref", default=None)
    parser.add_argument("--requested-gpu-hours", type=float, default=None)
    parser.add_argument("--interface", default=None)
    parser.add_argument(
        "--fixture-stuck-seconds",
        type=float,
        default=None,
        help="**仅 CPU 反例用**：把 main 换成一个卡住的进程，验证外层 TERM/KILL 时序",
    )
    parser.add_argument("--dry-run", action="store_true", help="只回命令与预算，不启动进程")
    parser.add_argument(
        "--fixture-print-phases",
        action="store_true",
        help="**仅 CPU 反例用**：把三段换成不触碰磁盘的打印命令（沙箱里验证编排用）",
    )
    parser.add_argument("--out", default=None, help="监督报告输出路径")
    args = parser.parse_args(argv)

    if args.emit_prep:
        try:
            return _emit_prep()
        except FailClosed as exc:
            print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
            return exc.exit_code

    try:
        budget = strict_positive(args.approved_budget_seconds, "approved-budget-seconds") if (
            args.approved_budget_seconds is not None
        ) else None
    except ValueError as exc:
        print(json.dumps({"code": "supervised_budget_invalid", "message": str(exc)}, ensure_ascii=False))
        return 7
    if args.dest_dir is None:
        print(json.dumps({"code": "dest_dir_missing", "message": "必须给出 --dest-dir"}, ensure_ascii=False))
        return 4
    run_root = args.run_root or os.path.join(os.path.dirname(os.path.abspath(args.dest_dir)), "supervised-runs")
    approval_context = {}
    if args.approval:
        approval_context = {
            "approval": os.path.abspath(args.approval),
            "anchor": args.approval_expected_sha256,
            "evidence_ref": args.approval_evidence_ref,
            "requested_gpu_hours": args.requested_gpu_hours,
        }
    try:
        report = run_supervised_check(
            v6_dir=str(args.v6_dir),
            dest_dir=str(args.dest_dir),
            run_root=run_root,
            steps=int(args.steps),
            checkpoint_every=int(args.checkpoint_every),
            stop_at_step=None if args.stop_at_step is None else int(args.stop_at_step),
            approved_budget_seconds=budget,
            outer_deadline_epoch=args.outer_deadline_epoch,
            approval_context=approval_context,
            fixture_stuck_seconds=args.fixture_stuck_seconds,
            fixture_phases=(
                [
                    {"name": "prep", "kind": "prep", "command": [sys.executable, "-c", "print('prep')"]},
                    {"name": "main", "kind": "main", "command": [sys.executable, "-c", "print('main')"]},
                    {"name": "wrapup", "kind": "wrapup", "command": [sys.executable, "-c", "print('wrapup')"]},
                ]
                if args.fixture_print_phases
                else None
            ),
            dry_run=bool(args.dry_run),
            extra_child_args={
                "hashes": args.hashes,
                "backend": args.backend,
                "plan_json": args.plan_json,
                "probe_cuda": bool(args.probe_cuda),
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "target_modules": args.target_modules,
                "model_inputs_root": args.model_inputs_root,
                "local_model_dir": args.local_model_dir,
                "local_load_authorized": bool(args.local_load_authorized),
                "allow_download": bool(args.allow_download),
                "approval": args.approval,
                "approval_expected_sha256": args.approval_expected_sha256,
                "approval_evidence_ref": args.approval_evidence_ref,
                "requested_gpu_hours": args.requested_gpu_hours,
                "interface": args.interface,
            },
        )
    except FailClosed as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
        return exc.exit_code

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    summary = {
        "verdict": report.get("verdict"),
        "overall_exit_code": report.get("overall_exit_code"),
        "overall_status": report.get("overall_status"),
        "main_exit_code": report.get("main_exit_code"),
        "wrapup_exit_code": report.get("wrapup_exit_code"),
        "budget_respected": report.get("budget_respected"),
        "binding_source": (report.get("budget") or {}).get("binding_source"),
        "budget_seconds": (report.get("budget") or {}).get("budget_seconds"),
        "child_env_stripped_keys": report.get("child_env_stripped_keys"),
        "termination": report.get("termination"),
        "phases": report.get("phases"),
        "run_dir": report.get("run_dir"),
        "evidence_sha256": report.get("evidence_sha256"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("verdict") in ("pass", "planned") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
