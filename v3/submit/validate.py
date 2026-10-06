"""官方提交载体校验（KAGGLE-26 / Q0 报告 Y14 ②，KAGGLE-26 复核后 🟡-2 修订）。

## 数字来源（修订要点）

上一版说"这一整组官方限额在本 checkout 内找不到可引用的原文"，只强制了设计稿里
有出处的两条，其余记入 `UNVERIFIED_OFFICIAL_LIMITS`。**这个判断是错的**：
KAGGLE-27 A 段已经用官方 wheel 探针把官方口径实测出来了，出处是
`KAGGLE-27-V0-A-section.zip → A-evidence.json → a2_limits`。本模块现在直接引用它：

- `built_limits_extensions`（**7 种**）：`.json .md .py .safetensors .txt .yaml .yml`
- `built_limits_adapter_extensions`：`.safetensors`（仅此一种）
- `built_limits_structural`：
  `max_total_size_bytes 3221225472` · `max_file_count 10000` · `max_yaml_files 1000`
  · `max_agents 500` · `max_sub_agent_depth 50` · `max_skills 1000`
  · `max_loop_iterations 500` · `max_instruction_chars 1000000`
  · `max_total_instruction_chars 10000000` · `max_yaml_size_bytes` / `max_skill_size_bytes`
  各 `52428800`
- 对照口径：`adk_submission.SubmissionLimits()` 默认是 **29** 种扩展名 / 6 种 adapter
  扩展名（含 `.bin .pt .pth .gguf .ggml`）——**库默认值不是官方口径**，用它得到的"通过"
  不代表官方通过。

## 本模块强制什么、不强制什么

**强制**（都能从一个已生成目录机械判定，逐条带边界反例）：

1. 扩展名必须落在官方 7 种内（`submission_extension_not_allowed`）；
2. 禁止权重格式（`.bin/.pt/...`）出现，且 adapter 只能是 `.safetensors`；
3. `max_file_count` 10000（超过即拒，恰好在 10000 通过）；
4. `max_yaml_files` 1000（`.yaml` + `.yml` 合计）；
5. `max_yaml_size_bytes` 52428800（单个 YAML 文件）；
6. `max_total_size_bytes` 3221225472 —— 官方是 `≤` 口径，而设计稿
   （KAGGLE-20:10 / KAGGLE-21:10）要求"总解压 **<**3GiB"。本模块取**更严**的严格小于，
   并在报告里同时给出官方 `≤` 值，边界反例是"恰好 3 GiB 被拒"；
7. adapter 文件名必须是 `adapter.safetensors`。

**不强制**（`not_locally_checkable`）：`max_agents` / `max_sub_agent_depth` /
`max_skills` / `max_loop_iterations` / `max_instruction_chars` /
`max_total_instruction_chars` / `max_skill_size_bytes`。它们是**提交 YAML 内容**的属性，
判定要经过官方 compiler/schema；本机没有官方包，**不猜** YAML 结构去近似它。
它们现在有出处，所以**不再**记在 `UNVERIFIED_OFFICIAL_LIMITS` 里。

在官方 compiler 可用之前，任何"已通过官方提交校验"的说法都不成立：
`official_limits_verified` 只有在 `official_submission_limits()` 真的返回时才为 True。
"""

from __future__ import annotations

import os

from ..common.errors import Blocked, MissingInput, PolicyViolation

#: adapter 文件名（提交载体只认这一个）。
ADAPTER_FILENAME = "adapter.safetensors"
#: adapter 只允许的扩展名（官方 `built_limits_adapter_extensions`，仅一种）。
REQUIRED_ADAPTER_EXTENSION = ".safetensors"
#: 明确禁止的权重格式：即使体积够小，也不是官方认可的 adapter 载体。
FORBIDDEN_WEIGHT_EXTENSIONS = (".bin", ".pt", ".ckpt", ".pth", ".gguf", ".h5", ".msgpack")

#: KAGGLE-27 A 段证据的出处前缀（一手 wheel 探针产物）。
KAGGLE27_EVIDENCE = "KAGGLE-27-V0-A-section.zip:A-evidence.json"
_A2_LIMITS = KAGGLE27_EVIDENCE + ":a2_limits"

#: 官方允许的扩展名清单（`a2_limits.built_limits_extensions`，7 种，逐项带出处）。
OFFICIAL_ALLOWED_EXTENSIONS = {
    ".json": _A2_LIMITS + ".built_limits_extensions",
    ".md": _A2_LIMITS + ".built_limits_extensions",
    ".py": _A2_LIMITS + ".built_limits_extensions",
    ".safetensors": _A2_LIMITS + ".built_limits_extensions（adapter 唯一载体）",
    ".txt": _A2_LIMITS + ".built_limits_extensions",
    ".yaml": _A2_LIMITS + ".built_limits_extensions",
    ".yml": _A2_LIMITS + ".built_limits_extensions",
}

#: 兼容旧名字（外部脚本可能引用）：现在指向官方清单。
DECLARED_ALLOWED_EXTENSIONS = OFFICIAL_ALLOWED_EXTENSIONS

#: 有出处的官方结构限额（`a2_limits.built_limits_structural`，外加设计稿的更严约束）。
SOURCED_LIMITS: dict[str, dict] = {
    "max_total_size_bytes": {
        "value": 3221225472,
        "citation": _A2_LIMITS + ".built_limits_structural.max_total_size_bytes（3221225472 = 3 GiB，≤ 口径）",
    },
    "max_total_unpacked_bytes": {
        "value": 3 * (1 << 30),
        "strictly_less_than": True,
        "citation": (
            "docs/v3/design/KAGGLE-20-V3-LoRA-joint-design.md:10（总解压大小 <3 GiB）；"
            "docs/v3/design/KAGGLE-21-V3-data-eval-spec.md:10（提交解包 <3GiB）。"
            "官方 max_total_size_bytes 同为 3221225472 但为 ≤ 口径，本模块取更严的 <"
        ),
    },
    "max_adapter_files": {
        "value": 8,
        "citation": "docs/v3/design/KAGGLE-20-V3-LoRA-joint-design.md:10（最多8个 adapter）",
    },
    "max_file_count": {
        "value": 10000,
        "citation": _A2_LIMITS + ".built_limits_structural.max_file_count",
    },
    "max_yaml_files": {
        "value": 1000,
        "citation": _A2_LIMITS + ".built_limits_structural.max_yaml_files",
    },
    "max_yaml_size_bytes": {
        "value": 52428800,
        "citation": _A2_LIMITS + ".built_limits_structural.max_yaml_size_bytes",
    },
}

#: 有出处、但**本模块无法从目录机械判定**的限额（提交 YAML 内容的属性，
#: 需要官方 compiler/schema）。有出处 ≠ 已强制，这里如实分开列。
NOT_LOCALLY_CHECKABLE_LIMITS: dict[str, dict] = {
    "max_agents": {"value": 500, "citation": _A2_LIMITS + ".built_limits_structural.max_agents"},
    "max_sub_agent_depth": {
        "value": 50,
        "citation": _A2_LIMITS + ".built_limits_structural.max_sub_agent_depth",
    },
    "max_skills": {"value": 1000, "citation": _A2_LIMITS + ".built_limits_structural.max_skills"},
    "max_loop_iterations": {
        "value": 500,
        "citation": _A2_LIMITS + ".built_limits_structural.max_loop_iterations",
    },
    "max_instruction_chars": {
        "value": 1000000,
        "citation": _A2_LIMITS + ".built_limits_structural.max_instruction_chars",
    },
    "max_total_instruction_chars": {
        "value": 10000000,
        "citation": _A2_LIMITS + ".built_limits_structural.max_total_instruction_chars",
    },
    "max_skill_size_bytes": {
        "value": 52428800,
        "citation": _A2_LIMITS + ".built_limits_structural.max_skill_size_bytes",
    },
}

#: 出现在任务/审查文本里、且**至今仍无出处**的官方限额。
#: KAGGLE-27 A 段证据把原先列在这里的五条全部落实了出处，因此本元组现在是空的；
#: 将来发现新的无出处数字继续追加在这里（不得手抄臆测数字当官方约束）。
UNVERIFIED_OFFICIAL_LIMITS: tuple[str, ...] = ()

#: 需要按 YAML 计数的扩展名。
YAML_EXTENSIONS = (".yaml", ".yml")


def declared_limits() -> dict:
    """本地声明限额 + 出处。

    `source` 仍是 `declared`（因为**没有**调用官方 builder），但每条数字都带
    KAGGLE-27 的一手出处；`verified_against_official_compiler` 仍为 False。
    """
    return {
        "source": "declared",
        "verified_against_official_compiler": False,
        "evidence": KAGGLE27_EVIDENCE,
        "allowed_extensions": dict(OFFICIAL_ALLOWED_EXTENSIONS),
        "adapter_extensions": [REQUIRED_ADAPTER_EXTENSION],
        "sourced_limits": {key: entry["value"] for key, entry in SOURCED_LIMITS.items()},
        "sourced_citations": {key: entry["citation"] for key, entry in SOURCED_LIMITS.items()},
        "not_locally_checkable": {
            key: entry["value"] for key, entry in NOT_LOCALLY_CHECKABLE_LIMITS.items()
        },
        "not_locally_checkable_citations": {
            key: entry["citation"] for key, entry in NOT_LOCALLY_CHECKABLE_LIMITS.items()
        },
        "unverified_official_limits": list(UNVERIFIED_OFFICIAL_LIMITS),
        "library_default_contrast": {
            "adk_default_extension_count": 29,
            "adk_default_adapter_extensions": [".bin", ".ggml", ".gguf", ".pt", ".pth", ".safetensors"],
            "citation": _A2_LIMITS + ".adk_default_extensions / .adk_default_adapter_extensions",
            "note": "库默认值不是官方口径：用 adk_submission 默认 limits 得到的通过不算官方通过",
        },
        "note": (
            "扩展名与结构限额取自 KAGGLE-27 A-evidence.json（官方 wheel 探针实测）；"
            "YAML 内容级限额列在 not_locally_checkable，需官方 compiler 才判定"
        ),
    }


def official_submission_limits():
    """尝试取官方限额。取不到即 `Blocked`，**不回退**成手抄常量冒充官方结果。"""
    try:  # pragma: no cover - 本机无官方包
        from sweegemma import build_submission_limits  # type: ignore
    except Exception as exc:
        raise Blocked(
            "official_compiler_unavailable",
            "无法导入官方 compiler/parser，提交限额只能作为声明值使用：%s" % exc,
        ) from exc
    return build_submission_limits()


def _collect_files(root: str) -> list[dict]:
    files: list[dict] = []
    for current, _dirs, names in os.walk(root):
        for name in names:
            full = os.path.join(current, name)
            files.append(
                {
                    "relative": os.path.relpath(full, root).replace("\\", "/"),
                    "extension": os.path.splitext(name)[1].lower(),
                    "bytes": os.path.getsize(full),
                }
            )
    return files


def validate_submission_dir(root: str, *, limits: dict | None = None) -> dict:
    """校验一个**已生成**的提交目录，返回结构化报告；不合规即抛 fail-closed 异常。

    强制项（每条都有出处，见模块 docstring）：
    目录存在且非空 · 扩展名在官方 7 种内 · 不得出现被禁止的权重扩展名 ·
    `max_file_count` · `max_yaml_files` · `max_yaml_size_bytes` ·
    总解压 **< 3 GiB** · adapter 必须是 `adapter.safetensors` 且 ≤ 8 个。
    """
    if not root or not os.path.isdir(root):
        raise MissingInput("submission_dir_missing", "提交目录不存在：%r" % root)

    declared = declared_limits()
    allowed = dict(declared["allowed_extensions"])
    if limits:
        for extension in limits.get("allowed_extensions", ()):  # 调用方覆盖
            allowed.setdefault(str(extension).lower(), "caller-provided")
    total_limit = int(
        (limits or {}).get("max_total_unpacked_bytes", SOURCED_LIMITS["max_total_unpacked_bytes"]["value"])
    )
    adapter_limit = int((limits or {}).get("max_adapter_files", SOURCED_LIMITS["max_adapter_files"]["value"]))
    file_count_limit = int((limits or {}).get("max_file_count", SOURCED_LIMITS["max_file_count"]["value"]))
    yaml_count_limit = int((limits or {}).get("max_yaml_files", SOURCED_LIMITS["max_yaml_files"]["value"]))
    yaml_size_limit = int(
        (limits or {}).get("max_yaml_size_bytes", SOURCED_LIMITS["max_yaml_size_bytes"]["value"])
    )
    extension_limit = int(
        (limits or {}).get("max_extension_count", len(OFFICIAL_ALLOWED_EXTENSIONS))
    )

    files = _collect_files(root)
    if not files:
        raise MissingInput("submission_empty", "提交目录里没有任何文件：%s" % root)

    # 1) 扩展名接受面 —— 官方恰好 7 种，多一种都不行。
    if len(allowed) > extension_limit:
        raise PolicyViolation(
            "submission_extension_allowlist_too_wide",
            "允许扩展名 %d 种 > 官方口径 %d 种" % (len(allowed), extension_limit),
            allowed=sorted(allowed),
            official_limit=extension_limit,
        )
    for entry in files:
        extension = entry["extension"]
        if extension in FORBIDDEN_WEIGHT_EXTENSIONS:
            raise PolicyViolation(
                "submission_adapter_not_safetensors",
                "提交包里出现被禁止的权重格式 %s：adapter 只能是 %s"
                % (extension, REQUIRED_ADAPTER_EXTENSION),
                file=entry["relative"],
            )
        if extension not in allowed:
            raise PolicyViolation(
                "submission_extension_not_allowed",
                "扩展名 %r 不在官方允许清单内" % extension,
                file=entry["relative"],
                allowed=sorted(allowed),
            )

    # 2) 结构限额（KAGGLE-27 `built_limits_structural`）。
    if len(files) > file_count_limit:
        raise PolicyViolation(
            "submission_file_count_exceeded",
            "文件数 %d 超过官方上限 %d" % (len(files), file_count_limit),
            file_count=len(files),
            limit=file_count_limit,
            citation=SOURCED_LIMITS["max_file_count"]["citation"],
        )
    yaml_files = [entry for entry in files if entry["extension"] in YAML_EXTENSIONS]
    if len(yaml_files) > yaml_count_limit:
        raise PolicyViolation(
            "submission_yaml_count_exceeded",
            "YAML 文件数 %d 超过官方上限 %d" % (len(yaml_files), yaml_count_limit),
            yaml_files=len(yaml_files),
            limit=yaml_count_limit,
            citation=SOURCED_LIMITS["max_yaml_files"]["citation"],
        )
    oversized_yaml = [entry for entry in yaml_files if entry["bytes"] > yaml_size_limit]
    if oversized_yaml:
        raise PolicyViolation(
            "submission_yaml_size_exceeded",
            "YAML 文件超过官方单文件上限 %d 字节" % yaml_size_limit,
            files=[entry["relative"] for entry in oversized_yaml][:10],
            limit=yaml_size_limit,
            citation=SOURCED_LIMITS["max_yaml_size_bytes"]["citation"],
        )

    # 3) 总大小：官方 ≤ 3GiB，设计稿要求严格小于 —— 取更严的一条。
    total_bytes = sum(entry["bytes"] for entry in files)
    if total_bytes >= total_limit:
        raise PolicyViolation(
            "submission_total_size_exceeded",
            "总解压大小 %d 字节不满足「严格小于 %d 字节」（3 GiB）" % (total_bytes, total_limit),
            total_bytes=total_bytes,
            limit_bytes=total_limit,
            citation=SOURCED_LIMITS["max_total_unpacked_bytes"]["citation"],
        )

    # 4) adapter 载体。
    adapters = [entry for entry in files if entry["extension"] == REQUIRED_ADAPTER_EXTENSION]
    if not adapters:
        raise PolicyViolation(
            "submission_adapter_missing",
            "提交包里没有 %s" % ADAPTER_FILENAME,
            files=[entry["relative"] for entry in files][:20],
        )
    if len(adapters) > adapter_limit:
        raise PolicyViolation(
            "submission_adapter_count_exceeded",
            "adapter 文件数 %d 超过上限 %d" % (len(adapters), adapter_limit),
            count=len(adapters),
        )
    misnamed = [entry["relative"] for entry in adapters if os.path.basename(entry["relative"]) != ADAPTER_FILENAME]
    if misnamed:
        raise PolicyViolation(
            "submission_adapter_name_invalid",
            "adapter 文件名必须是 %s：%s" % (ADAPTER_FILENAME, misnamed),
            files=misnamed,
        )

    return {
        "ok": True,
        "root": root,
        "files": len(files),
        "yaml_files": len(yaml_files),
        "adapter_files": len(adapters),
        "total_unpacked_bytes": total_bytes,
        "total_unpacked_limit_bytes": total_limit,
        "limits_source": "caller-provided" if limits else "declared",
        "official_limits_verified": False,
        "official_limits_evidence": KAGGLE27_EVIDENCE,
        "enforced_limits": {
            "max_total_unpacked_bytes": total_limit,
            "max_adapter_files": adapter_limit,
            "max_file_count": file_count_limit,
            "max_yaml_files": yaml_count_limit,
            "max_yaml_size_bytes": yaml_size_limit,
            "max_extension_count": extension_limit,
        },
        "enforced_citations": declared["sourced_citations"],
        "not_locally_checkable": declared["not_locally_checkable"],
        "unverified_official_limits": declared["unverified_official_limits"],
        "violations": [],
        "note": (
            "官方 compiler 未安装：本报告基于**有出处**的声明式限额（KAGGLE-27 A-evidence.json），"
            "并强制官方扩展名清单与四项结构限额；YAML 内容级限额需官方 compiler 才判定。"
            "不得据此声称已通过官方提交校验。"
        ),
    }
