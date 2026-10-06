"""官方提交载体校验（KAGGLE-26 / Q0 报告 Y14 ②）。

Q0 指出的现状：官方扩展名清单、adapter 仅 `.safetensors`、各类结构限额
—— **一条都没实现**，`ADAPTER_NAME = "adapter.safetensors"` 只是文件名常量。
本模块把这些限额落成代码。

**关于数字来源（很重要，直接决定本模块是否为"不编造"）**：

Q0 报告里出现的「7 种扩展名 / 3GiB / 10000 文件 / 1000 YAML / 500 agents / 深度 50」
这一整组口径，我在本 checkout 内**没有找到可引用的原文**。仓库里**确实**记录的是：

- `docs/v3/design/KAGGLE-20-V3-LoRA-joint-design.md:10`：
  「声明式 YAML；PEFT adapter 经 `adapter:` 挂载；**总解压大小 <3 GiB**，rank≤128，最多8个」
- `docs/v3/design/KAGGLE-21-V3-data-eval-spec.md:10`：
  「声明式提交；允许 adapters、受控 skill 脚本；评分环境无网络，**提交解包 <3GiB**」
- `docs/v3/design/KAGGLE-20-V3-LoRA-joint-design.md:75`：
  「导出只存 adapter 两文件……FP32 adapter 约 109.38MiB 仍远低于 3GiB」

所以本模块只强制**有出处**的四条：总解压 <3GiB、adapter 只 `.safetensors`、
adapter 文件数 ≤8、扩展名必须落在声明清单内（声明清单明确标 `declared`）。
其余数字（10000 文件 / 1000 YAML / 500 agents / 深度 50）**一律不硬编码、不冒充官方**：
它们出现在 `UNVERIFIED_OFFICIAL_LIMITS` 里，报告中标为"未取得出处，未强制"。

在官方 compiler 可用之前，任何"已通过官方提交校验"的说法都不成立。
"""

from __future__ import annotations

import os

from ..common.errors import Blocked, MissingInput, PolicyViolation

#: adapter 文件名（提交载体只认这一个）。
ADAPTER_FILENAME = "adapter.safetensors"
#: adapter 只允许的扩展名。
REQUIRED_ADAPTER_EXTENSION = ".safetensors"
#: 明确禁止的权重格式：即使体积够小，也不是官方认可的 adapter 载体。
FORBIDDEN_WEIGHT_EXTENSIONS = (".bin", ".pt", ".ckpt", ".pth", ".gguf", ".h5", ".msgpack")

#: 允许的扩展名清单。**标 declared**：本 checkout 内没有完整官方清单原文，
#: 这组值来自 KAGGLE-20:10 / KAGGLE-21:10 的"声明式 YAML + adapters + 受控 skill 脚本"
#: 与提交面常识，未经官方取值核对。官方包可用时必须覆盖。
DECLARED_ALLOWED_EXTENSIONS = {
    ".yaml": "docs/v3/design/KAGGLE-20-V3-LoRA-joint-design.md:10（声明式 YAML）",
    ".yml": "同上（YAML 常见别名）",
    ".json": "declared：配置文件常见格式，未见官方逐项清单原文",
    ".py": "docs/v3/design/KAGGLE-21-V3-data-eval-spec.md:10（受控 skill 脚本）",
    ".md": "declared：说明文件，未见官方逐项清单原文",
    ".txt": "declared：说明文件，未见官方逐项清单原文",
    ".safetensors": "docs/v3/design/KAGGLE-20-V3-LoRA-joint-design.md:10（PEFT adapter 挂载）",
}

#: 有出处的限额。
SOURCED_LIMITS: dict[str, dict] = {
    "max_total_unpacked_bytes": {
        "value": 3 * (1 << 30),
        "strictly_less_than": True,
        "citation": (
            "docs/v3/design/KAGGLE-20-V3-LoRA-joint-design.md:10"
            "（总解压大小 <3 GiB）；docs/v3/design/KAGGLE-21-V3-data-eval-spec.md:10"
            "（提交解包 <3GiB）"
        ),
    },
    "max_adapter_files": {
        "value": 8,
        "citation": "docs/v3/design/KAGGLE-20-V3-LoRA-joint-design.md:10（最多8个 adapter）",
    },
}

#: 出现在任务/审查文本里、但**在本 checkout 内找不到出处**的官方限额。
#: 记录在此以便复核，**不参与强制**（不得手抄臆测数字当官方约束）。
UNVERIFIED_OFFICIAL_LIMITS = (
    "最多 10000 个文件",
    "最多 1000 个 YAML",
    "最多 500 个 agents",
    "目录深度 ≤ 50",
    "官方允许扩展名恰好 7 种",
)


def declared_limits() -> dict:
    """本地声明限额 + 出处，明确标注**未经官方核对**。"""
    return {
        "source": "declared",
        "verified_against_official_compiler": False,
        "allowed_extensions": dict(DECLARED_ALLOWED_EXTENSIONS),
        "sourced_limits": {key: entry["value"] for key, entry in SOURCED_LIMITS.items()},
        "sourced_citations": {key: entry["citation"] for key, entry in SOURCED_LIMITS.items()},
        "unverified_official_limits": list(UNVERIFIED_OFFICIAL_LIMITS),
        "note": (
            "只强制有出处的限额；未见出处的官方数字记入 unverified_official_limits，"
            "既不硬编码也不声称已满足"
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


def validate_submission_dir(root: str, *, limits: dict | None = None) -> dict:
    """校验一个**已生成**的提交目录，返回结构化报告；不合规即抛 fail-closed 异常。

    强制项（全部有出处）：

    - 目录存在且非空；
    - 不得出现被禁止的权重扩展名；
    - 扩展名必须落在声明清单内；
    - adapter 必须是 `adapter.safetensors`，且至少存在一个；
    - **总解压字节数 < 3 GiB**（严格小于）；
    - adapter 文件数 ≤ 8。
    """
    if not root or not os.path.isdir(root):
        raise MissingInput("submission_dir_missing", "提交目录不存在：%r" % root)

    declared = declared_limits()
    allowed = dict(declared["allowed_extensions"])
    if limits:
        for extension in limits.get("allowed_extensions", ()):  # 调用方覆盖
            allowed.setdefault(str(extension).lower(), "caller-provided")
    total_limit = int((limits or {}).get("max_total_unpacked_bytes", SOURCED_LIMITS["max_total_unpacked_bytes"]["value"]))
    adapter_limit = int((limits or {}).get("max_adapter_files", SOURCED_LIMITS["max_adapter_files"]["value"]))

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
    if not files:
        raise MissingInput("submission_empty", "提交目录里没有任何文件：%s" % root)

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
                "扩展名 %r 不在声明允许清单内" % extension,
                file=entry["relative"],
                allowed=sorted(allowed),
            )

    total_bytes = sum(entry["bytes"] for entry in files)
    if total_bytes >= total_limit:
        raise PolicyViolation(
            "submission_total_size_exceeded",
            "总解压大小 %d 字节不满足「严格小于 %d 字节」（3 GiB）"
            % (total_bytes, total_limit),
            total_bytes=total_bytes,
            limit_bytes=total_limit,
        )

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
        "adapter_files": len(adapters),
        "total_unpacked_bytes": total_bytes,
        "total_unpacked_limit_bytes": total_limit,
        "limits_source": "caller-provided" if limits else "declared",
        "official_limits_verified": False,
        "enforced_limits": {
            "max_total_unpacked_bytes": total_limit,
            "max_adapter_files": adapter_limit,
        },
        "enforced_citations": declared["sourced_citations"],
        "unverified_official_limits": declared["unverified_official_limits"],
        "violations": [],
        "note": (
            "官方 compiler 未安装、且本 checkout 内没有完整官方扩展名清单原文："
            "本报告基于**声明式**限额，只强制有出处的两条。"
            "不得据此声称已通过官方提交校验。"
        ),
    }
