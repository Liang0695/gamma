"""官方 PEFT adapter 载体契约（KAGGLE-27 三项官方契约整改之①）。

## 为什么需要这个模块

E0 上一版只校验 adapter 的 **basename**（`adapter.safetensors`），
而官方口径是 **PEFT 目录** `adapters/<adapter_name>/adapter_model.safetensors`，
且该目录里必须有 `adapter_config.json`。

KAGGLE-27 用官方 `adk_submission.discovery.discover_adapters()` 实测出命名矩阵，
其中一格是**能过 E0 旧校验、却会让官方编译器失败**的组合：

    adapters/v3_policy/adapter.safetensors（旧命名）+ 缺 adapter_config.json
      → 官方发现的 adapter 名退化成本文件名 stem `adapter`
      → `resolve_model` / `compile_submission` 抛
         AdapterNotFoundError: Adapter 'v3_policy' not found in discovered
         adapters. Available adapters: adapter

根因（KAGGLE-27 源码读数 + E0 本轮独立复现，见 `discovered_adapter_name`）：
目录名成为 adapter 名的判据是 **有 `adapter_config.json`** 或 **stem == "adapter_model"**。
`adapter_model.safetensors` 自带 stem 判据，缺 config 也保住目录名；
`adapter.safetensors` 只能靠 config 触发，缺失即退化为 stem。

## 本模块强制什么

1. 权重文件必须落在 `adapters/<adapter_name>/` 且 basename **恰好**是
   `adapter_model.safetensors`（`adapter_carrier_wrong_filename` /
   `adapter_carrier_not_in_adapters_dir`）；
2. 旧命名 `adapter.safetensors` 一律拒（`adapter_carrier_legacy_name`）——
   **不是"仅改名"**，它的失败模式是缺 config 时静默退化；
3. 同目录必须有 `adapter_config.json`，且必须是**可解析的 JSON 对象**
   （`adapter_carrier_config_missing` / `adapter_carrier_config_unparsable`）；
4. 指定了 `declared_adapter_name`（来自 `agent.yaml` 的 `adapter:`）时，该名字必须
   能被官方规则解析到（`adapter_carrier_declared_name_unresolved`）。

**真实加载仍待验证**：本模块只判**静态命名/config 契约**，不导入 torch/peft，
不加载任何权重。通过本模块 ≠ 官方 compiler 跑过 ≠ 真实 adapter 可加载。

## 出处

- `HARNESS_README.md` §3.4 第 202 行：*"Place PEFT LoRA directories (containing
  `adapter_config.json` and `adapter_model.safetensors`) inside `adapters/<adapter_name>/`."*
- 同文档 §2 上限表：`adapter_extensions` = `.safetensors` only
  (`adapters/<name>/adapter_model.safetensors`)。
- 命名矩阵：`KAGGLE-27-supplements-and-protocol.zip:S1-adapter-matrix-full.json`
  （六格，E0 本轮独立重跑得到**字节相同**的产物）；
  E0 另补四格消除歧义，冻结在
  `docs/v3/design/kaggle-27-s1b-discovery-rule-cells.json`。
"""

from __future__ import annotations

import json
import os
from typing import Mapping

from ..common.errors import MissingInput, PolicyViolation

#: 官方 adapter 根目录名。
ADAPTERS_DIRNAME = "adapters"
#: 官方 PEFT 权重文件名（**唯一**合法 basename）。
ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
#: 官方 PEFT 配置文件名（同目录必需，且必须可解析）。
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
#: E0 旧命名：现在**一律拒绝**（缺 config 时会被官方发现成本文件名 stem）。
LEGACY_ADAPTER_FILENAME = "adapter.safetensors"
#: 触发"目录名即 adapter 名"判据的特殊 stem。
DIRNAME_JUDGEMENT_STEM = "adapter_model"
#: E0 在提交 YAML 里声明的 adapter 名（必须与 `agent.yaml` 的 `adapter:` 字段一致）。
#: 这是**产物命名常量**，不是模型 pin 的替身：模型与 revision 仍必须来自锁定参数。
DEFAULT_ADAPTER_NAME = "v3_policy"

#: 静态命名契约的出处（一手官方文档，可离线复核）。
README_CITATION = "HARNESS_README.md §3.4（adapters/<adapter_name>/ 含 adapter_config.json 与 adapter_model.safetensors）；§2 上限表 adapter_extensions 行"
#: 命名矩阵证据出处（KAGGLE-27 S1，E0 独立重跑字节一致）。
MATRIX_CITATION = "KAGGLE-27-supplements-and-protocol.zip:S1-adapter-matrix-full.json（六格矩阵 + 官方 compiler AdapterNotFoundError 实测）"
#: E0 本轮补的四格（消除"目录名 vs stem"歧义）。
RULE_CELLS_CITATION = "docs/v3/design/kaggle-27-s1b-discovery-rule-cells.json（E0 独立探针，四格）"


def adapter_dir_relative_path(adapter_name: str) -> str:
    """`adapters/<adapter_name>`（POSIX 分隔符，便于与提交包内相对路径比对）。"""
    name = str(adapter_name or "").strip()
    if not name:
        raise MissingInput("adapter_name_missing", "adapter 名称为空，无法定位官方 PEFT 目录")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise PolicyViolation(
            "adapter_name_invalid",
            "adapter 名称不得包含路径分隔符或 . / ..：%r" % (adapter_name,),
        )
    return "%s/%s" % (ADAPTERS_DIRNAME, name)


def carrier_weights_relative_path(adapter_name: str) -> str:
    """官方权重相对路径：`adapters/<adapter_name>/adapter_model.safetensors`。"""
    return "%s/%s" % (adapter_dir_relative_path(adapter_name), ADAPTER_WEIGHTS_FILENAME)


def carrier_config_relative_path(adapter_name: str) -> str:
    """官方配置相对路径：`adapters/<adapter_name>/adapter_config.json`。"""
    return "%s/%s" % (adapter_dir_relative_path(adapter_name), ADAPTER_CONFIG_FILENAME)


def discovered_adapter_name(*, weights_basename: str, directory_name: str, has_config: bool) -> str:
    """官方 `discover_adapters()` 的命名规则（E0 复现版）。

    规则（由 KAGGLE-27 六格 + E0 四格实测确定，非源码猜测）::

        有 adapter_config.json            -> directory_name
        无 config 且 stem == adapter_model -> directory_name
        否则                              -> stem

    实测依据（全部为官方 `discover_adapters()` 的真实返回）::

        adapter_model.safetensors / 无 config / adapters/v3_policy -> v3_policy
        adapter_model.safetensors / 无 config / adapters/custom_dir -> custom_dir
        adapter.safetensors       / 无 config / adapters/v3_policy -> adapter
        other.safetensors         / 无 config / adapters/v3_policy -> other
        other.safetensors         / 有 config / adapters/v3_policy -> v3_policy
    """
    stem = os.path.splitext(str(weights_basename))[0]
    if has_config or stem == DIRNAME_JUDGEMENT_STEM:
        return str(directory_name)
    return stem


def _read_config(path: str) -> dict:
    """读并解析 adapter_config.json；缺失/不可解析都按 fail-closed 抛错。"""
    if not os.path.exists(path):
        raise PolicyViolation(
            "adapter_carrier_config_missing",
            "官方 PEFT 目录里缺少 %s：缺 config 时官方 discover_adapters() 会把 adapter 名"
            "退化成本文件名 stem，声明名将解析不到（AdapterNotFoundError）" % ADAPTER_CONFIG_FILENAME,
            file=path,
        )
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        raise PolicyViolation(
            "adapter_carrier_config_unparsable",
            "%s 不是可解析的 JSON：%s" % (ADAPTER_CONFIG_FILENAME, exc),
            file=path,
        ) from exc
    if not isinstance(payload, Mapping):
        raise PolicyViolation(
            "adapter_carrier_config_not_object",
            "%s 解析结果不是 JSON 对象（实际 %s）" % (ADAPTER_CONFIG_FILENAME, type(payload).__name__),
            file=path,
        )
    return dict(payload)


def _scan_safetensors(root: str) -> list[dict]:
    """列出提交目录下全部 `.safetensors`（相对路径 + basename + 所属目录名）。"""
    found: list[dict] = []
    for current, _dirs, names in os.walk(root):
        for name in names:
            if os.path.splitext(name)[1].lower() != ".safetensors":
                continue
            full = os.path.join(current, name)
            relative = os.path.relpath(full, root).replace("\\", "/")
            parent = os.path.dirname(relative)
            found.append(
                {
                    "relative": relative,
                    "basename": name,
                    "parent": parent,
                    "parent_name": parent.split("/")[-1] if parent else "",
                    "full": full,
                }
            )
    return sorted(found, key=lambda item: item["relative"])


def assert_official_adapter_carrier(
    root: str,
    *,
    declared_adapter_name: str | None = None,
    require_config: bool = True,
) -> dict:
    """校验一个已生成的提交/导出目录是否符合官方 PEFT adapter 载体契约。

    不合规即抛 `PolicyViolation`（fail-closed），合规时返回结构化报告。
    只做静态判定：不导入 torch/peft、不加载权重、不调用官方 compiler。
    """
    if not root or not os.path.isdir(root):
        raise MissingInput("submission_dir_missing", "提交目录不存在：%r" % root)

    entries = _scan_safetensors(root)
    if not entries:
        raise PolicyViolation(
            "submission_adapter_missing",
            "提交包里没有 .safetensors：官方 adapter 载体是 %s"
            % carrier_weights_relative_path("<adapter_name>"),
            files=sorted(os.listdir(root))[:20],
        )

    legacy = [entry["relative"] for entry in entries if entry["basename"] == LEGACY_ADAPTER_FILENAME]
    if legacy:
        raise PolicyViolation(
            "adapter_carrier_legacy_name",
            "出现 E0 旧命名 %s：官方载体是 %s。旧命名在缺 %s 时会被官方"
            " discover_adapters() 退化成 stem `adapter`，声明名解析不到"
            % (LEGACY_ADAPTER_FILENAME, carrier_weights_relative_path("<adapter_name>"), ADAPTER_CONFIG_FILENAME),
            files=legacy,
            citation=MATRIX_CITATION,
        )

    wrong_basename = [
        entry["relative"] for entry in entries if entry["basename"] != ADAPTER_WEIGHTS_FILENAME
    ]
    if wrong_basename:
        raise PolicyViolation(
            "adapter_carrier_wrong_filename",
            "adapter 权重的 basename 必须是 %s：%s" % (ADAPTER_WEIGHTS_FILENAME, wrong_basename),
            files=wrong_basename,
            citation=README_CITATION,
        )

    expected_parent_of = lambda entry: "%s/%s" % (ADAPTERS_DIRNAME, entry["parent_name"])  # noqa: E731
    outside = [
        entry["relative"]
        for entry in entries
        if not entry["parent"].startswith(ADAPTERS_DIRNAME + "/")
        or entry["parent"] != expected_parent_of(entry)
    ]
    if outside:
        raise PolicyViolation(
            "adapter_carrier_not_in_adapters_dir",
            "adapter 权重必须位于 %s/<adapter_name>/ 之下：%s" % (ADAPTERS_DIRNAME, outside),
            files=outside,
            citation=README_CITATION,
        )

    details: list[dict] = []
    discovered: list[str] = []
    for entry in entries:
        config_path = os.path.join(os.path.dirname(entry["full"]), ADAPTER_CONFIG_FILENAME)
        has_config = os.path.exists(config_path)
        if require_config and not has_config:
            raise PolicyViolation(
                "adapter_carrier_config_missing",
                "官方 PEFT 目录 %s 里缺少 %s：缺 config 时官方 discover_adapters() 只能靠"
                " stem 判据保名，声明名有退化风险" % (entry["parent"], ADAPTER_CONFIG_FILENAME),
                file=entry["relative"],
                citation=README_CITATION,
            )
        config = _read_config(config_path) if has_config else None
        name = discovered_adapter_name(
            weights_basename=entry["basename"],
            directory_name=entry["parent_name"],
            has_config=has_config,
        )
        discovered.append(name)
        details.append(
            {
                "relative": entry["relative"],
                "adapter_dir": entry["parent"],
                "directory_name": entry["parent_name"],
                "has_adapter_config_json": has_config,
                "adapter_config_parsable": bool(has_config),
                "adapter_config_keys": sorted(config.keys()) if config else [],
                "discovered_adapter_name": name,
            }
        )

    if declared_adapter_name is not None:
        declared = str(declared_adapter_name).strip()
        if declared and declared not in discovered:
            raise PolicyViolation(
                "adapter_carrier_declared_name_unresolved",
                "声明名 %r 无法被官方命名规则解析（会抛 AdapterNotFoundError）：发现到 %s"
                % (declared_adapter_name, sorted(discovered)),
                declared=declared_adapter_name,
                discovered=sorted(discovered),
                citation=MATRIX_CITATION,
            )

    return {
        "ok": True,
        "carrier": "official-peft-directory",
        "weights_filename": ADAPTER_WEIGHTS_FILENAME,
        "config_filename": ADAPTER_CONFIG_FILENAME,
        "adapters": details,
        "discovered_adapter_names": sorted(discovered),
        "declared_adapter_name": (
            None if declared_adapter_name is None else str(declared_adapter_name).strip()
        ),
        "config_required": bool(require_config),
        "static_contract_only": True,
        "real_adapter_loading_verified": False,
        "citations": [README_CITATION, MATRIX_CITATION, RULE_CELLS_CITATION],
    }


#: 冻结的官方六格矩阵（KAGGLE-27 S1）在仓库内的路径。
S1_MATRIX_RELATIVE_PATH = os.path.join(
    "docs", "v3", "design", "kaggle-27-s1-adapter-matrix.json"
)
#: E0 补的四格在仓库内的路径。
RULE_CELLS_RELATIVE_PATH = os.path.join(
    "docs", "v3", "design", "kaggle-27-s1b-discovery-rule-cells.json"
)

#: 六格矩阵里的期望值（本模块独立抄一份，用于交叉核对冻结产物没被改动）。
EXPECTED_S1_CELLS: dict[str, tuple[str, bool, str]] = {
    # tag -> (filename, has_config, discovered_name)
    "official_name__no_config": ("adapter_model.safetensors", False, "v3_policy"),
    "official_name__with_config": ("adapter_model.safetensors", True, "v3_policy"),
    "e0_name__with_config": ("adapter.safetensors", True, "v3_policy"),
    "e0_name__no_config": ("adapter.safetensors", False, "adapter"),
    "custom_dirname__with_config": ("v3_policy.safetensors", True, "v3_policy"),
    "custom_dirname__no_config": ("v3_policy.safetensors", False, "v3_policy"),
}


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_rule_against_frozen_matrix(root: str | None = None) -> dict:
    """用本模块实现的命名规则重算冻结的官方矩阵，逐格比对。

    冻结产物（S1 六格 + E0 四格）必须**完好**且**与规则一致**；
    任何一格不符即 `IntegrityError`（规则被改坏，或冻结证据被改动）。
    """
    from ..common.errors import IntegrityError

    base = root or _repo_root()
    report: dict = {"checked": [], "problems": []}

    s1_path = os.path.join(base, S1_MATRIX_RELATIVE_PATH)
    if not os.path.exists(s1_path):
        raise MissingInput("s1_matrix_missing", "缺少官方命名矩阵冻结副本：%s" % s1_path)
    with open(s1_path, "rb") as handle:
        s1 = json.loads(handle.read().decode("utf-8-sig"))
    cells = s1.get("cells") or {}
    for tag, (filename, has_config, expected) in EXPECTED_S1_CELLS.items():
        if tag not in cells:
            report["problems"].append("S1 缺少格 %s" % tag)
            continue
        cell = cells[tag]
        if cell.get("filename") != filename or bool(cell.get("has_adapter_config_json")) != has_config:
            report["problems"].append("S1 格 %s 的 fixture 描述与期望不符" % tag)
            continue
        names = sorted(cell.get("discovered_adapter_names") or [])
        dirname = str(s1.get("adapter_dir") or "adapters/v3_policy").replace("\\", "/").split("/")[-1]
        recomputed = discovered_adapter_name(
            weights_basename=filename,
            directory_name=dirname,
            has_config=has_config,
        )
        if names != [expected]:
            report["problems"].append(
                "S1 格 %s 的官方发现名 %s 与期望 %s 不符" % (tag, names, [expected])
            )
            continue
        if recomputed != expected:
            report["problems"].append(
                "本模块规则在 S1 格 %s 上算出 %r，期望 %r" % (tag, recomputed, expected)
            )
            continue
        report["checked"].append({"source": "s1", "tag": tag, "name": expected})

    rule_path = os.path.join(base, RULE_CELLS_RELATIVE_PATH)
    if not os.path.exists(rule_path):
        raise MissingInput("rule_cells_missing", "缺少 E0 补格冻结副本：%s" % rule_path)
    with open(rule_path, "rb") as handle:
        extra = json.loads(handle.read().decode("utf-8-sig"))
    for cell in extra.get("cases") or []:
        filename = str(cell.get("filename"))
        has_config = bool(cell.get("has_adapter_config_json"))
        directory_name = str(cell.get("adapter_dir_name"))
        names = sorted(cell.get("discovered_adapter_names") or [])
        recomputed = discovered_adapter_name(
            weights_basename=filename, directory_name=directory_name, has_config=has_config
        )
        if names != [recomputed]:
            report["problems"].append(
                "E0 补格 %s/%s 官方发现名 %s 与规则算出的 %r 不符"
                % (directory_name, filename, names, recomputed)
            )
            continue
        report["checked"].append(
            {"source": "e0-supplement", "tag": "%s/%s" % (directory_name, filename), "name": recomputed}
        )

    if report["problems"]:
        raise IntegrityError(
            "adapter_naming_rule_mismatch",
            "命名规则与冻结证据不一致：%s" % "；".join(report["problems"]),
            problems=report["problems"],
        )
    report["ok"] = True
    report["cell_count"] = len(report["checked"])
    return report


def assert_adapter_config_present(directory: str) -> dict:
    """单独断言某个 PEFT 目录里有可解析的 `adapter_config.json`（供训练导出路径复用）。"""
    config_path = os.path.join(directory, ADAPTER_CONFIG_FILENAME)
    payload = _read_config(config_path)
    return {
        "path": config_path,
        "parsable": True,
        "keys": sorted(payload.keys()),
    }
