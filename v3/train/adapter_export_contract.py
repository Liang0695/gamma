"""G8：adapter 导出清单的**版本化契约**（生产与消费用同一份 schema）。

## 缺陷出处

KAGGLE-32 v6 `GAPS-AND-OWNERS.md` §G8 与 `EVIDENCE-static-pins.md` §A.7.3：

- 生产者 `checkpoint.build_export_manifest()`（`v3/train/checkpoint.py:409-455`）产出
  **嵌套** `training_hashes`，且**没有** `adapter_only` / `files`；
- 消费者 `entry.measure_gates()`（`v3/train/entry.py:205-241`）要的是**扁平**结构，
  且从清单里自报的 `load_ok` / `params_changed` / `fixture_pass` / `fixture_total`
  取值（`entry.py:234-238` 全是 `export_manifest.get(...)`）。

两处 schema 不一致，且"通过"可以由手写 JSON 拼出来 —— 这是本模块要关掉的口子。

## 本模块的契约（`v3-adapter-export/2`）

生产侧 :func:`build_export_manifest`：

1. **只认实际产物**：遍历 `export_dir`，逐文件算 `sha256` 与字节数，写进 `files`；
2. 复用 `v3.submit.adapter_contract.assert_official_adapter_carrier()` 校验官方 PEFT 载体
   （`adapters/<name>/adapter_model.safetensors` + 同目录可解析 `adapter_config.json`）；
3. 哈希键**扁平**放在顶层（`source_sha256` / `data_sha256` / `config_sha256` /
   `code_sha256` / `deps_sha256`），不再嵌套 `training_hashes`；
4. 产出 `artifact_digest` = 对文件清单做规范 JSON 的 SHA-256：消费侧据此判"清单与产物同源"。

消费侧 :func:`validate_export_manifest`：

1. `format` 必须是本模块支持的版本；`v3-adapter-export/1` **一律拒绝**
   （`export_manifest_legacy_schema`），消息里指出改用 v2 生产者；
2. 出现自报通过字段（`load_ok` / `params_changed` / `fixture_pass` / `fixture_total` /
   `adapter_valid` / `frozen_spec_satisfied`）即拒绝
   （`export_manifest_self_reported_fields`）—— 通过必须来自**物**，不是来自清单里的一句话；
3. 必须绑定 `export_dir`，并**逐个重算**文件哈希与 `artifact_digest`
   （`export_manifest_file_mismatch` / `export_manifest_artifact_digest_mismatch`）；
4. 官方载体契约**重新跑一遍**（不是复用清单里的 `carrier` 段）；
5. 出现 P0 **数据**导出的痕迹（`windows_sha256` / `window_count` / `window_tokens` /
   `export_config`）即拒绝（`export_manifest_kind_mismatch`）——
   adapter export 与 P0 数据 export 是两件事，不得互相冒充。

诚实边界：本模块只做**静态 + 字节级**校验，不导入 torch/peft、不加载权重。
`real_adapter_loading_verified` 恒为 `False`；真实加载证据必须来自获批 GPU 上的
工程检查（`v3/train/engineering_check.py`）。
"""

from __future__ import annotations

import hashlib
import os
from typing import Mapping

from ..common.canonical import canonical_json_bytes, sha256_bytes
from ..common.errors import FailClosed, MissingInput, PolicyViolation
from ..submit import adapter_contract
from .checkpoint import REQUIRED_HASH_KEYS

#: 历史（E0 上一版）格式：嵌套 `training_hashes`、无 `files` / `adapter_only`。
FORMAT_V1 = "v3-adapter-export/1"
#: 本模块定义的格式：扁平哈希键 + `files` + `artifact_digest`。
FORMAT_V2 = "v3-adapter-export/2"
#: 生产侧写出的格式（唯一）。
PRODUCTION_FORMAT = FORMAT_V2
#: 消费侧接受的格式（**只有** v2：v1 的缺陷正是本轮要关掉的东西）。
SUPPORTED_FORMATS = (FORMAT_V2,)

#: 禁止出现在 adapter 导出清单里的"自报通过"字段。
SELF_REPORT_KEYS = (
    "load_ok",
    "params_changed",
    "fixture_pass",
    "fixture_total",
    "adapter_valid",
    "frozen_spec_satisfied",
    "adapter_only",  # 允许！见下方白名单说明
)
#: 上表里真正禁止的字段（`adapter_only` 是**载体事实声明**，允许存在，但必须与
#: `contains_base_weights=false` 同时出现并由消费侧核对）。
FORBIDDEN_SELF_REPORT_KEYS = tuple(key for key in SELF_REPORT_KEYS if key != "adapter_only")

#: P0 数据导出清单的特征键：出现即说明"拿错清单了"。
DATA_EXPORT_MARKER_KEYS = ("windows_sha256", "window_count", "window_tokens", "export_config")

#: 非 safetensors 的权重后缀（出现即拒绝）。
FORBIDDEN_WEIGHT_SUFFIXES = (".bin", ".pt", ".ckpt", ".pth")

#: 清单里 `files` 条目的必需键。
FILE_ENTRY_KEYS = ("path", "sha256", "bytes")

#: 校验阶段。**训练阶段与 serving 阶段必须分开判**：
#: `vllm_mapper_revision` 是**目标 serving 环境**的软件事实（本轮锁里仍是 `null`），
#: 把它当训练导出的生产前置条件，会让一个只能在评分宿主上观测的量去阻塞训练入口。
#: 生产者 :func:`build_export_manifest` 与消费者 :func:`collect_export_manifest_problems`
#: 的**默认**阶段仍是 `serving`（保持既有 fail-closed 行为不变）；
#: 训练工程检查显式传 `stage=STAGE_TRAINING`，并把 `serving_pins_verified: false`
#: 原样写进清单 —— **不把 `null` 当通过**，只是把两阶段的判据分开。
STAGE_TRAINING = "training"
STAGE_SERVING = "serving"
STAGES = (STAGE_TRAINING, STAGE_SERVING)


def _normalize_stage(stage) -> str:
    value = str(stage or STAGE_SERVING).strip().lower()
    if value not in STAGES:
        raise PolicyViolation(
            "export_manifest_unknown_stage",
            "未知的导出清单校验阶段 %r：只支持 %s" % (stage, list(STAGES)),
        )
    return value


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure_export_dir(export_dir: str) -> list[dict]:
    """遍历导出目录，逐文件实测 `(相对路径, sha256, 字节数)`（按路径排序）。"""
    if not export_dir or not os.path.isdir(export_dir):
        raise MissingInput("export_dir_missing", "导出目录不存在：%r" % (export_dir,))
    measured: list[dict] = []
    for current, dirs, names in os.walk(export_dir):
        dirs.sort()
        for name in sorted(names):
            full = os.path.join(current, name)
            relative = os.path.relpath(full, export_dir).replace("\\", "/")
            measured.append(
                {
                    "path": relative,
                    "sha256": _sha256_file(full),
                    "bytes": os.path.getsize(full),
                }
            )
    return sorted(measured, key=lambda item: item["path"])


def artifact_digest(files: list[Mapping]) -> str:
    """文件清单的规范摘要：消费侧据此判定"清单描述的正是这批字节"。"""
    normalized = [
        {
            "path": str(item.get("path")),
            "sha256": str(item.get("sha256")),
            "bytes": int(item.get("bytes", 0)),
        }
        for item in files
    ]
    return sha256_bytes(canonical_json_bytes(sorted(normalized, key=lambda item: item["path"])))


def build_export_manifest(
    export_dir: str,
    *,
    adapter_name: str,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules_regex: str,
    matched_module_count: int,
    base_repo_id: str,
    base_revision: str,
    vllm_mapper_revision: str | None,
    training_hashes: Mapping,
    known_ranking: Mapping | None = None,
    produced_by: str = "v3.train.adapter_export_contract",
    stage: str = STAGE_SERVING,
    serving_pin_verified: bool = False,
) -> dict:
    """生产侧：从**实际产物目录**产出 `v3-adapter-export/2` 清单。

    fail-closed 点：缺哈希键、缺 mapper revision（**serving 阶段**）、模块数为 0、
    载体不符官方 PEFT 契约、出现非 safetensors 权重、目录为空 —— 任一即抛异常，
    不返回半成品清单。

    分阶段校验（本轮整改）：`stage=STAGE_TRAINING` 时 `vllm_mapper_revision` 允许为
    `None`，但清单里会**显式写出** `base.vllm_mapper_revision = None` 与
    `base.serving_pins_verified = false`，并且消费侧在训练阶段会把
    `serving_stage_validation_required` 置真 —— `null` 不会被当作通过。
    """
    stage = _normalize_stage(stage)
    missing = [key for key in REQUIRED_HASH_KEYS if key not in (training_hashes or {})]
    if missing:
        raise MissingInput("export_missing_hashes", "导出清单缺少哈希", missing=missing)
    if not vllm_mapper_revision:
        if stage == STAGE_SERVING:
            raise MissingInput(
                "mapper_revision_unknown",
                "vLLM 实际 mapper revision 未知：serving 阶段不得声称加载兼容（fail-closed）",
            )
        serving_pin_verified = False
    if int(matched_module_count) <= 0:
        raise PolicyViolation("export_no_modules", "target 模块数为 0，无效 adapter 不得通过")

    name = adapter_contract.adapter_dir_relative_path(adapter_name).split("/")[-1]
    try:
        carrier = adapter_contract.assert_official_adapter_carrier(
            export_dir, declared_adapter_name=name
        )
    except FailClosed as exc:
        raise PolicyViolation(
            "export_carrier_contract_failed",
            "导出目录未通过官方 PEFT 载体契约：%s（%s）" % (exc.code, exc.message),
            detail=exc.to_dict(),
        ) from exc

    files = measure_export_dir(export_dir)
    if not files:
        raise PolicyViolation("export_dir_empty", "导出目录里没有任何文件：空清单不是证据")
    bad_weights = [
        item["path"]
        for item in files
        if item["path"].lower().endswith(FORBIDDEN_WEIGHT_SUFFIXES)
    ]
    if bad_weights:
        raise PolicyViolation(
            "export_manifest_forbidden_weight_format",
            "导出目录里出现非 .safetensors 权重：%s" % bad_weights,
            files=bad_weights,
        )

    carrier_weights = adapter_contract.carrier_weights_relative_path(name)
    carrier_config = adapter_contract.carrier_config_relative_path(name)
    paths = {item["path"] for item in files}
    if carrier_weights not in paths:
        raise PolicyViolation(
            "export_manifest_carrier_missing",
            "导出目录里没有官方载体路径 %s（实际：%s）" % (carrier_weights, sorted(paths)),
            files=sorted(paths),
        )
    if carrier_config not in paths:
        raise PolicyViolation(
            "export_manifest_config_missing",
            "导出目录里没有 %s：官方 PEFT 目录缺 config 时 adapter 名会退化"
            % carrier_config,
            files=sorted(paths),
        )

    config_item = next(item for item in files if item["path"] == carrier_config)
    return {
        "format": PRODUCTION_FORMAT,
        "adapter_name": name,
        "adapter_only": True,
        "contains_base_weights": False,
        "files": files,
        "artifact_digest": artifact_digest(files),
        "file_count": len(files),
        # 哈希键**扁平**在顶层：与消费侧同源，不再嵌套 training_hashes。
        **{key: training_hashes[key] for key in REQUIRED_HASH_KEYS},
        "lora": {
            "r": int(rank),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "bias": "none",
            "use_dora": False,
            "use_rslora": False,
            "target_modules_regex": target_modules_regex,
            "matched_module_count": int(matched_module_count),
        },
        "base": {
            "repo_id": base_repo_id,
            "revision": base_revision,
            "vllm_mapper_revision": vllm_mapper_revision,
            "serving_pins_verified": bool(serving_pin_verified and vllm_mapper_revision),
            "serving_pins_note": (
                "serving 阶段的 mapper revision 已验证"
                if (serving_pin_verified and vllm_mapper_revision)
                else "serving 阶段的 mapper revision **未验证**（本轮为 null）："
                "本清单只通过训练阶段校验，serving 阶段必须单独判"
            ),
        },
        "validation_stage": stage,
        "carrier": {
            "kind": "official-peft-directory",
            "weights_relative_path": carrier_weights,
            "config_relative_path": carrier_config,
            "config_sha256": config_item["sha256"],
            "discovered_adapter_names": carrier.get("discovered_adapter_names"),
            "citation": "HARNESS_README.md §3.4 / §2（adapters/<adapter_name>/ + adapter_config.json）",
        },
        "measurement": {
            "probe": "filesystem-walk + sha256",
            "measured_by": produced_by,
            "measured_from": "export_dir-bytes（不是清单自报）",
            "is_real_measurement": True,
            "note": "这是字节级实测；**不**包含真实加载/前向证据。",
        },
        "known_ranking": dict(known_ranking or {}),
        "note": (
            "本清单只声明载体与字节事实；加载成功、参数变化、fixture 计数一律不在此处自报，"
            "由获批工程检查另行产出。"
        ),
    }


def collect_export_manifest_problems(
    manifest: Mapping | None,
    *,
    export_dir: str | None = None,
    expected_adapter_name: str | None = None,
    require_artifacts: bool = True,
    stage: str = STAGE_SERVING,
) -> list[str]:
    """消费侧：列出全部问题（不抛异常），供 `measure_gates` 汇总成闸门理由。

    `stage` 决定 `vllm_mapper_revision` 这一 **serving 侧事实**是否算问题：

    - `STAGE_SERVING`（默认，保持既有行为）：必须非空**且** `serving_pins_verified=true`；
    - `STAGE_TRAINING`：允许为 `null`（该值只能在评分宿主上观测，训练入口观测不到），
      但不允许"填了一个未验证的值"冒充已验证。
    """
    stage = _normalize_stage(stage)
    problems: list[str] = []
    if not isinstance(manifest, Mapping) or not manifest:
        return ["没有提供导出清单：必须先有一份可校验的 adapter 导出清单"]

    for key in FORBIDDEN_SELF_REPORT_KEYS:
        if key in manifest:
            problems.append(
                "清单里出现自报通过字段 %r：通过必须来自产物实测，不得由清单自述" % key
            )
    for key in DATA_EXPORT_MARKER_KEYS:
        if key in manifest:
            problems.append(
                "清单里出现 P0 数据导出的特征键 %r：adapter export 与数据 export 不得混用" % key
            )
            break

    fmt = manifest.get("format")
    if fmt == FORMAT_V1:
        problems.append(
            "清单是旧格式 %s（嵌套 training_hashes、无 files/adapter_only）："
            "请用 v3.train.adapter_export_contract.build_export_manifest() 重新产出 %s"
            % (FORMAT_V1, FORMAT_V2)
        )
    elif fmt not in SUPPORTED_FORMATS:
        problems.append(
            "不支持的导出清单格式 %r：支持 %s" % (fmt, list(SUPPORTED_FORMATS))
        )

    missing = [key for key in REQUIRED_HASH_KEYS if not manifest.get(key)]
    if missing:
        problems.append("导出清单缺少扁平哈希键：%s" % ", ".join(sorted(missing)))
    if manifest.get("training_hashes") is not None:
        problems.append("导出清单仍使用嵌套 training_hashes：v2 契约要求扁平哈希键")

    if manifest.get("adapter_only") is not True:
        problems.append("导出清单未声明 adapter_only=true（禁止整权重当 adapter）")
    if manifest.get("contains_base_weights") is not False:
        problems.append("导出清单未声明 contains_base_weights=false（adapter-only 契约）")

    # —— 分阶段校验：训练侧不得被 serving 侧不可观测量阻塞，也不得用 null 冒充通过 ——
    base = manifest.get("base")
    base = base if isinstance(base, Mapping) else {}
    mapper = base.get("vllm_mapper_revision")
    serving_verified = base.get("serving_pins_verified") is True
    if stage == STAGE_SERVING:
        if not mapper:
            problems.append(
                "serving 阶段校验要求 base.vllm_mapper_revision 非空（当前 %r）："
                "vLLM mapper 版本未知时不得声称加载兼容" % (mapper,)
            )
    elif mapper and not serving_verified:
        problems.append(
            "训练阶段清单写了 base.vllm_mapper_revision=%r 但没有 serving_pins_verified=true："
            "不得拿未验证的值冒充已验证" % (mapper,)
        )

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        problems.append("导出清单没有任何文件条目")
        return problems

    entry_problems: list[str] = []
    for index, item in enumerate(files):
        if not isinstance(item, Mapping):
            entry_problems.append("files[%d] 不是对象" % index)
            continue
        absent = [key for key in FILE_ENTRY_KEYS if key not in item]
        if absent:
            entry_problems.append("files[%d] 缺键 %s" % (index, absent))
    if entry_problems:
        problems.extend(entry_problems)
        return problems

    bad_weights = [
        str(item["path"])
        for item in files
        if str(item["path"]).lower().endswith(FORBIDDEN_WEIGHT_SUFFIXES)
    ]
    if bad_weights:
        problems.append("导出清单里出现非 .safetensors 权重文件：%s" % bad_weights)

    name = str(expected_adapter_name or manifest.get("adapter_name") or adapter_contract.DEFAULT_ADAPTER_NAME)
    try:
        expected_carrier = adapter_contract.carrier_weights_relative_path(name)
    except FailClosed as exc:
        problems.append("adapter 名非法：%s（%s）" % (name, exc.code))
        return problems
    normalized = {str(item["path"]).replace("\\", "/") for item in files}
    if expected_carrier not in normalized:
        problems.append(
            "导出清单缺少官方 PEFT 载体路径 %s（现有：%s）" % (expected_carrier, sorted(normalized))
        )

    # —— 与产物对账（本模块的核心：清单不得脱离字节）——
    if not export_dir:
        if require_artifacts:
            problems.append(
                "没有绑定实际产物目录（--export-dir）：清单必须能与字节对账，"
                "否则退回「照清单自述即通过」的口子"
            )
        return problems

    if not os.path.isdir(export_dir):
        problems.append("绑定的导出目录不存在：%s" % export_dir)
        return problems

    try:
        measured = measure_export_dir(export_dir)
    except FailClosed as exc:
        problems.append("无法实测导出目录：%s（%s）" % (exc.code, exc.message))
        return problems

    measured_map = {item["path"]: item for item in measured}
    for item in files:
        path = str(item["path"]).replace("\\", "/")
        actual = measured_map.get(path)
        if actual is None:
            problems.append("清单声明的文件在产物目录里不存在：%s" % path)
            continue
        if actual["sha256"] != str(item["sha256"]):
            problems.append(
                "文件哈希与产物不符：%s（清单 %s，实测 %s）"
                % (path, str(item["sha256"])[:16], actual["sha256"][:16])
            )
        if int(item.get("bytes", -1)) != int(actual["bytes"]):
            problems.append(
                "文件字节数与产物不符：%s（清单 %s，实测 %s）"
                % (path, item.get("bytes"), actual["bytes"])
            )
    extra = sorted(set(measured_map) - {str(item["path"]).replace("\\", "/") for item in files})
    if extra:
        problems.append("产物目录里存在清单未登记的文件：%s" % extra)

    declared_digest = manifest.get("artifact_digest")
    recomputed = artifact_digest([dict(item) for item in files])
    if declared_digest != recomputed:
        problems.append(
            "artifact_digest 与清单内容不符（清单 %r，重算 %s）"
            % (declared_digest, recomputed[:16])
        )

    try:
        adapter_contract.assert_official_adapter_carrier(
            export_dir, declared_adapter_name=name
        )
    except FailClosed as exc:
        problems.append(
            "官方载体契约复核未过（消费侧重跑，不复用清单里的 carrier 段）：%s（%s）"
            % (exc.code, exc.message)
        )
    return problems


def validate_export_manifest(
    manifest: Mapping | None,
    *,
    export_dir: str | None = None,
    expected_adapter_name: str | None = None,
    require_artifacts: bool = True,
    stage: str = STAGE_SERVING,
) -> dict:
    """消费侧严格入口：有问题即 `PolicyViolation("export_manifest_invalid")`。"""
    stage = _normalize_stage(stage)
    problems = collect_export_manifest_problems(
        manifest,
        export_dir=export_dir,
        expected_adapter_name=expected_adapter_name,
        require_artifacts=require_artifacts,
        stage=stage,
    )
    if problems:
        raise PolicyViolation(
            "export_manifest_invalid",
            "adapter 导出清单未通过 v2 契约（%d 项）：%s" % (len(problems), problems[0]),
            problems=problems,
            stage=stage,
        )
    files = list(manifest.get("files") or [])
    base = manifest.get("base") if isinstance(manifest.get("base"), Mapping) else {}
    serving_verified = bool(base.get("serving_pins_verified")) and bool(
        base.get("vllm_mapper_revision")
    )
    return {
        "ok": True,
        "stage": stage,
        "format": manifest.get("format"),
        "adapter_name": manifest.get("adapter_name"),
        "file_count": len(files),
        "artifact_digest": manifest.get("artifact_digest"),
        "verified_from_artifacts": True,
        "artifacts_dir": os.path.realpath(export_dir) if export_dir else None,
        "hash_keys": {key: manifest.get(key) for key in REQUIRED_HASH_KEYS},
        "self_reported_fields_used": [],
        "static_contract_only": True,
        "real_adapter_loading_verified": False,
        "serving_pins_verified": serving_verified,
        "serving_stage_validation_required": not serving_verified,
        "note": (
            "清单与产物字节已对账；装得下/对得上**不等于**能训练、也不等于评分端兼容。"
            + (
                ""
                if serving_verified
                else "本清单只表示**训练阶段**的载体与字节事实通过；"
                "serving 阶段必须用 stage='serving' 单独校验（当前 mapper revision 未验证）。"
            )
        ),
    }
