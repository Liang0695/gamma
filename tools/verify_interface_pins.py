"""G9：官方接口锁里未验证 pin 的**取证核对**工具（不下载大模型、不自行放行）。

## 缺陷出处

KAGGLE-32 v6 `GAPS-AND-OWNERS.md` §G9：

    `interface_pins_unverified`：`tokenizer_config_sha256` / `tokenizer_vocab_sha256` /
    `vllm_mapper_revision` 三个 pin 未验证（`docs/v3/evidence/train-preflight.json`）

当前状态（`v3/locks/official-interface.json`）：

| pin | 现值 | verified |
|---|---|---|
| `tokenizer_config_sha256` | `b8045a45…394b` | false |
| `tokenizer_vocab_sha256` | `null` | false |
| `chat_template_sha256` | `ae53464b…c6d4` | false |
| `vllm_mapper_revision` | `null` | false |

四个 pin 是**互相独立**的：各自只能由自己的证据文件字节判定，
`tokenizer_config.json` 的哈希**绝不**可以用来满足 `chat_template_sha256`（Mika 裁定）。

## 判定词表（六种，互不折叠）

旧版把"未锁定""不可达""确实没装"都塞进 `unavailable` / `missing_evidence`，
导致读者分不清"本机查不了"和"锁里本来就没值"。现在一律用下面六个常量：

| verdict | 何时出现 |
|---|---|
| `match` | 证据测得值与锁里声明的值逐位一致 |
| `mismatch` | 测得值与锁里声明不一致（不得把 `verified` 置真） |
| `not_locked` | 锁里该 pin 仍是 `null`（**尚未锁定**）：本次只产出候选值 |
| `unreachable` | 端点/证据**不可达或取回失败**（网络超时、目录不存在、manifest 声明取回失败） |
| `probed_not_installed` | 已探测环境**全部**报告 `vllm_installed=false`，且没有评分宿主版本证据 |
| `missing_evidence` | 证据齐备但**不含该事实**（缺文件、缺字段、元数据 API 不返回 SHA-256） |

`not_locked` 需要"有测得值"才会出现：没有值又没有"未安装"结论时如实报 `missing_evidence`。
兼容旧名：`UNAVAILABLE` 仍然可导入，等价于 `UNREACHABLE`（语义已收窄，见该常量处的说明）。

## 端点可参数化（评审 Mika 的 1 号要求）

`--endpoint <url-template>` 指定元数据端点，模板支持两种占位符写法：

    {repo} / {revision} 具名：https://hf-mirror.com/api/models/{repo}/revision/{revision}
    {} 位置（repo、revision 顺序）：https://hf-mirror.com/api/models/{}/revision/{}

默认仍是 `https://huggingface.co/api/models/{repo}/revision/{revision}`；报告里恒定记录
`endpoint.template` / `endpoint.source` / `endpoint.is_mirror` / `endpoint_reachable` 与锁定 revision，
所以"这台机器够不着"永远显示为 `unreachable`，而不会被误读成"这个 pin 查不了"。

## 镜像只算自洽校验（评审 Mika 的 4 号要求）

镜像（`hf-mirror.com`）上的文件字节与**同一镜像自己**的元数据/清单一致，
只能证明镜像内部自洽；它**不是**对官方源（`huggingface.co`）的独立认证。
凡出现这类比较，报告都会写 `consistency_check_only: true` 并附上同义措辞。

## 这个工具做什么

1. `--evidence-dir` 指向在**具备访问条件的执行环境**（107 / 有 HF 出口的机器）上取回的
   证据目录，其中：
   - `evidence-manifest.json`：必须写明 `model_repo_id` / `model_revision` /
     `retrieved_at_utc` / `retrieval_command` / `source`（下载来源）；
   - `tokenizer_config.json`：按锁定 revision 取回的 tokenizer 配置；
   - `tokenizer.json`（或 `vocab_file`）：词表/分词器数据；
   - `chat_template.jinja`：按锁定 revision 取回的聊天模板（对应 `chat_template_sha256`，独立 pin）；
   - `env-report.json`：逐环境记录 `vllm` 是否安装（见下）。
   工具会**重算 SHA-256**，与锁里声明的值逐位比对；锁里是 `null` 的项只报"候选值"。
2. `--metadata-only`：只查 HF **元数据**（`/api/models/<repo>/revision/<revision>`），
   不发文件下载。取不到就报 `unreachable` 并写清端点与原因。

### `env-report.json` 的两种形状

新形状（G9 要求，用于判定 `vllm_mapper_revision`）：

    {
      "probed_environments": [
        {"name": "107-h3", "vllm_installed": false},
        {"name": "local-win", "vllm_installed": false}
      ],
      "scoring_host_version_evidence": false,
      "vllm_public_source_revision": "vllm@<git-rev>"
    }

- 所有探测环境 `vllm_installed=false` 且 `scoring_host_version_evidence` 为假 →
  `probed_not_installed`，理由措辞恒为
  "已探测的环境未安装 vllm；本轮未取得评分宿主版本证据"（只陈述本轮范围，不宣称永远不可观测）；
- `vllm_public_source_revision` 一类**公开源码 revision 参考**只放进
  `public_source_reference`（`is_a_pin: false`），**永不**当作 `measured` 实测值；
- 传统形状（顶层 `vllm_mapper_revision` / `vllm_version`）继续支持，作为实测值使用。

## 这个工具**不做**什么

- **不下载权重**（只读元数据；`--evidence-dir` 模式读的是别人取回的小文件）；
- **不改写** `official-interface.json`，**不把** `verified` 置真；报告恒定含
  `lock_modified: false` / `verified_flipped: false`，只给出 `proposed_pin_patch` 与
  `required_review`；最终翻 `verified` 需 107 现场取证 + 独立审查（KAGGLE-26/Q0 口径）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: 需要核对的 pin（顺序即报告顺序）。
TARGET_PINS = (
    "tokenizer_config_sha256",
    "tokenizer_vocab_sha256",
    "chat_template_sha256",
    "vllm_mapper_revision",
)

#: 证据目录里各类文件的候选文件名（取第一个存在的）。
TOKENIZER_CONFIG_NAMES = ("tokenizer_config.json",)
TOKENIZER_DATA_NAMES = ("tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model")
#: 聊天模板是**独立 pin**：绝不允许用 config/vocab 的哈希顶替它（Mika 裁定）。
TOKENIZER_TEMPLATE_NAMES = ("chat_template.jinja",)

#: 三个 SHA pin ↔ 各自的证据文件候选名（顺序即报告顺序）。
SHA_PIN_FILES = {
    "tokenizer_config_sha256": TOKENIZER_CONFIG_NAMES,
    "tokenizer_vocab_sha256": TOKENIZER_DATA_NAMES,
    "chat_template_sha256": TOKENIZER_TEMPLATE_NAMES,
}

#: 默认端点：官方 HF。本机实测 TCP 超时，但**默认值不因此改写**，要用镜像请显式传 --endpoint。
DEFAULT_ENDPOINT = "https://huggingface.co/api/models/{repo}/revision/{revision}"

#: 镜像端点：同一 API 路径，评审在**本机**实测可用（见 LOCAL_REACHABILITY_NOTE）。
MIRROR_ENDPOINT = "https://hf-mirror.com/api/models/{repo}/revision/{revision}"

#: 已知端点 → 报告里的来源标签（其余模板一律记 `cli-override`）。
KNOWN_ENDPOINT_SOURCES = {
    DEFAULT_ENDPOINT: "default-huggingface",
    MIRROR_ENDPOINT: "hf-mirror",
}

#: 判定"这是镜像来源"的字符串标记（端点模板与证据 manifest 的 source 都用它判断）。
MIRROR_HOST_MARKERS = ("hf-mirror.com", "mirror")

# ---- 判定常量（六种，互不折叠）------------------------------------------------
MATCH = "match"
MISMATCH = "mismatch"
NOT_LOCKED = "not_locked"
UNREACHABLE = "unreachable"
PROBED_NOT_INSTALLED = "probed_not_installed"
MISSING_EVIDENCE = "missing_evidence"

#: 全部判定值（供 `verdict_counts` 与测试遍历）。
ALL_VERDICTS = (MATCH, MISMATCH, NOT_LOCKED, UNREACHABLE, PROBED_NOT_INSTALLED, MISSING_EVIDENCE)

#: 兼容旧名：旧版把"未锁定"和"不可达"合称 `unavailable`。
#: 现在语义收窄为**不可达**；旧读者若用它判 `null` pin，请改用 `NOT_LOCKED`。
UNAVAILABLE = UNREACHABLE

#: `probed_not_installed` 的固定理由措辞（评审指定，不得改写为"永远不可观测"）。
VLLM_PROBED_NOT_INSTALLED_REASON = "已探测的环境未安装 vllm；本轮未取得评分宿主版本证据"

#: 镜像自洽校验的固定措辞（必须在任何镜像比较出现的地方写出来）。
MIRROR_CONSISTENCY_STATEMENT = (
    "该文件字节与**同一镜像自己**的元数据/清单一致，只构成镜像自洽校验"
    "（consistency check only），不构成对官方源 huggingface.co 的独立认证。"
)

#: 本机 HF 可达性结论（写进报告，避免读者误以为"没查"）。
LOCAL_REACHABILITY_NOTE = (
    "本机（Windows 工作区）实测：huggingface.co TCP 超时；同一 API 路径在 hf-mirror.com 可用。"
    "默认端点仍是 huggingface.co，需要走镜像请显式传 --endpoint；"
    "注意镜像结果只是镜像自洽校验，不等于官方源独立认证。"
)


def sha256_file(path: str) -> str:
    """对文件字节重算 SHA-256（本工具唯一的"测量"动作）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(directory: str, names) -> str | None:
    for name in names:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


def load_lock(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---- 端点工具 -----------------------------------------------------------------


def resolve_endpoint(template: str, repo: str, revision: str) -> str:
    """把端点模板里的占位符替换成实际 `repo` / `revision`。

    支持两种写法：

    - 具名：`{repo}` / `{revision}`，例如
      `https://hf-mirror.com/api/models/{repo}/revision/{revision}`
    - 位置：`{}`，按 repo、revision 顺序替换，例如
      `https://hf-mirror.com/api/models/{}/revision/{}`

    两种都没有时抛 `ValueError`（CLI 会以退出码 2 报错，不静默拼出一个坏 URL）。
    """
    text = str(template)
    if "{repo}" in text or "{revision}" in text:
        return text.format(repo=repo, revision=revision)
    if "{}" in text:
        return text.format(repo, revision)
    raise ValueError(
        "端点模板必须含 {repo}/{revision} 或两个 {} 占位符，实际收到：%r" % template
    )


def describe_endpoint(template: str) -> tuple[str, bool]:
    """返回 `(来源标签, 是否是镜像端点)`。"""
    label = KNOWN_ENDPOINT_SOURCES.get(str(template), "cli-override")
    lowered = str(template).lower()
    is_mirror = any(marker in lowered for marker in MIRROR_HOST_MARKERS)
    return label, is_mirror


def _is_mirror_source(source) -> bool:
    """证据 manifest 的 `source` 是否指向镜像（含 `source_is_mirror` 标记）。"""
    if isinstance(source, bool):
        return source
    if source is None:
        return False
    if isinstance(source, dict):
        source = " ".join(str(value) for value in source.values())
    text = str(source).lower()
    return any(marker in text for marker in MIRROR_HOST_MARKERS)


def _declared(pins: dict, name: str):
    return (pins.get(name) or {}).get("value")


def _lock_state(declared) -> str:
    """`not_locked`：锁里该 pin 还是 `null`；`locked`：已写死一个值。"""
    return "not_locked" if declared is None else "locked"


def _verdict_for_declared(declared, measured_value) -> tuple[str, str]:
    """把"测得值 vs 锁声明值"落成 `match` / `mismatch` / `not_locked`。"""
    if declared is None:
        return (
            NOT_LOCKED,
            "锁里该 pin 仍是 null（尚未锁定）：本次测得的是**候选值**，需要作者线写入并独立审查",
        )
    if str(declared).strip().lower() == str(measured_value).strip().lower():
        return MATCH, "与锁里声明的值逐位一致"
    return MISMATCH, "与锁里声明的值不一致：不得把 verified 置真，需查清来源"


def _finalize(report: dict) -> dict:
    """补齐恒定字段：候选补丁、汇总、`lock_modified` / `verified_flipped` 恒假。"""
    pins = report.get("pins") or {}
    report["verdict_counts"] = {
        verdict: sum(
            1
            for item in pins.values()
            if isinstance(item, dict) and item.get("verdict") == verdict
        )
        for verdict in ALL_VERDICTS
    }
    report["proposed_pin_patch"] = {
        name: pins[name].get("measured")
        for name in TARGET_PINS
        if isinstance(pins.get(name), dict) and pins[name].get("measured") is not None
    }
    report["all_match"] = all(
        isinstance(pins.get(name), dict) and pins[name].get("verdict") == MATCH
        for name in TARGET_PINS
    )
    report["lock_modified"] = False
    report["verified_flipped"] = False
    report["required_review"] = (
        "本工具**不改** official-interface.json、**不**把 verified 置真："
        "翻 verified 需要作者线按 proposed_pin_patch 提交 + KAGGLE-26/Q0 独立复核。"
    )
    return report


def _blank_pins(pins: dict, verdict: str, reason: str) -> dict:
    """给所有目标 pin 铺一条同判定的记录（缺 manifest / 不可达时用）。"""
    return {
        name: {
            "declared": _declared(pins, name),
            "lock_state": _lock_state(_declared(pins, name)),
            "measured": None,
            "verdict": verdict,
            "reason": reason,
        }
        for name in TARGET_PINS
    }


# ---- vllm 判定 ----------------------------------------------------------------


def _probed_summary(probed_list) -> list[dict]:
    summary = []
    for item in probed_list:
        if not isinstance(item, dict):
            continue
        summary.append(
            {
                "name": item.get("name") or item.get("environment") or "unnamed",
                "vllm_installed": item.get("vllm_installed"),
                "vllm_version": item.get("vllm_version") or item.get("vllm_mapper_revision"),
            }
        )
    return summary


def _judge_vllm(env: dict, declared) -> dict:
    """按 `env-report.json` 判定 `vllm_mapper_revision`（六词表中的一种）。"""
    entry: dict = {
        "declared": declared,
        "lock_state": _lock_state(declared),
        "measured": None,
    }

    # 公开源码 revision 参考：只是参考，**永不**当实测值。
    public_ref = (
        env.get("vllm_public_source_revision")
        or env.get("public_source_revision")
        or env.get("source_reference")
    )
    if public_ref:
        entry["public_source_reference"] = {
            "value": public_ref,
            "is_a_pin": False,
            "note": "这是另列的公开源码 revision 参考（如 vLLM 仓库 git revision），"
            "不是任何环境的实测值，不得写进 measured。",
        }

    probed_list = env.get("probed_environments")
    probed_list = probed_list if isinstance(probed_list, list) else []
    if probed_list:
        entry["probed_environments"] = _probed_summary(probed_list)
    scoring_evidence = bool(env.get("scoring_host_version_evidence"))

    installed_flags = [
        item.get("vllm_installed") for item in probed_list if isinstance(item, dict)
    ]
    has_installed_flag = any("vllm_installed" in item for item in probed_list if isinstance(item, dict))
    # 必须是**显式** false；None（没记录）不算"确认未安装"。
    all_probed_absent = bool(installed_flags) and all(flag is False for flag in installed_flags)

    if all_probed_absent and not scoring_evidence:
        entry["scoring_host_version_evidence"] = scoring_evidence
        entry["verdict"] = PROBED_NOT_INSTALLED
        entry["reason"] = VLLM_PROBED_NOT_INSTALLED_REASON
        entry["scope_note"] = (
            "该结论只覆盖本轮已探测的环境；评分宿主上是否安装、版本为何，本轮**没有**取得证据，"
            "不得表述为永远不可观测。"
        )
        return entry

    measured = None
    measured_from = None
    if scoring_evidence:
        scoring_host = env.get("scoring_host")
        scoring_host = scoring_host if isinstance(scoring_host, dict) else {}
        measured = (
            env.get("scoring_host_vllm_revision")
            or env.get("scoring_host_vllm_version")
            or scoring_host.get("vllm_mapper_revision")
            or scoring_host.get("vllm_version")
        )
        if measured:
            measured_from = "scoring_host"
    if measured is None:
        for item in probed_list:
            if isinstance(item, dict) and item.get("vllm_installed"):
                candidate = item.get("vllm_mapper_revision") or item.get("vllm_version")
                if candidate:
                    measured = candidate
                    measured_from = "probed_environment:%s" % (
                        item.get("name") or item.get("environment") or "unnamed"
                    )
                    break
    if measured is None:
        # 传统形状：顶层直接给 vllm 版本 / mapper revision（视为该环境实测值）。
        measured = env.get("vllm_mapper_revision") or env.get("vllm_version") or env.get("vllm")
        if measured:
            measured_from = "env_report_legacy_field"
    if isinstance(measured, dict):
        measured = measured.get("value") or measured.get("revision") or measured.get("version")

    if not measured:
        entry["scoring_host_version_evidence"] = scoring_evidence
        entry["verdict"] = MISSING_EVIDENCE
        if probed_list and not has_installed_flag:
            entry["reason"] = (
                "env-report.json 的 probed_environments 里没有 vllm_installed 标记，"
                "既拿不到版本、也判不出'未安装'"
            )
        elif probed_list:
            entry["reason"] = (
                "env-report.json 里没有可用的 vllm 版本/mapper revision 字段"
                "（probed_environments 没给出已安装且带版本的环境，评分宿主也无版本证据）"
            )
        else:
            entry["reason"] = (
                "env-report.json 里既没有 probed_environments，也没有 vllm 版本/mapper revision 字段"
            )
        return entry

    entry["measured"] = measured
    entry["measured_from"] = measured_from
    verdict, reason = _verdict_for_declared(declared, measured)
    entry["verdict"] = verdict
    entry["reason"] = reason
    return entry


# ---- 主路径：证据目录 ---------------------------------------------------------


def verify_from_evidence(lock: dict, evidence_dir: str, *, endpoint: str = DEFAULT_ENDPOINT) -> dict:
    """按证据目录做字节级核对（本工具的主路径）。

    只读：**不修改**锁、**不**置 `verified`。任何东西读不到都落成 `unreachable` /
    `missing_evidence`，不抛异常（CLI 需要恒定产出一份报告）。
    """
    pins = lock.get("pins") or {}
    repo_pin = _declared(pins, "model_repo_id")
    revision_pin = _declared(pins, "model_revision")
    endpoint_label, endpoint_is_mirror = describe_endpoint(endpoint)
    report: dict = {
        "mode": "evidence-dir",
        "evidence_dir": os.path.abspath(evidence_dir),
        "pinned_repo_id": repo_pin,
        "pinned_revision": revision_pin,
        "endpoint": {
            "template": endpoint,
            "source": endpoint_label,
            "is_mirror": endpoint_is_mirror,
            "used_for_fetch": False,
            "note": "evidence-dir 模式不发起网络请求，这里只记录端点配置，便于与 metadata-only 报告对齐。",
        },
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "local_reachability_note": LOCAL_REACHABILITY_NOTE,
        "downloaded_any_model_file": False,
        "pins": {},
        "problems": [],
    }

    if not os.path.isdir(evidence_dir):
        report["pins"] = _blank_pins(
            pins, UNREACHABLE, "证据目录不存在或不可达：%s" % evidence_dir
        )
        report["problems"].append("证据目录不可达：%s" % evidence_dir)
        return _finalize(report)

    manifest_path = os.path.join(evidence_dir, "evidence-manifest.json")
    if not os.path.exists(manifest_path):
        report["pins"] = _blank_pins(pins, MISSING_EVIDENCE, "缺少 evidence-manifest.json")
        report["problems"].append("缺少 evidence-manifest.json：无法判定证据来自哪个 repo/revision")
        return _finalize(report)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        report["pins"] = _blank_pins(
            pins, UNREACHABLE, "evidence-manifest.json 读取/解析失败：%s" % exc
        )
        report["problems"].append("evidence-manifest.json 不可读：%s" % exc)
        return _finalize(report)
    if not isinstance(manifest, dict):
        report["pins"] = _blank_pins(pins, UNREACHABLE, "evidence-manifest.json 不是 JSON 对象")
        report["problems"].append("evidence-manifest.json 不是 JSON 对象")
        return _finalize(report)

    report["evidence_manifest"] = manifest

    retrieval_status = manifest.get("retrieval_status") or manifest.get("status")
    failed_status = (
        str(retrieval_status).lower()
        if isinstance(retrieval_status, str)
        else ""
    )
    if manifest.get("error") or failed_status in ("failed", "error", "unreachable", "timeout", "timed_out"):
        reason = "证据 manifest 声明本轮取回失败（retrieval_status=%r）：证据本身不可达" % (
            retrieval_status,
        )
        report["pins"] = _blank_pins(pins, UNREACHABLE, reason)
        report["problems"].append(reason)
        report["retrieval_failed"] = True
        return _finalize(report)

    for field, expected in (("model_repo_id", repo_pin), ("model_revision", revision_pin)):
        actual = manifest.get(field)
        if actual != expected:
            report["problems"].append(
                "证据的 %s=%r 与锁里的 pin %r 不一致：这份证据不适用于当前锁定版本"
                % (field, actual, expected)
            )
    for field in ("retrieved_at_utc", "retrieval_command", "source"):
        if not manifest.get(field):
            report["problems"].append("证据 manifest 缺少 %s：来源不可追溯" % field)

    # 镜像来源：字节与"同一镜像自己的元数据"一致只能算自洽校验（评审 4 号要求）。
    mirror_only = _is_mirror_source(manifest.get("source")) or _is_mirror_source(
        manifest.get("source_is_mirror")
    )
    report["evidence_source_authenticity"] = {
        "source": manifest.get("source"),
        "source_is_mirror": mirror_only,
        "consistency_check_only": mirror_only,
        "statement": MIRROR_CONSISTENCY_STATEMENT if mirror_only else "证据来源非镜像，无需降级为自洽校验。",
    }
    file_shas = manifest.get("file_sha256")
    file_shas = file_shas if isinstance(file_shas, dict) else {}

    measured: dict = {}
    for name, candidate_names in SHA_PIN_FILES.items():
        path = _first_existing(evidence_dir, candidate_names)
        if not path:
            continue
        measured[name] = {
            "value": sha256_file(path),
            "source_file": os.path.basename(path),
            "bytes": os.path.getsize(path),
        }

    for name, candidate_names in SHA_PIN_FILES.items():
        declared = _declared(pins, name)
        if name not in measured:
            report["pins"][name] = {
                "declared": declared,
                "lock_state": _lock_state(declared),
                "measured": None,
                "verdict": MISSING_EVIDENCE,
                "reason": "证据目录里没有对应的证据文件（%s）：该 pin 只能由**自己的**文件字节判定，"
                "不得用相邻 pin 的哈希顶替" % (candidate_names,),
            }
            continue
        entry = {
            "declared": declared,
            "lock_state": _lock_state(declared),
            "measured": measured[name],
        }
        value = measured[name]["value"]
        verdict, reason = _verdict_for_declared(declared, value)

        # 与"同一镜像自己的元数据"比对 → 只算自洽校验。
        mirror_sha = file_shas.get(measured[name]["source_file"])
        if mirror_sha:
            entry["mirror_metadata_sha256"] = mirror_sha
            entry["mirror_metadata_agrees"] = str(mirror_sha).lower() == value
            if entry["mirror_metadata_agrees"]:
                entry["consistency_check_only"] = True
                entry["mirror_metadata_note"] = MIRROR_CONSISTENCY_STATEMENT
                reason = "%s；%s" % (reason, MIRROR_CONSISTENCY_STATEMENT)
        elif mirror_only:
            entry["consistency_check_only"] = True
            reason = "%s；%s" % (reason, MIRROR_CONSISTENCY_STATEMENT)

        entry["verdict"] = verdict
        entry["reason"] = reason
        report["pins"][name] = entry
        if verdict == MISMATCH:
            report["problems"].append("%s 与证据不一致" % name)

    # vLLM mapper revision
    env_path = os.path.join(evidence_dir, "env-report.json")
    declared = _declared(pins, "vllm_mapper_revision")
    if not os.path.exists(env_path):
        report["pins"]["vllm_mapper_revision"] = {
            "declared": declared,
            "lock_state": _lock_state(declared),
            "measured": None,
            "verdict": MISSING_EVIDENCE,
            "reason": "缺少 env-report.json（需要目标环境里 vllm 的实际版本与 mapper 来源）",
        }
    else:
        try:
            with open(env_path, "r", encoding="utf-8") as handle:
                env = json.load(handle)
        except (OSError, ValueError) as exc:
            env = None
            report["problems"].append("env-report.json 不可读：%s" % exc)
        if not isinstance(env, dict):
            report["pins"]["vllm_mapper_revision"] = {
                "declared": declared,
                "lock_state": _lock_state(declared),
                "measured": None,
                "verdict": UNREACHABLE if env is None else MISSING_EVIDENCE,
                "reason": "env-report.json 读取/解析失败或不是 JSON 对象",
            }
        else:
            entry = _judge_vllm(env, declared)
            report["pins"]["vllm_mapper_revision"] = entry
            if entry.get("verdict") == MISMATCH:
                report["problems"].append("vllm_mapper_revision 与证据不一致")
            if entry.get("public_source_reference"):
                report["public_source_reference"] = entry["public_source_reference"]

    return _finalize(report)


# ---- 只读元数据路径 -----------------------------------------------------------


def _metadata_only_pin_stub(pins: dict, verdict: str, reason: str) -> dict:
    return {
        name: {
            "declared": _declared(pins, name),
            "lock_state": _lock_state(_declared(pins, name)),
            "measured": None,
            "verdict": verdict,
            "reason": reason,
        }
        for name in TARGET_PINS
    }


def verify_metadata_only(
    lock: dict,
    *,
    fetcher=None,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 20.0,
) -> dict:
    """只查 HF **元数据**（不下文件）；取不到就如实报 `unreachable`。"""
    pins = lock.get("pins") or {}
    repo = _declared(pins, "model_repo_id")
    revision = _declared(pins, "model_revision")
    endpoint_label, endpoint_is_mirror = describe_endpoint(endpoint)
    result: dict = {
        "mode": "metadata-only",
        "repo_id": repo,
        "revision": revision,
        "pinned_revision": revision,
        "endpoint": {
            "template": endpoint,
            "source": endpoint_label,
            "is_mirror": endpoint_is_mirror,
            "used_for_fetch": bool(repo and revision),
        },
        "endpoint_reachable": None,
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "local_reachability_note": LOCAL_REACHABILITY_NOTE,
        "pins": _metadata_only_pin_stub(pins, UNREACHABLE, "元数据查询未发起"),
        "problems": [],
        "downloaded_any_model_file": False,
    }
    if endpoint_is_mirror:
        # 镜像返回的元数据只是镜像自己的视图：可以看，不构成官方源认证。
        result["consistency_check_only"] = True
        result["source_authentication_note"] = MIRROR_CONSISTENCY_STATEMENT

    if not repo or not revision:
        reason = "锁里缺 model_repo_id / model_revision，无法发起查询"
        result["error"] = reason
        result["pins"] = _metadata_only_pin_stub(pins, UNREACHABLE, reason)
        return _finalize(result)

    url = resolve_endpoint(endpoint, repo, revision)
    result["url"] = url
    fetch = fetcher or (lambda target: _default_fetch(target, timeout=timeout))
    try:
        payload = fetch(url)
        if not isinstance(payload, dict):
            raise ValueError("端点返回的不是 JSON 对象：%r" % type(payload).__name__)
    except Exception as exc:  # noqa: BLE001 - 网络不可达是预期分支
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        result["endpoint_reachable"] = False
        reason = "元数据端点不可达：%s" % result["error"]
        result["pins"] = _metadata_only_pin_stub(pins, UNREACHABLE, reason)
        result["problems"].append(reason)
        return _finalize(result)

    result["endpoint_reachable"] = True
    result["response_sha"] = payload.get("sha")
    result["response_sha_matches_pinned_revision"] = payload.get("sha") == revision
    if payload.get("sha") and payload.get("sha") != revision:
        result["problems"].append(
            "端点返回的 sha=%r 与锁定 revision=%r 不一致" % (payload.get("sha"), revision)
        )
    siblings = [item.get("rfilename") for item in (payload.get("siblings") or []) if isinstance(item, dict)]
    result["file_count"] = len(siblings)
    result["tokenizer_files_present"] = sorted(
        name for name in siblings if name and ("tokenizer" in name or "vocab" in name)
    )
    # 端点可达但这类证据回答不了这三个 SHA pin：报 missing_evidence，而不是 unreachable。
    for name in SHA_PIN_FILES:
        result["pins"][name] = {
            "declared": _declared(pins, name),
            "lock_state": _lock_state(_declared(pins, name)),
            "measured": None,
            "verdict": MISSING_EVIDENCE,
            "reason": (
                "端点可达，但元数据 API 不返回非 LFS 小文件（含 chat_template.jinja）的 SHA-256："
                "仍需在具备出口的环境取回文件后重算"
            ),
        }
    result["pins"]["vllm_mapper_revision"] = {
        "declared": _declared(pins, "vllm_mapper_revision"),
        "lock_state": _lock_state(_declared(pins, "vllm_mapper_revision")),
        "measured": None,
        "verdict": MISSING_EVIDENCE,
        "reason": "端点可达，但 mapper revision 属于目标环境的软件事实，模型元数据里没有",
    }
    return _finalize(result)


def _default_fetch(url: str, *, timeout: float = 20.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "v3-pin-verifier/2"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


# ---- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="G9：核对 official-interface 的未验证 pin（只报不改）",
        epilog=(
            "示例：\n"
            "  python tools/verify_interface_pins.py --metadata-only\n"
            "  python tools/verify_interface_pins.py --metadata-only "
            "--endpoint https://hf-mirror.com/api/models/{repo}/revision/{revision}\n"
            "  python tools/verify_interface_pins.py --evidence-dir docs/v3/evidence/g9 --out g9.json\n"
            "退出码：产出报告即 0；端点模板不合法为 2（可达性写在报告字段里，不用退出码表达）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interface", default=os.path.join(REPO_ROOT, "v3", "locks", "official-interface.json")
    )
    parser.add_argument("--evidence-dir", default=None, help="取证目录（见模块文档）")
    parser.add_argument("--metadata-only", action="store_true", help="只查 HF 元数据，不下文件")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=(
            "元数据端点模板，支持 {repo}/{revision} 或两个 {} 占位符"
            "（默认 %s；镜像可用 https://hf-mirror.com/api/models/{repo}/revision/{revision}）"
            % DEFAULT_ENDPOINT
        ),
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="元数据请求超时秒数（默认 20）")
    parser.add_argument("--out", default=None, help="报告输出路径（默认打印到 stdout）")
    args = parser.parse_args(argv)

    _label, _is_mirror = describe_endpoint(args.endpoint)
    try:
        resolve_endpoint(args.endpoint, "repo", "revision")
    except ValueError as exc:
        print("端点模板不合法：%s" % exc, file=sys.stderr)
        return 2

    lock = load_lock(args.interface)
    if args.evidence_dir:
        report = verify_from_evidence(lock, args.evidence_dir, endpoint=args.endpoint)
    else:
        report = verify_metadata_only(
            lock, endpoint=args.endpoint, timeout=args.timeout
        )
    report["interface_lock"] = {
        "path": os.path.abspath(args.interface),
        "writable_by_this_tool": False,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
