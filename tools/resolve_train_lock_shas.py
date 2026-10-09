"""G6：从**公开 PyPI 元数据**解析训练依赖锁的精确 wheel SHA（不安装、不下载大文件）。

## 缺陷出处

KAGGLE-32 v6 `GAPS-AND-OWNERS.md` §G6：

    train 依赖锁 `verified=false`，7 个包 wheel SHA 全为 `PENDING`
    （`v3/locks/train.lock.json`，E0 `128d9b98…`）

## 这个工具做什么

对 `v3/locks/train.lock.json` 里每个**已固定版本**的包，读取 PyPI 的
`https://pypi.org/pypi/<name>/<version>/json`（只有元数据，**不下载 wheel**），
按目标平台标签挑出唯一的 wheel，并记下它的
`filename` / `sha256`（来自 `digests.sha256`）/ `size` / `yanked` / 元数据 URL。

产出 `train.lock.resolved.json`，逐包状态是：

- `resolved`：拿到精确 wheel 与 SHA，并且 `resolution_source.url` 可复核；
- `excluded`：该条目带**依赖审计块**，结论是"不属于训练路径的安装集"。
  这类条目**不再算 pending**，但审计理由必须原样带进结果
  （`exclusion_audit` / `exclusion_reason`）—— **不允许静默消失**；
- `pending`：**拿不到就如实挂起**，并写明 `pending_reason`
  （例如源锁里版本仍是 `PENDING`、或该版本没有匹配目标平台的 wheel）。

### 源锁里 `excluded` 结论的**两个**位置（KAGGLE-38 r3 起支持）

**(a) 顶层 `excluded_packages`（现在的正位）** —— 被排除的包**不在** `packages`
安装集里，审计结论单独放在顶层：

    "packages": { "torch": {…}, … },          // 安装集：必须无 PENDING
    "excluded_packages": {
      "trl": {
        "reason": "not_a_dependency_of_the_training_path",
        "audit": ["…"],
        "evidence": ["https://pypi.org/pypi/trl/json"],
        "recorded_at_utc": "…",
        "reintroduce_if": "…"
      }
    }

结果里这类条目进**独立的** `excluded_packages` 映射（带
`exclusion_location = "top_level:excluded_packages"`），**不进** `packages` ——
这样"安装集合里没有 TRL"在输出里是**结构性的**，而不只是一句说明。

为什么放顶层：`v3/t0/deps.py::DependencyLock.validate()` 在冻结执行闭包里，
它不认识 `excluded`，只要被排除的包还留在 `packages` 里，它的 `PENDING`
就会被报成 problem —— 于是"安装集无 PENDING"与"validate() 为空"无法同时成立。

**(b) 包条目内联的 `excluded` 块（兼容旧形态）** —— 包仍在 `packages` 里，
但条目自带审计块：

    "trl": {
      "source": "excluded",
      "version": "PENDING",
      "wheel_sha256": "PENDING",
      "excluded": {"reason": "…", "audit": ["…"], "evidence": ["…"]}
    }

结果里这类条目留在 `packages` 映射内、状态 `excluded`，带
`exclusion_location = "packages.<name>.excluded"`。

两种位置都**不**联网取数（排除结论不依赖任何网络），
`version` / `wheel_sha256` 保持 `PENDING` 是**故意的**：任何忽略 `excluded` 的朴素
消费者仍然 fail-closed，不会把"已排除"误读成"已固定版本"。
同一个包**不得**同时出现在两个位置（`resolve_lock` 直接报错）。

## 这个工具**不**做什么

- **不下载 wheel、不安装任何包、不建 venv**；
- **不把 `verified` 置真**：输出文件恒为 `"verified": false`，并写明为什么 ——
  `verified` 的真正含义是"在目标解释器上按 SHA 装上了并能 import"，
  那必须在 107 上跑 `pip download --require-hashes` + 独立审查，本工具给不了；
- **不用 latest 顶替**：源锁里版本未固定时，只把"观察到的最新版"记进
  `observed_latest`（明确标注**不是 pin**），状态仍是 `pending`；
- **不解析传递闭包**：本工具只处理锁里**已声明**的条目。完整传递依赖（含 CUDA
  运行时轮子）由 `docs/v3/evidence/kaggle-38-train-stack-manifest.json` 承载，
  锁里以顶层 `train_stack_manifest` 指针引用它。

## 用法

    # 在线（默认）：直接读 PyPI 元数据
    python tools/resolve_train_lock_shas.py

    # 离线 / 可复现：从本地缓存的元数据目录读（每个包一个 <name>.json）
    python tools/resolve_train_lock_shas.py --metadata-dir ./cache --out ./resolved.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: 目标平台（107 的登录/计算节点：Linux x86_64 + glibc 2.39 + 训练锁要求的 Python 3.11.9）。
DEFAULT_TARGET = {
    "python": "3.11.9",
    "abi_tag": "cp311",
    "glibc": "2.39",
    #: glibc 2.39 上**可安装**的 manylinux 标签阶梯，按 pip 的偏好序（新 → 旧；
    #: 别名紧跟在其等价标签之后）。列全 2_39…2_5 是必须的：CUDA 运行时轮子
    #: （`nvidia-*-cu12`、`cublas`、`cudnn`、`triton`）与 numpy 都发 `manylinux_2_27_x86_64`，
    #: 云端 glibc 2.39 完全能装，只列 2_28/2_24/2_17 会把这些包误报成 pending。
    "platform_tags": [
        "manylinux_2_39_x86_64",
        "manylinux_2_38_x86_64",
        "manylinux_2_37_x86_64",
        "manylinux_2_36_x86_64",
        "manylinux_2_35_x86_64",
        "manylinux_2_34_x86_64",
        "manylinux_2_33_x86_64",
        "manylinux_2_32_x86_64",
        "manylinux_2_31_x86_64",
        "manylinux_2_30_x86_64",
        "manylinux_2_29_x86_64",
        "manylinux_2_28_x86_64",
        "manylinux_2_27_x86_64",
        "manylinux_2_26_x86_64",
        "manylinux_2_25_x86_64",
        "manylinux_2_24_x86_64",
        "manylinux_2_17_x86_64",
        "manylinux2014_x86_64",
        "manylinux_2_12_x86_64",
        "manylinux2010_x86_64",
        "manylinux_2_5_x86_64",
        "manylinux1_x86_64",
        "linux_x86_64",
        "any",
    ],
}

#: 未固定版本的字面量（与 `v3/common/errors.py::reject_unpinned` 同口径）。
UNPINNED = {"", "pending", "latest", "head", "main", "master", "*", "any", "none", "null"}

PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"
PYPI_JSON_ANY = "https://pypi.org/pypi/{name}/json"

WHEEL_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"(?:-(?P<build>[^-]+))?"
    r"-(?P<pytag>[^-]+)-(?P<abitag>[^-]+)-(?P<platform>[^-]+)\.whl$"
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str, *, timeout: float = 30.0) -> dict:
    """默认取数器：只做一次 GET，失败直接抛出，不重试、不静默降级。"""
    request = urllib.request.Request(url, headers={"User-Agent": "v3-lock-resolver/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def load_metadata(*, name: str, version: str | None, metadata_dir: str | None, fetcher=fetch_json):
    """取一个包的元数据；`metadata_dir` 给出时从本地 JSON 读（离线可复现）。"""
    if metadata_dir:
        path = os.path.join(metadata_dir, "%s.json" % name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), "file://%s" % os.path.abspath(path)
    url = PYPI_JSON_ANY.format(name=name) if not version else PYPI_JSON.format(name=name, version=version)
    return fetcher(url), url


def _python_tag_ok(pytag: str, abi_tag: str) -> bool:
    """Python tag 是否匹配目标 ABI（含 `abi3` 稳定 ABI 与 `py3` 纯 Python 轮子）。

    `abi3` 轮子（如 safetensors 的 `cp38-abi3-...`）对**更高**的 CPython 也适用，
    因此 `cp38` 必须被判为"能在 3.11 上装"，否则会误报 pending。
    """
    tags = pytag.split(".")
    if abi_tag in tags:
        return True
    if any(tag in ("py3", "py2.py3", "py2") for tag in tags):
        return True
    match = re.match(r"^cp3(\d+)$", abi_tag)
    target_minor = int(match.group(1)) if match else None
    if target_minor is None:
        return False
    for tag in tags:
        found = re.match(r"^cp3(\d+)$", tag)
        if found and int(found.group(1)) <= target_minor:
            return True
    return False


def _platform_tag_ok(platform: str, allowed: list[str]) -> bool:
    """平台标签是否匹配（复合标签按 `.` 拆开逐个比对）。"""
    parts = platform.split(".")
    for part in parts:
        for tag in allowed:
            if part == tag or part.endswith(tag):
                return True
    return False


def pick_wheel(files: list[dict], target: dict) -> tuple[dict | None, list[dict]]:
    """按目标平台标签挑唯一的 wheel；返回 `(best, candidates)`。"""
    candidates: list[dict] = []
    for item in files:
        filename = str(item.get("filename") or "")
        match = WHEEL_RE.match(filename)
        if not match:
            continue
        pytag = match.group("pytag")
        abitag = match.group("abitag")
        platform = match.group("platform")
        abi_ok = target["abi_tag"] in abitag.split(".") or abitag in ("none", "abi3")
        if not abi_ok or not _python_tag_ok(pytag, target["abi_tag"]):
            continue
        if not _platform_tag_ok(platform, target["platform_tags"]):
            continue
        candidates.append(
            {
                "filename": filename,
                "sha256": (item.get("digests") or {}).get("sha256"),
                "size": item.get("size"),
                "url": item.get("url"),
                "yanked": bool(item.get("yanked")),
                "pytag": pytag,
                "abitag": abitag,
                "platform": platform,
            }
        )

    def rank(item: dict) -> tuple:
        parts = item["platform"].split(".")
        platform_rank = next(
            (index for index, tag in enumerate(target["platform_tags"]) if tag in parts),
            len(target["platform_tags"]),
        )
        specific = 0 if item["platform"] != "any" else 1
        return (1 if item["yanked"] else 0, specific, platform_rank, item["filename"])

    usable = [item for item in candidates if item["sha256"]]
    usable.sort(key=rank)
    return (usable[0] if usable else None), candidates


def resolve_package(
    name: str,
    spec: dict,
    *,
    target: dict,
    metadata_dir: str | None = None,
    fetcher=fetch_json,
) -> dict:
    """解析单个包：带审计块 → `excluded`；能确定 → `resolved`；不能 → `pending` 并写明原因。"""
    version = str(spec.get("version") or "").strip()
    result: dict = {
        "source_declared": dict(spec),
        "version": version or None,
        "status": "pending",
        "wheel": None,
        "candidates": [],
        "pending_reason": None,
        "resolution_source": None,
        "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    excluded = spec.get("excluded")
    if excluded:
        # 依赖审计结论：该包**不属于**训练路径的安装集。不当作 pending，也**不**丢掉：
        # 审计块原样进结果，且 `reason` 必须非空，否则视为"静默排除"直接报错。
        audit = dict(excluded) if isinstance(excluded, dict) else {"reason": str(excluded)}
        reason = str(audit.get("reason") or "").strip()
        if not reason:
            raise AssertionError("包 %s 标记为 excluded 但审计块没写 reason" % name)
        result["status"] = "excluded"
        result["exclusion_audit"] = audit
        result["exclusion_reason"] = reason
        result["exclusion_location"] = "packages.%s.excluded" % name
        result["pending_reason"] = None
        return result
    if version.lower() in UNPINNED:
        result["pending_reason"] = (
            "源锁里该包版本未固定（%r）：本工具**不**用 latest 顶替，需先由作者写出精确版本"
            % (spec.get("version"),)
        )
        try:
            payload, url = load_metadata(
                name=name, version=None, metadata_dir=metadata_dir, fetcher=fetcher
            )
            result["observed_latest"] = {
                "version": (payload.get("info") or {}).get("version"),
                "url": url,
                "is_a_pin": False,
                "note": "仅为观察值，**不是** pin，不得写进依赖锁",
            }
        except Exception as exc:  # noqa: BLE001 - 离线时只记原因
            result["observed_latest"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
        return result

    try:
        payload, url = load_metadata(
            name=name, version=version, metadata_dir=metadata_dir, fetcher=fetcher
        )
    except urllib.error.HTTPError as exc:
        result["pending_reason"] = "该版本在 PyPI 上取不到元数据（HTTP %s）：不得自造 SHA" % exc.code
        return result
    except FileNotFoundError:
        result["pending_reason"] = "本地元数据缓存里没有 %s.json" % name
        return result
    except Exception as exc:  # noqa: BLE001
        result["pending_reason"] = "取元数据失败（%s）：不得自造 SHA" % type(exc).__name__
        result["resolution_error"] = str(exc)
        return result

    files = list((payload.get("releases") or {}).get(version) or payload.get("urls") or [])
    best, candidates = pick_wheel(files, target)
    result["candidates"] = candidates
    result["resolution_source"] = {
        "api": "pypi-json-v1",
        "url": url,
        "file_count_for_version": len(files),
    }
    if best is None:
        result["pending_reason"] = (
            "版本 %s 存在，但没有匹配目标平台 %s 的 wheel（候选 %d 个）"
            % (version, target["platform_tags"][0], len(candidates))
        )
        return result
    result["wheel"] = best
    result["status"] = "resolved"
    result["pending_reason"] = None
    return result


def resolve_excluded_package(name: str, block, *, location: str = "top_level:excluded_packages") -> dict:
    """顶层 `excluded_packages` 条目：报 `excluded`、**不**联网、**不**算 pending。

    审计块本身就是结论（`reason` / `audit` / `evidence` / `recorded_at_utc` /
    `reintroduce_if`），这里原样带进结果，只补 `exclusion_location` 以标明它来自
    顶层而不是包条目内部。缺 `reason` 视为"静默排除"，直接报错。
    """
    if not isinstance(block, dict):
        raise AssertionError("excluded_packages.%s 不是对象（%r）" % (name, type(block).__name__))
    audit = dict(block)
    reason = str(audit.get("reason") or "").strip()
    if not reason:
        raise AssertionError("包 %s 标在 %s 里但审计块没写 reason" % (name, location))
    return {
        "source_declared": audit,
        "version": audit.get("version"),
        "status": "excluded",
        "wheel": None,
        "candidates": [],
        "pending_reason": None,
        "resolution_source": None,
        "exclusion_audit": audit,
        "exclusion_reason": reason,
        "exclusion_location": location,
        "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def resolve_lock(
    lock: dict,
    *,
    target: dict | None = None,
    metadata_dir: str | None = None,
    fetcher=fetch_json,
    source_path: str | None = None,
) -> dict:
    """解析整份锁，返回**不会**把 verified 置真的结果对象。

    输出形状（`format = "v3-train-lock-resolved/2"`）刻意与锁同构：

    - `packages`：源锁 `packages` 里的条目（`resolved` / `pending`，以及**旧形态**
      的内联 `excluded` 条目）；
    - `excluded_packages`：源锁顶层 `excluded_packages` 里的审计条目 ——
      它们是**结论**，不进 `packages`，所以"安装集合里没有 TRL"在输出里是结构性的；
    - `summary`：两处合起来计数，`excluded` 含两个位置的排除项。
    """
    target = dict(target or DEFAULT_TARGET)
    packages = lock.get("packages") or {}
    excluded_declared = lock.get("excluded_packages") or {}
    if not isinstance(excluded_declared, dict):
        raise AssertionError("excluded_packages 必须是对象（name -> 审计块）")
    both = sorted(set(packages) & set(excluded_declared))
    if both:
        raise AssertionError(
            "包 %s 同时出现在 packages 与 excluded_packages：一个包只能有一个位置" % both
        )

    resolved: dict[str, dict] = {}
    for name in sorted(packages):
        resolved[name] = resolve_package(
            name,
            packages[name],
            target=target,
            metadata_dir=metadata_dir,
            fetcher=fetcher,
        )
    excluded_top: dict[str, dict] = {}
    for name in sorted(excluded_declared):
        excluded_top[name] = resolve_excluded_package(name, excluded_declared[name])

    ok = sorted(name for name, item in resolved.items() if item["status"] == "resolved")
    pending = sorted(
        name
        for name, item in resolved.items()
        if item["status"] not in ("resolved", "excluded")
    )
    inline_excluded = sorted(
        name for name, item in resolved.items() if item["status"] == "excluded"
    )
    excluded = sorted(set(inline_excluded) | set(excluded_top))
    accounted = len(ok) + len(pending) + len(excluded)
    declared = len(packages) + len(excluded_declared)
    out = {
        "channel": lock.get("channel", "train"),
        "format": "v3-train-lock-resolved/2",
        "target": target,
        "source_lock": {
            "path": source_path,
            "sha256": sha256_file(source_path) if source_path and os.path.exists(source_path) else None,
            "declared_python": lock.get("python"),
            "declared_verified": bool(lock.get("verified")),
            "declared_packages": sorted(packages),
            "declared_excluded_packages": sorted(excluded_declared),
        },
        "train_stack_manifest": lock.get("train_stack_manifest"),
        "resolved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "packages": resolved,
        "excluded_packages": excluded_top,
        "summary": {
            "resolved": len(ok),
            "excluded": len(excluded),
            "pending": len(pending),
            "resolved_packages": ok,
            "excluded_packages": excluded,
            "pending_packages": pending,
            "excluded_inline_in_packages": inline_excluded,
            "excluded_from_top_level": sorted(excluded_top),
            "declared_packages": len(packages),
            "declared_excluded_packages": len(excluded_declared),
            "every_declared_package_has_a_verdict": accounted == declared,
        },
        # 恒为 false：见下面这段说明，不由本工具翻转。
        "verified": False,
        "verified_note": (
            "本文件**不**把 verified 置真，也**不**改写 v3/locks/train.lock.json。"
            "`verified` 的真正含义是「在目标解释器上按这些 SHA 装上了、能 import、"
            "并且独立审查过」—— 需要在 107 上执行 "
            "`pip download --require-hashes -r <requirements>` + `pip install --no-index` 并留证据，"
            "由作者线提交、独审确认。本工具只提供可复核的公开元数据来源。"
        ),
        "next_steps": [
            "把 resolved 条目回填到 v3/locks/train.lock.json 的 wheel_sha256（由 E0 作者线做）",
            "在 107 上按 SHA 实装并记录 pip 输出（`--require-hashes` 会拒绝任何不一致）",
            "pending 条目需要先由作者给出精确版本；不得用 latest 顶替",
            "excluded 条目是**依赖审计**结论（不在安装集里）：审计理由随锁一起评审；"
            "若要重新引入，必须给出精确版本 + wheel SHA 并**删掉**对应的审计块、"
            "把精确 pin 写回 `packages`",
            "完整传递闭包（含 CUDA 轮子）见锁里的 train_stack_manifest 指针，"
            "本工具**不**解析传递闭包（那由 docs/v3/evidence/kaggle-38-train-stack-manifest.json 负责）",
        ],
    }
    assert out["verified"] is False, "本工具绝不能把 verified 置真"
    return out


def assert_no_pending_without_reason(result: dict) -> None:
    """自检：pending 必须写明原因，excluded 必须写明审计理由（都不许"静默"）。

    两个位置都查：`packages`（含旧形态的内联 `excluded`）与顶层 `excluded_packages`。
    """
    items = list((result.get("packages") or {}).items())
    items += list((result.get("excluded_packages") or {}).items())
    for name, item in items:
        status = item["status"]
        if status == "pending" and not item.get("pending_reason"):
            raise AssertionError("包 %s 处于 pending 但没写原因" % name)
        if status == "excluded" and not item.get("exclusion_reason"):
            raise AssertionError("包 %s 标记为 excluded 但没写审计理由" % name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G6：解析训练依赖锁的精确 wheel SHA（不安装、不下载）")
    parser.add_argument("--lock", default=os.path.join(REPO_ROOT, "v3", "locks", "train.lock.json"))
    parser.add_argument(
        "--out", default=os.path.join(REPO_ROOT, "v3", "locks", "train.lock.resolved.json")
    )
    parser.add_argument("--metadata-dir", default=None, help="离线元数据目录（每个包一个 <name>.json）")
    parser.add_argument("--python", default=DEFAULT_TARGET["python"])
    parser.add_argument("--abi", default=DEFAULT_TARGET["abi_tag"])
    args = parser.parse_args(argv)

    with open(args.lock, "r", encoding="utf-8") as handle:
        lock = json.load(handle)
    target = dict(DEFAULT_TARGET)
    target["python"] = args.python
    target["abi_tag"] = args.abi
    result = resolve_lock(
        lock, target=target, metadata_dir=args.metadata_dir, source_path=args.lock
    )
    assert_no_pending_without_reason(result)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    summary = dict(result["summary"])
    summary["out"] = os.path.abspath(args.out)
    summary["verified"] = result["verified"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
