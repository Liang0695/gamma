"""G9：按**固定 revision** 取回接口 pin 所需的**非权重小文件**，产出可复核证据目录。

## 为什么需要这个工具

KAGGLE-28 的 G9 取证报告把三个 pin 的结论交了上来，但**附件里只有
`tokenizer_config.json` 原件**：`tokenizer.json`（词表）与 `chat_template.jinja`
（独立于 config 的 chat template）只有哈希、没有原件，评审无法独立逐字节复核。
本工具补的正是这一段：把固定 revision 上的**小文件**取回来放进取证目录，写出
`evidence-manifest.json`，然后交给 `tools/verify_interface_pins.py --evidence-dir` 核对。

## 硬边界

- 只取**非权重**文件：`tokenizer_config.json` / `tokenizer.json` / `chat_template.jinja`
  / `config.json`。**绝不下载** `*.safetensors`（本工具显式拒绝）；
- revision 必须精确给定，禁止 `latest` / `main` / `HEAD`；
- 一条命令只打一个端点，且把端点写进 manifest（`source` 字段）：默认
  `hf-mirror.com`（本机实测唯一可达的通道），可换 `huggingface.co`。
  **镜像取回的字节与"同一镜像自己的元数据"一致只构成自洽校验**，
  不构成对官方源的独立认证 —— manifest 与报告里都这么写；
- 工具**不**改 `v3/locks/official-interface.json`、**不**翻任何 `verified`。

## 用法

    python tools/capture_interface_pin_evidence.py \
        --repo-id google/gemma-4-31B-it-qat-w4a16-ct \
        --revision 52f3f65bc7a02d555763bc923bd1d9094898219d \
        --out docs/v3/evidence/kaggle-38-g9-evidence

产出目录内容：

    evidence-manifest.json   repo/revision/来源/时间/命令/逐文件实测 SHA-256
    env-report.json          vLLM mapper pin 的环境探测事实（**手工填**，见下）
    tokenizer_config.json    固定 revision 原件
    tokenizer.json           固定 revision 原件（词表）
    chat_template.jinja      固定 revision 原件（独立于 config）
    hf-metadata-blobs.json   同镜像元数据原文（`?blobs=true`），用于自洽比对

`env-report.json` **不由本工具编造**：目标环境里 vllm 装没装、装的是哪个版本，
只能在真实执行环境里探测。本工具只写一个 `_template` 形状，实际证据必须由执行链
（KAGGLE-28 的 107 只读探测）填入 `probed_environments` 与
`scoring_host_version_evidence`。缺这份文件时，`verify_interface_pins.py` 会把
vllm pin 如实报成 `missing_evidence` —— 那是诚实的，不是缺陷。
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

#: 允许取回的**非权重**文件白名单（顺序即报告顺序）。
NON_WEIGHT_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "config.json",
)

#: 明确禁止下载的权重后缀（出现即拒绝，不做"反正也没列进白名单"的默认放过）。
FORBIDDEN_SUFFIXES = (".safetensors", ".bin", ".pt", ".ckpt", ".pth", ".gguf")

#: 端点。默认走本机实测可达的只读镜像。
DEFAULT_ENDPOINT = "https://hf-mirror.com"

#: 被拒绝的 revision 写法。
UNPINNED_REVISIONS = ("", "latest", "main", "head", "master")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_pinned_revision(revision: str) -> str:
    value = str(revision or "").strip()
    if value.lower() in UNPINNED_REVISIONS:
        raise SystemExit("revision 必须精确锁定，禁止 latest/main/HEAD：%r" % (revision,))
    if len(value) < 7:
        raise SystemExit("revision 太短，不像一个精确 commit：%r" % (revision,))
    return value


def assert_not_a_weight(name: str) -> None:
    lowered = str(name).lower()
    if lowered.endswith(FORBIDDEN_SUFFIXES):
        raise SystemExit("拒绝下载权重文件 %r：本工具只取非权重小文件" % (name,))


def _fetch(url: str, *, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "v3-pin-capture/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def capture(
    *,
    repo_id: str,
    revision: str,
    out_dir: str,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 120.0,
    files=NON_WEIGHT_FILES,
    fetcher=None,
) -> dict:
    """取回文件并写 manifest；返回报告（同时落盘 `evidence-manifest.json`）。"""
    revision = assert_pinned_revision(revision)
    endpoint = endpoint.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)
    fetch = fetcher or (lambda url: _fetch(url, timeout=timeout))

    meta_url = "%s/api/models/%s/revision/%s?blobs=true" % (endpoint, repo_id, revision)
    report: dict = {
        "mode": "capture",
        "repo_id": repo_id,
        "revision": revision,
        "endpoint": endpoint,
        "source": endpoint.replace("https://", "").split("/")[0],
        "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata_url": meta_url,
        "downloaded_weight_files": [],
        "only_non_weight_files": True,
        "lock_modified": False,
        "verified_flipped": False,
        "problems": [],
    }

    metadata = None
    metadata_sha = None
    try:
        raw = fetch(meta_url)
        metadata = json.loads(raw.decode("utf-8"))
        metadata_sha = sha256_bytes(raw)
        with open(os.path.join(out_dir, "hf-metadata-blobs.json"), "wb") as handle:
            handle.write(raw)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        report["problems"].append("元数据端点不可达：%s: %s" % (type(exc).__name__, exc))
        report["metadata_sha256"] = None
    else:
        report["metadata_sha256"] = metadata_sha
        report["metadata_sha_field"] = metadata.get("sha") if isinstance(metadata, dict) else None
        report["metadata_sha_matches_revision"] = (
            isinstance(metadata, dict) and str(metadata.get("sha")) == revision
        )

    # 同镜像元数据里声明的逐文件 SHA（LFS 才有；小文件通常只有 size）——自洽比对用。
    declared: dict = {}
    if isinstance(metadata, dict):
        for item in metadata.get("siblings") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("rfilename")
            lfs = item.get("lfs") or {}
            if name and lfs.get("sha256"):
                declared[name] = str(lfs["sha256"])

    measured: dict = {}
    for name in files:
        assert_not_a_weight(name)
        url = "%s/%s/resolve/%s/%s" % (endpoint, repo_id, revision, name)
        try:
            blob = fetch(url)
        except (urllib.error.URLError, OSError) as exc:
            report["problems"].append("取回 %s 失败：%s: %s" % (name, type(exc).__name__, exc))
            continue
        path = os.path.join(out_dir, name)
        with open(path, "wb") as handle:
            handle.write(blob)
        measured[name] = {
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
            "url": url,
            "mirror_metadata_sha256": declared.get(name),
            "mirror_metadata_agrees": (
                None if name not in declared else declared[name] == sha256_bytes(blob)
            ),
        }
    report["files"] = measured
    report["file_sha256"] = {name: item["sha256"] for name, item in measured.items()}

    manifest = {
        "model_repo_id": repo_id,
        "model_revision": revision,
        "retrieved_at_utc": report["retrieved_at_utc"],
        "retrieval_command": report.get("capture_command") or _command_line(repo_id, revision, endpoint),
        "source": report["source"],
        "source_is_mirror": True,
        "retrieval_status": "ok" if measured else "failed",
        "file_sha256": report["file_sha256"],
        "files": measured,
        "metadata_sha256": report.get("metadata_sha256"),
        "consistency_check_only": True,
        "source_authentication_note": (
            "本目录里的文件字节取自同一个只读**镜像**，与**该镜像自己**的元数据清单"
            "互为一致只能算镜像自洽校验（consistency check only）；"
            "**不构成**对官方源 huggingface.co 的独立来源认证。"
        ),
        "downloaded_weight_files": [],
        "note": (
            "只取非权重小文件；revision 为精确 commit，不是 latest。"
            "本 manifest 不构成任何 verified 翻转依据。"
        ),
    }
    with open(os.path.join(out_dir, "evidence-manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    report["evidence_manifest"] = os.path.join(out_dir, "evidence-manifest.json")
    return report


def _command_line(repo_id: str, revision: str, endpoint: str) -> str:
    return (
        "python tools/capture_interface_pin_evidence.py --repo-id %s --revision %s "
        "--endpoint %s" % (repo_id, revision, endpoint)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="G9：按固定 revision 取回非权重小文件，产出可复核的接口 pin 证据目录"
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, help="证据目录（会创建）")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="只读端点，默认 hf-mirror.com")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args(argv)

    report = capture(
        repo_id=args.repo_id,
        revision=args.revision,
        out_dir=args.out,
        endpoint=args.endpoint,
        timeout=float(args.timeout),
    )
    report["capture_command"] = _command_line(args.repo_id, args.revision, args.endpoint)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out_report:
        with open(args.out_report, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)
    return 0 if report.get("files") else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
