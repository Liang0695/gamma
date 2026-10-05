"""adapter-only 断点续训与导出 manifest（KAGGLE-20 §6，KAGGLE-19 §3 4h 续训）。

规则：
- 保存内容：adapter safetensors + optimizer/scheduler + Python/NumPy/torch RNG +
  sampler 顺序与 cursor + global_step + 已消费输入/监督 token + 梯度 accum 边界 +
  source/data/config/code/依赖 SHA；
- **不保存完整基座或 FP32 主权重**；内部 optimizer 状态不进提交包；
- 写临时目录 → 校验 → **原子标记 complete**；
- 下个 4h 作业必须重建同 revision 基座、恢复全部状态并**拒绝哈希不一致**；
- 断点续训不是"重新加载 adapter 再起一个新 optimizer"；
- 中断发生在 accum 中途则**回滚最近完整步**，最多损失一个 accum 窗口且日志注明；
- **数据游标和 RNG 不可只恢复其一**。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Mapping

from ..common.canonical import canonical_json_bytes, sha256_bytes, sha256_json
from ..common.errors import IntegrityError, MissingInput, PolicyViolation

#: 提交包总量上限（KAGGLE-22 §8.5：总包 <3 GiB）。
MAX_PACKAGE_BYTES = 3 * (1 << 30)
#: 小 checkpoint 保留数量。
KEEP_LAST = 2

COMPLETE_MARKER = "COMPLETE"
MANIFEST_NAME = "checkpoint_manifest.json"
ADAPTER_NAME = "adapter.safetensors"
OPTIMIZER_NAME = "optimizer_state.json"
RNG_NAME = "rng_state.json"
CURSOR_NAME = "sampler_cursor.json"

#: 禁止出现在 checkpoint 里的整模型标记。
FORBIDDEN_KEY_MARKERS = (
    "base_model",
    "model_state_dict",
    "full_model",
    "fp32_master",
    "dequantized",
)

REQUIRED_HASH_KEYS = ("source_sha256", "data_sha256", "config_sha256", "code_sha256", "deps_sha256")


@dataclass
class ResumeState:
    global_step: int
    consumed_input_tokens: int
    consumed_supervised_tokens: int
    accum_boundary: int
    adapter_sha256: str
    optimizer: dict
    rng: dict
    cursor: dict
    hashes: dict
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "consumed_input_tokens": self.consumed_input_tokens,
            "consumed_supervised_tokens": self.consumed_supervised_tokens,
            "accum_boundary": self.accum_boundary,
            "adapter_sha256": self.adapter_sha256,
            "optimizer": dict(self.optimizer),
            "rng": dict(self.rng),
            "cursor": dict(self.cursor),
            "hashes": dict(self.hashes),
            "notes": list(self.notes),
        }


def _reject_full_model_payload(payload: Mapping, where: str) -> None:
    def walk(node, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                lowered = str(key).lower()
                if any(marker in lowered for marker in FORBIDDEN_KEY_MARKERS):
                    raise PolicyViolation(
                        "checkpoint_contains_full_model",
                        "checkpoint 里出现整模型标记 %s（%s）" % (key, path),
                    )
                walk(value, "%s.%s" % (path, key))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, "%s[%d]" % (path, index))

    walk(payload, where)


@dataclass
class CheckpointStore:
    """本地 checkpoint 目录（每次保存一个子目录 + 原子 complete 标记）。"""

    root: str
    keep_last: int = KEEP_LAST

    def checkpoint_dir(self, global_step: int) -> str:
        return os.path.join(self.root, "step_%06d" % int(global_step))

    # ---- 保存 ----

    def save(
        self,
        *,
        global_step: int,
        adapter_bytes: bytes,
        optimizer: Mapping,
        rng: Mapping,
        cursor: Mapping,
        hashes: Mapping,
        consumed_input_tokens: int,
        consumed_supervised_tokens: int,
        accum_boundary: int,
        optional_state: Mapping | None = None,
    ) -> dict:
        missing = [key for key in REQUIRED_HASH_KEYS if key not in hashes]
        if missing:
            raise MissingInput(
                "checkpoint_missing_hashes", "checkpoint 缺少哈希", missing=missing
            )
        if not rng.get("python_state") or not rng.get("numpy_state") or not rng.get("torch_state"):
            raise MissingInput(
                "rng_incomplete",
                "RNG 必须同时包含 python / numpy / torch 三份状态（不可只恢复其一）",
            )
        if not cursor.get("sampler_order_sha256") or cursor.get("position") is None:
            raise MissingInput(
                "cursor_incomplete", "数据游标必须含 sampler_order_sha256 与 position"
            )
        if len(adapter_bytes) > MAX_PACKAGE_BYTES:
            raise PolicyViolation(
                "adapter_too_large", "adapter 超过 3GiB 提交包上限"
            )
        _reject_full_model_payload(optimizer, "optimizer")
        _reject_full_model_payload(rng, "rng")
        _reject_full_model_payload(dict(optional_state or {}), "optional_state")

        target = self.checkpoint_dir(global_step)
        temp = target + ".tmp"
        if os.path.exists(temp):
            shutil.rmtree(temp)
        os.makedirs(temp, exist_ok=True)

        adapter_sha = sha256_bytes(adapter_bytes)
        with open(os.path.join(temp, ADAPTER_NAME), "wb") as handle:
            handle.write(adapter_bytes)
        for name, payload in (
            (OPTIMIZER_NAME, dict(optimizer)),
            (RNG_NAME, dict(rng)),
            (CURSOR_NAME, dict(cursor)),
        ):
            with open(os.path.join(temp, name), "wb") as handle:
                handle.write(canonical_json_bytes(payload))

        manifest = {
            "global_step": int(global_step),
            "consumed_input_tokens": int(consumed_input_tokens),
            "consumed_supervised_tokens": int(consumed_supervised_tokens),
            "accum_boundary": int(accum_boundary),
            "adapter_sha256": adapter_sha,
            "adapter_bytes": len(adapter_bytes),
            "hashes": dict(hashes),
            "files": {
                ADAPTER_NAME: sha256_bytes(adapter_bytes),
                OPTIMIZER_NAME: sha256_json(dict(optimizer)),
                RNG_NAME: sha256_json(dict(rng)),
                CURSOR_NAME: sha256_json(dict(cursor)),
            },
            "policy": "adapter-only：不含基座权重，不含 FP32 主权重，optimizer 状态不进提交包",
        }
        manifest_sha = sha256_json(manifest)
        with open(os.path.join(temp, MANIFEST_NAME), "wb") as handle:
            handle.write(canonical_json_bytes(manifest))

        # 校验：逐文件回读比对哈希。
        self._verify_dir(temp, manifest)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.rename(temp, target)
        with open(os.path.join(target, COMPLETE_MARKER), "wb") as handle:
            handle.write(canonical_json_bytes({"manifest_sha256": manifest_sha}))
        self._prune()
        return {
            "checkpoint_dir": target,
            "manifest_sha256": manifest_sha,
            "adapter_sha256": adapter_sha,
            "adapter_bytes": len(adapter_bytes),
        }

    # ---- 校验 ----

    def _verify_dir(self, path: str, manifest: Mapping) -> None:
        for name, expected in (manifest.get("files") or {}).items():
            full = os.path.join(path, name)
            if not os.path.exists(full):
                raise IntegrityError("checkpoint_file_missing", "checkpoint 缺少 %s" % name)
            with open(full, "rb") as handle:
                payload = handle.read()
            actual = sha256_bytes(payload) if name == ADAPTER_NAME else sha256_json(json.loads(payload))
            if actual != expected:
                raise IntegrityError(
                    "checkpoint_file_hash_mismatch",
                    "checkpoint 文件 %s 哈希不一致" % name,
                    expected=expected,
                    actual=actual,
                )

    def _prune(self) -> None:
        if not os.path.isdir(self.root):
            return
        complete = sorted(
            name
            for name in os.listdir(self.root)
            if name.startswith("step_") and os.path.exists(os.path.join(self.root, name, COMPLETE_MARKER))
        )
        for name in complete[: max(0, len(complete) - self.keep_last)]:
            shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)

    # ---- 恢复 ----

    def load(self, path: str) -> ResumeState:
        marker = os.path.join(path, COMPLETE_MARKER)
        if not os.path.exists(marker):
            raise IntegrityError(
                "checkpoint_incomplete",
                "checkpoint 没有 COMPLETE 原子标记：视为未完成，不得用于续训",
            )
        with open(os.path.join(path, MANIFEST_NAME), "rb") as handle:
            manifest = json.loads(handle.read())
        with open(marker, "rb") as handle:
            marker_payload = json.loads(handle.read())
        if marker_payload.get("manifest_sha256") != sha256_json(manifest):
            raise IntegrityError("checkpoint_manifest_mismatch", "COMPLETE 标记与 manifest 不匹配")
        self._verify_dir(path, manifest)
        with open(os.path.join(path, ADAPTER_NAME), "rb") as handle:
            adapter_sha = sha256_bytes(handle.read())
        if adapter_sha != manifest["adapter_sha256"]:
            raise IntegrityError("adapter_hash_mismatch", "adapter 哈希与 manifest 不一致")
        with open(os.path.join(path, OPTIMIZER_NAME), "rb") as handle:
            optimizer = json.loads(handle.read())
        with open(os.path.join(path, RNG_NAME), "rb") as handle:
            rng = json.loads(handle.read())
        with open(os.path.join(path, CURSOR_NAME), "rb") as handle:
            cursor = json.loads(handle.read())
        return ResumeState(
            global_step=int(manifest["global_step"]),
            consumed_input_tokens=int(manifest["consumed_input_tokens"]),
            consumed_supervised_tokens=int(manifest["consumed_supervised_tokens"]),
            accum_boundary=int(manifest["accum_boundary"]),
            adapter_sha256=adapter_sha,
            optimizer=optimizer,
            rng=rng,
            cursor=cursor,
            hashes=dict(manifest["hashes"]),
        )


def resume(
    store: CheckpointStore,
    path: str,
    current: Mapping,
    *,
    interrupted_mid_accum: bool = False,
) -> ResumeState:
    """恢复并强制校验环境哈希；不一致即拒绝续训。"""
    state = store.load(path)
    mismatch = {
        key: {"expected": state.hashes.get(key), "actual": current.get(key)}
        for key in REQUIRED_HASH_KEYS
        if state.hashes.get(key) != current.get(key)
    }
    if mismatch:
        raise IntegrityError(
            "resume_hash_mismatch",
            "环境/数据/配置哈希与 checkpoint 不一致，拒绝续训",
            mismatch=mismatch,
        )
    if interrupted_mid_accum:
        state.notes.append(
            "中断发生在 accum 中途：回滚最近的完整 optimizer 步，最多损失一个 accum 窗口"
        )
        state.accum_boundary = 0
    return state


def build_export_manifest(
    *,
    adapter_sha256: str,
    rank: int,
    target_modules_regex: str,
    matched_module_count: int,
    base_repo_id: str,
    base_revision: str,
    lora_alpha: int,
    lora_dropout: float,
    training_hashes: Mapping,
    known_ranking: Mapping | None = None,
    vllm_mapper_revision: str | None = None,
) -> dict:
    """导出 manifest：记录 rank / target_modules / base revision（KAGGLE-22 §8.5 要求）。"""
    missing = [key for key in REQUIRED_HASH_KEYS if key not in training_hashes]
    if missing:
        raise MissingInput("export_missing_hashes", "导出 manifest 缺少哈希", missing=missing)
    if not vllm_mapper_revision:
        raise MissingInput(
            "mapper_revision_unknown",
            "vLLM 实际 mapper revision 未知：不得声称加载兼容（fail-closed）",
        )
    if matched_module_count <= 0:
        raise PolicyViolation("export_no_modules", "target 模块数为 0，无效 adapter 不得通过")
    return {
        "format": "v3-adapter-export/1",
        "adapter_sha256": adapter_sha256,
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
        },
        "training_hashes": dict(training_hashes),
        "known_ranking": dict(known_ranking or {}),
        "note": "加载成功不等于有效；本 manifest 不构成评分端兼容证据。",
    }


def assert_adapter_valid(manifest: Mapping, *, load_ok: bool, params_changed: bool, fixture_pass: int, fixture_total: int, min_pass: int = 7) -> None:
    """禁止「无效 adapter 静默过关」。"""
    problems = []
    if not load_ok:
        problems.append("adapter 未成功加载")
    if not params_changed:
        problems.append("参数 delta 不可见：可能是零初始化/未路由")
    if fixture_total <= 0:
        problems.append("没有 fixture 证据")
    elif fixture_pass < min_pass:
        problems.append("fixture 目标动作通过 %d/%d < %d" % (fixture_pass, fixture_total, min_pass))
    if int((manifest.get("lora") or {}).get("matched_module_count", 0)) <= 0:
        problems.append("matched_module_count 为 0")
    if problems:
        raise PolicyViolation("adapter_invalid", "; ".join(problems), problems=problems)
