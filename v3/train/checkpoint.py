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

#: 冻结验收规格：20 项 mask fixture 必须全部通过（KAGGLE-20 §4 验收项，
#: fixture 集见 `v3/train/fixtures.py::build_fixtures()`，共 20 项）。
#: 出处 = KAGGLE-20 §4「无效 adapter 不得静默过关」+ 20 项 mask fixture 的验收口径。
REQUIRED_FIXTURE_TOTAL = 20
REQUIRED_FIXTURE_PASS = 20


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
    #: 本状态来自哪个 checkpoint 目录（快照路径，供调用方核对）。
    checkpoint_dir: str | None = None
    #: save() 记录的上一个完整 optimizer 步（没有则为 None）。
    prev_complete_step: int | None = None
    #: 是否**真的**执行了回滚。中止在 accum 中途但无可用快照时必须是 False。
    rollback_applied: bool = False
    #: 回滚的来源目录（被放弃的那个 checkpoint）。
    rolled_back_from: str | None = None
    #: 回滚的目标目录（切过去的那份快照）。
    rolled_back_to: str | None = None

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
            "checkpoint_dir": self.checkpoint_dir,
            "prev_complete_step": self.prev_complete_step,
            "rollback_applied": self.rollback_applied,
            "rolled_back_from": self.rolled_back_from,
            "rolled_back_to": self.rolled_back_to,
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

    def _complete_steps(self) -> list[int]:
        """root 下所有带 COMPLETE 原子标记的 checkpoint 步号（升序）。"""
        if not os.path.isdir(self.root):
            return []
        steps: list[int] = []
        for name in os.listdir(self.root):
            if not name.startswith("step_"):
                continue
            directory = os.path.join(self.root, name)
            if not os.path.exists(os.path.join(directory, COMPLETE_MARKER)):
                continue
            try:
                steps.append(int(name[len("step_"):]))
            except ValueError:
                continue
        return sorted(steps)

    def previous_complete_dir(self, global_step: int) -> str | None:
        """严格小于 `global_step` 的最新一份完整快照目录；没有则 None。

        这是"回滚最近的完整 optimizer 步"唯一的合法落点：宁可返回 None
        （由调用方判定不可回滚），也不要拿一份不完整的 checkpoint 充数。
        """
        earlier = [step for step in self._complete_steps() if step < int(global_step)]
        return self.checkpoint_dir(earlier[-1]) if earlier else None

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
        # 先解析"上一个完整 optimizer 步"，再写新目录：这是 accum 中途中断时的回滚落点。
        prev_snapshot = self.previous_complete_dir(global_step)
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
            # 上一个完整 optimizer 步的快照引用（KAGGLE-20 §6：中断在 accum 中途要能回滚）。
            "prev_complete_step": (
                int(os.path.basename(prev_snapshot)[len("step_"):])
                if prev_snapshot is not None
                else None
            ),
            "prev_complete_dir": prev_snapshot,
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
            "prev_complete_step": manifest["prev_complete_step"],
            "prev_complete_dir": prev_snapshot,
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
            checkpoint_dir=path,
            prev_complete_step=(
                int(manifest["prev_complete_step"])
                if manifest.get("prev_complete_step") is not None
                else None
            ),
        )


def resume(
    store: CheckpointStore,
    path: str,
    current: Mapping,
    *,
    interrupted_mid_accum: bool = False,
) -> ResumeState:
    """恢复并强制校验环境哈希；不一致即拒绝续训。

    `interrupted_mid_accum=True` 时**真正回滚**到上一份完整 optimizer 步快照
    （即 `CheckpointStore.save()` 写进 manifest 的 `prev_complete_step` 指针）：
    `global_step` / `optimizer` / `cursor` / `consumed_input_tokens` /
    `consumed_supervised_tokens` / `accum_boundary` 全部换成该快照里的值，
    并置 `rollback_applied=True`、`rolled_back_from=<原 checkpoint 目录>`、
    `rolled_back_to=<快照目录>`。

    找不到可用快照（第一份 checkpoint、或快照已被 prune）时**不回滚**：返回原状态、
    `rollback_applied=False`、`rolled_back_from/rolled_back_to=None`，并在 `notes`
    里写明"未执行回滚"。调用方必须依据 `rollback_applied` 判定，不得假定一定回滚成功。
    """
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
    if not interrupted_mid_accum:
        return state

    snapshot_dir: str | None = None
    if state.prev_complete_step is not None:
        candidate = store.checkpoint_dir(state.prev_complete_step)
        if os.path.exists(os.path.join(candidate, COMPLETE_MARKER)):
            snapshot_dir = candidate
    if snapshot_dir is None:
        # 兼容旧 checkpoint（manifest 无指针）或指针指向的目录已不存在：重新扫描 root。
        snapshot_dir = store.previous_complete_dir(state.global_step)
    if snapshot_dir is None:
        state.rollback_applied = False
        state.rolled_back_from = None
        state.rolled_back_to = None
        state.notes.append(
            "中断发生在 accum 中途：找不到上一份完整 optimizer 步快照，未执行回滚"
            "（rollback_applied=False）；不得声称已回滚，最多损失一个 accum 窗口由调用方判定"
        )
        return state

    snapshot = store.load(snapshot_dir)
    snapshot_mismatch = {
        key: {"expected": snapshot.hashes.get(key), "actual": current.get(key)}
        for key in REQUIRED_HASH_KEYS
        if snapshot.hashes.get(key) != current.get(key)
    }
    if snapshot_mismatch:
        raise IntegrityError(
            "rollback_hash_mismatch",
            "回滚目标快照 %s 的环境/数据/配置哈希与当前不一致，拒绝回滚续训" % snapshot_dir,
            mismatch=snapshot_mismatch,
        )
    snapshot.rollback_applied = True
    snapshot.rolled_back_from = path
    snapshot.rolled_back_to = snapshot_dir
    snapshot.notes.append(
        "中断发生在 accum 中途：已回滚到上一份完整 optimizer 步快照 %s"
        "（global_step / optimizer / cursor / accum_boundary 全部取自该快照，最多损失一个 accum 窗口）"
        % snapshot_dir
    )
    return snapshot


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


def assert_adapter_valid(
    manifest: Mapping,
    *,
    load_ok: bool,
    params_changed: bool,
    fixture_pass: int,
    fixture_total: int,
    require_frozen_suite: bool = False,
) -> dict:
    """禁止「无效 adapter 静默过关」（KAGGLE-20 §4 验收项）。

    缺陷修复说明（原 `min_pass=7` 魔法默认值已删除）：

    - `fixture_pass` / `fixture_total` 非整数 → `MissingInput("fixture_count_not_integer")`；
    - 任一为负 → `PolicyViolation("fixture_count_negative")`；
    - `fixture_pass > fixture_total` → `PolicyViolation("fixture_count_inconsistent")`
      （原来的 `7/2` 会被静默放行）；
    - `fixture_total == REQUIRED_FIXTURE_TOTAL`（20）而
      `fixture_pass < REQUIRED_FIXTURE_PASS`（20）→
      `PolicyViolation("adapter_fixture_regression")`：**20 项 mask fixture 必须全过**；
    - 声明少于 20 项的自洽小集（历史 smoke 口径）必须**整份全绿**，否则走
      `PolicyViolation("adapter_invalid")`；返回值里 `frozen_spec_satisfied=False`，
      生产 preflight 应改用 `assert_adapter_valid_frozen_spec()`；
    - `require_frozen_suite=True` 时 `fixture_total` 必须**恰好**等于
      `REQUIRED_FIXTURE_TOTAL`，否则 `PolicyViolation("adapter_fixture_suite_incomplete")`。

    返回（未抛异常时）::

        {"adapter_valid": True, "load_ok": bool, "params_changed": bool,
         "fixture_pass": int, "fixture_total": int,
         "required_fixture_pass": 20, "required_fixture_total": 20,
         "frozen_spec_satisfied": bool, "matched_module_count": int, "note": str}
    """
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (fixture_pass, fixture_total)):
        raise MissingInput(
            "fixture_count_not_integer",
            "fixture_pass / fixture_total 必须是整数：%r" % ((fixture_pass, fixture_total),),
        )
    if fixture_pass < 0 or fixture_total < 0:
        raise PolicyViolation(
            "fixture_count_negative",
            "fixture 计数不得为负：fixture_pass=%d, fixture_total=%d"
            % (fixture_pass, fixture_total),
            fixture_pass=fixture_pass,
            fixture_total=fixture_total,
        )
    if fixture_pass > fixture_total:
        raise PolicyViolation(
            "fixture_count_inconsistent",
            "fixture_pass=%d > fixture_total=%d：计数自相矛盾，不得当作通过证据"
            % (fixture_pass, fixture_total),
            fixture_pass=fixture_pass,
            fixture_total=fixture_total,
        )
    if require_frozen_suite and fixture_total != REQUIRED_FIXTURE_TOTAL:
        raise PolicyViolation(
            "adapter_fixture_suite_incomplete",
            "冻结验收口径要求 fixture_total == %d，实际 %d"
            % (REQUIRED_FIXTURE_TOTAL, fixture_total),
            fixture_total=fixture_total,
            required_fixture_total=REQUIRED_FIXTURE_TOTAL,
        )
    if fixture_total > REQUIRED_FIXTURE_TOTAL:
        raise PolicyViolation(
            "adapter_fixture_suite_oversized",
            "fixture_total=%d 超过冻结口径 %d：不得用自定义大集合冒充验收证据"
            % (fixture_total, REQUIRED_FIXTURE_TOTAL),
            fixture_total=fixture_total,
            required_fixture_total=REQUIRED_FIXTURE_TOTAL,
        )
    if fixture_total == REQUIRED_FIXTURE_TOTAL and fixture_pass < REQUIRED_FIXTURE_PASS:
        raise PolicyViolation(
            "adapter_fixture_regression",
            "%d 项 mask fixture 必须全部通过：实际 %d/%d < %d/%d"
            % (
                REQUIRED_FIXTURE_TOTAL,
                fixture_pass,
                fixture_total,
                REQUIRED_FIXTURE_PASS,
                REQUIRED_FIXTURE_TOTAL,
            ),
            fixture_pass=fixture_pass,
            fixture_total=fixture_total,
            required_fixture_pass=REQUIRED_FIXTURE_PASS,
            required_fixture_total=REQUIRED_FIXTURE_TOTAL,
        )

    lora = manifest.get("lora") if isinstance(manifest, Mapping) else None
    matched_module_count = (lora or {}).get("matched_module_count", 0) if isinstance(lora, Mapping) else 0
    try:
        matched_module_count = int(matched_module_count)
    except (TypeError, ValueError):
        raise MissingInput(
            "matched_module_count_invalid",
            "manifest.lora.matched_module_count 不是整数：%r" % (matched_module_count,),
        )

    problems = []
    if not load_ok:
        problems.append("adapter 未成功加载")
    if not params_changed:
        problems.append("参数 delta 不可见：可能是零初始化/未路由")
    if fixture_total <= 0:
        problems.append("没有 fixture 证据")
    elif fixture_total < REQUIRED_FIXTURE_TOTAL and fixture_pass < fixture_total:
        problems.append(
            "声明了 %d 项的非冻结 fixture 集（< %d）时必须整份全绿：实际 %d/%d"
            % (fixture_total, REQUIRED_FIXTURE_TOTAL, fixture_pass, fixture_total)
        )
    if matched_module_count <= 0:
        problems.append("matched_module_count 为 0")
    if problems:
        raise PolicyViolation("adapter_invalid", "; ".join(problems), problems=problems)

    frozen_spec_satisfied = (
        fixture_total == REQUIRED_FIXTURE_TOTAL and fixture_pass >= REQUIRED_FIXTURE_PASS
    )
    return {
        "adapter_valid": True,
        "load_ok": bool(load_ok),
        "params_changed": bool(params_changed),
        "fixture_pass": fixture_pass,
        "fixture_total": fixture_total,
        "required_fixture_pass": REQUIRED_FIXTURE_PASS,
        "required_fixture_total": REQUIRED_FIXTURE_TOTAL,
        "frozen_spec_satisfied": frozen_spec_satisfied,
        "matched_module_count": matched_module_count,
        "note": (
            "冻结验收口径（20/20）已满足。"
            if frozen_spec_satisfied
            else "已通过自洽性检查，但未提供 %d/%d 的冻结验收证据：不得当作冻结验收通过。"
            % (REQUIRED_FIXTURE_PASS, REQUIRED_FIXTURE_TOTAL)
        ),
    }


def assert_adapter_valid_frozen_spec(
    manifest: Mapping,
    *,
    load_ok: bool,
    params_changed: bool,
    fixture_pass: int,
    fixture_total: int,
) -> dict:
    """生产 preflight 入口：必须提交 20/20 的冻结验收证据，缺一即阻断。

    与 `assert_adapter_valid(..., require_frozen_suite=True)` 完全等价，供
    `entry.preflight()` 直接接线（返回结构见 `assert_adapter_valid`）。
    """
    return assert_adapter_valid(
        manifest,
        load_ok=load_ok,
        params_changed=params_changed,
        fixture_pass=fixture_pass,
        fixture_total=fixture_total,
        require_frozen_suite=True,
    )
