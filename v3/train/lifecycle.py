"""G14：停止 / 恢复接线（中途 checkpoint、信号处理、终止收尾）。

## 缺陷出处

KAGGLE-32 v6 `GAPS-AND-OWNERS.md` §G14：

    runner.run_training 不处理信号、不落中途 checkpoint、不调用 CheckpointStore/resume()
    —— 对 runner.py grep resume|checkpoint|CheckpointStore|signal|SIGTERM|KeyboardInterrupt
       只有 `def run_training` 一处命中。

`v3/train/checkpoint.py` 里 `CheckpointStore` / `resume()` 早就写好了（含 accum 中途回滚），
但**没有任何生产接线**：保存的只是训练**结束后**的 adapter 副本。

## 本模块接线什么

1. :class:`StopRequest` —— `SIGTERM` / `SIGINT` 处理器 + 外部轮询标志 + 总截止
   （`deadline_monotonic`，monotonic 计时，墙钟只用于审计）。停止请求是**幂等**的：
   第一次触发记录原因，后续不再覆盖。
2. :class:`TrainingLifecycle` —— 把 `CheckpointStore` 接到训练循环上：

   - `begin(backend)`：有 `resume_from` 时调 `checkpoint.resume()`（强制环境哈希一致；
     `interrupted_mid_accum=True` 时按已有实现回滚到上一份完整 optimizer 步），
     然后把 optimizer / scheduler / scaler / RNG / 游标 / global_step 灌回后端；
   - `after_step(info)`：按 `checkpoint_every` 落中途 checkpoint（**含** optimizer、
     scheduler、scaler、三份 RNG、数据游标、global_step、已消费 token、accum 边界）；
   - `finalize(backend, ...)`：训练结束或停止时再落一份，并给出完整账目。

3. :func:`jsonify_tensors` / :func:`unjsonify_tensors` —— 把 torch 的
   `optimizer.state_dict()` 等含张量的结构转成规范 JSON（张量 base64 编码），
   因为 `CheckpointStore` 只写 JSON。**未在真实 GPU 上验证**，代码里如实标注。

保留判据：本模块**不**放宽任何原有门槛 —— mask fixture、基座冻结、adapter 实际路由
仍由 `run_training` 与 `train_steps` 判定，checkpoint 只是多存一份状态。
"""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..common.errors import Blocked, IntegrityError, MissingInput, PolicyViolation
from . import checkpoint as checkpoint_mod
from .checkpoint import REQUIRED_HASH_KEYS, CheckpointStore

#: 停止原因码。
STOP_SIGNAL = "signal"
STOP_DEADLINE = "deadline"
STOP_EXTERNAL = "external-request"
STOP_STEP_LIMIT = "step_limit"

#: `train_steps(on_step=...)` 的约定：返回非空字符串 = 立刻停止，该字符串是停止原因。
CONTINUE = None


@dataclass
class StopRequest:
    """停止请求（信号 / 截止时间 / 外部标志），幂等且可审计。"""

    signals: Sequence[int] = field(default_factory=lambda: _default_signals())
    deadline_monotonic: float | None = None
    #: 记录到报告里的总截止（墙钟，审计用；判定一律用 monotonic）。
    deadline_epoch: float | None = None
    _requested: bool = False
    _reason: str | None = None
    _detail: dict = field(default_factory=dict)
    _previous: dict = field(default_factory=dict)
    _installed: bool = False
    _monotonic = staticmethod(time.monotonic)

    # ---- 请求 ----
    def request(self, reason: str, **detail: object) -> bool:
        """发出停止请求；返回是否为**首次**触发（首次 True）。"""
        if self._requested:
            return False
        self._requested = True
        self._reason = str(reason)
        self._detail = dict(detail)
        return True

    @property
    def requested(self) -> bool:
        return self._requested

    @property
    def reason(self) -> str | None:
        return self._reason

    def _handler(self, signum, frame):  # pragma: no cover - 真实信号路径
        self.request(STOP_SIGNAL, signal=int(signum), signal_name=_signal_name(signum))

    # ---- 安装 / 卸载 ----
    def install(self) -> "StopRequest":
        """安装信号处理器。

        只允许在主线程安装（`signal.signal` 在子线程里会抛 ValueError）；子线程里
        调用相当于"只用截止时间与外部标志"，并记录 `installed=False`。
        """
        if self._installed:
            return self
        if threading.current_thread() is not threading.main_thread():
            self._detail["install_skipped"] = "not-main-thread"
            return self
        for signum in self.signals:
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handler)
            except (ValueError, OSError, RuntimeError) as exc:  # 平台不支持该信号
                self._previous.pop(signum, None)
                self._detail.setdefault("signal_install_errors", {})[_signal_name(signum)] = str(exc)
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        for signum, previous in self._previous.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError, RuntimeError):  # pragma: no cover
                pass
        self._installed = False
        self._previous = {}

    def __enter__(self) -> "StopRequest":
        return self.install()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.uninstall()
        return False

    # ---- 判定 ----
    def poll(self, *, now: float | None = None) -> str | None:
        """返回停止原因（无请求则 None）。截止时间到时**自动**发出请求。

        注意（KAGGLE-38 r3 缺陷 A）：`request()` 是**首次生效**的，一旦业务原因
        （`step_limit` / `signal`）先置位，本方法就只会返回那个业务原因，
        **不再看时钟**。所以"总截止"的硬判定不能只依赖本方法的返回值 ——
        用 :meth:`hard_deadline_exceeded` 独立判。
        """
        if not self._requested and self.deadline_monotonic is not None:
            moment = self._monotonic() if now is None else now
            if moment >= float(self.deadline_monotonic):
                self.request(
                    STOP_DEADLINE,
                    remaining_seconds=round(moment - float(self.deadline_monotonic), 3),
                )
        return self._reason

    def remaining_seconds(self, *, now: float | None = None) -> float | None:
        """距总截止还剩多少秒（无截止则 None）；已过期返回负数。"""
        if self.deadline_monotonic is None:
            return None
        moment = self._monotonic() if now is None else now
        return round(float(self.deadline_monotonic) - moment, 6)

    def hard_deadline_exceeded(self, *, now: float | None = None) -> bool:
        """**独立于业务停止原因**的硬截止判定（KAGGLE-38 r3 缺陷 A）。

        这是修复 `BudgetLedger.check()` 只看 `poll() == deadline` 的那个洞：
        `step_limit` 先置位时 `poll()` 永远返回 `step_limit`，于是越过总截止
        仍然放行了下一阶段（实测 remaining = -1 s）。硬截止必须自己看时钟。
        """
        remaining = self.remaining_seconds(now=now)
        return remaining is not None and remaining <= 0.0

    def to_dict(self) -> dict:
        return {
            "requested": self._requested,
            "reason": self._reason,
            "detail": dict(self._detail),
            "deadline_monotonic": self.deadline_monotonic,
            "deadline_epoch": self.deadline_epoch,
            "installed": self._installed,
            "signals": sorted(_signal_name(item) for item in self.signals),
        }


def _signal_name(signum) -> str:
    try:
        return signal.Signals(signum).name
    except Exception:  # noqa: BLE001 - 平台可能没有该枚举
        return str(signum)


def _default_signals() -> tuple[int, ...]:
    """默认监听 SIGTERM + SIGINT（`SIGBREAK` 在 Windows 上等价于 Ctrl-Break）。"""
    names = ("SIGTERM", "SIGINT", "SIGBREAK")
    found: list[int] = []
    for name in names:
        value = getattr(signal, name, None)
        if isinstance(value, int) and value not in found:
            found.append(value)
    return tuple(found)


# ------------------------------------------------------------------ 张量编解码


def jsonify_tensors(value, *, where: str = "state") -> object:
    """把含张量的嵌套结构转成规范 JSON（张量 → base64 字典）。

    张量判据是**鸭子类型**：有 `detach()`、`cpu()`、`numpy()`、`shape`、`dtype`
    且 numpy 结果有 `tobytes()`。没有 torch 也能测（测试里用替身对象）。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonify_tensors(item, where="%s.%s" % (where, key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify_tensors(item, where="%s[%d]" % (where, index)) for index, item in enumerate(value)]
    tensor = _as_tensor_payload(value, where=where)
    if tensor is not None:
        return tensor
    raise Blocked(
        "checkpoint_state_not_serializable",
        "状态里出现无法序列化的对象（%s）：%s" % (where, type(value).__name__),
    )


def _as_tensor_payload(value, *, where: str) -> dict | None:
    required = ("detach", "cpu", "numpy", "shape", "dtype")
    if not all(hasattr(value, name) for name in required):
        return None
    array = value.detach().cpu().numpy()
    if not hasattr(array, "tobytes"):
        return None
    return {
        "__tensor__": {
            "dtype": str(value.dtype),
            "shape": [int(item) for item in tuple(value.shape)],
            "numpy_dtype": str(getattr(array, "dtype", "")),
            "data_base64": base64.b64encode(array.tobytes()).decode("ascii"),
            "where": where,
        }
    }


def unjsonify_tensors(value, *, decoder=None) -> object:
    """`jsonify_tensors` 的逆运算；`decoder(payload) -> 张量` 省略时需 torch。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if set(value.keys()) == {"__tensor__"}:
            payload = dict(value["__tensor__"])
            if decoder is None:
                decoder = _torch_tensor_decoder()
            return decoder(payload)
        return {key: unjsonify_tensors(item, decoder=decoder) for key, item in value.items()}
    if isinstance(value, list):
        return [unjsonify_tensors(item, decoder=decoder) for item in value]
    raise Blocked(
        "checkpoint_state_not_decodable",
        "JSON 里出现无法反序列化的值：%s" % type(value).__name__,
    )


def _torch_tensor_decoder():  # pragma: no cover - 需要真实 torch
    try:
        import numpy  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise Blocked(
            "torch_unavailable_for_state_decode",
            "解码张量状态需要 torch/numpy，本机不可用：%s" % exc,
        ) from exc

    def decode(payload: Mapping):
        raw = base64.b64decode(str(payload["data_base64"]))
        array = numpy.frombuffer(raw, dtype=numpy.dtype(str(payload.get("numpy_dtype") or "float32")))
        shape = tuple(int(item) for item in payload.get("shape") or ())
        if shape:
            array = array.reshape(shape)
        return torch.from_numpy(array.copy())

    return decode


# ------------------------------------------------------------------ 生命周期


@dataclass
class LifecyclePolicy:
    """checkpoint 节奏与停止点。"""

    checkpoint_every: int = 0
    """每多少 optimizer 步落一份中途 checkpoint；0 表示只在停止/结束时落。"""

    stop_at_step: int | None = None
    """硬停止点（到达该步后立刻停止并保存），用于"同一总截止内停机"。"""

    keep_last: int = checkpoint_mod.KEEP_LAST

    def validate(self) -> None:
        if self.checkpoint_every < 0:
            raise PolicyViolation("checkpoint_every_negative", "checkpoint_every 不得为负")
        if self.stop_at_step is not None and self.stop_at_step <= 0:
            raise PolicyViolation("stop_at_step_invalid", "stop_at_step 必须为正")


@dataclass
class TrainingLifecycle:
    """把 `CheckpointStore` 接到训练循环上（保存 / 恢复 / 停止收尾）。"""

    store_root: str
    hashes: Mapping
    policy: LifecyclePolicy = field(default_factory=LifecyclePolicy)
    resume_from: str | None = None
    interrupted_mid_accum: bool = False
    stop: StopRequest | None = None
    save_on_stop: bool = True
    keep_last: int = checkpoint_mod.KEEP_LAST

    #: 运行时账目。
    checkpoints: list[dict] = field(default_factory=list)
    resume_report: dict | None = None
    stop_reason: str | None = None
    saved_on_stop: dict | None = None
    resumed_step: int | None = None
    _store: CheckpointStore | None = None

    def __post_init__(self) -> None:
        self.policy.validate()
        missing = [key for key in REQUIRED_HASH_KEYS if not self.hashes.get(key)]
        if missing:
            raise MissingInput(
                "lifecycle_missing_hashes",
                "启用 checkpoint 必须提供全部环境/数据/配置哈希（否则续训无法判同源）",
                missing=missing,
            )
        self._store = CheckpointStore(root=self.store_root, keep_last=self.keep_last)

    @property
    def store(self) -> CheckpointStore:
        assert self._store is not None
        return self._store

    # ---- 恢复 ----
    def begin(self, backend) -> dict:
        """恢复（如有）并返回恢复报告；无 resume_from 时返回 `{"resumed": False}`。"""
        if not hasattr(backend, "import_training_state"):
            raise PolicyViolation(
                "backend_cannot_resume",
                "后端 %s 没有 import_training_state()：不得假装已接续（fail-closed）"
                % type(backend).__name__,
            )
        if not self.resume_from:
            self.resume_report = {"resumed": False, "reason": "未提供 resume_from（全新开训）"}
            return self.resume_report
        state = checkpoint_mod.resume(
            self.store,
            self.resume_from,
            dict(self.hashes),
            interrupted_mid_accum=self.interrupted_mid_accum,
        )
        applied = backend.import_training_state(state)
        self.resumed_step = int(state.global_step)
        self.resume_report = {
            "resumed": True,
            "checkpoint_dir": state.checkpoint_dir,
            "global_step": int(state.global_step),
            "prev_complete_step": state.prev_complete_step,
            "rollback_applied": bool(state.rollback_applied),
            "rolled_back_from": state.rolled_back_from,
            "rolled_back_to": state.rolled_back_to,
            "notes": list(state.notes),
            "hash_check": "pass",
            "backend_import": applied,
        }
        return self.resume_report

    # ---- 逐步钩子 ----
    def after_step(self, backend, info: Mapping) -> str | None:
        """训练循环每步调用：按节奏落 checkpoint，并返回停止原因（None = 继续）。"""
        step = int(info.get("step") or 0)
        if self.policy.checkpoint_every and step % self.policy.checkpoint_every == 0:
            self.save_checkpoint(backend, step=step, reason="interval", info=info)
        reason = self.stop.poll() if self.stop is not None else None
        if reason is None and self.policy.stop_at_step is not None and step >= self.policy.stop_at_step:
            if self.stop is not None:
                self.stop.request(STOP_STEP_LIMIT, stop_at_step=self.policy.stop_at_step)
            reason = STOP_STEP_LIMIT
        if reason is not None:
            self.stop_reason = reason
        return reason

    # ---- 保存 ----
    def save_checkpoint(self, backend, *, step: int, reason: str, info: Mapping | None = None) -> dict:
        exporter = getattr(backend, "export_training_state", None)
        if exporter is None:
            raise PolicyViolation(
                "backend_cannot_export_state",
                "后端 %s 没有 export_training_state()：无法落中途 checkpoint（fail-closed）"
                % type(backend).__name__,
            )
        bytes_exporter = getattr(backend, "export_adapter_bytes", None)
        if bytes_exporter is None:
            raise PolicyViolation(
                "backend_cannot_export_adapter_bytes",
                "后端 %s 没有 export_adapter_bytes()：checkpoint 里的 adapter 副本无法产出"
                % type(backend).__name__,
            )
        state = dict(exporter())
        adapter_bytes = bytes_exporter()
        for key in ("optimizer", "rng", "cursor", "consumed_input_tokens",
                    "consumed_supervised_tokens", "accum_boundary"):
            if key not in state:
                raise MissingInput(
                    "backend_state_incomplete",
                    "后端导出的训练状态缺少 %r：续训不可只恢复其一" % key,
                    missing=[key],
                )
        saved = self.store.save(
            global_step=int(step),
            adapter_bytes=adapter_bytes,
            optimizer=state["optimizer"],
            rng=state["rng"],
            cursor=state["cursor"],
            hashes=dict(self.hashes),
            consumed_input_tokens=int(state["consumed_input_tokens"]),
            consumed_supervised_tokens=int(state["consumed_supervised_tokens"]),
            accum_boundary=int(state["accum_boundary"]),
            optional_state={
                "scheduler": state.get("scheduler"),
                "scaler": state.get("scaler"),
                "global_step_from_backend": int(state.get("global_step", step)),
                "rng_scope": state.get("rng_scope"),
                "tensor_encoding": state.get("tensor_encoding"),
            },
        )
        record = {
            "step": int(step),
            "reason": str(reason),
            "checkpoint_dir": saved["checkpoint_dir"],
            "manifest_sha256": saved["manifest_sha256"],
            "adapter_sha256": saved["adapter_sha256"],
            "prev_complete_step": saved["prev_complete_step"],
            "loss": (info or {}).get("loss"),
            "grad_norm": (info or {}).get("grad_norm"),
            "has_scheduler": state.get("scheduler") is not None,
            "has_scaler": state.get("scaler") is not None,
            "rng_scope": state.get("rng_scope"),
            "cursor_position": (state.get("cursor") or {}).get("position"),
            "consumed_input_tokens": int(state["consumed_input_tokens"]),
            "consumed_supervised_tokens": int(state["consumed_supervised_tokens"]),
            "accum_boundary": int(state["accum_boundary"]),
        }
        self.checkpoints.append(record)
        return record

    def finalize(self, backend, *, step: int, trained: Mapping) -> dict:
        """训练结束/停止后的收尾：必要时再落一份，并给出生命周期账目。"""
        if self.stop is not None:
            self.stop.poll()
        if (self.stop is not None and self.stop.requested) or self.stop_reason is not None:
            self.stop_reason = self.stop_reason or (self.stop.reason if self.stop else None)
            if self.save_on_stop:
                self.saved_on_stop = self.save_checkpoint(
                    backend,
                    step=int(step),
                    reason="stop:%s" % (self.stop_reason or "unknown"),
                    info={"optimizer_step_count": trained.get("optimizer_step_count")},
                )
        elif trained.get("optimizer_step_count"):
            self.save_checkpoint(
                backend,
                step=int(step),
                reason="final",
                info={"optimizer_step_count": trained.get("optimizer_step_count")},
            )
        return {
            "enabled": True,
            "store_root": self.store.root,
            "keep_last": self.keep_last,
            "policy": {
                "checkpoint_every": self.policy.checkpoint_every,
                "stop_at_step": self.policy.stop_at_step,
            },
            "resume": self.resume_report,
            "resumed_step": self.resumed_step,
            "checkpoint_count": len(self.checkpoints),
            "checkpoints": list(self.checkpoints),
            "saved_on_stop": self.saved_on_stop,
            "stop": self.stop.to_dict() if self.stop is not None else None,
            "stop_reason": self.stop_reason,
            "stopped": bool(self.stop_reason),
            "honest_scope": (
                "checkpoint 只证明状态被写盘并通过逐文件哈希校验；"
                "**不**证明它能在真实 GPU 上恢复训练（需现场验证）。"
            ),
        }


def verify_checkpoint_roundtrip(store_root: str, expect_step: int, hashes: Mapping) -> dict:
    """从盘上读回最近一份 checkpoint 并核对哈希（工程检查用的正例）。"""
    store = CheckpointStore(root=store_root, keep_last=checkpoint_mod.KEEP_LAST)
    complete = store._complete_steps()
    if not complete:
        raise IntegrityError("no_complete_checkpoint", "checkpoint 目录里没有完整快照")
    step = complete[-1]
    state = checkpoint_mod.resume(store, store.checkpoint_dir(step), dict(hashes))
    if expect_step and int(state.global_step) != int(expect_step):
        raise IntegrityError(
            "checkpoint_step_mismatch",
            "读回的 global_step 与期望不符",
            expected=expect_step,
            actual=state.global_step,
        )
    return {
        "checkpoint_dir": state.checkpoint_dir,
        "global_step": int(state.global_step),
        "adapter_sha256": state.adapter_sha256,
        "prev_complete_step": state.prev_complete_step,
        "consumed_input_tokens": int(state.consumed_input_tokens),
        "consumed_supervised_tokens": int(state.consumed_supervised_tokens),
        "accum_boundary": int(state.accum_boundary),
        "hash_check": "pass",
        "complete_steps": complete,
    }


#: 只接受 64 位**小写** hex。KAGGLE-32 v6 `shard_guard.py::HEX64_RE` 同口径。
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def hashes_from_json(path: str) -> dict:
    """从 JSON 文件读 5 项环境哈希（107 作业里由上游脚本产出）。

    独立审查（KAGGLE-38 评论 01a11aa9…，阻断 2）实测：五个字段都填字符串 ``"1"``
    时旧实现照样返回成功——「真实哈希」实际上只是**调用方自报**，既没有格式校验，
    也没有和批准记录绑定。这里先补最小的一层：**逐字段严格 hex64 格式校验**。

    格式校验不是授权。真实模式还必须在加载前核对 v6 批准记录
    （见 ``v3/train/v6_approval.py``）；本函数只负责「这不是随手填的字符串」。
    """
    if not os.path.exists(path):
        raise MissingInput("hashes_file_missing", "哈希文件不存在：%s" % path)
    with open(path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise MissingInput("hashes_file_invalid", "哈希文件不是 JSON 对象：%s" % path)
    missing = [key for key in REQUIRED_HASH_KEYS if not payload.get(key)]
    if missing:
        raise MissingInput(
            "hashes_file_incomplete", "哈希文件缺少键：%s" % missing, missing=missing
        )
    malformed = []
    for key in REQUIRED_HASH_KEYS:
        value = payload[key]
        if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
            malformed.append("%s=%r" % (key, value))
    if malformed:
        raise IntegrityError(
            "hashes_not_sha256_hex64",
            "哈希字段必须是 64 位小写 hex（拒绝自报值/占位值/大小写混写）：%s"
            % ", ".join(malformed),
            malformed=malformed,
            required_keys=list(REQUIRED_HASH_KEYS),
        )
    return {key: str(payload[key]) for key in REQUIRED_HASH_KEYS}
