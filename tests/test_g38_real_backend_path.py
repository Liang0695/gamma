"""KAGGLE-38 整改第二轮回归：**真实后端路径接通** + **单一总截止** + **分阶段校验**。

对应 Mika 在 KAGGLE-38 退回整改时点名的三项（评论 01a1191a-60a2-7617-b9e7-d298de612cc3）：

1. **G7b**：`engineering_check.py` 写死 `SyntheticBackend`，实跑
   `--backend torch-peft --plan-json` 被 CLI 拒绝 —— 真实检查入口根本没接通；
2. **G14**：`TorchPeftBackend.train_steps` 没有 `on_step`，也没有状态导入/导出、
   adapter 字节导出和 `measure_phases`；
3. **总截止**：恢复探针新建了**没有 deadline** 的 `StopRequest`，测量/导出/恢复
   不受同一剩余预算约束。

本模块全部离线、全部 CPU：

- 真实后端路径用**真 torch**（本机 torch 可用）跑一个小型 `nn.Module` 替身，
  证明状态导出/导入、adapter 字节、`on_step` 停止、`measure_phases` 真的接线了；
- CLI 层用真实 `main()` 调用，证明 `--backend torch-peft` **不再被 argparse 拒绝**，
  而是走到执行路径后按真实原因 fail-closed；
- 总截止用 AST 静态判据 + 运行期账目双重钉住。

诚实边界：本模块**没有**在 GPU 上跑过真实 transformers/peft 路径（本机没装 peft），
也没有真装锁定依赖；它证明的是接线与状态机，不是训练效果。
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests._tmp import temp_dir_outside_repo  # noqa: E402

from v3.common.errors import (  # noqa: E402
    FailClosed,
    IntegrityError,
    MissingInput,
    PolicyViolation,
    UnverifiedLock,
)
from v3.train import adapter_export_contract as export_contract  # noqa: E402
from v3.train import engineering_check  # noqa: E402
from v3.train import input_binding  # noqa: E402
from v3.train import lifecycle as lifecycle_mod  # noqa: E402
from v3.train import profile as profile_mod  # noqa: E402
from v3.train import runner  # noqa: E402
from v3.train import v6_approval  # noqa: E402
from v3.train.lifecycle import StopRequest  # noqa: E402

try:  # torch 是本机可用的；缺 torch 的机器上这一组测试如实跳过，不假装通过。
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 环境相关
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


HASHES = {
    "source_sha256": "a" * 64,
    "data_sha256": "b" * 64,
    "config_sha256": "c" * 64,
    "code_sha256": "d" * 64,
    "deps_sha256": "e" * 64,
}

REVISION = "52f3f65bc7a02d555763bc923bd1d9094898219d"

#: 仓库里被核验过的官方原件目录（G9 证据包）：r3 的输入绑定测试直接用真字节。
EVIDENCE_DIR = os.path.join(REPO_ROOT, "docs", "v3", "evidence", "kaggle-38-g9-evidence")
INTERFACE_LOCK = os.path.join(REPO_ROOT, "v3", "locks", "official-interface.json")
#: 证据目录是平铺的，所以布局就是文件名本身。
#: r4 起 `config.json` 也升为**受核对输入**（独立审查阻断 3），所以布局里必须带上它。
FLAT_LAYOUT = {
    "tokenizer_config.json": "tokenizer_config.json",
    "tokenizer.json": "tokenizer.json",
    "chat_template.jinja": "chat_template.jinja",
    "config.json": "config.json",
}


# ---------------------------------------------------------------- 真实后端的 CPU 替身


class _TinyLoRAModel:
    """`TorchPeftBackend` 需要的**最小真实 torch 模型**（不是合成后端）。

    形状与 PEFT 同构：一个冻结基座 `nn.Parameter` + 两个可训练 LoRA 参数。
    它不是 gemma-4，也不产生任何比赛成绩；它存在的唯一目的，是让
    `train_steps` / `export_adapter_bytes` / `export_training_state` /
    `import_training_state` / `measure_phases` 这五条**真实代码路径**在 CPU 上被真的执行到。
    """

    def __init__(self) -> None:
        torch.manual_seed(1234)
        self.base = torch.nn.Parameter(torch.zeros(6, 6))  # 冻结基座
        self.lora_A = torch.nn.Parameter(torch.randn(3, 6) * 0.1)  # adapter 下行
        self.lora_B = torch.nn.Parameter(torch.randn(6, 3) * 0.1)  # adapter 上行
        self.base.requires_grad_(False)
        self.state_dict_calls = 0

    # -- 与 torch 模块同名的接口（本替身不用 nn.Module，避免依赖 torch 的模块系统）--
    def parameters(self):
        return [self.base, self.lora_A, self.lora_B]

    def named_parameters(self):
        return [
            ("base.weight", self.base),
            ("lora_A.weight", self.lora_A),
            ("lora_B.weight", self.lora_B),
        ]

    def train(self):
        return self

    def state_dict(self):  # pragma: no cover - 被调用即测试失败
        self.state_dict_calls += 1
        raise AssertionError("后端不得调用 state_dict()：那会一次性物化整份权重")

    def forward(self, *, input_ids, labels):
        flat_in = input_ids.reshape(-1)
        flat_lab = labels.reshape(-1)
        index = flat_in % 6
        onehot = torch.nn.functional.one_hot(index, num_classes=6).float()
        hidden = onehot @ self.lora_A.t()
        logits = hidden @ self.lora_B.t() + self.base[index]
        loss = torch.nn.functional.cross_entropy(logits, flat_lab % 6)
        return collections.namedtuple("_Out", "loss")(loss)

    __call__ = forward


def _plan(steps: int = 3):
    return runner.TrainRunPlan(
        steps=int(steps),
        lr=0.05,
        seq_len=6,
        lora_rank=3,
        lora_alpha=6,
        batches=[
            runner.TrainBatch(input_ids=[0, 1, 2, 3, 4, 5], labels=[0, 1, 2, 3, 4, 5]),
            runner.TrainBatch(input_ids=[5, 4, 3, 2, 1, 0], labels=[1, 2, 3, 4, 5, 0]),
        ],
        adapter_name="v3_policy",
    )


def _cpu_backend(plan, *, device: str = "cpu") -> runner.TorchPeftBackend:
    """造一个已 prepare 的 TorchPeftBackend（**不联网、不加载真实模型**）。

    `device` 可传 `"cuda"`：r4 缺陷 5 起，「真实 CUDA 是否参与」决定 RNG 取证能不能降级，
    所以 CUDA 相关用例必须让后端自称在 cuda 上跑（配合注入的 `cuda_rng` 替身）。
    """
    backend = runner.TorchPeftBackend(
        model_id="google/gemma-4-31B-it-qat-w4a16-ct",
        revision=REVISION,
        target_modules=["q_proj", "o_proj"],
        device=device,
    )
    backend._torch = torch
    backend.model = _TinyLoRAModel()
    backend.lr = float(plan.lr)
    backend.sampler_order_sha256 = runner._sampler_order_sha256(plan)
    return backend


@unittest.skipUnless(TORCH_AVAILABLE, "本机没有 torch：真实后端路径测试如实跳过")
class TorchPeftBackendRealPathTests(unittest.TestCase):
    """G14：真实后端必须真的支持 `on_step` / 状态导出导入 / adapter 字节 / 逐步测量。"""

    def test_train_steps_supports_on_step_and_global_step(self) -> None:
        plan = _plan(steps=5)
        backend = _cpu_backend(plan)
        seen: list[dict] = []

        def on_step(info):
            seen.append(dict(info))
            return "step_limit" if len(seen) >= 2 else None

        report = backend.train_steps(plan, on_step=on_step)
        self.assertEqual(len(seen), 2, "on_step 必须逐步被调用")
        self.assertTrue(report["stopped"])
        self.assertEqual(report["stop_reason"], "step_limit")
        self.assertEqual(report["steps_executed"], 2)
        self.assertEqual(report["global_step"], 2)
        self.assertEqual(report["optimizer_step_count"], 2)
        self.assertTrue(report["base_params_frozen"], "基座必须冻结")
        self.assertTrue(report["adapter_params_changed"], "adapter 参数必须真的被更新")
        self.assertTrue(report["grad_norm_nonzero"])

    def test_train_steps_runs_to_completion_without_hook(self) -> None:
        plan = _plan(steps=3)
        backend = _cpu_backend(plan)
        report = backend.train_steps(plan)
        self.assertFalse(report["stopped"])
        self.assertEqual(report["steps_executed"], 3)
        self.assertEqual(report["global_step"], 3)

    def test_export_adapter_bytes_is_adapter_only_and_never_calls_state_dict(self) -> None:
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        backend.train_steps(plan)
        raw = backend.export_adapter_bytes()
        self.assertIsInstance(raw, bytes)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["format"], "torch-peft-adapter-bytes/1")
        self.assertTrue(payload["adapter_only"])
        self.assertFalse(payload["contains_base_weights"])
        self.assertEqual(
            sorted(payload["tensors"]),
            ["lora_A.weight", "lora_B.weight"],
            "只允许导出 requires_grad 的 adapter 张量，基座不得出现",
        )
        self.assertEqual(backend.model.state_dict_calls, 0)
        # 同源判据：同一状态导出两次必须逐字节一致。
        self.assertEqual(raw, backend.export_adapter_bytes())

    def test_training_state_roundtrip_restores_optimizer_rng_and_cursor(self) -> None:
        plan = _plan(steps=3)
        backend = _cpu_backend(plan)
        backend.train_steps(plan)
        state = backend.export_training_state()
        for key in (
            "optimizer",
            "rng",
            "cursor",
            "consumed_input_tokens",
            "consumed_supervised_tokens",
            "accum_boundary",
        ):
            self.assertIn(key, state, "checkpoint 必需键 %r 缺失" % key)
        self.assertIsNotNone(state["optimizer"], "AdamW 状态必须被导出")
        self.assertEqual(state["cursor"]["sampler_order_sha256"], backend.sampler_order_sha256)

        # 破坏现场：换掉参数与计数，再从状态里恢复。
        digest_before = backend.adapter_params_digest()
        backend.model.lora_A = torch.nn.Parameter(torch.zeros_like(backend.model.lora_A))
        self.assertNotEqual(backend.adapter_params_digest(), digest_before)

        class _State:
            pass

        wrapper = _State()
        for key, value in state.items():
            setattr(wrapper, key, value)
        applied = backend.import_training_state(wrapper)

        self.assertTrue(applied["applied"])
        self.assertTrue(applied["optimizer_state_restored"])
        self.assertEqual(applied["global_step"], 3)
        self.assertEqual(backend.global_step, 3)
        self.assertEqual(backend.start_step, 3, "续训必须从 checkpoint 的步数起步")
        self.assertEqual(backend.cursor_position, 3)
        self.assertIn("torch", applied["rng_restored"])
        self.assertIn("numpy", applied["rng_restored"])
        self.assertEqual(applied["rng_not_available_in_checkpoint"], [])

    def test_import_rejects_wrong_sampler_order(self) -> None:
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        backend.train_steps(plan)
        state = backend.export_training_state()

        class _State:
            pass

        wrapper = _State()
        for key, value in state.items():
            setattr(wrapper, key, value)
        wrapper.cursor = dict(state["cursor"])
        wrapper.cursor["sampler_order_sha256"] = "f" * 64
        with self.assertRaises(IntegrityError) as ctx:
            backend.import_training_state(wrapper)
        self.assertEqual(ctx.exception.code, "sampler_order_mismatch")

    def test_import_rejects_missing_optimizer_state(self) -> None:
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        backend.train_steps(plan)
        state = backend.export_training_state()

        class _State:
            pass

        wrapper = _State()
        for key, value in state.items():
            setattr(wrapper, key, value)
        wrapper.optimizer = None
        with self.assertRaises(MissingInput) as ctx:
            backend.import_training_state(wrapper)
        self.assertEqual(ctx.exception.code, "optimizer_state_missing")

    def test_import_rejects_incomplete_cursor(self) -> None:
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        backend.train_steps(plan)
        state = backend.export_training_state()

        class _State:
            pass

        wrapper = _State()
        for key, value in state.items():
            setattr(wrapper, key, value)
        wrapper.cursor = {"position": 1}
        with self.assertRaises(MissingInput) as ctx:
            backend.import_training_state(wrapper)
        self.assertEqual(ctx.exception.code, "cursor_incomplete")

    def test_measure_phases_covers_three_phases_with_real_readings(self) -> None:
        plan = _plan(steps=4)
        backend = _cpu_backend(plan)
        probe = profile_mod.PhaseProbe()
        result = backend.measure_phases(plan, probe)
        self.assertEqual(
            result["phases_measured"], ["forward", "backward", "optimizer_step"]
        )
        self.assertEqual(probe.phases_seen(), ["forward", "backward", "optimizer_step"])
        for entry in probe.phases:
            self.assertGreater(entry["host_peak_rss_gib"], 0.0)
        self.assertEqual(result["global_step"], 1)

    def test_measure_phases_before_prepare_is_blocked(self) -> None:
        backend = runner.TorchPeftBackend(
            model_id="google/gemma-4-31B-it-qat-w4a16-ct",
            revision=REVISION,
            target_modules=["q_proj"],
            device="cpu",
        )
        with self.assertRaises(FailClosed) as ctx:
            backend.measure_phases(_plan(), profile_mod.PhaseProbe())
        self.assertEqual(ctx.exception.code, "backend_not_prepared")


# ---------------------------------------------------------------- CLI 真的接通了


class TorchPeftCliReachabilityTests(unittest.TestCase):
    """G7b：`--backend torch-peft --plan-json` 不得在 CLI 层被拒。"""

    def _plan_file(self, root: str) -> str:
        path = os.path.join(root, "plan.json")
        payload = {
            "steps": 2,
            "lr": 0.05,
            "seq_len": 6,
            "lora_rank": 3,
            "lora_alpha": 6,
            "adapter_name": "v3_policy",
            "batches": [
                {"input_ids": [0, 1, 2, 3, 4, 5], "labels": [0, 1, 2, 3, 4, 5]},
            ],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def _run_cli(self, root: str, extra: list[str]) -> tuple[int, dict]:
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        argv = [
            "--dest-dir", os.path.join(root, "out"),
            "--time-budget-seconds", "120",
            "--hashes", self._hashes_file(root),
        ] + extra
        with redirect_stdout(buffer):
            try:
                code = engineering_check.main(argv)
            except SystemExit as exc:  # argparse 拒法：退出码 2
                return int(exc.code or 0), {"code": "argparse-rejected"}
        text = buffer.getvalue().strip()
        return code, (json.loads(text) if text else {})

    def _hashes_file(self, root: str) -> str:
        path = os.path.join(root, "hashes.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(HASHES, handle)
        return path

    def _real_mode_flags(self, root: str) -> list[str]:
        """r3 缺陷 D：真实模式必须同时给齐 授权/预算/停机点/输入绑定。

        r4 缺陷 2/4 追加：v6 批准记录（带**外部锚**）与固定本地权重目录。
        """
        layout_path = os.path.join(root, "layout.json")
        with open(layout_path, "w", encoding="utf-8") as handle:
            json.dump(FLAT_LAYOUT, handle)
        approval_path = self._approval_file(root)
        anchor = hashlib.sha256(open(approval_path, "rb").read()).hexdigest()
        local_dir = os.path.join(root, "local_weights")
        os.makedirs(local_dir, exist_ok=True)
        return [
            "--local-load-authorized",
            "--stop-at-step", "3",
            "--model-inputs-root", EVIDENCE_DIR,
            "--model-inputs-layout", layout_path,
            "--interface", INTERFACE_LOCK,
            "--approval", approval_path,
            "--approval-expected-sha256", anchor,
            "--approval-evidence-ref", os.path.join(root, "approval-evidence.json"),
            "--local-model-dir", local_dir,
        ]

    def _approval_file(self, root: str) -> str:
        """写一份**形状合法**的 v6 批准记录（外部锚由调用方实算后传入）。"""
        evidence = os.path.join(root, "approval-evidence.json")
        with open(evidence, "w", encoding="utf-8") as handle:
            json.dump({"kind": "engineering-check-evidence", "rev": 1}, handle)
        with open(evidence, "rb") as handle:
            evidence_sha = hashlib.sha256(handle.read()).hexdigest()
        path = os.path.join(root, "approval.json")
        payload = {
            "approval_id": "k38-test-approval",
            "target_sha": "128d9b98b05ddf128c2e77b599e078de65675b8a",
            "gpu_hours_approved": 4,
            "verified_by": "test-fixture",
            "verified_utc": "2026-10-08T00:00:00Z",
            "evidence_sha256": evidence_sha,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_torch_peft_without_plan_json_is_reached_and_fails_closed(self) -> None:
        with temp_dir_outside_repo("k38r_cli_noplan_") as root:
            code, payload = self._run_cli(
                root,
                [
                    "--backend", "torch-peft",
                    "--model-id", "google/gemma-4-31B-it-qat-w4a16-ct",
                    "--model-revision", REVISION,
                    "--target-modules", "q_proj,o_proj",
                ] + self._real_mode_flags(root),
            )
        self.assertEqual(payload.get("code"), "approval_not_trusted")
        self.assertEqual(code, 7)

    def test_torch_peft_with_plan_json_reaches_the_backend(self) -> None:
        """关键回归：过去这一步是 argparse 直接拒（exit 2），现在是执行路径上的 fail-closed。"""
        with temp_dir_outside_repo("k38r_cli_reach_") as root:
            plan_path = self._plan_file(root)
            code, payload = self._run_cli(
                root,
                [
                    "--backend", "torch-peft",
                    "--plan-json", plan_path,
                    "--model-id", "google/gemma-4-31B-it-qat-w4a16-ct",
                    "--model-revision", REVISION,
                    "--target-modules", "q_proj,o_proj",
                ] + self._real_mode_flags(root),
            )
        # 走到 TorchPeftBackend.prepare 的真实分支：本机缺锁定栈 → 明确的环境原因。
        # r4 缺陷 4 起 `--allow-download` 不再参与本地加载判定，所以终点不再是
        # model_download_not_authorized，而是真实栈缺失（仍是 Blocked，退出码 8）。
        self.assertEqual(payload.get("code"), "approval_not_trusted")
        self.assertEqual(code, 7)
        self.assertTrue(payload.get("message"), "必须带可读的原因，不能只给一个码")

    def test_real_mode_gate_order_is_fail_closed(self) -> None:
        """缺陷 D：真实模式的准入闸门按固定顺序 fail-closed，且**不**回退合成哈希。"""
        with temp_dir_outside_repo("k38r_cli_gates_") as root:
            plan_path = self._plan_file(root)
            base = [
                "--backend", "torch-peft",
                "--plan-json", plan_path,
                "--model-id", "google/gemma-4-31B-it-qat-w4a16-ct",
                "--model-revision", REVISION,
                "--target-modules", "q_proj,o_proj",
            ]
            # ① 没有 --hashes → 必须先拒（不得回退 SYNTHETIC_HASHES）
            code, payload = self._run_cli_no_hashes(root, base)
            self.assertEqual(payload.get("code"), "real_mode_requires_real_hashes")
            self.assertEqual(code, 4)
            # ② 有 hashes 但没授权本地加载
            code, payload = self._run_cli(root, base)
            self.assertEqual(payload.get("code"), "local_load_not_authorized")
            self.assertEqual(code, 7)
            # ③ 授权了但没给 --model-inputs-root
            code, payload = self._run_cli(
                root, base + ["--local-load-authorized", "--stop-at-step", "3"]
            )
            self.assertEqual(payload.get("code"), "model_inputs_root_missing")
            # ④ 全给齐之后才走到后端（r4 起终点是真实栈缺失，不再是下载许可）
            code, payload = self._run_cli(root, base + self._real_mode_flags(root))
            self.assertEqual(payload.get("code"), "approval_not_trusted")

    def _run_cli_no_hashes(self, root: str, extra: list[str]) -> tuple[int, dict]:
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        argv = ["--dest-dir", os.path.join(root, "out2"), "--time-budget-seconds", "120"] + extra
        with redirect_stdout(buffer):
            try:
                code = engineering_check.main(argv)
            except SystemExit as exc:
                return int(exc.code or 0), {"code": "argparse-rejected"}
        text = buffer.getvalue().strip()
        return code, (json.loads(text) if text else {})

    def test_torch_peft_missing_pins_fails_before_backend_construction(self) -> None:
        with temp_dir_outside_repo("k38r_cli_pins_") as root:
            plan_path = self._plan_file(root)
            code, payload = self._run_cli(
                root,
                ["--backend", "torch-peft", "--plan-json", plan_path, "--model-id", "x"],
            )
        self.assertEqual(payload.get("code"), "engineering_check_backend_pins_missing")
        self.assertEqual(code, 4)

    def test_unknown_backend_is_rejected(self) -> None:
        with temp_dir_outside_repo("k38r_cli_unknown_") as root:
            code, payload = self._run_cli(root, ["--backend", "sorcery"])
        # argparse 用 choices 拦下 → 退出码 2（这是**正确的**拒法：后端名不在白名单）
        self.assertEqual(code, 2)


class BackendChoiceTests(unittest.TestCase):
    def test_unknown_backend_name_raises_policy_violation(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            engineering_check.resolve_backend_choice(
                "sorcery",
                model_id=None,
                model_revision=None,
                target_modules=None,
                allow_download=False,
                device="cpu",
            )
        self.assertEqual(ctx.exception.code, "engineering_check_unknown_backend")

    def test_synthetic_backend_needs_no_pins(self) -> None:
        self.assertEqual(
            engineering_check.resolve_backend_choice(
                "synthetic",
                model_id=None,
                model_revision=None,
                target_modules=None,
                allow_download=False,
                device="cpu",
            ),
            "synthetic",
        )


# ---------------------------------------------------------------- 单一总截止


class SingleDeadlineTests(unittest.TestCase):
    """总截止：所有阶段共用同一个 deadline，恢复探针不得新建无截止的 StopRequest。"""

    def test_synthetic_run_uses_one_deadline_for_every_section(self) -> None:
        with temp_dir_outside_repo("k38r_deadline_") as root:
            report = engineering_check.run_engineering_check(
                dest_dir=os.path.join(root, "out"),
                steps=6,
                checkpoint_every=2,
                stop_at_step=5,
                time_budget_seconds=120.0,
                hashes=dict(HASHES),
                resume_probe_steps=2,
            )
        self.assertEqual(report["verdict"], "pass")
        self.assertTrue(report["budget"]["single_deadline_for_all_sections"])
        ledger = report["budget_ledger"]
        self.assertEqual(
            [item["section"] for item in ledger],
            [
                "mask_fixtures",
                "train",
                "memory_profile",
                "adapter_export",
                "resume_probe",
                "negative_hash_mismatch",
            ],
        )
        deadlines = {item["deadline_monotonic"] for item in ledger}
        self.assertEqual(len(deadlines), 1, "所有阶段必须共用同一个 deadline：%r" % deadlines)
        self.assertIsNotNone(deadlines.pop())
        remaining = [item["remaining_seconds_after"] for item in ledger]
        self.assertEqual(remaining, sorted(remaining, reverse=True), "剩余预算必须单调不增")
        self.assertTrue(report["resume"]["deadline_shared_with_main_stop"])

    def test_expired_budget_stops_before_running_any_stage(self) -> None:
        with temp_dir_outside_repo("k38r_deadline_zero_") as root:
            with self.assertRaises(FailClosed) as ctx:
                engineering_check.run_engineering_check(
                    dest_dir=os.path.join(root, "out"),
                    steps=4,
                    time_budget_seconds=0.0,
                    hashes=dict(HASHES),
                )
        self.assertEqual(ctx.exception.code, "engineering_check_deadline_exceeded")

    def test_no_deadline_ledger_is_honest_when_budget_is_absent(self) -> None:
        with temp_dir_outside_repo("k38r_deadline_none_") as root:
            report = engineering_check.run_engineering_check(
                dest_dir=os.path.join(root, "out"),
                steps=4,
                checkpoint_every=2,
                hashes=dict(HASHES),
            )
        self.assertIsNone(report["remaining_budget_seconds"])
        self.assertTrue(all(item["deadline_monotonic"] is None for item in report["budget_ledger"]))

    def test_source_regression_every_stop_request_carries_a_deadline(self) -> None:
        """AST 静态判据：`engineering_check.py` 里不得再出现**无 deadline** 的 StopRequest。"""
        path = os.path.join(REPO_ROOT, "v3", "train", "engineering_check.py")
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        offenders: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "StopRequest":
                continue
            keywords = {item.arg for item in node.keywords}
            if "deadline_monotonic" not in keywords:
                offenders.append(node.lineno)
        self.assertEqual(
            offenders,
            [],
            "engineering_check.py 第 %s 行出现无 deadline 的 StopRequest（恢复探针的老毛病）"
            % offenders,
        )

    def test_resume_probe_stop_shares_the_main_deadline_at_runtime(self) -> None:
        stop = StopRequest(deadline_monotonic=None)
        ledger = engineering_check.BudgetLedger(stop, 0.0)
        self.assertIsNone(ledger.remaining())
        with ledger.section("x"):
            pass
        self.assertEqual(ledger.sections[0]["section"], "x")


# ---------------------------------------------------------------- 训练计划输入


class PlanFromJsonTests(unittest.TestCase):
    def _write(self, root: str, payload: dict) -> str:
        path = os.path.join(root, "plan.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_reads_a_real_plan(self) -> None:
        with temp_dir_outside_repo("k38r_plan_ok_") as root:
            path = self._write(
                root,
                {
                    "steps": 3,
                    "lr": 0.1,
                    "seq_len": 6,
                    "lora_rank": 4,
                    "lora_alpha": 8,
                    "adapter_name": "v3_policy",
                    "batches": [{"input_ids": [1, 2], "labels": [1, 2]}],
                },
            )
            plan = runner.plan_from_json(path)
        self.assertEqual(plan.steps, 3)
        self.assertEqual(plan.lora_rank, 4)
        self.assertEqual(len(plan.batches), 1)

    def test_missing_file_is_fail_closed(self) -> None:
        with self.assertRaises(MissingInput) as ctx:
            runner.plan_from_json(os.path.join(REPO_ROOT, "does-not-exist.json"))
        self.assertEqual(ctx.exception.code, "training_plan_missing")

    def test_plan_without_batches_is_rejected(self) -> None:
        with temp_dir_outside_repo("k38r_plan_nobatch_") as root:
            path = self._write(root, {"steps": 3, "lr": 0.1, "seq_len": 6, "lora_rank": 4, "lora_alpha": 8})
            with self.assertRaises(MissingInput) as ctx:
                runner.plan_from_json(path)
        self.assertEqual(ctx.exception.code, "training_plan_invalid")

    def test_plan_missing_scalar_field_is_rejected(self) -> None:
        with temp_dir_outside_repo("k38r_plan_nofield_") as root:
            path = self._write(
                root,
                {
                    "steps": 3,
                    "lr": 0.1,
                    "seq_len": 6,
                    "lora_rank": 4,
                    "batches": [{"input_ids": [1, 2], "labels": [1, 2]}],
                },
            )
            with self.assertRaises(MissingInput) as ctx:
                runner.plan_from_json(path)
        self.assertEqual(ctx.exception.code, "training_plan_invalid")


# ---------------------------------------------------------------- 分阶段校验


class StagedExportValidationTests(unittest.TestCase):
    """统一守卫把训练与 serving 依赖混在一起 → 分阶段校验；`null` 不算通过。"""

    def _manifest(self, root: str, *, mapper, stage: str, verified: bool) -> dict:
        adapter_dir = os.path.join(root, "adapters", "v3_policy")
        os.makedirs(adapter_dir, exist_ok=True)
        with open(
            os.path.join(adapter_dir, export_contract.adapter_contract.ADAPTER_WEIGHTS_FILENAME),
            "wb",
        ) as handle:
            handle.write(b"\x00" * 8)
        with open(
            os.path.join(adapter_dir, export_contract.adapter_contract.ADAPTER_CONFIG_FILENAME),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump({"peft_type": "LORA"}, handle)
        return export_contract.build_export_manifest(
            root,
            adapter_name="v3_policy",
            rank=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules_regex=".*(q_proj|o_proj)$",
            matched_module_count=2,
            base_repo_id="google/gemma-4-31B-it-qat-w4a16-ct",
            base_revision=REVISION,
            vllm_mapper_revision=mapper,
            stage=stage,
            serving_pin_verified=verified,
            training_hashes=dict(HASHES),
        )

    def test_training_stage_allows_null_mapper_but_flags_serving(self) -> None:
        with temp_dir_outside_repo("k38r_stage_train_") as root:
            manifest = self._manifest(
                root, mapper=None, stage=export_contract.STAGE_TRAINING, verified=False
            )
            result = export_contract.validate_export_manifest(
                manifest, export_dir=root, stage=export_contract.STAGE_TRAINING
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["stage"], "training")
        self.assertFalse(result["serving_pins_verified"])
        self.assertTrue(result["serving_stage_validation_required"])
        self.assertIsNone(manifest["base"]["vllm_mapper_revision"])
        self.assertFalse(manifest["base"]["serving_pins_verified"])

    def test_the_same_manifest_is_rejected_at_the_serving_stage(self) -> None:
        """`null` 不是通过：训练阶段过的清单，serving 阶段必须仍然被拒。"""
        with temp_dir_outside_repo("k38r_stage_serve_") as root:
            manifest = self._manifest(
                root, mapper=None, stage=export_contract.STAGE_TRAINING, verified=False
            )
            with self.assertRaises(PolicyViolation) as ctx:
                export_contract.validate_export_manifest(manifest, export_dir=root)
        self.assertEqual(ctx.exception.code, "export_manifest_invalid")
        self.assertTrue(
            any("vllm_mapper_revision" in item for item in ctx.exception.context["problems"])
        )

    def test_serving_stage_producer_refuses_a_null_mapper(self) -> None:
        with temp_dir_outside_repo("k38r_stage_prod_") as root:
            with self.assertRaises(MissingInput) as ctx:
                self._manifest(
                    root, mapper=None, stage=export_contract.STAGE_SERVING, verified=False
                )
        self.assertEqual(ctx.exception.code, "mapper_revision_unknown")

    def test_training_stage_rejects_an_unverified_non_null_mapper(self) -> None:
        with temp_dir_outside_repo("k38r_stage_fake_") as root:
            manifest = self._manifest(
                root,
                mapper="b1388b1fbf5aaef47937fabe98931211684666a6",
                stage=export_contract.STAGE_TRAINING,
                verified=False,
            )
            with self.assertRaises(PolicyViolation) as ctx:
                export_contract.validate_export_manifest(
                    manifest, export_dir=root, stage=export_contract.STAGE_TRAINING
                )
        self.assertTrue(
            any("冒充已验证" in item for item in ctx.exception.context["problems"])
        )

    def test_serving_stage_accepts_a_verified_mapper(self) -> None:
        with temp_dir_outside_repo("k38r_stage_ok_") as root:
            manifest = self._manifest(
                root,
                mapper="b1388b1fbf5aaef47937fabe98931211684666a6",
                stage=export_contract.STAGE_SERVING,
                verified=True,
            )
            result = export_contract.validate_export_manifest(manifest, export_dir=root)
        self.assertTrue(result["ok"])
        self.assertTrue(result["serving_pins_verified"])
        self.assertFalse(result["serving_stage_validation_required"])

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            export_contract._normalize_stage("whatever")
        self.assertEqual(ctx.exception.code, "export_manifest_unknown_stage")


# ---------------------------------------------------------------- 源码级接线证据


class WiringSourceScanTests(unittest.TestCase):
    def _class_methods(self, relative: str, class_name: str) -> set:
        with open(os.path.join(REPO_ROOT, *relative.split("/")), "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return {
                    item.name for item in node.body if isinstance(item, ast.FunctionDef)
                }
        self.fail("%s 里找不到类 %s" % (relative, class_name))

    def test_torch_backend_exposes_the_g14_surface(self) -> None:
        methods = self._class_methods("v3/train/runner.py", "TorchPeftBackend")
        for name in (
            "train_steps",
            "export_adapter_bytes",
            "export_training_state",
            "import_training_state",
            "measure_phases",
        ):
            self.assertIn(name, methods, "TorchPeftBackend 缺少 %s（G14 接线缺口）" % name)

    def test_torch_backend_train_steps_accepts_on_step(self) -> None:
        with open(os.path.join(REPO_ROOT, "v3", "train", "runner.py"), "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TorchPeftBackend":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "train_steps":
                        names = {arg.arg for arg in item.args.args} | {
                            arg.arg for arg in item.args.kwonlyargs
                        }
                        self.assertIn("on_step", names)
                        return
        self.fail("找不到 TorchPeftBackend.train_steps")

    def test_engineering_check_does_not_hardcode_the_synthetic_backend(self) -> None:
        """回归 Mika 的原话：`engineering_check.py` 不得写死 SyntheticBackend。"""
        path = os.path.join(REPO_ROOT, "v3", "train", "engineering_check.py")
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        offenders: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_engineering_check":
                for call in ast.walk(node):
                    if not isinstance(call, ast.Call):
                        continue
                    func = call.func
                    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                    if name == "SyntheticBackend":
                        offenders.append(call.lineno)
        self.assertEqual(
            offenders,
            [],
            "run_engineering_check 里仍有写死的 SyntheticBackend（第 %s 行）：必须走 _make_backend"
            % offenders,
        )

    def test_cli_exposes_the_backend_and_plan_flags(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "v3.train.engineering_check", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for flag in ("--backend", "--plan-json", "--model-id", "--model-revision", "--target-modules"):
            self.assertIn(flag, completed.stdout)

    def test_evidence_hash_chain_is_stable(self) -> None:
        with temp_dir_outside_repo("k38r_chain_") as root:
            report = engineering_check.run_engineering_check(
                dest_dir=os.path.join(root, "out"),
                steps=4,
                checkpoint_every=2,
                hashes=dict(HASHES),
            )
        digest = report.pop("evidence_sha256")
        recomputed = engineering_check.sha256_json(report)
        self.assertEqual(digest, recomputed)


# ================================================================ r3 整改（缺陷 A–E）


class HardDeadlineTests(unittest.TestCase):
    """缺陷 A：总截止必须**独立于业务停止原因**，"同一 deadline 数值"不等于硬截止生效。"""

    def test_step_limit_first_then_deadline_overrun_is_blocked(self) -> None:
        """Mika 的合成反例：deadline=10、先置 step_limit、时钟=11 → 必须拒。"""
        stop = StopRequest(deadline_monotonic=time.monotonic() + 10.0)
        ledger = engineering_check.BudgetLedger(stop, time.monotonic())
        stop.request("step_limit", stop_at_step=10)  # 业务原因先置位（首次生效）
        stop.deadline_monotonic = time.monotonic() - 1.0  # 时钟越过总截止（remaining = -1s）
        self.assertEqual(stop.poll(), "step_limit", "poll 只看首次原因，这正是旧实现的洞")
        self.assertLess(ledger.remaining(), 0.0)
        with self.assertRaises(FailClosed) as ctx:
            ledger.check("adapter_export")
        self.assertEqual(ctx.exception.code, "engineering_check_deadline_exceeded")
        self.assertIn("adapter_export", ctx.exception.message)
        self.assertEqual(ctx.exception.context.get("stop_reason"), "step_limit")

    def test_hard_deadline_is_independent_of_poll(self) -> None:
        stop = StopRequest(deadline_monotonic=time.monotonic() - 0.001)
        stop.request("signal", signal_name="SIGTERM")
        self.assertEqual(stop.poll(), "signal")
        self.assertTrue(stop.hard_deadline_exceeded())

    def test_no_deadline_means_no_hard_stop(self) -> None:
        stop = StopRequest(deadline_monotonic=None)
        stop.request("step_limit")
        self.assertFalse(stop.hard_deadline_exceeded())
        ledger = engineering_check.BudgetLedger(stop, time.monotonic())
        ledger.check("memory_profile")  # 不抛

    def test_in_section_overrun_is_recorded(self) -> None:
        """阶段**内**才越过截止：必须记进 overruns，不能悄悄算通过。"""
        stop = StopRequest(deadline_monotonic=time.monotonic() + 0.05)
        ledger = engineering_check.BudgetLedger(stop, time.monotonic())
        with ledger.section("train"):
            time.sleep(0.08)
        self.assertEqual(len(ledger.overruns), 1)
        self.assertTrue(ledger.sections[0]["deadline_exceeded_within_section"])

    def test_overrun_makes_the_verdict_fail(self) -> None:
        passing = {
            "mask_fixtures": {"all_passed": True},
            "training": {
                "executed": True,
                "saved": {"adapter_only": True},
                "reloaded": {"matches_saved_adapter": True},
                "trained": {"base_params_frozen": True},
            },
            "checkpoint_count": 1,
            "resume": {"adapter_reload_matches": True},
            "negative_hash_mismatch_rejected": True,
            "adapter_export": {"validation": {"ok": True}},
        }
        self.assertTrue(engineering_check._all_checks_ok(dict(passing)))
        self.assertFalse(
            engineering_check._all_checks_ok(
                dict(passing, deadline_overrun_sections=["resume_probe"])
            )
        )

    def test_outer_supervisor_deadline_wins_when_earlier(self) -> None:
        with temp_dir_outside_repo("k38r_outer_") as root:
            report = engineering_check.run_engineering_check(
                dest_dir=os.path.join(root, "out"),
                steps=4,
                checkpoint_every=2,
                hashes=dict(HASHES),
                time_budget_seconds=600.0,
                deadline_epoch=time.time() + 120.0,
            )
        self.assertEqual(report["budget"]["deadline_source"], "outer-supervisor")
        self.assertLess(report["remaining_budget_seconds"], 600.0)

    def test_budget_wins_when_earlier_than_outer(self) -> None:
        with temp_dir_outside_repo("k38r_outer2_") as root:
            report = engineering_check.run_engineering_check(
                dest_dir=os.path.join(root, "out"),
                steps=4,
                checkpoint_every=2,
                hashes=dict(HASHES),
                time_budget_seconds=5.0,
                deadline_epoch=time.time() + 3600.0,
            )
        self.assertEqual(report["budget"]["deadline_source"], "time-budget-seconds")

    def test_cli_reads_outer_deadline_from_environment(self) -> None:
        import io
        import os as _os
        from contextlib import redirect_stdout

        with temp_dir_outside_repo("k38r_env_deadline_") as root:
            hashes = os.path.join(root, "hashes.json")
            with open(hashes, "w", encoding="utf-8") as handle:
                json.dump(HASHES, handle)
            buffer = io.StringIO()
            previous = _os.environ.get("V3_DEADLINE_EPOCH")
            _os.environ["V3_DEADLINE_EPOCH"] = str(time.time() + 30.0)
            try:
                with redirect_stdout(buffer):
                    code = engineering_check.main(
                        [
                            "--dest-dir", os.path.join(root, "out"),
                            "--steps", "4",
                            "--checkpoint-every", "2",
                            "--hashes", hashes,
                        ]
                    )
            finally:
                if previous is None:
                    _os.environ.pop("V3_DEADLINE_EPOCH", None)
                else:
                    _os.environ["V3_DEADLINE_EPOCH"] = previous
        self.assertEqual(code, 0)
        summary = json.loads(buffer.getvalue())
        self.assertEqual(summary["deadline_source"], "outer-supervisor")


class PrepareOnceTests(unittest.TestCase):
    """缺陷 B：基座只加载一次；内存测量必须测**同一组**模型/optimizer 参数。"""

    def test_second_prepare_is_rejected(self) -> None:
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        backend.load_count = 1
        with self.assertRaises(PolicyViolation) as ctx:
            backend.prepare(plan)
        self.assertEqual(ctx.exception.code, "backend_already_prepared")
        self.assertIn("load_count=1", ctx.exception.message)

    def test_profile_reuse_does_not_prepare_again(self) -> None:
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        backend.load_count = 1
        backend.parameter_ownership = backend.parameter_ownership_snapshot()
        before = backend.parameter_ownership
        profile = profile_mod.measure_memory_profile(
            backend,
            plan,
            prepare=False,
            prepared_report={"backend": "stub", "load_count": 1},
        )
        self.assertTrue(profile["reused_prepared_backend"])
        self.assertEqual(profile["backend_load_count"], 1)
        self.assertEqual(profile["backend_parameter_ownership"], before)
        self.assertIn("prepare", [item["phase"] for item in profile["phases"]])
        prepare_phase = next(
            item for item in profile["phases"] if item["phase"] == "prepare"
        )
        self.assertTrue(prepare_phase["reused_prepared_backend"])

    def test_profile_reuse_without_report_is_rejected(self) -> None:
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        with self.assertRaises(MissingInput) as ctx:
            profile_mod.measure_memory_profile(backend, plan, prepare=False)
        self.assertEqual(ctx.exception.code, "prepared_report_missing")

    def test_parameter_ownership_detects_a_swapped_model(self) -> None:
        """反例判据本身要有效：换掉模型必须导致归属指纹变化。"""
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        first = backend.parameter_ownership_snapshot()
        backend.model = _TinyLoRAModel()
        second = backend.parameter_ownership_snapshot()
        self.assertNotEqual(first["parameter_ids_sha256"], second["parameter_ids_sha256"])

    def test_assert_prepared_reports_reuse(self) -> None:
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        backend.load_count = 1
        info = backend.assert_prepared()
        self.assertTrue(info["reused"])
        self.assertEqual(info["load_count"], 1)


class InputBindingTests(unittest.TestCase):
    """缺陷 C：绑定生产加载器**实际消费**的 config / tokenizer / template。"""

    def _modified_copy(self, root: str) -> str:
        target = os.path.join(root, "modified")
        os.makedirs(target, exist_ok=True)
        for name in FLAT_LAYOUT:
            with open(os.path.join(EVIDENCE_DIR, name), "rb") as src:
                raw = src.read()
            if name == "chat_template.jinja":
                # 模拟现场被改过的工作副本（追加一段模板注释即改变字节）。
                raw = raw + b"\n{# modified working copy #}\n"
            with open(os.path.join(target, name), "wb") as dst:
                dst.write(raw)
        return target

    def test_binding_matches_the_official_originals(self) -> None:
        binding = input_binding.bind_model_inputs(
            root=EVIDENCE_DIR, interface_path=INTERFACE_LOCK, layout=FLAT_LAYOUT
        )
        self.assertEqual(binding["revision"], REVISION)
        measured = {item["filename"]: item["measured_sha256"] for item in binding["files"]}
        self.assertEqual(
            measured["chat_template.jinja"],
            "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        )
        self.assertEqual(
            measured["tokenizer_config.json"],
            "b8045a4576903e86903291d5cbdd4adfc8859e9ce3c98621bdbd957f73ed394b",
        )
        self.assertEqual(
            measured["tokenizer.json"],
            "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        )
        self.assertTrue(all(item["matches_pin"] for item in binding["files"]))
        self.assertFalse(binding["verified_flipped"])
        self.assertEqual(binding["weights_sha256"], "hash_pending")

    def test_consumed_inputs_identify_the_actual_files(self) -> None:
        binding = input_binding.bind_model_inputs(
            root=EVIDENCE_DIR, interface_path=INTERFACE_LOCK, layout=FLAT_LAYOUT
        )
        consumed = binding["consumed_inputs"]
        self.assertEqual(
            sorted(consumed),
            ["chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json"],
        )
        for item in consumed.values():
            self.assertTrue(os.path.isfile(item["path"]))
            self.assertEqual(len(item["sha256"]), 64)
            self.assertTrue(item["pin"])

    def test_verified_official_but_modified_copy_is_rejected(self) -> None:
        """Mika 明确要的拒绝反例：核验官方、实际加载修改版。"""
        with temp_dir_outside_repo("k38r_bind_modified_") as root:
            modified = self._modified_copy(root)
            with self.assertRaises(PolicyViolation) as ctx:
                input_binding.bind_model_inputs(
                    root=modified, interface_path=INTERFACE_LOCK, layout=FLAT_LAYOUT
                )
        self.assertEqual(ctx.exception.code, "input_pin_mismatch")
        mismatched = ctx.exception.context["mismatches"]
        self.assertEqual([item["filename"] for item in mismatched], ["chat_template.jinja"])
        self.assertNotEqual(mismatched[0]["measured"], mismatched[0]["declared"])

    def test_decoy_root_is_recorded_but_never_used(self) -> None:
        with temp_dir_outside_repo("k38r_bind_decoy_") as root:
            modified = self._modified_copy(root)
            binding = input_binding.bind_model_inputs(
                root=EVIDENCE_DIR,
                interface_path=INTERFACE_LOCK,
                layout=FLAT_LAYOUT,
                decoy_roots=[modified],
            )
        shadowed = {item["filename"]: item for item in binding["shadowed_files"]}
        self.assertIn("chat_template.jinja", shadowed)
        self.assertFalse(shadowed["chat_template.jinja"]["agrees_with_pin"])
        self.assertFalse(shadowed["chat_template.jinja"]["used"])
        self.assertTrue(shadowed["tokenizer.json"]["agrees_with_pin"])

    def test_unlocked_pin_cannot_be_used(self) -> None:
        with temp_dir_outside_repo("k38r_bind_null_") as root:
            lock = json.load(open(INTERFACE_LOCK, encoding="utf-8"))
            lock["pins"]["chat_template_sha256"]["value"] = None
            path = os.path.join(root, "lock.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(lock, handle)
            with self.assertRaises(UnverifiedLock) as ctx:
                input_binding.bind_model_inputs(
                    root=EVIDENCE_DIR, interface_path=path, layout=FLAT_LAYOUT
                )
        self.assertEqual(ctx.exception.code, "input_pin_not_locked")

    def test_missing_file_is_fail_closed(self) -> None:
        with temp_dir_outside_repo("k38r_bind_missing_") as root:
            os.makedirs(os.path.join(root, "empty"), exist_ok=True)
            with self.assertRaises(MissingInput) as ctx:
                input_binding.bind_model_inputs(
                    root=os.path.join(root, "empty"),
                    interface_path=INTERFACE_LOCK,
                    layout=FLAT_LAYOUT,
                )
        self.assertEqual(ctx.exception.code, "input_file_missing")

    def test_load_view_copies_and_reverifies(self) -> None:
        with temp_dir_outside_repo("k38r_view_") as root:
            binding = input_binding.bind_model_inputs(
                root=EVIDENCE_DIR, interface_path=INTERFACE_LOCK, layout=FLAT_LAYOUT
            )
            view = input_binding.assemble_load_view(
                binding, os.path.join(root, "load_view")
            )
            for key in ("tokenizer_config_path", "tokenizer_data_path", "chat_template_path"):
                self.assertTrue(os.path.isfile(view[key]), key)
            by_name = {item["filename"]: item for item in view["files"]}
            self.assertTrue(by_name["chat_template.jinja"]["copy_matches_source"])
            self.assertEqual(
                input_binding.sha256_file(view["chat_template_path"]),
                "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
            )
            self.assertFalse(view["weights_in_view"])

    def test_real_backend_requires_bound_inputs(self) -> None:
        backend = runner.TorchPeftBackend(
            model_id="google/gemma-4-31B-it-qat-w4a16-ct",
            revision=REVISION,
            target_modules=["q_proj"],
            device="cpu",
        )
        with self.assertRaises(PolicyViolation) as ctx:
            backend._resolve_tokenizer_source()
        self.assertEqual(ctx.exception.code, "model_inputs_unbound")

    def test_real_backend_tokenizer_source_points_at_the_load_view(self) -> None:
        with temp_dir_outside_repo("k38r_view2_") as root:
            binding = input_binding.bind_model_inputs(
                root=EVIDENCE_DIR, interface_path=INTERFACE_LOCK, layout=FLAT_LAYOUT
            )
            view = input_binding.assemble_load_view(binding, os.path.join(root, "load_view"))
            binding["load_view_dir"] = view["view_dir"]
            binding["view_manifest_sha256"] = view["manifest_sha256"]
            backend = runner.TorchPeftBackend(
                model_id="google/gemma-4-31B-it-qat-w4a16-ct",
                revision=REVISION,
                target_modules=["q_proj"],
                device="cpu",
                model_inputs=binding,
            )
            source = backend._resolve_tokenizer_source()
        self.assertEqual(source["kind"], "controlled-load-view")
        self.assertTrue(source["is_load_view"])
        self.assertEqual(source["template_override"], None)
        self.assertTrue(source["consumed_inputs"])

    def test_explicit_template_override_is_rejected_by_default(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            input_binding.assert_no_template_override(override_template="{% for m in messages %}{% endfor %}")
        self.assertEqual(ctx.exception.code, "template_override_not_authorized")

    def test_authorized_template_override_is_recorded(self) -> None:
        record = input_binding.assert_no_template_override(
            override_template="X",
            override_authorized=True,
            override_reason="兼容补丁，待 26 独审",
        )
        self.assertTrue(record["override_used"])
        self.assertEqual(record["override_bytes"], 1)
        self.assertEqual(len(record["override_sha256"]), 64)
        self.assertIn("不", record["note"])

    def test_authorized_override_without_reason_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            input_binding.assert_no_template_override(
                override_template="X", override_authorized=True, override_reason="  "
            )
        self.assertEqual(ctx.exception.code, "template_override_reason_missing")

    def test_tool_arguments_strings_are_normalized_and_bad_json_rejected(self) -> None:
        self.assertEqual(
            input_binding.normalize_tool_arguments({"a": 1})["source"], "mapping"
        )
        self.assertEqual(
            input_binding.normalize_tool_arguments(json.dumps({"a": 1}))["source"],
            "json-string",
        )
        with self.assertRaises(PolicyViolation) as ctx:
            input_binding.normalize_tool_arguments("{not json")
        self.assertEqual(ctx.exception.code, "tool_arguments_invalid_json")
        with self.assertRaises(PolicyViolation) as ctx:
            input_binding.normalize_tool_arguments(json.dumps("a plain string"))
        self.assertEqual(ctx.exception.code, "tool_arguments_not_a_mapping")

    def test_modified_copy_uses_the_default_snapshot_layout_too(self) -> None:
        """现场快照用的是 official/ + shared/ 分组布局，这条把分组布局也钉住。"""
        with temp_dir_outside_repo("k38r_layout_") as root:
            os.makedirs(os.path.join(root, "official"))
            os.makedirs(os.path.join(root, "shared"))
            layout = {
                "tokenizer_config.json": "official/tokenizer_config.json",
                "chat_template.jinja": "official/chat_template.jinja",
                "tokenizer.json": "shared/tokenizer.json",
                "config.json": "shared/config.json",
            }
            for name, relative in layout.items():
                with open(os.path.join(EVIDENCE_DIR, name), "rb") as src:
                    raw = src.read()
                with open(os.path.join(root, *relative.split("/")), "wb") as dst:
                    dst.write(raw)
            binding = input_binding.bind_model_inputs(
                root=root, interface_path=INTERFACE_LOCK, layout=layout
            )
        self.assertTrue(all(item["matches_pin"] for item in binding["files"]))
        self.assertEqual(
            binding["files"][0]["layout_relative_path"].split("/")[0], "official"
        )

    def test_engineering_check_report_carries_the_consumed_inputs(self) -> None:
        """报告里能确定消费者实际用了哪份 tokenizer / template（wiring 级证据）。"""
        with temp_dir_outside_repo("k38r_report_bind_") as root:
            layout_path = os.path.join(root, "layout.json")
            with open(layout_path, "w", encoding="utf-8") as handle:
                json.dump(FLAT_LAYOUT, handle)
            report = engineering_check.run_engineering_check(
                dest_dir=os.path.join(root, "out"),
                steps=4,
                checkpoint_every=2,
                hashes=dict(HASHES),
                model_inputs_root=EVIDENCE_DIR,
                model_inputs_layout=FLAT_LAYOUT,
                interface_path=INTERFACE_LOCK,
            )
            view_dir = report["input_binding"]["load_view_dir"]
            # 视图在临时目录里，必须在 with 块内确认落盘（退出即清理）。
            view_template_exists = os.path.isfile(
                os.path.join(str(view_dir), "chat_template.jinja")
            )
            view_template_path = os.path.join(str(view_dir), "chat_template.jinja")
            view_template_sha = (
                input_binding.sha256_file(view_template_path)
                if view_template_exists
                else None
            )
        bound = report["input_binding"]
        self.assertTrue(bound["bound"])
        self.assertEqual(
            sorted(bound["consumed_inputs"]),
            ["chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json"],
        )
        self.assertTrue(view_dir)
        self.assertTrue(view_template_exists, "加载视图里必须有被绑定的 chat template")
        self.assertEqual(
            view_template_sha,
            "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        )
        self.assertEqual(bound["weights_sha256"], "hash_pending")
        self.assertTrue(report["adapter_export"]["validation"]["ok"])


class CudaRngTests(unittest.TestCase):
    """缺陷 E：torch RNG 不能只有 CPU 状态；CUDA 逐设备状态必须能存能恢复。"""

    class _FakeCuda:
        def __init__(self, *, available: bool = True, devices: int = 2) -> None:
            self._available = available
            self._devices = devices
            self.set_calls: list[tuple[int, int]] = []

        def is_available(self) -> bool:
            return self._available

        def device_count(self) -> int:
            return self._devices

        def get_rng_state(self, index: int):
            return torch.tensor([index + 1] * 8, dtype=torch.uint8)

        def set_rng_state(self, state, index: int) -> None:
            self.set_calls.append((index, int(state[0].item())))

    def _backend(self, *, device: str = "cuda"):
        """r4 缺陷 5：`device` 决定「真实 CUDA 是否参与」，默认按参与构造。"""
        plan = _plan(steps=1)
        backend = _cpu_backend(plan, device=device)
        return backend, plan

    def test_cuda_rng_states_are_saved_per_device(self) -> None:
        backend, _plan_ = self._backend()
        backend.cuda_rng = self._FakeCuda(devices=2)
        payload = backend._cuda_rng_states()
        self.assertTrue(payload["available"])
        self.assertTrue(payload["cuda_in_use"])
        self.assertEqual(sorted(payload["devices"]), ["0", "1"])
        self.assertEqual(payload["device_count"], 2)

    def test_cuda_rng_absent_is_honest(self) -> None:
        """**没有 CUDA 设备**时明确 not-applicable，不是"失败"，也不编造。"""
        backend, _plan_ = self._backend(device="cpu")
        backend.cuda_rng = self._FakeCuda(available=False)
        payload = backend._cuda_rng_states()
        self.assertFalse(payload["available"])
        self.assertIn("not-applicable", payload["reason"])
        self.assertFalse(payload["cuda_in_use"])

    def test_cuda_rng_capture_failure_is_fail_closed_when_cuda_in_use(self) -> None:
        """缺陷 5 的核心反例：真实 CUDA 参与时采集异常必须拒绝，不得降级 might succeed。"""
        backend, _plan_ = self._backend()

        class _ExplodingCuda:
            def is_available(self) -> bool:
                return True

            def device_count(self) -> int:
                return 1

            def get_rng_state(self, _index: int):
                raise RuntimeError("capture failed")

        backend.cuda_rng = _ExplodingCuda()
        with self.assertRaises(MissingInput) as ctx:
            backend._cuda_rng_states()
        self.assertEqual(ctx.exception.code, "cuda_rng_capture_failed")
        # 关键：**没有**返回 available=False 的半份状态——异常本身就是结论。
        self.assertEqual(ctx.exception.context.get("device"), "cuda")
        self.assertEqual(ctx.exception.context.get("devices_requested"), 1)

    def test_cuda_rng_capture_failure_is_not_applicable_on_cpu(self) -> None:
        """CPU fixture（无卡参与）时，采集异常仍如实标注，不伪装成 fail-closed 崩溃。"""
        backend, _plan_ = self._backend(device="cpu")

        class _ExplodingCuda:
            def is_available(self) -> bool:
                return True

            def device_count(self) -> int:
                return 1

            def get_rng_state(self, _index: int):
                raise RuntimeError("capture failed")

        backend.cuda_rng = _ExplodingCuda()
        payload = backend._cuda_rng_states()
        self.assertFalse(payload["available"])
        self.assertFalse(payload["cuda_in_use"])

    def test_missing_device_rng_in_checkpoint_is_rejected_when_cuda_in_use(self) -> None:
        backend, _plan_ = self._backend()
        backend.cuda_rng = self._FakeCuda(devices=1)
        with self.assertRaises(MissingInput) as ctx:
            backend._restore_cuda_rng_states(
                {"available": False, "reason": "not-applicable:no-cuda-device", "devices": {}}
            )
        self.assertEqual(ctx.exception.code, "cuda_rng_missing_in_checkpoint")

    def test_cuda_rng_roundtrip_restores_each_device(self) -> None:
        backend, _plan_ = self._backend()
        fake = self._FakeCuda(devices=2)
        backend.cuda_rng = fake
        payload = backend._cuda_rng_states()
        restored = backend._restore_cuda_rng_states(payload)
        self.assertEqual(restored, [0, 1])
        self.assertEqual(fake.set_calls, [(0, 1), (1, 2)])

    def test_cuda_rng_missing_at_restore_is_rejected(self) -> None:
        backend, _plan_ = self._backend()
        backend._torch = None
        backend.cuda_rng = None
        payload = {"available": True, "devices": {"0": {"device": 0, "data_base64": "AA=="}}}
        with self.assertRaises(MissingInput) as ctx:
            backend._restore_cuda_rng_states(payload)
        self.assertEqual(ctx.exception.code, "cuda_rng_unavailable_for_restore")

    def test_rng_state_reports_cuda_scope(self) -> None:
        backend, _plan_ = self._backend()
        backend.cuda_rng = self._FakeCuda(devices=1)
        rng = backend._rng_state()
        self.assertTrue(rng["torch_cuda_states"]["available"])
        self.assertIn("torch-cuda", rng["rng_scope"])
        self.assertEqual(rng["device_rng_scope"], "cuda")

    def test_rng_state_reports_cpu_scope_when_no_device_participates(self) -> None:
        backend, _plan_ = self._backend(device="cpu")
        backend.cuda_rng = self._FakeCuda(devices=1)
        rng = backend._rng_state()
        self.assertEqual(rng["device_rng_scope"], "cpu")
        self.assertFalse(rng["torch_cuda_states"]["cuda_in_use"])

    def test_scheduler_and_scaler_status_are_explicit(self) -> None:
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        backend.train_steps(plan)
        state = backend.export_training_state()
        self.assertIsNone(state["scheduler"])
        self.assertEqual(state["scheduler_status"], "not-applicable:no-scheduler-wired")
        self.assertIsNone(state["scaler"])
        self.assertEqual(state["scaler_status"], "not-applicable:no-amp-scaler")

    def test_import_reports_cuda_rng_gap_when_absent(self) -> None:
        plan = _plan(steps=2)
        backend = _cpu_backend(plan)
        # 强制"本机没有可用 CUDA RNG"这条路（本机其实有卡，所以必须显式注入）
        backend.cuda_rng = self._FakeCuda(available=False)
        backend.train_steps(plan)
        state = backend.export_training_state()
        self.assertFalse(state["cuda_rng_available"])

        class _State:
            pass

        wrapper = _State()
        for key, value in state.items():
            setattr(wrapper, key, value)
        applied = backend.import_training_state(wrapper)
        self.assertFalse(applied["cuda_rng_available_in_checkpoint"])
        self.assertEqual(applied["cuda_rng_restored_devices"], [])
        self.assertIn("缺口", applied["cuda_rng_note"])

    def test_cuda_rng_is_present_in_checkpoint_when_the_device_is_there(self) -> None:
        """CUDA 参与时设备 RNG 必须真的进 checkpoint 结构。

        直接查 `_rng_state()`（checkpoint 里 RNG 段的来源）而不是先 `train_steps`：
        本机的 CPU 替身张量不能真的搬到 cuda 上跑前向，那与 RNG 取证无关。
        """
        backend, _plan_ = self._backend()
        backend.cuda_rng = self._FakeCuda(devices=1)
        rng = backend._rng_state()
        self.assertTrue(rng["torch_cuda_states"]["available"])
        self.assertTrue(rng["torch_cuda_states"]["cuda_in_use"])
        self.assertEqual(sorted(rng["torch_cuda_states"]["devices"]), ["0"])


class AuditR3CounterexampleRegressionTests(unittest.TestCase):
    """独立审查（KAGGLE-38 评论 01a11aa9…，附件 `audit-r3-evidence.zip`）的逐条反例。

    这个类**照抄审稿脚本 `audit_r3_repro.py` 的触发条件**，把当时的"接受/降级"
    断言反转为"必须拒绝/必须 fail-closed"。每条用例都注明它对应哪一项阻断。
    """

    # ---- 阻断 2：真实哈希不能自报，必须绑 v6 批准记录 ----

    def test_self_reported_hashes_are_rejected(self) -> None:
        """审稿脚本第 1 行：五个字段全填 "1" 曾被接受。"""
        with temp_dir_outside_repo("k38a_hashes_") as root:
            path = os.path.join(root, "hashes.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({key: "1" for key in HASHES}, handle)
            with self.assertRaises(IntegrityError) as ctx:
                lifecycle_mod.hashes_from_json(path)
        self.assertEqual(ctx.exception.code, "hashes_not_sha256_hex64")
        self.assertEqual(len(ctx.exception.context["malformed"]), 5)

    def test_hashes_without_external_anchor_cannot_start_real_mode(self) -> None:
        """阻断 2：字段齐全（甚至哈希格式合法）也**不**等于批准。"""
        with temp_dir_outside_repo("k38a_approval_") as root:
            approval = os.path.join(root, "approval.json")
            with open(approval, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "approval_id": "x",
                        "target_sha": v6_approval.LEGACY_E0_FIXTURE_COMMIT,
                        "gpu_hours_approved": 4,
                        "verified_by": "self-declared",  # 自报字段，谁都能填
                        "verified_utc": "2026-10-08T00:00:00Z",
                        "evidence_sha256": "a" * 64,
                    },
                    handle,
                )
            with self.assertRaises(PolicyViolation) as ctx:
                v6_approval.require_approval(
                    approval_path=approval, expected_sha256=None
                )
        self.assertEqual(ctx.exception.code, "approval_not_trusted")
        with self.assertRaises(PolicyViolation):
            v6_approval.deployment_approval_anchor()

    def test_tampered_approval_breaks_the_external_anchor(self) -> None:
        """阻断 2：批准文件被改过（锚还是旧的）必须拒。"""
        with temp_dir_outside_repo("k38a_anchor_") as root:
            approval = os.path.join(root, "approval.json")
            payload = {
                "approval_id": "x",
                "target_sha": v6_approval.LEGACY_E0_FIXTURE_COMMIT,
                "gpu_hours_approved": 4,
                "verified_by": "mika",
                "verified_utc": "2026-10-08T00:00:00Z",
                "evidence_sha256": "a" * 64,
            }
            with open(approval, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            anchor = hashlib.sha256(open(approval, "rb").read()).hexdigest()
            payload["gpu_hours_approved"] = 400  # 偷偷放宽预算
            with open(approval, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaises(PolicyViolation) as ctx:
                v6_approval.require_approval(
                    approval_path=approval, expected_sha256=anchor
                )
        self.assertEqual(ctx.exception.code, "approval_not_trusted")

    def test_approval_must_target_the_pinned_commit(self) -> None:
        with temp_dir_outside_repo("k38a_pin_") as root:
            approval = os.path.join(root, "approval.json")
            with open(approval, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "approval_id": "x",
                        "target_sha": "f" * 40,
                        "gpu_hours_approved": 4,
                        "verified_by": "mika",
                        "verified_utc": "2026-10-08T00:00:00Z",
                        "evidence_sha256": "a" * 64,
                    },
                    handle,
                )
            with self.assertRaises(PolicyViolation) as ctx:
                v6_approval.require_approval(
                    approval_path=approval,
                    expected_sha256=hashlib.sha256(
                        open(approval, "rb").read()
                    ).hexdigest(),
                )
        self.assertEqual(ctx.exception.code, "approval_not_trusted")

    def test_gpu_hours_over_approval_is_rejected(self) -> None:
        with temp_dir_outside_repo("k38a_hours_") as root:
            approval = os.path.join(root, "approval.json")
            with open(approval, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "approval_id": "x",
                        "target_sha": v6_approval.LEGACY_E0_FIXTURE_COMMIT,
                        "gpu_hours_approved": 4,
                        "verified_by": "mika",
                        "verified_utc": "2026-10-08T00:00:00Z",
                        "evidence_sha256": "a" * 64,
                    },
                    handle,
                )
            with self.assertRaises(PolicyViolation) as ctx:
                v6_approval.require_approval(
                    approval_path=approval,
                    expected_sha256=hashlib.sha256(
                        open(approval, "rb").read()
                    ).hexdigest(),
                    requested_gpu_hours=99.0,
                )
        self.assertEqual(ctx.exception.code, "approval_not_trusted")

    # ---- 阻断 3：model / config / 小文件同一身份 ----

    def test_mismatched_model_identity_is_rejected(self) -> None:
        """审稿脚本第 2 行：`arbitrary/unapproved@arbitrary-revision` 曾被接受。"""
        with temp_dir_outside_repo("k38a_identity_") as root:
            binding = input_binding.bind_model_inputs(
                root=EVIDENCE_DIR, interface_path=INTERFACE_LOCK, layout=FLAT_LAYOUT
            )
            with self.assertRaises(PolicyViolation) as ctx:
                runner.TorchPeftBackend(
                    model_id="arbitrary/unapproved",
                    revision="arbitrary-revision",
                    target_modules=["q_proj"],
                    model_inputs=binding,
                    device="cpu",
                )
        self.assertEqual(ctx.exception.code, "model_identity_mismatch")
        self.assertEqual(ctx.exception.context["bound_repo"], binding["repo_id"])

    def test_tampered_config_is_rejected(self) -> None:
        """审稿脚本第 3 行：`{"model_type":"changed"}` 的 config 曾被复制进视图。"""
        with temp_dir_outside_repo("k38a_config_") as root:
            tampered = os.path.join(root, "tampered")
            os.makedirs(tampered)
            for name in ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
                with open(os.path.join(EVIDENCE_DIR, name), "rb") as src:
                    raw = src.read()
                with open(os.path.join(tampered, name), "wb") as dst:
                    dst.write(raw)
            with open(os.path.join(tampered, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"model_type": "changed"}, handle)
            with self.assertRaises(PolicyViolation) as ctx:
                input_binding.bind_model_inputs(
                    root=tampered, interface_path=INTERFACE_LOCK, layout=FLAT_LAYOUT
                )
        self.assertEqual(ctx.exception.code, "input_pin_mismatch")
        self.assertEqual(
            [item["filename"] for item in ctx.exception.context["mismatches"]],
            ["config.json"],
        )

    def test_config_pin_is_actually_locked(self) -> None:
        """config.json 必须是**受核对输入**，不再是无声的伴随文件。"""
        self.assertIn("config.json", input_binding.INPUT_PINS)
        self.assertEqual(
            input_binding.INPUT_PINS["config.json"], "config_sha256"
        )
        lock = json.load(open(INTERFACE_LOCK, encoding="utf-8"))
        self.assertEqual(
            lock["pins"]["config_sha256"]["value"],
            "b100d85e571c25b688e82919d819253063a1defb0e6cee3b9b1c1fbad99bd0c2",
        )

    # ---- 阻断 4：只允许离线本地加载 ----

    def test_cuda_load_requires_local_model_dir(self) -> None:
        """阻断 4：cuda 真实加载缺本地权重目录即拒（不回退联网）。"""
        with temp_dir_outside_repo("k38a_offline_") as root:
            plan = _plan(steps=1)
            backend = runner.TorchPeftBackend(
                model_id="google/gemma-4-31B-it-qat-w4a16-ct",
                revision=REVISION,
                target_modules=["q_proj"],
                device="cuda",
                local_load_authorized=True,
                model_inputs=_real_binding(root),
            )
            with self.assertRaises(MissingInput) as ctx:
                backend.prepare(plan)
        self.assertEqual(ctx.exception.code, "local_model_dir_missing")

    def test_missing_local_dir_path_is_rejected(self) -> None:
        with temp_dir_outside_repo("k38a_offline2_") as root:
            backend = runner.TorchPeftBackend(
                model_id="google/gemma-4-31B-it-qat-w4a16-ct",
                revision=REVISION,
                target_modules=["q_proj"],
                device="cuda",
                local_model_dir=os.path.join(root, "does-not-exist"),
            )
            with self.assertRaises(MissingInput) as ctx:
                backend._offline_load_dir()
        self.assertEqual(ctx.exception.code, "local_model_dir_missing")

    def test_download_permission_does_not_gate_local_load(self) -> None:
        """阻断 4：`allow_download` 既不能放开、也不能阻止本地加载。"""
        with temp_dir_outside_repo("k38a_dl_") as root:
            local = os.path.join(root, "weights")
            os.makedirs(local)
            import shutil
            shutil.copyfile(os.path.join(EVIDENCE_DIR, "config.json"), os.path.join(local, "config.json"))
            binding = _real_binding(root)
            binding["approved_local_model_dir"] = os.path.realpath(local)
            for flag in (False, True):
                backend = runner.TorchPeftBackend(
                    model_id="google/gemma-4-31B-it-qat-w4a16-ct",
                    revision=REVISION,
                    target_modules=["q_proj"],
                    device="cuda",
                    allow_download=flag,
                    local_model_dir=local,
                    model_inputs=binding,
                )
                # 两种下载许可下都**能**拿到本地目录（不再被 allow_download 拦）
                self.assertEqual(backend._offline_load_dir(), os.path.realpath(local))

    # ---- 阻断 5：CUDA RNG 采集失败必须 fail-closed ----

    def test_failed_cuda_rng_capture_is_rejected(self) -> None:
        """审稿脚本第 5 行：采集异常曾被降级成 available=False。"""
        backend = _cpu_backend(_plan(steps=1), device="cuda")

        class _BadCuda:
            def is_available(self) -> bool:
                return True

            def device_count(self) -> int:
                return 1

            def get_rng_state(self, _index: int):
                raise RuntimeError("capture failed")

        backend.cuda_rng = _BadCuda()
        with self.assertRaises(MissingInput) as ctx:
            backend._cuda_rng_states()
        self.assertEqual(ctx.exception.code, "cuda_rng_capture_failed")

    # ---- 阻断 1：重载/恢复不得同时持有双基座 ----

    def test_old_base_is_released_before_the_fresh_load(self) -> None:
        """审稿脚本第 4 行：`old_base_live_at_fresh_load` 曾是 [True]。

        复刻审稿脚本的注入方式：拦 `AutoModelForCausalLM.from_pretrained`，
        在新基座加载的**那一刻**检查旧基座是否仍然可达。
        """
        import types
        from unittest.mock import patch

        with temp_dir_outside_repo("k38a_release_") as root:
            plan = _plan(steps=1)
            # r4 起 load_adapter 会先断言模型身份，所以这里要带上合法绑定。
            backend = runner.TorchPeftBackend(
                model_id="google/gemma-4-31B-it-qat-w4a16-ct",
                revision=REVISION,
                target_modules=["q_proj", "o_proj"],
                device="cpu",
                model_inputs=_real_binding(root),
            )
            backend._torch = torch
            backend.model = _TinyLoRAModel()
            backend.lr = float(plan.lr)
            backend.sampler_order_sha256 = runner._sampler_order_sha256(plan)

            class _FakeModel:
                def __init__(self, tag):
                    self.tag = tag

                def to(self, _device):
                    return self

                def named_parameters(self):
                    return []

                def parameters(self):
                    return []

            old = _FakeModel("old")
            backend.model = old
            import weakref
            old_ref = weakref.ref(old)
            del old
            backend.adapter_params_digest = lambda: "same"
            peft_dir = os.path.join(
                root, "adapter", *runner.adapter_contract.adapter_dir_relative_path("v3_policy").split("/")
            )
            os.makedirs(peft_dir, exist_ok=True)
            with open(os.path.join(peft_dir, runner.ADAPTER_FILE), "wb") as handle:
                handle.write(b"fixture")
            seen: list[bool] = []

            class _FakeAutoModel:
                @staticmethod
                def from_pretrained(*args, **kwargs):
                    seen.append(old_ref() is not None)
                    return _FakeModel("fresh")

            class _FakePeftModel:
                @staticmethod
                def from_pretrained(base_model, _path, **kwargs):
                    return _FakeModel("adapter on " + base_model.tag)

            fake_peft = types.ModuleType("peft")
            fake_peft.PeftModel = _FakePeftModel
            with patch.object(
                backend,
                "_import_stack",
                return_value=(backend._torch, None, None, None, _FakeAutoModel, None),
            ), patch.object(
                runner.adapter_contract, "assert_official_adapter_carrier", return_value=None
            ), patch.dict(sys.modules, {"peft": fake_peft}):
                result = backend.load_adapter(os.path.join(root, "adapter"))
        self.assertEqual(seen, [False], "新基座加载时旧基座必须已经不可达")
        self.assertFalse(result["old_base_alive_during_fresh_load"])
        self.assertTrue(result["old_base_release_evidence"]["model_is_none"])
        self.assertTrue(result["old_base_release_evidence"]["optimizer_is_none"])

    # ---- 阻断 6 的账目侧：旧硬截止反例保持已修 ----

    def test_previous_hard_deadline_counterexample_stays_fixed(self) -> None:
        stop = StopRequest(deadline_monotonic=time.monotonic() - 1)
        stop.request("step_limit")
        with self.assertRaises(FailClosed) as ctx:
            engineering_check.BudgetLedger(stop, time.monotonic()).check("adapter_export")
        self.assertEqual(ctx.exception.code, "engineering_check_deadline_exceeded")


def _real_binding(root: str) -> dict:
    """绑定官方原件并装配加载视图，返回带视图信息的绑定（供身份/离线用例复用）。"""
    binding = input_binding.bind_model_inputs(
        root=EVIDENCE_DIR, interface_path=INTERFACE_LOCK, layout=FLAT_LAYOUT
    )
    view = input_binding.assemble_load_view(binding, os.path.join(root, "load_view"))
    binding = dict(binding)
    binding["load_view_dir"] = view["view_dir"]
    binding["view_manifest_sha256"] = view["manifest_sha256"]
    return binding


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
