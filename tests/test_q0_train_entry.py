"""KAGGLE-26 / Q0 报告 🔴-1 与「可执行训练入口」的回归测试（E0 作者自测）。

覆盖两条必须"能失败"的检查：

1. 🔴-1：assistant 的 `content` 必须能在渲染后的 `input_ids` 里按 token 子序列定位到。
   旧实现一旦 assistant 带 `arguments` 就静默丢弃 `content`，本文件的
   `AssistantContentPresenceTests.test_content_token_subsequence_is_present` 会失败。
2. 🔴 §7：`entry.start()` 必须是**可执行训练入口**。本文件断言仓库里存在真实训练代码
   （`import torch` / `get_peft_model` / `.backward()` / `optimizer.step()`），
   并端到端跑通 CPU 自检（前向 / 反向 / 优化器步进 / adapter-only 保存 / 重载）。
"""

from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests._adapter_fixtures import (  # noqa: E402
    legacy_self_report_manifest,
    make_export_manifest,
    make_official_export_dir,
)
from tests._tmp import temp_dir, temp_dir_outside_repo  # noqa: E402

from v3.common.errors import FailClosed, PolicyViolation  # noqa: E402
from v3.train import entry, fixtures, profile as profile_mod, runner  # noqa: E402
from v3.train.masks import LabelResult, build_labels, render_and_mask  # noqa: E402
from v3.train.template import (  # noqa: E402
    ShimRenderer,
    assert_roundtrip_arguments,
    assert_text_in_input_ids,
)

RUNNER_PATH = os.path.join(REPO_ROOT, "v3", "train", "runner.py")


def measured_profile(*, peak_gpu_gib: float, host_rss_gib: float) -> dict:
    """构造一份**结构完整、来源可核**的测量记录（本机没有 GPU，因此是合成证据）。

    刻意复用生产模块的字段名与白名单 provenance，用来单独验证**消费者**的判定逻辑；
    真实测量只在获批 GPU 作业上由 `v3/train/profile.py` 产出。
    """
    return {
        "seq_len": 2048,
        "peak_gpu_gib": float(peak_gpu_gib),
        "host_rss_gib": float(host_rss_gib),
        "gpu_measurement_status": profile_mod.GPU_MEASURED,
        "gpu_provenance": profile_mod.REAL_GPU_PROVENANCE,
        "is_real_measurement": True,
        "source": "measured-%s" % profile_mod.REAL_GPU_PROVENANCE,
        "host_measurement": {
            "probe": "proc/self/status:VmHWM",
            "is_real_measurement": True,
        },
        "workload": {
            "backend": "torch-peft-bf16-lora",
            "requires_gpu": True,
            "uses_gpu": True,
            "steps": 8,
        },
        "phases": [{"phase": name} for name in profile_mod.REQUIRED_PHASES],
        "missing_phases": [],
    }


class AssistantContentPresenceTests(unittest.TestCase):
    """🔴-1：assistant content 不得被静默丢弃。"""

    def test_content_token_subsequence_is_present(self) -> None:
        renderer = ShimRenderer()
        rendered = renderer.render(
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "REASONING_PLAN_XYZ",
                    "tool_name": "run_command",
                    "arguments": {"command": "ls"},
                    "loss_eligible": True,
                    "step": 1,
                },
            ]
        )
        position = assert_text_in_input_ids(renderer, rendered, "REASONING_PLAN_XYZ")
        self.assertGreaterEqual(position, 0)

    def test_every_fixture_assistant_content_is_locatable(self) -> None:
        renderer = ShimRenderer()
        checked = 0
        for fixture in fixtures.build_fixtures():
            sample = renderer.render(fixture.messages, thinking=fixture.thinking)
            for message in fixture.messages:
                if message.get("role") != "assistant":
                    continue
                content = str(message.get("content", ""))
                if not content:
                    continue
                checked += 1
                self.assertGreaterEqual(
                    assert_text_in_input_ids(renderer, sample, content),
                    0,
                    "%s 的 assistant content 没有进入 input_ids" % fixture.fixture_id,
                )
        self.assertEqual(checked, 24)

    def test_empty_arguments_object_keeps_content(self) -> None:
        """F19 的 `arguments={}` 与 F01 的 `arguments=None` 必须同样保留 content。"""
        renderer = ShimRenderer()
        with_empty_args = renderer.render(
            [
                {"role": "assistant", "content": "LOCATE v1 anchor: none", "tool_name": "submit_patch",
                 "arguments": {}, "loss_eligible": True, "step": 1},
            ]
        )
        self.assertGreaterEqual(
            assert_text_in_input_ids(renderer, with_empty_args, "LOCATE v1 anchor: none"), 0
        )


class ThinkingChannelTests(unittest.TestCase):
    """Y3：thinking 开关必须有可观测差异，reasoning 通道永远 context-only。"""

    def _render(self, thinking: bool):
        messages = [
            {"role": "user", "content": "Find it."},
            {
                "role": "assistant",
                "content": "thinking... then act",
                "tool_name": "run_command",
                "arguments": {"command": "ls"},
                "loss_eligible": True,
                "step": 1,
                "phase": "localize",
            },
        ]
        return ShimRenderer().render(messages, thinking=thinking)

    def test_thinking_flag_changes_input_ids(self) -> None:
        self.assertNotEqual(self._render(True).input_ids, self._render(False).input_ids)

    def test_reasoning_channel_is_never_supervised(self) -> None:
        rendered = self._render(True)
        reasoning = [span for span in rendered.spans if span.channel == "reasoning"]
        self.assertTrue(reasoning)
        labels = build_labels(rendered)
        for span in reasoning:
            self.assertFalse(span.supervised)
            for index in range(span.start, span.end):
                self.assertEqual(labels.labels[index], -100)

    def test_thinking_off_has_no_reasoning_channel(self) -> None:
        self.assertEqual([s for s in self._render(False).spans if s.channel == "reasoning"], [])

    def test_fixture_suite_reports_thinking_channels(self) -> None:
        report = fixtures.run_all()
        by_id = {item["fixture_id"]: item for item in report["results"]}
        self.assertGreater(by_id["F04-thinking-on"]["stats"]["reasoning_context_tokens"], 0)
        self.assertEqual(by_id["F05-thinking-off"]["stats"]["reasoning_span_count"], 0)


class ArgsRoundtripTests(unittest.TestCase):
    """Y2：roundtrip 必须真的经过 renderer。"""

    def test_fake_renderer_fails(self) -> None:
        class NeverCalled:
            def encode(self, text):
                raise AssertionError("encode 不该被调用")

            def render(self, messages, thinking=False):
                raise AssertionError("render 必须被调用")

        with self.assertRaises(AssertionError):
            assert_roundtrip_arguments(NeverCalled(), {"a": 1})

    def test_real_renderer_recovers_arguments(self) -> None:
        outcome = assert_roundtrip_arguments(
            ShimRenderer(), {"path": "a/b.py", "nested": {"k": [1, 2, "三"]}}
        )
        self.assertEqual(outcome["recovered"], {"path": "a/b.py", "nested": {"k": [1, 2, "三"]}})
        self.assertEqual(outcome["channel"], "action")


class MaskAccountingTests(unittest.TestCase):
    """Y13 与 🔵-1。"""

    def test_loss_counts_buckets_are_not_constant_zero(self) -> None:
        bare = LabelResult(input_ids=[1, 2, 3], labels=[-100, 2, 3]).loss_counts()
        self.assertEqual(bare["supervised"], 2)

        renderer = ShimRenderer()
        _, labels, _ = render_and_mask(
            renderer,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "ask"},
                {"role": "tool", "content": "obs", "tool_name": "run_command", "step": 1},
                {"role": "assistant", "content": "thinking step", "tool_name": "run_command",
                 "arguments": {"command": "ls"}, "loss_eligible": True, "step": 2},
            ],
            thinking=True,
        )
        counts = labels.loss_counts()
        self.assertGreater(counts["system"] + counts["user"] + counts["tool"], 0)
        self.assertGreater(counts["reasoning_context"], 0)
        self.assertGreater(counts["supervised"], 0)
        # 分桶必须随样本变化，而不是恒定的同一组数字
        other = LabelResult(
            input_ids=[1, 2, 3],
            labels=[-100, 2, 3],
            span_report=[
                {"role": "tool", "start": 0, "end": 1, "loss_tokens": 0,
                 "channel": "action", "supervised": False},
                {"role": "system", "start": 1, "end": 3, "loss_tokens": 0,
                 "channel": "action", "supervised": False},
            ],
        ).loss_counts()
        self.assertNotEqual(counts["system"], other["system"])
        self.assertNotEqual(counts["tool"], other["tool"])

    def test_build_labels_does_not_mutate_input(self) -> None:
        renderer = ShimRenderer()
        sample = renderer.render([{"role": "user", "content": "x"}])
        before = list(sample.input_ids)
        build_labels(sample, pad_to=len(before) + 5)
        self.assertEqual(sample.input_ids, before)
        build_labels(sample, pad_to=len(before) + 5)  # 二次调用不得抛异常


class TrainingEntryImplementationTests(unittest.TestCase):
    """§7：训练实现必须在仓库里，而不是注释里的承诺。"""

    def test_runner_contains_real_training_calls(self) -> None:
        with open(RUNNER_PATH, "r", encoding="utf-8") as handle:
            source = handle.read()
        for needle in ("import torch", "get_peft_model", "loss.backward()", "optimizer.step()",
                       "save_pretrained", "PeftModel.from_pretrained", "LoraConfig"):
            self.assertIn(needle, source, "runner.py 缺少真实训练调用：%s" % needle)

    def test_gpu_execution_not_implemented_is_gone(self) -> None:
        """旧实现的"无条件抛出"必须消失。"""
        offenders = []
        for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, "v3")):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                with open(path, "r", encoding="utf-8") as handle:
                    if "gpu_execution_not_implemented" in handle.read():
                        offenders.append(os.path.relpath(path, REPO_ROOT))
        self.assertEqual(offenders, [])

    def test_cpu_self_check_passes_every_check(self) -> None:
        report = runner.self_check_cpu()
        self.assertTrue(report["all_passed"], report)
        for name, passed in report["checks"].items():
            self.assertTrue(passed, "自检项未通过：%s" % name)
        self.assertFalse(report["is_real_model_training"])
        # loss 是否下降只作观察值，不作为通过条件（否则会造出会骗人的绿灯）
        self.assertIn("loss_decreased_observation_only", report)
        self.assertNotIn("loss_decreased", report["checks"])

    def test_out_of_range_token_is_fail_closed(self) -> None:
        plan = runner.TrainRunPlan(
            steps=1, lr=0.1, seq_len=4, lora_rank=2, lora_alpha=4,
            batches=[runner.TrainBatch(input_ids=[1, 999, 3], labels=[1, 999, 3])],
        )
        backend = runner.SyntheticBackend(vocab=8, dim=4)
        with self.assertRaises(PolicyViolation) as ctx:
            backend.prepare(plan)
        self.assertIn(ctx.exception.code, ("token_id_out_of_range", "label_id_out_of_range"))


class TrainingLoopTests(unittest.TestCase):
    """训练循环本身：前向 / 反向 / 步进 / 保存 / 重载都要真的发生。

    临时目录用仓库自带的 `tests._tmp.temp_dir`：本机禁写系统临时目录，
    且 `tempfile.mkdtemp` 建的目录带受限 ACL（写入会 Permission denied）。
    """

    def _plan(self, steps: int = 8) -> runner.TrainRunPlan:
        return runner.TrainRunPlan(
            steps=steps,
            lr=0.2,
            seq_len=10,
            lora_rank=4,
            lora_alpha=8,
            batches=runner.build_smoke_batches(seq_len=10),
        )

    def test_full_loop_produces_evidence(self) -> None:
        with temp_dir("train_") as workdir:
            backend = runner.SyntheticBackend(vocab=32, dim=6)
            report = runner.run_training(backend, self._plan(steps=8), workdir)
            trained = report["trained"]
            self.assertTrue(report["executed"])
            self.assertTrue(trained["grad_norm_nonzero"])
            self.assertEqual(trained["optimizer_step_count"], 8)
            self.assertTrue(trained["base_params_frozen"])
            self.assertTrue(trained["adapter_params_changed"])
            self.assertTrue(report["saved"]["adapter_only"])
            self.assertFalse(report["saved"]["is_official_submission_carrier"])
            self.assertTrue(report["reloaded"]["matches_saved_adapter"])
            self.assertTrue(os.path.exists(os.path.join(workdir, runner.EVIDENCE_FILE)))

    def test_batch_without_supervised_tokens_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation):
            runner.TrainBatch(input_ids=[1, 2, 3], labels=[-100, -100, -100])

    def test_zero_steps_is_rejected(self) -> None:
        plan = self._plan(steps=0)
        with self.assertRaises(FailClosed):
            plan.assert_runnable()

    def test_adapter_reload_detects_tampering(self) -> None:
        with temp_dir("train_tamper_") as workdir:
            backend = runner.SyntheticBackend(vocab=32, dim=6)
            runner.run_training(backend, self._plan(steps=4), workdir)
            path = os.path.join(workdir, runner.SYNTHETIC_ADAPTER_FILE)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["lora_b"][0][0] = float(payload["lora_b"][0][0]) + 1.0
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            reloader = runner.SyntheticBackend(vocab=32, dim=6)
            reloader.prepare(self._plan(steps=4))
            with self.assertRaises(FailClosed):
                reloader.load_adapter(workdir)


class GateMeasurementTests(unittest.TestCase):
    """闸门必须是测量结果，不是硬编码常量。"""

    def test_gates_are_measured(self) -> None:
        state = entry.measure_gates()
        self.assertEqual(sorted(state["gates"]), sorted(entry.GATE_NAMES))
        # fixture 全过 + CPU 自检全过 => T1 工程闸门为真（这是实测结果）
        self.assertTrue(state["gates"]["t1_engineering_pass"])
        # 没有实测内存 profile / 导出 manifest => 必须为假，并给出原因
        self.assertFalse(state["gates"]["memory_plan_pass"])
        self.assertFalse(state["gates"]["export_manifest_pass"])
        self.assertTrue(state["evidence"]["memory_plan_reason"])

    def test_memory_gate_uses_measurement(self) -> None:
        """G7b：内存闸门只认**真实测量**，手打的数字不再能过门。"""
        hand_written = entry.measure_gates(
            memory_profile={"peak_gpu_gib": 40.0, "host_rss_gib": 8.0, "source": "unit-test"},
            run_cpu_self_check=False,
        )
        self.assertFalse(
            hand_written["gates"]["memory_plan_pass"],
            "手打 profile 不得通过内存闸门（G7b）",
        )
        self.assertIn(
            "来源校验", hand_written["evidence"]["memory_plan_reason"] or ""
        )
        measured = entry.measure_gates(
            memory_profile=measured_profile(peak_gpu_gib=40.0, host_rss_gib=8.0),
            run_cpu_self_check=False,
        )
        self.assertTrue(measured["gates"]["memory_plan_pass"])
        self.assertTrue(measured["evidence"]["memory_profile_provenance"]["ok"])
        too_big = entry.measure_gates(
            memory_profile=measured_profile(peak_gpu_gib=999.0, host_rss_gib=8.0),
            run_cpu_self_check=False,
        )
        self.assertFalse(too_big["gates"]["memory_plan_pass"])

    def test_export_gate_requires_artifacts_and_rejects_self_report(self) -> None:
        """G8：导出闸门要求 v2 契约 + 绑定的产物目录，并拒绝自报通过字段。"""
        empty = entry.measure_gates(export_manifest={"files": []}, run_cpu_self_check=False)
        self.assertFalse(empty["gates"]["export_manifest_pass"])

        with temp_dir_outside_repo("gate_export_") as export_dir:
            make_official_export_dir(export_dir)
            manifest = make_export_manifest(export_dir)
            complete = entry.measure_gates(
                export_manifest=manifest,
                export_dir=export_dir,
                run_cpu_self_check=False,
            )
            self.assertTrue(complete["gates"]["export_manifest_pass"])
            self.assertTrue(
                complete["evidence"]["export_manifest"]["adapter_validation"]["verified_from_artifacts"]
            )
            # 清单与产物脱钩（不给 --export-dir）即不通过。
            unbound = entry.measure_gates(
                export_manifest=manifest, run_cpu_self_check=False
            )
            self.assertFalse(unbound["gates"]["export_manifest_pass"])
            # 旧口径的手拼清单（自报 load_ok / fixture_pass）必须被拒绝。
            legacy = entry.measure_gates(
                export_manifest=legacy_self_report_manifest(),
                export_dir=export_dir,
                run_cpu_self_check=False,
            )
            self.assertFalse(legacy["gates"]["export_manifest_pass"])
            problems = " ".join(legacy["evidence"]["export_manifest"]["problems"])
            self.assertIn("自报通过字段", problems)

    def test_start_is_fail_closed_without_authorization(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            entry.start(operator=None, gpu_hours=0.0, stage="T1")
        self.assertEqual(ctx.exception.code, "start_not_allowed")
        problems = ctx.exception.context.get("problems", [])
        self.assertTrue(any("GPU" in item or "授权" in item for item in problems), problems)
        # 闸门状态必须一并回报，而不是只说"不允许"
        self.assertFalse(ctx.exception.context["gates"]["memory_plan_pass"])
        self.assertTrue(ctx.exception.context["gates"]["t1_engineering_pass"])

    def test_start_requires_a_plan_even_when_gates_pass(self) -> None:
        """闸门全过但没有训练计划时，入口**不会自己编造数据**。"""
        manifest = {
            **{key: "a" * 64 for key in ("source_sha256", "data_sha256", "config_sha256",
                                         "code_sha256", "deps_sha256")},
            "adapter_only": True,
            "files": ["adapters/v3_policy/adapter_model.safetensors"],
        }
        with self.assertRaises(FailClosed) as ctx:
            entry.start(
                operator="unit-test",
                gpu_hours=1.0,
                stage="T1",
                v2_conflict_checked=True,
                memory_profile={"peak_gpu_gib": 40.0, "host_rss_gib": 8.0},
                export_manifest=manifest,
                backend=runner.SyntheticBackend(vocab=16, dim=4),
                plan=None,
            )
        self.assertIn(ctx.exception.code, ("start_not_allowed", "training_plan_missing"))

    def test_preflight_reports_implemented_entry(self) -> None:
        report = entry.preflight()
        self.assertTrue(report["training_entry"]["implemented"])
        self.assertIn("gates", report)
        self.assertIn("gates_evidence", report)
        self.assertFalse(report["can_start_training_now"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
