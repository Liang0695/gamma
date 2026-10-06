"""训练侧测试：目标模块、mask fixture、模板锁定、断点续训、配置闸门。"""

from __future__ import annotations

import os
import unittest

from tests._tmp import temp_dir
from v3.common.errors import Blocked, IntegrityError, MissingInput, PolicyViolation, UnverifiedLock
from v3.train.checkpoint import (
    CheckpointStore,
    assert_adapter_valid_lenient_for_tests,
    build_export_manifest,
    resume,
)
from v3.train.config import DEFAULT_CONFIG, TrainingConfig, assert_start_allowed
from v3.train.entry import main as entry_main
from v3.train.entry import preflight, start
from v3.train.fixtures import build_fixtures, run_all
from v3.train.masks import build_labels, render_and_mask
from v3.train.targets import (
    MemoryPlan,
    ModulePlan,
    adapter_bytes,
    adapter_train_state_gib,
    assert_no_auto_gpu,
    expected_module_count,
    is_frozen,
    lora_param_count,
)
from v3.train.template import (
    CANONICAL_TEMPLATE_SHA256,
    OfficialTemplateRenderer,
    ShimRenderer,
)

HASHES = {
    "source_sha256": "1" * 64,
    "data_sha256": "2" * 64,
    "config_sha256": "3" * 64,
    "code_sha256": "4" * 64,
    "deps_sha256": "5" * 64,
}


def rng_state() -> dict:
    return {
        "python_state": "deadbeef",
        "numpy_state": "cafebabe",
        "torch_state": "feedface",
        "cuda_state": None,
    }


def cursor_state() -> dict:
    return {"sampler_order_sha256": "9" * 64, "position": 128}


class TargetTests(unittest.TestCase):
    def test_module_and_parameter_arithmetic(self) -> None:
        self.assertEqual(expected_module_count(), 120)
        self.assertEqual(lora_param_count(16), 28_672_000)
        self.assertEqual(adapter_bytes(16, "bf16"), 57_344_000)
        self.assertAlmostEqual(adapter_bytes(16, "bf16") / (1 << 20), 54.6875, places=3)
        self.assertAlmostEqual(adapter_train_state_gib(16), 0.4272, places=3)
        self.assertGreater(lora_param_count(16, include_mlp=True), lora_param_count(16))

    def test_full_module_plan_validates(self) -> None:
        names = [
            "model.language_model.layers.%d.self_attn.%s" % (layer, projection)
            for layer in range(60)
            for projection in ("q_proj", "o_proj")
        ]
        plan = ModulePlan.from_names(names)
        self.assertEqual(len(plan.matched), 120)
        plan.validate()
        summary = plan.to_dict()
        self.assertEqual(summary["trainable_params"], 28_672_000)
        self.assertEqual(summary["adapter_bytes_bf16"], 57_344_000)

    def test_incomplete_module_plan_is_blocked_without_explanation(self) -> None:
        names = [
            "model.language_model.layers.%d.self_attn.%s" % (layer, projection)
            for layer in range(59)
            for projection in ("q_proj", "o_proj")
        ]
        plan = ModulePlan.from_names(names)
        with self.assertRaises(MissingInput):
            plan.validate()

    def test_incomplete_module_plan_allowed_with_explained_mapping(self) -> None:
        names = [
            "model.language_model.layers.%d.self_attn.%s" % (layer, projection)
            for layer in range(59)
            for projection in ("q_proj", "o_proj")
        ]
        plan = ModulePlan.from_names(names)
        plan.explained_mapping = {
            "reason": "global K=V 层在锁定版本里没有独立 v_proj，q/o 命名不同",
            "evidence": "module-inventory.json#layer-59",
        }
        plan.validate()

    def test_unexplained_fused_names_are_rejected(self) -> None:
        plan = ModulePlan.from_names(
            ["model.language_model.layers.0.self_attn.qkv_proj",
             "model.language_model.layers.0.self_attn.o_proj"]
        )
        with self.assertRaises(PolicyViolation):
            plan.validate()

    def test_frozen_patterns(self) -> None:
        self.assertTrue(is_frozen("model.vision_tower.blocks.0.attn.q_proj"))
        self.assertTrue(is_frozen("lm_head"))
        self.assertTrue(is_frozen("model.language_model.embed_tokens"))
        self.assertTrue(is_frozen("model.language_model.layers.0.input_layernorm"))
        self.assertFalse(is_frozen("model.language_model.layers.0.self_attn.q_proj"))

    def test_memory_plan_gates(self) -> None:
        self.assertEqual(MemoryPlan(peak_gpu_gib=70.0, host_rss_gib=11.0).evaluate()["status"], "pass")
        retry = MemoryPlan(peak_gpu_gib=74.0, host_rss_gib=11.0).evaluate()
        self.assertEqual(retry["status"], "retry_seq1024")
        stopped = MemoryPlan(peak_gpu_gib=74.0, host_rss_gib=11.0, seq_len=1024, downgrade_used=True).evaluate()
        self.assertEqual(stopped["status"], "stop")
        rss = MemoryPlan(peak_gpu_gib=10.0, host_rss_gib=13.0, seq_len=1024, downgrade_used=True).evaluate()
        self.assertEqual(rss["status"], "stop")
        available = MemoryPlan(peak_gpu_gib=10.0, host_rss_gib=1.0, available_gpu_gib=60.0)
        self.assertEqual(available.effective_gpu_limit(), 56.0)

    def test_gpu_auto_start_is_forbidden(self) -> None:
        with self.assertRaises(PolicyViolation):
            assert_no_auto_gpu({})
        with self.assertRaises(PolicyViolation):
            assert_no_auto_gpu({"allow_gpu": True, "gates_passed": False, "budget_gpu_hours": 6})
        with self.assertRaises(PolicyViolation):
            assert_no_auto_gpu({"allow_gpu": True, "gates_passed": True, "budget_gpu_hours": 0})
        assert_no_auto_gpu({"allow_gpu": True, "gates_passed": True, "budget_gpu_hours": 6})


class TemplateAndMaskTests(unittest.TestCase):
    def test_shim_renderer_never_claims_official_parity(self) -> None:
        renderer = ShimRenderer()
        self.assertFalse(renderer.official_template_verified)
        with self.assertRaises(UnverifiedLock):
            renderer.verify_template_bytes(b"whatever")

    def test_official_renderer_rejects_wrong_template_sha(self) -> None:
        with self.assertRaises(IntegrityError):
            OfficialTemplateRenderer(object(), "f" * 64)

    def test_official_renderer_requires_explicit_download_authorization(self) -> None:
        with self.assertRaises(Blocked):
            OfficialTemplateRenderer.from_pretrained("google/gemma-4-31B-it-qat-w4a16-ct", "0" * 40)

    def test_official_renderer_verifies_template_bytes(self) -> None:
        renderer = object.__new__(OfficialTemplateRenderer)
        renderer.official_template_verified = True
        with self.assertRaises(IntegrityError):
            renderer.verify_template_bytes(b"not the canonical template")

    def test_system_and_tool_tokens_never_get_labels(self) -> None:
        messages = [
            {"role": "system", "content": "sys prompt here"},
            {"role": "user", "content": "user question"},
            {"role": "assistant", "content": "call", "tool_name": "run_command", "arguments": {"command": "ls"}, "loss_eligible": True, "step": 1, "phase": "localize"},
            {"role": "tool", "content": "tool output", "tool_name": "run_command", "step": 2, "phase": "localize"},
            {"role": "assistant", "content": "final", "tool_name": "submit_patch", "arguments": {}, "loss_eligible": True, "step": 3, "phase": "finish"},
        ]
        rendered, labels, stats = render_and_mask(ShimRenderer(), messages, pad_to=128)
        self.assertEqual(stats["system_loss_tokens"], 0)
        self.assertEqual(stats["tool_loss_tokens"], 0)
        self.assertEqual(stats["user_loss_tokens"], 0)
        self.assertGreater(stats["target_loss_tokens"], 0)
        self.assertEqual(len(labels.input_ids), 128)
        self.assertEqual(stats["padding"], labels.padding)
        self.assertGreater(stats["padding"], 0)
        self.assertEqual(labels.labels[-1], -100)

    def test_unsupervised_assistant_action_is_context_only(self) -> None:
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "bad", "tool_name": "read_file", "arguments": {"filepath": "x"}, "loss_eligible": False, "step": 1, "phase": "recover"},
            {"role": "assistant", "content": "good", "tool_name": "run_command", "arguments": {"command": "ls"}, "loss_eligible": True, "step": 2, "phase": "recover"},
        ]
        rendered, labels, stats = render_and_mask(ShimRenderer(), messages)
        self.assertEqual(stats["bad_action_loss_tokens"], 0)
        bad_span = [s for s in rendered.spans if s.step == 1][0]
        self.assertTrue(all(value == -100 for value in labels.labels[bad_span.start:bad_span.end]))

    def test_no_supervised_action_is_rejected(self) -> None:
        messages = [
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "obs", "tool_name": "run_command", "step": 1, "phase": "localize"},
        ]
        with self.assertRaises(PolicyViolation):
            render_and_mask(ShimRenderer(), messages)

    def test_twenty_fixtures_all_pass(self) -> None:
        report = run_all()
        self.assertEqual(report["fixture_count"], 20)
        self.assertTrue(report["all_passed"], [r for r in report["results"] if r["status"] != "pass"])
        self.assertIn("serving_render_parity", report["unverified_claims"])
        self.assertFalse(report["results"][0]["stats"]["official_template_verified"])

    def test_fixture_set_covers_required_topics(self) -> None:
        tags = {tag for fixture in build_fixtures() for tag in fixture.tags}
        for required in (
            "empty_search",
            "multi_tool_call",
            "recovery",
            "thinking",
            "marker_injection",
            "unicode",
            "args",
            "bos",
            "padding",
            "context",
            "truncation",
            "windowing",
            "parity",
            "patch_verify",
            "fail_closed",
            "tool_masking",
        ):
            self.assertIn(required, tags)

    def test_build_labels_pads_with_ignore_index(self) -> None:
        rendered = ShimRenderer().render([{"role": "user", "content": "x"}])
        labels = build_labels(rendered, pad_to=rendered.length + 5)
        self.assertEqual(labels.padding, 5)
        self.assertEqual(labels.labels[-5:], [-100] * 5)


class CheckpointTests(unittest.TestCase):
    def _save(self, store: CheckpointStore, step: int = 4, adapter: bytes = b"adapter-bytes"):
        return store.save(
            global_step=step,
            adapter_bytes=adapter,
            optimizer={"exp_avg": [0.1, 0.2], "step": step},
            rng=rng_state(),
            cursor=cursor_state(),
            hashes=HASHES,
            consumed_input_tokens=1024,
            consumed_supervised_tokens=256,
            accum_boundary=0,
        )

    def test_save_load_and_resume_roundtrip(self) -> None:
        with temp_dir("ckpt_") as root:
            store = CheckpointStore(root)
            info = self._save(store)
            self.assertTrue(os.path.exists(os.path.join(info["checkpoint_dir"], "COMPLETE")))
            state = store.load(info["checkpoint_dir"])
            self.assertEqual(state.global_step, 4)
            self.assertEqual(state.consumed_input_tokens, 1024)
            resumed = resume(store, info["checkpoint_dir"], HASHES)
            self.assertEqual(resumed.hashes, HASHES)

    def test_resume_rejects_hash_mismatch(self) -> None:
        with temp_dir("ckpt_") as root:
            store = CheckpointStore(root)
            info = self._save(store)
            changed = dict(HASHES, data_sha256="f" * 64)
            with self.assertRaises(IntegrityError):
                resume(store, info["checkpoint_dir"], changed)

    def test_missing_complete_marker_is_not_usable(self) -> None:
        with temp_dir("ckpt_") as root:
            store = CheckpointStore(root)
            info = self._save(store)
            os.remove(os.path.join(info["checkpoint_dir"], "COMPLETE"))
            with self.assertRaises(IntegrityError):
                store.load(info["checkpoint_dir"])

    def test_full_model_payload_is_rejected(self) -> None:
        with temp_dir("ckpt_") as root:
            store = CheckpointStore(root)
            with self.assertRaises(PolicyViolation):
                store.save(
                    global_step=1,
                    adapter_bytes=b"x",
                    optimizer={"base_model_state_dict": {"w": [1.0]}},
                    rng=rng_state(),
                    cursor=cursor_state(),
                    hashes=HASHES,
                    consumed_input_tokens=1,
                    consumed_supervised_tokens=1,
                    accum_boundary=0,
                )

    def test_partial_rng_or_cursor_is_rejected(self) -> None:
        with temp_dir("ckpt_") as root:
            store = CheckpointStore(root)
            with self.assertRaises(MissingInput):
                store.save(
                    global_step=1,
                    adapter_bytes=b"x",
                    optimizer={},
                    rng={"python_state": "a"},
                    cursor=cursor_state(),
                    hashes=HASHES,
                    consumed_input_tokens=1,
                    consumed_supervised_tokens=1,
                    accum_boundary=0,
                )
            with self.assertRaises(MissingInput):
                store.save(
                    global_step=2,
                    adapter_bytes=b"x",
                    optimizer={},
                    rng=rng_state(),
                    cursor={"position": 3},
                    hashes=HASHES,
                    consumed_input_tokens=1,
                    consumed_supervised_tokens=1,
                    accum_boundary=0,
                )

    def test_mid_accum_interruption_rolls_back_boundary(self) -> None:
        with temp_dir("ckpt_") as root:
            store = CheckpointStore(root)
            info = self._save(store)
            state = resume(store, info["checkpoint_dir"], HASHES, interrupted_mid_accum=True)
            self.assertEqual(state.accum_boundary, 0)
            self.assertTrue(any("accum" in note for note in state.notes))

    def test_prune_keeps_only_last_two(self) -> None:
        with temp_dir("ckpt_") as root:
            store = CheckpointStore(root)
            for step in (1, 2, 3):
                self._save(store, step=step)
            remaining = sorted(
                name for name in os.listdir(root) if name.startswith("step_")
            )
            self.assertEqual(len(remaining), 2)
            self.assertNotIn("step_000001", remaining)

    def test_export_manifest_requires_module_count_and_mapper_revision(self) -> None:
        with self.assertRaises(MissingInput):
            build_export_manifest(
                adapter_sha256="a" * 64,
                rank=16,
                target_modules_regex="^x$",
                matched_module_count=120,
                base_repo_id="google/gemma-4-31B-it-qat-w4a16-ct",
                base_revision="0" * 40,
                lora_alpha=32,
                lora_dropout=0.05,
                training_hashes=HASHES,
            )
        manifest = build_export_manifest(
            adapter_sha256="a" * 64,
            rank=16,
            target_modules_regex="^x$",
            matched_module_count=120,
            base_repo_id="google/gemma-4-31B-it-qat-w4a16-ct",
            base_revision="0" * 40,
            lora_alpha=32,
            lora_dropout=0.05,
            training_hashes=HASHES,
            vllm_mapper_revision="0.19.1",
        )
        self.assertEqual(manifest["lora"]["matched_module_count"], 120)
        with self.assertRaises(PolicyViolation):
            build_export_manifest(
                adapter_sha256="a" * 64,
                rank=16,
                target_modules_regex="^x$",
                matched_module_count=0,
                base_repo_id="r",
                base_revision="0" * 40,
                lora_alpha=32,
                lora_dropout=0.05,
                training_hashes=HASHES,
                vllm_mapper_revision="0.19.1",
            )

    def test_invalid_adapter_cannot_pass_silently(self) -> None:
        manifest = {"lora": {"matched_module_count": 120}}
        assert_adapter_valid_lenient_for_tests(manifest, load_ok=True, params_changed=True, fixture_pass=8, fixture_total=8)
        with self.assertRaises(PolicyViolation):
            assert_adapter_valid_lenient_for_tests(manifest, load_ok=True, params_changed=False, fixture_pass=8, fixture_total=8)
        with self.assertRaises(PolicyViolation):
            assert_adapter_valid_lenient_for_tests(manifest, load_ok=True, params_changed=True, fixture_pass=3, fixture_total=8)
        with self.assertRaises(PolicyViolation):
            assert_adapter_valid_lenient_for_tests({"lora": {"matched_module_count": 0}}, load_ok=True, params_changed=True, fixture_pass=8, fixture_total=8)


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid_and_pinned(self) -> None:
        config = TrainingConfig.default()
        config.assert_valid()
        self.assertEqual(config.seq_len, 2048)
        self.assertEqual(config.lora["r"], 16)
        self.assertEqual(config.budget["max_optimizer_updates"], 32)

    def test_tampered_configs_are_rejected(self) -> None:
        for mutate in (
            lambda payload: payload["batching"].update({"packing": True}),
            lambda payload: payload["batching"].update({"seq_len": 8192}),
            lambda payload: payload["lora"].update({"r": 8}),
            lambda payload: payload["lora"].update({"use_dora": True}),
            lambda payload: payload["optim"].update({"lr": 1e-3}),
            lambda payload: payload["runtime"].update({"cpu_offload": True}),
            lambda payload: payload["budget"].update({"max_optimizer_updates": 64}),
        ):
            payload = TrainingConfig.default().to_dict()
            mutate(payload)
            with self.assertRaises(PolicyViolation):
                TrainingConfig(payload).assert_valid()

    def test_start_gate_blocks_on_every_missing_precondition(self) -> None:
        config = TrainingConfig.default()
        base = {
            "config": config,
            "train_lock_summary": {"verified": False},
            "gates": {"t1_engineering_pass": False, "memory_plan_pass": False, "export_manifest_pass": False},
            "authorization": {"gpu_hours_released": 0, "operator": None, "stage": None, "v2_conflict_checked": False},
        }
        with self.assertRaises(PolicyViolation) as ctx:
            assert_start_allowed(**base)
        problems = ctx.exception.context["problems"]
        self.assertTrue(any("verified" in p for p in problems))
        self.assertTrue(any("t1_engineering_pass" in p for p in problems))
        self.assertTrue(any("GPU" in p for p in problems))
        self.assertTrue(any("operator" in p or "操作者" in p for p in problems))

        allowed = dict(base)
        allowed["train_lock_summary"] = {"verified": True}
        allowed["gates"] = {"t1_engineering_pass": True, "memory_plan_pass": True, "export_manifest_pass": True}
        allowed["authorization"] = {
            "gpu_hours_released": 6,
            "operator": "Liang",
            "stage": "T1",
            "v2_conflict_checked": True,
        }
        self.assertTrue(assert_start_allowed(**allowed)["allowed"])

    def test_config_payload_roundtrip_is_canonical(self) -> None:
        config = TrainingConfig.default()
        self.assertEqual(config.config_sha256(), config.config_sha256())
        self.assertEqual(len(config.config_sha256()), 64)
        self.assertEqual(DEFAULT_CONFIG["seed"], 17)


class EntryTests(unittest.TestCase):
    def test_preflight_reports_blockers_and_never_claims_ready(self) -> None:
        report = preflight()
        self.assertFalse(report["can_start_training_now"])
        codes = {item["code"] for item in report["blockers"]}
        self.assertIn("train_lock_unverified", codes)
        self.assertIn("serving_lock_unverified", codes)
        self.assertIn("interface_pins_unverified", codes)
        self.assertEqual(report["mask_fixtures"]["passed"], 20)
        self.assertEqual(report["target_plan"]["expected_modules"], 120)
        self.assertIn("实际训练收益（未跑真实 GPU）", report["unverified_claims"])

    def test_start_is_blocked_without_authorization(self) -> None:
        with self.assertRaises(PolicyViolation):
            start(operator=None, gpu_hours=0.0, stage=None)

    def test_start_is_blocked_even_with_authorization_but_missing_gates(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            start(operator="Liang", gpu_hours=6.0, stage="T1", v2_conflict_checked=True)
        problems = ctx.exception.context["problems"]
        self.assertTrue(any("t1_engineering_pass" in item or "verified" in item for item in problems))

    def test_cli_preflight_exit_code(self) -> None:
        self.assertEqual(entry_main(["--preflight"]), 5)
        self.assertNotEqual(entry_main(["--start"]), 0)

    def test_cli_start_exit_code_is_fail_closed(self) -> None:
        self.assertEqual(entry_main(["--start", "--operator", "Liang", "--gpu-hours", "6"]), 7)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

