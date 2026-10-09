"""Q0 独立审查应修缺陷 Y4 / Y5 / Y9 / Y11 的回归测试（仅标准库，纯 CPU）。

每个测试都对应 Q0 报告里一条具体结论：

- Y4：默认 fixture 入口的 `min_pass=7` 魔数、且不校验 `fixture_pass <= fixture_total`；
  该默认入口现名 `assert_adapter_valid_lenient_for_tests`（🟡-4：生产路径禁止调用）；
- Y5：`resume(interrupted_mid_accum=True)` 只加注解、不回滚 optimizer/global_step/游标；
- Y9：`deps.load_pair` / `MemoryPlan.evaluate` / `assert_no_full_state_dict` 三处未接线、
  且 `assert_no_full_state_dict` 的方向是"装得下才抛"；
- Y11：`TrainingConfig.from_file` 浅合并导致部分配置抛裸 `KeyError`（非 FailClosed）。

运行：`python run_tests.py test_q0_ckpt_config`
"""

from __future__ import annotations

import json
import os
import unittest

from tests._tmp import temp_dir
from v3.common.errors import FailClosed, MissingInput, PolicyViolation
from v3.t0.deps import DependencyLock, assert_distinct_lock_channels, load_pair
from v3.train import fixtures as fixture_module
from v3.train.checkpoint import (
    REQUIRED_FIXTURE_PASS,
    REQUIRED_FIXTURE_TOTAL,
    REQUIRED_HASH_KEYS,
    CheckpointStore,
    assert_adapter_valid_frozen_spec,
    assert_adapter_valid_lenient_for_tests,
    resume,
)
from v3.train.config import TrainingConfig
from v3.train.streaming import StreamingPlan
from v3.train.targets import MemoryPlan

GiB = 1 << 30

HASHES = {key: "a" * 64 for key in REQUIRED_HASH_KEYS}

ADAPTER_EXPORT_MANIFEST = {"lora": {"matched_module_count": 120}}


# ---------------------------------------------------------------- 公共小工具


def _rng_state() -> dict:
    return {"python_state": "p", "numpy_state": "n", "torch_state": "t"}


def _cursor_state(position: int) -> dict:
    return {"sampler_order_sha256": "s" * 64, "position": position}


def _save(
    store: CheckpointStore,
    step: int,
    *,
    accum_boundary: int = 0,
    position: int = 42,
    optimizer: dict | None = None,
) -> dict:
    """写一份完整 checkpoint（哈希/RNG/游标都齐）。"""
    return store.save(
        global_step=step,
        adapter_bytes=b"ADAPTER",
        optimizer=optimizer if optimizer is not None else {"exp_avg": [1.0, 2.0]},
        rng=_rng_state(),
        cursor=_cursor_state(position),
        hashes=HASHES,
        consumed_input_tokens=step * 10,
        consumed_supervised_tokens=step,
        accum_boundary=accum_boundary,
    )


def _write_json(path: str, payload) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path


def _lock_payload(channel: str, packages: dict) -> dict:
    return {
        "channel": channel,
        "python": "3.11.9",
        "verified": False,
        "notes": [],
        "packages": packages,
    }


# ---------------------------------------------------------------- Y4


class Y4AdapterFixtureCountTests(unittest.TestCase):
    """Y4：魔数 min_pass 删除 + fixture 计数自洽 + 20/20 冻结口径。"""

    def test_constants_match_frozen_fixture_suite(self) -> None:
        """具名常量必须与仓库里真实的 20 项 mask fixture 集一致（出处可核）。"""
        self.assertEqual(REQUIRED_FIXTURE_TOTAL, 20)
        self.assertEqual(REQUIRED_FIXTURE_PASS, 20)
        self.assertEqual(len(fixture_module.build_fixtures()), REQUIRED_FIXTURE_TOTAL)

    def test_pass_exceeding_total_is_rejected(self) -> None:
        """Y4 反例：fixture_pass=7 / fixture_total=2 必须被拒（原来会静默放行）。"""
        with self.assertRaises(FailClosed) as ctx:
            assert_adapter_valid_lenient_for_tests(
                ADAPTER_EXPORT_MANIFEST,
                load_ok=True,
                params_changed=True,
                fixture_pass=7,
                fixture_total=2,
            )
        self.assertEqual(ctx.exception.code, "fixture_count_inconsistent")

    def test_frozen_20_must_all_pass(self) -> None:
        """Y4：7/20 不达冻结验收（20 项 mask fixture 必须全过）。"""
        with self.assertRaises(FailClosed) as ctx:
            assert_adapter_valid_lenient_for_tests(
                ADAPTER_EXPORT_MANIFEST,
                load_ok=True,
                params_changed=True,
                fixture_pass=7,
                fixture_total=20,
            )
        self.assertEqual(ctx.exception.code, "adapter_fixture_regression")

    def test_frozen_20_of_20_passes(self) -> None:
        result = assert_adapter_valid_lenient_for_tests(
            ADAPTER_EXPORT_MANIFEST,
            load_ok=True,
            params_changed=True,
            fixture_pass=20,
            fixture_total=20,
        )
        self.assertTrue(result["adapter_valid"])
        self.assertTrue(result["frozen_spec_satisfied"])
        self.assertEqual(result["required_fixture_pass"], REQUIRED_FIXTURE_PASS)

    def test_strict_entry_rejects_smaller_suite(self) -> None:
        """生产入口：fixture_total != 20 直接阻断（不给"小集全绿"留口子）。"""
        with self.assertRaises(FailClosed) as ctx:
            assert_adapter_valid_frozen_spec(
                ADAPTER_EXPORT_MANIFEST,
                load_ok=True,
                params_changed=True,
                fixture_pass=8,
                fixture_total=8,
            )
        self.assertEqual(ctx.exception.code, "adapter_fixture_suite_incomplete")

    def test_default_entry_marks_small_suite_as_not_frozen(self) -> None:
        """向后兼容入口：8/8 这类非冻结小集可以过，但必须自报未满足冻结口径。"""
        result = assert_adapter_valid_lenient_for_tests(
            ADAPTER_EXPORT_MANIFEST,
            load_ok=True,
            params_changed=True,
            fixture_pass=8,
            fixture_total=8,
        )
        self.assertFalse(result["frozen_spec_satisfied"])

    def test_oversized_suite_is_rejected(self) -> None:
        """不得用自定义大集合冒充验收证据。"""
        with self.assertRaises(FailClosed) as ctx:
            assert_adapter_valid_lenient_for_tests(
                ADAPTER_EXPORT_MANIFEST,
                load_ok=True,
                params_changed=True,
                fixture_pass=21,
                fixture_total=21,
            )
        self.assertEqual(ctx.exception.code, "adapter_fixture_suite_oversized")

    def test_negative_counts_and_bad_types_are_fail_closed(self) -> None:
        with self.assertRaises(FailClosed) as ctx:
            assert_adapter_valid_lenient_for_tests(
                ADAPTER_EXPORT_MANIFEST,
                load_ok=True,
                params_changed=True,
                fixture_pass=-1,
                fixture_total=20,
            )
        self.assertEqual(ctx.exception.code, "fixture_count_negative")
        with self.assertRaises(MissingInput):
            assert_adapter_valid_lenient_for_tests(
                ADAPTER_EXPORT_MANIFEST,
                load_ok=True,
                params_changed=True,
                fixture_pass="20",
                fixture_total=20,
            )


# ---------------------------------------------------------------- Y5


class Y5MidAccumRollbackTests(unittest.TestCase):
    """Y5：accum 中途中断必须**真的**回滚到上一份完整 optimizer 步。"""

    def test_save_records_previous_complete_step(self) -> None:
        with temp_dir("q0ckpt_") as root:
            store = CheckpointStore(root)
            first = _save(store, step=10)
            second = _save(store, step=20)
            self.assertIsNone(first["prev_complete_step"])
            self.assertEqual(second["prev_complete_step"], 10)
            self.assertEqual(second["prev_complete_dir"], first["checkpoint_dir"])

    def test_rollback_switches_every_stateful_field(self) -> None:
        with temp_dir("q0ckpt_") as root:
            store = CheckpointStore(root)
            first = _save(
                store, step=10, accum_boundary=5, position=42, optimizer={"exp_avg": [1.0, 2.0]}
            )
            second = _save(
                store, step=20, accum_boundary=7, position=99, optimizer={"exp_avg": [9.0]}
            )
            state = resume(store, second["checkpoint_dir"], HASHES, interrupted_mid_accum=True)

            self.assertTrue(state.rollback_applied)
            self.assertEqual(state.rolled_back_from, second["checkpoint_dir"])
            self.assertEqual(state.rolled_back_to, first["checkpoint_dir"])
            # 四个有状态字段都必须取自快照，而不是"只把 accum_boundary 置 0"。
            self.assertEqual(state.global_step, 10)
            self.assertEqual(state.accum_boundary, 5)
            self.assertEqual(state.cursor["position"], 42)
            self.assertEqual(state.optimizer, {"exp_avg": [1.0, 2.0]})
            self.assertEqual(state.consumed_input_tokens, 100)
            self.assertTrue(any("回滚" in note for note in state.notes))

            payload = state.to_dict()
            self.assertIs(payload["rollback_applied"], True)
            self.assertEqual(payload["rolled_back_to"], first["checkpoint_dir"])

    def test_without_snapshot_rollback_is_not_faked(self) -> None:
        """只有一份 checkpoint 时不得假装回滚：rollback_applied=False 且不改字段。"""
        with temp_dir("q0ckpt_") as root:
            store = CheckpointStore(root)
            only = _save(store, step=10, accum_boundary=3, position=7)
            state = resume(store, only["checkpoint_dir"], HASHES, interrupted_mid_accum=True)

            self.assertFalse(state.rollback_applied)
            self.assertIsNone(state.rolled_back_from)
            self.assertIsNone(state.rolled_back_to)
            self.assertEqual(state.to_dict()["rollback_applied"], False)
            self.assertTrue(any("未执行回滚" in note for note in state.notes))
            self.assertEqual(state.global_step, 10)
            self.assertEqual(state.accum_boundary, 3)  # 不假装回滚成 0
            self.assertEqual(state.cursor["position"], 7)

    def test_plain_resume_never_rolls_back(self) -> None:
        with temp_dir("q0ckpt_") as root:
            store = CheckpointStore(root)
            _save(store, step=10)
            second = _save(store, step=20, position=99)
            state = resume(store, second["checkpoint_dir"], HASHES)
            self.assertFalse(state.rollback_applied)
            self.assertEqual(state.global_step, 20)
            self.assertEqual(state.cursor["position"], 99)


# ---------------------------------------------------------------- Y9-1 依赖锁隔离


class Y9LockChannelIsolationTests(unittest.TestCase):
    """Y9(1)：训练锁与推理锁的隔离判定必须可被生产调用并返回结构化结果。"""

    def test_distinct_locks_pass_with_structured_result(self) -> None:
        with temp_dir("q0locks_") as root:
            train_path = _write_json(
                os.path.join(root, "train.lock.json"),
                _lock_payload("train", {"torch": {"version": "2.10.0"}}),
            )
            serving_path = _write_json(
                os.path.join(root, "serving.lock.json"),
                _lock_payload("serving", {"vllm": {"version": "0.19.1"}}),
            )
            train, serving = load_pair(train_path, serving_path)
            result = assert_distinct_lock_channels(train, serving)

            self.assertTrue(result["distinct"])
            self.assertEqual(len(result["train_lock_sha256"]), 64)
            self.assertEqual(len(result["serving_lock_sha256"]), 64)
            self.assertEqual(len(result["train_source_sha256"]), 64)
            self.assertEqual(result["train_channel"], "train")
            self.assertEqual(result["serving_channel"], "serving")
            self.assertEqual(result["package_overlap"], [])
            self.assertFalse(result["package_versions_identical"])

    def test_same_file_is_aliased(self) -> None:
        with temp_dir("q0locks_") as root:
            path = _write_json(
                os.path.join(root, "train.lock.json"),
                _lock_payload("train", {"torch": {"version": "2.10.0"}}),
            )
            train = DependencyLock.from_file(path)
            with self.assertRaises(PolicyViolation) as ctx:
                assert_distinct_lock_channels(train, train)
            self.assertEqual(ctx.exception.code, "training_serving_lock_aliased")
            # 唯一调用 assert_channels_isolated 的生产入口也要拦得住同一份锁。
            with self.assertRaises(FailClosed):
                load_pair(path, path)

    def test_same_bytes_in_two_paths_are_aliased(self) -> None:
        with temp_dir("q0locks_") as root:
            payload = _lock_payload("train", {"torch": {"version": "2.10.0"}})
            left = DependencyLock.from_file(
                _write_json(os.path.join(root, "a.lock.json"), payload)
            )
            right = DependencyLock.from_file(
                _write_json(os.path.join(root, "b.lock.json"), payload)
            )
            with self.assertRaises(FailClosed) as ctx:
                assert_distinct_lock_channels(left, right)
            self.assertEqual(ctx.exception.code, "training_serving_lock_aliased")

    def test_identical_package_versions_are_aliased(self) -> None:
        """包名→版本映射完全一致 = 两侧没有分面锁定。"""
        with temp_dir("q0locks_") as root:
            packages = {"torch": {"version": "2.10.0"}}
            train = DependencyLock.from_file(
                _write_json(
                    os.path.join(root, "train.lock.json"), _lock_payload("train", packages)
                )
            )
            serving = DependencyLock.from_file(
                _write_json(
                    os.path.join(root, "serving.lock.json"), _lock_payload("serving", packages)
                )
            )
            with self.assertRaises(FailClosed) as ctx:
                assert_distinct_lock_channels(train, serving)
            self.assertEqual(ctx.exception.code, "training_serving_lock_aliased")

    def test_channel_mismatch_is_aliased(self) -> None:
        with temp_dir("q0locks_") as root:
            train = DependencyLock.from_file(
                _write_json(
                    os.path.join(root, "train.lock.json"),
                    _lock_payload("train", {"torch": {"version": "2.10.0"}}),
                )
            )
            serving = DependencyLock.from_file(
                _write_json(
                    os.path.join(root, "serving.lock.json"),
                    _lock_payload("train", {"vllm": {"version": "0.19.1"}}),
                )
            )
            with self.assertRaises(FailClosed):
                assert_distinct_lock_channels(train, serving)


# ---------------------------------------------------------------- Y9-2 内存门槛


class Y9MemoryPlanTests(unittest.TestCase):
    """Y9(2)：没有实测就不能当作门槛通过。"""

    def test_evaluate_or_block_requires_measurement(self) -> None:
        plan = MemoryPlan(peak_gpu_gib=70.0, host_rss_gib=11.0)
        with self.assertRaises(MissingInput) as ctx:
            plan.evaluate_or_block(None)
        self.assertEqual(ctx.exception.code, "memory_plan_unmeasured")

    def test_incomplete_measurement_is_rejected(self) -> None:
        plan = MemoryPlan(peak_gpu_gib=70.0, host_rss_gib=11.0)
        with self.assertRaises(MissingInput) as ctx:
            plan.evaluate_or_block({"peak_gpu_gib": 70.0})
        self.assertEqual(ctx.exception.code, "memory_plan_measured_incomplete")
        with self.assertRaises(MissingInput) as ctx2:
            plan.evaluate_or_block({"peak_gpu_gib": "很多", "host_rss_gib": 11.0})
        self.assertEqual(ctx2.exception.code, "memory_plan_measured_invalid")

    def test_evaluate_or_block_uses_measured_values(self) -> None:
        # 自报估计很小（1GiB），实测却是 70GiB：必须按实测判，不能按估计值放行。
        plan = MemoryPlan(peak_gpu_gib=1.0, host_rss_gib=1.0)
        report = plan.evaluate_or_block({"peak_gpu_gib": 70.0, "host_rss_gib": 11.0})
        self.assertTrue(report["measured"])
        self.assertTrue(report["pass"])
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["measured_source"], "measured")
        self.assertEqual(report["measured_peak_gpu_gib"], 70.0)

    def test_measured_gate_failure_is_policy_violation(self) -> None:
        plan = MemoryPlan(peak_gpu_gib=1.0, host_rss_gib=1.0, seq_len=1024, downgrade_used=True)
        with self.assertRaises(PolicyViolation) as ctx:
            plan.evaluate_or_block({"peak_gpu_gib": 90.0, "host_rss_gib": 1.0})
        self.assertEqual(ctx.exception.code, "memory_plan_gate_failed")
        self.assertIn("report", ctx.exception.context)

    def test_declared_estimate_is_not_measured_evidence(self) -> None:
        report = MemoryPlan(peak_gpu_gib=70.0, host_rss_gib=11.0).evaluate()
        self.assertFalse(report["measured"])
        self.assertEqual(report["measured_source"], "declared_estimate")
        self.assertEqual(report["status"], "pass")  # 估计值仍可静态判定，但不是实测证据


# ---------------------------------------------------------------- Y9-3 整体驻留


class Y9FullStateDictTests(unittest.TestCase):
    """Y9(3)：断言必须针对"整体驻留"行为本身，而不是"装得下才抛"。"""

    def _plan(self, *shards: int) -> StreamingPlan:
        return StreamingPlan.from_manifest(
            {"shards": [{"name": "s%d" % i, "bytes": size} for i, size in enumerate(shards)]}
        )

    def test_resident_full_state_dict_is_blocked(self) -> None:
        plan = self._plan(1 * GiB)
        with self.assertRaises(PolicyViolation) as ctx:
            plan.assert_no_full_state_dict(peak_full_state_dict_bytes=1 * GiB, allowed_bytes=0)
        self.assertEqual(ctx.exception.code, "full_state_dict_resident")

    def test_no_evidence_is_fail_closed(self) -> None:
        """不提供证据（旧调用形态）不得视为通过。"""
        plan = self._plan(1 * GiB)
        with self.assertRaises(PolicyViolation) as ctx:
            plan.assert_no_full_state_dict()
        self.assertEqual(ctx.exception.code, "full_state_dict_resident_unevidenced")

    def test_zero_residency_passes(self) -> None:
        plan = self._plan(1 * GiB, 2 * GiB)
        result = plan.assert_no_full_state_dict(
            peak_full_state_dict_bytes=0, allowed_bytes=0, evidence={"probe": "shard_loader"}
        )
        self.assertTrue(result["checked"])
        self.assertFalse(result["full_state_dict_resident"])
        self.assertEqual(result["evidence"], {"probe": "shard_loader"})

    def test_negative_counter_is_rejected(self) -> None:
        plan = self._plan(1 * GiB)
        with self.assertRaises(PolicyViolation) as ctx:
            plan.assert_no_full_state_dict(peak_full_state_dict_bytes=-1, allowed_bytes=0)
        self.assertEqual(ctx.exception.code, "full_state_dict_counter_invalid")


# ---------------------------------------------------------------- Y11 配置合并


class Y11ConfigMergeTests(unittest.TestCase):
    """Y11：部分配置必须能加载（深合并），且未知键 fail-closed。"""

    def test_partial_config_never_raises_bare_exception(self) -> None:
        """反例原文：只写 {"lora": {"r": 16}} 时原来抛裸 KeyError: 'lora_alpha'。"""
        with temp_dir("q0cfg_") as root:
            path = _write_json(os.path.join(root, "partial.json"), {"lora": {"r": 16}})
            try:
                config = TrainingConfig.from_file(path)
                problems = config.validate()
            except FailClosed as exc:
                self.fail("部分配置不应 fail-closed：%s" % exc.code)
            except Exception as exc:  # noqa: BLE001 - 这里就是要抓住任何裸异常
                self.fail("抛了非 FailClosed 异常：%s: %s" % (type(exc).__name__, exc))
            self.assertIsInstance(problems, list)
            self.assertEqual(problems, [])
            self.assertEqual(config.lora["r"], 16)
            self.assertEqual(config.lora["lora_alpha"], 32)  # 兄弟键保留默认值
            self.assertEqual(config.seq_len, 2048)
            self.assertEqual(config.budget["max_optimizer_updates"], 32)

    def test_partial_section_keeps_siblings(self) -> None:
        with temp_dir("q0cfg_") as root:
            path = _write_json(os.path.join(root, "batch.json"), {"batching": {"seq_len": 1024}})
            config = TrainingConfig.from_file(path)
            self.assertEqual(config.payload["batching"]["seq_len"], 1024)
            self.assertFalse(config.payload["batching"]["packing"])
            self.assertEqual(config.payload["batching"]["gradient_accumulation_steps"], 8)
            self.assertEqual(config.validate(), [])

    def test_unknown_keys_are_rejected(self) -> None:
        with temp_dir("q0cfg_") as root:
            for index, payload in enumerate(
                (
                    {"nope": 1},
                    {"lora": {"r": 16, "rank": 8}},
                    {"optim": {"unknown_flag": True}},
                )
            ):
                path = _write_json(os.path.join(root, "unknown_%d.json" % index), payload)
                with self.assertRaises(PolicyViolation) as ctx:
                    TrainingConfig.from_file(path)
                self.assertEqual(ctx.exception.code, "unknown_config_key")

    def test_non_object_config_is_rejected(self) -> None:
        with temp_dir("q0cfg_") as root:
            path = _write_json(os.path.join(root, "list.json"), [1, 2, 3])
            with self.assertRaises(MissingInput):
                TrainingConfig.from_file(path)

    def test_missing_fields_and_bad_types_are_fail_closed(self) -> None:
        """缺字段 → MissingInput；字段类型错 → 记入 problems，绝不抛裸 TypeError。"""
        with self.assertRaises(MissingInput):
            TrainingConfig({"lora": "not-a-dict"}).validate()

        payload = TrainingConfig.default().to_dict()
        payload["lora"]["lora_dropout"] = "不是数值"
        payload["budget"]["max_optimizer_updates"] = {"nested": 1}
        problems = TrainingConfig(payload).validate()
        self.assertTrue(any("lora_dropout" in item for item in problems), problems)
        self.assertTrue(any("max_optimizer_updates" in item for item in problems), problems)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

