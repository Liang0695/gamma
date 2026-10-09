"""KAGGLE-38 增量回归：G13（--dest-dir）/ G8（导出契约）/ G7b（内存测量）/ G14（保存恢复）。

全部在 CPU 上跑、不联网、不加载模型、不训练真实模型。
唯一会碰到真实 GPU 的用例（`RealCudaProbeTests`）默认**跳过**，
只在显式设置环境变量 `V3_RUN_CUDA_PROBE=1` 时才执行，且只做一次 64 MiB 的
"分配 → 加一 → 释放"（约 70 ms），不加载模型、不训练。
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
    FIXTURE_HASHES,
    legacy_self_report_manifest,
    make_export_manifest,
    make_official_export_dir,
)
from tests._tmp import temp_dir_outside_repo  # noqa: E402

from v3.common.errors import FailClosed, IntegrityError, MissingInput, PolicyViolation  # noqa: E402
from v3.train import adapter_export_contract as contract  # noqa: E402
from v3.train import destdir, engineering_check, entry, lifecycle, profile as profile_mod, runner  # noqa: E402
from v3.train.checkpoint import ResumeState  # noqa: E402


# --------------------------------------------------------------------- G13


class DestDirTests(unittest.TestCase):
    """G13：`--dest-dir` 必须显式、绝对、仓库外、可写。"""

    def test_missing_dest_dir_is_rejected(self) -> None:
        for value in (None, "", "   "):
            with self.assertRaises(MissingInput) as ctx:
                destdir.resolve_dest_dir(value)
            self.assertEqual(ctx.exception.code, "dest_dir_missing")

    def test_relative_path_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            destdir.resolve_dest_dir("relative/out")
        self.assertEqual(ctx.exception.code, "dest_dir_not_absolute")

    def test_path_inside_repo_is_rejected(self) -> None:
        inside = os.path.join(destdir.repo_root(), "v3", "train", "_run_adapter")
        with self.assertRaises(PolicyViolation) as ctx:
            destdir.resolve_dest_dir(inside)
        self.assertEqual(ctx.exception.code, "dest_dir_inside_repo")

    def test_repo_ancestor_is_rejected(self) -> None:
        ancestor = os.path.dirname(destdir.repo_root())
        with self.assertRaises(PolicyViolation) as ctx:
            destdir.resolve_dest_dir(ancestor)
        self.assertEqual(ctx.exception.code, "dest_dir_contains_repo")

    def test_existing_file_is_rejected(self) -> None:
        with temp_dir_outside_repo("g13_file_") as root:
            path = os.path.join(root, "not_a_dir")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("x")
            with self.assertRaises(PolicyViolation) as ctx:
                destdir.resolve_dest_dir(path)
            self.assertEqual(ctx.exception.code, "dest_dir_is_file")

    def test_valid_outside_path_is_created_and_probed(self) -> None:
        with temp_dir_outside_repo("g13_ok_") as root:
            target = os.path.join(root, "nested", "runs")
            resolution = destdir.resolve_dest_dir(target)
        self.assertEqual(resolution["resolved"], os.path.realpath(target))
        self.assertTrue(resolution["outside_repo"])
        self.assertFalse(resolution["contains_repo"])
        self.assertIn("writable_probe_ok", resolution["checks"])
        self.assertFalse(
            os.path.exists(os.path.join(target, destdir.PROBE_FILENAME)),
            "探针文件必须被清理",
        )

    def test_paths_consistency(self) -> None:
        with temp_dir_outside_repo("g13_consistency_") as root:
            resolution = destdir.resolve_dest_dir(root)
            ok = destdir.verify_paths_consistent(
                resolution,
                [
                    ("saved", os.path.join(root, "adapter.synthetic.json")),
                    ("evidence", os.path.join(root, "training_evidence.json")),
                ],
            )
            self.assertTrue(ok["consistent"])
            with self.assertRaises(PolicyViolation) as ctx:
                destdir.verify_paths_consistent(
                    resolution, [("elsewhere", os.path.join(os.path.dirname(root), "other.json"))]
                )
            self.assertEqual(ctx.exception.code, "dest_dir_path_inconsistent")

    def test_cli_start_without_dest_dir_never_writes_into_repo(self) -> None:
        """缺 `--dest-dir` 时 CLI 必须非零退出，且**不得**在仓库里建默认产物目录。"""
        legacy_default = os.path.join(destdir.repo_root(), "v3", "train", "_run_adapter")
        existed = os.path.exists(legacy_default)
        code = entry.main(["--start", "--operator", "unit", "--gpu-hours", "1", "--stage", "T1"])
        self.assertNotEqual(code, 0)
        self.assertFalse(
            os.path.exists(legacy_default) and not existed,
            "G13 之前的行为（默认写进仓库）不得回归",
        )


# --------------------------------------------------------------------- G8


class AdapterExportContractTests(unittest.TestCase):
    """G8：一个版本化契约同时管生产与消费，并强制与字节对账。"""

    def test_producer_and_consumer_agree(self) -> None:
        with temp_dir_outside_repo("g8_ok_") as export_dir:
            make_official_export_dir(export_dir)
            manifest = make_export_manifest(export_dir)
            self.assertEqual(manifest["format"], contract.PRODUCTION_FORMAT)
            self.assertEqual(manifest["artifact_digest"], contract.artifact_digest(manifest["files"]))
            for key in ("source_sha256", "data_sha256", "config_sha256", "code_sha256", "deps_sha256"):
                self.assertIn(key, manifest, "哈希键必须扁平在顶层")
            self.assertNotIn("training_hashes", manifest)
            report = contract.validate_export_manifest(manifest, export_dir=export_dir)
            self.assertTrue(report["verified_from_artifacts"])
            self.assertFalse(report["real_adapter_loading_verified"])
            self.assertEqual(report["self_reported_fields_used"], [])

    def test_legacy_v1_format_is_rejected(self) -> None:
        problems = contract.collect_export_manifest_problems(
            {"format": contract.FORMAT_V1, "training_hashes": {}, "files": []}, export_dir=None
        )
        self.assertTrue(any(contract.FORMAT_V1 in item for item in problems), problems)

    def test_self_reported_fields_are_rejected(self) -> None:
        with temp_dir_outside_repo("g8_self_") as export_dir:
            make_official_export_dir(export_dir)
            problems = contract.collect_export_manifest_problems(
                legacy_self_report_manifest(), export_dir=export_dir
            )
        joined = " ".join(problems)
        self.assertIn("自报通过字段", joined)
        self.assertIn("load_ok", joined)

    def test_data_export_is_not_an_adapter_export(self) -> None:
        problems = contract.collect_export_manifest_problems(
            {"windows_sha256": "a" * 64, "window_count": 3, "format": contract.PRODUCTION_FORMAT},
            export_dir=None,
        )
        self.assertTrue(any("P0 数据导出" in item for item in problems), problems)

    def test_artifacts_must_be_bound(self) -> None:
        with temp_dir_outside_repo("g8_unbound_") as export_dir:
            make_official_export_dir(export_dir)
            manifest = make_export_manifest(export_dir)
            problems = contract.collect_export_manifest_problems(manifest, export_dir=None)
        self.assertTrue(any("--export-dir" in item for item in problems), problems)

    def test_tampered_bytes_are_detected(self) -> None:
        with temp_dir_outside_repo("g8_tamper_") as export_dir:
            make_official_export_dir(export_dir)
            manifest = make_export_manifest(export_dir)
            weights = os.path.join(
                export_dir, *contract.adapter_contract.carrier_weights_relative_path("v3_policy").split("/")
            )
            with open(weights, "wb") as handle:
                handle.write(b"TAMPERED")
            problems = contract.collect_export_manifest_problems(manifest, export_dir=export_dir)
        self.assertTrue(any("文件哈希与产物不符" in item for item in problems), problems)

    def test_extra_unlisted_file_is_detected(self) -> None:
        with temp_dir_outside_repo("g8_extra_") as export_dir:
            make_official_export_dir(export_dir)
            manifest = make_export_manifest(export_dir)
            with open(os.path.join(export_dir, "README.txt"), "w", encoding="utf-8") as handle:
                handle.write("surprise")
            problems = contract.collect_export_manifest_problems(manifest, export_dir=export_dir)
        self.assertTrue(any("未登记的文件" in item for item in problems), problems)

    def test_artifact_digest_mismatch_is_detected(self) -> None:
        with temp_dir_outside_repo("g8_digest_") as export_dir:
            make_official_export_dir(export_dir)
            manifest = make_export_manifest(export_dir)
            manifest["artifact_digest"] = "0" * 64
            problems = contract.collect_export_manifest_problems(manifest, export_dir=export_dir)
        self.assertTrue(any("artifact_digest" in item for item in problems), problems)

    def test_legacy_carrier_name_fails_the_producer(self) -> None:
        with temp_dir_outside_repo("g8_legacy_") as export_dir:
            directory = os.path.join(export_dir, "adapters", "v3_policy")
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "adapter.safetensors"), "wb") as handle:
                handle.write(b"legacy")
            with open(os.path.join(directory, "adapter_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"peft_type": "LORA"}, handle)
            with self.assertRaises(FailClosed):
                make_export_manifest(export_dir)

    def test_producer_requires_pins_and_module_count(self) -> None:
        with temp_dir_outside_repo("g8_pins_") as export_dir:
            make_official_export_dir(export_dir)
            with self.assertRaises(MissingInput) as ctx:
                contract.build_export_manifest(
                    export_dir,
                    adapter_name="v3_policy",
                    rank=16,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    target_modules_regex=".*",
                    matched_module_count=8,
                    base_repo_id="google/gemma-4-31B-it-qat-w4a16-ct",
                    base_revision="0" * 40,
                    vllm_mapper_revision=None,
                    training_hashes=dict(FIXTURE_HASHES),
                )
            self.assertEqual(ctx.exception.code, "mapper_revision_unknown")
            with self.assertRaises(PolicyViolation) as ctx2:
                contract.build_export_manifest(
                    export_dir,
                    adapter_name="v3_policy",
                    rank=16,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    target_modules_regex=".*",
                    matched_module_count=0,
                    base_repo_id="google/gemma-4-31B-it-qat-w4a16-ct",
                    base_revision="0" * 40,
                    vllm_mapper_revision="v",
                    training_hashes=dict(FIXTURE_HASHES),
                )
            self.assertEqual(ctx2.exception.code, "export_no_modules")


# --------------------------------------------------------------------- G7b


class FakeGpuReader:
    """测试替身：显式声明**非真实** provenance，用于验证消费者会拒绝它。"""

    provenance = "test-double"
    available = True
    reason = ""

    def reset(self) -> None:  # pragma: no cover - 占位
        pass

    def peaks_gib(self) -> dict:
        return {"peak_gpu_gib": 1.0, "peak_gpu_reserved_gib": 1.2}

    def free_gib(self) -> float:
        return 40.0


class MemoryProfileTests(unittest.TestCase):
    """G7b：内存 profile 必须来自测量，且工作负载真的上卡。"""

    def _plan(self, steps: int = 4):
        return runner.TrainRunPlan(
            steps=steps,
            lr=0.3,
            seq_len=12,
            lora_rank=4,
            lora_alpha=8,
            batches=runner.build_smoke_batches(seq_len=12),
        )

    def test_host_peak_probe_is_real(self) -> None:
        probe = profile_mod.host_peak_rss_bytes()
        self.assertGreater(probe["bytes"], 0)
        self.assertTrue(probe["probe"])
        self.assertIn(":", probe["probe"])

    def test_profile_covers_all_required_phases(self) -> None:
        backend = runner.SyntheticBackend(vocab=32, dim=8)
        measured = profile_mod.measure_memory_profile(backend, self._plan())
        self.assertEqual(measured["missing_phases"], [])
        self.assertGreater(measured["host_rss_gib"], 0.0)
        self.assertEqual(measured["workload"]["uses_gpu"], False)
        # CPU 玩具模型的 profile 不构成显存证据：payload 必须为 None。
        self.assertIsNone(profile_mod.profile_to_memory_plan_payload(measured))
        report = profile_mod.check_profile_provenance(measured)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("没有使用 GPU" in item for item in report["problems"]), report["problems"]
        )

    def test_synthetic_fixture_profile_is_rejected(self) -> None:
        fake = profile_mod.synthetic_fixture_profile(peak_gpu_gib=1.0, host_rss_gib=1.0)
        report = profile_mod.check_profile_provenance(fake)
        self.assertFalse(report["ok"])
        with self.assertRaises(PolicyViolation) as ctx:
            profile_mod.assert_profile_is_real_measurement(fake)
        self.assertEqual(ctx.exception.code, "memory_profile_not_a_measurement")
        self.assertIsNone(profile_mod.profile_to_memory_plan_payload(fake))

    def test_test_double_provenance_is_rejected(self) -> None:
        backend = runner.SyntheticBackend(vocab=32, dim=8)
        measured = profile_mod.measure_memory_profile(
            backend, self._plan(), gpu_reader=FakeGpuReader()
        )
        self.assertFalse(measured["is_real_measurement"])
        self.assertFalse(profile_mod.check_profile_provenance(measured)["ok"])

    def test_gpu_unavailable_is_recorded_not_guessed(self) -> None:
        class NoGpuReader(FakeGpuReader):
            provenance = profile_mod.REAL_GPU_PROVENANCE
            available = False
            reason = "no cuda in test"

        backend = runner.SyntheticBackend(vocab=32, dim=8)
        measured = profile_mod.measure_memory_profile(
            backend, self._plan(), gpu_reader=NoGpuReader()
        )
        self.assertEqual(measured["gpu_measurement_status"], profile_mod.GPU_UNAVAILABLE)
        self.assertNotIn("peak_gpu_gib", measured)
        self.assertEqual(measured["gpu_unavailable_reason"], "no cuda in test")

    def test_phases_can_be_asserted(self) -> None:
        backend = runner.SyntheticBackend(vocab=32, dim=8)
        measured = profile_mod.measure_memory_profile(backend, self._plan())
        profile_mod.assert_phases_covered(measured)
        with self.assertRaises(MissingInput):
            profile_mod.assert_phases_covered({"phases": [{"phase": "prepare"}]})


class RealCudaProbeTests(unittest.TestCase):
    """真实显存探针：默认跳过（需要显式 V3_RUN_CUDA_PROBE=1）。"""

    @unittest.skipUnless(
        os.environ.get("V3_RUN_CUDA_PROBE") == "1", "默认不碰真实 GPU（设 V3_RUN_CUDA_PROBE=1 才跑）"
    )
    def test_cuda_reader_returns_nonzero_peak(self) -> None:
        probe = profile_mod.probe_cuda_reader(mib=64)
        if not probe["available"]:
            self.skipTest("本机没有可用 CUDA：%s" % probe.get("reason"))
        self.assertTrue(probe["is_real_gpu_probe"])
        self.assertFalse(probe["is_training"])
        self.assertEqual(probe["allocated_bytes"], 64 * (1 << 20))
        self.assertGreater(probe["peak_gpu_gib"], 0.0)


# --------------------------------------------------------------------- G14


class LifecycleTests(unittest.TestCase):
    """G14：中途 checkpoint、停止、恢复都要真的接线。"""

    def _plan(self, steps: int, adapter_name: str = "v3_policy"):
        return runner.TrainRunPlan(
            steps=steps,
            lr=0.3,
            seq_len=12,
            lora_rank=4,
            lora_alpha=8,
            batches=runner.build_smoke_batches(seq_len=12),
            adapter_name=adapter_name,
        )

    def test_checkpoints_are_written_and_stop_is_honoured(self) -> None:
        with temp_dir_outside_repo("g14_stop_") as root:
            dest = os.path.join(root, "adapter_out")
            ckpt = os.path.join(root, "checkpoints")
            stop = lifecycle.StopRequest()
            life = lifecycle.TrainingLifecycle(
                store_root=ckpt,
                hashes=dict(FIXTURE_HASHES),
                policy=lifecycle.LifecyclePolicy(checkpoint_every=4, stop_at_step=6),
                stop=stop,
            )
            report = runner.run_training(
                runner.SyntheticBackend(vocab=32, dim=8),
                self._plan(8),
                dest,
                lifecycle=life,
                stop=stop,
            )
            self.assertTrue(report["stopped"])
            self.assertEqual(report["stop_reason"], lifecycle.STOP_STEP_LIMIT)
            self.assertEqual(report["trained"]["steps_executed"], 6)
            self.assertEqual(report["trained"]["global_step"], 6)
            self.assertGreaterEqual(report["lifecycle"]["checkpoint_count"], 2)
            self.assertIsNotNone(report["lifecycle"]["saved_on_stop"])
            self.assertTrue(report["reloaded"]["matches_saved_adapter"])
            self.assertTrue(report["dest_dir"])
            # 停止不跳过完整性检查
            self.assertTrue(report["trained"]["base_params_frozen"])
            # 盘上确实有完整快照
            roundtrip = lifecycle.verify_checkpoint_roundtrip(ckpt, 6, dict(FIXTURE_HASHES))
            self.assertEqual(roundtrip["global_step"], 6)

    def test_resume_continues_from_checkpoint(self) -> None:
        with temp_dir_outside_repo("g14_resume_") as root:
            ckpt = os.path.join(root, "checkpoints")
            stop = lifecycle.StopRequest()
            first = lifecycle.TrainingLifecycle(
                store_root=ckpt,
                hashes=dict(FIXTURE_HASHES),
                policy=lifecycle.LifecyclePolicy(checkpoint_every=4, stop_at_step=6),
                stop=stop,
            )
            runner.run_training(
                runner.SyntheticBackend(vocab=32, dim=8),
                self._plan(8),
                os.path.join(root, "out1"),
                lifecycle=first,
                stop=stop,
            )
            last = sorted(
                name for name in os.listdir(ckpt) if name.startswith("step_")
            )[-1]
            resume_dir = os.path.join(ckpt, last)
            backend = runner.SyntheticBackend(vocab=32, dim=8)
            second = lifecycle.TrainingLifecycle(
                store_root=ckpt,
                hashes=dict(FIXTURE_HASHES),
                policy=lifecycle.LifecyclePolicy(checkpoint_every=0),
                resume_from=resume_dir,
                stop=lifecycle.StopRequest(),
            )
            report = runner.run_training(
                backend,
                self._plan(3),
                os.path.join(root, "out2"),
                lifecycle=second,
            )
            resumed = report["lifecycle"]["resume"]
            self.assertTrue(resumed["resumed"])
            self.assertEqual(resumed["global_step"], 6)
            self.assertEqual(report["trained"]["start_step"], 6)
            self.assertEqual(report["trained"]["global_step"], 9)
            self.assertTrue(report["reloaded"]["matches_saved_adapter"])

    def test_resume_rejects_hash_mismatch(self) -> None:
        with temp_dir_outside_repo("g14_hash_") as root:
            ckpt = os.path.join(root, "checkpoints")
            stop = lifecycle.StopRequest()
            first = lifecycle.TrainingLifecycle(
                store_root=ckpt,
                hashes=dict(FIXTURE_HASHES),
                policy=lifecycle.LifecyclePolicy(checkpoint_every=2),
                stop=stop,
            )
            runner.run_training(
                runner.SyntheticBackend(vocab=32, dim=8),
                self._plan(2),
                os.path.join(root, "out"),
                lifecycle=first,
                stop=stop,
            )
            step_dir = os.path.join(ckpt, sorted(os.listdir(ckpt))[-1])
            wrong = dict(FIXTURE_HASHES)
            wrong["data_sha256"] = "f" * 64
            backend = runner.SyntheticBackend(vocab=32, dim=8)
            second = lifecycle.TrainingLifecycle(
                store_root=ckpt,
                hashes=wrong,
                policy=lifecycle.LifecyclePolicy(),
                resume_from=step_dir,
                stop=lifecycle.StopRequest(),
            )
            with self.assertRaises(IntegrityError) as ctx:
                runner.run_training(
                    backend, self._plan(2), os.path.join(root, "out2"), lifecycle=second
                )
            self.assertEqual(ctx.exception.code, "resume_hash_mismatch")

    def test_backend_without_step_hook_is_refused(self) -> None:
        class NoHookBackend(runner.SyntheticBackend):
            name = "no-hook"

            def train_steps(self, plan):  # 故意不接受 on_step
                return super().train_steps(plan)

        with temp_dir_outside_repo("g14_nohook_") as root:
            life = lifecycle.TrainingLifecycle(
                store_root=os.path.join(root, "ckpt"),
                hashes=dict(FIXTURE_HASHES),
                policy=lifecycle.LifecyclePolicy(),
                stop=lifecycle.StopRequest(),
            )
            with self.assertRaises(PolicyViolation) as ctx:
                runner.run_training(
                    NoHookBackend(vocab=32, dim=8),
                    self._plan(2),
                    os.path.join(root, "out"),
                    lifecycle=life,
                )
            self.assertEqual(ctx.exception.code, "backend_lacks_step_hook")

    def test_deadline_stop_reason(self) -> None:
        stop = lifecycle.StopRequest(deadline_monotonic=0.0, signals=())
        with temp_dir_outside_repo("g14_deadline_") as root:
            life = lifecycle.TrainingLifecycle(
                store_root=os.path.join(root, "ckpt"),
                hashes=dict(FIXTURE_HASHES),
                policy=lifecycle.LifecyclePolicy(),
                stop=stop,
            )
            report = runner.run_training(
                runner.SyntheticBackend(vocab=32, dim=8),
                self._plan(4),
                os.path.join(root, "out"),
                lifecycle=life,
                stop=stop,
            )
        self.assertTrue(report["stopped"])
        self.assertEqual(report["stop_reason"], lifecycle.STOP_DEADLINE)
        self.assertEqual(report["trained"]["steps_executed"], 1)

    def test_signal_request_is_idempotent_and_restored(self) -> None:
        import signal as signal_mod

        stop = lifecycle.StopRequest(signals=(signal_mod.SIGTERM,))
        original = signal_mod.getsignal(signal_mod.SIGTERM)
        with stop:
            handler = signal_mod.getsignal(signal_mod.SIGTERM)
            self.assertNotEqual(handler, original)
            handler(signal_mod.SIGTERM, None)
            self.assertTrue(stop.requested)
            self.assertEqual(stop.reason, lifecycle.STOP_SIGNAL)
            self.assertFalse(stop.request(lifecycle.STOP_EXTERNAL), "第二次请求应被忽略")
            self.assertEqual(stop.reason, lifecycle.STOP_SIGNAL)
        self.assertEqual(signal_mod.getsignal(signal_mod.SIGTERM), original)

    def test_import_state_requires_rng_and_cursor(self) -> None:
        backend = runner.SyntheticBackend(vocab=32, dim=8)
        backend.prepare(self._plan(2))
        state = ResumeState(
            global_step=3,
            consumed_input_tokens=1,
            consumed_supervised_tokens=1,
            accum_boundary=1,
            adapter_sha256="x",
            optimizer={},
            rng={},
            cursor={},
            hashes=dict(FIXTURE_HASHES),
        )
        with self.assertRaises(MissingInput):
            backend.import_training_state(state)

    def test_lifecycle_requires_all_hashes(self) -> None:
        with temp_dir_outside_repo("g14_hashes_") as root:
            with self.assertRaises(MissingInput) as ctx:
                lifecycle.TrainingLifecycle(
                    store_root=os.path.join(root, "ckpt"), hashes={"source_sha256": "a" * 64}
                )
            self.assertEqual(ctx.exception.code, "lifecycle_missing_hashes")

    def test_tensor_codec_roundtrip_with_duck_typed_tensor(self) -> None:
        class FakeArray:
            def __init__(self, payload: bytes, dtype: str, shape) -> None:
                self._payload = payload
                self.dtype = dtype
                self.shape = shape

            def tobytes(self) -> bytes:
                return self._payload

        class FakeTensor:
            def __init__(self) -> None:
                self.dtype = "float32"
                self.shape = (2,)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return FakeArray(b"\x01\x02\x03\x04\x05\x06\x07\x08", "float32", (2,))

        encoded = lifecycle.jsonify_tensors({"weights": FakeTensor()})
        payload = encoded["weights"]["__tensor__"]
        self.assertEqual(payload["data_base64"], "AQIDBAUGBwg=")
        self.assertEqual(payload["shape"], [2])

        def decoder(item):
            import base64

            return base64.b64decode(item["data_base64"])

        decoded = lifecycle.unjsonify_tensors(encoded, decoder=decoder)
        self.assertEqual(decoded["weights"], b"\x01\x02\x03\x04\x05\x06\x07\x08")

    def test_jsonify_rejects_unserializable(self) -> None:
        with self.assertRaises(FailClosed) as ctx:
            lifecycle.jsonify_tensors({"bad": object()})
        self.assertEqual(ctx.exception.code, "checkpoint_state_not_serializable")


# --------------------------------------------------------- 工程检查入口


class EngineeringCheckTests(unittest.TestCase):
    """G7b：合成工程检查入口必须能停、能存、能恢复、能出证据。"""

    def test_run_engineering_check_passes_and_reports_honestly(self) -> None:
        with temp_dir_outside_repo("engchk_") as root:
            report = engineering_check.run_engineering_check(
                dest_dir=root,
                steps=8,
                checkpoint_every=4,
                stop_at_step=6,
                time_budget_seconds=300.0,
            )
        self.assertEqual(report["verdict"], "pass")
        self.assertTrue(report["stopped"])
        self.assertEqual(report["stop_reason"], lifecycle.STOP_STEP_LIMIT)
        self.assertFalse(report["is_real_gpu"])
        self.assertFalse(report["is_real_training"])
        self.assertTrue(report["hashes"]["are_synthetic_fixture"])
        self.assertTrue(report["negative_hash_mismatch_rejected"])
        self.assertIsNone(report["memory_profile_payload"])
        self.assertGreaterEqual(report["checkpoint_count"], 1)
        self.assertTrue(report["resume"]["adapter_reload_matches"])
        self.assertTrue(report["adapter_export"]["validation"]["ok"])
        self.assertTrue(report["adapter_export"]["artifacts_are_synthetic"])
        self.assertTrue(report["evidence_sha256"])
        self.assertGreaterEqual(report["elapsed_seconds"], 0.0)
        self.assertEqual(
            set(report["phases_covered"]),
            {"prepare", "forward", "backward", "optimizer_step"},
        )

    def test_engineering_check_refuses_repo_internal_dest(self) -> None:
        inside = os.path.join(destdir.repo_root(), "v3", "train", "_engineering_check")
        with self.assertRaises(PolicyViolation) as ctx:
            engineering_check.run_engineering_check(dest_dir=inside, steps=2)
        self.assertEqual(ctx.exception.code, "dest_dir_inside_repo")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
