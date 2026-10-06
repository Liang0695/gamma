"""🟡-3 / 🟡-4 / 🟡-8 的接线回归：pin 绑定、fixture 入口隔离、内存纪律门槛。

三条都来自 KAGGLE-26 独立复核：

- 🟡-3 `start()` 的默认真实后端路径必然失败：`model_id` 落到硬编码
  `"google/gemma-4-1b-it"`（不是锁定模型），`target_modules` 读一个**不存在**的
  `DEFAULT_CONFIG["lora"]["target_suffixes"]` → 空清单；
- 🟡-4 默认 fixture 入口对 8/8 放行，等于把"小集全绿"当"20/20 冻结验收"；
- 🟡-8 `assert_no_full_state_dict` 改对了方向，但**生产路径没有调用点**。

本文件只做 CPU 接线验证：**不声称**跑过真实 GPU 训练，真实显存峰值仍未验证。
"""

from __future__ import annotations

import ast
import json
import os
import re
import unittest

from tests._tmp import temp_dir
from v3.common.errors import FailClosed, MissingInput, PolicyViolation, UnverifiedLock
from v3.train import checkpoint, entry, runner
from v3.train.entry import resolve_backend_pins
from v3.train.streaming import assert_full_state_dict_guard
from v3.train.targets import TARGET_REGEX, target_module_names

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
V3_ROOT = os.path.join(REPO_ROOT, "v3")
INTERFACE_LOCK = os.path.join(V3_ROOT, "locks", "official-interface.json")
#: 锁定的真实模型（KAGGLE-20 §3 / official-interface.json pins）。
LOCKED_MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"
LOCKED_MODEL_REVISION = "52f3f65bc7a02d555763bc923bd1d9094898219d"
#: 旧实现硬编码的错误模型（不是锁定模型）。
OLD_HARDCODED_MODEL_ID = "google/gemma-4-1b-it"


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """去掉模块/类/函数的 docstring，只留可执行代码（注释本来就不在 AST 里）。"""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return tree


def _v3_sources(keep_docstrings: bool = False) -> dict[str, str]:
    """`v3/` 下全部 .py 源码。

    默认**剥掉 docstring 与注释**：要断言的是"可执行代码里没有旧硬编码/旧死键"，
    而不是"文档里不许提到它"—— 恰恰相反，文档必须如实记录原来的缺陷。
    """
    out: dict[str, str] = {}
    for current, _dirs, names in os.walk(V3_ROOT):
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(current, name)
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            key = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
            if not keep_docstrings:
                try:
                    text = ast.unparse(_strip_docstrings(ast.parse(text)))
                except SyntaxError:  # pragma: no cover - 仓库里的 .py 应当都能解析
                    pass
            out[key] = text
    return out


class TargetModuleDerivationTests(unittest.TestCase):
    def test_names_are_derived_from_the_regex(self) -> None:
        self.assertEqual(target_module_names(), ["o_proj", "q_proj"])
        self.assertEqual(target_module_names(TARGET_REGEX), ["o_proj", "q_proj"])

    def test_underivable_regex_fails_closed(self) -> None:
        for bad in ("", "model.layers.*.self_attn", "^foo$"):
            with self.assertRaises(MissingInput):
                target_module_names(bad)


class ResolveBackendPinsTests(unittest.TestCase):
    """🟡-3：`(model_id, revision, target_modules)` 来自锁，**没有硬编码回退**。"""

    def test_pins_come_from_the_official_interface_lock(self) -> None:
        pins = resolve_backend_pins(interface_path=INTERFACE_LOCK)
        self.assertEqual(pins["model_id"], LOCKED_MODEL_ID)
        self.assertEqual(pins["model_revision"], LOCKED_MODEL_REVISION)
        self.assertEqual(pins["target_modules"], ["o_proj", "q_proj"])
        self.assertEqual(pins["source"], "official-interface.json:pins")

    def test_explicit_arguments_override_the_lock(self) -> None:
        pins = resolve_backend_pins(
            interface_path=INTERFACE_LOCK,
            model_id="someone/other-model",
            model_revision="b" * 40,
            target_modules=["q_proj"],
        )
        self.assertEqual(pins["model_id"], "someone/other-model")
        self.assertEqual(pins["model_revision"], "b" * 40)
        self.assertEqual(pins["target_modules"], ["q_proj"])

    def _interface(self, workdir: str, **pin_overrides) -> str:
        with open(INTERFACE_LOCK, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["pins"].update(pin_overrides)
        path = os.path.join(workdir, "interface.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_unverified_revision_pin_is_rejected(self) -> None:
        with temp_dir("pins_unver_") as workdir:
            path = self._interface(workdir, model_revision={"source": "s", "value": LOCKED_MODEL_REVISION, "verified": False})
            with self.assertRaises(UnverifiedLock) as ctx:
                resolve_backend_pins(interface_path=path)
            self.assertEqual(ctx.exception.code, "pin_unverified")

    def test_missing_revision_pin_is_rejected(self) -> None:
        with temp_dir("pins_missing_") as workdir:
            path = self._interface(workdir, model_revision={"source": "s", "value": None, "verified": True})
            with self.assertRaises(UnverifiedLock) as ctx:
                resolve_backend_pins(interface_path=path)
            self.assertEqual(ctx.exception.code, "model_revision_pin_missing")

    def test_latest_revision_is_rejected(self) -> None:
        for bad in ("latest", "main", "HEAD", "  "):
            with self.assertRaises(FailClosed):
                resolve_backend_pins(
                    interface_path=INTERFACE_LOCK, model_revision=bad
                )

    def test_missing_interface_file_is_fail_closed(self) -> None:
        with self.assertRaises(FailClosed):
            resolve_backend_pins(interface_path=os.path.join(HERE, "no-such-lock.json"))


class MissingPinStopsBeforeBackendConstructionTests(unittest.TestCase):
    """缺 pin 必须在**构造真实后端之前**拒绝，而不是等到 GPU 上才炸。"""

    def test_start_raises_before_constructing_the_backend(self) -> None:
        calls: list[dict] = []

        class SpyBackend:
            def __init__(self, **kwargs) -> None:  # pragma: no cover - 不应被调用
                calls.append(kwargs)

        original = entry.runner.TorchPeftBackend
        entry.runner.TorchPeftBackend = SpyBackend
        try:
            with temp_dir("pins_start_") as workdir:
                with self.assertRaises(FailClosed) as ctx:
                    entry.start(
                        operator="Liang",
                        gpu_hours=6.0,
                        stage="T1",
                        model_revision="latest",
                        interface_path=INTERFACE_LOCK,
                        dest_dir=workdir,
                    )
        finally:
            entry.runner.TorchPeftBackend = original
        self.assertEqual(ctx.exception.code, "model_revision_pin_missing")
        self.assertEqual(calls, [], "缺 pin 时不得构造后端")


class StartWiringTests(unittest.TestCase):
    """在 CPU 上用替身证明 pin **确实到达真实后端构造**（不代表跑过真实 GPU 训练）。"""

    def _run_start(self, capture: list[dict]) -> dict:
        class SpyBackend:
            name = "spy"
            requires_gpu = True
            verified = False

            def __init__(self, **kwargs) -> None:
                capture.append(kwargs)

        original_backend = entry.runner.TorchPeftBackend
        original_run = entry.runner.run_training
        original_gates = entry.measure_gates
        original_assert = entry.assert_start_allowed
        original_no_gpu = entry.assert_no_auto_gpu
        entry.runner.TorchPeftBackend = SpyBackend
        entry.runner.run_training = lambda backend, plan, dest, **kw: {
            "backend": "spy",
            "executed": True,
            "prepared": {"unverified_claims": []},
        }
        entry.measure_gates = lambda **kwargs: {
            "gates": {name: True for name in entry.GATE_NAMES},
            "evidence": {},
        }
        entry.assert_start_allowed = lambda **kwargs: {"allowed": True}
        entry.assert_no_auto_gpu = lambda flags: None
        try:
            with temp_dir("start_wire_") as workdir:
                report = entry.start(
                    operator="Liang",
                    gpu_hours=6.0,
                    stage="T1",
                    v2_conflict_checked=True,
                    interface_path=INTERFACE_LOCK,
                    plan=object(),
                    dest_dir=workdir,
                )
        finally:
            entry.runner.TorchPeftBackend = original_backend
            entry.runner.run_training = original_run
            entry.measure_gates = original_gates
            entry.assert_start_allowed = original_assert
            entry.assert_no_auto_gpu = original_no_gpu
        return report

    def test_pins_reach_the_real_backend_constructor(self) -> None:
        capture: list[dict] = []
        report = self._run_start(capture)
        self.assertEqual(len(capture), 1, "真实后端应被构造一次")
        self.assertEqual(capture[0]["model_id"], LOCKED_MODEL_ID)
        self.assertEqual(capture[0]["revision"], LOCKED_MODEL_REVISION)
        self.assertEqual(list(capture[0]["target_modules"]), ["o_proj", "q_proj"])
        # 旧的错误模型名绝不能出现在构造参数里
        self.assertNotEqual(capture[0]["model_id"], OLD_HARDCODED_MODEL_ID)
        self.assertEqual(report["backend_pins"]["model_id"], LOCKED_MODEL_ID)
        self.assertEqual(report["authorization"]["model_revision"], LOCKED_MODEL_REVISION)

    def test_resolved_pins_are_accepted_by_the_real_backend_class(self) -> None:
        """用**真实** `TorchPeftBackend` 构造（不 prepare，不联网、不加载权重）。"""
        pins = resolve_backend_pins(interface_path=INTERFACE_LOCK)
        backend = runner.TorchPeftBackend(
            model_id=pins["model_id"],
            revision=pins["model_revision"],
            target_modules=pins["target_modules"],
        )
        self.assertEqual(backend.model_id, LOCKED_MODEL_ID)
        self.assertEqual(backend.revision, LOCKED_MODEL_REVISION)
        self.assertEqual(backend.target_modules, ["o_proj", "q_proj"])
        self.assertTrue(backend.requires_gpu)
        self.assertFalse(backend.verified)  # 仍标未验证：本机没有 GPU 实测

    def test_empty_revision_is_still_rejected_by_the_real_backend(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            runner.TorchPeftBackend(model_id=LOCKED_MODEL_ID, revision="", target_modules=["q_proj"])
        self.assertEqual(ctx.exception.code, "unpinned_model_revision")


class NoHardcodedFallbackSourceScanTests(unittest.TestCase):
    """🟡-3 的源码级证据：错误模型与不存在的键在 `v3/` 里已消失。"""

    def test_hardcoded_model_id_is_gone(self) -> None:
        offenders = [
            path for path, text in _v3_sources().items() if OLD_HARDCODED_MODEL_ID in text
        ]
        self.assertEqual(offenders, [], "旧硬编码模型名仍在：%s" % offenders)

    def test_target_suffixes_key_is_gone(self) -> None:
        offenders = [
            path
            for path, text in _v3_sources().items()
            if re.search(r"target_suffixes", text)
        ]
        self.assertEqual(offenders, [], "不存在的 target_suffixes 键仍被读取：%s" % offenders)

    def test_authorization_does_not_pretend_to_carry_pins(self) -> None:
        """旧实现从 `authorization.get("model_id")` 取值 —— 那个字典里从来没有该键。"""
        text = _v3_sources()["v3/train/entry.py"]
        self.assertNotIn("authorization.get('model_id')", text)
        self.assertNotIn('authorization.get("model_id")', text)
        self.assertIn("'model_id': pins['model_id']", text)


class FixtureEntryIsolationTests(unittest.TestCase):
    """🟡-4：默认 fixture 入口不得被当成 20/20 冻结验收。"""

    def test_old_unqualified_name_no_longer_exists(self) -> None:
        self.assertFalse(hasattr(checkpoint, "assert_adapter_valid"))
        self.assertTrue(hasattr(checkpoint, "assert_adapter_valid_lenient_for_tests"))
        self.assertTrue(hasattr(checkpoint, "assert_adapter_valid_frozen_spec"))

    def test_lenient_entry_marks_itself_as_test_only(self) -> None:
        manifest = {"lora": {"matched_module_count": 120}}
        result = checkpoint.assert_adapter_valid_lenient_for_tests(
            manifest, load_ok=True, params_changed=True, fixture_pass=8, fixture_total=8
        )
        self.assertTrue(result["adapter_valid"])
        self.assertFalse(result["frozen_spec_satisfied"])
        self.assertTrue(result["production_use_forbidden"])
        self.assertEqual(result["entry_kind"], "lenient-for-tests")
        self.assertEqual(result["required_fixture_total"], 20)

    def test_frozen_entry_rejects_8_of_8(self) -> None:
        manifest = {"lora": {"matched_module_count": 120}}
        with self.assertRaises(PolicyViolation) as ctx:
            checkpoint.assert_adapter_valid_frozen_spec(
                manifest, load_ok=True, params_changed=True, fixture_pass=8, fixture_total=8
            )
        self.assertEqual(ctx.exception.code, "adapter_fixture_suite_incomplete")

    def test_frozen_entry_accepts_20_of_20(self) -> None:
        manifest = {"lora": {"matched_module_count": 120}}
        result = checkpoint.assert_adapter_valid_frozen_spec(
            manifest, load_ok=True, params_changed=True, fixture_pass=20, fixture_total=20
        )
        self.assertTrue(result["frozen_spec_satisfied"])
        self.assertFalse(result["production_use_forbidden"])

    def test_no_production_call_site_for_the_lenient_entry(self) -> None:
        """生产零调用证据：`v3/` 下**没有任何**非注释调用/导入。"""
        offenders = []
        for path, text in _v3_sources().items():
            for number, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if "assert_adapter_valid_lenient_for_tests" not in stripped:
                    continue
                # 允许在说明文字里提到它（本模块的 docstring 就要点名），
                # 但不允许出现可执行的导入/调用。
                if stripped.startswith("#") or stripped.startswith("*") or stripped.startswith("-"):
                    continue
                if re.match(r"^[A-Za-z_]", stripped):  # 代码行（包括 import / 调用 / def）
                    if stripped.startswith("def "):
                        continue
                    offenders.append("%s:%d: %s" % (path, number, stripped))
        self.assertEqual(offenders, [], offenders)

    def test_production_entry_uses_the_frozen_spec(self) -> None:
        text = _v3_sources()["v3/train/entry.py"]
        self.assertIn("assert_adapter_valid_frozen_spec", text)
        self.assertIn("from .checkpoint import REQUIRED_HASH_KEYS, assert_adapter_valid_frozen_spec", text)


class FullStateDictGuardWiringTests(unittest.TestCase):
    """🟡-8：门槛接在**真实的保存路径**上，且观测缺失即 fail-closed。"""

    def _plan(self) -> runner.TrainRunPlan:
        return runner.TrainRunPlan(
            steps=2,
            lr=0.2,
            seq_len=8,
            lora_rank=4,
            lora_alpha=8,
            batches=runner.build_smoke_batches(count=2, seq_len=8),
        )

    def _backend(self, observation, saved: list[str]):
        class FakeBackend(runner.TrainBackend):
            name = "fake-observation-backend"
            requires_gpu = False
            verified = False

            def prepare(self, plan):
                return {"backend": self.name, "unverified_claims": []}

            def train_steps(self, plan):
                return {
                    "grad_norm_nonzero": True,
                    "base_params_frozen": True,
                    "optimizer_step_count": plan.steps,
                }

            def observe_full_state_dict(self):
                return observation

            def save_adapter(self, dest_dir):
                saved.append(dest_dir)  # pragma: no cover - 违规路径不应到达
                return {"adapter_only": True, "contains_base_weights": False}

            def load_adapter(self, src_dir):  # pragma: no cover
                return {"matches_saved_adapter": True}

        return FakeBackend()

    def test_resident_full_state_dict_blocks_the_save(self) -> None:
        saved: list[str] = []
        backend = self._backend(
            {
                "probe": "test-probe",
                "method": "test",
                "peak_full_state_dict_bytes": 58 * (1 << 30),
                "declared_weight_bytes": 58 * (1 << 30),
            },
            saved,
        )
        with temp_dir("guard_hit_") as workdir:
            with self.assertRaises(PolicyViolation) as ctx:
                runner.run_training(backend, self._plan(), workdir)
        self.assertEqual(ctx.exception.code, "full_state_dict_resident")
        self.assertEqual(saved, [], "门槛未过时不得保存 adapter")

    def test_missing_observation_blocks_the_save(self) -> None:
        saved: list[str] = []
        backend = self._backend(None, saved)
        with temp_dir("guard_none_") as workdir:
            with self.assertRaises(PolicyViolation) as ctx:
                runner.run_training(backend, self._plan(), workdir)
        self.assertEqual(ctx.exception.code, "full_state_dict_observation_missing")
        self.assertEqual(saved, [])

    def test_clean_observation_passes_and_is_reported(self) -> None:
        with temp_dir("guard_ok_") as workdir:
            backend = runner.SyntheticBackend(vocab=32, dim=4)
            report = runner.run_training(backend, self._plan(), workdir)
        guard = report["memory_guard"]
        self.assertTrue(guard["checked"])
        self.assertFalse(guard["full_state_dict_resident"])
        self.assertEqual(guard["allowed_bytes"], 0)
        self.assertEqual(guard["observation"]["probe"], "synthetic-parameter-residency")

    def test_explicit_observation_overrides_the_backend(self) -> None:
        saved: list[str] = []
        backend = self._backend({"peak_full_state_dict_bytes": 0, "declared_weight_bytes": 0}, saved)
        with temp_dir("guard_override_") as workdir:
            with self.assertRaises(PolicyViolation) as ctx:
                runner.run_training(
                    backend,
                    self._plan(),
                    workdir,
                    full_state_dict_observation={
                        "probe": "caller",
                        "method": "caller",
                        "peak_full_state_dict_bytes": 1,
                        "declared_weight_bytes": 1,
                    },
                )
        self.assertEqual(ctx.exception.code, "full_state_dict_resident")

    def test_guard_rejects_incomplete_observation(self) -> None:
        with self.assertRaises(MissingInput) as ctx:
            assert_full_state_dict_guard({"peak_full_state_dict_bytes": 0})
        self.assertEqual(ctx.exception.code, "full_state_dict_observation_incomplete")

    def test_production_call_site_exists_in_runner(self) -> None:
        text = _v3_sources()["v3/train/runner.py"]
        self.assertIn("assert_full_state_dict_guard(observation", text)
        # 且它排在 save_adapter 之前
        self.assertLess(
            text.index("assert_full_state_dict_guard(observation"),
            text.index("saved = backend.save_adapter(dest_dir, adapter_name=adapter_name)"),
        )

    def test_self_check_reports_the_guard(self) -> None:
        report = runner.self_check_cpu(steps=6)
        self.assertTrue(report["all_passed"])
        self.assertTrue(report["full_state_dict_guard"]["checked"])

    def test_torch_backend_probe_walks_without_state_dict(self) -> None:
        """真实后端的探针**不调用** `state_dict()`（一次性物化全部权重 = 硬规则禁止）。"""
        with open(os.path.join(V3_ROOT, "train", "runner.py"), "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        target = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TorchPeftBackend":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "observe_full_state_dict":
                        target = item
        self.assertIsNotNone(target, "TorchPeftBackend.observe_full_state_dict 不存在")
        called = {
            node.func.attr
            for node in ast.walk(target)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("state_dict", called, called)
        self.assertIn("named_parameters", called, called)
        used_names = {node.id for node in ast.walk(target) if isinstance(node, ast.Name)}
        literals = {node.value for node in ast.walk(target) if isinstance(node, ast.Constant)}
        self.assertTrue(
            any("_full_state_dict_materializations" in str(item) for item in literals | used_names),
            "探针没有引用整份材料化计数器：%r" % (literals,),
        )
