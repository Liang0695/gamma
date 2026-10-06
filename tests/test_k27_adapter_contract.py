"""KAGGLE-27 整改①：官方 PEFT adapter 目录契约的正例与全部反例。

Mika 的整改要求（KAGGLE-24 描述「KAGGLE-27 新发现限定 CPU 整改」第 1 条）：

    导出、打包及校验统一官方 PEFT 目录 `adapters/v3_policy/adapter_model.safetensors`，
    并强制 `adapter_config.json` 存在且可解析；**不要仅改名就宣称可加载**。
    补官方命名正例、缺 config／错声明／旧命名反例，使用官方 discovery/resolver
    验证名称契约；真实加载仍待验。

本文件覆盖 E0 侧的**静态契约**回归（官方 discovery/resolver 的实机验证由
`tools/k27_adapter_naming_probe.py` 在带官方包的环境里产出，证据冻结在
`docs/v3/design/kaggle-27-s1-adapter-matrix.json` 与 `…-s1b-discovery-rule-cells.json`）。

**不声称**：真实 adapter 已加载、已路由、官方 compiler 跑过完整提交包。
"""

from __future__ import annotations

import json
import os
import unittest

from tests._tmp import temp_dir
from v3.common.errors import FailClosed, MissingInput, PolicyViolation
from v3.submit import adapter_contract as ac
from v3.submit import validate as submit

CONFIG = b'{"r": 16, "lora_alpha": 32, "peft_type": "LORA"}'


def _canonical_bytes(raw: bytes) -> bytes:
    """把 CRLF 收敛成 LF 后返回，用于**内容级**哈希比对。

    为什么需要它：本机与干净克隆的 `core.autocrlf=true` 会在 checkout 时把 LF 翻成
    CRLF，让工作树字节与提交里的 blob 不是同一份（实测：这在干净克隆上导致 6 条
    断言失败）。仓库已用 `.gitattributes` 给 `docs/v3/design/*.json` 打了 `-text`
    （不做任何换行转换），这里再做一层内容级兜底。

    它只吞掉"整份文件都是 CRLF"这一种差异；**混合换行**视为损坏并直接失败，
    以免把真正的字节改动当成换行转换放过。
    """
    if b"\r\n" in raw:
        if b"\r\n" in raw.replace(b"\r\n", b""):
            raise AssertionError("文件混用 CRLF 与 LF，视为损坏：拒绝按换行转换放过")
        return raw.replace(b"\r\n", b"\n")
    return raw


def _write(path: str, payload: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def _official(root: str, name: str = ac.DEFAULT_ADAPTER_NAME, *, config: bytes | None = CONFIG) -> str:
    directory = os.path.join(root, *ac.adapter_dir_relative_path(name).split("/"))
    weights = _write(os.path.join(directory, ac.ADAPTER_WEIGHTS_FILENAME), b"\x00" * 64)
    if config is not None:
        _write(os.path.join(directory, ac.ADAPTER_CONFIG_FILENAME), config)
    return weights


class NamingRuleTests(unittest.TestCase):
    """把官方 `discover_adapters()` 的规则固化成可断言的函数。"""

    def test_each_disambiguating_case(self) -> None:
        # 有 config → 目录名
        self.assertEqual(
            ac.discovered_adapter_name(
                weights_basename="other.safetensors", directory_name="v3_policy", has_config=True
            ),
            "v3_policy",
        )
        # 无 config 且 stem == adapter_model → 目录名（官方文件名自带 stem 判据）
        self.assertEqual(
            ac.discovered_adapter_name(
                weights_basename="adapter_model.safetensors",
                directory_name="custom_dir",
                has_config=False,
            ),
            "custom_dir",
        )
        # 无 config 且 stem != adapter_model → stem（这正是旧命名的失败模式）
        for filename, stem in (("adapter.safetensors", "adapter"), ("other.safetensors", "other")):
            self.assertEqual(
                ac.discovered_adapter_name(
                    weights_basename=filename, directory_name="v3_policy", has_config=False
                ),
                stem,
            )

    def test_rule_matches_frozen_official_matrix(self) -> None:
        """规则必须能逐格复现官方 6 格（S1）+ E0 补的 4 格，共 10 格。"""
        report = ac.verify_rule_against_frozen_matrix()
        self.assertTrue(report["ok"])
        self.assertEqual(report["cell_count"], 10)
        self.assertEqual(report["problems"], [])

    def test_frozen_s1_matrix_is_present_and_byte_stable(self) -> None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ac.S1_MATRIX_RELATIVE_PATH,
        )
        self.assertTrue(os.path.exists(path), path)
        with open(path, "rb") as handle:
            payload = handle.read()
        # 与 KAGGLE-27 附件里 S1 的字节完全相同（E0 独立重跑得到同一份）。
        # 哈希按**规范化 LF** 求，兼容 autocrlf=true 的 checkout（见 `_canonical_bytes`）。
        import hashlib

        self.assertEqual(
            hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
            "fe918929f679165623e6aadc78bfb6a52363cd875ca4c59a7d5a214d7edb4f35",
        )
        # 冻结副本里不得出现本机绝对路径
        self.assertNotIn(b"multica", payload)
        self.assertNotIn(b"C:\\", payload)
        self.assertNotIn(b"L:\\", payload)

    def test_rule_rejects_a_tampered_frozen_matrix(self) -> None:
        """反例：冻结证据被改动时，规则自检必须失败，而不是静默放过。"""
        import shutil

        source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with temp_dir("rule_tamper_") as sandbox:
            design = os.path.join(sandbox, "docs", "v3", "design")
            os.makedirs(design, exist_ok=True)
            shutil.copy(
                os.path.join(source_root, ac.S1_MATRIX_RELATIVE_PATH),
                os.path.join(design, os.path.basename(ac.S1_MATRIX_RELATIVE_PATH)),
            )
            shutil.copy(
                os.path.join(source_root, ac.RULE_CELLS_RELATIVE_PATH),
                os.path.join(design, os.path.basename(ac.RULE_CELLS_RELATIVE_PATH)),
            )
            target = os.path.join(design, os.path.basename(ac.RULE_CELLS_RELATIVE_PATH))
            with open(target, "rb") as handle:
                payload = json.loads(handle.read())
            payload["cases"][0]["discovered_adapter_names"] = ["v3_policy"]  # 篡改
            with open(target, "wb") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
            with self.assertRaises(FailClosed) as ctx:
                ac.verify_rule_against_frozen_matrix(sandbox)
            self.assertEqual(ctx.exception.code, "adapter_naming_rule_mismatch")


class CarrierPositiveTests(unittest.TestCase):
    def test_official_layout_passes(self) -> None:
        with temp_dir("carrier_ok_") as root:
            _official(root)
            report = submit.validate_submission_dir(
                root, declared_adapter_name=ac.DEFAULT_ADAPTER_NAME
            )
        self.assertTrue(report["ok"])
        carrier = report["adapter_carrier"]
        self.assertEqual(carrier["carrier"], "official-peft-directory")
        self.assertEqual(carrier["discovered_adapter_names"], [ac.DEFAULT_ADAPTER_NAME])
        self.assertTrue(carrier["adapters"][0]["has_adapter_config_json"])
        self.assertTrue(carrier["adapters"][0]["adapter_config_parsable"])
        self.assertTrue(carrier["static_contract_only"])
        # 明确不声称真实加载
        self.assertFalse(carrier["real_adapter_loading_verified"])
        self.assertFalse(report["official_limits_verified"])

    def test_official_layout_passes_without_declared_name(self) -> None:
        """不给声明名时只做载体形态校验，也应通过；且仍报告发现名。"""
        with temp_dir("carrier_nodecl_") as root:
            _official(root)
            report = submit.validate_submission_dir(root)
        self.assertEqual(report["adapter_carrier"]["discovered_adapter_names"], ["v3_policy"])

    def test_custom_adapter_name_is_supported(self) -> None:
        with temp_dir("carrier_custom_") as root:
            _official(root, "policy_alpha")
            report = submit.validate_submission_dir(root, declared_adapter_name="policy_alpha")
        self.assertEqual(report["adapter_carrier"]["discovered_adapter_names"], ["policy_alpha"])

    def test_carrier_paths_are_the_official_ones(self) -> None:
        self.assertEqual(
            ac.carrier_weights_relative_path(ac.DEFAULT_ADAPTER_NAME),
            "adapters/v3_policy/adapter_model.safetensors",
        )
        self.assertEqual(
            ac.carrier_config_relative_path(ac.DEFAULT_ADAPTER_NAME),
            "adapters/v3_policy/adapter_config.json",
        )

    def test_declared_limits_expose_the_contract(self) -> None:
        block = submit.declared_limits()["adapter_carrier_contract"]
        self.assertEqual(block["layout"], "adapters/<adapter_name>/adapter_model.safetensors")
        self.assertEqual(block["weights_filename"], "adapter_model.safetensors")
        self.assertEqual(block["config_filename"], "adapter_config.json")
        self.assertTrue(block["config_required"])
        self.assertEqual(block["legacy_filename_rejected"], "adapter.safetensors")
        self.assertIn("adapter_model", block["discovery_rule"])


class CarrierNegativeTests(unittest.TestCase):
    """逐条反例：每一条都必须**拒**，并给出稳定的错误码。"""

    def _expect(self, root: str, code: str, **kwargs) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            submit.validate_submission_dir(root, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_legacy_e0_name_is_rejected(self) -> None:
        """旧命名 `adapter.safetensors`：即便有 config 也拒（不再接受旧形态）。"""
        for with_config in (True, False):
            with temp_dir("carrier_legacy_") as root:
                directory = os.path.join(root, "adapters", ac.DEFAULT_ADAPTER_NAME)
                _write(os.path.join(directory, "adapter.safetensors"), b"\x00" * 64)
                if with_config:
                    _write(os.path.join(directory, ac.ADAPTER_CONFIG_FILENAME), CONFIG)
                self._expect(root, "adapter_carrier_legacy_name")

    def test_missing_config_is_rejected(self) -> None:
        with temp_dir("carrier_noconf_") as root:
            _official(root, config=None)
            self._expect(root, "adapter_carrier_config_missing")

    def test_unparsable_config_is_rejected(self) -> None:
        with temp_dir("carrier_badconf_") as root:
            _official(root, config=b"{not json")
            self._expect(root, "adapter_carrier_config_unparsable")

    def test_non_object_config_is_rejected(self) -> None:
        with temp_dir("carrier_listconf_") as root:
            _official(root, config=b'["not", "an", "object"]')
            self._expect(root, "adapter_carrier_config_not_object")

    def test_wrong_basename_is_rejected(self) -> None:
        with temp_dir("carrier_wrongname_") as root:
            directory = os.path.join(root, "adapters", ac.DEFAULT_ADAPTER_NAME)
            _write(os.path.join(directory, "v3_policy.safetensors"), b"\x00" * 64)
            _write(os.path.join(directory, ac.ADAPTER_CONFIG_FILENAME), CONFIG)
            self._expect(root, "adapter_carrier_wrong_filename")

    def test_weights_outside_adapters_dir_are_rejected(self) -> None:
        with temp_dir("carrier_flat_") as root:
            _write(
                os.path.join(root, ac.ADAPTER_WEIGHTS_FILENAME),
                b"\x00" * 64,
            )
            self._expect(root, "adapter_carrier_not_in_adapters_dir")

    def test_declared_name_mismatch_is_rejected(self) -> None:
        """错声明：目录叫 v3_policy，YAML 却声明 policy_beta → 官方会 AdapterNotFoundError。"""
        with temp_dir("carrier_mismatch_") as root:
            _official(root, "v3_policy")
            self._expect(
                root, "adapter_carrier_declared_name_unresolved", declared_adapter_name="policy_beta"
            )

    def test_adapter_name_with_separator_is_rejected(self) -> None:
        for bad in ("", "a/b", "..", "a\\b"):
            with self.assertRaises(FailClosed):
                ac.adapter_dir_relative_path(bad)

    def test_production_default_requires_config(self) -> None:
        """默认入口必须要求 config；只有显式 require_config=False 才放宽（测试辅助语义）。"""
        with temp_dir("carrier_lenient_") as root:
            _official(root, config=None)
            report = ac.assert_official_adapter_carrier(root, require_config=False)
        self.assertFalse(report["adapters"][0]["has_adapter_config_json"])
        self.assertFalse(report["config_required"])

    def test_missing_adapter_directory_is_rejected(self) -> None:
        with temp_dir("carrier_none_") as root:
            _write(os.path.join(root, "agent.yaml"), b"name: demo\n")
            self._expect(root, "submission_adapter_missing")


class ExportWiringTests(unittest.TestCase):
    """导出面（`TorchPeftBackend.save_adapter`）必须落到官方 PEFT 目录。

    用 CPU 替身替换真实 `save_pretrained`：**只验证接线与落盘形态**，
    不加载任何真实模型/权重（真实训练仍未验证）。
    """

    class _StubModel:
        def __init__(self, *, with_config: bool = True) -> None:
            self.with_config = with_config
            self.saved_to: str | None = None

        def save_pretrained(self, directory: str, safe_serialization: bool = True) -> None:
            self.saved_to = directory
            _write(os.path.join(directory, ac.ADAPTER_WEIGHTS_FILENAME), b"\x00" * 128)
            if self.with_config:
                _write(os.path.join(directory, ac.ADAPTER_CONFIG_FILENAME), CONFIG)

    def _backend(self, model) -> object:
        from v3.train import runner

        backend = runner.TorchPeftBackend(
            model_id="google/gemma-4-31B-it-qat-w4a16-ct",
            revision="52f3f65bc7a02d555763bc923bd1d9094898219d",
            target_modules=["o_proj", "q_proj"],
        )
        backend.model = model
        return backend

    def test_save_adapter_writes_official_carrier(self) -> None:
        with temp_dir("export_ok_") as root:
            model = self._StubModel()
            backend = self._backend(model)
            saved = backend.save_adapter(root, adapter_name=ac.DEFAULT_ADAPTER_NAME)
        self.assertTrue(saved["is_official_submission_carrier"])
        self.assertEqual(saved["filename"], ac.ADAPTER_WEIGHTS_FILENAME)
        self.assertEqual(saved["relative"], "adapters/v3_policy/adapter_model.safetensors")
        self.assertEqual(
            os.path.normpath(model.saved_to),
            os.path.normpath(os.path.join(root, "adapters", "v3_policy")),
        )
        self.assertTrue(saved["adapter_config"]["parsable"])

    def test_save_adapter_rejects_backend_without_config(self) -> None:
        """后端没写 config 时必须拦住（而不是等到官方 compiler 报 AdapterNotFoundError）。"""
        with temp_dir("export_noconf_") as root:
            backend = self._backend(self._StubModel(with_config=False))
            with self.assertRaises(PolicyViolation) as ctx:
                backend.save_adapter(root, adapter_name=ac.DEFAULT_ADAPTER_NAME)
        self.assertEqual(ctx.exception.code, "adapter_carrier_config_missing")

    def test_plan_adapter_name_must_be_valid(self) -> None:
        from v3.train import runner

        plan = runner.TrainRunPlan(
            steps=1,
            lr=1e-4,
            seq_len=8,
            lora_rank=4,
            lora_alpha=8,
            adapter_name="..",
            batches=runner.build_smoke_batches(count=1, seq_len=8),
        )
        with self.assertRaises(PolicyViolation) as ctx:
            plan.assert_runnable()
        self.assertEqual(ctx.exception.code, "adapter_name_invalid")

    def test_plan_default_adapter_name_is_official(self) -> None:
        from v3.train import runner

        self.assertEqual(runner.TrainRunPlan(steps=1, lr=1e-4, seq_len=8, lora_rank=4, lora_alpha=8).adapter_name, ac.DEFAULT_ADAPTER_NAME)
        self.assertEqual(runner.ADAPTER_FILE, ac.ADAPTER_WEIGHTS_FILENAME)


class ExportGateTests(unittest.TestCase):
    """`measure_gates` 的导出闸门必须认官方载体路径。"""

    def _manifest(self, files: list[str]) -> dict:
        return {
            **{
                key: "a" * 64
                for key in (
                    "source_sha256",
                    "data_sha256",
                    "config_sha256",
                    "code_sha256",
                    "deps_sha256",
                )
            },
            "adapter_only": True,
            "files": files,
        }

    def test_official_carrier_path_satisfies_the_gate(self) -> None:
        from v3.train import entry

        manifest = self._manifest([ac.carrier_weights_relative_path(ac.DEFAULT_ADAPTER_NAME)])
        report = entry.measure_gates(export_manifest=manifest, run_cpu_self_check=False)
        self.assertTrue(report["gates"]["export_manifest_pass"])

    def test_legacy_file_list_fails_the_gate(self) -> None:
        from v3.train import entry

        report = entry.measure_gates(
            export_manifest=self._manifest(["adapter.safetensors"]), run_cpu_self_check=False
        )
        self.assertFalse(report["gates"]["export_manifest_pass"])
        problems = report["evidence"]["export_manifest"]["problems"]
        self.assertTrue(
            any("adapters/v3_policy/adapter_model.safetensors" in item for item in problems),
            problems,
        )


class ToolAvailabilityTests(unittest.TestCase):
    """官方包不可用时探针必须 fail-closed（返回 3，不产出结果文件）。"""

    def test_probe_is_fail_closed_without_official_packages(self) -> None:
        import importlib.util

        if importlib.util.find_spec("adk_submission") is not None:
            self.skipTest("本环境装了官方包，无法验证『缺包即 fail-closed』这支分支")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tool_path = os.path.join(repo_root, "tools", "k27_adapter_naming_probe.py")
        spec = importlib.util.spec_from_file_location("k27_adapter_naming_probe", tool_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with temp_dir("probe_fc_") as sandbox:
            out = os.path.join(sandbox, "out.json")
            code = module.main(
                ["probe", "--work", os.path.join(sandbox, "work"), "--out", out]
            )
            self.assertEqual(code, 3)
            self.assertFalse(os.path.exists(out), "官方包缺失时不得产出结果文件")


class NoLegacyNameTests(unittest.TestCase):
    """全仓 `v3/` 可执行代码里不得再出现旧命名（注释/文档不计）。"""

    def test_v3_sources_do_not_hardcode_legacy_adapter_filename(self) -> None:
        """旧名字面量只允许出现在 `v3/submit/adapter_contract.py` 的常量定义处。"""
        import ast

        v3_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v3"
        )
        offenders: list[str] = []
        definition_hits: list[str] = []
        for current, _dirs, names in os.walk(v3_root):
            for name in names:
                if not name.endswith(".py"):
                    continue
                full = os.path.join(current, name)
                relative = os.path.relpath(full, v3_root).replace("\\", "/")
                with open(full, "r", encoding="utf-8-sig") as handle:
                    text = handle.read()
                tree = ast.parse(text)
                docstrings = set()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                        doc = ast.get_docstring(node, clean=False)
                        if doc:
                            docstrings.add(doc)
                stripped = text
                for doc in docstrings:
                    stripped = stripped.replace(doc, "")
                for line in stripped.splitlines():
                    code = line.split("#", 1)[0]
                    if '"adapter.safetensors"' not in code and "'adapter.safetensors'" not in code:
                        continue
                    entry = "%s: %s" % (relative, line.strip())
                    if relative == "submit/adapter_contract.py":
                        definition_hits.append(entry)
                    else:
                        offenders.append(entry)
        self.assertEqual(offenders, [], offenders)
        # adapter_contract.py 里的命中只允许两类：
        #  ① 常量定义 `LEGACY_ADAPTER_FILENAME = "adapter.safetensors"`（恰好一处）；
        #  ② EXPECTED_S1_CELLS 里对官方矩阵 fixture 文件名的描述（`"e0_name__…"` 条目）。
        assignments = [item for item in definition_hits if "LEGACY_ADAPTER_FILENAME =" in item]
        self.assertEqual(len(assignments), 1, definition_hits)
        leftovers = [
            item
            for item in definition_hits
            if item not in assignments and '"e0_name__' not in item
        ]
        self.assertEqual(leftovers, [], leftovers)
        self.assertEqual(ac.LEGACY_ADAPTER_FILENAME, "adapter.safetensors")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
