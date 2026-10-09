"""🟡-2：官方提交限额（KAGGLE-27 A 段证据）的强制项与**边界反例**。

KAGGLE-26 复核意见：E0 上一版说"这组官方限额在本 checkout 内找不到出处"是错的
—— 出处就在 KAGGLE-27 的 `A-evidence.json → a2_limits`。本文件验证：

1. 引用的数值与 KAGGLE-27 证据**逐位一致**（不是凭记忆抄的）；
2. 扩展名接受面恰为官方 7 种；
3. 结构限额真的被强制，且**边界行为**明确（恰好等于上限 = 通过，超一个 = 拒；
   总大小走设计稿"严格小于"口径，恰好 3 GiB = 拒）；
4. 内容级限额（agents/skills/深度/…）有出处但本模块判不了，必须单列、不得假装已强制。
"""

from __future__ import annotations

import json
import os
import unittest

from tests._tmp import temp_dir
from v3.common.errors import PolicyViolation
from v3.submit import adapter_contract
from v3.submit import validate as submit

#: KAGGLE-27 A 段证据里的官方结构限额（本文件独立抄一份用于交叉核对）。
EVIDENCE_BUILT_LIMITS_STRUCTURAL = {
    "max_total_size_bytes": 3221225472,
    "max_file_count": 10000,
    "max_yaml_files": 1000,
    "max_agents": 500,
    "max_sub_agent_depth": 50,
    "max_skills": 1000,
    "max_loop_iterations": 500,
    "max_instruction_chars": 1000000,
    "max_total_instruction_chars": 10000000,
    "max_yaml_size_bytes": 52428800,
    "max_skill_size_bytes": 52428800,
}
EVIDENCE_EXTENSIONS = [".json", ".md", ".py", ".safetensors", ".txt", ".yaml", ".yml"]
EVIDENCE_ADAPTER_EXTENSIONS = [".safetensors"]


def _write(root: str, name: str, payload: bytes = b"x") -> str:
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def _adapter(root: str, *, name: str = adapter_contract.DEFAULT_ADAPTER_NAME, size: int = 1) -> str:
    """写一个**官方形态**的 PEFT adapter 载体：`adapters/<name>/adapter_model.safetensors`
    加同目录 `adapter_config.json`（KAGGLE-27 整改①，本文件全部正例都用它）。"""
    directory = os.path.join(root, *adapter_contract.adapter_dir_relative_path(name).split("/"))
    os.makedirs(directory, exist_ok=True)
    weights = os.path.join(directory, adapter_contract.ADAPTER_WEIGHTS_FILENAME)
    with open(weights, "wb") as handle:
        handle.write(b"w" * int(size))
    with open(os.path.join(directory, adapter_contract.ADAPTER_CONFIG_FILENAME), "wb") as handle:
        handle.write(b'{"r": 16, "lora_alpha": 32, "peft_type": "LORA"}')
    return weights


class EvidenceCrossCheckTests(unittest.TestCase):
    def test_values_match_the_kaggle27_evidence(self) -> None:
        for key, value in EVIDENCE_BUILT_LIMITS_STRUCTURAL.items():
            pool = dict(submit.SOURCED_LIMITS)
            pool.update(submit.NOT_LOCALLY_CHECKABLE_LIMITS)
            self.assertIn(key, pool, key)
            self.assertEqual(pool[key]["value"], value, key)
            self.assertIn("KAGGLE-27", pool[key]["citation"], key)

    def test_extension_allowlist_is_exactly_the_official_seven(self) -> None:
        self.assertEqual(
            sorted(submit.OFFICIAL_ALLOWED_EXTENSIONS), sorted(EVIDENCE_EXTENSIONS)
        )
        self.assertEqual(len(submit.OFFICIAL_ALLOWED_EXTENSIONS), 7)
        self.assertEqual([submit.REQUIRED_ADAPTER_EXTENSION], EVIDENCE_ADAPTER_EXTENSIONS)

    def test_unverified_bucket_is_now_empty_and_documented(self) -> None:
        self.assertEqual(submit.UNVERIFIED_OFFICIAL_LIMITS, ())
        limits = submit.declared_limits()
        self.assertEqual(limits["unverified_official_limits"], [])
        # 但"有出处"不等于"已实测官方 builder"
        self.assertFalse(limits["verified_against_official_compiler"])


class BoundaryTests(unittest.TestCase):
    """边界反例：恰好等于上限 → 通过；超一点点 → 拒。"""

    def test_structured_limits_are_exposed_in_the_report(self) -> None:
        with temp_dir("lim_ok_") as root:
            _adapter(root)
            _write(root, "agent.yaml", b"name: demo\n")
            report = submit.validate_submission_dir(root)
        self.assertTrue(report["ok"])
        self.assertEqual(report["enforced_limits"]["max_file_count"], 10000)
        self.assertEqual(report["enforced_limits"]["max_yaml_files"], 1000)
        self.assertEqual(report["enforced_limits"]["max_yaml_size_bytes"], 52428800)
        self.assertEqual(report["enforced_limits"]["max_extension_count"], 7)
        self.assertEqual(report["official_limits_evidence"], submit.KAGGLE27_EVIDENCE)

    def test_file_count_boundary(self) -> None:
        # 官方载体占 2 个文件（权重 + config）。恰好等于上限：通过
        with temp_dir("lim_fc_ok_") as root:
            _adapter(root)
            for index in range(2):
                _write(root, "f%d.txt" % index, b"x")
            report = submit.validate_submission_dir(root, limits={"max_file_count": 4})
            self.assertEqual(report["files"], 4)
        # 上限 + 1：拒绝
        with temp_dir("lim_fc_bad_") as root:
            _adapter(root)
            for index in range(3):
                _write(root, "f%d.txt" % index, b"x")
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root, limits={"max_file_count": 4})
            self.assertEqual(ctx.exception.code, "submission_file_count_exceeded")

    def test_yaml_count_boundary_counts_yml_too(self) -> None:
        with temp_dir("lim_yc_ok_") as root:
            _adapter(root)
            _write(root, "a.yaml", b"x")
            _write(root, "b.yml", b"x")
            report = submit.validate_submission_dir(root, limits={"max_yaml_files": 2})
            self.assertEqual(report["yaml_files"], 2)
        with temp_dir("lim_yc_bad_") as root:
            _adapter(root)
            _write(root, "a.yaml", b"x")
            _write(root, "b.yml", b"x")
            _write(root, "c.yaml", b"x")
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root, limits={"max_yaml_files": 2})
            self.assertEqual(ctx.exception.code, "submission_yaml_count_exceeded")

    def test_yaml_size_boundary(self) -> None:
        with temp_dir("lim_ys_ok_") as root:
            _adapter(root)
            _write(root, "a.yaml", b"x" * 10)
            report = submit.validate_submission_dir(root, limits={"max_yaml_size_bytes": 10})
            self.assertTrue(report["ok"])
        with temp_dir("lim_ys_bad_") as root:
            _adapter(root)
            _write(root, "a.yaml", b"x" * 11)
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root, limits={"max_yaml_size_bytes": 10})
            self.assertEqual(ctx.exception.code, "submission_yaml_size_exceeded")

    def test_total_size_is_strictly_less_than_three_gib(self) -> None:
        """官方 `max_total_size_bytes` 是 ≤ 3GiB，设计稿要求 < 3GiB：本模块取更严的。"""
        limit = submit.SOURCED_LIMITS["max_total_unpacked_bytes"]["value"]
        self.assertEqual(limit, 3 * (1 << 30))
        self.assertEqual(
            submit.SOURCED_LIMITS["max_total_size_bytes"]["value"], 3 * (1 << 30)
        )
        self.assertTrue(submit.SOURCED_LIMITS["max_total_unpacked_bytes"]["strictly_less_than"])
        # 恰好 3 GiB（权重 + config 合计）→ 拒（严格小于的边界）
        with temp_dir("lim_3gib_") as root:
            config_bytes = len(b'{"r": 16, "lora_alpha": 32, "peft_type": "LORA"}')
            directory = os.path.join(
                root,
                *adapter_contract.adapter_dir_relative_path(
                    adapter_contract.DEFAULT_ADAPTER_NAME
                ).split("/")
            )
            os.makedirs(directory, exist_ok=True)
            big = os.path.join(directory, adapter_contract.ADAPTER_WEIGHTS_FILENAME)
            # 稀疏写：不在内存里物化 3 GiB。
            with open(big, "wb") as handle:
                handle.seek(limit - config_bytes - 1)
                handle.write(b"\0")
            with open(
                os.path.join(directory, adapter_contract.ADAPTER_CONFIG_FILENAME), "wb"
            ) as handle:
                handle.write(b'{"r": 16, "lora_alpha": 32, "peft_type": "LORA"}')
            self.assertEqual(os.path.getsize(big), limit - config_bytes)
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root)
            self.assertEqual(ctx.exception.code, "submission_total_size_exceeded")
            self.assertEqual(ctx.exception.context["total_bytes"], limit)

    def test_extension_allowlist_cannot_be_widened_by_caller(self) -> None:
        with temp_dir("lim_ext_") as root:
            _adapter(root)
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(
                    root, limits={"allowed_extensions": [".sh"], "max_extension_count": 7}
                )
            self.assertEqual(ctx.exception.code, "submission_extension_allowlist_too_wide")

    def test_library_default_extensions_are_not_official(self) -> None:
        """`.sh` 在库默认的 29 种里，但不在官方 7 种里 —— 必须拒。"""
        with temp_dir("lim_sh_") as root:
            _adapter(root)
            _write(root, "setup.sh", b"echo hi\n")
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root)
            self.assertEqual(ctx.exception.code, "submission_extension_not_allowed")

    def test_non_safetensors_adapter_from_library_defaults_is_rejected(self) -> None:
        """.gguf 在库默认的 6 种 adapter 扩展名里，但官方只认 `.safetensors`。"""
        with temp_dir("lim_gguf_") as root:
            _write(root, "adapter.gguf", b"w")
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root)
            self.assertEqual(ctx.exception.code, "submission_adapter_not_safetensors")

    def test_content_level_limits_are_not_claimed_as_enforced(self) -> None:
        limits = submit.declared_limits()
        for key in (
            "max_agents",
            "max_sub_agent_depth",
            "max_skills",
            "max_loop_iterations",
            "max_instruction_chars",
            "max_total_instruction_chars",
            "max_skill_size_bytes",
        ):
            self.assertIn(key, limits["not_locally_checkable"])
            self.assertIn("KAGGLE-27", limits["not_locally_checkable_citations"][key])
            self.assertNotIn(key, limits["enforced_limits"] if "enforced_limits" in limits else {})

    def test_official_builder_is_still_blocked_when_absent(self) -> None:
        from v3.common.errors import FailClosed

        with self.assertRaises(FailClosed) as ctx:
            submit.official_submission_limits()
        self.assertEqual(ctx.exception.code, "official_compiler_unavailable")


class EvidenceFileShapeTests(unittest.TestCase):
    """直接读 KAGGLE-27 A 段证据的冻结副本，逐位核对阈值（这是"出处可离线复核"的证据）。"""

    EVIDENCE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "docs",
        "v3",
        "design",
        "kaggle-27-a2-limits-evidence.json",
    )

    def test_frozen_evidence_copy_is_present(self) -> None:
        self.assertTrue(
            os.path.exists(self.EVIDENCE_PATH),
            "缺少 KAGGLE-27 A 段证据冻结副本：%s" % self.EVIDENCE_PATH,
        )

    def test_extracted_evidence_matches(self) -> None:
        path = os.environ.get("KAGGLE27_A_EVIDENCE") or self.EVIDENCE_PATH
        with open(path, "r", encoding="utf-8-sig") as handle:
            evidence = json.load(handle)
        limits = evidence["a2_limits"]
        structural = limits["built_limits_structural"]
        for key, value in EVIDENCE_BUILT_LIMITS_STRUCTURAL.items():
            self.assertEqual(structural[key], value, key)
        self.assertEqual(sorted(limits["built_limits_extensions"]), sorted(EVIDENCE_EXTENSIONS))
        self.assertEqual(
            limits["built_limits_adapter_extensions"], EVIDENCE_ADAPTER_EXTENSIONS
        )
        self.assertEqual(limits["swegemma_config_extensions"], limits["built_limits_extensions"])
        # 库默认值与官方口径确实不同（29 vs 7 / 6 vs 1）
        self.assertEqual(limits["adk_default_extension_count"], 29)
        self.assertEqual(len(limits["adk_default_adapter_extensions"]), 6)
        # serving 侧载体版本（Y14 锁里引用的就是这两个）
        self.assertEqual(limits["adk_submission_version"], "0.2.12")
        self.assertEqual(limits["swegemma_version"], "0.2.7")
