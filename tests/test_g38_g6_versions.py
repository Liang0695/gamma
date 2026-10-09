"""KAGGLE-38 G6 回归：训练依赖锁的精确版本 / wheel SHA 与依赖审计结论。

全部离线：取数一律走**注入的取数器**或**本地元数据目录**，测试进程不联网、
不下载 wheel、不安装任何包。真实联网取回的 SHA 不写进断言（那属于执行证据，
见 `docs/v3/evidence/kaggle-38-train-lock-resolved.json`）；这里只核对仓库内的
几份产物是否**互相一致**（锁 ↔ 再生证据 ↔ 官方 wheelhouse 物料记录 ↔ 推理面锁 ↔
完整传递依赖清单）。

r3 形状变更（Mika 裁定）：被审计排除的包**移出 `packages`**，放到锁顶层的
`excluded_packages` 审计块。理由是这个锁的 `packages` 段同时被
`v3/t0/deps.py::DependencyLock` 当作**安装集合**消费，而 `deps.py` 在
`v3/data/g2_integrity.py` 的固定执行闭包里、不许原地改动。移出后：

- 安装集合（`packages`）里**没有** TRL、**没有**任何 `PENDING` → `validate()` 为空；
- `verified=false` 仍然是唯一的 fail-closed 闸门（`verify()` 抛 `UnverifiedLock`）；
- 排除结论仍然带 reason / audit / evidence / `reintroduce_if`，不是静默删除。

钉住六件事：

1. 真实 `v3/locks/train.lock.json` 的 `packages` 里**不得**有任何
   `version` / `wheel_sha256` 是 `PENDING`，也**不得**出现 `trl`；
2. 安装集里的包，`wheel_sha256` 必须是 64 位小写 hex，且都指着一个具体 wheel 文件名；
3. `verified` 在锁里、在再生证据文件里都仍然是 `false`（本轮不得偷偷放行）；
4. 解析器把顶层 `excluded_packages` 报成 `excluded`（带审计理由），**不**计入 pending，
   且**不**为它联网取数；缺 `reason` 的审计块直接报错（不许静默排除）；
5. `safetensors` 的既存冲突已按 Mika 授权修正：版本满足 `transformers 5.17.0` 的
   `requires_dist` 下界，且训练面修 pin **不**动 serving 锁；
6. 完整传递依赖清单（`train_stack_manifest`）自洽：条目数、总字节数、
   每个条目都有精确版本/wheel 文件名/64 位 hex SHA/来源/平台标签，
   并且清单里同样没有 `trl`、没有 `PENDING`。
"""

from __future__ import annotations

import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests._tmp import temp_dir_outside_repo  # noqa: E402

from tools import resolve_train_lock_shas as lock_tool  # noqa: E402

TRAIN_LOCK = os.path.join(REPO_ROOT, "v3", "locks", "train.lock.json")
SERVING_LOCK = os.path.join(REPO_ROOT, "v3", "locks", "serving.lock.json")
EVIDENCE = os.path.join(
    REPO_ROOT, "docs", "v3", "evidence", "kaggle-38-train-lock-resolved.json"
)
STACK_MANIFEST = os.path.join(
    REPO_ROOT, "docs", "v3", "evidence", "kaggle-38-train-stack-manifest.json"
)
WHEELHOUSE_MATERIAL = os.path.join(
    REPO_ROOT, "docs", "v3", "design", "kaggle-27-wheelhouse-material-sha256.json"
)

SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
#: 未固定版本的字面量：**直接取自解析器**，避免两处口径漂移。
PENDING_LITERALS = tuple(sorted(lock_tool.UNPINNED))
#: r3 起被审计排除的包（必须**不在** `packages` 安装集合里）。
EXCLUDED_BY_AUDIT = ("trl",)


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _excluded_blocks(lock: dict) -> dict:
    """锁里**所有**位置的排除审计块：顶层 `excluded_packages` + 旧形态的内联 `excluded`。"""
    blocks: dict = {}
    top = lock.get("excluded_packages")
    if isinstance(top, dict):
        for name, block in top.items():
            blocks[name] = block
    for name, spec in (lock.get("packages") or {}).items():
        if isinstance(spec, dict) and isinstance(spec.get("excluded"), dict):
            blocks[name] = spec["excluded"]
    return blocks


class TrainLockPinningTests(unittest.TestCase):
    """真实锁：安装集合里没有未说明的 PENDING，也没有被排除的包。"""

    def setUp(self) -> None:
        self.lock = _load(TRAIN_LOCK)

    def test_no_pending_in_the_install_set(self) -> None:
        unexplained: list[str] = []
        for name, spec in self.lock["packages"].items():
            for field in ("version", "wheel_sha256"):
                raw = str(spec.get(field) or "").strip().lower()
                if raw in PENDING_LITERALS or raw in ("pending", "null"):
                    unexplained.append("%s.%s=%r" % (name, field, spec.get(field)))
        self.assertEqual(
            unexplained,
            [],
            "安装集合里仍有未固定的字段（G6 要求：packages 全部精确固定）：%s" % unexplained,
        )

    def test_excluded_packages_are_not_in_the_install_set(self) -> None:
        """r3 裁定：被审计排除的包必须在**顶层** `excluded_packages`，不在 `packages`。"""
        for name in EXCLUDED_BY_AUDIT:
            self.assertNotIn(
                name,
                self.lock["packages"],
                "%s 仍在安装集合 packages 里：deps.py 会把它当安装目标（Mika 裁定要移出）"
                % name,
            )
            self.assertIn(
                name,
                self.lock.get("excluded_packages") or {},
                "%s 必须留在顶层 excluded_packages 的审计块里（不许静默删除）" % name,
            )

    def test_kept_packages_have_64_lowercase_hex(self) -> None:
        kept = 0
        for name, spec in sorted(self.lock["packages"].items()):
            kept += 1
            digest = str(spec.get("wheel_sha256") or "")
            self.assertRegex(
                digest, SHA256_HEX, "包 %s 的 wheel_sha256 不是 64 位小写 hex：%r" % (name, digest)
            )
        self.assertGreaterEqual(kept, 6, "安装集不该缩到 6 个包以下")

    def test_kept_packages_record_a_concrete_wheel_file(self) -> None:
        """每个 kept 包都要能指着**一个具体 wheel 文件名**，不许含糊。"""
        for name, spec in sorted(self.lock["packages"].items()):
            filename = str(spec.get("wheel_filename") or "")
            self.assertTrue(filename.endswith(".whl"), "包 %s 缺 wheel_filename" % name)
            self.assertNotIn("latest", filename.lower(), "包 %s 指向 latest 式文件名" % name)

    def test_excluded_entry_records_a_real_audit_and_stays_fail_closed(self) -> None:
        blocks = _excluded_blocks(self.lock)
        self.assertTrue(blocks, "本轮至少有一个包走依赖审计（excluded），不该全为空")
        for name, block in blocks.items():
            self.assertTrue(str(block.get("reason") or "").strip(), "%s 缺 reason" % name)
            self.assertTrue(block.get("audit"), "%s 缺 audit 明细" % name)
            self.assertTrue(block.get("evidence"), "%s 缺 evidence 出处" % name)
            self.assertTrue(block.get("reintroduce_if"), "%s 缺 reintroduce_if" % name)
            # 顶层审计块**故意**保持 PENDING：把 excluded_packages 当安装集读的
            # 朴素消费者仍然 fail-closed，不会把"已排除"误读成"已固定版本"。
            self.assertEqual(str(block.get("version")).upper(), "PENDING")
            self.assertEqual(str(block.get("wheel_sha256")).upper(), "PENDING")

    def test_install_set_passes_the_frozen_dependency_lock_validator(self) -> None:
        """不改冻结的 `deps.py`，靠"把 trl 移出 packages"让 `validate()` 变干净。"""
        import importlib.util

        try:
            from v3.t0.deps import DependencyLock  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - 环境相关
            self.skipTest("无法导入 v3.t0.deps：%s" % exc)
        lock = DependencyLock.from_file(TRAIN_LOCK)
        self.assertEqual(lock.validate(), [], "安装集合必须没有任何 validate 问题")
        # 但 fail-closed 一点没松：verified=false 仍然阻断。
        from v3.common.errors import UnverifiedLock  # noqa: PLC0415

        with self.assertRaises(UnverifiedLock) as ctx:
            lock.verify()
        self.assertEqual(ctx.exception.code, "dependency_lock_unverified")


class SafetensorsConflictTests(unittest.TestCase):
    """Mika 授权修正训练面 `safetensors` 既存 pin：满足 transformers 的下界，且不动 serving。"""

    def setUp(self) -> None:
        self.lock = _load(TRAIN_LOCK)
        self.entry = self.lock["packages"]["safetensors"]

    def test_version_satisfies_transformers_lower_bound(self) -> None:
        basis = self.entry.get("compatibility") or {}
        intersection = basis.get("constraint_intersection") or {}
        transformers = intersection.get("transformers 5.17.0") or {}
        constraint = str(transformers.get("constraint_string") or "")
        self.assertIn(
            "safetensors>=", constraint, "必须记录 transformers 的 requires_dist 依据"
        )
        match = re.search(r"safetensors>=([0-9][0-9.]*)", constraint)
        self.assertIsNotNone(match, "requires_dist 里读不出下界：%r" % constraint)
        floor = tuple(int(part) for part in match.group(1).split("."))
        actual = tuple(int(part) for part in str(self.entry["version"]).split("."))
        self.assertGreaterEqual(
            actual, floor, "锁里的 safetensors %s 低于 transformers 要求的 %s" % (actual, floor)
        )
        # 依据必须能复核：URL + 字段名。
        self.assertIn("pypi.org/pypi/transformers/", str(transformers.get("url") or ""))
        self.assertEqual(transformers.get("field"), "info.requires_dist")

    def test_pin_is_a_concrete_wheel_with_sha(self) -> None:
        self.assertTrue(str(self.entry.get("wheel_filename") or "").endswith(".whl"))
        self.assertRegex(str(self.entry.get("wheel_sha256") or ""), SHA256_HEX)
        self.assertNotIn("latest", str(self.entry.get("version")).lower())

    def test_old_conflicting_pin_is_recorded_as_superseded(self) -> None:
        note = str(self.entry.get("note") or "")
        self.assertIn("0.6.2", note, "必须记下旧值，审计才知道改了什么")
        self.assertIn("safetensors>=0.8.0", note, "必须写明冲突来源")

    def test_serving_lock_was_not_touched(self) -> None:
        """训练面修 pin **不得**顺带改推理面。"""
        serving = _load(SERVING_LOCK).get("packages") or {}
        self.assertNotIn(
            "safetensors",
            serving,
            "推理面锁里出现了 safetensors：本轮授权只修训练面，不该顺带改 serving",
        )
        # 推理面真正共用的那个包（compressed-tensors）必须与训练面逐位一致。
        self.assertEqual(
            serving["compressed-tensors"]["wheel_sha256"],
            self.lock["packages"]["compressed-tensors"]["wheel_sha256"],
        )


class TrainStackManifestTests(unittest.TestCase):
    """完整传递依赖清单：不是"只锁七个顶层包"。"""

    def setUp(self) -> None:
        self.lock = _load(TRAIN_LOCK)
        self.manifest = _load(STACK_MANIFEST)

    def test_lock_points_at_the_manifest(self) -> None:
        pointer = self.lock.get("train_stack_manifest") or {}
        self.assertEqual(
            str(pointer.get("path") or "").replace("\\", "/"),
            "docs/v3/evidence/kaggle-38-train-stack-manifest.json",
        )
        self.assertGreaterEqual(int(pointer.get("entry_count") or 0), 7)

    def test_manifest_targets_the_isolated_interpreter(self) -> None:
        target = self.manifest["target"]
        self.assertEqual(target["interpreter"], "/home/scc/pb24511961/v3/envs/py3119/bin/python")
        self.assertEqual(target["python_version"], "3.11.9")
        self.assertIn("--require-hashes", target["install_flags"])
        self.assertIn("--only-binary=:all:", target["install_flags"])

    def test_every_entry_is_fully_pinned(self) -> None:
        entries = self.manifest["entries"]
        self.assertGreater(len(entries), 6, "传递闭包不能只有顶层包")
        top_level = set(self.lock["packages"])
        names = {str(item["name"]) for item in entries}
        self.assertTrue(
            top_level.issubset(names), "清单必须覆盖全部顶层包：缺 %s" % sorted(top_level - names)
        )
        bad: list[str] = []
        for item in entries:
            digest = str(item.get("sha256") or "")
            if not SHA256_HEX.match(digest):
                bad.append("%s.sha256=%r" % (item.get("name"), digest))
            if not str(item.get("wheel_filename") or "").endswith(".whl"):
                bad.append("%s.wheel_filename=%r" % (item.get("name"), item.get("wheel_filename")))
            if str(item.get("version") or "").strip().lower() in PENDING_LITERALS:
                bad.append("%s.version=%r" % (item.get("name"), item.get("version")))
            if not str(item.get("source") or "").startswith("http"):
                bad.append("%s.source=%r" % (item.get("name"), item.get("source")))
            if int(item.get("size_bytes") or 0) <= 0:
                bad.append("%s.size_bytes=%r" % (item.get("name"), item.get("size_bytes")))
            if not str(item.get("tag") or "").strip():
                bad.append("%s.tag=%r" % (item.get("name"), item.get("tag")))
        self.assertEqual(bad, [], "清单里有没钉死的条目：%s" % bad[:10])

    def test_pure_python_wheels_are_allowed_and_present(self) -> None:
        """Mika：允许 py3-none-any，不能只收 cp311/abi3 manylinux 轮子。"""
        tags = {str(item.get("tag") or "") for item in self.manifest["entries"]}
        self.assertIn("py3-none-any", tags, "清单里一个纯 Python 轮子都没有：筛选口径过窄")
        self.assertTrue(
            any("manylinux" in value for value in tags), "缺少 manylinux 平台标签"
        )
        self.assertTrue(all(value for value in tags), "条目缺平台标签")

    def test_cuda_runtime_wheels_are_covered(self) -> None:
        names = {str(item["name"]) for item in self.manifest["entries"]}
        cuda = sorted(name for name in names if name.startswith("nvidia-") or name.startswith("cuda-"))
        self.assertTrue(cuda, "torch 2.10.0 在 Linux 上要求的 CUDA 运行时轮子没有被列进清单")

    def test_manifest_has_no_trl_and_no_pending(self) -> None:
        names = {str(item["name"]) for item in self.manifest["entries"]}
        for name in EXCLUDED_BY_AUDIT:
            self.assertNotIn(name, names, "%s 已被审计排除，不该出现在安装清单里" % name)
        self.assertEqual(
            [item for item in self.manifest["entries"] if item["version"] in PENDING_LITERALS],
            [],
        )

    def test_total_bytes_matches_the_sum_of_entries(self) -> None:
        pointer = self.lock["train_stack_manifest"]
        total = int(pointer["total_download_bytes"])
        self.assertEqual(
            total,
            sum(int(item["size_bytes"]) for item in self.manifest["entries"]),
            "总下载体积与逐条之和不一致",
        )
        self.assertIn("estimates", self.manifest)
        self.assertTrue(self.manifest["estimates"].get("caveats"), "估算必须带不确定性说明")


class CompressedTensorsBasisTests(unittest.TestCase):
    """compressed-tensors 的 pin 必须与官方 wheelhouse 物料、推理面锁**互相一致**。"""

    def setUp(self) -> None:
        self.lock = _load(TRAIN_LOCK)
        self.entry = self.lock["packages"]["compressed-tensors"]

    def test_matches_official_wheelhouse_material(self) -> None:
        """PyPI 的 wheel SHA 与官方 wheelhouse 物料记录逐位相同（离线交叉核对）。"""
        material = {
            item["filename"]: item for item in _load(WHEELHOUSE_MATERIAL)["wheels"]
        }
        filename = self.entry["wheel_filename"]
        self.assertIn(filename, material, "锁里的 wheel 不在官方 wheelhouse 物料清单里")
        self.assertEqual(material[filename]["sha256"], self.entry["wheel_sha256"])

    def test_train_and_serving_pin_the_same_quantization_runtime(self) -> None:
        """训练面不得为了加载基座升级评分器：两侧的 CT 版本/SHA 必须一致。"""
        serving = _load(SERVING_LOCK)["packages"]["compressed-tensors"]
        self.assertEqual(serving["version"], self.entry["version"])
        self.assertEqual(serving["wheel_sha256"], self.entry["wheel_sha256"])

    def test_compatibility_basis_is_recorded(self) -> None:
        """兼容依据必须是**可复核的公开字段**，不是一句"应该能用"。"""
        basis = self.entry["compatibility"]
        self.assertIn("pypi.org/pypi/compressed-tensors/", basis["evidence_url"])
        requires = " ".join(basis["requires_dist"])
        self.assertIn("torch>=", requires)
        self.assertIn("transformers>=", requires)
        self.assertEqual(
            basis["satisfied_by"],
            {
                "torch": self.lock["packages"]["torch"]["version"],
                "transformers": self.lock["packages"]["transformers"]["version"],
            },
        )


class VerifiedStaysFalseTests(unittest.TestCase):
    """本轮**没有**把 verified 置真：锁与再生证据都必须仍是 false。"""

    def test_lock_verified_is_false(self) -> None:
        self.assertFalse(_load(TRAIN_LOCK)["verified"], "train.lock.json 的 verified 不得置真")

    def test_evidence_verified_is_false(self) -> None:
        evidence = _load(EVIDENCE)
        self.assertFalse(evidence["verified"], "再生证据文件的 verified 不得置真")
        self.assertIn("pip download --require-hashes", evidence["verified_note"])

    def test_evidence_has_no_pending_left(self) -> None:
        evidence = _load(EVIDENCE)
        self.assertEqual(evidence["summary"]["pending"], 0)
        self.assertEqual(evidence["summary"]["pending_packages"], [])
        for name, item in evidence["packages"].items():
            self.assertIn(item["status"], ("resolved", "excluded"), "%s 仍是挂起状态" % name)

    def test_lock_and_evidence_agree_on_every_sha(self) -> None:
        lock = _load(TRAIN_LOCK)
        evidence = _load(EVIDENCE)
        self.assertEqual(
            sorted(lock["packages"]), sorted(evidence["packages"]), "锁与证据的安装集合必须一致"
        )
        self.assertEqual(
            sorted(lock.get("excluded_packages") or {}),
            sorted(evidence.get("excluded_packages") or {}),
            "锁与证据的排除集合必须一致",
        )
        for name, spec in sorted(lock["packages"].items()):
            item = evidence["packages"][name]
            self.assertEqual(item["status"], "resolved")
            self.assertEqual(item["wheel"]["sha256"], spec["wheel_sha256"], name)
            self.assertEqual(item["wheel"]["filename"], spec["wheel_filename"], name)
        for name in (lock.get("excluded_packages") or {}):
            self.assertNotIn(name, evidence["packages"], "%s 不该出现在安装集合里" % name)
            item = evidence["excluded_packages"][name]
            self.assertEqual(item["status"], "excluded")
            self.assertTrue(str(item.get("exclusion_reason") or "").strip())


class ResolverExclusionTests(unittest.TestCase):
    """解析器：`excluded` 报成 `excluded` 且不算 pending；缺 reason 直接报错。"""

    TARGET = dict(lock_tool.DEFAULT_TARGET)

    def _wheel(self, filename: str = "demo-1.0-py3-none-any.whl", sha: str = "a" * 64) -> dict:
        return {
            "filename": filename,
            "digests": {"sha256": sha},
            "size": 1,
            "url": "https://example.invalid/%s" % filename,
            "yanked": False,
        }

    def _payload(self, files: list[dict], version: str = "1.0") -> dict:
        return {"info": {"version": version}, "releases": {version: files}}

    def _lock(self) -> dict:
        """r3 形状：被排除的包在**顶层** `excluded_packages`，不在 `packages` 安装集合里。"""
        return {
            "channel": "train",
            "python": "3.11.9",
            "packages": {
                "torch": {"source": "pypi", "version": "2.10.0", "wheel_sha256": "PENDING"},
            },
            "excluded_packages": {
                "trl": {
                    "source": "excluded",
                    "version": "PENDING",
                    "wheel_sha256": "PENDING",
                    "reason": "not_a_dependency_of_the_training_path",
                    "audit": ["v3/ 下零命中"],
                    "evidence": ["https://pypi.org/pypi/trl/json"],
                    "reintroduce_if": "任何训练代码 import trl 时重新引入并 pin",
                },
            },
            "verified": False,
        }

    def test_excluded_is_reported_not_pending(self) -> None:
        seen: list[str] = []

        def fetcher(url: str) -> dict:
            seen.append(url)
            return self._payload([self._wheel("torch-2.10.0-cp311-cp311-manylinux_2_28_x86_64.whl", "b" * 64)], "2.10.0")

        result = lock_tool.resolve_lock(self._lock(), fetcher=fetcher)
        self.assertNotIn("trl", result["packages"], "排除的包不得混进安装集合")
        item = result["excluded_packages"]["trl"]
        self.assertEqual(item["status"], "excluded")
        self.assertEqual(item["exclusion_reason"], "not_a_dependency_of_the_training_path")
        self.assertEqual(item["exclusion_audit"]["audit"], ["v3/ 下零命中"])
        self.assertIsNone(item["pending_reason"], "excluded 不是 pending，不得挂 pending_reason")
        self.assertEqual(result["summary"]["pending"], 0)
        self.assertEqual(result["summary"]["pending_packages"], [])
        self.assertEqual(result["summary"]["excluded"], 1)
        self.assertEqual(result["summary"]["excluded_packages"], ["trl"])
        self.assertEqual(result["summary"]["excluded_from_top_level"], ["trl"])
        self.assertEqual(result["summary"]["resolved"], 1)
        self.assertEqual(result["summary"]["resolved_packages"], ["torch"])
        self.assertFalse(result["verified"])
        lock_tool.assert_no_pending_without_reason(result)
        self.assertFalse(
            [url for url in seen if "trl" in url], "excluded 条目不该再去联网取数：%s" % seen
        )

    def test_excluded_without_reason_is_a_bug(self) -> None:
        with self.assertRaises(AssertionError):
            lock_tool.resolve_excluded_package("x", {"audit": ["没有 reason"]})
        with self.assertRaises(AssertionError):
            lock_tool.resolve_lock(
                {
                    "channel": "train",
                    "python": "3.11.9",
                    "packages": {},
                    "excluded_packages": {"x": {"audit": ["没有 reason"]}},
                },
                fetcher=lambda url: {},
            )

    def test_assert_selfcheck_covers_both_kinds(self) -> None:
        with self.assertRaises(AssertionError):
            lock_tool.assert_no_pending_without_reason(
                {"packages": {"x": {"status": "pending", "pending_reason": None}}}
            )
        with self.assertRaises(AssertionError):
            lock_tool.assert_no_pending_without_reason(
                {"packages": {"y": {"status": "excluded", "exclusion_reason": ""}}}
            )
        lock_tool.assert_no_pending_without_reason(
            {"packages": {"z": {"status": "excluded", "exclusion_reason": "audited-out"}}}
        )

    def test_real_lock_has_no_pending_and_reports_trl_excluded(self) -> None:
        """对**仓库里真实的**锁跑一遍（注入取数器按 URL 里的版本造元数据，不联网）。"""
        seen: list[str] = []

        def fetcher(url: str) -> dict:
            seen.append(url)
            found = re.search(r"/pypi/(?P<name>[^/]+)/(?P<version>[^/]+)/json$", url)
            assert found, "取数器只接受带版本号的 PyPI URL：%s" % url
            version = found.group("version")
            return self._payload(
                [self._wheel("pkg-%s-cp311-cp311-manylinux_2_28_x86_64.whl" % version, "c" * 64)],
                version,
            )

        result = lock_tool.resolve_lock(_load(TRAIN_LOCK), fetcher=fetcher)
        self.assertEqual(result["summary"]["pending"], 0)
        self.assertEqual(result["summary"]["excluded_packages"], ["trl"])
        self.assertEqual(result["summary"]["excluded_from_top_level"], ["trl"])
        self.assertEqual(result["summary"]["excluded_inline_in_packages"], [])
        self.assertNotIn("trl", result["packages"])
        self.assertEqual(
            result["summary"]["resolved"] + result["summary"]["excluded"],
            len(_load(TRAIN_LOCK)["packages"]) + len(_load(TRAIN_LOCK)["excluded_packages"]),
        )
        self.assertTrue(result["excluded_packages"]["trl"]["exclusion_reason"])
        self.assertFalse(result["verified"])
        self.assertFalse(
            [url for url in seen if "trl" in url], "excluded 条目不该联网取数：%s" % seen
        )

    def test_offline_metadata_dir_still_reports_excluded(self) -> None:
        """离线元数据目录模式：excluded 结论不依赖任何本地缓存文件。"""
        with temp_dir_outside_repo("g38_g6_cache_") as cache:
            with open(os.path.join(cache, "torch.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    self._payload(
                        [self._wheel(
                            "torch-2.10.0-cp311-cp311-manylinux_2_28_x86_64.whl", "d" * 64
                        )], "2.10.0"
                    ),
                    handle,
                )
            result = lock_tool.resolve_lock(
                self._lock(), metadata_dir=cache, fetcher=lambda url: {}
            )
        self.assertEqual(result["packages"]["torch"]["status"], "resolved")
        self.assertEqual(result["packages"]["torch"]["wheel"]["sha256"], "d" * 64)
        self.assertEqual(result["excluded_packages"]["trl"]["status"], "excluded")
        self.assertEqual(result["summary"]["pending"], 0)

    def test_a_package_cannot_be_in_both_places(self) -> None:
        """同一个包不能既在安装集合、又在排除集合里。"""
        lock = self._lock()
        lock["packages"]["trl"] = {"source": "pypi", "version": "1.0", "wheel_sha256": "e" * 64}
        with self.assertRaises(AssertionError):
            lock_tool.resolve_lock(lock, fetcher=lambda url: {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
