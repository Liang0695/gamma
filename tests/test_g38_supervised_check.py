"""KAGGLE-38 r5：v6 监督器接线（独审阻断 6）的 CPU 回归。

覆盖 Mika 点名的三类正反例：

1. **卡住子进程被 TERM/KILL**（真跑 v6 监督器，卡住的 main 必须在总预算内被杀掉）；
2. **批准上下文被篡改被拒**（外部锚不符 / 目标版本不符 / GPU-h 超额）；
3. **截止缺失或被放宽被拒**（缺文件、缺自锚、锚不符、CLI 请求更晚、环境里重新注入通道）。

## 本机平台限制（实测，如实登记）

本机（Windows + DSH 沙箱）实测：**v6 监督器的子进程无法访问工作区文件**
（`python <脚本>` 退出码 2 = 打不开脚本；`-c` 里 `open(工作区文件)` 抛异常退出 1），
而同一命令由本进程直接启动则正常。因此：

- 需要"监督器子进程真的写盘"的用例走 `WriteThroughSupervisorTests`，
  由 `child_can_touch_workspace()` 运行时探测决定 `skip`（在 Linux/107 上会执行）；
- 三类必需反例**不依赖**该能力：终止时序只用内存里的 `-c` 子进程，
  截止/批准拒绝用子进程 CLI 的**直接**调用（不经过监督器）。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests._tmp import temp_dir_outside_repo  # noqa: E402

from v3.common.errors import FailClosed, IntegrityError, MissingInput, PolicyViolation  # noqa: E402
from v3.train import engineering_check as ec  # noqa: E402
from v3.train import supervised_check as sc  # noqa: E402

V6_DIR = os.path.join(REPO_ROOT, "tools", "v6")
#: 仓库里被核验过的官方小文件目录（与 r4 真实模式用例同源）。
EVIDENCE_DIR = os.path.join(REPO_ROOT, "docs", "v3", "evidence", "kaggle-38-g9-evidence")
INTERFACE_LOCK = os.path.join(REPO_ROOT, "v3", "locks", "official-interface.json")
HASHES = {key: ("%x" % (index + 1)) * 64 for index, key in enumerate(
    ("source_sha256", "data_sha256", "config_sha256", "code_sha256", "deps_sha256")
)}
APPROVAL_PIN = "128d9b98b05ddf128c2e77b599e078de65675b8a"


def child_can_touch_workspace() -> dict:
    """运行时探测：v6 监督器的子进程能否读到工作区里的文件。

    做法：让监督器跑一个 `-c` 子进程去读一个刚写好的一行文件；
    读到 → 退出 70，读不到（异常）→ 非 0 但不是 70。
    """
    with temp_dir_outside_repo("g38r5_reach_") as root:
        probe_file = os.path.join(root, "probe.txt")
        with open(probe_file, "w", encoding="utf-8") as handle:
            handle.write("reachable\n")
        probe = [
            "import sys",
            "p=%r" % probe_file,
            "sys.exit(70 if open(p).read().strip()=='reachable' else 71)",
        ]
        phases = [
            {"name": "prep", "kind": "prep", "command": [sys.executable, "-c", ";".join(probe)]},
            {"name": "main", "kind": "main", "command": [sys.executable, "-c", "print('ok')"]},
            {"name": "wrapup", "kind": "wrapup", "command": [sys.executable, "-c", "print('ok')"]},
        ]
        report = sc.run_supervised_check(
            v6_dir=V6_DIR,
            dest_dir=os.path.join(root, "dest"),
            run_root=os.path.join(root, "runs"),
            approved_budget_seconds=20.0,
            fixture_phases=phases,
        )
        prep = next(
            (item for item in report.get("phases") or [] if item.get("phase") == "prep"), {}
        )
        return {
            "reachable": prep.get("exit_code") == 70,
            "prep_exit_code": prep.get("exit_code"),
            "platform": sys.platform,
            "evidence": "监督器子进程读工作区文件：70=读到（有访问权），其它=被拒",
        }


# ------------------------------------------------------------------ v6 包核对


class V6PackageTests(unittest.TestCase):
    """复用 = 逐字节核对 v6 原件，而不是拿一个同名改写版当监督器。"""

    def test_repository_copy_is_the_pinned_v6_package(self) -> None:
        report = sc.verify_v6_package(V6_DIR)
        self.assertEqual(
            sorted(report["verified_files"]), sorted(sc.V6_REQUIRED_FILES)
        )
        self.assertEqual(
            report["verified_files"]["shard_supervisor.py"],
            sc.V6_REQUIRED_FILES["shard_supervisor.py"],
        )
        self.assertEqual(report["optional_files"].get("deadline_wrapper.py"),
                         sc.V6_OPTIONAL_FILES["deadline_wrapper.py"])

    def test_tampered_supervisor_byte_is_rejected(self) -> None:
        with temp_dir_outside_repo("g38r5_v6_tamper_") as root:
            for name in sc.V6_REQUIRED_FILES:
                shutil.copy(os.path.join(V6_DIR, name), os.path.join(root, name))
            target = os.path.join(root, "shard_guard.py")
            with open(target, "rb") as handle:
                raw = handle.read()
            with open(target, "wb") as handle:
                handle.write(raw.replace(b"PIN_COMMIT", b"pin_commit", 1))
            with self.assertRaises(IntegrityError) as ctx:
                sc.verify_v6_package(root)
        self.assertEqual(ctx.exception.code, "v6_package_tampered")
        self.assertTrue(any("shard_guard.py" in item for item in ctx.exception.context["problems"]))

    def test_missing_package_is_refused(self) -> None:
        with self.assertRaises(MissingInput) as ctx:
            sc.verify_v6_package(os.path.join(REPO_ROOT, "tools", "definitely-not-here"))
        self.assertEqual(ctx.exception.code, "v6_supervisor_missing")


# ------------------------------------------------------------------ 预算与截止


class BudgetTests(unittest.TestCase):
    """`min(批准预算, 外层截止)` —— 只能收紧，不能放宽，两者都缺即拒绝。"""

    def test_earlier_bound_wins(self) -> None:
        now_epoch, now_mono = 1_000_000.0, 5_000.0
        approved = sc.compute_budget(
            approved_budget_seconds=600.0,
            outer_deadline_epoch=now_epoch + 120.0,
            now_epoch=now_epoch,
            now_monotonic=now_mono,
        )
        self.assertAlmostEqual(approved["budget_seconds"], 120.0, places=6)
        self.assertEqual(approved["binding_source"], "outer-deadline")
        self.assertAlmostEqual(approved["deadline_monotonic"], now_mono + 120.0, places=6)

    def test_approved_budget_wins_when_tighter(self) -> None:
        now_epoch, now_mono = 2_000_000.0, 900.0
        result = sc.compute_budget(
            approved_budget_seconds=90.0,
            outer_deadline_epoch=now_epoch + 3600.0,
            now_epoch=now_epoch,
            now_monotonic=now_mono,
        )
        self.assertAlmostEqual(result["budget_seconds"], 90.0, places=6)
        self.assertEqual(result["binding_source"], "approved-budget")

    def test_no_bound_is_refused(self) -> None:
        with self.assertRaises(MissingInput) as ctx:
            sc.compute_budget(approved_budget_seconds=None, outer_deadline_epoch=None)
        self.assertEqual(ctx.exception.code, "supervised_budget_missing")

    def test_zero_or_negative_outer_deadline_is_refused(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            sc.compute_budget(
                approved_budget_seconds=None,
                outer_deadline_epoch=1_000.0,
                now_epoch=1_500.0,
                now_monotonic=10.0,
            )
        self.assertEqual(ctx.exception.code, "supervised_budget_invalid")

    def test_phase_caps_are_reserves_inside_the_budget(self) -> None:
        caps = sc.phase_caps(10.0)
        self.assertLess(caps["prep_cap"], 10.0)
        self.assertLess(caps["wrapup_cap"], 10.0)
        self.assertLess(caps["kill_after"], 10.0)
        self.assertLessEqual(
            caps["prep_cap"] + caps["wrapup_cap"] + caps["kill_after"], 10.0
        )

    def test_strict_positive_rejects_weird_numbers(self) -> None:
        for bad in (None, "", " 1", "+1", "nan", "inf", "-1", "0", "."):
            with self.assertRaises(ValueError):
                sc.strict_positive(bad, "x")
        self.assertEqual(sc.strict_positive("2.5", "x"), 2.5)


class EnvSanitizeTests(unittest.TestCase):
    """子进程环境里不允许残留任何可以用来扩时的通道。"""

    def test_deadline_and_budget_channels_are_stripped(self) -> None:
        env = {
            "PATH": "/usr/bin",
            "V3_DEADLINE_EPOCH": "9999999999",
            "V3_TOTAL_BUDGET_SECONDS": "99999",
            "V3_TIME_BUDGET_SECONDS": "88888",
            "V3_KILL_AFTER": "777",
            "V3_REPO": "/home/x/gamma",
        }
        result = sc.sanitize_child_env(env)
        self.assertEqual(
            result["stripped_keys"],
            ["V3_DEADLINE_EPOCH", "V3_KILL_AFTER", "V3_TIME_BUDGET_SECONDS", "V3_TOTAL_BUDGET_SECONDS"],
        )
        self.assertNotIn("V3_DEADLINE_EPOCH", result["env"])
        self.assertNotIn("V3_TOTAL_BUDGET_SECONDS", result["env"])
        # 非截止类变量原样保留（不能顺手把无关配置也删掉）。
        self.assertEqual(result["env"]["V3_REPO"], "/home/x/gamma")
        self.assertEqual(result["env"]["PATH"], "/usr/bin")

    def test_injected_channel_in_supervised_child_is_rejected(self) -> None:
        supervised = {"deadline_monotonic": 1_000.0, "budget_seconds": 60.0, "source": "v6-supervisor"}
        with self.assertRaises(PolicyViolation) as ctx:
            sc.reject_widening(
                supervised=supervised,
                requested_budget_seconds=None,
                requested_deadline_epoch=None,
                now_epoch=0.0,
                now_monotonic=0.0,
                env={"V3_DEADLINE_EPOCH": "9999", "V3_SUPERVISED": "1"},
            )
        self.assertEqual(ctx.exception.code, "deadline_widening_env_present")
        # 只读的监督通道不算"重新注入"。
        ok = sc.reject_widening(
            supervised=supervised,
            requested_budget_seconds=None,
            requested_deadline_epoch=None,
            now_epoch=0.0,
            now_monotonic=0.0,
            env={
                sc.SUPERVISED_FLAG_ENV: "1",
                sc.CUTOFF_FILE_ENV: "/tmp/x.json",
                sc.CUTOFF_SHA_ENV: "a" * 64,
            },
        )
        self.assertTrue(ok["ok"])


class DeadlineFileTests(unittest.TestCase):
    """父进程签发的截止文件：读它必须带自锚，换掉即发现。"""

    def _payload(self, budget: float = 60.0) -> dict:
        return {
            "source": "v6-supervisor-adapter",
            "budget_seconds": budget,
            "binding_source": "approved-budget",
            "deadline_monotonic": 12345.0 + budget,
            "deadline_epoch": 1_700_000_000.0 + budget,
            "issued_utc": "2026-10-08T00:00:00Z",
            "never_widens": True,
        }

    def test_anchor_is_verified(self) -> None:
        with temp_dir_outside_repo("g38r5_dl_") as root:
            path = os.path.join(root, "deadline.json")
            record = sc.write_deadline_file(path, self._payload())
            loaded = sc.read_deadline_file(path, record["file_sha256"])
        self.assertTrue(loaded["verified_against_anchor"])
        self.assertAlmostEqual(loaded["budget_seconds"], 60.0, places=6)

    def test_wrong_anchor_is_rejected(self) -> None:
        with temp_dir_outside_repo("g38r5_dl_bad_") as root:
            path = os.path.join(root, "deadline.json")
            sc.write_deadline_file(path, self._payload())
            with self.assertRaises(IntegrityError) as ctx:
                sc.read_deadline_file(path, "0" * 64)
        self.assertEqual(ctx.exception.code, "supervised_deadline_tampered")

    def test_relaxed_rewrite_is_detected(self) -> None:
        """把截止往后改（放宽）必须被发现——这正是原来只能靠自报的那个洞。"""
        with temp_dir_outside_repo("g38r5_dl_relax_") as root:
            path = os.path.join(root, "deadline.json")
            record = sc.write_deadline_file(path, self._payload(budget=60.0))
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self._payload(budget=3600.0), handle)
            with self.assertRaises(IntegrityError) as ctx:
                sc.read_deadline_file(path, record["file_sha256"])
        self.assertEqual(ctx.exception.code, "supervised_deadline_tampered")

    def test_missing_file_and_missing_anchor(self) -> None:
        with temp_dir_outside_repo("g38r5_dl_missing_") as root:
            with self.assertRaises(MissingInput) as ctx:
                sc.read_deadline_file(None, "a" * 64)
            self.assertEqual(ctx.exception.code, "supervised_deadline_missing")
            path = os.path.join(root, "deadline.json")
            record = sc.write_deadline_file(path, self._payload())
            with self.assertRaises(MissingInput) as ctx2:
                sc.read_deadline_file(path, None)
            self.assertEqual(ctx2.exception.code, "supervised_deadline_anchor_missing")
            with self.assertRaises(MissingInput) as ctx3:
                sc.read_deadline_file(os.path.join(root, "nope.json"), record["file_sha256"])
            self.assertEqual(ctx3.exception.code, "supervised_deadline_missing")


class WideningRefusalTests(unittest.TestCase):
    """请求更晚的截止 = 放宽 = 显式拒绝（不是静默夹紧）。"""

    SUPERVISED = {"deadline_monotonic": 1_000.0, "budget_seconds": 60.0, "source": "v6-supervisor"}

    def test_later_budget_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            sc.reject_widening(
                supervised=self.SUPERVISED,
                requested_budget_seconds=6000.0,
                requested_deadline_epoch=None,
                now_epoch=0.0,
                now_monotonic=0.0,
                env={},
            )
        self.assertEqual(ctx.exception.code, "deadline_widening_rejected")
        self.assertEqual(ctx.exception.context["requested_source"], "time-budget-seconds")

    def test_later_outer_epoch_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            sc.reject_widening(
                supervised=self.SUPERVISED,
                requested_budget_seconds=None,
                requested_deadline_epoch=999_999.0,
                now_epoch=0.0,
                now_monotonic=0.0,
                env={},
            )
        self.assertEqual(ctx.exception.code, "deadline_widening_rejected")
        self.assertEqual(ctx.exception.context["requested_source"], "deadline-epoch")

    def test_tighter_or_equal_request_is_allowed(self) -> None:
        ok = sc.reject_widening(
            supervised=self.SUPERVISED,
            requested_budget_seconds=30.0,
            requested_deadline_epoch=None,
            now_epoch=0.0,
            now_monotonic=0.0,
            env={},
        )
        self.assertTrue(ok["ok"])
        self.assertFalse(ok["widened"])
        self.assertTrue(ok["requested_tighter"])

    def test_no_request_is_allowed(self) -> None:
        ok = sc.reject_widening(
            supervised=self.SUPERVISED,
            requested_budget_seconds=None,
            requested_deadline_epoch=None,
            now_epoch=0.0,
            now_monotonic=0.0,
            env={},
        )
        self.assertTrue(ok["ok"])
        self.assertIsNone(ok["details"]["requested"])


# ------------------------------------------------------------------ 真跑监督器


class SupervisorOrchestrationTests(unittest.TestCase):
    """真跑 v6 监督器：三段俱在、同一总截止、卡住的子进程被杀掉。"""

    def _print_phases(self) -> list:
        return [
            {"name": "prep", "kind": "prep", "command": [sys.executable, "-c", "print('prep')"]},
            {"name": "main", "kind": "main", "command": [sys.executable, "-c", "print('main')"]},
            {"name": "wrapup", "kind": "wrapup", "command": [sys.executable, "-c", "print('wrapup')"]},
        ]

    def test_three_phases_run_under_one_deadline(self) -> None:
        reach = child_can_touch_workspace()
        with temp_dir_outside_repo("g38r5_orch_") as root:
            report = sc.run_supervised_check(
                v6_dir=V6_DIR,
                dest_dir=os.path.join(root, "dest"),
                run_root=os.path.join(root, "runs"),
                approved_budget_seconds=25.0,
                fixture_phases=self._print_phases(),
            )
            deadline_written = os.path.exists(report["deadline_file"]["path"])
            plan_written = os.path.exists(report["plan_path"])
        self.assertTrue(deadline_written)
        self.assertTrue(plan_written)
        self.assertEqual([item["phase"] for item in report["phases"]], ["prep", "main", "wrapup"])
        self.assertEqual(report["main_exit_code"], 0)
        self.assertEqual(report["wrapup_exit_code"], 0)
        self.assertEqual([item["exit_code"] for item in report["phases"]], [0, 0, 0])
        self.assertTrue(report["budget_respected"])
        self.assertEqual(report["termination"]["terminate_signals_sent"], 0)
        self.assertEqual(report["budget"]["binding_source"], "approved-budget")
        self.assertTrue(report["v6_package"]["verified_files"])
        if reach["reachable"]:
            self.assertEqual(report["overall_exit_code"], 0)
            self.assertNotEqual(report["overall_status"], "audit_failed")
        else:
            # 本机沙箱：监督器的**审计**子进程也要跑脚本文件，因此必然 audit_failed。
            # 这是平台限制（阶段退出码全部为 0 已另行断言），在 Linux/107 上不成立。
            self.assertEqual(report["overall_status"], "audit_failed")

    def test_stuck_child_is_terminated_within_the_total_budget(self) -> None:
        """必需反例 ①：卡住的 main 必须在总预算内被 TERM/KILL，且 wrapup 仍要跑。"""
        with temp_dir_outside_repo("g38r5_stuck_") as root:
            report = sc.run_supervised_check(
                v6_dir=V6_DIR,
                dest_dir=os.path.join(root, "dest"),
                run_root=os.path.join(root, "runs"),
                approved_budget_seconds=8.0,
                fixture_stuck_seconds=60.0,
            )
        self.assertIn(report["main_exit_code"], (124, 137))
        self.assertGreaterEqual(report["termination"]["terminate_signals_sent"], 1)
        self.assertEqual(report["termination"]["phases_terminated"], ["main"])
        main_phase = next(item for item in report["phases"] if item["phase"] == "main")
        self.assertEqual(main_phase["stop_reason"], "stopped_by_term")
        self.assertTrue(main_phase["child_settled"])
        # 外层真的在总截止内停了：不是"事后记一笔超时"。
        self.assertTrue(report["budget_respected"])
        self.assertNotEqual(report["overall_exit_code"], 0)
        # wrapup 仍然执行（v6 的必需收尾）。
        self.assertIn("wrapup", [item["phase"] for item in report["phases"]])

    def test_dry_run_reports_the_exact_command(self) -> None:
        with temp_dir_outside_repo("g38r5_dry_") as root:
            report = sc.run_supervised_check(
                v6_dir=V6_DIR,
                dest_dir=os.path.join(root, "dest"),
                run_root=os.path.join(root, "runs"),
                approved_budget_seconds=15.0,
                dry_run=True,
            )
        self.assertEqual(report["verdict"], "planned")
        self.assertIn("shard_supervisor.py", " ".join(report["supervisor_command"]))
        self.assertIn("--total-budget", report["supervisor_command"])
        self.assertEqual(report["budget"]["binding_source"], "approved-budget")

    def test_stripped_env_keys_are_recorded(self) -> None:
        env = dict(os.environ)
        env["V3_DEADLINE_EPOCH"] = "9999999999"
        env["V3_TOTAL_BUDGET_SECONDS"] = "60000"
        with temp_dir_outside_repo("g38r5_envstrip_") as root:
            report = sc.run_supervised_check(
                v6_dir=V6_DIR,
                dest_dir=os.path.join(root, "dest"),
                run_root=os.path.join(root, "runs"),
                approved_budget_seconds=15.0,
                fixture_phases=self._print_phases(),
                env=env,
            )
        self.assertIn("V3_DEADLINE_EPOCH", report["child_env_stripped_keys"])
        self.assertIn("V3_TOTAL_BUDGET_SECONDS", report["child_env_stripped_keys"])


class WriteThroughSupervisorTests(unittest.TestCase):
    """需要"监督器子进程真的写盘"的端到端用例（本机沙箱下自动跳过）。"""

    def test_child_evidence_is_written_end_to_end(self) -> None:
        probe = child_can_touch_workspace()
        if not probe["reachable"]:
            self.skipTest(
                "本机沙箱拒绝 v6 监督器子进程访问工作区（prep 退出码 %s）；"
                "在 Linux/107 上此用例会执行" % probe["prep_exit_code"]
            )
        with temp_dir_outside_repo("g38r5_e2e_") as root:
            report = sc.run_supervised_check(
                v6_dir=V6_DIR,
                dest_dir=os.path.join(root, "dest"),
                run_root=os.path.join(root, "runs"),
                approved_budget_seconds=120.0,
                steps=6,
                checkpoint_every=3,
                stop_at_step=4,
            )
            evidence = os.path.join(report["run_dir"], "child_evidence.json")
            evidence_exists = os.path.exists(evidence)
            ledger_exists = os.path.exists(report["ledger_path"])
            with open(evidence, encoding="utf-8") as handle:
                child = json.load(handle)
        self.assertEqual(report["overall_exit_code"], 0)
        self.assertEqual(report["overall_status"], "succeeded")
        self.assertTrue(evidence_exists)
        self.assertTrue(ledger_exists)
        self.assertTrue(child["budget"]["supervised_active"])
        self.assertTrue(child["budget"]["supervised"]["effective_deadline_is_supervised"])
        self.assertEqual(child["budget"]["deadline_source"], "v6-supervisor")


# ------------------------------------------------------------------ 子进程 CLI 直接反例


class ChildCliRefusalTests(unittest.TestCase):
    """必需反例 ②③：CLI 直接调用（不经过监督器）时的截止与批准拒绝。"""

    def _run_cli(self, argv: list) -> tuple[int, dict]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            try:
                code = ec.main(argv)
            except SystemExit as exc:  # argparse
                return int(exc.code or 0), {"code": "argparse-rejected"}
        text = buffer.getvalue().strip()
        return code, (json.loads(text) if text else {})

    def _deadline_file(self, root: str, budget: float = 30.0) -> tuple[str, str]:
        """用**真实的单调钟**写一份外层签发的截止（假数字会让子进程立刻判定超时）。"""
        computed = sc.compute_budget(approved_budget_seconds=budget)
        path = os.path.join(root, "deadline.json")
        record = sc.write_deadline_file(
            path,
            {
                "source": "v6-supervisor-adapter",
                "budget_seconds": computed["budget_seconds"],
                "binding_source": computed["binding_source"],
                "deadline_monotonic": computed["deadline_monotonic"],
                "deadline_epoch": computed["deadline_epoch"],
                "issued_utc": "2026-10-08T00:00:00Z",
                "never_widens": True,
            },
        )
        return path, record["file_sha256"]

    def _hashes_file(self, root: str) -> str:
        path = os.path.join(root, "hashes.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(HASHES, handle)
        return path

    def test_supervised_without_deadline_file_is_missing(self) -> None:
        with temp_dir_outside_repo("g38r5_cli_nodl_") as root:
            code, payload = self._run_cli(
                [
                    "--supervised",
                    "--dest-dir", os.path.join(root, "out"),
                    "--steps", "3",
                    "--checkpoint-every", "2",
                    "--stop-at-step", "2",
                ]
            )
        self.assertEqual(payload.get("code"), "supervised_deadline_missing")
        self.assertEqual(code, MissingInput.exit_code)

    def test_widened_cli_budget_is_rejected(self) -> None:
        with temp_dir_outside_repo("g38r5_cli_wide_") as root:
            path, anchor = self._deadline_file(root, budget=30.0)
            code, payload = self._run_cli(
                [
                    "--supervised",
                    "--deadline-file", path,
                    "--deadline-file-sha256", anchor,
                    "--time-budget-seconds", "6000",
                    "--dest-dir", os.path.join(root, "out"),
                    "--steps", "3",
                    "--checkpoint-every", "2",
                    "--stop-at-step", "2",
                ]
            )
        self.assertEqual(payload.get("code"), "deadline_widening_rejected")
        self.assertEqual(code, PolicyViolation.exit_code)

    def test_relaxed_deadline_file_is_rejected(self) -> None:
        with temp_dir_outside_repo("g38r5_cli_relax_") as root:
            path, anchor = self._deadline_file(root, budget=30.0)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "source": "v6-supervisor-adapter",
                        "budget_seconds": 36000.0,
                        "binding_source": "approved-budget",
                        "deadline_monotonic": 999999.0,
                        "deadline_epoch": 1_700_000_000.0 + 36000.0,
                        "issued_utc": "2026-10-08T00:00:00Z",
                    },
                    handle,
                )
            code, payload = self._run_cli(
                [
                    "--supervised",
                    "--deadline-file", path,
                    "--deadline-file-sha256", anchor,
                    "--dest-dir", os.path.join(root, "out"),
                ]
            )
        self.assertEqual(payload.get("code"), "supervised_deadline_tampered")
        self.assertEqual(code, IntegrityError.exit_code)

    def test_injected_deadline_env_is_rejected(self) -> None:
        with temp_dir_outside_repo("g38r5_cli_env_") as root:
            path, anchor = self._deadline_file(root, budget=30.0)
            original = os.environ.get("V3_DEADLINE_EPOCH")
            os.environ["V3_DEADLINE_EPOCH"] = "9999999999"
            try:
                code, payload = self._run_cli(
                    [
                        "--supervised",
                        "--deadline-file", path,
                        "--deadline-file-sha256", anchor,
                        "--dest-dir", os.path.join(root, "out"),
                        "--steps", "3",
                        "--checkpoint-every", "2",
                        "--stop-at-step", "2",
                    ]
                )
            finally:
                if original is None:
                    os.environ.pop("V3_DEADLINE_EPOCH", None)
                else:
                    os.environ["V3_DEADLINE_EPOCH"] = original
        self.assertEqual(payload.get("code"), "deadline_widening_env_present")
        self.assertEqual(code, PolicyViolation.exit_code)

    def test_supervised_child_consumes_the_issued_deadline(self) -> None:
        """正例：受监督子进程真的按**外层签发**的截止运行（不是自报预算）。"""
        with temp_dir_outside_repo("g38r5_cli_ok_") as root:
            path, anchor = self._deadline_file(root, budget=60.0)
            out = os.path.join(root, "child.json")
            code, payload = self._run_cli(
                [
                    "--supervised",
                    "--deadline-file", path,
                    "--deadline-file-sha256", anchor,
                    "--dest-dir", os.path.join(root, "out"),
                    "--steps", "4",
                    "--checkpoint-every", "2",
                    "--stop-at-step", "2",
                    "--out", out,
                ]
            )
            self.assertEqual(code, 0, payload)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
        self.assertEqual(report["budget"]["deadline_source"], "v6-supervisor")
        self.assertTrue(report["budget"]["supervised"]["active"])
        self.assertTrue(report["budget"]["supervised"]["anchor_verified"])
        self.assertTrue(report["budget"]["supervised"]["effective_deadline_is_supervised"])
        self.assertEqual(payload.get("deadline_source"), "v6-supervisor")

    # ---- 批准上下文篡改（真实模式路径，加载权重之前就拒绝） ----

    def _approval(self, root: str, **overrides) -> tuple[str, str]:
        evidence = os.path.join(root, "approval-evidence.json")
        with open(evidence, "w", encoding="utf-8") as handle:
            json.dump({"kind": "engineering-check-evidence", "rev": 1}, handle)
        with open(evidence, "rb") as handle:
            evidence_sha = hashlib.sha256(handle.read()).hexdigest()
        payload = {
            "approval_id": "k38r5-approval",
            "target_sha": APPROVAL_PIN,
            "gpu_hours_approved": 4,
            "verified_by": "test-fixture",
            "verified_utc": "2026-10-08T00:00:00Z",
            "evidence_sha256": evidence_sha,
        }
        payload.update(overrides)
        path = os.path.join(root, "approval.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path, hashlib.sha256(open(path, "rb").read()).hexdigest()

    def _real_mode_argv(self, root: str, approval: str, anchor: str, extra: list | None = None) -> list:
        local_dir = os.path.join(root, "local_weights")
        os.makedirs(local_dir, exist_ok=True)
        # 真实模式的**前置**要求（输入绑定/固定本地目录）都要先满足，
        # 否则会在批准闸门之前就以 MissingInput 停下 —— 那测的就不是批准篡改了。
        layout_path = os.path.join(root, "layout.json")
        with open(layout_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "tokenizer_config.json": "tokenizer_config.json",
                    "tokenizer.json": "tokenizer.json",
                    "chat_template.jinja": "chat_template.jinja",
                    "config.json": "config.json",
                },
                handle,
            )
        argv = [
            "--backend", "torch-peft",
            "--dest-dir", os.path.join(root, "out"),
            "--time-budget-seconds", "120",
            "--stop-at-step", "3",
            "--hashes", self._hashes_file(root),
            "--local-load-authorized",
            "--local-model-dir", local_dir,
            "--model-inputs-root", EVIDENCE_DIR,
            "--model-inputs-layout", layout_path,
            "--interface", INTERFACE_LOCK,
            "--model-id", "google/gemma-4-31B-it-qat-w4a16-ct",
            "--model-revision", "52f3f65bc7a02d555763bc923bd1d9094898219d",
            "--target-modules", "q_proj,o_proj",
            "--approval", approval,
            "--approval-expected-sha256", anchor,
            "--requested-gpu-hours", "4",
        ]
        return argv + list(extra or ())

    def test_tampered_approval_context_is_refused(self) -> None:
        with temp_dir_outside_repo("g38r5_appr_") as root:
            approval, anchor = self._approval(root)
            # 篡改批准内容但保留旧锚 → 外部锚核对失败
            with open(approval, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "approval_id": "k38r5-approval",
                        "target_sha": APPROVAL_PIN,
                        "gpu_hours_approved": 400,
                        "verified_by": "test-fixture",
                        "verified_utc": "2026-10-08T00:00:00Z",
                        "evidence_sha256": "b" * 64,
                    },
                    handle,
                )
            code, payload = self._run_cli(self._real_mode_argv(root, approval, anchor))
        self.assertEqual(code, PolicyViolation.exit_code)
        self.assertIn(
            payload.get("code"),
            ("approval_not_trusted", "approval_external_anchor_mismatch"),
        )

    def test_approval_for_another_commit_is_refused(self) -> None:
        with temp_dir_outside_repo("g38r5_appr_sha_") as root:
            approval, anchor = self._approval(root, target_sha="f" * 40)
            code, payload = self._run_cli(self._real_mode_argv(root, approval, anchor))
        self.assertEqual(code, PolicyViolation.exit_code)
        self.assertIn(
            payload.get("code"), ("approval_target_sha_mismatch", "approval_not_trusted")
        )

    def test_gpu_hours_over_approval_is_refused(self) -> None:
        with temp_dir_outside_repo("g38r5_appr_hours_") as root:
            approval, anchor = self._approval(root, gpu_hours_approved=1)
            code, payload = self._run_cli(
                self._real_mode_argv(root, approval, anchor, ["--requested-gpu-hours", "9"])
            )
        self.assertEqual(code, PolicyViolation.exit_code)
        self.assertIn(payload.get("code"), ("approval_gpu_hours_exceeded", "approval_not_trusted"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
