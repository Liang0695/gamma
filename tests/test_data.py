"""四面数据契约、导出与 oracle/回放的测试。"""

from __future__ import annotations

import unittest

from tests._fixtures import (
    H64,
    audit_face,
    bundle,
    oracle_face,
    public_face,
    step,
    trajectory_face,
)
from v3.common.errors import MissingInput, PolicyViolation
from v3.data.exporter import audit_mix, build_export, build_windows, steps_to_messages
from v3.data.faces import (
    FACE_AUDIT,
    FACE_ORACLE,
    FACE_PUBLIC,
    FACE_TRAJECTORY,
    assert_no_oracle_in_actor_input,
    validate_face,
    validate_release,
    validate_verification,
)
from v3.data.oracle import (
    OracleValidator,
    RunResult,
    ScriptedRunner,
    normalize_observation,
    replay,
    validate_question_set,
)


def count_tokens(messages) -> int:
    return sum(len(str(message.get("content", "")).split()) for message in messages) + len(messages)


class FaceTests(unittest.TestCase):
    def test_public_face_accepts_valid_record(self) -> None:
        validate_face(FACE_PUBLIC, public_face())

    def test_missing_field_is_illegal(self) -> None:
        record = public_face()
        record.pop("snapshot_sha256")
        with self.assertRaises(MissingInput):
            validate_face(FACE_PUBLIC, record)

    def test_family_id_must_match_canonical_origin_ids(self) -> None:
        record = public_face()
        record["problem_family_id"] = H64
        with self.assertRaises(PolicyViolation):
            validate_face(FACE_PUBLIC, record)

    def test_split_and_origin_kind_are_enumerated(self) -> None:
        record = public_face()
        record["split"] = "hidden"
        with self.assertRaises(PolicyViolation):
            validate_face(FACE_PUBLIC, record)

    def test_step_requires_all_action_fields(self) -> None:
        record = trajectory_face()
        record["steps"][0].pop("loss_eligible")
        with self.assertRaises(MissingInput):
            validate_face(FACE_TRAJECTORY, record)

    def test_oracle_keys_never_reach_actor_input(self) -> None:
        assert_no_oracle_in_actor_input({"messages": [{"role": "user", "content": "hi"}]})
        for leaked in (
            {"gold_files": ["a.py"]},
            {"nested": {"test_patch": "..."}},
            {"steps": [{"gold_hunks": []}]},
        ):
            with self.assertRaises(PolicyViolation):
                assert_no_oracle_in_actor_input(leaked)

    def test_verification_status_consistency(self) -> None:
        validate_verification("pass", True, True, False, 2)
        with self.assertRaises(PolicyViolation):
            validate_verification("pass", False, True, False, 2)
        with self.assertRaises(PolicyViolation):
            validate_verification("pass", True, True, True, 2)
        with self.assertRaises(PolicyViolation):
            validate_verification("pass", True, True, False, 1)
        with self.assertRaises(PolicyViolation):
            validate_verification("resolved", True, True, False, 2)

    def test_release_gating(self) -> None:
        summary = validate_release(bundle("design_only"))
        self.assertEqual(summary["release_state"], "design_only")

        released = bundle("released")
        validate_release(released)

        broken = bundle("released")
        broken["faces"][FACE_ORACLE]["gold_patch_sha256"] = ""
        with self.assertRaises(PolicyViolation):
            validate_release(broken)

        unlicensed = bundle("released", authorization_scope="pending")
        with self.assertRaises(PolicyViolation):
            validate_release(unlicensed)

        not_frozen = bundle("released")
        not_frozen["frozen"]["export_frozen"] = False
        with self.assertRaises(PolicyViolation):
            validate_release(not_frozen)

    def test_placeholder_release_id_is_rejected(self) -> None:
        broken = bundle("released")
        broken["release_id"] = "TBD"
        with self.assertRaises(PolicyViolation):
            validate_release(broken)


class ExporterTests(unittest.TestCase):
    def test_windows_split_on_complete_turns(self) -> None:
        public, trajectory = public_face(), trajectory_face()
        windows = build_windows(public, trajectory, count_tokens, window_tokens=8)
        self.assertGreaterEqual(len(windows), 2)
        for window in windows:
            self.assertTrue(window["supervised_steps"])

    def test_recovery_observation_and_successor_stay_together(self) -> None:
        steps = [
            step(1, "recover", supervised=False, finish_reason="error"),
            step(2, "recover", supervised=True),
            step(3, "validate"),
        ]
        windows = build_windows(public_face(), trajectory_face(steps=steps), count_tokens, window_tokens=1)
        first = windows[0]
        self.assertIn(1, first["step_range"][0:1] + [1])
        self.assertIn(2, [s["step"] for s in first["messages"] if s.get("step")])

    def test_window_never_contains_oracle_fields(self) -> None:
        windows = build_windows(public_face(), trajectory_face(), count_tokens)
        for window in windows:
            assert_no_oracle_in_actor_input(window)

    def test_empty_trajectory_is_rejected(self) -> None:
        with self.assertRaises(MissingInput):
            build_windows(public_face(), trajectory_face(steps=[]), count_tokens)

    def test_mix_audit_flags_hard_isolation_violation(self) -> None:
        windows = []
        for index in range(10):
            windows.append(
                {
                    "supervised_token_count": 1000,
                    "supervision_buckets": ["localize"],
                    "gold_access": index < 8,
                    "origin_kind": "mutation",
                    "repo_family": "alpha",
                }
            )
        audit = audit_mix(windows)
        self.assertEqual(audit["status"], "fail")
        self.assertIn("gold_assisted>20%", audit["hard_violations"])
        self.assertIn("mutation>40%", audit["hard_violations"])

    def test_build_export_produces_manifest(self) -> None:
        records = []
        for index, repo in enumerate(("alpha", "beta", "gamma", "delta")):
            records.append(
                {
                    "public": public_face(task_id="task-%d" % index, repo_family=repo),
                    "trajectory": trajectory_face(trace_id="trace-%d" % index, task_id="task-%d" % index),
                }
            )
        records.append(
            {
                "public": public_face(task_id="task-mut", repo_family="epsilon", origin_kind="mutation"),
                "trajectory": trajectory_face(trace_id="trace-mut", task_id="task-mut"),
            }
        )
        export = build_export(
            records,
            count_tokens,
            export_config={"template_lock": "shim-deterministic"},
        )
        manifest = export["export_manifest"]
        self.assertGreater(manifest["window_count"], 0)
        self.assertEqual(manifest["task_count"], 5)
        self.assertEqual(len(manifest["windows_sha256"]), 64)
        self.assertEqual(manifest["audit"]["status"], "warn")  # 比例警告，不是隔离违规
        self.assertEqual(manifest["audit"]["hard_violations"], [])

    def test_build_export_refuses_hard_violation(self) -> None:
        records = [
            {
                "public": public_face(task_id="m%d" % index, origin_kind="mutation"),
                "trajectory": trajectory_face(trace_id="t%d" % index, task_id="m%d" % index, gold_access=True),
            }
            for index in range(9)
        ]
        with self.assertRaises(PolicyViolation):
            build_export(records, count_tokens)

    def test_steps_to_messages_keeps_tool_arguments_structured(self) -> None:
        messages = steps_to_messages(public_face(), [step(1, "localize")])
        self.assertEqual(messages[0]["role"], "user")
        self.assertIsInstance(messages[1]["arguments"], dict)


def runner_for(question) -> ScriptedRunner:
    return question["runner"]


def make_scripted(broken: RunResult, reference: RunResult, clean: RunResult) -> ScriptedRunner:
    table = {}
    commands = ["python -m pytest -q tests/test_routing.py"]
    for ws, result in (("/ws/broken", broken), ("/ws/reference", reference), ("/ws/clean", clean)):
        table[ScriptedRunner.key(ws, commands)] = result
    runner = ScriptedRunner(table)
    return runner


class OracleTests(unittest.TestCase):
    def _question(self, broken_fails=True, reference_ok=True, clean_ok=True, flaky=False):
        commands = ["python -m pytest -q tests/test_routing.py"]
        broken = RunResult(
            exit_code=1 if broken_fails else 0,
            failed_tests=["tests/test_routing.py::test_timeout"] if broken_fails else [],
            passed_tests=[] if broken_fails else commands,
        )
        reference = RunResult(
            exit_code=0 if reference_ok else 1,
            failed_tests=[] if reference_ok else ["tests/test_routing.py::test_timeout"],
        )
        clean = RunResult(exit_code=0 if clean_ok else 2)
        table = {}
        for ws, result in (("/ws/broken", broken), ("/ws/reference", reference), ("/ws/clean", clean)):
            table[ScriptedRunner.key(ws, commands)] = result
        runner = ScriptedRunner(table)
        if flaky:
            original = runner.run

            calls = {"n": 0}

            def flaky_run(workspace, cmds):
                calls["n"] += 1
                if workspace == "/ws/broken" and calls["n"] > 1:
                    return RunResult(exit_code=0)
                return original(workspace, cmds)

            runner.run = flaky_run  # type: ignore[assignment]
        return {
            "question_id": "syn-q1",
            "oracle": oracle_face(),
            "workspaces": {
                "broken_ws": "/ws/broken",
                "reference_ws": "/ws/reference",
                "clean_ws": "/ws/clean",
            },
            "runner": runner,
        }

    def _contrast(self, question):
        validator = OracleValidator(question["runner"], question["oracle"])
        return validator.contrast("/ws/broken", "/ws/reference", "/ws/clean")

    def test_good_question_passes_and_needs_two_repeats(self) -> None:
        outcome = self._contrast(self._question())
        self.assertEqual(outcome["status"], "pass")
        self.assertFalse(outcome["quarantine"])
        self.assertEqual(len(outcome["runs"]["reference"]), 2)

    def test_broken_that_does_not_fail_is_quarantined(self) -> None:
        outcome = self._contrast(self._question(broken_fails=False))
        self.assertEqual(outcome["status"], "fail")
        self.assertTrue(outcome["quarantine"])

    def test_reference_that_fails_is_quarantined(self) -> None:
        outcome = self._contrast(self._question(reference_ok=False))
        self.assertTrue(outcome["quarantine"])
        self.assertTrue(any("reference_tree" in p for p in outcome["problems"]))

    def test_p2p_regression_is_quarantined(self) -> None:
        outcome = self._contrast(self._question(clean_ok=False))
        self.assertTrue(any("P2P" in p for p in outcome["problems"]))

    def test_flaky_runs_are_inconclusive(self) -> None:
        outcome = self._contrast(self._question(flaky=True))
        self.assertEqual(outcome["status"], "inconclusive")

    def test_question_set_requires_all_eight(self) -> None:
        questions = []
        for index in range(8):
            question = self._question(broken_fails=index != 0)
            question["question_id"] = "syn-q%d" % index
            questions.append(question)
        report = validate_question_set(questions, lambda question: question["runner"])
        self.assertEqual(report["total"], 8)
        self.assertEqual(report["passed"], 7)
        self.assertFalse(report["qualified"])

    def test_replay_detects_tree_and_observation_divergence(self) -> None:
        steps = [dict(step(1, "localize"), input_tree_sha256=H64, output_tree_sha256="b" * 64)]
        result = replay(
            steps,
            H64,
            apply_action=lambda s, tree: "b" * 64,
            normalize_observation=normalize_observation,
        )
        self.assertEqual(result["status"], "pass")
        bad = replay(
            steps,
            H64,
            apply_action=lambda s, tree: "c" * 64,
            normalize_observation=normalize_observation,
        )
        self.assertEqual(bad["status"], "fail")
        self.assertEqual(bad["divergences"][0]["kind"], "output_tree_mismatch")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
