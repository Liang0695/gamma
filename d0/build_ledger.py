"""Build the verified V3 training-family ledger (D0 deliverable).

For every family this script re-derives, from the pinned git objects:
  * base_commit / fix_commit full SHAs and the parent link between them
  * sha256 of the oracle patch, of each touched file at base and at fix
  * whether the FAIL_TO_PASS test node exists at base (it must NOT)
    and at fix (it must)
  * the added assertion lines that constitute the oracle basis
  * dependency material for the pinned revision

A family is only marked released=true when every SHA/hash field is non-empty
and the structural checks pass.  Empty-SHA records can never be released.
"""
import hashlib
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "out")

POLICY_ID = "v3-time-policy-2026-10-05-mika"
TRAIN_WINDOW_END = "2026-01-01T00:00:00+00:00"      # == fix time <= 2025-12-31
DEVSEALED_START = "2026-01-01T00:00:00+00:00"
DEVSEALED_END_EXCLUSIVE = "2026-10-05T00:00:00+00:00"   # == through 2026-10-04
WITHDRAWN_WINDOWS = {
    "train_end_exclusive": "2025-01-01T00:00:00+00:00",
    "dev_start": "2025-01-01T00:00:00+00:00",
    "dev_end_exclusive": "2026-01-01T00:00:00+00:00",
    "sealed_start": "2026-01-01T00:00:00+00:00",
    "status": ("withdrawn 2026-10-05: never approved; superseded by the windows above. "
               "dev and sealed share ONE window and no ordering between them is claimed."),
}
BACKPORT_RE = re.compile(
    r"cherry[- ]?picked?\s+from|cherry[- ]?pick\s+of|back[- ]?port|"
    r"backport|cherry-pick|re-?land",
    re.I,
)

REPOS = {
    "click":          {"slug": "pallets/click", "spdx": "BSD-3-Clause",
                       "license_file": "LICENSE.txt", "split": "train"},
    "more-itertools": {"slug": "more-itertools/more-itertools", "spdx": "MIT",
                       "license_file": "LICENSE", "split": "train"},
    "pluggy":         {"slug": "pytest-dev/pluggy", "spdx": "MIT",
                       "license_file": "LICENSE", "split": "train"},
    "boltons":        {"slug": "mahmoud/boltons", "spdx": "BSD-3-Clause",
                       "license_file": "LICENSE", "split": "train"},
    "attrs":          {"slug": "python-attrs/attrs", "spdx": "MIT",
                       "license_file": "LICENSE", "split": "dev"},
    "dateutil":       {"slug": "dateutil/dateutil", "spdx": "Apache-2.0 OR BSD-3-Clause",
                       "license_file": "LICENSE", "split": "dev"},
    "packaging":      {"slug": "pypa/packaging", "spdx": "Apache-2.0 OR BSD-2-Clause",
                       "license_file": "LICENSE", "split": "sealed"},
    "marshmallow":    {"slug": "marshmallow-code/marshmallow", "spdx": "MIT",
                       "license_file": "LICENSE", "split": "sealed"},
}

# ---- the four real P0 training families (one per training repo) -------------
REAL_FAMILIES = [
    {
        "family_id": "v3-train-click-001",
        "repo": "click",
        "fix_commit": "9da1791476",
        "issue_refs": ["222"],
        "symptom_restatement": (
            "An option declared with nargs=-1 accepts a value list whose length is "
            "not validated, so a CLI definition that is internally inconsistent is "
            "accepted at construction time and only misbehaves later. The task is to "
            "make the option constructor reject the unsupported nargs value eagerly "
            "with a clear error instead of silently building a broken parameter."),
        "failure_mechanism": "missing eager validation in the option/parameter constructor path",
    },
    {
        "family_id": "v3-train-more-itertools-001",
        "repo": "more-itertools",
        "fix_commit": "62411c1618",
        "issue_refs": ["409"],
        "symptom_restatement": (
            "A helper that wraps an iterable and records what was consumed returns an "
            "object that callers can still mutate, so a consumer that appends to the "
            "returned container corrupts the recorded history. The task is to make the "
            "returned container reject mutation while keeping the read path unchanged."),
        "failure_mechanism": "returned recording container is not immutable",
    },
    {
        "family_id": "v3-train-pluggy-001",
        "repo": "pluggy",
        "fix_commit": "9cf2eaa50d",
        "issue_refs": ["544"],
        "symptom_restatement": (
            "When a hook implementation signals termination by raising StopIteration and "
            "the hook is wrapped, the wrapper teardown path turns the signal into a "
            "RuntimeError, so the caller sees a generic generator error instead of the "
            "original signal. The task is to make the wrapper teardown path resume with "
            "the original signal when the RuntimeError was caused by it, for both the "
            "modern and the legacy wrapper form."),
        "failure_mechanism": "generator StopIteration converted to RuntimeError in teardown path",
    },
    {
        "family_id": "v3-train-boltons-001",
        "repo": "boltons",
        "fix_commit": "ae21ed2a78",
        "issue_refs": ["30"],
        "symptom_restatement": (
            "Exception formatting fails when the frames referenced by a traceback have "
            "no retrievable source, for example for code compiled from a string, so "
            "printing the traceback raises instead of displaying it. The task is to "
            "make the formatter degrade gracefully for frameless/sourceless frames."),
        "failure_mechanism": "source-unavailable frames not handled during formatting",
    },
]

# ---- the four P0 variant families derived from the real ones ---------------
VARIANT_FAMILIES = [
    {
        "family_id": "v3-train-click-001-var-rename",
        "derived_from": "v3-train-click-001",
        "repo": "click",
        "mutation_class": "symbol-rename",
        "mutation_recipe": (
            "On top of the real family's base_commit, commit a pure identifier rename of "
            "the validation helper and of the public parameter class attribute used by the "
            "fix (no behaviour change). Re-express the task text against the renamed API."),
        "why": ("Blocks solutions that hard-code the upstream symbol name or replay the "
                "upstream patch textually; the agent must re-locate the code."),
        "expected_patch_shape": {"files": 2, "hunks": 2},
    },
    {
        "family_id": "v3-train-more-itertools-001-var-api",
        "derived_from": "v3-train-more-itertools-001",
        "repo": "more-itertools",
        "mutation_class": "public-api-change",
        "mutation_recipe": (
            "On top of the real family's base_commit, move the recording helper into a "
            "submodule and expose it under a new public name with a thin alias, then "
            "re-target the task text and the test import at the new name."),
        "why": ("Tests whether the patch follows behaviour rather than a remembered "
                "import path, and exercises multi-file patch assembly."),
        "expected_patch_shape": {"files": 3, "hunks": 3},
    },
    {
        "family_id": "v3-train-pluggy-001-var-backport",
        "derived_from": "v3-train-pluggy-001",
        "repo": "pluggy",
        "mutation_class": "backport",
        "mutation_recipe": (
            "Forward-port the real family's defect onto the pinned 1.6.0 revision: at the "
            "pinned revision remove the neutral-value branch introduced by the fix while "
            "keeping the later surrounding refactor, then author the task text against the "
            "modern layout."),
        "why": ("The upstream patch does not apply and the surrounding code differs, so the "
                "agent must re-derive the fix in an evolved codebase."),
        "expected_patch_shape": {"files": 2, "hunks": 2},
    },
    {
        "family_id": "v3-train-boltons-001-var-multidefect",
        "derived_from": "v3-train-boltons-001",
        "repo": "boltons",
        "mutation_class": "multi-defect-split",
        "mutation_recipe": (
            "Split the real family's single fix into two independently observable defects "
            "on the same base_commit (one in the frame-lookup path, one in the rendering "
            "path), requiring a two-part patch before the reporting test passes."),
        "why": ("Forces a repair that spans more than one function and rewards recovery "
                "after a partial patch, matching the patch/verification and recovery "
                "supervision buckets."),
        "expected_patch_shape": {"files": 2, "hunks": 3},
    },
]

def decorator_block(source, func_name):
    """Return the contiguous decorator block immediately above ``def func_name``.

    Walking backwards with paren-depth tracking is what keeps a neighbouring
    test's decorator from being treated as this function's parametrization.
    """
    m = re.search(r"^([ \t]*)def %s\b" % re.escape(func_name), source, re.M)
    if not m:
        return None
    lines = source[:m.start()].splitlines()
    i = len(lines) - 1
    block = []
    depth = 0
    while i >= 0:
        line = lines[i]
        stripped = line.strip()
        if not block and stripped == "":
            i -= 1
            continue
        if depth == 0 and not stripped.startswith("@"):
            break
        block.insert(0, line)
        depth += line.count("(") - line.count(")")
        if depth < 0:
            depth = 0
        i -= 1
    text = "\n".join(block)
    found = re.search(r"@pytest\.mark\.parametrize\((?P<args>.*)\)\s*$", text, re.S)
    return found


TEST_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$", re.I)
CODE_RE = re.compile(r"\.pyi?$")
# CI / packaging / changelog material that upstream fixes often touch but that is
# never part of the minimal oracle patch
CONFIG_RE = re.compile(
    r"(^|/)\.(travis|github|pre-commit|gitignore|readthedocs)|"
    r"(^|/)(tox\.ini|setup\.cfg|setup\.py|pyproject\.toml|MANIFEST\.in|"
    r"CHANGES|CHANGELOG|changelog/|docs/|doc/)|\.(yml|yaml|cfg|ini|toml|rst|md)$",
    re.I)


def file_role(path):
    if TEST_RE.search(path):
        return "test"
    if CODE_RE.search(path) and not CONFIG_RE.search(path):
        return "code"
    return "other"


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def blob_sha256(repo, rev, path):
    rc, out, err = git(repo, "show", "%s:%s" % (rev, path))
    if rc != 0:
        return None
    return hashlib.sha256(out.encode("utf-8", "replace")).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    lock = json.load(open(os.path.join(OUT, "source-lock.json"), encoding="utf-8"))
    lic = json.load(open(os.path.join(OUT, "license-files.json"), encoding="utf-8"))
    ledger = {
        "policy_id": POLICY_ID,
        "spec": {
            "train_window": {"end_exclusive": TRAIN_WINDOW_END,
                             "means": "family original fix time <= 2025-12-31"},
            "dev_sealed_window": {"start": DEVSEALED_START,
                                  "end_exclusive": DEVSEALED_END_EXCLUSIVE,
                                  "means": "family original fix time in 2026-01-01 .. 2026-10-04",
                                  "dev_vs_sealed_ordering_claimed": False},
            "withdrawn_windows": WITHDRAWN_WINDOWS,
            "time_axis_authority": (
                "a family's time is the ORIGINAL upstream fix commit's own time, recorded as "
                "BOTH author date and committer date with both required inside the window. "
                "Release/tag dates, snapshot or pin dates, backport dates and cherry-pick "
                "dates are rejected substitutes and must never classify a family."),
            "snapshot_axis_is_separate": (
                "the pinned revision decides which CONTENT exists (the fix commit must be an "
                "ancestor of it); it never decides which WINDOW a family belongs to."),
            "p0_requirement": "8 training families = 4 real + 4 variant, from >= 2 repos",
            "issue_text_policy": (
                "No upstream issue/PR prose is copied into the deliverable. Only the "
                "numeric issue reference is recorded for traceability; every task "
                "statement is rewritten from the observed defect, per the integrated "
                "review's rule that code being open source does not license its comments."),
            "runtime_verification_status": (
                "not_run_in_d0: the D0 research runtime has no reachable package index, so "
                "pytest could not be installed and no FAIL_TO_PASS run was executed. Every "
                "field below is derived statically from the pinned git objects. Building the "
                "environment and actually running the oracle is E0's deliverable, and the "
                "8-task cross-check is later engineering acceptance."),
            "variant_release_policy": (
                "variant families are released=false by construction: their variant commit "
                "does not exist until an author builds it. Only the upstream source base "
                "commit is pinned, so no record can be published with an empty SHA."),
        },
        "families": [],
    }

    for spec in REAL_FAMILIES:
        name = spec["repo"]
        repo = os.path.join(SRC, name)
        rc, fix_full, _ = git(repo, "rev-parse", spec["fix_commit"] + "^{commit}")
        fix_full = fix_full.strip()
        rc, meta, _ = git(repo, "log", "-1", "--format=%P%x1f%aI%x1f%cI%x1f%s%x1f%b",
                          fix_full)
        parts = (meta.strip().split("\x1f") + ["", "", "", "", ""])[:5]
        parents, adate, cdate, subject, body = parts
        base_full = parents.split()[0]
        pinned = lock["repos"][name]["pinned_commit"]
        rc_anc, _, _ = git(repo, "merge-base", "--is-ancestor", fix_full, pinned)
        fix_is_ancestor = rc_anc == 0
        backport_marker = BACKPORT_RE.search(subject + "\n" + body)
        rc, patch, _ = git(repo, "show", "--format=", "--no-renames", fix_full)
        patch_sha = hashlib.sha256(patch.encode("utf-8", "replace")).hexdigest()
        _, numstat, _ = git(repo, "show", "--numstat", "--format=", "--no-renames", fix_full)

        touched = []
        for line in numstat.strip().splitlines():
            a, d, p = (line.split("\t") + ["", "", ""])[:3]
            role = file_role(p)
            is_test = role == "test"
            at_base = blob_sha256(repo, base_full, p)
            at_fix = blob_sha256(repo, fix_full, p)
            if at_base is None and at_fix is not None:
                status = "added"
            elif at_base is not None and at_fix is None:
                status = "deleted"
            else:
                status = "modified"
            touched.append({
                "path": p,
                "kind": "test" if is_test else "source",
                "role": role,
                "status_at_fix": status,
                "added": None if a == "-" else int(a),
                "deleted": None if d == "-" else int(d),
                "sha256_at_base": at_base,
                "sha256_at_fix": at_fix,
            })

        added_tests = sorted(set(re.findall(r"^\+\s*(?:async\s+)?def (test[A-Za-z0-9_]*)",
                                            patch, re.M)))
        f2p = []
        for tf in [t for t in touched if t["kind"] == "test"]:
            rc2, fix_body, _ = git(repo, "show", "%s:%s" % (fix_full, tf["path"]))
            rc3, base_body, _ = git(repo, "show", "%s:%s" % (base_full, tf["path"]))
            if rc3 != 0:
                base_body = ""
            for tn in added_tests:
                if not re.search(r"\bdef %s\b" % re.escape(tn), fix_body):
                    continue
                if re.search(r"\bdef %s\b" % re.escape(tn), base_body):
                    continue  # pre-existing test: not a fail-to-pass node
                # Only a parametrize decorator DIRECTLY attached to this function may
                # multiply its node ids.  Collect the contiguous decorator block that
                # sits immediately above the def, so a neighbouring test's decorator
                # can never leak its parameter names into this node id.
                attached = decorator_block(fix_body, tn)
                ids = []
                if attached:
                    lst = re.search(r"\[(.*?)\]", attached.group("args"), re.S)
                    if lst:
                        body_items = lst.group(1)
                        depth = 0
                        cur = ""
                        items = []
                        for ch in body_items:
                            if ch in "[(":
                                depth += 1
                            elif ch in ")]":
                                depth -= 1
                            if ch == "," and depth == 0:
                                items.append(cur)
                                cur = ""
                            else:
                                cur += ch
                        if cur.strip():
                            items.append(cur)
                        for it in items:
                            it = it.strip()
                            q = re.fullmatch(r"['\"](.*)['\"]", it)
                            if q:
                                ids.append(q.group(1))
                            elif re.fullmatch(r"(True|False|None|-?\d+\.?\d*)", it):
                                ids.append(it)
                            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", it):
                                ids.append(it)
                if ids:
                    for vid in ids:
                        f2p.append({
                            "node_id": "%s::%s[%s]" % (tf["path"], tn, vid),
                            "present_at_base": False,
                            "present_at_fix": True,
                            "test_added_by_fix": True,
                            "derived_from_parametrize": True,
                        })
                else:
                    f2p.append({
                        "node_id": "%s::%s" % (tf["path"], tn),
                        "present_at_base": False,
                        "present_at_fix": True,
                        "test_added_by_fix": True,
                    })
        f2p = {x["node_id"]: x for x in f2p}
        f2p = list(f2p.values())

        # test files that the fix also edited but did not add -> regression watch (P2P)
        p2p_watch = sorted({t["path"] for t in touched
                            if t["kind"] == "test" and t["status_at_fix"] == "modified"})

        added_asserts = [ln[1:].strip() for ln in patch.splitlines()
                         if re.match(r"^\+\s*assert\b", ln)]

        # Split the fix into the two halves the oracle needs: the test patch that is
        # applied to the base checkout to expose the defect, and the gold patch that
        # resolves it.  Evaluation of a solved task must use ONLY the test patch.
        test_paths = [t["path"] for t in touched if t["role"] == "test"]
        gold_paths = [t["path"] for t in touched if t["role"] != "test"]
        code_paths = [t["path"] for t in touched if t["role"] == "code"]
        def sub_patch(paths):
            if not paths:
                return ""
            _, out, _ = git(repo, "diff", "--no-renames", base_full, fix_full, "--", *paths)
            return out
        test_patch = sub_patch(test_paths)
        gold_patch = sub_patch(gold_paths)
        code_patch = sub_patch(code_paths)
        oracle_split = {
            "test_patch_sha256": hashlib.sha256(test_patch.encode("utf-8", "replace")).hexdigest(),
            "test_patch_files": sorted(test_paths),
            "test_patch_bytes": len(test_patch),
            "upstream_reference_patch_sha256": hashlib.sha256(
                gold_patch.encode("utf-8", "replace")).hexdigest(),
            "upstream_reference_patch_files": sorted(gold_paths),
            "upstream_reference_patch_bytes": len(gold_patch),
            "code_only_patch_sha256": hashlib.sha256(
                code_patch.encode("utf-8", "replace")).hexdigest(),
            "code_only_patch_files": sorted(code_paths),
            "code_only_patch_bytes": len(code_patch),
            "which_patch_is_the_minimal_oracle": (
                "code_only: the upstream commit also edits CI and changelog material, which "
                "is not needed to make the fail_to_pass nodes pass and must not be counted "
                "as part of the required fix"),
            "apply_test_patch_to_base": ("git apply the test patch onto base_commit, run the "
                                         "fail_to_pass nodes, expect failure"),
            "apply_both_then_run": ("apply the test patch and the code-only patch onto "
                                    "base_commit, run the fail_to_pass nodes, expect pass"),
            "gold_patch_must_not_reach_actor": True,
        }

        meta_pk = lock["repos"][name]["packaging_metadata"]
        requires_python = None
        deps = []
        pp = os.path.join(repo, "pyproject.toml")
        if os.path.isfile(pp):
            txt = open(pp, encoding="utf-8", errors="replace").read()
            m = re.search(r'requires-python\s*=\s*"([^"]+)"', txt)
            if m:
                requires_python = m.group(1)
            for m in re.finditer(r'dependencies\s*=\s*\[(.*?)\]', txt, re.S):
                deps += [d.strip().strip('",\'') for d in m.group(1).splitlines()
                         if d.strip().strip(',')]

        lic_entry = [x for x in lic["by_repo"][name]
                     if x["path"] == REPOS[name]["license_file"]][0]

        structural = {
            "fix_commit_resolved": bool(re.fullmatch(r"[0-9a-f]{40}", fix_full)),
            "base_commit_resolved": bool(re.fullmatch(r"[0-9a-f]{40}", base_full)),
            "base_is_parent_of_fix": base_full in parents.split(),
            "patch_hash_present": bool(re.fullmatch(r"[0-9a-f]{64}", patch_sha)),
            "touched_files_hashed_at_fix": all(t["sha256_at_fix"] for t in touched),
            "modified_files_hashed_at_base": all(
                t["sha256_at_base"] for t in touched if t["status_at_fix"] != "added"),
            "has_test_file": any(t["kind"] == "test" for t in touched),
            "has_source_file": any(t["kind"] == "source" for t in touched),
            "source_file_actually_changed": any(
                t["kind"] == "source" and t["sha256_at_base"] != t["sha256_at_fix"]
                for t in touched),
            "f2p_non_empty": len(f2p) > 0,
            "f2p_absent_at_base": all(not x["present_at_base"] for x in f2p) if f2p else False,
            "oracle_split_both_halves_non_empty": bool(test_patch) and bool(gold_patch),
            "code_only_oracle_patch_non_empty": bool(code_patch),
            "in_train_window_author_date": adate < TRAIN_WINDOW_END,
            "in_train_window_committer_date": cdate < TRAIN_WINDOW_END,
            "fix_commit_is_ancestor_of_pinned_revision": fix_is_ancestor,
            "fix_commit_is_not_a_backport_or_cherry_pick": backport_marker is None,
            "license_approved": lock["repos"][name]["license_review"]["decision"] == "approved",
            "license_approved_for_this_exact_revision":
                lock["repos"][name]["license_review"]["decided_against_revision"]
                == lock["repos"][name]["pinned_commit"],
        }
        released = all(structural.values()) and not any(
            v is None or v == "" for v in (base_full, fix_full, patch_sha))
        ledger["families"].append({
            "family_id": spec["family_id"],
            "kind": "real",
            "split": "train",
            "released": released,
            "repo": name,
            "upstream_slug": REPOS[name]["slug"],
            "pinned_revision": lock["repos"][name]["pinned_commit"],
            "license": {"spdx": REPOS[name]["spdx"],
                        "license_file": lic_entry["path"],
                        "license_file_sha256": lic_entry["sha256"],
                        "license_review_decision":
                            lock["repos"][name]["license_review"]["decision"],
                        "license_review_path": "repos.%s.license_review in source-lock.json" % name},
            "base_commit": base_full,
            "oracle_fix_commit": fix_full,
            "oracle_patch_sha256": patch_sha,
            "fix_time": {
                "author_date": adate,
                "committer_date": cdate,
                "primary": adate,
                "basis": "author date of the original upstream fix commit",
                "both_dates_in_train_window": (adate < TRAIN_WINDOW_END
                                               and cdate < TRAIN_WINDOW_END),
                "not_derived_from": ["release/tag date", "snapshot or pin date",
                                     "backport date", "cherry-pick date"],
            },
            "fix_date": adate,
            "fix_date_committer": cdate,
            "fix_commit_is_ancestor_of_pinned_revision": fix_is_ancestor,
            "backport_or_cherry_pick_marker": (backport_marker.group(0)
                                               if backport_marker else None),
            "snapshot_axis_note": (
                "the pinned revision decides which content exists; the family's own fix time "
                "decides the window. The two axes are checked separately."),
            "upstream_subject": subject,
            "upstream_issue_refs": spec["issue_refs"],
            "symptom_restatement": spec["symptom_restatement"],
            "failure_mechanism": spec["failure_mechanism"],
            "touched_files": touched,
            "fail_to_pass": f2p,
            "regression_watch": p2p_watch,
            "oracle_split": oracle_split,
            "oracle_assertions": added_asserts[:10],
            "runtime_verified": False,
            "env": {
                "requires_python": requires_python,
                "runtime_dependencies": deps[:20],
                "test_runner": "pytest (node ids above are pytest node ids)",
                "packaging_metadata_files": sorted(meta_pk.keys()),
            },
            "checks": structural,
        })

    for spec in VARIANT_FAMILIES:
        name = spec["repo"]
        repo = os.path.join(SRC, name)
        parent_fam = [f for f in ledger["families"]
                      if f["family_id"] == spec["derived_from"]][0]
        # the variant's source base is the real family's base_commit (rename/api/split)
        # or the pinned revision (backport) -- both are real, non-empty upstream SHAs
        variant_base = (parent_fam["pinned_revision"] if spec["mutation_class"] == "backport"
                        else parent_fam["base_commit"])
        rc, vb_full, _ = git(repo, "rev-parse", variant_base + "^{commit}")
        vb_full = vb_full.strip()
        _, when, _ = git(repo, "log", "-1", "--format=%cI", vb_full)
        released = False
        ledger["families"].append({
            "family_id": spec["family_id"],
            "kind": "variant",
            "split": "train",
            "released": released,
            "release_blocker": ("variant commit does not exist yet: it must be constructed "
                                "by the coding officer; only the upstream source base is "
                                "pinned, so no empty SHA is ever published as an example"),
            "repo": name,
            "upstream_slug": REPOS[name]["slug"],
            "derived_from": spec["derived_from"],
            "mutation_class": spec["mutation_class"],
            "mutation_recipe": spec["mutation_recipe"],
            "rationale": spec["why"],
            "expected_patch_shape": spec["expected_patch_shape"],
            "source_base_commit": vb_full,
            "source_base_date": when.strip(),
            "source_base_date_role": (
                "SNAPSHOT-axis date of the upstream source base, recorded for reproducibility. "
                "It is NOT this variant's fix time and does not classify the variant into a "
                "window. A constructed variant inherits its parent family's train window."),
            "source_base_released": True,
            "variant_commit": None,
            "oracle_fix_commit": None,
            "license": parent_fam["license"],
        })

    with open(os.path.join(OUT, "family-ledger.json"), "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)

    print("families=%d" % len(ledger["families"]))
    for fam in ledger["families"]:
        if fam["kind"] == "real":
            bad = [k for k, v in fam["checks"].items() if not v]
            print("  %-40s %-14s released=%s f2p=%d %s" % (
                fam["family_id"], fam["fix_date"][:10], fam["released"],
                len(fam["fail_to_pass"]), ("FAILED:" + ",".join(bad)) if bad else "ok"))
        else:
            print("  %-40s %-14s released=%s(%s) source_base=%s" % (
                fam["family_id"], fam["source_base_date"][:10], fam["released"],
                fam["mutation_class"], fam["source_base_commit"][:12]))


if __name__ == "__main__":
    main()
