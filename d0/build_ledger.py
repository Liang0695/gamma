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
STATUS_CONTRACT_ID = "v3-status-axes-2026-10-06-mika"

SOURCE_PREPARATION_SCOPE = (
    "source preparation ONLY: the static artefacts for this family (pinned base and fix "
    "commits, hashed patches, oracle split, the licence decision for the exact pinned "
    "revision, and a window set by a verified ORIGINAL merge event) are complete. This is "
    "NOT a training release and NOT a data release: it carries no runtime, no independent "
    "licence countersignature and no permission-isolation evidence.")

STATUS_CONTRACT_TEXT = (
    "Two independent axes. Axis 1, source_preparation.status, is a static-artefact "
    "statement and is the ONLY thing the compatibility field `released` mirrors. Axis 2, "
    "training_release.status, is the release gate and additionally requires verified "
    "runtime oracle results on the actor environment, an independent licence review, "
    "demonstrated isolation of gold/dev/sealed material from the actor, and a constructed "
    "variant half. A consumer MUST read `training_release.status`; reading `released` and "
    "inferring a training approval from it is a contract violation. `released` is retained "
    "only so an existing consumer keyed on it keeps working, and it is scoped by the "
    "companion `released_scope` string that travels with it in every emitted file.")

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
        "fix_commit": "ee56925bc4f5",
        "issue_refs": ["1925"],
        "replaces": "click-9da1791476fe79ce77aa7a2a2db370c91a455251",
        "symptom_restatement": (
            "An option object reports itself as a boolean flag whenever its value type "
            "happens to be boolean, even when it was never declared as a flag. Downstream "
            "code that branches on that report therefore takes the flag code path for "
            "ordinary options, so a plain option with a boolean value type is treated as "
            "something the user can toggle. The task is to make the reported flag-ness "
            "depend on the declaration and on the value type together, so a non-flag option "
            "reports False while a genuine flag still reports True."),
        "failure_mechanism": (
            "flag-ness predicate tests only the value type and ignores the declaration"),
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
        "family_id": "v3-train-click-001-var-predicate",
        "derived_from": "v3-train-click-001",
        "repo": "click",
        "mutation_class": "predicate-relocation-and-propagation",
        "mutation_recipe": (
            "On top of the real family's base_commit, move the flag-ness predicate out of "
            "the option constructor into a module-level helper that the constructor calls, "
            "and have a second consumer read flag-ness through that helper on a different "
            "code path (the help/usage rendering path) instead of the stored attribute. "
            "Leave the helper's body with the same defect the real family fixes. Re-express "
            "the task text against the helper's new name and location."),
        "why": ("Blocks a solution that reproduces the upstream one-line diff by pattern or "
                "by remembered file/line: the predicate no longer lives where upstream put "
                "it, and a fix applied only to the constructor leaves the rendering path "
                "still reporting the wrong answer, so the agent must re-locate the "
                "behaviour and fix it where it is actually computed."),
        "expected_patch_shape": {"files": 2, "hunks": 3},
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

def source_preparation_status(structural, base_full, fix_full, patch_sha):
    """Static source-preparation state, and NOTHING more.

    This is the field that used to be called ``released``. It answers exactly one
    question: are this family's static source artefacts -- pinned base and fix
    commits, hashed patches, oracle split, licence decision for the exact pinned
    revision, and a window set by a verified ORIGINAL merge event -- internally
    complete? It does *not* say the family may train an agent.
    """
    ok = bool(all(structural.values())) and not any(
        v is None or v == "" for v in (base_full, fix_full, patch_sha))
    return {
        "status": "ready" if ok else "not_ready",
        "basis": ("every structural check in `checks` passed; this covers static source "
                  "artefacts only"),
        "blocking_checks": sorted(k for k, v in structural.items() if not v),
    }


def training_release_decision(structural, base_full, fix_full, patch_sha, lic):
    """May this family be used to train or evaluate an agent? Currently never, by design.

    The gate that separates "the static source material is complete" from "the task is
    fit to train on" needs positive evidence for each of these. None of it exists in
    this delivery, so every family is blocked -- and the reason is enumerated rather
    than implied, so no consumer can read a source-preparation status as a training
    approval.
    """
    prep_ok = (bool(all(structural.values()))
               and not any(v is None or v == "" for v in (base_full, fix_full, patch_sha)))
    evidence = {
        # runtime: broken/reference/P2P double-clean control on a real environment
        "runtime_oracle_verified_on_the_actor_environment": {
            "verified": False,
            "reason": "no FAIL_TO_PASS / reference run has been executed anywhere yet; "
                      "every oracle field in this ledger is static evidence",
            "owner": "E0 (builds the environment and runs the oracles)",
        },
        # the licence decision is self-declared until Q0 signs it
        "licence_independently_reviewed": {
            "verified": False,
            "reason": "independent_review.status is '%s' for this repository; the decision "
                      "field is author-declared and must not be read as a countersignature"
                      % (lic.get("independent_review") or {}).get("status"),
            "owner": "Q0 (independent review)",
        },
        # gold / dev / sealed material has no demonstrated isolation yet
        "gold_and_sealed_material_isolated_from_the_actor": {
            "verified": False,
            "reason": "no negative permission test has been produced; the restricted "
                      "material's split is still declared pending and there is no evidence "
                      "that a training/search actor cannot read or re-derive it",
            "owner": "custodian + Q0 (permission and exclusion evidence)",
        },
        # the variant half does not exist yet
        "variant_half_constructed": {
            "verified": False,
            "reason": "variant families are construction recipes; their mutation commits "
                      "do not exist yet",
            "owner": "E0 (coding officer)",
        },
    }
    missing = sorted(k for k, v in evidence.items() if not v["verified"])
    return {
        "status": "blocked" if missing else "approved",
        "approved_for_training": False if missing else True,
        "reason": ("blocked: static source preparation is not a training release. "
                   "Missing positive evidence: %s" % ", ".join(missing)
                   if missing else "all training-release evidence present"),
        "requires": evidence,
        "precondition_source_preparation": "ready" if prep_ok else "not_ready",
        "precondition_owner": ("a family whose source preparation is not ready can never be "
                              "training-released"),
        "owner": "Mika / custodian decision; D0 records the state and does not self-approve",
    }


def install_status_separation(fam, structural, base_full, fix_full, patch_sha, lic):
    """Mount the two status axes on a family record with one shared scope contract."""
    prep = source_preparation_status(structural, base_full, fix_full, patch_sha)
    train = training_release_decision(structural, base_full, fix_full, patch_sha, lic)
    fam["source_preparation"] = prep
    # compatibility field, EXPLICITLY scoped: it mirrors the static preparation axis
    # only, and the scope string travels with it so a consumer cannot mistake it for a
    # training release even if it reads this single key by habit.
    fam["released"] = (prep["status"] == "ready")
    fam["released_scope"] = SOURCE_PREPARATION_SCOPE
    fam["training_release"] = train
    fam["status_axes"] = {
        "contract_id": STATUS_CONTRACT_ID,
        "contract": STATUS_CONTRACT_TEXT,
        "training_release_admissible": train["approved_for_training"],
        "consumer_rule": ("consume `training_release.status`, never `released`; `released` "
                          "is a source-preparation alias kept for backward compatibility"),
    }
    return fam


def gold_derivability(families):
    """Can the actor reconstruct the gold patch from what it is given?

    Yellow item 6 from the review. The answer is stated per family instead of being
    left for a reader to infer, and the argument names the concrete thing that keeps
    the gold out rather than asserting good intentions.
    """
    per_family = {}
    for f in families:
        fid = f["family_id"]
        if f["kind"] == "variant":
            per_family[fid] = {
                "kind": "variant",
                "gold_patch_reaches_the_actor": False,
                "actor_receives": ["nothing yet: the mutation commit does not exist"],
                "derivability_argument": (
                    "there is no gold patch to derive because the defective revision has "
                    "not been constructed. This is a statement about the current state, "
                    "not a solved problem: once the variant is built, its gold becomes "
                    "derivable in exactly the way the real family's is -- from the task "
                    "text and the failing test -- and must be withheld the same way."),
                "leak_channels_closed": [],
                "leak_channels_open": ["the variant's oracle does not exist, so nothing "
                                       "has been closed for it yet"],
            }
            continue
        per_family[fid] = {
            "kind": "real",
            "gold_patch_reaches_the_actor": False,
            "actor_receives": [
                "the rewritten task statement (D0 writes it; no upstream prose is copied)",
                "the failing test node ids and the test patch",
                "the pinned base revision",
                "environment material (dependency metadata, requires-python, test runner)",
            ],
            "not_given_to_the_actor": [
                "oracle_assertions (withheld into the restricted file)",
                "the upstream commit message, PR body, issue thread and review comments",
                "the upstream repository history at the pinned revision beyond the base",
                "the code-only gold patch",
            ],
            "derivability_argument": (
                "the actor is given the failing behaviour and the test that exposes it, "
                "which is the intended difficulty. What it is NOT given is the fix: the "
                "oracle split keeps the code-only patch out, the restricted file holds the "
                "assertions, and the task text is a rewrite rather than the upstream "
                "message. The remaining derivability of a *correct fix* is inherent to a "
                "SWE-style task and is not treated here as a leak."),
            "leak_channels_closed": [
                "oracle_assertions excluded from public-manifest.json",
                "gold patch not emitted into any public artifact",
                "task text rewritten, not copied",
            ],
            "leak_channels_open": [
                "the upstream repository is PUBLIC: an actor with network access and the "
                "pinned commit could locate the real fix upstream. No actor-side network "
                "control is demonstrated in this delivery.",
                "no actor-side history control is demonstrated: if the actor is given a "
                "git checkout whose history reaches past the base commit, the gold is in "
                "that history.",
            ],
        }
    return {
        "generated_by": "d0/build_ledger.py",
        "question": ("can an actor reconstruct the gold patch from what it is handed?"),
        "per_family": per_family,
        "summary": {
            "gold_patch_reaches_the_actor_in_any_family": False,
            "isolation_established": False,
            "honest_position": (
                "the oracle material is withheld from the public artifacts, but ISOLATION "
                "IS NOT ESTABLISHED: the upstream repositories are public and this "
                "delivery demonstrates no actor-side network or history control. Any "
                "consumer must treat these families as gold-withheld-but-publicly-"
                "discoverable until the custodian produces negative permission tests."),
            "owner_of_the_remaining_work": "custodian + Q0",
        },
    }


def actor_isolation():
    """The actor-side isolation gaps, declared instead of implied (yellow item 7)."""
    return {
        "generated_by": "d0/build_ledger.py",
        "scope": "the training/search actor, not the D0 author",
        "gaps": [
            {
                "id": "actor_network_access_not_controlled",
                "status": "unproven",
                "statement": (
                    "every counted family comes from a public upstream repository whose "
                    "history contains the original fix. An actor with network access can "
                    "find it. No sandbox, proxy rule or deny-list is demonstrated here, "
                    "and D0 is not the party that can demonstrate it."),
                "what_would_close_it": (
                    "a negative test showing the actor environment cannot reach the "
                    "upstream remote, plus a run recorded without network"),
                "owner": "custodian / E0 environment owner",
            },
            {
                "id": "actor_git_history_not_controlled",
                "status": "unproven",
                "statement": (
                    "the pinned clones here carry FULL history (they had to, to mine the "
                    "families). If the actor receives such a checkout, the fix commit is "
                    "reachable by `git log`. The actor's checkout has no demonstrated "
                    "history truncation and no demonstrated future-history control."),
                "what_would_close_it": (
                    "hand the actor a base-commit-only checkout (no descendants, no "
                    "future commits) and record a negative test that the fix is not "
                    "reachable from it, including via reflog, packed refs or remotes"),
                "owner": "custodian / E0 environment owner",
            },
            {
                "id": "restricted_oracle_split_not_adjudicated",
                "status": "unproven",
                "statement": (
                    "restricted-oracle.json still carries split_declaration_pending: a "
                    "custodian holding the gold material has not ruled which split its "
                    "contents belong to. Naming a file 'restricted' establishes nothing."),
                "what_would_close_it": (
                    "the custodian adjudicates the split and records who may read it"),
                "owner": "custodian",
            },
        ],
        "isolation_established": False,
        "note": ("none of these gaps blocks the static source-preparation work; all three "
                 "block the training-release gate, which is why training_release.status "
                 "is 'blocked' for every family"),
    }


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


def merge_record_for(index, fix_full):
    """Return the merge-evidence record for a fix commit, or None.

    Absence is reported as ``not_collected`` and then fails the qualification
    predicate -- it is never treated as "probably merged".
    """
    return index.get(fix_full)


def merge_qualifies(rec):
    return bool(rec) and rec.get("status") == "verified" \
        and rec.get("window_qualified_by_merge_event") is True


def merge_evidence_block(rec):
    """The per-family view of merge-evidence.json, with its provenance."""
    if not rec:
        return {
            "status": "not_collected",
            "window_qualified_by_merge_event": False,
            "reason": ("no merge-evidence record exists for this fix commit; per "
                       "policy it is counted in no window"),
            "artifact": "out/merge-evidence.json",
            "provenance": {
                "producer": "d0/fetch_merge_evidence.py",
                "metadata_source": "out/merge-evidence.json (GitHub REST API v3)",
            },
        }
    pr = rec.get("pull_request") or {}
    lookup = rec.get("commit_to_pr_lookup") or {}
    return {
        "status": rec.get("status"),
        "pull_request_number": pr.get("number"),
        "pull_request_url": rec.get("pull_request_url"),
        "merged_at_utc": rec.get("merged_at_utc"),
        "merge_commit_sha": rec.get("merge_commit_sha"),
        "merge_commit_sha_matches_fix_commit":
            rec.get("merge_commit_sha_matches_fix_commit"),
        "merge_commit_geometry": rec.get("merge_commit_geometry"),
        # THE landing-event answer. The reviewer asked for the PR head vs merge SHA
        # difference to be established rather than glossed, so the adjudicated landing
        # commit, the shape it took and the patch-equivalence result are promoted to the
        # top of the block instead of being buried in the geometry.
        "landing_event_adjudication": rec.get("landing_event_adjudication") or (
            rec.get("merge_commit_geometry") or {}).get("landing_event_adjudication"),
        "landing_event_shape": (rec.get("merge_commit_geometry") or {}).get(
            "landing_event_shape"),
        "adjudicated_landing_commit": (rec.get("merge_commit_geometry") or {}).get(
            "adjudicated_landing_commit"),
        "adjudicated_landing_is_ancestor_of_pinned_revision": (
            rec.get("merge_commit_geometry") or {}).get(
                "adjudicated_landing_is_ancestor_of_pinned_revision"),
        "landing_event_evidence": rec.get("landing_event_evidence"),
        "patch_equivalence": (rec.get("merge_commit_geometry") or {}).get(
            "patch_equivalence"),
        "window_qualified_by_merge_event": rec.get("window_qualified_by_merge_event"),
        "reason": rec.get("reason"),
        "provenance": {
            "producer": "d0/fetch_merge_evidence.py",
            "artifact": "out/merge-evidence.json",
            "commit_to_pr_lookup_url": lookup.get("url"),
            "pull_request_metadata_url": pr.get("url"),
            "raw_response_paths": [lookup.get("raw_path"), pr.get("raw_path")],
            "raw_response_sha256": {
                "commit_to_pr_lookup": lookup.get("raw_response_sha256"),
                "pull_request": pr.get("raw_response_sha256"),
            },
            "authentication": "unauthenticated; no token used",
        },
    }


def license_block(lock, name, lic):
    """Licence facts for a family, taken from source-lock so the two cannot drift."""
    review = lock["repos"][name]["license_review"]
    entry = [x for x in lic["by_repo"][name]
             if x["path"] == REPOS[name]["license_file"]]
    entry = entry[0] if entry else None
    return {
        "spdx": review["approved_spdx"],
        "spdx_is_derived": True,
        "license_file": entry["path"] if entry else REPOS[name]["license_file"],
        "license_file_checkout_sha256": entry["sha256"] if entry else None,
        "license_file_upstream_blob_sha256":
            entry.get("upstream_blob_sha256") if entry else None,
        "license_file_newline_transformation":
            entry.get("newline_transformation") if entry else None,
        "license_review_decision": review["decision"],
        "license_conflicts": review["license_conflicts"],
        "license_review_path": "repos.%s.license_review in source-lock.json" % name,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    lock = json.load(open(os.path.join(OUT, "source-lock.json"), encoding="utf-8"))
    lic = json.load(open(os.path.join(OUT, "license-files.json"), encoding="utf-8"))
    merge_doc = json.load(open(os.path.join(OUT, "merge-evidence.json"),
                               encoding="utf-8"))
    merge_index = {r["fix_commit"]: r for r in merge_doc["records"]}
    ledger = {
        "policy_id": POLICY_ID,
        "status_contract": {
            "contract_id": STATUS_CONTRACT_ID,
            "axes": ["source_preparation.status", "training_release.status"],
            "compatibility_field": "released",
            "compatibility_field_scope": SOURCE_PREPARATION_SCOPE,
            "contract": STATUS_CONTRACT_TEXT,
            "why_two_axes": (
                "the previous revision published a single `released` boolean that mixed a "
                "static-artefact statement with a release decision, so a reader could "
                "mistake 'the hashes and the merge event are complete' for 'this task may "
                "train an agent'. Splitting the axes makes the release gate explicit and "
                "keeps the old key working for consumers that already read it."),
        },
        "spec": {
            "train_window": {"end_exclusive": TRAIN_WINDOW_END,
                             "means": "family original fix time <= 2025-12-31"},
            "dev_sealed_window": {"start": DEVSEALED_START,
                                  "end_exclusive": DEVSEALED_END_EXCLUSIVE,
                                  "means": "family original fix time in 2026-01-01 .. 2026-10-04",
                                  "dev_vs_sealed_ordering_claimed": False},
            "withdrawn_windows": WITHDRAWN_WINDOWS,
            "time_axis_authority": (
                "a family's window is decided by the ORIGINAL upstream fix's MERGE "
                "event -- merged_at_utc of the pull request the fix commit belongs "
                "to, recorded with its metadata source and raw-response hash. The "
                "commit's own author and committer dates are kept ONLY as audit "
                "corroboration: a rebase, squash or re-land can rewrite them, so "
                "they never qualify a family. Release/tag dates, snapshot or pin "
                "dates, backport dates and cherry-pick dates are rejected "
                "substitutes."),
            "merge_evidence_artifact": {
                "path": "out/merge-evidence.json",
                "producer": "d0/fetch_merge_evidence.py",
                "policy": merge_doc["policy"]["window_qualification_basis"],
                "missing_merge_evidence": merge_doc["policy"]["missing_merge_evidence"],
            },
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

        merc = merge_record_for(merge_index, fix_full)
        merge_ok = merge_qualifies(merc)

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
            # THE window check: the original fix's merge event, not a commit date.
            "merge_event_evidence_present": bool(merc)
            and merc.get("status") == "verified",
            "merge_event_qualifies_window": merge_ok,
            # audit-only, kept so a reviewer can see the two dates corroborate
            "in_train_window_author_date_audit_only": adate < TRAIN_WINDOW_END,
            "in_train_window_committer_date_audit_only": cdate < TRAIN_WINDOW_END,
            "fix_commit_is_ancestor_of_pinned_revision": fix_is_ancestor,
            "fix_commit_is_not_a_backport_or_cherry_pick": backport_marker is None,
            "license_approved": lock["repos"][name]["license_review"]["decision"] == "approved",
            "license_approved_for_this_exact_revision":
                lock["repos"][name]["license_review"]["decided_against_revision"]
                == lock["repos"][name]["pinned_commit"],
            "license_metadata_has_no_conflict":
                not lock["repos"][name]["license_review"]["license_conflicts"],
        }
        fam = {
            "family_id": spec["family_id"],
            "kind": "real",
            "split": "train",
            "release_decision": {
                "window_decided_by": "merge_event" if merge_ok else "none",
                "blocking_checks": sorted(k for k, v in structural.items() if not v),
                "note": ("not ready: at least one structural check failed. Such a family is "
                         "counted in no window and is never promoted by a commit date."),
            },
            "repo": name,
            "replaces_fix_commit": spec.get("replaces"),
            "upstream_slug": REPOS[name]["slug"],
            "pinned_revision": lock["repos"][name]["pinned_commit"],
            "license": license_block(lock, name, lic),
            "base_commit": base_full,
            "oracle_fix_commit": fix_full,
            "oracle_patch_sha256": patch_sha,
            "merge_evidence": merge_evidence_block(merc),
            "fix_time": {
                "author_date": adate,
                "committer_date": cdate,
                "author_and_committer_dates_role": (
                    "audit corroboration ONLY. They are recorded so a reviewer can "
                    "see whether the commit dates agree with the merge event, and "
                    "they never qualify a family into a window."),
                "merge_event_utc": merc.get("merged_at_utc") if merc else None,
                "qualifies_by_merge_event": merge_ok,
                "primary": (merc.get("merged_at_utc") if merge_ok else None),
                "basis": ("merged_at_utc of the original upstream pull request the fix "
                          "commit belongs to; falls back to nothing when the merge "
                          "event could not be retrieved"),
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
                "the pinned revision decides which content exists; the family's own "
                "original merge event decides the window. The two axes are checked "
                "separately."),
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
            "provenance_note": (
                "replaces_fix_commit records the family this slot previously held. It is "
                "kept as a failure record only: the replaced commit stays unverified, is "
                "counted in no quota, and must not be read as a released example."
                if spec.get("replaces") else None),
        }
        ledger["families"].append(
            install_status_separation(fam, structural, base_full, fix_full, patch_sha, lic))

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
        vfam = {
            "family_id": spec["family_id"],
            "kind": "variant",
            "split": "train",
            "released": released,
            "released_scope": SOURCE_PREPARATION_SCOPE,
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
            "source_base_released": parent_fam["released"],
            "source_base_release_note": (
                "a variant can only inherit a window from a parent family whose source "
                "preparation is ready; here the parent is %s (source_preparation=%s)"
                % (parent_fam["family_id"],
                   parent_fam["source_preparation"]["status"])),
            "inherits_window_from": parent_fam["family_id"],
            "variant_commit": None,
            "oracle_fix_commit": None,
            "license": parent_fam["license"],
            "source_preparation": {
                "status": "specification_only",
                "basis": ("the mutation recipe and its upstream source base are pinned; the "
                          "mutation commit itself does not exist, so there is nothing to "
                          "prepare yet"),
                "blocking_checks": ["mutation_commit_not_constructed"],
            },
            "training_release": {
                "status": "blocked",
                "approved_for_training": False,
                "reason": ("blocked: a variant family is a construction recipe. It has no "
                           "commit, no oracle and no runtime evidence, so it can never be "
                           "training-released in this state."),
                "requires": dict(parent_fam["training_release"]["requires"]),
                "precondition_source_preparation": "ready" if parent_fam["released"]
                                                   else "not_ready",
                "owner": "Mika / custodian decision; D0 records the state only",
            },
            "status_axes": {
                "contract_id": STATUS_CONTRACT_ID,
                "contract": STATUS_CONTRACT_TEXT,
                "training_release_admissible": False,
                "consumer_rule": ("consume `training_release.status`, never `released`; "
                                  "`released` is a source-preparation alias"),
            },
        }
        ledger["families"].append(vfam)

    ledger["gold_derivability"] = gold_derivability(ledger["families"])
    ledger["actor_isolation"] = actor_isolation()

    with open(os.path.join(OUT, "family-ledger.json"), "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)

    print("families=%d" % len(ledger["families"]))
    for fam in ledger["families"]:
        if fam["kind"] == "real":
            bad = [k for k, v in fam["checks"].items() if not v]
            print("  %-40s %-14s prep=%-9s train=%-7s f2p=%d %s" % (
                fam["family_id"], fam["fix_date"][:10],
                fam["source_preparation"]["status"], fam["training_release"]["status"],
                len(fam["fail_to_pass"]), ("FAILED:" + ",".join(bad)) if bad else "ok"))
        else:
            print("  %-40s %-14s released=%s(%s) source_base=%s" % (
                fam["family_id"], fam["source_base_date"][:10], fam["released"],
                fam["mutation_class"], fam["source_base_commit"][:12]))


if __name__ == "__main__":
    main()
