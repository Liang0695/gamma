"""Derive the time-isolation evidence, the shortfall report, and the
public (gamma-committable) / restricted (custodian) manifest split.

--------------------------------------------------------------------------
TIME AXIS -- revised 2026-10-05 (policy v3-time-policy-2026-10-05-mika)
--------------------------------------------------------------------------
    train       : family original fix time <= 2025-12-31
    dev/sealed  : family original fix time in 2026-01-01 .. 2026-10-04
    dev and sealed share ONE window; no ordering between them is claimed.

The withdrawn (never approved) rule -- train < 2025-01-01, dev = calendar
2025, sealed >= 2026-01-01 -- is recorded in the output so no downstream
reader can mix the two policies.

Two axes are kept strictly separate and both are checked:
  * SNAPSHOT axis: the pinned revision decides which CONTENT exists
    (a family's fix commit must be an ancestor of the pin).
  * TIME axis: the family's own ORIGINAL fix time decides its window.
A release/tag date, a snapshot/pin date, a backport date and a cherry-pick
date are all rejected substitutes for the family fix time.
"""
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SRC = os.path.join(HERE, "src")

POLICY_ID = "v3-time-policy-2026-10-05-mika"
TRAIN_END_EXCLUSIVE = "2026-01-01T00:00:00+00:00"
DEVSEALED_START = "2026-01-01T00:00:00+00:00"
DEVSEALED_END_EXCLUSIVE = "2026-10-05T00:00:00+00:00"

TRAIN_REPOS = ["click", "more-itertools", "pluggy", "boltons"]
DEV_REPOS = ["attrs", "dateutil"]
SEALED_REPOS = ["packaging", "marshmallow"]
ALTERNATIVE_DEV = "python-dotenv"


def load(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f:
        return json.load(f)


def env_material(name):
    """Dependency material for a pinned revision, read from the checkout."""
    repo = os.path.join(SRC, name)
    material = {"files": [], "requires_python": None, "runtime_dependencies": [],
                "test_runner": None}
    for cand in ("pyproject.toml", "setup.cfg", "setup.py", "tox.ini",
                 "requirements.txt", "requirements-dev.txt", "Makefile",
                 "MANIFEST.in", ".pre-commit-config.yaml"):
        p = os.path.join(repo, cand)
        if os.path.isfile(p):
            raw = open(p, "rb").read()
            material["files"].append({
                "path": cand,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            })
    pp = os.path.join(repo, "pyproject.toml")
    if os.path.isfile(pp):
        txt = open(pp, encoding="utf-8", errors="replace").read()
        m = re.search(r'requires-python\s*=\s*"([^"]+)"', txt)
        if m:
            material["requires_python"] = m.group(1)
        for m in re.finditer(r'dependencies\s*=\s*\[(.*?)\]', txt, re.S):
            material["runtime_dependencies"] += [
                d.strip().strip('",\'') for d in m.group(1).splitlines()
                if d.strip().strip(',')]
    rt = os.path.join(repo, "requirements.txt")
    if os.path.isfile(rt):
        material["requirements_txt"] = [
            l.strip() for l in open(rt, encoding="utf-8", errors="replace").read().splitlines()
            if l.strip() and not l.strip().startswith("#")][:30]
    if os.path.isfile(os.path.join(repo, "tox.ini")) or \
       os.path.isfile(os.path.join(repo, "tests")):
        material["test_runner"] = "pytest"
    return material


def main():
    lock = load("source-lock.json")
    cand = load("family-candidates.json")
    ledger = load("family-ledger.json")
    merge = load("merge-evidence.json")
    merge_index = {r["fix_commit"]: r for r in merge["records"]}

    # ---------- 1. time isolation ----------
    iso = {
        "policy_id": POLICY_ID,
        "supersedes": ("v3-time-policy-d0-first-pass "
                       "(train<2025-01-01 / dev=2025 / sealed>=2026-01-01)"),
        "rule": {
            "axis_order": [
                "1. repository family -> split role (fixed by the integrated review)",
                "2. defect family within the repository (issue / PR / backport / variant)",
                "3. time window, evaluated last: a family may only be used by the split "
                "whose window contains its ORIGINAL fix time",
            ],
            "windows": {
                "train": {"start": None, "end_exclusive": TRAIN_END_EXCLUSIVE,
                          "means": "family original fix time <= 2025-12-31"},
                "dev_sealed": {"start": DEVSEALED_START,
                               "end_exclusive": DEVSEALED_END_EXCLUSIVE,
                               "means": "family original fix time in 2026-01-01 .. 2026-10-04",
                               "shared_by_roles": ["dev", "sealed"],
                               "dev_vs_sealed_ordering_claimed": False},
            },
            "withdrawn_windows": {
                "train_end_exclusive": "2025-01-01T00:00:00+00:00",
                "dev_start": "2025-01-01T00:00:00+00:00",
                "dev_end_exclusive": "2026-01-01T00:00:00+00:00",
                "sealed_start": "2026-01-01T00:00:00+00:00",
                "status": "withdrawn 2026-10-05 after Mika's ruling; never approved",
            },
            "time_axis_authority": (
                "the ORIGINAL upstream fix's MERGE event -- merged_at_utc of the pull "
                "request the fix commit belongs to -- recorded with its metadata source "
                "and raw-response sha256 in out/merge-evidence.json. The commit's own "
                "author and committer dates are audit corroboration ONLY and never "
                "qualify a family."),
            "merge_event_evidence": {
                "artifact": "out/merge-evidence.json",
                "producer": "d0/fetch_merge_evidence.py",
                "records": merge["summary"]["records"],
                "verified": merge["summary"]["verified"],
                "unverified": merge["summary"]["unverified"],
                "unverified_commits": merge["summary"]["unverified_commits"],
                "missing_evidence_rule": merge["policy"]["missing_merge_evidence"],
                "scope_caveat": (
                    "merge evidence has been collected for the families this delivery "
                    "INTENDS TO COUNT and for the alternative candidate's in-window "
                    "commits. The bulk per-repository inventory counts below are still "
                    "commit-date SCREENINGS: they are headroom observations, and no "
                    "individual count in them is window-qualified until that commit's "
                    "merge event is retrieved and verified."),
            },
            "rejected_time_substitutes": [
                "release/tag date", "snapshot or pin date",
                "backport date", "cherry-pick date",
            ],
            "enforcement": (
                "dev/sealed share one window and are separated by REPOSITORY ROLE, not by time. "
                "Within the train split the guarantee is two-sided: (a) a candidate whose "
                "original fix time falls in the dev/sealed window can never be downgraded into "
                "train, and (b) every train family's fix commit must be an ancestor of the "
                "pinned snapshot, so a family cannot be pulled in from outside the pin. "
                "Backport/cherry-pick commits are rejected outright, so a re-landed change "
                "cannot carry an older-original defect into a later window."),
            "snapshot_axis_is_separate": True,
        },
        "source_level_split": {},
        "train_inventory": {},
        "dev_sealed_inventory": {},
        "alternative_dev_candidate": {},
        "evidence": {},
        "open_items": [],
    }

    for name, rec in lock["repos"].items():
        if name == ALTERNATIVE_DEV:
            continue
        iso["source_level_split"][name] = {
            "role": rec["split_role"],
            "pinned_tag": rec["pinned_tag"],
            "pinned_commit": rec["pinned_commit"],
            "tree_sha": rec["tree_sha"],
            "snapshot_date": rec["commit_date"],
            "snapshot_date_role": (
                "SNAPSHOT axis only -- the pinned revision's own date. It is NOT a family fix "
                "time, it must never classify a family into a window, and no family in this "
                "delivery is dated by it."),
            "license_spdx": rec["design_license_expectation"],
            "license_review_decision": rec["license_review"]["decision"],
            "license_review_revision": rec["license_review"]["decided_against_revision"],
        }

    def inventory(names):
        out = {}
        for name in names:
            cs = cand["candidates_by_repo"].get(name, [])
            dates = sorted(c["fix_time_primary"] for c in cs)
            rec = lock["repos"][name]
            role = rec["split_role"]
            floor = TRAIN_END_EXCLUSIVE if role == "train" else DEVSEALED_START
            qualified = [c for c in cs if merge_index.get(c["fix_commit"], {})
                         .get("window_qualified_by_merge_event") is True]
            out[name] = {
                "role": role,
                "window": "train" if role == "train" else "dev_sealed",
                "screened_candidate_commits_in_window": len(cs),
                "oldest_fix_time": dates[0] if dates else None,
                "newest_fix_time": dates[-1] if dates else None,
                "merge_event_verified_and_in_window": len(qualified),
                "merge_event_verified_commits": sorted(
                    c["fix_commit"] for c in qualified),
                "count_semantics": (
                    "screened_candidate_commits_in_window is a COMMIT-DATE screening "
                    "count (both author and committer dates inside the window). It is "
                    "headroom evidence, NOT a qualified quota. "
                    "merge_event_verified_and_in_window is the only count that has the "
                    "merge-event evidence the time rule requires."),
                "all_fix_times_inside_window": all(
                    c["fix_time_both_dates_in_window"] for c in cs),
                "all_fix_commits_ancestors_of_pinned_revision": all(
                    c["is_ancestor_of_pinned_revision"] for c in cs),
                "pinned_snapshot_date": rec["commit_date"],
                "pinned_snapshot_predates_window": rec["commit_date"] < floor,
                "pinned_snapshot_date_not_used_as_family_time": True,
            }
        return out

    iso["train_inventory"] = inventory(TRAIN_REPOS)
    iso["dev_sealed_inventory"] = inventory(DEV_REPOS + SEALED_REPOS)

    total = sum(v["screened_candidate_commits_in_window"]
                for v in iso["train_inventory"].values())
    dev_total = sum(v["screened_candidate_commits_in_window"]
                    for v in iso["dev_sealed_inventory"].values()
                    if v["role"] == "dev")
    iso["evidence"]["train_candidate_total_screened_in_window"] = total
    iso["evidence"]["dev_candidate_total_screened_in_window"] = dev_total
    iso["evidence"]["train_candidate_total_merge_verified"] = sum(
        1 for c in cand["candidates_by_repo"].get("click", []) +
        cand["candidates_by_repo"].get("more-itertools", []) +
        cand["candidates_by_repo"].get("pluggy", []) +
        cand["candidates_by_repo"].get("boltons", [])
        if merge_index.get(c["fix_commit"], {}).get("window_qualified_by_merge_event")
        is True)
    iso["evidence"]["train_window_upper_bound_observed"] = max(
        (v["newest_fix_time"] for v in iso["train_inventory"].values()
         if v["newest_fix_time"]), default=None)
    iso["evidence"]["backport_marked_commits_observed_in_window"] = {
        k: v.get("observed_backport_marked_in_window", 0)
        for k, v in cand["dropped_by_repo"].items()}
    iso["evidence"]["backport_marked_candidates_excluded"] = \
        cand["dropped_totals"].get("excluded_backport_marker", 0)
    iso["evidence"]["no_family_time_derived_from_release_or_snapshot_date"] = True

    chosen = [f for f in ledger["families"] if f["kind"] == "real"]
    iso["evidence"]["chosen_real_families"] = [
        {"family_id": f["family_id"], "repo": f["repo"],
         "fix_time_author": f["fix_time"]["author_date"],
         "fix_time_committer": f["fix_time"]["committer_date"],
         "merge_event_utc": f["fix_time"]["merge_event_utc"],
         "qualifies_by_merge_event": f["fix_time"]["qualifies_by_merge_event"],
         "merge_evidence_status": f["merge_evidence"]["status"],
         "released": f["released"],
         "both_dates_in_train_window": f["fix_time"]["both_dates_in_train_window"],
         "is_ancestor_of_pinned_revision": f["fix_commit_is_ancestor_of_pinned_revision"],
         "backport_or_cherry_pick_marker": f["backport_or_cherry_pick_marker"],
         "base_commit": f["base_commit"], "oracle_fix_commit": f["oracle_fix_commit"]}
        for f in chosen]

    # ---------- 1b. alternative dev candidate (python-dotenv) ----------
    alt = cand["alternative_candidates"][ALTERNATIVE_DEV]
    alt_lock = lock["repos"][ALTERNATIVE_DEV]

    def alt_family(c):
        merc = merge_index.get(c["fix_commit"]) or {}
        pr = merc.get("pull_request") or {}
        qualified = merc.get("window_qualified_by_merge_event") is True
        return {
            "fix_commit": c["fix_commit"], "base_commit": c["base_commit"],
            "fix_time_author": c["fix_time_primary"],
            "fix_time_committer": c["fix_time_committer"],
            "fix_time_author_and_committer_role":
                "audit corroboration only; not the window basis",
            "is_ancestor_of_pinned_revision": c["is_ancestor_of_pinned_revision"],
            "merge_evidence": {
                "status": merc.get("status", "not_collected"),
                "pull_request": merc.get("pull_request_url"),
                "merged_at_utc": merc.get("merged_at_utc"),
                "merge_commit_sha": merc.get("merge_commit_sha"),
                "merge_commit_sha_matches_fix_commit":
                    merc.get("merge_commit_sha_matches_fix_commit"),
                "window_qualified_by_merge_event": qualified,
                "reason": merc.get("reason"),
                "metadata_source_url": pr.get("url"),
                "raw_response_sha256": {
                    "commit_to_pr_lookup": (merc.get("commit_to_pr_lookup") or {})
                    .get("raw_response_sha256"),
                    "pull_request": pr.get("raw_response_sha256"),
                },
                "artifact": "out/merge-evidence.json",
            },
            "counted_in_qualified_quota": qualified,
            "test_files": c["test_files"], "src_files": c["src_files"],
            "lines_changed": c["lines_changed"],
        }

    alt_families = [alt_family(c) for c in alt["candidates"]]
    alt_qualified = [f for f in alt_families if f["counted_in_qualified_quota"]]
    alt_qualified_dates = sorted(f["merge_evidence"]["merged_at_utc"]
                                 for f in alt_qualified)
    iso["alternative_dev_candidate"] = {
        "repo": ALTERNATIVE_DEV,
        "upstream_slug": alt_lock["upstream_slug"],
        "role": alt_lock["split_role"],
        "approval_status": (
            "approved as an ALTERNATIVE CANDIDATE only (license/family/environment "
            "verification). It is NOT a replacement for dateutil and NOT approved for "
            "release until the custodian rules."),
        "replaces": None,
        "pinned_tag": alt_lock["pinned_tag"],
        "pinned_commit": alt_lock["pinned_commit"],
        "commit_matches_plan": alt_lock["commit_matches_plan"],
        "snapshot_date": alt_lock["commit_date"],
        "snapshot_date_role": "SNAPSHOT axis only; not a family fix time",
        "license": {
            "decision": alt_lock["license_review"]["decision"],
            "approved_spdx": alt_lock["license_review"]["approved_spdx"],
            "spdx_correction": (
                "corrected 2026-10-06 from a mistaken MIT preset to BSD-3-Clause, "
                "which is what the pinned LICENSE and pyproject.toml both say; this "
                "preset was a transcription error, not a licence change upstream"),
            "license_conflicts": alt_lock["license_review"]["license_conflicts"],
            "license_facts": alt_lock["license_review"]["license_facts"],
            "primary_license_file": alt_lock["license_review"]["evidence"][
                "primary_license_file"],
            "hash_basis": alt_lock["license_review"]["hash_basis"]["per_file"],
            "copyleft_marker_hits": alt_lock["license_review"]["copyleft_marker_hits"],
            "restrictive_marker_hits": alt_lock["license_review"]["restrictive_marker_hits"],
            "path_in_source_lock": "repos.%s.license_review" % ALTERNATIVE_DEV,
        },
        "family_inventory": {
            "window": "dev_sealed (2026-01-01 .. 2026-10-04)",
            "count_semantics": (
                "screened_candidate_commits_in_window is a commit-date screening "
                "count of headroom. merge_event_verified_and_in_window is the only "
                "count backed by the merge-event evidence the time rule requires."),
            "screened_candidate_commits_in_window": len(alt["candidates"]),
            "merge_event_verified_and_in_window": len(alt_qualified),
            "oldest_merge_event_utc": alt_qualified_dates[0] if alt_qualified_dates else None,
            "newest_merge_event_utc": alt_qualified_dates[-1] if alt_qualified_dates else None,
            "families": alt_families,
        },
        "environment_material": env_material(ALTERNATIVE_DEV),
        "open_items": [
            "custodian has not ruled on replacing dateutil; this record does not replace it",
            "no FAIL_TO_PASS run was executed for any of its families",
            ("one in-window commit (f5485a61eefa5e686d6d5bdc7aa9ad6b104b1e92) has NO "
             "retrievable merge event: its commit message references #600, which is an "
             "ISSUE, not a pull request. It is unverified and counted in no quota."),
        ],
    }

    # ---------- 1c. open items ----------
    risks = []
    for name in DEV_REPOS + SEALED_REPOS:
        rec = lock["repos"][name]
        role = rec["split_role"]
        n = iso["dev_sealed_inventory"][name]["screened_candidate_commits_in_window"]
        if n == 0:
            risks.append({
                "repo": name,
                "role": role,
                "observed_family_candidates_in_window": 0,
                "pinned_revision_date": rec["commit_date"],
                "required_window_start": DEVSEALED_START,
                "status": ("cannot supply families in its window at the pinned revision: the "
                           "pinned snapshot itself predates the window"),
                "consequence": ("any %s family drawn from this pinned revision would be older "
                                "than %s; qualifying it would require re-pinning, which "
                                "invalidates the per-file licence approval bound to this "
                                "revision" % (role, DEVSEALED_START)),
            })
    iso["open_items"] = risks
    with open(os.path.join(OUT, "d0-time-isolation.json"), "w", encoding="utf-8") as f:
        json.dump(iso, f, indent=2, ensure_ascii=False)

    # ---------- 2. shortfall ----------
    real = [f for f in ledger["families"] if f["kind"] == "real"]
    variant = [f for f in ledger["families"] if f["kind"] == "variant"]
    repos_covered = sorted({f["repo"] for f in ledger["families"]})
    shortfall = {
        "policy_id": POLICY_ID,
        "p0": {
            "requirement": "8 training families = 4 real + 4 variant from >= 2 repositories",
            "real_families_released": sum(1 for f in real if f["released"]),
            "real_families_total": len(real),
            "real_families_blocked_on_window_evidence": [
                f["family_id"] for f in real if not f["released"]],
            "variant_specs_ready": sum(1 for f in variant if f["mutation_recipe"]),
            "variant_commits_built": 0,
            "repositories_covered": repos_covered,
            "verdict": ("real half SHORT: %d/%d released under the merge-event rule. "
                        "v3-train-click-001 has no retrievable merge event (the 2015 fix "
                        "was pushed straight to main; its issue #222 was closed by a "
                        "direct commit reference and pull requests #258/#259 were closed "
                        "UNMERGED), so it is released=false and counted in no quota. "
                        "Variant half has 4/4 specifications but zero constructed "
                        "commits." % (sum(1 for f in real if f["released"]), len(real))),
        },
        "p1_headroom": {
            "requirement": "train split of 96 families = 24 real + 40 variant",
            "screened_candidate_commits_in_train_window": total,
            "merge_event_verified_train_candidates":
                iso["evidence"]["train_candidate_total_merge_verified"],
            "per_repo": {k: v["screened_candidate_commits_in_window"]
                         for k, v in iso["train_inventory"].items()},
            "assessment": ("%d commit-date SCREENED candidates against a need for 24 real "
                           "families is nominal headroom of about %.1fx, but the count is a "
                           "screening, not a qualified quota: each candidate still needs its "
                           "merge event retrieved (%d have one today), a verifier run, an "
                           "actor-reachability check and a bounded test patch. Do not read "
                           "%d as 24 usable."
                           % (total, total / 24.0,
                              iso["evidence"]["train_candidate_total_merge_verified"],
                              total)),
            "weakest_repo": "more-itertools (%d screened candidates) -- thin if it must carry "
                            "a share" % iso["train_inventory"]["more-itertools"][
                                "screened_candidate_commits_in_window"],
        },
        "dev_supply": {
            "requirement": ("a dev split needs families whose original fix time is in "
                            "2026-01-01 .. 2026-10-04, from the dev-role repositories"),
            "screened_dev_candidates_found": dev_total,
            "per_repo": {k: v["screened_candidate_commits_in_window"]
                         for k, v in iso["dev_sealed_inventory"].items()},
            "dateutil_verdict": ("0 qualified dev families. dateutil's pinned revision is dated "
                                 "2024-02-29, entirely before the window, so no family in that "
                                 "pinned revision can be a dev family. This confirms Mika's "
                                 "provisional count of 0."),
            "attrs_observation": ("attrs is the only locked dev-role repository that does yield "
                                  "in-window dev families. The count is thin and unverified; "
                                  "it is headroom evidence, not a released dev set."),
            "assessment": "dev family supply is thin and depends on attrs plus an alternative.",
        },
        "blocking_gaps": [],
        "non_blocking_gaps": [],
    }

    if any(f["repo"] == "dateutil" for f in real):
        shortfall["blocking_gaps"].append("dateutil used as a train source (unexpected)")
    for f in real:
        if f["released"]:
            continue
        me = f["merge_evidence"]
        shortfall["blocking_gaps"].append({
            "gap": ("%s is released=false: the original fix's MERGE event could not be "
                    "retrieved, and the time rule requires the merge event, not a "
                    "commit date" % f["family_id"]),
            "impact": ("the P0 real half is %d/4 instead of 4/4, so the 8-task set is "
                       "short by one REAL family until this is ruled on"
                       % sum(1 for x in real if x["released"])),
            "evidence": {
                "family_id": f["family_id"],
                "repo": f["repo"],
                "fix_commit": f["oracle_fix_commit"],
                "merge_evidence_status": me["status"],
                "commit_to_pr_lookup_url": me["provenance"].get(
                    "commit_to_pr_lookup_url"),
                "author_date_audit_only": f["fix_time"]["author_date"],
                "committer_date_audit_only": f["fix_time"]["committer_date"],
                "extra_observation": (
                    "the two pull requests cross-referenced from the upstream issue "
                    "(pallets/click#258 and #259) were both closed UNMERGED, so no "
                    "pull-request merge event exists for this fix at all"),
            },
            "options": [
                "accept a defined alternative landing-event evidence standard for "
                "direct-to-default-branch pushes (a policy decision, not taken here)",
                "or substitute another train-split family whose merge event IS "
                "retrievable, from the screened inventory, after running the same "
                "derivation and review",
                "or drop the 4th real family and record the P0 quota as short",
            ],
            "owner": "Mika / custodian decision; D0 does not silently swap a family",
        })
    for r in risks:
        shortfall["non_blocking_gaps"].append({
            "gap": ("%s cannot carry %s families at the pinned revision under the time rule"
                    % (r["repo"], r["role"])),
            "evidence": ("pinned snapshot %s < required window start %s"
                         % (r["pinned_revision_date"], r["required_window_start"])),
            "options": [
                "drop it from the %s role and let the remaining dev/ sealed-role repositories "
                "carry that split" % r["role"],
                "or re-pin it to a revision inside the window and redo the per-file licence "
                "ledger AND the licence approval for the new revision",
                "or approve the already-named alternative candidate if the custodian rules "
                "that way",
            ],
            "owner": "needs a decision from Mika / the custodian, not taken here",
        })
    shortfall["non_blocking_gaps"].append({
        "gap": ("dev/sealed family inventories exist but are unverified headroom, and only one "
                "locked dev-role repository (attrs) yields in-window dev families"),
        "evidence": ("counts in dev_supply above; every candidate still needs its merge "
                     "event retrieved, a verifier run and an actor-reachability check"),
        "options": ["treat attrs plus an approved alternative as the dev supply, or re-pin"],
        "owner": "D0 follow-up / custodian decision",
    })
    shortfall["non_blocking_gaps"].append({
        "gap": ("the bulk per-repository inventory counts are commit-date SCREENINGS, not "
                "window-qualified families"),
        "evidence": ("out/d0-time-isolation.json records screened_candidate_commits_in_window "
                     "and merge_event_verified_and_in_window separately; merge events were "
                     "retrieved only for the families this delivery intends to count and for "
                     "the alternative candidate's in-window commits (%d train candidates "
                     "verified so far)"
                     % iso["evidence"]["train_candidate_total_merge_verified"]),
        "options": ["fetch the merge event for a candidate before promoting it, or keep it "
                    "labelled as headroom"],
        "owner": "D0 follow-up; no quota claim is made from a screening count",
    })
    shortfall["non_blocking_gaps"].append({
        "gap": "no FAIL_TO_PASS run was executed",
        "evidence": ("the research runtime has no reachable package index, so pytest could not "
                     "be installed; every oracle field here is static evidence"),
        "options": ["E0 builds the environment and runs the four oracles as its T0 deliverable"],
        "owner": "E0 (coding officer)",
    })
    shortfall["non_blocking_gaps"].append({
        "gap": "licence approval is self-declared and not independently countersigned",
        "evidence": ("repos.<name>.license_review.independent_review.status == 'pending' for "
                     "every repository; the decision field is machine-readable but its author "
                     "is the D0 owner"),
        "options": ["Q0 countersigns or refutes the license_review records"],
        "owner": "Q0",
    })
    shortfall["non_blocking_gaps"].append({
        "gap": "licence of the upstream issue/PR *prose* was not cleared",
        "evidence": ("the candidate repositories are code-licensed; a report body or comment "
                     "thread is a separate work. The design already requires rewritten task "
                     "text, and this ledger copies no prose, so nothing is blocked -- but the "
                     "rewritten text must be reviewed as original writing"),
        "options": ["keep the rewrite policy; have Q0 spot-check the task texts for paraphrase"],
        "owner": "Q0",
    })
    with open(os.path.join(OUT, "d0-shortfall.json"), "w", encoding="utf-8") as f:
        json.dump(shortfall, f, indent=2, ensure_ascii=False)

    # ---------- 3. public vs restricted split ----------
    GOLD_KEYS = {"oracle_assertions"}
    public_families = []
    restricted_families = []
    for f in ledger["families"]:
        pub = {k: v for k, v in f.items() if k not in GOLD_KEYS}
        public_families.append(pub)
        restricted_families.append({
            "family_id": f["family_id"],
            "restricted_reason": "oracle-side hint; must not reach a training or search author",
            "oracle_assertions": f.get("oracle_assertions") or [],
        })

    sealed = set(SEALED_REPOS)
    public_lock = {
        "generated_by": "d0/build_extras.py",
        "policy_id": POLICY_ID,
        "note": ("Public face of the D0 source lock. Contains only metadata, hashes and "
                 "identifiers from permissively licensed repositories. No gold patch body, no "
                 "sealed-split source text, no credentials."),
        "repos": {k: v for k, v in lock["repos"].items()
                  if k not in sealed and k != ALTERNATIVE_DEV},
        "sealed_repos_metadata_only": {
            k: {
                "upstream_slug": v["upstream_slug"],
                "pinned_tag": v["pinned_tag"],
                "pinned_commit": v["pinned_commit"],
                "tree_sha": v["tree_sha"],
                "commit_date": v["commit_date"],
                "license_spdx": v["design_license_expectation"],
                "license_review": v["license_review"],
                "content_committed": False,
                "reason": "sealed split: keep content and families out of the public manifest",
            } for k, v in lock["repos"].items() if k in sealed},
        "alternative_candidates_metadata": {
            ALTERNATIVE_DEV: {
                "upstream_slug": lock["repos"][ALTERNATIVE_DEV]["upstream_slug"],
                "pinned_tag": lock["repos"][ALTERNATIVE_DEV]["pinned_tag"],
                "pinned_commit": lock["repos"][ALTERNATIVE_DEV]["pinned_commit"],
                "commit_date": lock["repos"][ALTERNATIVE_DEV]["commit_date"],
                "license_review": lock["repos"][ALTERNATIVE_DEV]["license_review"],
                "status": ("alternative candidate only; not a replacement, not approved for "
                           "release"),
                "content_committed": False,
            }},
        "families": public_families,
        "time_isolation": iso["rule"],
        "license_approvals": {},
        "license_review_schema": lock["license_review_schema"],
    }
    for name, rec in lock["repos"].items():
        if name in sealed:
            continue
        review = rec["license_review"]
        public_lock["license_approvals"][name] = {
            # machine-readable approval, taken straight from source-lock.json so the two
            # files can never disagree
            "schema_version": "1.1",
            "decision": review["decision"],
            "approval": "approved" if review["decision"] == "approved" else review["decision"],
            "spdx": review["approved_spdx"],
            "spdx_is_derived": True,
            "license_facts": review["license_facts"],
            "license_conflicts": review["license_conflicts"],
            "osi_permissive": review["osi_permissive"],
            "decided_against_revision": review["decided_against_revision"],
            "copyleft_marker_hits": review["copyleft_marker_hits"],
            "restrictive_marker_hits": review["restrictive_marker_hits"],
            "independent_review": review["independent_review"],
            "source_of_truth": "repos.%s.license_review in source-lock.json" % name,
            "basis": review["decision_basis"],
            "primary_license_file": review["evidence"]["primary_license_file"],
            "all_license_files": review["evidence"]["all_license_file_hashes"],
            "hash_basis": review["hash_basis"],
            "declared_in_packaging_metadata": rec["packaging_metadata"],
            "per_file_scan": review["evidence"]["per_file_scan"],
        }
    with open(os.path.join(OUT, "public-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(public_lock, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "restricted-oracle.json"), "w", encoding="utf-8") as f:
        json.dump({
            "generated_by": "d0/build_extras.py",
            "policy_id": POLICY_ID,
            "handling": ("custodian/Q0 and E0 only. Not for the training author, not for the "
                         "search author, and not to be committed to the shared gamma repo."),
            "split_declaration_pending": (
                "the split of THIS file's contents must be decided by a custodian that holds "
                "the gold material. This delivery does not claim the file is already safely "
                "split or already uncontaminated."),
            "families": restricted_families,
        }, f, indent=2, ensure_ascii=False)

    print("time-isolation open items:", [r["repo"] for r in risks])
    print("train candidates SCREENED by commit date:", total, "| per repo:",
          {k: v["screened_candidate_commits_in_window"]
           for k, v in iso["train_inventory"].items()})
    print("train candidates MERGE-EVENT verified:",
          iso["evidence"]["train_candidate_total_merge_verified"])
    print("dev/sealed inventory (screened):",
          {k: v["screened_candidate_commits_in_window"]
           for k, v in iso["dev_sealed_inventory"].items()})
    print("alternative dev candidate families: screened=%d merge-verified=%d" % (
        iso["alternative_dev_candidate"]["family_inventory"][
            "screened_candidate_commits_in_window"],
        iso["alternative_dev_candidate"]["family_inventory"][
            "merge_event_verified_and_in_window"]))
    print("blocking gaps:", [g if isinstance(g, str) else g["gap"][:70]
                             for g in shortfall["blocking_gaps"]])
    print("public manifest families:", len(public_families))
    print("restricted families:", len(restricted_families))


if __name__ == "__main__":
    main()
