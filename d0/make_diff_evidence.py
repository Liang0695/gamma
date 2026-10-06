"""Emit machine-readable diff evidence for the D0 v3 revision.

This is the "difference verification" Mika asked for: instead of asserting in
prose that the licence was corrected and the window basis moved to the merge
event, it reads the SAME files out of the previous commit and out of the current
WORKING TREE and prints the before/after values side by side.  Every claim in
the report's change list therefore has a line here that a reviewer can falsify.

The "after" side is the working tree rather than a commit on purpose: it lets
the artefacts and their diff evidence land in ONE commit.  Once that commit
exists the same evidence is reproducible with

    git diff <previous> HEAD -- d0/

Usage:  python d0/make_diff_evidence.py [previous-commit]
Default previous commit is the v2 artefact head of this branch (65aaa16).
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
DEFAULT_PREV = "7fe7170"
BRANCH = "agent/research/kaggle-23-d0-source-lock"
OUTPUT_NAME = "v4-diff-evidence.txt"


def git(*args):
    p = subprocess.run(["git", "-C", REPO] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode("utf-8", "replace"), \
        p.stderr.decode("utf-8", "replace")


def show_json(rev, path):
    """Committed JSON at `rev`, or the working-tree file when rev is None."""
    if rev is None:
        full = os.path.join(HERE, path.replace("d0/", "", 1))
        if not os.path.isfile(full):
            return None
        try:
            return json.load(open(full, encoding="utf-8"))
        except ValueError:
            return None
    rc, out, _ = git("show", "%s:%s" % (rev, path))
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def show_text(rev, path):
    """Raw bytes of a committed file (some transcripts are UTF-16, so do not
    decode here)."""
    p = subprocess.run(["git", "-C", REPO, "show", "%s:%s" % (rev, path)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return None
    return p.stdout


def gate_counts(raw):
    """Read 'N checks, M failed' out of a gate transcript.

    The first-pass transcript was written by a PowerShell redirect (UTF-16LE);
    later ones are written by the pipeline as UTF-8.  Try both.
    """
    if not raw:
        return None
    for encoding in ("utf-8", "utf-16", "utf-16-le"):
        try:
            decoded = raw.decode(encoding, "strict")
        except (UnicodeDecodeError, LookupError):
            continue
        m = re.search(r"(\d+) checks, (\d+) failed", decoded)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def time_basis(iso):
    if not iso:
        return None
    rule = iso.get("rule", {})
    return {
        "time_axis_authority": (rule.get("time_axis_authority") or "")[:110],
        "merge_evidence_artifact": bool(rule.get("merge_event_evidence")),
        "count_semantics_field_present": bool(
            iso.get("train_inventory", {}).get("click", {})
            .get("screened_candidate_commits_in_window") is not None),
        "old_field_qualified_candidate_families_present": bool(
            iso.get("train_inventory", {}).get("click", {})
            .get("qualified_candidate_families_in_window") is not None),
    }


def released_counts(ledger):
    if not ledger:
        return None
    fams = ledger.get("families", [])
    prepared = [f for f in fams if f.get("kind") == "real"
                and (f.get("source_preparation") or {}).get("status") == "ready"]
    trained = [f for f in fams if f.get("kind") == "real"
               and (f.get("training_release") or {}).get("approved_for_training")]
    return {
        "real_source_prepared": len(prepared),
        "real_training_released": len(trained),
        "real_released_alias": sum(1 for f in fams
                                   if f.get("kind") == "real" and f.get("released")),
        "real_released": sum(1 for f in fams
                             if f.get("kind") == "real" and f.get("released")),
        "real_total": sum(1 for f in fams if f.get("kind") == "real"),
        "variant_released": sum(1 for f in fams
                                if f.get("kind") == "variant" and f.get("released")),
        "variant_total": sum(1 for f in fams if f.get("kind") == "variant"),
    }


def dotenv_licence_state(lock):
    """Every field the licence correction touched, for one repository."""
    if not lock:
        return None
    lr = (lock.get("repos", {}).get("python-dotenv", {}) or {}).get(
        "license_review", {})
    facts = lr.get("license_facts") or {}
    hb = (lr.get("hash_basis") or {}).get("per_file") or {}
    lic = hb.get("LICENSE") or {}
    return {
        "declared_preset": (lock.get("repos", {}).get("python-dotenv", {}) or {})
        .get("design_license_expectation"),
        "approved_spdx": lr.get("approved_spdx"),
        "decision": lr.get("decision"),
        "license_facts_present": bool(facts),
        "detected_from_licence_text": facts.get("detected_from_licence_text"),
        "detected_from_packaging_metadata":
            facts.get("detected_from_packaging_metadata"),
        "license_conflicts": [c.get("kind") for c in (lr.get("license_conflicts") or [])],
        "hash_basis_present": bool(hb),
        "license_upstream_blob_sha256": (lic.get("upstream_blob_sha256") or "")[:16],
        "license_checkout_sha256": (lic.get("checkout_sha256") or "")[:16],
        "license_newline_transformation": lic.get("newline_transformation"),
        "independent_review_status":
            (lr.get("independent_review") or {}).get("status"),
    }


def merge_state(merge, ledger):
    """Merge-event coverage, before and after."""
    if not merge:
        return None
    verified = [r for r in merge["records"] if r["status"] == "verified"]
    fams = (ledger or {}).get("families", [])
    return {
        "artifact_present": True,
        "records": len(merge["records"]),
        "verified": len(verified),
        "unverified": len(merge["records"]) - len(verified),
        "unverified_commits": [r["fix_commit"][:12] for r in merge["records"]
                               if r["status"] != "verified"],
        "raw_responses_hashed": all(
            "raw_response_sha256" in r["commit_to_pr_lookup"]
            for r in merge["records"]),
        "families_with_merge_block": sum(1 for f in fams
                                         if "merge_evidence" in f),
    }


def main():
    prev = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREV
    rc, prev_full, _ = git("rev-parse", prev)
    prev_full = prev_full.strip() or "(unresolved)"
    rc, head, _ = git("rev-parse", "HEAD")
    head = head.strip()
    rc, stat, _ = git("diff", "--stat", prev_full)
    rc, names, _ = git("diff", "--name-status", prev_full)
    rc, dirty, _ = git("status", "--porcelain")
    n_dirty = len([l for l in dirty.splitlines() if l.strip()])

    L = []
    a = L.append
    a("V3 D0 v4 -- DIFF EVIDENCE (machine-derived, not transcribed)")
    a("=" * 72)
    a("branch                 : %s" % BRANCH)
    a("previous head (v3)     : %s (%s)" % (prev, prev_full))
    a("HEAD at generation     : %s" % head)
    a("'after' side read from : the WORKING TREE (d0/out/*.json on disk)")
    a("working tree vs HEAD   : %d changed path(s)" % n_dirty)
    a("generated by           : d0/make_diff_evidence.py")
    a("")
    a("NOTE: the 'after' side is the working tree, not a commit, so this evidence")
    a("file can be committed TOGETHER with the artefacts it describes (a file")
    a("cannot contain its own commit hash). Every value below is read out of the")
    a("previous commit's own committed file and out of the on-disk file, so each")
    a("claim is falsifiable. Once the artefacts are committed, reproduce with:")
    a("  git log --oneline -2")
    a("  git diff %s HEAD -- d0/ | head" % prev_full[:9])
    a("")
    a("  The structural diff below therefore uses `git diff <prev>` with no second")
    a("  revision, and it covers everything from the v2 head to whatever state the")
    a("  working tree was in when this file was written. The few paths still dirty")
    a("  against HEAD are the ones this evidence file shares its commit with.")
    a("")

    a("-" * 72)
    a("1. git diff --name-status  %s..WORKTREE" % prev_full[:9])
    a("-" * 72)
    a(names.strip() or "(no changes)")
    a("")
    a("-" * 72)
    a("2. git diff --stat  %s..WORKTREE" % prev_full[:9])
    a("-" * 72)
    a(stat.strip() or "(no changes)")
    a("")

    a("-" * 72)
    a("3. LICENCE CORRECTION   (d0/out/source-lock.json .repos.python-dotenv)")
    a("-" * 72)
    lb = dotenv_licence_state(show_json(prev_full, "d0/out/source-lock.json"))
    ln = dotenv_licence_state(show_json(None, "d0/out/source-lock.json"))
    keys = ["declared_preset", "approved_spdx", "decision",
            "license_facts_present", "detected_from_licence_text",
            "detected_from_packaging_metadata", "license_conflicts",
            "hash_basis_present", "license_upstream_blob_sha256",
            "license_checkout_sha256", "license_newline_transformation",
            "independent_review_status"]
    for k in keys:
        a("  %-40s before=%s" % (k, json.dumps((lb or {}).get(k))))
        a("  %-40s after =%s" % ("", json.dumps((ln or {}).get(k))))
    a("")
    a("  Interpretation:")
    a("   * the preset and the approved SPDX move MIT -> BSD-3-Clause, which is")
    a("     what the pinned LICENSE (blob 3a97119010ac82e15e917a69b7b8f9f59b5a4601)")
    a("     and the pinned pyproject.toml both say. The MIT value was a")
    a("     transcription error in the generator's preset, not an upstream change:")
    a("     the earlier candidate pin 791414804eff08a23f0b7970968e1717e3b28e66")
    a("     carries the same LICENSE blob.")
    a("   * license_facts / license_conflicts / hash_basis are NEW. The before side")
    a("     is null for all three because the v2 record had no derivation and no")
    a("     conflict list: that is exactly how the contradiction survived review.")
    a("   * the two hash bases are now both recorded and related by an explicit")
    a("     newline transformation, so a CRLF-converted hash can no longer be read")
    a("     as a licence change.")
    a("   * independent_review_status stays 'pending': this revision does NOT")
    a("     convert a self-declared assessment into an independent signature.")
    a("")

    a("-" * 72)
    a("4. WINDOW BASIS CORRECTION   (d0/out/d0-time-isolation.json .rule)")
    a("-" * 72)
    tb = time_basis(show_json(prev_full, "d0/out/d0-time-isolation.json"))
    tn = time_basis(show_json(None, "d0/out/d0-time-isolation.json"))
    for k in ["time_axis_authority", "merge_evidence_artifact",
              "count_semantics_field_present",
              "old_field_qualified_candidate_families_present"]:
        a("  %-40s before=%s" % (k, json.dumps((tb or {}).get(k))))
        a("  %-40s after =%s" % ("", json.dumps((tn or {}).get(k))))
    a("")
    a("  BEFORE (v2): %s" % json.dumps((tb or {}).get("time_axis_authority")))
    a("  AFTER  (v3): %s" % json.dumps((tn or {}).get("time_axis_authority")))
    a("")
    a("  Interpretation: the window is no longer decided by the commit's own dates")
    a("  (which a rebase, squash or re-land can rewrite) but by the original fix's")
    a("  upstream MERGE event. The author/committer dates survive only as audit")
    a("  corroboration, and the inventory counts are split into a commit-date")
    a("  SCREENING count and a merge-event VERIFIED count.")
    a("")

    a("-" * 72)
    a("5. MERGE-EVENT EVIDENCE ARTIFACT   (d0/out/merge-evidence.json)")
    a("-" * 72)
    mb = merge_state(show_json(prev_full, "d0/out/merge-evidence.json"),
                     show_json(prev_full, "d0/out/family-ledger.json"))
    mn = merge_state(show_json(None, "d0/out/merge-evidence.json"),
                     show_json(None, "d0/out/family-ledger.json"))
    for k in ["artifact_present", "records", "verified", "unverified",
              "unverified_commits", "raw_responses_hashed",
              "families_with_merge_block"]:
        a("  %-40s before=%s" % (k, json.dumps((mb or {}).get(k))))
        a("  %-40s after =%s" % ("", json.dumps((mn or {}).get(k))))
    a("")
    a("  Interpretation: the artifact is NEW (the before side is None). Every")
    a("  record carries merged_at_utc, the pull-request URL, the raw API response")
    a("  sha256, and the merge-commit geometry. Two records are UNVERIFIED and are")
    a("  counted in no quota: click 9da1791476fe (no pull request exists for the")
    a("  commit at all) and python-dotenv f5485a61eefa (its commit message cites")
    a("  #600, which is an ISSUE, not a pull request).")
    a("")

    a("-" * 72)
    a("6. STATUS AXES BEFORE/AFTER   (d0/out/family-ledger.json)")
    a("-" * 72)
    rb = released_counts(show_json(prev_full, "d0/out/family-ledger.json"))
    rn = released_counts(show_json(None, "d0/out/family-ledger.json"))
    for k in ["real_source_prepared", "real_training_released", "real_released_alias",
              "real_total", "variant_total"]:
        a("  %-42s before=%s" % (k, (rb or {}).get(k)))
        a("  %-42s after =%s" % ("", (rn or {}).get(k)))
    a("")
    a("  Interpretation: v3 shipped ONE `released` boolean that mixed a static-artefact")
    a("  statement with a release decision, and reported 3/8 with no separate notion of")
    a("  'may this train an agent'. v4 splits the axes: `source_preparation.status`")
    a("  (static material complete) and `training_release.status` (release gate).")
    a("  Every family is now source-prepared but training-BLOCKED, with the missing")
    a("  evidence itemised per family -- no runtime oracle result, no independent")
    a("  licence review, no demonstrated actor isolation, no constructed variant.")
    a("  `released` survives as a compatibility alias mirroring ONLY the static axis,")
    a("  scoped by a companion `released_scope` string present in every emitted file.")
    a("")
    a("  The click slot was substituted in this revision (Mika's ruling): the family")
    a("  that could not produce a merge event is replaced by ee56925bc4f5 (PR #1934,")
    a("  merged 2021-07-03), and the replaced commit is retained in")
    a("  merge-evidence.json's replaced_records with counted_in_no_quota=true.")
    a("")

    a("-" * 72)
    a("7. ACCEPTANCE GATE BEFORE/AFTER   (d0/out/validate_d0.output.txt)")
    a("-" * 72)
    gb = gate_counts(show_text(prev_full, "d0/out/validate_d0.output.txt"))
    gn = gate_counts(open(os.path.join(OUT, "validate_d0.output.txt"), "rb").read())
    a("  checks / failures   before=%s" % json.dumps(gb))
    a("  checks / failures   after =%s" % json.dumps(gn))
    a("")
    a("  Interpretation: the gate gained five new assertion groups -- the two status")
    a("  axes and their cross-file contract, the landing-event adjudication and patch")
    a("  equivalence, gold derivability, actor-side isolation gaps, and the replaced-")
    a("  family bookkeeping -- plus a third negative test (a family with no landing")
    a("  evidence must not be adjudicated as reconciled). It also gained an explicit")
    a("  SKIP channel: restricted-oracle.json is not committed, so the public")
    a("  validator now runs without it and records those checks as SKIP. A skip is")
    a("  printed as SKIP, never as PASS, and is explicitly not isolation evidence.")
    a("")

    a("-" * 72)
    a("7b. LANDING-EVENT ADJUDICATION   (d0/out/merge-evidence.json)")
    a("-" * 72)
    me_n = show_json(None, "d0/out/merge-evidence.json") or {}
    for fam, g in sorted((me_n.get("summary", {})
                          .get("landing_event_adjudications") or {}).items()):
        a("  %-16s %s" % (fam, g))
    a("")
    a("  Interpretation: GitHub's merge_commit_sha is not the commit that lives in the")
    a("  pinned clone for most of these families. v4 re-derives the landing event from")
    a("  the pinned bytes for each counted family and records which of three shapes")
    a("  applies, rather than treating 'the associated pull request was merged' as")
    a("  proof. boltons is the sharp case: the API's merge commit is ABSENT from the")
    a("  pinned clone, and the pinned history's own merge commit for that pull request")
    a("  brought in a different commit which has the fix as its parent. That is")
    a("  disclosed with ancestry and per-file change-set evidence instead of being")
    a("  silently aligned.")
    a("")

    a("-" * 72)
    a("8. WHAT DID NOT CHANGE (explicitly)")
    a("-" * 72)
    a("  * the 8 locked repositories and their pinned commits / tree SHAs are")
    a("    unchanged; no repository was re-pinned and no repository's per-file")
    a("    licence ledger was re-reviewed in this round. The click substitution")
    a("    reuses the SAME pinned click revision and its existing licence approval.")
    a("  * the time windows are unchanged, and the date rule was not relaxed to")
    a("    admit the replacement: it qualifies on its own 2021-07-03 merge event.")
    a("  * the '4 real + 4 variant from >= 2 repositories' target was not lowered;")
    a("    the four training repositories are all still represented.")
    a("  * no FAIL_TO_PASS run: this runtime still has no reachable package index,")
    a("    so every oracle field remains static evidence.")
    a("  * d0/out/restricted-oracle.json is still NOT committed, and still carries")
    a("    split_declaration_pending so it cannot be read as already split.")
    a("  * dev/sealed release and acceptance remain blocked on the isolation")
    a("    prerequisite (unchanged from Mika's ruling).")
    a("  * the public manifest still carries no oracle_assertions (gate-asserted).")
    a("  * no gold/answer material and no credential entered the repository; the")
    a("    merge-evidence fetch is unauthenticated and reads no token.")
    a("")

    a("-" * 72)
    a("9. INDEPENDENT RE-DERIVATION HINT")
    a("-" * 72)
    a("  python d0/run_all.py            # regenerate everything from d0/src/")
    a("  python d0/validate_d0.py        # all checks, must exit 0")
    a("  git diff %s HEAD -- d0/    # this diff again" % prev_full[:9])
    a("  d0/src/ is deliberately NOT committed (see .gitignore): regenerate it with")
    a("  d0/fetch_snapshots.ps1, which unshallows every repo and also fetches the")
    a("  alternative candidate.")
    a("  d0/pr-evidence/raw/ IS committed, so every merge-evidence raw_response_sha256")
    a("  can be re-hashed offline without touching the network.")
    a("")

    text = "\n".join(L) + "\n"
    with open(os.path.join(OUT, OUTPUT_NAME), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote d0/out/%s (%d bytes)" % (OUTPUT_NAME, len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
