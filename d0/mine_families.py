"""Mine SWE-bench-style candidate families from every pinned candidate source.

A candidate family needs, at a minimum:
  * a fix commit that changes BOTH package source and test code
  * a commit message that references an upstream issue / PR number
  * a bounded diff (so the oracle patch stays reviewable)
  * a fix time inside the declared split window for that repository's role

--------------------------------------------------------------------------
TIME AXIS -- revised 2026-10-05, supersedes the first D0 pass
--------------------------------------------------------------------------
    train       : family original fix time <= 2025-12-31
                  (end exclusive 2026-01-01T00:00:00+00:00)
    dev/sealed  : 2026-01-01 .. 2026-10-04 inclusive
                  (start 2026-01-01T00:00:00+00:00, end exclusive 2026-10-05T00:00:00+00:00)
    dev and sealed share ONE window.  No ordering between dev and sealed is
    claimed -- they are separated by repository role, not by time.

The withdrawn (never approved) rule was train < 2025-01-01 / dev = calendar
2025 / sealed >= 2026-01-01.  It is recorded as withdrawn so that no downstream
reader can mix the two policies.

A family's WINDOW is decided by the ORIGINAL upstream fix's MERGE event -- the
merged_at_utc of the pull request the fix commit belongs to (see
fetch_merge_evidence.py / out/merge-evidence.json).  The commit's own author and
committer dates are recorded TWICE but as AUDIT CORROBORATION ONLY: a rebase,
squash or re-land can rewrite them, so they never qualify a family.  Release
tag dates, snapshot/pin dates, backport dates and cherry-pick dates are
explicitly NOT accepted as a family's time either:

  * the fix commit must be an ANCESTOR of the pinned snapshot, so the pin
    decides which content exists but never what time a family belongs to;
  * a commit whose message carries a backport / cherry-pick marker is rejected
    outright and the marker is recorded, so a re-landed change cannot smuggle
    an older-original defect into a later window.

Output: d0/out/family-candidates.json  (+ printed shortlist)
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "out")

# ---------------------------------------------------------------------------
# split windows (single source of truth for this script; build_ledger.py,
# build_extras.py and validate_d0.py carry the same constants and the gate
# re-asserts them against the emitted JSON)
# ---------------------------------------------------------------------------
POLICY_ID = "v3-time-policy-2026-10-05-mika"
TRAIN_END_EXCLUSIVE = "2026-01-01T00:00:00+00:00"   # == fix time <= 2025-12-31
DEVSEALED_START = "2026-01-01T00:00:00+00:00"
DEVSEALED_END_EXCLUSIVE = "2026-10-05T00:00:00+00:00"   # == through 2026-10-04
WITHDRAWN_WINDOWS = {
    "train_end_exclusive": "2025-01-01T00:00:00+00:00",
    "dev_start": "2025-01-01T00:00:00+00:00",
    "dev_end_exclusive": "2026-01-01T00:00:00+00:00",
    "sealed_start": "2026-01-01T00:00:00+00:00",
}

WINDOWS = {
    "train": (None, TRAIN_END_EXCLUSIVE),
    "dev": (DEVSEALED_START, DEVSEALED_END_EXCLUSIVE),
    "sealed": (DEVSEALED_START, DEVSEALED_END_EXCLUSIVE),
}

# ---------------------------------------------------------------------------
# the 8 candidate repositories fixed by the integrated review, each mined at
# its pinned revision (never at a moving branch head)
# ---------------------------------------------------------------------------
SOURCES = {
    "click": ("train", "8b19813f2bfca99f1018a587a8cf54fc959f2e5d"),
    "more-itertools": ("train", "64be96ceb2a6e836f76f069f4a96d2394d59fd0c"),
    "pluggy": ("train", "fd08ab5f811a9b2fa9124ae8cbbd393221151e2c"),
    "boltons": ("train", "4332b35a278d694f30c99881faa61cde695c7a96"),
    "attrs": ("dev", "7bfc49e9b22d5ba25b6e429524c3d49fee27cb36"),
    "dateutil": ("dev", "db9d018944c41ddc740015cf5f64717c2ba64a5c"),
    "packaging": ("sealed", "929fd4b1410ac7ef61ef3f45b2f5d7e87711a9b5"),
    "marshmallow": ("sealed", "c7b559a1fa3aba57ca6dba0ab336841c5038a782"),
}

# approved ALTERNATIVE candidate for the dev slot (license/family/environment
# verification only -- not a 9th member of the locked set, and not a replacement
# until the custodian says so)
ALTERNATIVES = {
    "python-dotenv": {
        "role": "dev",
        "pinned_tag": "v1.2.4",
        "pinned_commit": "a565c2cc41599c48eabc6b7b7f5b826d43c5a6d7",
        "upstream_slug": "theskumar/python-dotenv",
        "status": "candidate_under_verification",
        "replaces": None,
        "note": ("Mika approved this only as an alternative dev candidate to be "
                 "license/family/environment-verified; it does not replace dateutil "
                 "and is not approved for release."),
    },
}

ISSUE_RE = re.compile(r"(?:#|GH-|gh-)(\d{2,6})")
FIXWORD_RE = re.compile(r"\b(fix(?:es|ed)?|clos(?:e|es|ed)|resolv(?:e|es|ed)|bug)\b", re.I)
TEST_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$", re.I)
SRC_RE = re.compile(r"\.pyi?$")

# A re-landed change must never be counted as a family in its own right, and
# its own commit time must never stand in for the original fix time.
BACKPORT_RE = re.compile(
    r"cherry[- ]?picked?\s+from|cherry[- ]?pick\s+of|back[- ]?port|"
    r"backport|cherry-pick|re-?land",
    re.I,
)


def parse_iso(s):
    return dt.datetime.fromisoformat(s)


def in_window(iso, role):
    """True when this ISO timestamp lies inside the role's window."""
    lo, hi = WINDOWS[role]
    t = parse_iso(iso)
    if lo is not None and t < parse_iso(lo):
        return False
    if hi is not None and t >= parse_iso(hi):
        return False
    return True


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def numstat(repo, sha):
    rc, out, _ = git(repo, "show", "--numstat", "--format=", "--no-renames", sha)
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        add, dele, path = parts
        rows.append({
            "path": path,
            "add": None if add == "-" else int(add),
            "del": None if dele == "-" else int(dele),
        })
    return rows


def is_binaryish(r):
    return r["add"] is None


def is_ancestor(repo, sha, ref):
    rc, _, _ = git(repo, "merge-base", "--is-ancestor", sha, ref)
    return rc == 0


def mine(name, role, ref):
    repo = os.path.join(SRC, name)
    rc, out, err = git(repo, "log", "--no-merges",
                       "--format=%H%x1f%P%x1f%aI%x1f%cI%x1f%s%x1f%b%x1e", ref)
    if rc != 0:
        print("log failed for %s: %s" % (name, err[:200]))
        return [], {"excluded_out_of_window": 0, "observed_backport_marked_in_window": 0,
                    "excluded_backport_marker": 0,
                    "excluded_merged_parents": 0, "excluded_no_fixword": 0,
                    "excluded_no_issue_ref": 0, "excluded_binary": 0,
                    "excluded_no_test_or_src": 0, "excluded_too_large": 0}
    cands = []
    dropped = {
        "excluded_out_of_window": 0, "observed_backport_marked_in_window": 0,
        "excluded_backport_marker": 0,
        "excluded_merged_parents": 0, "excluded_no_fixword": 0,
        "excluded_no_issue_ref": 0, "excluded_binary": 0,
        "excluded_no_test_or_src": 0, "excluded_too_large": 0,
    }
    for rec in out.split("\x1e"):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        fields = rec.split("\x1f")
        if len(fields) < 6:
            continue
        sha, parents, adate, cdate, subject, body = fields[:6]
        # BOTH dates must be inside the window: the classification must not hang
        # on a single date that a rebase or a cherry-pick could rewrite.
        if not in_window(adate, role) or not in_window(cdate, role):
            dropped["excluded_out_of_window"] += 1
            continue
        text = subject + "\n" + body
        marker = BACKPORT_RE.search(text)
        if marker:
            # counted across EVERY in-window commit, not only the ones that would
            # otherwise have qualified, so the counter is a real observation
            dropped["observed_backport_marked_in_window"] += 1
        parents = parents.split()
        if len(parents) != 1:
            dropped["excluded_merged_parents"] += 1
            continue
        issues = sorted(set(ISSUE_RE.findall(text)), key=int)
        if not FIXWORD_RE.search(text):
            dropped["excluded_no_fixword"] += 1
            continue
        if not issues:
            dropped["excluded_no_issue_ref"] += 1
            continue
        rows = numstat(repo, sha)
        if not rows or any(is_binaryish(r) for r in rows):
            dropped["excluded_binary"] += 1
            continue
        tests = [r for r in rows if TEST_RE.search(r["path"])]
        srcs = [r for r in rows if (not TEST_RE.search(r["path"])) and SRC_RE.search(r["path"])]
        if not tests or not srcs:
            dropped["excluded_no_test_or_src"] += 1
            continue
        changed = sum((r["add"] or 0) + (r["del"] or 0) for r in rows)
        if changed > 200 or len(rows) > 10:
            dropped["excluded_too_large"] += 1
            continue
        if marker:
            dropped["excluded_backport_marker"] += 1
            continue
        cands.append({
            "repo": name,
            "role": role,
            "fix_commit": sha,
            "base_commit": parents[0],
            "fix_time_author": adate,
            "fix_time_committer": cdate,
            "fix_time_primary": adate,
            "fix_time_basis": "author date of the original upstream fix commit",
            "fix_time_both_dates_in_window": True,
            "is_ancestor_of_pinned_revision": is_ancestor(repo, sha, ref),
            "backport_or_cherry_pick_marker": None,
            "subject": subject,
            "issue_refs": issues,
            "files_changed": len(rows),
            "lines_changed": changed,
            "src_files": [r["path"] for r in srcs],
            "test_files": [r["path"] for r in tests],
            "all_files": [r["path"] for r in rows],
        })
    return cands, dropped


def main():
    os.makedirs(OUT, exist_ok=True)
    allc = {}
    dropped_by_repo = {}
    total_dropped = {}
    for name, (role, ref) in SOURCES.items():
        c, dropped = mine(name, role, ref)
        allc[name] = c
        dropped_by_repo[name] = dropped
        for k, v in dropped.items():
            total_dropped[k] = total_dropped.get(k, 0) + v
        print("%-15s role=%-7s candidates=%d" % (name, role, len(c)))
        c.sort(key=lambda d: (-len(d["test_files"]), d["lines_changed"]))
        for d in c[:6]:
            print("   %s %s  %-58s L=%-4d tests=%d %s" % (
                d["fix_time_primary"][:10], d["fix_commit"][:10], d["subject"][:58],
                d["lines_changed"], len(d["test_files"]), ",".join(d["issue_refs"][:3])))

    alt_out = {}
    for name, spec in ALTERNATIVES.items():
        role, ref = spec["role"], spec["pinned_commit"]
        c, dropped = mine(name, role, ref)
        alt_out[name] = {
            "spec": spec,
            "candidate_families_in_window": len(c),
            "candidates": c,
            "dropped": dropped,
        }
        print("%-15s role=%-7s (ALTERNATIVE) candidates=%d" % (name, role, len(c)))

    with open(os.path.join(OUT, "family-candidates.json"), "w", encoding="utf-8") as f:
        json.dump({
            "policy_id": POLICY_ID,
            "criteria": {
                "revision_mined": "the PINNED commit of each repository, never a moving branch head",
                "time_axis_authority": (
                    "the ORIGINAL upstream fix's MERGE event (merged_at_utc of the pull "
                    "request the fix commit belongs to), evidenced in "
                    "out/merge-evidence.json. The commit's own author and committer "
                    "dates are audit corroboration ONLY and never qualify a family."),
                "count_semantics": (
                    "the candidate lists below are COMMIT-DATE SCREENINGS: both dates "
                    "inside the window. A screened candidate is headroom, NOT a "
                    "window-qualified family; qualification requires its merge event "
                    "to be retrieved and verified (status=verified and "
                    "window_qualified_by_merge_event=true)."),
                "time_substitutes_rejected": [
                    "release/tag date", "snapshot or pin date",
                    "backport date", "cherry-pick date",
                ],
                "train_window": {"end_exclusive": TRAIN_END_EXCLUSIVE,
                                 "means": "family fix time <= 2025-12-31"},
                "dev_sealed_window": {"start": DEVSEALED_START,
                                      "end_exclusive": DEVSEALED_END_EXCLUSIVE,
                                      "means": "family fix time in 2026-01-01 .. 2026-10-04",
                                      "dev_vs_sealed_ordering_claimed": False},
                "withdrawn_windows": WITHDRAWN_WINDOWS,
                "requires_issue_ref": True,
                "requires_source_and_test_change": True,
                "requires_fix_commit_ancestor_of_pinned_revision": True,
                "rejects_backport_or_cherry_pick_commits": True,
                "max_lines_changed": 200,
                "max_files_changed": 10,
                "single_parent_only": True,
            },
            "candidates_by_repo": allc,
            "counts": {k: len(v) for k, v in allc.items()},
            "dropped_by_repo": dropped_by_repo,
            "dropped_totals": total_dropped,
            "alternative_candidates": alt_out,
        }, f, indent=2, ensure_ascii=False)
    print("total=%d -> family-candidates.json" % sum(len(v) for v in allc.values()))
    print("alternative=%d" % sum(v["candidate_families_in_window"] for v in alt_out.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
