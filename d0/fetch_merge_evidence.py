"""Fetch the ORIGINAL upstream MERGE event for every family we intend to count.

Why this exists (Mika's targeted return, 2026-10-06)
---------------------------------------------------
A defect family's split window must be decided by the *original family fix's
upstream merge event* -- the ``merged_at`` timestamp of the pull request whose
``merge_commit_sha`` is the fix commit.  The commit's own author/committer dates
are kept **only as audit corroboration**: a rebase, a squash or a re-land can
rewrite them, so they must never qualify a family on their own.

Anything without a retrievable merge event stays ``unverified`` and is counted
in **no** window: its commit-message issue reference is not merge evidence
(#600 in python-dotenv is an ISSUE, so it proves nothing about a merge).

Evidence discipline
-------------------
* Every response body is written verbatim to ``d0/pr-evidence/raw/`` and its
  sha256 is recorded in the emitted record, so a reviewer can re-hash the exact
  bytes offline instead of trusting a summary.
* Raw bodies are cached: re-running this script does not spend API quota for a
  response that is already on disk (delete the file to force a refetch).
* Unauthenticated GitHub REST API only; no token is read, written or logged.

Output: d0/out/merge-evidence.json  (+ d0/pr-evidence/raw/*.json)

Usage:  python d0/fetch_merge_evidence.py [--refresh]
"""
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RAW = os.path.join(HERE, "pr-evidence", "raw")
SRC = os.path.join(HERE, "src")

API = "https://api.github.com"
UA = "gamma-kaggle-v3-d0-merge-evidence/1.0 (+research; unauthenticated)"
ACCEPT = "application/vnd.github+json"

TRAIN_END_EXCLUSIVE = "2026-01-01T00:00:00+00:00"
DEVSEALED_START = "2026-01-01T00:00:00+00:00"
DEVSEALED_END_EXCLUSIVE = "2026-10-05T00:00:00+00:00"

# ---- the commits we intend to COUNT, so each needs a merge event -------------
# real P0 training families
TRAIN_TARGETS = [
    # the click slot was substituted 2026-10-06 under Mika's ruling: the family now
    # counted is ee56925bc4f5 (PR #1934, merged 2021-07-03). The replaced commit
    # 9da1791476fe is kept in REPLACED_RECORDS so its failed evidence stays on the
    # record instead of disappearing.
    ("pallets/click", "click",
     "ee56925bc4f5451a125317e183f498e8bd1aecb3"),
    ("more-itertools/more-itertools", "more-itertools",
     "62411c1618493f94b16901746c34e72ad415061e"),
    ("pytest-dev/pluggy", "pluggy",
     "9cf2eaa50dd1ad3ebf042978629e78c695197095"),
    ("mahmoud/boltons", "boltons",
     "ae21ed2a78064ca1090db069e3f755aa1853b885"),
]

# Families whose evidence gathering FAILED. They are not counted anywhere, but the
# negative result is part of the delivery: a replaced or rejected slot must leave a
# retrievable record of why it was replaced or rejected.
REPLACED_RECORDS = [
    {"repo": "click", "upstream_slug": "pallets/click",
     "fix_commit": "9da1791476fe79ce77aa7a2a2db370c91a455251",
     "replaced_by": "ee56925bc4f5451a125317e183f498e8bd1aecb3",
     "note": ("2025-03-31 fix pushed straight to the default branch: GitHub reports no "
              "associated pull request and the two cross-referenced pull requests #258/#259 "
              "were both closed unmerged. It therefore has no original merge event, is "
              "counted in no quota, and was substituted by Mika's ruling.")},
]

# the alternative dev candidate's in-window candidates
ALT_SLUG = "theskumar/python-dotenv"
ALT_NAME = "python-dotenv"
ALT_TARGETS = [
    "da0c82054f1ec1e03d57356d49b0b9ad09eb4209",
    "bca6644d9aedbe287b792b756b3ae3d650cd0d3a",
    "f5485a61eefa5e686d6d5bdc7aa9ad6b104b1e92",
    "f7b18d9c72d1abcc2ad4023424b84f5bee30d266",
    "e0310e5bb3f2b701b11bbc007ca0c82c1bd56f38",
    "f215c0274dc4c310e5ca5feb4c6bddade25ece21",
]


# the pinned revision each family is mined at (mirrors collect_licenses.py)
PINNED = {
    "click": "8b19813f2bfca99f1018a587a8cf54fc959f2e5d",
    "more-itertools": "64be96ceb2a6e836f76f069f4a96d2394d59fd0c",
    "pluggy": "fd08ab5f811a9b2fa9124ae8cbbd393221151e2c",
    "boltons": "4332b35a278d694f30c99881faa61cde695c7a96",
    "python-dotenv": "a565c2cc41599c48eabc6b7b7f5b826d43c5a6d7",
}


def patch_equivalence(name, base_sha, fix_sha, landing_sha, shape):
    """Is the content carried by the landing event the same change as this family's fix?

    Ancestry alone cannot distinguish "this commit was merged" from "a different commit
    for the same pull request was merged" -- and boltons is exactly that case, where the
    commit that landed (078a215b) is not the commit GitHub reports (1efa5112), which the
    pinned clone does not contain at all. So the review compares diff text:

      * landing event is a two-parent merge commit -> compare the diff the MERGE
        introduces (merge vs its FIRST parent) with the diff the FIX introduces (fix vs
        its own parent). Diffing both against the pull request's ``base.sha`` would be
        wrong: the base branch usually advanced while the pull request was open, so that
        comparison picks up unrelated commits. pluggy PR#545 is exactly that case.
      * landing event is the fix commit itself (squash/rebase) -> the fix's own diff is
        the landed content by construction; reported as such rather than re-diffed.
    """
    repo = os.path.join(SRC, name)
    if not landing_sha or not base_sha:
        return {"computed": False, "reason": "no adjudicated landing commit or no base sha"}
    rc, fix_meta = git(repo, "log", "-1", "--format=%P", fix_sha)
    fix_parents = fix_meta.split() if rc == 0 else []
    if not fix_parents:
        return {"computed": False, "reason": "could not resolve the fix commit's parent"}
    fix_base = fix_parents[0]
    rc, fix_diff = git(repo, "diff", "--no-renames", fix_base, fix_sha)
    if rc != 0:
        return {"computed": False, "reason": "could not diff the fix commit"}
    fix_sha256 = sha256_bytes(fix_diff.encode("utf-8", "replace"))
    if shape == "fix_commit_is_the_landing_commit":
        return {
            "computed": True,
            "method": ("the fix commit is itself the landing event (squash or rebase "
                       "landing on the pinned branch), so the landed content is the fix "
                       "commit's own diff against its parent by construction"),
            "landing_event_is_the_fix_commit": True,
            "fix_diff_base": fix_base,
            "fix_diff_sha256": fix_sha256,
            "landing_diff_sha256": fix_sha256,
            "diff_text_equal": True,
            "fix_diff_bytes": len(fix_diff),
            "landing_diff_bytes": len(fix_diff),
            "not_claimed": ("this says nothing about whether some OTHER pull request "
                            "re-landed the same fix elsewhere in history"),
        }
    rc, land_meta = git(repo, "log", "-1", "--format=%P", landing_sha)
    land_parents = land_meta.split() if rc == 0 else []
    if not land_parents:
        return {"computed": False, "reason": "could not resolve the landing commit's parents"}
    # the diff the MERGE introduces = merge vs its first parent
    rc, land_diff = git(repo, "diff", "--no-renames", land_parents[0], landing_sha)
    if rc != 0:
        return {"computed": False, "reason": "could not diff the landing commit"}
    fix_by_file = diff_change_index(fix_diff)
    land_by_file = diff_change_index(land_diff)
    # A file the fix changed must be changed by the landing event with a change set that
    # contains the fix's. Equality is the clean case; containment is accepted when a
    # later commit on the pull request branch refined the same file, which is exactly the
    # boltons PR#31 shape -- and the refining commits are listed so the reviewer sees it.
    merged_in = land_parents[1] if len(land_parents) > 1 else None
    later_refinements = []
    if merged_in and merged_in != fix_sha:
        rc, log_lines = git(repo, "rev-list", "--reverse", "--format=%H %s",
                            "%s..%s" % (fix_sha, merged_in))
        if rc == 0:
            for ln in log_lines.splitlines():
                if not ln or ln.startswith("commit "):
                    continue
                parts = ln.split(" ", 1)
                if parts and re.fullmatch(r"[0-9a-f]{40}", parts[0]):
                    later_refinements.append({"sha": parts[0],
                                              "subject": parts[1] if len(parts) > 1 else None})
    file_results = []
    for path, (fix_add, fix_del) in fix_by_file.items():
        if path not in land_by_file:
            state = "missing_from_the_landing_event"
            extra = None
        else:
            land_add, land_del = land_by_file[path]
            identical = (fix_add == land_add and fix_del == land_del)
            contained = (_multiset_contains(land_add, fix_add)
                         and _multiset_contains(land_del, fix_del))
            state = ("identical" if identical else
                     ("contained_superset" if contained else "refined_by_a_later_commit"))
            extra = None if identical else {
                "landing_added_lines": len(land_add), "fix_added_lines": len(fix_add),
                "landing_removed_lines": len(land_del), "fix_removed_lines": len(fix_del),
                "lines_the_landing_event_adds_beyond_the_fix":
                    _multiset_extra(land_add, fix_add)[:10],
            }
        file_results.append({"path": path, "state": state, "detail": extra})
    not_covered = [f["path"] for f in file_results
                   if f["state"] == "missing_from_the_landing_event"]
    refined = [f["path"] for f in file_results if f["state"] == "refined_by_a_later_commit"]
    return {
        "computed": True,
        "method": ("the diff the merge introduces (merge vs its FIRST parent) compared, "
                   "file by file, with the diff the fix introduces (fix vs its own "
                   "parent). Both are diffed against their own parent because the base "
                   "branch usually advanced while the pull request was open"),
        "landing_event_is_the_fix_commit": False,
        "fix_diff_base": fix_base,
        "landing_first_parent": land_parents[0],
        "landing_merged_in_commit": merged_in,
        "fix_diff_sha256": fix_sha256,
        "landing_diff_sha256": sha256_bytes(land_diff.encode("utf-8", "replace")),
        "diff_text_equal": fix_diff == land_diff,
        "closed_under_the_landing_event": not not_covered,
        "every_fix_file_present_in_the_landing_event": not not_covered,
        "files_identical_or_contained": [f["path"] for f in file_results
                                         if f["state"] in ("identical", "contained_superset")],
        "files_refined_by_a_later_commit_on_the_pull_request": refined,
        "fix_files_not_covered": not_covered,
        "file_level": file_results,
        "commits_on_the_pull_request_after_the_fix": later_refinements,
        "fix_commit_is_ancestor_of_the_landing_merged_in_commit":
            _is_ancestor(repo, fix_sha, merged_in) if merged_in else None,
        "landing_touches_extra_files": sorted(set(land_by_file) - set(fix_by_file)),
        "fix_diff_bytes": len(fix_diff),
        "landing_diff_bytes": len(land_diff),
        "not_claimed": ("a change-set comparison over the files involved. It is NOT a proof "
                        "that no other commit re-landed the same fix elsewhere in history"),
    }


def _multiset_contains(superset, subset):
    """Is every element of ``subset`` present in ``superset`` with at least that count?"""
    from collections import Counter
    return not (Counter(subset) - Counter(superset))


def _multiset_extra(superset, subset):
    """Elements of ``superset`` beyond the counts in ``subset``."""
    from collections import Counter
    return sorted((Counter(superset) - Counter(subset)).elements())


def diff_change_index(diff_text):
    """Map each file in a unified diff to the multiset of added/removed lines.

    Stable across context-line differences, which is the point: two branches that carry
    the same change but sit on different trees produce different hunk context yet the
    same added/removed lines.
    """
    index = {}
    path = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            index.setdefault(path, {"added": [], "removed": []})
        elif path and line.startswith("+") and not line.startswith("+++"):
            index[path]["added"].append(line[1:])
        elif path and line.startswith("-") and not line.startswith("---"):
            index[path]["removed"].append(line[1:])
    return {p: (sorted(v["added"]), sorted(v["removed"])) for p, v in index.items()}


def landing_event_evidence(geometry):
    """The per-family landing answer a reviewer can check without re-deriving it.

    The counts come from the ADJUDICATED landing commit, not from the API's
    ``merge_commit_sha``: those are different commits for most of these families, which
    is the whole point of the block.
    """
    rec = geometry.get("reconciliation") or {}
    landing = geometry.get("adjudicated_landing_commit")
    chosen = None
    for c in rec.get("candidates_examined") or []:
        if c["sha"] == landing:
            chosen = c
            break
    return {
        "adjudication": geometry.get("landing_event_adjudication"),
        "github_merge_commit_present_in_pinned_clone":
            geometry.get("merge_commit_present_in_pinned_clone"),
        "landing_commit_in_pinned_history": landing,
        "landing_commit_is_ancestor_of_pinned_revision":
            geometry.get("adjudicated_landing_is_ancestor_of_pinned_revision"),
        "landing_commit_is_a_two_parent_merge":
            bool(chosen) and chosen["parent_count"] == 2,
        "fix_commit_is_the_second_parent":
            bool(chosen) and chosen["second_parent_is_the_fix_commit"],
        "landing_commit_parent_count": (chosen or {}).get("parent_count"),
        "search_command": rec.get("search"),
        "patch_equivalence": geometry.get("patch_equivalence"),
        "reviewer_note": ("the landing event is re-derived from the pinned repository, not "
                          "taken from the API, because the API's merge_commit_sha disagrees "
                          "with the pinned history for several of these families"),
    }


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def parse_iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def in_window(iso, role):
    t = parse_iso(iso)
    if role == "train":
        return t < parse_iso(TRAIN_END_EXCLUSIVE)
    return parse_iso(DEVSEALED_START) <= t < parse_iso(DEVSEALED_END_EXCLUSIVE)


def fetch(url, cache_path, refresh=False):
    """Return (raw_bytes, from_cache). Raw bytes are cached verbatim."""
    if not refresh and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            return f.read(), True
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": ACCEPT, "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=45) as r:
        body = r.read()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(body)
    return body, False


def slug_file(slug, name):
    return slug.split("/")[-1] + "-" + name


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode("utf-8", "replace").strip()


def real_merge_candidate(name, pinned, fix_sha, pr_number):
    """Independently locate the PR's landing event in the PINNED repository.

    Motivating fact (2026-10-06 review): GitHub's ``merge_commit_sha`` is NOT a reliable
    identifier of the commit that lives in the pinned clone. Three distinct shapes occur
    in this very delivery, and the reviewer asked for the difference to be established
    rather than asserted::

        more-itertools PR#412  api a5a14f61  -> pinned merge a5a14f61, fix is 2nd parent
        pluggy         PR#545  api 9d19d4b8  -> pinned merge 9d19d4b8, fix is 2nd parent
        boltons        PR#31   api 1efa5112  -> ABSENT from the pinned clone; the pinned
                                               history's own merge commit 1d7d8c4e brings
                                               in a different commit (078a215b) for the
                                               same pull request
        python-dotenv  PR#606  api == fix    -> squash: no merge commit exists at all

    So the landing event is re-derived from the pinned bytes. Two shapes are accepted:
    a two-parent "Merge pull request #N" commit, or the fix commit itself landing as a
    single-parent (squash) commit. Anything else is reported as unreconciled.
    """
    repo = os.path.join(SRC, name)
    rc, _ = git(repo, "cat-file", "-t", fix_sha)
    if rc != 0:
        return {"found": False, "reason": "the fix commit is absent from the pinned clone"}
    rc, fix_meta = git(repo, "log", "-1", "--format=%P%n%cI%n%s", fix_sha)
    fix_lines = fix_meta.splitlines()
    fix_parents = fix_lines[0].split() if fix_lines else []
    rc, listing = git(repo, "rev-list", pinned, "--grep=Merge pull request #%d" % pr_number)
    if rc != 0:
        return {"found": False, "reason": "git rev-list --grep failed"}
    funnels = []
    for sha in [s for s in listing.split() if s]:
        rc, meta = git(repo, "log", "-1", "--format=%P%n%cI%n%s", sha)
        lines = meta.splitlines()
        parents = lines[0].split() if lines else []
        funnels.append({
            "sha": sha,
            "committer_date": lines[1] if len(lines) > 1 else None,
            "subject": lines[2] if len(lines) > 2 else None,
            "parents": parents,
            "parent_count": len(parents),
            "second_parent_is_the_fix_commit": len(parents) > 1 and parents[1] == fix_sha,
        })
    matches = [c for c in funnels if c["second_parent_is_the_fix_commit"]]
    return {
        "found": bool(matches),
        "search": "git rev-list %s --grep='Merge pull request #%d'" % (pinned[:12], pr_number),
        "candidates_examined": funnels,
        "matching_merge_commit": matches[0]["sha"] if matches else None,
        "matching_merge_commit_detail": matches[0] if matches else None,
        "reason": ("exactly one pinned-history merge commit has this pull request in its "
                   "subject and the fix commit as its second parent" if matches else
                   "no merge commit in the pinned history matches both the pull-request "
                   "subject and the fix commit parent"),
        "fix_commit_parent_count_in_pinned_history": len(fix_parents),
        "fix_commit_is_itself_a_two_parent_merge": len(fix_parents) > 1,
        "fix_commit_is_ancestor_of_pinned_revision": _is_ancestor(repo, fix_sha, pinned),
    }


def merge_commit_geometry(name, pinned, merge_sha, fix_sha, pr_number):
    """Where does GitHub's merge_commit_sha sit relative to the pinned snapshot?

    A rebase-merge or a squash-merge legitimately produces a merge commit that
    differs from the fix commit, so a mismatch is *disclosed*, never hidden and
    never by itself a qualification. If the merge commit is absent from the
    pinned clone that is recorded too, and the pinned history is then searched for
    the real landing event so the commit is not simply written off.
    """
    repo = os.path.join(SRC, name)
    rc, typed = git(repo, "cat-file", "-t", merge_sha)
    present = rc == 0 and typed.strip() == "commit"
    if present:
        rc2, _ = git(repo, "merge-base", "--is-ancestor", merge_sha, pinned)
    else:
        rc2 = 1
    reconciled = real_merge_candidate(name, pinned, fix_sha, pr_number)
    # Adjudicate WHICH commit in the pinned history is the landing event. The merge
    # commit naming the pull request, the fix commit, and GitHub's merge_commit_sha can
    # all three differ, and boltons is exactly that case: the pinned merge commit
    # 1d7d8c4e names PR#31 but brings in 078a215b, whose parent is the fix ae21ed2a.
    # A landing event is therefore accepted on evidence, not on identity:
    #   1. the pinned two-parent merge commit that names this pull request, preferring
    #      the one that lists the fix as a parent, and confirming content equivalence;
    #   2. otherwise the fix commit itself, when it lands on the pinned branch.
    landing = None
    landing_shape = "unreconciled"
    adjudication = ("unreconciled: no landing event could be established from the pinned "
                    "history")
    cands = reconciled.get("candidates_examined") or []
    fix_is_parent = [c for c in cands if c["second_parent_is_the_fix_commit"]]
    ordered = fix_is_parent + [c for c in cands if c not in fix_is_parent]
    if ordered:
        chosen = ordered[0]
        landing = chosen["sha"]
        landing_shape = "two_parent_merge_commit"
        if chosen["second_parent_is_the_fix_commit"]:
            detail = ""
        else:
            # the merge brought in some other commit; establish that the fix still
            # reached the branch through it, by ancestry or by identical content
            other = chosen["parents"][1] if chosen["parent_count"] > 1 else None
            rc, eq = git(repo, "diff", "--no-renames", chosen["parents"][0], fix_sha) \
                if chosen["parents"] else (1, "")
            rc2, eq2 = git(repo, "diff", "--no-renames", chosen["parents"][0], other) \
                if other and chosen["parents"] else (1, "")
            same_content = rc == 0 and rc2 == 0 and eq == eq2
            fix_below_other = bool(other) and _is_ancestor(repo, fix_sha, other)
            detail = (
                "the pinned merge commit names this pull request but its merged-in "
                "commit is %s, not the fix commit %s; the fix reaches the branch beneath "
                "it (fix_is_ancestor_of_merged_in_commit=%s) and the content introduced "
                "by the merge equals the fix's own content (content_equal=%s)"
                % ((other or "-")[:12], fix_sha[:12], fix_below_other, same_content))
        adjudication = (
            "api_merge_commit_is_the_pinned_history_merge_event"
            if landing == merge_sha
            else ("api_merge_commit_absent_or_different_from_pinned_history; the pinned "
                  "history's own two-parent merge commit that names this pull request is "
                  "recorded as the landing event" + (". " + detail if detail else "")))
    elif present and reconciled.get("fix_commit_is_ancestor_of_pinned_revision"):
        landing = fix_sha
        landing_shape = "fix_commit_is_the_landing_commit"
        adjudication = ("api_merge_commit_present_and_the_fix_commit_itself_lands_on_the_"
                        "pinned_branch: squash or rebase merge, so the fix commit IS the "
                        "landing event and no separate merge commit exists")
    elif reconciled.get("fix_commit_is_ancestor_of_pinned_revision"):
        landing = fix_sha
        landing_shape = "fix_commit_is_the_landing_commit"
        adjudication = ("api_merge_commit_absent_from_the_pinned_clone and no two-parent "
                        "merge commit names this pull request, but the fix commit itself "
                        "lands on the pinned branch: squash/rebase landing")
    return {
        "merge_commit_present_in_pinned_clone": present,
        "merge_commit_is_ancestor_of_pinned_revision": (present and rc2 == 0),
        "fix_commit_is_the_pull_request_head": None,  # filled in by the caller
        "reconciliation": reconciled,
        "landing_event_adjudication": adjudication,
        "adjudicated_landing_commit": landing,
        "landing_event_shape": landing_shape,
        "adjudicated_landing_is_ancestor_of_pinned_revision":
            _is_ancestor(repo, landing, pinned) if landing else False,
    }


def _is_ancestor(repo, sha, ref):
    rc, _ = git(repo, "merge-base", "--is-ancestor", sha, ref)
    return rc == 0


def probe_commit(slug, name, sha, role, refresh):
    """Ask GitHub which pull requests this commit belongs to."""
    url = "%s/repos/%s/commits/%s/pulls" % (API, slug, sha)
    cache = os.path.join(RAW, "%s-commit-%s-pulls.json" % (name, sha[:12]))
    body, cached = fetch(url, cache, refresh)
    try:
        prs = json.loads(body.decode("utf-8"))
    except ValueError:
        prs = []
    return {
        "url": url,
        "raw_path": os.path.relpath(cache, HERE).replace(os.sep, "/"),
        "raw_response_sha256": sha256_bytes(body),
        "from_cache": cached,
        "pull_requests": [
            {"number": p.get("number"),
             "merge_commit_sha": p.get("merge_commit_sha"),
             "merged_at": p.get("merged_at"),
             "state": p.get("state")}
            for p in prs if isinstance(p, dict)],
    }


def pull_detail(slug, name, number, refresh):
    url = "%s/repos/%s/pulls/%d" % (API, slug, number)
    cache = os.path.join(RAW, "%s-pr%d.json" % (name, number))
    body, cached = fetch(url, cache, refresh)
    try:
        d = json.loads(body.decode("utf-8"))
    except ValueError:
        d = {}
    return {
        "url": url,
        "raw_path": os.path.relpath(cache, HERE).replace(os.sep, "/"),
        "raw_response_sha256": sha256_bytes(body),
        "from_cache": cached,
        "number": d.get("number", number),
        "html_url": d.get("html_url"),
        "title": d.get("title"),
        "state": d.get("state"),
        "merged": d.get("merged"),
        "merged_at": d.get("merged_at"),
        "merged_at_utc": (parse_iso(d["merged_at"]).astimezone(dt.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ") if d.get("merged_at") else None),
        "merge_commit_sha": d.get("merge_commit_sha"),
        "base_sha": (d.get("base") or {}).get("sha"),
        "head_sha": (d.get("head") or {}).get("sha"),
    }


def build_record(slug, name, sha, role, issue_refs, refresh):
    rec = {
        "repo": name,
        "upstream_slug": slug,
        "fix_commit": sha,
        "split_role": role,
        "window": "train" if role == "train" else "dev_sealed",
        "commit_message_issue_refs": issue_refs,
        "issue_reference_is_not_merge_evidence": True,
    }
    probe = probe_commit(slug, name, sha, role, refresh)
    rec["commit_to_pr_lookup"] = probe
    if not probe["pull_requests"]:
        rec.update({
            "status": "unverified",
            "pull_request": None,
            "merged_at_utc": None,
            "merge_commit_sha": None,
            "merge_commit_sha_matches_fix_commit": False,
            "window_qualified_by_merge_event": False,
            "reason": ("GitHub reports no pull request associated with this commit. "
                       "Absence of merge evidence is NOT evidence that no original "
                       "fix exists; the record simply cannot be counted."),
        })
        return rec

    # prefer the PR whose merge_commit_sha is exactly this commit
    prs = sorted(probe["pull_requests"],
                 key=lambda p: (p.get("merge_commit_sha") != sha, p.get("merged_at") or ""))
    detail = pull_detail(slug, name, prs[0]["number"], refresh)
    rec["pull_request"] = detail
    rec["pull_request_url"] = detail["html_url"]
    rec["merged_at_utc"] = detail["merged_at_utc"]
    rec["merge_commit_sha"] = detail["merge_commit_sha"]
    rec["merge_commit_sha_matches_fix_commit"] = detail["merge_commit_sha"] == sha
    rec["pull_request_head_sha_matches_fix_commit"] = detail.get("head_sha") == sha
    geometry = merge_commit_geometry(name, PINNED[name], detail["merge_commit_sha"],
                                     sha, detail.get("number") or prs[0]["number"])
    geometry["fix_commit_is_the_pull_request_head"] = \
        detail.get("head_sha") == sha
    geometry["pr_base_sha"] = detail.get("base_sha")
    geometry["pr_head_sha"] = detail.get("head_sha")
    geometry["patch_equivalence"] = patch_equivalence(
        name, detail.get("base_sha"), sha, geometry.get("adjudicated_landing_commit"),
        geometry.get("landing_event_shape"))
    if rec["merge_commit_sha_matches_fix_commit"]:
        geometry["mismatch_pattern"] = "exact_merge"
    elif geometry["fix_commit_is_the_pull_request_head"]:
        geometry["mismatch_pattern"] = (
            "rebase_or_squash_merge: the fix commit is the pull request head and GitHub's "
            "merge_commit_sha is the commit the base branch moved to")
    else:
        geometry["mismatch_pattern"] = (
            "merge_commit_sha is neither the fix commit nor the pull request head; the "
            "pinned history's own merge event is recorded in "
            "geometry.reconciliation.matching_merge_commit and adjudicated there")
    rec["merge_commit_geometry"] = geometry
    rec["landing_event_evidence"] = landing_event_evidence(geometry)
    if detail["merged_at_utc"] is None:
        rec["status"] = "unverified"
        rec["window_qualified_by_merge_event"] = False
        rec["reason"] = "the pull request was not merged, so it supplies no merge event"
        return rec
    rec["status"] = "verified"
    rec["window_qualified_by_merge_event"] = in_window(detail["merged_at_utc"], role)
    rec["reason"] = (
        "merge event retrieved from the pull-request metadata; qualification uses "
        "merged_at_utc, not the commit's author/committer dates")
    return rec


def main():
    refresh = "--refresh" in sys.argv
    os.makedirs(RAW, exist_ok=True)
    records = []
    for slug, name, sha in TRAIN_TARGETS:
        records.append(build_record(slug, name, sha, "train", [], refresh))
    for sha in ALT_TARGETS:
        records.append(build_record(ALT_SLUG, ALT_NAME, sha, "dev", [], refresh))

    live_lookups = sum(1 for r in records
                       if r["commit_to_pr_lookup"]["from_cache"] is False)
    # keep the original retrieval time when nothing was actually re-fetched, so a
    # cache-only re-run does not churn the artifact's timestamp
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prev_path = os.path.join(OUT, "merge-evidence.json")
    if live_lookups == 0 and os.path.isfile(prev_path):
        try:
            with open(prev_path, encoding="utf-8") as f:
                stamp = json.load(f).get("retrieved_at_utc", stamp)
        except (OSError, ValueError):
            pass

    verified = [r for r in records if r["status"] == "verified"
                and r["window_qualified_by_merge_event"]]
    doc = {
        "generated_by": "d0/fetch_merge_evidence.py",
        "retrieved_at_utc": stamp,
        "retrieval_mode": ("live" if live_lookups else
                           "cache-only re-run; no network request was made"),
        "metadata_source": API + " (GitHub REST API v3)",
        "authentication": ("unauthenticated; no token is read, written or logged. "
                           "Cache-first: an already-downloaded response is re-used, "
                           "so re-runs do not spend API quota."),
        "policy": {
            "window_qualification_basis": (
                "the ORIGINAL family fix's upstream MERGE event: the merged_at_utc of "
                "the pull request whose merge_commit_sha equals the fix commit"),
            "author_and_committer_dates_role": (
                "audit corroboration ONLY. A rebase, squash or re-land can rewrite "
                "them, so they never qualify a family into a window."),
            "backport_handling": (
                "a backport commit inherits the ORIGINAL fix's merge event; its own "
                "commit time is never used to classify it"),
            "missing_merge_evidence": (
                "status=unverified, window_qualified_by_merge_event=false, counted in "
                "no window. An issue reference in the commit message is not merge "
                "evidence and can never promote such a record."),
            "negative_test_required": (
                "a record without merge evidence must fail the gate's qualification "
                "predicate -- see validate_d0.py"),
        },
        "windows": {
            "train": {"end_exclusive": TRAIN_END_EXCLUSIVE},
            "dev_sealed": {"start": DEVSEALED_START,
                           "end_exclusive": DEVSEALED_END_EXCLUSIVE},
        },
        "records": records,
        "replaced_records": [
            dict(r, status="replaced", window_qualified_by_merge_event=False,
                 counted_in_no_quota=True,
                 issue_reference_is_not_merge_evidence=True)
            for r in REPLACED_RECORDS],
        "summary": {
            "records": len(records),
            "verified": sum(1 for r in records if r["status"] == "verified"),
            "unverified": sum(1 for r in records if r["status"] != "verified"),
            "verified_and_in_window": len(verified),
            "unverified_commits": [r["fix_commit"] for r in records
                                   if r["status"] != "verified"],
            "merge_commit_sha_matches_fix_commit": sum(
                1 for r in records if r.get("merge_commit_sha_matches_fix_commit")),
            "merge_commit_sha_differs_from_fix_commit": sum(
                1 for r in records if r["status"] == "verified"
                and not r.get("merge_commit_sha_matches_fix_commit")),
            "merge_commit_absent_from_pinned_clone": [
                {"repo": r["repo"], "fix_commit": r["fix_commit"],
                 "pull_request": r.get("pull_request_url"),
                 "merge_commit_sha": r.get("merge_commit_sha")}
                for r in records
                if (r.get("merge_commit_geometry") or {}).get(
                    "merge_commit_present_in_pinned_clone") is False],
            "api_lookups": live_lookups,
            "landing_event_adjudications": {
                r["repo"]: (r.get("merge_commit_geometry") or {}).get(
                    "landing_event_adjudication") for r in records},
            "reconciled_landing_commits": [
                {"repo": r["repo"], "fix_commit": r["fix_commit"],
                 "github_merge_commit_sha": r.get("merge_commit_sha"),
                 "adjudicated_landing_commit": (r.get("merge_commit_geometry") or {}).get(
                     "adjudicated_landing_commit"),
                 "adjudication": (r.get("merge_commit_geometry") or {}).get(
                     "landing_event_adjudication")}
                for r in records if r["status"] == "verified"],
            "patch_equivalence_all_true": all(
                ((r.get("merge_commit_geometry") or {}).get("patch_equivalence") or {})
                .get("diff_text_equal") is True
                for r in records if r["status"] == "verified"),
        },
        "disclosures": [
            ("an unverified record is NOT evidence that no original fix exists; it "
             "means the merge event could not be retrieved and the record is counted "
             "in no window"),
            ("for a rebase- or squash-merge the merge_commit_sha legitimately differs "
             "from the fix commit; the geometry block records which pattern applies "
             "and whether the merge commit is an ancestor of the pinned revision"),
            ("a merge commit that is absent from the pinned clone is listed in "
             "summary.merge_commit_absent_from_pinned_clone rather than reconciled "
             "silently"),
            ("this artifact only settles the WINDOW question. It does not establish "
             "family independence, licence closure, environment or FAIL_TO_PASS "
             "behaviour, or permission isolation"),
        ],
    }
    with open(os.path.join(OUT, "merge-evidence.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    for r in records:
        print("%-15s %s %-9s pr=%-5s merged=%-21s match=%-5s in_window=%s" % (
            r["repo"], r["fix_commit"][:12], r["status"],
            str(r.get("pull_request_url", "-")).split("/")[-1],
            str(r.get("merged_at_utc")), r.get("merge_commit_sha_matches_fix_commit"),
            r.get("window_qualified_by_merge_event")))
    print("wrote d0/out/merge-evidence.json: %s" % json.dumps(doc["summary"],
                                                              ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
