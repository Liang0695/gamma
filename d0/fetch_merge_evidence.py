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
    ("pallets/click", "click",
     "9da1791476fe79ce77aa7a2a2db370c91a455251"),
    ("more-itertools/more-itertools", "more-itertools",
     "62411c1618493f94b16901746c34e72ad415061e"),
    ("pytest-dev/pluggy", "pluggy",
     "9cf2eaa50dd1ad3ebf042978629e78c695197095"),
    ("mahmoud/boltons", "boltons",
     "ae21ed2a78064ca1090db069e3f755aa1853b885"),
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


def merge_commit_geometry(name, pinned, merge_sha, fix_sha):
    """Where does GitHub's merge_commit_sha sit relative to the pinned snapshot?

    A rebase-merge or a squash-merge legitimately produces a merge commit that
    differs from the fix commit, so a mismatch is *disclosed*, never hidden and
    never by itself a qualification. If the merge commit is absent from the
    pinned clone that is recorded too: it means the upstream history no longer
    contains what GitHub reports, which a reviewer must be able to see.
    """
    repo = os.path.join(SRC, name)
    rc, _ = git(repo, "cat-file", "-t", merge_sha)
    present = rc == 0
    rc2, _ = git(repo, "merge-base", "--is-ancestor", merge_sha, pinned)
    return {
        "merge_commit_present_in_pinned_clone": present,
        "merge_commit_is_ancestor_of_pinned_revision": (present and rc2 == 0),
        "fix_commit_is_the_pull_request_head": None,  # filled in by the caller
    }


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
    geometry = merge_commit_geometry(name, PINNED[name], detail["merge_commit_sha"], sha)
    geometry["fix_commit_is_the_pull_request_head"] = \
        detail.get("head_sha") == sha
    geometry["mismatch_pattern"] = (
        "exact_merge" if rec["merge_commit_sha_matches_fix_commit"]
        else ("rebase_or_squash_merge: the fix commit is the pull request head and "
              "GitHub's merge_commit_sha is the commit the base branch moved to"
              if geometry["fix_commit_is_the_pull_request_head"]
              else "merge_commit_sha is neither the fix commit nor the pull request "
                   "head; disclosed for review, not reconciled here"))
    rec["merge_commit_geometry"] = geometry
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
