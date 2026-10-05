"""Mine SWE-bench-style candidate families from the training repos.

A candidate family needs, at a minimum:
  * a fix commit that changes BOTH package source and test code
  * a commit message that references an upstream issue / PR number
  * a bounded diff (so the oracle patch stays reviewable)
  * a fix date inside the declared V3 train window

Output: d0/out/family-candidates.json  (+ printed shortlist)
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "out")

TRAIN = {
    "click": "origin/main",
    "more-itertools": "origin/master",
    "pluggy": "origin/main",
    "boltons": "origin/master",
}

TRAIN_WINDOW_END = "2025-01-01T00:00:00+00:00"   # exclusive: train families must be strictly older
ISSUE_RE = re.compile(r"(?:#|GH-|gh-)(\d{2,6})")
FIXWORD_RE = re.compile(r"\b(fix(?:es|ed)?|clos(?:e|es|ed)|resolv(?:e|es|ed)|bug)\b", re.I)
TEST_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$", re.I)
SRC_RE = re.compile(r"\.pyi?$")


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


def mine(name, ref):
    repo = os.path.join(SRC, name)
    rc, out, err = git(repo, "log", "--no-merges", "--format=%H%x1f%P%x1f%cI%x1f%s%x1f%b%x1e",
                       ref)
    if rc != 0:
        print("log failed for %s: %s" % (name, err[:200]))
        return []
    cands = []
    for rec in out.split("\x1e"):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        fields = rec.split("\x1f")
        if len(fields) < 5:
            continue
        sha, parents, cdate, subject, body = fields[0], fields[1], fields[2], fields[3], fields[4]
        if cdate >= TRAIN_WINDOW_END:
            continue
        parents = parents.split()
        if len(parents) != 1:
            continue
        text = subject + "\n" + body
        issues = sorted(set(ISSUE_RE.findall(text)), key=int)
        if not issues:
            continue
        if not FIXWORD_RE.search(text):
            continue
        rows = numstat(repo, sha)
        if not rows or any(is_binaryish(r) for r in rows):
            continue
        tests = [r for r in rows if TEST_RE.search(r["path"])]
        srcs = [r for r in rows if (not TEST_RE.search(r["path"])) and SRC_RE.search(r["path"])]
        if not tests or not srcs:
            continue
        changed = sum((r["add"] or 0) + (r["del"] or 0) for r in rows)
        if changed > 200 or len(rows) > 10:
            continue
        cands.append({
            "repo": name,
            "fix_commit": sha,
            "base_commit": parents[0],
            "fix_date": cdate,
            "subject": subject,
            "issue_refs": issues,
            "files_changed": len(rows),
            "lines_changed": changed,
            "src_files": [r["path"] for r in srcs],
            "test_files": [r["path"] for r in tests],
            "all_files": [r["path"] for r in rows],
        })
    return cands


def main():
    os.makedirs(OUT, exist_ok=True)
    allc = {}
    for name, ref in TRAIN.items():
        c = mine(name, ref)
        allc[name] = c
        print("%-15s candidates=%d" % (name, len(c)))
        c.sort(key=lambda d: (-len(d["test_files"]), d["lines_changed"]))
        for d in c[:6]:
            print("   %s %s  %-58s L=%-4d tests=%d %s" % (
                d["fix_date"][:10], d["fix_commit"][:10], d["subject"][:58],
                d["lines_changed"], len(d["test_files"]), ",".join(d["issue_refs"][:3])))
    with open(os.path.join(OUT, "family-candidates.json"), "w", encoding="utf-8") as f:
        json.dump({
            "criteria": {
                "train_window_end_exclusive": TRAIN_WINDOW_END,
                "requires_issue_ref": True,
                "requires_source_and_test_change": True,
                "max_lines_changed": 200,
                "max_files_changed": 10,
                "single_parent_only": True,
            },
            "candidates_by_repo": allc,
            "counts": {k: len(v) for k, v in allc.items()},
        }, f, indent=2, ensure_ascii=False)
    print("total=%d -> family-candidates.json" % sum(len(v) for v in allc.values()))


if __name__ == "__main__":
    main()
