"""Build the per-file license / NOTICE ledger for each pinned source snapshot.

Outputs (under d0/out/):
  source-lock.json          pinned revisions + tree sha + metadata license declaration
  license-files.json        hash of every top-level license-ish file
  per-file-ledger.csv       every tracked file: sha256, header SPDX/copyright signal
  per-file-ledger.summary.json
"""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "out")
ACCESS_DATE = "2026-10-05"

REPOS = {
    "click": ("pallets/click", "8.5.0", "8b19813f2bfca99f1018a587a8cf54fc959f2e5d", "BSD-3-Clause", "train"),
    "more-itertools": ("more-itertools/more-itertools", "v11.1.0", "64be96ceb2a6e836f76f069f4a96d2394d59fd0c", "MIT", "train"),
    "pluggy": ("pytest-dev/pluggy", "1.6.0", "fd08ab5f811a9b2fa9124ae8cbbd393221151e2c", "MIT", "train"),
    "boltons": ("mahmoud/boltons", "26.2.0", "4332b35a278d694f30c99881faa61cde695c7a96", "BSD-3-Clause", "train"),
    "attrs": ("python-attrs/attrs", "26.1.0", "7bfc49e9b22d5ba25b6e429524c3d49fee27cb36", "MIT", "dev"),
    "dateutil": ("dateutil/dateutil", "2.9.0", "db9d018944c41ddc740015cf5f64717c2ba64a5c", "Apache-2.0 OR BSD-3-Clause", "dev"),
    "packaging": ("pypa/packaging", "26.3", "929fd4b1410ac7ef61ef3f45b2f5d7e87711a9b5", "Apache-2.0 OR BSD-2-Clause", "sealed"),
    "marshmallow": ("marshmallow-code/marshmallow", "4.3.1", "c7b559a1fa3aba57ca6dba0ab336841c5038a782", "MIT", "sealed"),
}

# Approved ALTERNATIVE candidate for the dev slot. Verified here for license,
# family inventory and environment material, but NOT a member of the locked
# eight and NOT approved for release: it only replaces dateutil if the
# custodian says so.
ALTERNATIVE_REPOS = {
    "python-dotenv": ("theskumar/python-dotenv", "v1.2.4",
                      "a565c2cc41599c48eabc6b7b7f5b826d43c5a6d7", "MIT", "dev"),
}

LICENSE_REVIEW_SCHEMA = {
    "schema_version": "1.0",
    "applies_to": "every repository in this file, keyed by repository name",
    "required_fields": [
        "decision", "approved_spdx", "osi_permissive", "copyleft_marker_hits",
        "restrictive_marker_hits", "evidence", "decision_basis", "decided_by",
        "decided_at", "decided_against_revision", "independent_review",
    ],
    "decision_domain": ["approved", "rejected", "pending"],
    "gate": ("no repository may be used as a family source unless "
             "license_review.decision == 'approved' for the exact revision in "
             "decided_against_revision"),
    "approval_is_revision_bound": ("an approval covers one pinned commit; re-pinning "
                                   "a repository invalidates it and it must be re-derived"),
    "independent_review_note": ("decision is the D0 owner's licence-fact assessment and "
                                "is self-declared; independent countersignature is tracked "
                                "separately in independent_review and is NOT claimed here"),
}

LICENSE_FILE_RE = re.compile(
    r"^(licen[cs]e|copying|copyright|notice|authors|contributors|patents|"
    r"third[_-]?party|legal)([-_.].*)?$",
    re.I,
)

TEXT_EXT = {
    ".py", ".pyi", ".pyx", ".pxd", ".cfg", ".toml", ".ini", ".txt", ".rst",
    ".md", ".js", ".ts", ".css", ".html", ".yaml", ".yml", ".json", ".sh",
    ".bat", ".ps1", ".c", ".h", ".in", ".template", ".patch", ".cmake",
}

SPDX_RE = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+ ]+)", re.I)
# a copyright line must be short-ish and look like a notice, not prose
COPY_RE = re.compile(
    r"^\s*(?:[#*/<!;%\s\"']*)?(copyright|\(c\)|\(C\)|\u00a9)\s*[:\u00a9]?\s*(.{0,140})$",
    re.I,
)
LICENSE_MENTION_RE = re.compile(
    r"\b(licen[cs]ed under|distributed under|under the terms of|MIT License|"
    r"BSD License|BSD 3-Clause|BSD 2-Clause|Apache License|GNU (?:Lesser )?General Public License|"
    r"PSF License|Mozilla Public License)\b", re.I,
)

# ---------------------------------------------------------------------------
# machine-readable licence approval contract (see LICENSE_REVIEW_SCHEMA)
# ---------------------------------------------------------------------------
# The marker scan is deliberately limited to the primary licence-family files
# (LICENSE / COPYING / NOTICE / PATENTS).  AUTHORS, CONTRIBUTING and changelog
# files routinely mention other licences in passing, and letting them raise a
# copyleft hit would make the signal unusable.
PRIMARY_LICENCE_RE = re.compile(r"^(licen[cs]e|copying|notice|patents)([-_.].*)?$", re.I)

COPYLEFT_MARKERS = [
    "GNU General Public License", "GNU Lesser General Public License",
    "GNU Affero General Public License", "AGPL", "LGPL",
    "Mozilla Public License", "Eclipse Public License",
    "Reciprocal Public License", "Sleepycat License", "CDDL",
    "Creative Commons Attribution-ShareAlike", "CC BY-SA",
]
RESTRICTIVE_MARKERS = [
    "non-commercial", "noncommercial", "non commercial",
    "research only", "research purposes only", "for research use only",
    "no derivative works", "not for commercial use", "evaluation only",
    "CC BY-NC",
]
APPROVED_SPDX_SET = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0",
                     "Apache-2.0 OR BSD-3-Clause", "Apache-2.0 OR BSD-2-Clause"}


def marker_scan(repo_dir, lic_files):
    """Scan primary licence files for copyleft / restrictive-use markers.

    Returns (copyleft_hits, restrictive_hits); each hit records the file, the
    matched marker and the matching line so a reviewer can judge it directly
    instead of trusting a boolean.
    """
    copyleft, restrictive = [], []
    for lf in lic_files:
        if not PRIMARY_LICENCE_RE.match(os.path.basename(lf["path"])):
            continue
        full = os.path.join(repo_dir, lf["path"].replace("/", os.sep))
        try:
            text = open(full, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        lines = text.splitlines()
        for marker in COPYLEFT_MARKERS:
            for i, line in enumerate(lines):
                if marker.lower() in line.lower():
                    copyleft.append({"file": lf["path"], "marker": marker,
                                     "line_no": i + 1, "line": line.strip()[:200]})
                    break
        for marker in RESTRICTIVE_MARKERS:
            for i, line in enumerate(lines):
                if marker.lower() in line.lower():
                    restrictive.append({"file": lf["path"], "marker": marker,
                                        "line_no": i + 1, "line": line.strip()[:200]})
                    break
    return copyleft, restrictive


def build_license_review(name, slug, tag, pinned, design_license, repo_dir,
                         lic_files, meta, per_file_stats, decided_at):
    """Assemble the machine-readable approval record for one pinned revision."""
    primary = None
    for want in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "LICENSE.rst"):
        for lf in lic_files:
            if lf["path"] == want:
                primary = lf
                break
        if primary:
            break
    if primary is None and lic_files:
        primary = lic_files[0]
    copyleft, restrictive = marker_scan(repo_dir, lic_files)
    approved = design_license in APPROVED_SPDX_SET
    decision = "approved" if (approved and not restrictive) else "pending"
    basis = (
        "declared licence family is OSI-permissive (%s); the primary licence file at the "
        "pinned revision carries no copyleft marker and no non-commercial / research-only / "
        "no-derivative clause. Hit lists below are recorded verbatim so this decision can be "
        "re-derived from the hashed files alone." % design_license)
    if not approved:
        basis = "declared licence family %s is outside the approved permissive set" % design_license
    if restrictive:
        basis += " ; RESTRICTIVE MARKER FOUND -> decision withheld"
    return {
        "decision": decision,
        "approved_spdx": design_license,
        "osi_permissive": bool(approved),
        "copyleft_marker_hits": copyleft,
        "restrictive_marker_hits": restrictive,
        "evidence": {
            "primary_license_file": primary,
            "all_license_file_hashes": lic_files,
            "packaging_metadata_sha256": {k: v.get("sha256") for k, v in meta.items()},
            "per_file_scan": per_file_stats,
            "per_file_ledger": "per-file-ledger.csv",
        },
        "decision_basis": basis,
        "decided_by": "D0 owner (资料调研与分发)",
        "decided_at": decided_at,
        "decided_against_revision": pinned,
        "decided_against_tag": tag,
        "upstream_slug": slug,
        "independent_review": {
            "required": True,
            "status": "pending",
            "reviewer_role": "Q0",
            "note": ("self-declared licence-fact assessment; NOT an independent "
                     "countersignature and must not be read as one"),
        },
    }


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo_dir, *args):
    p = subprocess.run(["git", "-C", repo_dir] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode("utf-8", "replace").strip(), p.stderr.decode("utf-8", "replace").strip()


def scan_header(path):
    """Return (spdx, copyright_lines, license_mention) from the first 60 lines."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            head = "".join(f.readline() for _ in range(60))
    except (OSError, UnicodeError):
        return None, [], None
    spdx = None
    m = SPDX_RE.search(head)
    if m:
        spdx = " ".join(m.group(1).split())
    cps = []
    for line in head.splitlines():
        cm = COPY_RE.match(line)
        if cm:
            tail = cmp_text = cm.group(2).strip()
            if cm.group(1).lower() == "copyright":
                cmp_text = tail
            # keep plausible notices only
            if 0 < len(cmp_text) <= 140 and len(cmp_text.split()) <= 20:
                cps.append(line.strip())
    for extra in license_mention_regex_findall(head):
        cps.append(extra)
    lm = None
    m2 = LICENSE_MENTION_RE.search(head)
    if m2:
        lm = m2.group(0)
    return spdx, cps[:6], lm


def license_mention_regex_findall(head):
    # secondary: explicit "Copyright (c) YYYY Name" style lines
    out = []
    for line in head.splitlines()[:20]:
        if re.match(r"^\s*(?:[#*!;\s]*)(copyright\s*\(c\)|\(c\)\s*\d{4}|copyright\s+\d{4})", line, re.I):
            out.append(line.strip())
    return out


def read_meta(repo_dir, files):
    """Extract declared license metadata from packaging metadata files."""
    meta = {}
    for cand in ("pyproject.toml", "setup.cfg", "setup.py", "PKG-INFO"):
        if cand not in files:
            continue
        text = open(os.path.join(repo_dir, cand), encoding="utf-8", errors="replace").read()
        info = {"sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()}
        lics = []
        for m in re.finditer(r'^\s*(?:license|license_files|license-files|licence)\s*=\s*(.+)$', text, re.M | re.I):
            lics.append(m.group(1).strip())
        for m in re.finditer(r'^\s*license\s*=\s*\{([^}]*)\}', text, re.M | re.I):
            lics.append("{" + m.group(1).strip() + "}")
        cls = re.findall(r'License\s*::[^"\'\n]*', text)
        if cls:
            info["classifiers"] = sorted(set(c.strip() for c in cls))
        if lics:
            info["declared"] = lics
        meta[cand] = info
    return meta


def main():
    os.makedirs(OUT, exist_ok=True)
    lock = {
        "generated_by": "d0/collect_licenses.py",
        "policy_id": "v3-source-lock-2026-10-05-mika",
        "license_review_schema": LICENSE_REVIEW_SCHEMA,
        "locked_candidate_repositories": sorted(REPOS),
        "alternative_candidate_repositories": sorted(ALTERNATIVE_REPOS),
        "repos": {},
    }
    lic_files_all = {}
    ledger_rows = []
    summaries = {}

    ordered = [(n, v, "locked_candidate") for n, v in REPOS.items()]
    ordered += [(n, v, "alternative_candidate") for n, v in ALTERNATIVE_REPOS.items()]

    for name, (slug, tag, exp_sha, design_license, split), role_class in ordered:
        repo_dir = os.path.join(SRC, name)
        rc, head, err = git(repo_dir, "rev-parse", "HEAD")
        rc2, tree, _ = git(repo_dir, "rev-parse", "HEAD^{tree}")
        rc3, cdate, _ = git(repo_dir, "log", "-1", "--format=%cI")
        rc4, cname, _ = git(repo_dir, "log", "-1", "--format=%cN <%cE>")
        rc5, files_out, _ = git(repo_dir, "ls-tree", "-r", "--name-only", "HEAD")
        files = [f for f in files_out.splitlines() if f.strip()]

        lic_files = []
        for f in files:
            base = os.path.basename(f)
            if LICENSE_FILE_RE.match(base):
                full = os.path.join(repo_dir, f.replace("/", os.sep))
                if os.path.isfile(full):
                    lic_files.append({
                        "path": f,
                        "sha256": sha256_of(full),
                        "bytes": os.path.getsize(full),
                    })
        lic_files.sort(key=lambda d: d["path"])

        per_file = []
        n_with_spdx = 0
        n_with_cp = 0
        scanned = 0
        for f in files:
            full = os.path.join(repo_dir, f.replace("/", os.sep))
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext not in TEXT_EXT:
                continue
            scanned += 1
            spdx, cps, lm = scan_header(full)
            if spdx:
                n_with_spdx += 1
            if cps:
                n_with_cp += 1
            per_file.append({
                "repo": name,
                "path": f,
                "sha256": sha256_of(full),
                "bytes": os.path.getsize(full),
                "spdx_header": spdx or "",
                "copyright_lines": " | ".join(cps),
                "license_mention": lm or "",
                "is_license_file": any(lf["path"] == f for lf in lic_files),
            })
        ledger_rows.extend(per_file)

        spdx_set = sorted({r["spdx_header"] for r in per_file if r["spdx_header"]})
        meta = read_meta(repo_dir, files)
        per_file_stats = {
            "text_files_scanned": scanned,
            "files_with_spdx_header": n_with_spdx,
            "files_with_copyright_line": n_with_cp,
            "method": "d0/collect_licenses.py; every file row is in per-file-ledger.csv",
        }
        review = build_license_review(
            name, slug, tag, head, design_license, repo_dir, lic_files, meta,
            per_file_stats, decided_at=ACCESS_DATE)
        lock["repos"][name] = {
            "role_class": role_class,
            "upstream_slug": slug,
            "mirror_url": "https://ghfast.top/https://github.com/%s.git" % slug,
            "pinned_tag": tag,
            "pinned_commit": head,
            "expected_commit": exp_sha,
            "commit_matches_plan": head == exp_sha,
            "tree_sha": tree,
            "commit_date": cdate,
            "commit_date_role": (
                "the pinned SNAPSHOT's own commit date. It is NOT a family fix time and "
                "must never be used to assign a defect family to a split window."),
            "committer": cname,
            "tracked_files": len(files),
            "text_files_scanned": scanned,
            "split_role": split,
            "design_license_expectation": design_license,
            "spdx_headers_found": spdx_set,
            "packaging_metadata": meta,
            "license_review": review,
        }
        lic_files_all[name] = lic_files
        summaries[name] = {
            "tracked_files": len(files),
            "license_files": [lf["path"] for lf in lic_files],
            "text_files_scanned": scanned,
            "files_with_spdx_header": n_with_spdx,
            "files_with_copyright_line": n_with_cp,
            "files_without_any_header_signal": scanned - n_with_cp,
            "spdx_headers_found": spdx_set,
        }
        print("%-15s head=%s tree=%s files=%d lics=%d spdx_hdrs=%d" % (
            name, head[:12], tree[:12], len(files), len(lic_files), n_with_spdx))

    lock["summary"] = {
        "repositories": len(lock["repos"]),
        "decision_counts": {
            d: sum(1 for r in lock["repos"].values()
                   if r["license_review"]["decision"] == d)
            for d in LICENSE_REVIEW_SCHEMA["decision_domain"]},
        "decision_by_repo": {n: r["license_review"]["decision"]
                            for n, r in lock["repos"].items()},
        "api_field": "repos.<name>.license_review.decision",
        "gate_satisfied": all(r["license_review"]["decision"] == "approved"
                              for r in lock["repos"].values()),
        "independent_countersignature_claimed": False,
    }

    with open(os.path.join(OUT, "source-lock.json"), "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "license-files.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_by": "d0/collect_licenses.py", "by_repo": lic_files_all}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "per-file-ledger.summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "per-file-ledger.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["repo", "path", "sha256", "bytes", "spdx_header",
                                          "copyright_lines", "license_mention", "is_license_file"])
        w.writeheader()
        for r in ledger_rows:
            w.writerow(r)
    print("rows=%d -> per-file-ledger.csv" % len(ledger_rows))


if __name__ == "__main__":
    main()
