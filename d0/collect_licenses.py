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
    lock = {"generated_by": "d0/collect_licenses.py", "repos": {}}
    lic_files_all = {}
    ledger_rows = []
    summaries = {}

    for name, (slug, tag, exp_sha, design_license, split) in REPOS.items():
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
        lock["repos"][name] = {
            "upstream_slug": slug,
            "mirror_url": "https://ghfast.top/https://github.com/%s.git" % slug,
            "pinned_tag": tag,
            "pinned_commit": head,
            "expected_commit": exp_sha,
            "commit_matches_plan": head == exp_sha,
            "tree_sha": tree,
            "commit_date": cdate,
            "committer": cname,
            "tracked_files": len(files),
            "text_files_scanned": scanned,
            "split_role": split,
            "design_license_expectation": design_license,
            "spdx_headers_found": spdx_set,
            "packaging_metadata": meta,
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
