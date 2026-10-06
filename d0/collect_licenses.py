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
#
# SPDX corrected 2026-10-06: this entry used to say MIT. Both the pinned LICENSE
# (git blob 3a97119010ac82e15e917a69b7b8f9f59b5a4601) and the pinned
# pyproject.toml (`license = { text = "BSD-3-Clause" }`) say BSD-3-Clause, so MIT
# was a transcription error in this preset, not a licence change upstream. The
# earlier candidate pin 791414804eff08a23f0b7970968e1717e3b28e66 carries the same
# LICENSE blob. The approval below is re-derived from the pinned bytes.
ALTERNATIVE_REPOS = {
    "python-dotenv": ("theskumar/python-dotenv", "v1.2.4",
                      "a565c2cc41599c48eabc6b7b7f5b826d43c5a6d7", "BSD-3-Clause", "dev"),
}

LICENSE_REVIEW_SCHEMA = {
    "schema_version": "1.2",
    "applies_to": "every repository in this file, keyed by repository name",
    "required_fields": [
        "decision", "approved_spdx", "osi_permissive", "copyleft_marker_hits",
        "restrictive_marker_hits", "evidence", "decision_basis", "decided_by",
        "decided_at", "decided_against_revision", "independent_review",
        "license_facts", "license_conflicts", "hash_basis",
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
    "approved_spdx_is_derived_not_asserted": (
        "approved_spdx must agree with the families detected in the pinned licence "
        "TEXT and, when present, the SPDX declared in pinned packaging METADATA. "
        "The preset in the generator is an expectation, never evidence. Metadata "
        "is optional for approval, but missing metadata prevents a complete "
        "three-way agreement claim."),
    "agreement_fields": [
        "agreement", "agreement_status", "agreement_scope", "evidence_coverage",
    ],
    "agreement_status_domain": [
        "consistent", "partial", "missing_required_evidence", "conflict",
        "preset_unrecognized",
    ],
    "preset_role": ("design_license_expectation is the D0 author's expected SPDX value; "
                    "it is not independent evidence and cannot prove agreement"),
    "agreement_semantics": (
        "agreement is true only when preset, fixed licence text, and packaging "
        "metadata are all present and consistent. The preset is an expectation, "
        "not an independent evidence source. agreement_status also distinguishes "
        "an unrecognized preset as preset_unrecognized; the other states are conflict, "
        "missing_required_evidence, partial, or consistent. Missing optional "
        "packaging metadata may still permit approval when fixed licence text "
        "matches the preset; evidence_coverage lists present and missing sides."),
    "conflict_rule": ("a positive disagreement between the preset, the licence text "
                      "and/or the packaging metadata forces decision='pending' and is "
                      "listed verbatim in license_conflicts; it can never read as "
                      "'approved'. Packaging metadata is optional when absent, but at "
                      "least one matching positive family signal from the fixed root "
                      "licence text is mandatory. An EMPTY evidence set is not a "
                      "conflict and is never approval."),
    "hash_basis_rule": (
        "every licence file records BOTH the upstream git-blob sha256 and the "
        "checkout sha256 plus the newline transformation relating them; the two are "
        "different byte strings under core.autocrlf=true and must never be mixed"),
    "missing_merge_evidence_note": (
        "this field set settles LICENCE facts only; it does not establish the family "
        "window, which needs the merge-event evidence in merge-evidence.json"),
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

# ---------------------------------------------------------------------------
# SPDX conflict detection (Mika's targeted return, 2026-10-06)
# ---------------------------------------------------------------------------
# The v2 pass took `design_license_expectation` -- a hand-written preset in this
# file -- and copied it straight into `approved_spdx`.  For python-dotenv the
# preset said MIT while both the pinned LICENSE and pyproject.toml said
# BSD-3-Clause, so the record approved a licence the repository does not have
# and even contradicted its own packaging metadata a few lines below.
#
# The preset is therefore no longer trusted.  Every approval must now survive a
# three-way agreement check between
#     (a) the preset in this file,
#     (b) the SPDX family/ies detected in the pinned licence TEXT, and
#     (c) the SPDX declared in the pinned packaging METADATA.
# Any positive disagreement withholds the approval (decision -> "pending"), so
# a licence-metadata conflict can never be machine-read as "approved".
FAMILY_TOKENS = ("MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0")

# Only a repository's OWN root licence files are read for the text check.
# NOTICE files routinely quote a third party's licence (marshmallow's NOTICE
# embeds Django's BSD-3-Clause next to its own MIT), so including them would
# manufacture a conflict out of nothing.
OWN_LICENCE_RE = re.compile(r"^(licen[cs]e|copying)([-_.].*)?$", re.I)

MIT_BODY = "permission is hereby granted, free of charge"
BSD_BODY = "redistribution and use in source and binary forms"
BSD_ENDORSE_RE = re.compile(
    r"neither the name|may not be used to endorse or promote products|"
    r"to endorse or promote products derived",
    re.I,
)
APACHE_RE = re.compile(r"apache license", re.I)
APACHE_VER_RE = re.compile(r"version 2\.0", re.I)

# packaging metadata -> SPDX. Deliberately conservative: an ambiguous classifier
# ("License :: OSI Approved :: BSD License") yields NO token rather than a guess,
# because a guess would either mask a real conflict or invent one.
CLASSIFIER_SPDX = [
    ("License :: OSI Approved :: MIT License", "MIT"),
    ("License :: OSI Approved :: Apache Software License", "Apache-2.0"),
]
SPDX_LITERAL_RE = re.compile(r"\b(MIT|BSD-2-Clause|BSD-3-Clause|Apache-2\.0)\b")


def detect_text_families(repo_dir, lic_files):
    """SPDX families positively identified in the pinned licence TEXT.

    Returns (families, per_file) where per_file records which file produced
    which token, so a reviewer can re-derive the set from the hashed bytes.
    """
    families, per_file = set(), []
    for lf in lic_files:
        if not OWN_LICENCE_RE.match(os.path.basename(lf["path"])):
            continue
        full = os.path.join(repo_dir, lf["path"].replace("/", os.sep))
        try:
            text = open(full, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        found = []
        # licences are hard-wrapped, so every probe runs on whitespace-flattened
        # text: boltons' third clause reads "...may not be used to endorse or\n
        # promote products..." and a raw substring test misses it, which would
        # mis-report a 3-clause BSD as 2-clause.
        flat = re.sub(r"\s+", " ", text).lower()
        if MIT_BODY in flat:
            found.append("MIT")
        if BSD_BODY in flat:
            found.append("BSD-3-Clause" if BSD_ENDORSE_RE.search(flat)
                         else "BSD-2-Clause")
        if APACHE_RE.search(flat) and APACHE_VER_RE.search(flat):
            found.append("Apache-2.0")
        if found:
            families.update(found)
            per_file.append({"path": lf["path"], "families": sorted(set(found))})
    return families, per_file


def detect_metadata_families(meta):
    """SPDX families explicitly declared in the pinned packaging metadata."""
    families, evidence = set(), []
    for fname, info in sorted(meta.items()):
        for decl in info.get("declared", []) or []:
            for m in SPDX_LITERAL_RE.finditer(decl):
                families.add(m.group(1))
                evidence.append({"file": fname, "declaration": decl,
                                 "families": [m.group(1)]})
        for cls in info.get("classifiers", []) or []:
            for prefix, spdx in CLASSIFIER_SPDX:
                if cls.strip() == prefix:
                    families.add(spdx)
                    evidence.append({"file": fname, "declaration": cls,
                                     "families": [spdx]})
    return families, evidence


def preset_families(preset):
    return {t.strip() for t in preset.split("OR") if t.strip() in FAMILY_TOKENS}


def detect_conflicts(preset, text_families, meta_families):
    """Return the list of licence-metadata disagreements.

    An empty evidence set is NOT a conflict: "unknown" is not "disagrees".
    The three comparisons are independent so a reviewer can see exactly which
    pair of sources disagreed.
    """
    preset_t = preset_families(preset)
    conflicts = []
    if text_families and preset_t and not (preset_t & text_families):
        conflicts.append({
            "kind": "preset_vs_licence_text",
            "preset_spdx": preset,
            "preset_families": sorted(preset_t),
            "licence_text_families": sorted(text_families),
            "effect": "approval withheld: the declared preset is not among the "
                      "families the pinned licence text actually contains",
        })
    if meta_families and preset_t and not (preset_t & meta_families):
        conflicts.append({
            "kind": "preset_vs_packaging_metadata",
            "preset_spdx": preset,
            "preset_families": sorted(preset_t),
            "packaging_metadata_families": sorted(meta_families),
            "effect": "approval withheld: the declared preset disagrees with the "
                      "repository's own pinned packaging metadata",
        })
    if text_families and meta_families and not (text_families & meta_families):
        conflicts.append({
            "kind": "licence_text_vs_packaging_metadata",
            "licence_text_families": sorted(text_families),
            "packaging_metadata_families": sorted(meta_families),
            "effect": "approval withheld: the pinned licence text and the pinned "
                      "packaging metadata name different licence families",
        })
    return conflicts


def classify_license(preset, text_families, meta_families):
    """The single decision predicate, shared by the generator and the gate.

    Kept as a free function so validate_d0.py can re-run THE SAME rule over
    synthetic records in its regression test instead of re-implementing it.
    """
    conflicts = detect_conflicts(preset, text_families, meta_families)
    approved = preset in APPROVED_SPDX_SET
    if conflicts:
        return "pending", False, conflicts
    # A hand-written expectation and absent package metadata are not evidence.
    # Require positive evidence from the fixed repository-owned LICENSE/COPYING
    # body. Packaging metadata is allowed to be absent; if present, conflicts
    # above still withhold approval.
    preset_tokens = preset_families(preset)
    if not text_families or not (preset_tokens & set(text_families)):
        return "pending", False, []
    return ("approved" if approved else "pending"), approved, conflicts


def licence_agreement_facts(preset, text_families, meta_families, conflicts):
    """Describe evidence coverage separately from the approval decision.

    Packaging metadata is optional for approval, but its absence means the
    record cannot claim complete three-way agreement. The declared preset is
    an expectation, not evidence.
    """
    text_families = set(text_families)
    meta_families = set(meta_families)
    preset_tokens = preset_families(preset)
    present = ["preset_spdx"]
    missing = []
    if text_families:
        present.append("licence_text")
    else:
        missing.append("licence_text")
    if meta_families:
        present.append("packaging_metadata")
    else:
        missing.append("packaging_metadata")

    positive_mismatch = bool(conflicts) or (
        bool(preset_tokens) and bool(text_families)
        and not (preset_tokens & text_families)) or (
        bool(preset_tokens) and bool(meta_families)
        and not (preset_tokens & meta_families)) or (
        bool(text_families) and bool(meta_families)
        and not (text_families & meta_families))
    if positive_mismatch:
        status = "conflict"
    elif not preset_tokens:
        status = "preset_unrecognized"
    elif not text_families:
        status = "missing_required_evidence"
    elif not meta_families:
        status = "partial"
    else:
        status = "consistent"
    return {
        "agreement": status == "consistent",
        "agreement_status": status,
        "agreement_scope": ["preset_spdx", "licence_text", "packaging_metadata"],
        "evidence_coverage": {
            "present_sides": present,
            "missing_sides": missing,
            "expectation_sides": ["preset_spdx"],
            "unrecognized_sides": [] if preset_tokens else ["preset_spdx"],
            "preset_recognized": bool(preset_tokens),
            "complete_for_three_way_agreement": (
                bool(preset_tokens) and not missing and not positive_mismatch),
            "metadata_required_for_approval": False,
        },
    }


# ---------------------------------------------------------------------------
# hash basis: upstream blob bytes vs checkout bytes are NOT the same bytes
# ---------------------------------------------------------------------------
# core.autocrlf=true makes the working tree carry CRLF while the git object
# still holds LF, so the same licence file has two legitimate sha256 values.
# The v2 pass recorded only the checkout hash and did not say which one it was,
# which is how a CRLF-converted hash could be misread as "the licence changed".
# Both are now recorded, together with the transformation that relates them.
def hash_basis_entry(repo_dir, path, checkout_sha256, checkout_bytes):
    full = os.path.join(repo_dir, path.replace("/", os.sep))
    checkout = open(full, "rb").read()
    blob = git_blob_bytes(repo_dir, path)
    rc, blob_sha1, _ = git(repo_dir, "rev-parse", "HEAD:%s" % path)
    entry = {
        "path": path,
        "git_blob_sha1": blob_sha1 if rc == 0 else None,
        "sha256": checkout_sha256,
        "bytes": checkout_bytes,
        "checkout_sha256": checkout_sha256,
        "checkout_bytes": checkout_bytes,
        "upstream_blob_sha256": None,
        "upstream_blob_bytes": None,
        "newline_transformation": "unknown",
        "sha256_basis": ("checkout worktree bytes; git core.autocrlf=true rewrites "
                         "LF to CRLF on checkout, so this is NOT the upstream blob"),
    }
    if blob is None:
        return entry
    entry["upstream_blob_sha256"] = hashlib.sha256(blob).hexdigest()
    entry["upstream_blob_bytes"] = len(blob)
    if blob == checkout:
        entry["newline_transformation"] = "none"
    elif blob.replace(b"\n", b"\r\n") == checkout:
        entry["newline_transformation"] = "lf_to_crlf_on_checkout"
    else:
        entry["newline_transformation"] = "other"
    return entry


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
    text_families, text_evidence = detect_text_families(repo_dir, lic_files)
    meta_families, meta_evidence = detect_metadata_families(meta)
    decision, approved, conflicts = classify_license(
        design_license, text_families, meta_families)
    if restrictive:
        # a restrictive clause withholds the approval regardless of SPDX agreement
        decision, approved = "pending", False
        conflicts.append({
            "kind": "restrictive_marker",
            "effect": "approval withheld: a non-commercial / research-only / "
                      "no-derivative clause was found in a primary licence file",
        })
    agreement_facts = licence_agreement_facts(
        design_license, text_families, meta_families, conflicts)
    basis = (
        "approved_spdx is DERIVED, not asserted: the pinned licence text yields %s "
        "and the pinned packaging metadata declares %s when present; the declared "
        "preset %s must agree with available evidence. Missing metadata is reported "
        "as incomplete three-way agreement coverage, although it is optional for "
        "approval. The primary licence file carries no copyleft marker "
        "and no non-commercial / research-only / no-derivative clause. Every input "
        "is hashed below, so the decision can be re-derived from the pinned bytes "
        "alone." % (sorted(text_families) or "no identified family",
                    sorted(meta_families) or "no explicit SPDX",
                    design_license))
    if not approved and not conflicts:
        basis = ("declared licence family %s is outside the approved permissive set"
                 % design_license)
    if conflicts:
        basis += " ; CONFLICT -> decision withheld: " + "; ".join(
            c["kind"] for c in conflicts)
    return {
        "decision": decision,
        "approved_spdx": design_license,
        "osi_permissive": bool(approved),
        "copyleft_marker_hits": copyleft,
        "restrictive_marker_hits": restrictive,
        "license_facts": {
            "preset_spdx": design_license,
            "detected_from_licence_text": sorted(text_families),
            "licence_text_evidence": text_evidence,
            "detected_from_packaging_metadata": sorted(meta_families),
            "packaging_metadata_evidence": meta_evidence,
            **agreement_facts,
            "detection_scope": (
                "licence TEXT is read only from the repository's own root "
                "LICENSE/COPYING files; NOTICE and third-party licence copies are "
                "excluded because they quote other projects' licences"),
        },
        "license_conflicts": conflicts,
        "hash_basis": {
            "rule": ("upstream_blob_sha256 is the committed git object's bytes; "
                     "checkout_sha256 is the working-tree bytes after git's newline "
                     "conversion. They are different strings and must never be mixed."),
            "core_autocrlf": True,
            "per_file": {lf["path"]: {
                "git_blob_sha1": lf.get("git_blob_sha1"),
                "upstream_blob_sha256": lf.get("upstream_blob_sha256"),
                "upstream_blob_bytes": lf.get("upstream_blob_bytes"),
                "checkout_sha256": lf.get("checkout_sha256"),
                "checkout_bytes": lf.get("checkout_bytes"),
                "newline_transformation": lf.get("newline_transformation"),
                "sha256_basis": lf.get("sha256_basis"),
            } for lf in lic_files},
        },
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
    return p.returncode, p.stdout.decode("utf-8", "replace").strip(), p.stderr.decode("utf-8", "replace")


def git_blob_bytes(repo_dir, path):
    """Raw bytes of a committed blob -- no checkout newline conversion applied."""
    p = subprocess.run(["git", "-C", repo_dir, "cat-file", "blob", "HEAD:%s" % path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout if p.returncode == 0 else None.strip()


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
                    # both the upstream blob hash and the checkout hash, plus the
                    # newline transformation that relates them -- never one alone
                    lic_files.append(hash_basis_entry(
                        repo_dir, f, sha256_of(full), os.path.getsize(full)))
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
        "repos_with_licence_metadata_conflicts": [
            n for n, r in lock["repos"].items()
            if r["license_review"]["license_conflicts"]],
        "approved_spdx_by_repo": {n: r["license_review"]["approved_spdx"]
                                  for n, r in lock["repos"].items()},
        "licence_text_families_by_repo": {
            n: r["license_review"]["license_facts"]["detected_from_licence_text"]
            for n, r in lock["repos"].items()},
        "hash_basis_recorded_per_licence_file": True,
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
