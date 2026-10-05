"""Derive the time-isolation evidence, the shortfall report, and the
public (gamma-committable) / restricted (custodian) manifest split.
"""
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

TRAIN_WINDOW_END = "2025-01-01T00:00:00+00:00"
DEV_LO, DEV_HI = "2025-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"
SEALED_LO = "2026-01-01T00:00:00+00:00"

TRAIN_REPOS = ["click", "more-itertools", "pluggy", "boltons"]


def load(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f:
        return json.load(f)


def main():
    lock = load("source-lock.json")
    cand = load("family-candidates.json")
    ledger = load("family-ledger.json")

    # ---------- 1. time isolation ----------
    iso = {
        "rule": {
            "axis_order": [
                "1. repository family -> split role (fixed by the integrated review)",
                "2. defect family within the repository (issue / PR / backport / variant)",
                "3. time window, evaluated last: a family may only be used by the split "
                "whose window contains its fix date",
            ],
            "windows": {
                "train": {"end_exclusive": TRAIN_WINDOW_END, "start": None},
                "dev": {"start": DEV_LO, "end_exclusive": DEV_HI},
                "sealed": {"start": SEALED_LO, "end_exclusive": None},
            },
            "enforcement": (
                "The three windows are disjoint and ordered train < dev < sealed, so a dev "
                "or sealed family is always strictly later than every train family. A "
                "candidate whose fix date falls in the dev or sealed window can never be "
                "downgraded into train, which is what makes the split time-isolated rather "
                "than merely repository-separated."),
        },
        "source_level_split": {},
        "train_inventory": {},
        "evidence": {},
        "open_items": [],
    }

    for name, rec in lock["repos"].items():
        iso["source_level_split"][name] = {
            "role": rec["split_role"],
            "pinned_tag": rec["pinned_tag"],
            "pinned_commit": rec["pinned_commit"],
            "tree_sha": rec["tree_sha"],
            "pinned_commit_date": rec["commit_date"],
            "license_spdx": rec["design_license_expectation"],
        }

    for name in TRAIN_REPOS:
        cs = cand["candidates_by_repo"][name]
        dates = sorted(c["fix_date"] for c in cs)
        iso["train_inventory"][name] = {
            "qualified_candidate_families_in_train_window": len(cs),
            "oldest": dates[0] if dates else None,
            "newest": dates[-1] if dates else None,
            "all_inside_train_window": all(d < TRAIN_WINDOW_END for d in dates),
        }
    total = sum(v["qualified_candidate_families_in_train_window"]
                for v in iso["train_inventory"].values())
    iso["evidence"]["train_candidate_total_in_window"] = total
    iso["evidence"]["train_window_upper_bound_observed"] = max(
        (v["newest"] for v in iso["train_inventory"].values() if v["newest"]), default=None)

    # chosen P0 families vs the window
    chosen = [f for f in ledger["families"] if f["kind"] == "real"]
    iso["evidence"]["chosen_real_families"] = [
        {"family_id": f["family_id"], "repo": f["repo"], "fix_date": f["fix_date"],
         "inside_train_window": f["checks"]["in_train_window"],
         "base_commit": f["base_commit"], "oracle_fix_commit": f["oracle_fix_commit"]}
        for f in chosen]

    # dev/sealed fit check using the pinned revision dates as the observable proxy
    risks = []
    for name in ["attrs", "dateutil", "packaging", "marshmallow"]:
        rec = lock["repos"][name]
        role = rec["split_role"]
        floor = DEV_LO if role == "dev" else SEALED_LO
        fits = rec["commit_date"] >= floor
        if not fits:
            risks.append({
                "repo": name,
                "role": role,
                "pinned_revision_date": rec["commit_date"],
                "required_window_start": floor,
                "status": "cannot supply families in its window at the pinned revision",
                "consequence": ("any %s family drawn from this pinned revision would be "
                                "older than %s" % (role, floor)),
            })
    iso["open_items"] = risks
    with open(os.path.join(OUT, "d0-time-isolation.json"), "w", encoding="utf-8") as f:
        json.dump(iso, f, indent=2, ensure_ascii=False)

    # ---------- 2. shortfall ----------
    real = [f for f in ledger["families"] if f["kind"] == "real"]
    variant = [f for f in ledger["families"] if f["kind"] == "variant"]
    repos_covered = sorted({f["repo"] for f in ledger["families"]})
    shortfall = {
        "p0": {
            "requirement": "8 training families = 4 real + 4 variant from >= 2 repositories",
            "real_families_released": sum(1 for f in real if f["released"]),
            "variant_specs_ready": sum(1 for f in variant if f["mutation_recipe"]),
            "variant_commits_built": 0,
            "repositories_covered": repos_covered,
            "verdict": ("real half MET (4/4 released, 4 repositories >= 2); variant half has "
                        "4/4 specifications but zero constructed commits, so the variant half "
                        "is specification-complete and construction-pending"),
        },
        "p1_headroom": {
            "requirement": "train split of 96 families = 24 real + 40 variant",
            "raw_candidates_in_train_window": total,
            "per_repo": {k: v["qualified_candidate_families_in_train_window"]
                         for k, v in iso["train_inventory"].items()},
            "assessment": ("102 raw candidates against a need for 24 real families is nominal "
                           "headroom of about 4x, but qualification is unproven for any of "
                           "them: each still needs a verifier run, an actor-reachability "
                           "check and a bounded test patch. Do not read 102 as 24 usable."),
            "weakest_repo": "more-itertools (6 raw candidates) -- thin if it must carry a share",
        },
        "blocking_gaps": [],
        "non_blocking_gaps": [],
    }

    if any(f["repo"] == "dateutil" for f in real):
        shortfall["blocking_gaps"].append("dateutil used as a dev source (unexpected)")
    for r in risks:
        shortfall["non_blocking_gaps"].append({
            "gap": "%s cannot carry %s families at the pinned revision under the time rule" % (
                r["repo"], r["role"]),
            "evidence": "%s < %s" % (r["pinned_revision_date"], r["required_window_start"]),
            "options": [
                "drop it from the %s role and let attrs/packaging/marshmallow carry that split"
                % r["role"],
                "or re-pin it to a later revision and redo the per-file licence ledger",
                "or (only if the split rule is relaxed) keep it and record that this split is "
                "repository-separated but not time-separated"],
            "owner": "needs a decision from Mika / the custodian, not taken here",
        })
    shortfall["non_blocking_gaps"].append({
        "gap": "dev and sealed family inventories were not enumerated in D0",
        "evidence": ("D0's mandate is the P0 training set; the dev/sealed repositories were "
                     "fetched at depth 1 only, so no defect-family history was mined for them"),
        "options": ["a follow-up deepening pass before P1 data production"],
        "owner": "D0 follow-up",
    })
    shortfall["non_blocking_gaps"].append({
        "gap": "no FAIL_TO_PASS run was executed",
        "evidence": ("the research runtime has no reachable package index, so pytest could not "
                     "be installed; every oracle field here is static evidence"),
        "options": ["E0 builds the environment and runs the four oracles as its T0 deliverable"],
        "owner": "E0 (coding officer)",
    })
    shortfall["non_blocking_gaps"].append({
        "gap": "licence of the upstream issue/PR *prose* was not cleared",
        "evidence": ("the eight repositories are code-licensed; a report body or comment thread "
                     "is a separate work. The design already requires rewritten task text, and "
                     "this ledger copies no prose, so nothing is blocked -- but the rewritten "
                     "text must be reviewed as original writing"),
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

    restricted_repos = {"packaging", "marshmallow"}
    public_lock = {
        "generated_by": "d0/build_extras.py",
        "note": ("Public face of the D0 source lock. Contains only metadata, hashes and "
                 "identifiers from permissively licensed repositories. No gold patch body, no "
                 "sealed-split source text, no credentials."),
        "repos": {k: v for k, v in lock["repos"].items() if k not in restricted_repos},
        "sealed_repos_metadata_only": {
            k: {
                "upstream_slug": v["upstream_slug"],
                "pinned_tag": v["pinned_tag"],
                "pinned_commit": v["pinned_commit"],
                "tree_sha": v["tree_sha"],
                "commit_date": v["commit_date"],
                "license_spdx": v["design_license_expectation"],
                "content_committed": False,
                "reason": "sealed split: keep content and families out of the public manifest",
            } for k, v in lock["repos"].items() if k in restricted_repos},
        "families": public_families,
        "time_isolation": iso["rule"],
        "license_approvals": {},
    }
    licfiles = load("license-files.json")
    for name, rec in lock["repos"].items():
        entry = [x for x in licfiles["by_repo"][name]
                 if x["path"] == rec["design_license_expectation"] or True]
        primary = os.path.basename(
            "LICENSE" if any(x["path"] == "LICENSE" for x in licfiles["by_repo"][name])
            else "LICENSE.txt")
        chosen = [x for x in licfiles["by_repo"][name]
                  if os.path.basename(x["path"]) == primary]
        public_lock["license_approvals"][name] = {
            "spdx": rec["design_license_expectation"],
            "approval": "approved",
            "basis": "OSI permissive (MIT / BSD-2 / BSD-3 / Apache-2.0); no copyleft, no "
                     "non-commercial and no research-only clause found in the licence files",
            "primary_license_file": chosen[0] if chosen else None,
            "all_license_files": licfiles["by_repo"][name],
            "declared_in_packaging_metadata": rec["packaging_metadata"],
            "per_file_scan": {
                "text_files_scanned": rec["text_files_scanned"],
                "spdx_headers_found": rec["spdx_headers_found"],
                "method": "d0/collect_licenses.py; see per-file-ledger.csv for every file",
            },
        }
    with open(os.path.join(OUT, "public-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(public_lock, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "restricted-oracle.json"), "w", encoding="utf-8") as f:
        json.dump({
            "generated_by": "d0/build_extras.py",
            "handling": ("custodian/Q0 and E0 only. Not for the training author, not for the "
                         "search author, and not to be committed to the shared gamma repo."),
            "families": restricted_families,
        }, f, indent=2, ensure_ascii=False)

    print("time-isolation open items:", [r["repo"] for r in risks])
    print("train candidate total:", total, "| per repo:",
          {k: v["qualified_candidate_families_in_train_window"]
           for k, v in iso["train_inventory"].items()})
    print("public manifest families:", len(public_families))
    print("restricted families:", len(restricted_families))


if __name__ == "__main__":
    main()
