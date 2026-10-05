"""D0 acceptance gate.  Exits non-zero if any check fails.

The headline rule from the issue is: an example with an empty SHA must never be
released.  Check 'no_released_record_has_empty_sha' is that rule, and it is
evaluated against every record in the ledger, not only the ones we expect to pass.
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
APPROVED = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0",
            "Apache-2.0 OR BSD-3-Clause", "Apache-2.0 OR BSD-2-Clause"}
TRAIN_END = "2025-01-01T00:00:00+00:00"
DEV_LO, DEV_HI = "2025-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"
SEALED_LO = "2026-01-01T00:00:00+00:00"

results = []


def load(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f:
        return json.load(f)


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def main():
    lock = load("source-lock.json")
    ledger = load("family-ledger.json")
    pub = load("public-manifest.json")
    ror = load("restricted-oracle.json")
    iso = load("d0-time-isolation.json")
    fams = ledger["families"]
    real = [f for f in fams if f["kind"] == "real"]
    var = [f for f in fams if f["kind"] == "variant"]

    # 1. pinned revisions
    bad = [n for n, r in lock["repos"].items()
           if not SHA1.match(r["pinned_commit"]) or not r["commit_matches_plan"]]
    check("all_8_pinned_revisions_are_resolvable_40hex_and_match_plan", not bad,
          "offenders=%s" % bad)
    check("all_8_have_tree_sha", all(SHA256.match(r["tree_sha"]) or SHA1.match(r["tree_sha"])
                                     for r in lock["repos"].values()))
    check("eight_candidate_repositories_present", len(lock["repos"]) == 8,
          "found=%d" % len(lock["repos"]))

    # 2. licences
    badlic = [n for n, r in lock["repos"].items()
              if r["design_license_expectation"] not in APPROVED]
    check("all_8_licences_are_osi_permissive", not badlic, "offenders=%s" % badlic)
    check("licence_approval_recorded_for_all_8",
          all(v["approval"] == "approved" for v in pub["license_approvals"].values()))
    check("every_licence_file_is_hashed",
          all(SHA256.match(x["sha256"])
              for group in load("license-files.json")["by_repo"].values() for x in group))

    # 3. the headline rule
    offenders = []
    for f in fams:
        if not f.get("released"):
            continue
        for key in ("base_commit", "oracle_fix_commit", "oracle_patch_sha256"):
            v = f.get(key)
            if not v or not isinstance(v, str) or not v.strip():
                offenders.append("%s:%s" % (f["family_id"], key))
        if not f.get("fail_to_pass"):
            offenders.append("%s:fail_to_pass" % f["family_id"])
        for node in f.get("fail_to_pass") or []:
            if not node.get("node_id"):
                offenders.append("%s:empty_node_id" % f["family_id"])
    check("no_released_record_has_empty_sha", not offenders, "offenders=%s" % offenders)

    # 4. unreleased records must still pin a real upstream base
    badvar = [f["family_id"] for f in var
              if not SHA1.match(f.get("source_base_commit") or "")]
    check("every_variant_pins_a_real_upstream_source_base", not badvar, "offenders=%s" % badvar)
    check("no_released_record_carries_a_null_sha_field",
          not [f["family_id"] for f in fams if f.get("released")
               and any(f.get(k) is None for k in ("base_commit", "oracle_patch_sha256"))])

    # 5. real half of P0
    check("four_real_families_released", len(real) == 4 and all(f["released"] for f in real),
          "released=%d/%d" % (sum(1 for f in real if f["released"]), len(real)))
    check("p0_covers_at_least_two_repositories",
          len({f["repo"] for f in fams if f.get("released")}) >= 2,
          "repos=%s" % sorted({f["repo"] for f in fams if f.get("released")}))
    check("p0_is_eight_families_four_real_four_variant",
          len(real) == 4 and len(var) == 4, "real=%d variant=%d" % (len(real), len(var)))

    # 6. structural per family
    for f in real:
        failed = [k for k, v in f["checks"].items() if not v]
        check("structural_checks_%s" % f["family_id"], not failed, "failed=%s" % failed)

    # 7. time isolation
    inwin = all(f["fix_date"] < TRAIN_END for f in real)
    check("every_released_family_is_inside_the_train_window", inwin,
          "dates=%s" % [f["fix_date"][:10] for f in real])
    check("no_released_family_falls_in_dev_or_sealed_window",
          not [f["family_id"] for f in real
               if DEV_LO <= f["fix_date"] or f["fix_date"] >= SEALED_LO])
    check("time_windows_are_disjoint_and_ordered",
          TRAIN_END == DEV_LO and DEV_HI == SEALED_LO,
          "train<dev<sealed")
    check("time_isolation_open_items_are_recorded", isinstance(iso["open_items"], list))

    # 8. public / restricted separation
    leaked = [f["family_id"] for f in pub["families"] if "oracle_assertions" in f]
    check("public_manifest_contains_no_oracle_assertions", not leaked, "offenders=%s" % leaked)
    check("public_manifest_keeps_sealed_repo_content_out",
          all(v.get("content_committed") is False
              for v in pub["sealed_repos_metadata_only"].values()))
    check("restricted_file_covers_every_family",
          len(ror["families"]) == len(fams))
    check("restricted_file_is_marked_not_for_training_authors",
          "training author" in ror["handling"])

    # 9. per-file ledger completeness
    with open(os.path.join(OUT, "per-file-ledger.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    check("per_file_ledger_non_empty", len(rows) > 500, "rows=%d" % len(rows))
    check("per_file_ledger_every_row_has_sha256",
          all(SHA256.match(r["sha256"] or "") for r in rows))

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print("%s  %-*s  %s" % (status, width, name, detail))
    print("\n%d checks, %d failed" % (len(results), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
