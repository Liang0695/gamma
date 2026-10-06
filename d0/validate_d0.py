"""D0 acceptance gate.  Exits non-zero if any check fails.

Two rules dominate this gate:

  * the headline rule from the issue -- an example with an empty SHA must never
    be released.  Check 'no_released_record_has_empty_sha' is that rule, and it
    is evaluated against every record in the ledger, not only the ones we
    expect to pass.
  * the revised time rule (policy v3-time-policy-2026-10-05-mika), which the
    gate re-asserts against the emitted JSON rather than trusting the scripts:
    train means family original fix time <= 2025-12-31, dev/sealed share
    2026-01-01 .. 2026-10-04, a family is never dated by a release or snapshot
    date, and dev/sealed carry no ordering claim.

The gate also re-asserts the machine-readable licence approval contract, so a
source-lock without an approved decision cannot pass.
"""
import csv
import datetime as dt
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

# revised windows (must match every generator script)
TRAIN_END_EXCLUSIVE = "2026-01-01T00:00:00+00:00"
DEVSEALED_START = "2026-01-01T00:00:00+00:00"
DEVSEALED_END_EXCLUSIVE = "2026-10-05T00:00:00+00:00"
WITHDRAWN = {
    "train_end_exclusive": "2025-01-01T00:00:00+00:00",
    "dev_start": "2025-01-01T00:00:00+00:00",
    "dev_end_exclusive": "2026-01-01T00:00:00+00:00",
    "sealed_start": "2026-01-01T00:00:00+00:00",
}
POLICY_ID = "v3-time-policy-2026-10-05-mika"

LICENSE_REVIEW_REQUIRED = [
    "decision", "approved_spdx", "osi_permissive", "copyleft_marker_hits",
    "restrictive_marker_hits", "evidence", "decision_basis", "decided_by",
    "decided_at", "decided_against_revision", "independent_review",
]
REJECTED_TIME_SUBSTITUTES = {"release/tag date", "snapshot or pin date",
                             "backport date", "cherry-pick date"}

results = []


def load(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f:
        return json.load(f)


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def parse(s):
    return dt.datetime.fromisoformat(s)


def main():
    lock = load("source-lock.json")
    ledger = load("family-ledger.json")
    pub = load("public-manifest.json")
    ror = load("restricted-oracle.json")
    iso = load("d0-time-isolation.json")
    short = load("d0-shortfall.json")
    fams = ledger["families"]
    real = [f for f in fams if f["kind"] == "real"]
    var = [f for f in fams if f["kind"] == "variant"]
    locked = lock["locked_candidate_repositories"]
    repos = lock["repos"]

    # ---------------- 1. pinned revisions ----------------
    bad = [n for n in locked
           if not SHA1.match(repos[n]["pinned_commit"]) or not repos[n]["commit_matches_plan"]]
    check("all_8_pinned_revisions_are_resolvable_40hex_and_match_plan", not bad,
          "offenders=%s" % bad)
    check("all_locked_repos_have_tree_sha",
          all(SHA256.match(repos[n]["tree_sha"]) or SHA1.match(repos[n]["tree_sha"])
              for n in locked))
    check("eight_locked_candidate_repositories_present", len(locked) == 8,
          "found=%d" % len(locked))
    check("locked_and_alternative_sets_are_disjoint",
          not (set(locked) & set(lock["alternative_candidate_repositories"])))

    # ---------------- 2. licence approval contract ----------------
    missing = []
    for n, r in repos.items():
        lr = r.get("license_review") or {}
        for k in LICENSE_REVIEW_REQUIRED:
            if k not in lr:
                missing.append("%s.%s" % (n, k))
    check("license_review_machine_readable_fields_present_for_every_repo",
          not missing, "missing=%s" % missing)

    notapproved = [n for n, r in repos.items()
                   if r["license_review"]["decision"] != "approved"]
    check("license_review_decision_is_approved_for_every_repo", not notapproved,
          "offenders=%s" % notapproved)
    check("license_review_decisions_cover_the_whole_locked_set",
          all(repos[n]["license_review"]["decision"] == "approved" for n in locked))
    check("license_review_records_the_exact_revision_it_covers",
          all(SHA1.match(r["license_review"]["decided_against_revision"])
              for r in repos.values()))
    check("license_review_approval_is_revision_bound_to_the_pin",
          all(r["license_review"]["decided_against_revision"] == r["pinned_commit"]
              for r in repos.values()))
    check("no_copyleft_or_restrictive_marker_found",
          all(not r["license_review"]["copyleft_marker_hits"]
              and not r["license_review"]["restrictive_marker_hits"]
              for r in repos.values()),
          "offenders=%s" % [n for n, r in repos.items()
                            if r["license_review"]["copyleft_marker_hits"]
                            or r["license_review"]["restrictive_marker_hits"]])
    check("license_review_declares_it_is_not_an_independent_countersignature",
          all(r["license_review"]["independent_review"]["status"] == "pending"
              for r in repos.values()))
    badlic = [n for n in locked if repos[n]["design_license_expectation"] not in APPROVED]
    check("all_8_licences_are_osi_permissive", not badlic, "offenders=%s" % badlic)
    check("licence_approval_recorded_for_all_locked_repos_in_public_manifest",
          all(pub["license_approvals"][n]["decision"] == "approved" for n in locked
              if n not in pub["sealed_repos_metadata_only"]))
    check("public_manifest_approval_agrees_with_source_lock",
          all(pub["license_approvals"][n]["decision"] == repos[n]["license_review"]["decision"]
              and pub["license_approvals"][n]["spdx"]
              == repos[n]["license_review"]["approved_spdx"]
              for n in pub["license_approvals"]))
    check("every_licence_file_is_hashed",
          all(SHA256.match(x["sha256"])
              for group in load("license-files.json")["by_repo"].values() for x in group))
    check("license_review_schema_declares_a_gate",
          "gate" in lock["license_review_schema"]
          and lock["license_review_schema"]["decision_domain"]
          == ["approved", "rejected", "pending"])

    # ---------------- 3. the headline rule ----------------
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

    # ---------------- 4. unreleased records still pin a real base ----------------
    badvar = [f["family_id"] for f in var
              if not SHA1.match(f.get("source_base_commit") or "")]
    check("every_variant_pins_a_real_upstream_source_base", not badvar,
          "offenders=%s" % badvar)
    check("no_released_record_carries_a_null_sha_field",
          not [f["family_id"] for f in fams if f.get("released")
               and any(f.get(k) is None for k in ("base_commit", "oracle_patch_sha256"))])

    # ---------------- 5. real half of P0 ----------------
    check("four_real_families_released", len(real) == 4 and all(f["released"] for f in real),
          "released=%d/%d" % (sum(1 for f in real if f["released"]), len(real)))
    check("p0_covers_at_least_two_repositories",
          len({f["repo"] for f in fams if f.get("released")}) >= 2,
          "repos=%s" % sorted({f["repo"] for f in fams if f.get("released")}))
    check("p0_is_eight_families_four_real_four_variant",
          len(real) == 4 and len(var) == 4, "real=%d variant=%d" % (len(real), len(var)))

    # ---------------- 6. structural checks per family ----------------
    for f in real:
        failed = [k for k, v in f["checks"].items() if not v]
        check("structural_checks_%s" % f["family_id"], not failed, "failed=%s" % failed)

    # ---------------- 7. revised time rule ----------------
    check("policy_id_is_recorded_on_ledger_and_isolation_and_shortfall",
          ledger.get("policy_id") == POLICY_ID and iso.get("policy_id") == POLICY_ID
          and short.get("policy_id") == POLICY_ID)
    check("train_window_end_is_2025_12_31_inclusive",
          iso["rule"]["windows"]["train"]["end_exclusive"] == TRAIN_END_EXCLUSIVE
          and parse(TRAIN_END_EXCLUSIVE) - dt.timedelta(days=1)
          == dt.datetime(2025, 12, 31, tzinfo=dt.timezone.utc),
          "end_exclusive=%s" % iso["rule"]["windows"]["train"]["end_exclusive"])
    ds = iso["rule"]["windows"]["dev_sealed"]
    ds_last_day = (parse(DEVSEALED_END_EXCLUSIVE) - dt.timedelta(days=1)).date()
    check("dev_sealed_window_is_2026_01_01_to_2026_10_04",
          ds["start"] == DEVSEALED_START
          and ds["end_exclusive"] == DEVSEALED_END_EXCLUSIVE
          and ds_last_day == dt.date(2026, 10, 4),
          "window=%s..%s (last day inclusive=%s)" % (
              ds["start"], ds["end_exclusive"], ds_last_day))
    check("dev_and_sealed_share_one_window_and_claim_no_ordering",
          ds.get("dev_vs_sealed_ordering_claimed") is False
          and set(ds.get("shared_by_roles", [])) == {"dev", "sealed"})
    check("withdrawn_unapproved_split_is_recorded",
          iso["rule"]["withdrawn_windows"]["train_end_exclusive"]
          == WITHDRAWN["train_end_exclusive"]
          and iso["rule"]["withdrawn_windows"]["dev_start"] == WITHDRAWN["dev_start"]
          and "withdrawn" in iso["rule"]["withdrawn_windows"]["status"]
          and "supersedes" in iso)
    check("time_axis_authority_is_the_original_fix_commit_time",
          "ORIGINAL upstream fix commit" in iso["rule"]["time_axis_authority"]
          and set(iso["rule"]["rejected_time_substitutes"]) == REJECTED_TIME_SUBSTITUTES,
          "substitutes=%s" % iso["rule"]["rejected_time_substitutes"])
    check("no_family_time_is_derived_from_a_release_or_snapshot_date",
          iso["evidence"].get("no_family_time_derived_from_release_or_snapshot_date") is True
          and all(set(f["fix_time"]["not_derived_from"]) == REJECTED_TIME_SUBSTITUTES
                  for f in real)
          and all(v.get("pinned_snapshot_date_not_used_as_family_time") is True
                  for v in list(iso["train_inventory"].values())
                  + list(iso["dev_sealed_inventory"].values())))
    check("every_snapshot_date_is_labelled_as_not_a_family_time",
          all("not a family fix time" in v["snapshot_date_role"].lower()
              for v in iso["source_level_split"].values()))
    for f in real:
        check("both_fix_dates_in_train_window_%s" % f["family_id"],
              parse(f["fix_time"]["author_date"]) < parse(TRAIN_END_EXCLUSIVE)
              and parse(f["fix_time"]["committer_date"]) < parse(TRAIN_END_EXCLUSIVE)
              and f["fix_time"]["both_dates_in_train_window"] is True,
              "author=%s committer=%s" % (f["fix_time"]["author_date"],
                                          f["fix_time"]["committer_date"]))
        check("fix_commit_inside_pinned_snapshot_%s" % f["family_id"],
              f["fix_commit_is_ancestor_of_pinned_revision"] is True)
        check("fix_commit_is_not_a_backport_or_cherry_pick_%s" % f["family_id"],
              f["backport_or_cherry_pick_marker"] is None)
        check("no_released_family_falls_in_dev_sealed_window_%s" % f["family_id"],
              not (parse(DEVSEALED_START) <= parse(f["fix_time"]["author_date"])
                   < parse(DEVSEALED_END_EXCLUSIVE)))
    check("time_isolation_open_items_are_recorded", isinstance(iso["open_items"], list))
    check("dateutil_dev_family_shortfall_is_recorded_and_counted_zero",
          any(r["repo"] == "dateutil" and r["observed_family_candidates_in_window"] == 0
              for r in iso["open_items"])
          and iso["dev_sealed_inventory"]["dateutil"][
              "qualified_candidate_families_in_window"] == 0
          and short["dev_supply"]["qualified_dev_families_found"]
          == sum(v["qualified_candidate_families_in_window"]
                 for v in iso["dev_sealed_inventory"].values() if v["role"] == "dev"))

    # ---------------- 8. alternative candidate stays a candidate ----------------
    alt = iso["alternative_dev_candidate"]
    check("alternative_candidate_is_recorded_for_python_dotenv",
          alt["repo"] == "python-dotenv"
          and SHA1.match(alt["pinned_commit"])
          and alt["license"]["decision"] == "approved"
          and alt["family_inventory"]["qualified_candidate_families_in_window"] > 0,
          "families=%d" % alt["family_inventory"][
              "qualified_candidate_families_in_window"])
    check("alternative_candidate_does_not_claim_to_replace_or_release",
          alt["replaces"] is None
          and "NOT a replacement" in alt["approval_status"]
          and "not approved for release"
          in pub["alternative_candidates_metadata"]["python-dotenv"]["status"].lower())
    check("alternative_candidate_has_environment_material",
          bool(alt["environment_material"]["files"])
          and bool(alt["environment_material"]["test_runner"]))
    check("no_family_in_the_ledger_comes_from_the_alternative_candidate",
          not [f for f in fams if f["repo"] == "python-dotenv"])

    # ---------------- 9. inventories cover the whole locked set ----------------
    covered = set(iso["train_inventory"]) | set(iso["dev_sealed_inventory"])
    check("family_inventory_covers_every_locked_repository",
          covered == set(locked), "missing=%s" % sorted(set(locked) - covered))
    check("every_inventory_entry_asserts_ancestry_and_window",
          all(v["all_fix_commits_ancestors_of_pinned_revision"] is True
              and v["all_fix_times_inside_window"] is True
              for v in list(iso["train_inventory"].values())
              + list(iso["dev_sealed_inventory"].values())))
    check("train_headroom_total_matches_per_repo_counts",
          iso["evidence"]["train_candidate_total_in_window"]
          == sum(v["qualified_candidate_families_in_window"]
                 for v in iso["train_inventory"].values()))

    # ---------------- 10. public / restricted separation ----------------
    leaked = [f["family_id"] for f in pub["families"] if "oracle_assertions" in f]
    check("public_manifest_contains_no_oracle_assertions", not leaked, "offenders=%s" % leaked)
    check("public_manifest_keeps_sealed_repo_content_out",
          all(v.get("content_committed") is False
              for v in pub["sealed_repos_metadata_only"].values()))
    check("restricted_file_covers_every_family", len(ror["families"]) == len(fams))
    check("restricted_file_is_marked_not_for_training_authors",
          "training author" in ror["handling"])
    check("restricted_file_does_not_claim_an_unverified_split",
          ror.get("split_declaration_pending") is not None)

    # ---------------- 11. per-file ledger completeness ----------------
    with open(os.path.join(OUT, "per-file-ledger.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    check("per_file_ledger_non_empty", len(rows) > 500, "rows=%d" % len(rows))
    check("per_file_ledger_every_row_has_sha256",
          all(SHA256.match(r["sha256"] or "") for r in rows))
    check("per_file_ledger_covers_every_scanned_repository",
          {r["repo"] for r in rows} == set(repos),
          "repos=%d" % len({r["repo"] for r in rows}))

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
