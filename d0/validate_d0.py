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
source-lock without an approved decision cannot pass, and it re-runs the licence
conflict predicate and the merge-evidence qualification predicate against
deliberately broken inputs, so a rule that was quietly loosened fails the gate.

It asserts CONSISTENCY, not universal success: a family may be released=false,
but then its blocking checks and its shortfall must be recorded, and it may not
be counted in any quota.
"""
import csv
import datetime as dt
import json
import os
import re
import sys
import tempfile
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
# the SAME predicate the generator used -- the gate must not re-implement the
# rule it is checking, or the two can drift apart silently
from collect_licenses import (classify_license, detect_conflicts,
                              detect_text_families)  # noqa: E402
# the status-contract constants, imported from the generator so the gate cannot
# drift from the values the emitted files were built with
from build_ledger import (STATUS_CONTRACT_ID,  # noqa: E402
                          SOURCE_PREPARATION_SCOPE, source_preparation_status,
                          training_release_decision)
import fetch_merge_evidence as merge_evidence  # noqa: E402

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
    "license_facts", "license_conflicts", "hash_basis",
]
REJECTED_TIME_SUBSTITUTES = {"release/tag date", "snapshot or pin date",
                             "backport date", "cherry-pick date"}

results = []
skipped = []


def load(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f:
        return json.load(f)


def load_optional(p):
    """Return the parsed file, or None when it is absent.

    Used for the inputs a public reviewer legitimately does not receive. An absent
    optional input yields a recorded SKIP, never a pass.
    """
    path = os.path.join(OUT, p)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def skip(name, why):
    """Record a check that CANNOT be evaluated here. Explicitly not a pass."""
    skipped.append((name, why))


def parse(s):
    return dt.datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# regression probes -- each drives the REAL predicate with a planted defect, so
# the gate fails if the rule is quietly loosened back to the old behaviour
# ---------------------------------------------------------------------------
def licence_conflict_probe():
    """Plant the python-dotenv defect the reviewer actually caught.

    The v2 record said MIT while the pinned LICENSE and pyproject.toml said
    BSD-3-Clause. Feeding exactly that mismatch to the live predicate must
    withhold the approval.
    """
    preset = "MIT"
    decision, approved, conflicts = classify_license(
        preset, {"BSD-3-Clause"}, {"BSD-3-Clause"})
    return preset, decision, approved, conflicts


def licence_evidence_probe():
    """Exercise the production licence predicate with positive, missing and conflict inputs."""
    missing = classify_license("MIT", set(), set())
    with tempfile.TemporaryDirectory() as temp_dir:
        license_path = os.path.join(temp_dir, "LICENSE")
        with open(license_path, "w", encoding="utf-8") as f:
            f.write("MIT License\nPermission is hereby granted, free of charge, "
                    "to any person obtaining a copy of this software.\n")
        positive_text_families, _ = detect_text_families(
            temp_dir, [{"path": "LICENSE"}])
    positive_without_metadata = classify_license(
        "MIT", positive_text_families, set())
    conflicting = classify_license("MIT", {"BSD-3-Clause"}, {"BSD-3-Clause"})
    return missing, positive_without_metadata, conflicting


def same_file_different_patch_probe():
    """Call the production patch predicate with the reviewer's planted same-file defect."""
    fix_diff = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
                "@@ -1 +1 @@\n-return 0\n+return 1\n")
    landing_diff = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
                    "@@ -1 +1 @@\n-return 0\n+return -999\n")

    def fake_git(repo, *args):
        if args == ("log", "-1", "--format=%P", "fix"):
            return 0, "base"
        if args == ("log", "-1", "--format=%P", "landing"):
            return 0, "base refined"
        if args == ("diff", "--no-renames", "base", "fix"):
            return 0, fix_diff
        if args == ("diff", "--no-renames", "base", "landing"):
            return 0, landing_diff
        if args[0] == "rev-list":
            return 0, ""
        if args[0] == "merge-base":
            return 1, ""
        raise AssertionError(args)

    with patch.object(merge_evidence, "git", fake_git):
        return merge_evidence.patch_equivalence(
            "fixture", "base", "fix", "landing", "two_parent_merge_commit")


def ancestor_without_landing_probe():
    """Ancestry without an attributable PR landing event must not create a landing commit."""
    reconciled = {"found": False, "candidates_examined": [],
                  "fix_commit_is_ancestor_of_pinned_revision": True}
    with patch.object(merge_evidence, "git", return_value=(1, "")), \
            patch.object(merge_evidence, "real_merge_candidate",
                         return_value=reconciled), \
            patch.object(merge_evidence, "_is_ancestor", return_value=True):
        return merge_evidence.merge_commit_geometry(
            "fixture", "pin", "missing_merge", "fix", 1)


def merge_qualification_probe(records):
    """Plant the f5485a61 defect: an in-window commit with no merge event.

    The commit message carries an issue reference (#600) but no pull request,
    so the qualifier must return False -- an issue reference is not merge
    evidence and cannot promote a record into the quota.
    """
    unverified = [r for r in records if r.get("status") != "verified"
                  or r.get("window_qualified_by_merge_event") is not True]
    if not unverified:
        return None
    probe = dict(unverified[0])
    qualified = (probe.get("status") == "verified"
                 and probe.get("window_qualified_by_merge_event") is True
                 and probe.get("merged_at_utc") is not None
                 and probe.get("merge_commit_sha") is not None)
    return probe, qualified


def main():
    lock = load("source-lock.json")
    ledger = load("family-ledger.json")
    pub = load("public-manifest.json")
    # Never open restricted-oracle.json in the public validation path. Its
    # presence, filename, or directory is not isolation evidence; the check is
    # therefore always SKIP here and remains the custodian/Q0 responsibility.
    ror = None
    iso = load("d0-time-isolation.json")
    short = load("d0-shortfall.json")
    merge = load("merge-evidence.json")
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

    # ---------------- 2b. licence-metadata conflict regression ----------------
    # Mika's second licence requirement: a licence-metadata conflict must REFUSE
    # approval. The test drives the real predicate with a conflict planted in it,
    # so it fails if the predicate is ever loosened back to "trust the preset".
    check("no_licence_metadata_conflict_is_approved",
          not [n for n, r in repos.items()
               if r["license_review"]["license_conflicts"]
               and r["license_review"]["decision"] == "approved"],
          "offenders=%s" % [n for n, r in repos.items()
                            if r["license_review"]["license_conflicts"]
                            and r["license_review"]["decision"] == "approved"])
    probe_preset, probe_decision, probe_approved, probe_conflicts = \
        licence_conflict_probe()
    check("negative_test_licence_metadata_conflict_withholds_approval",
          probe_decision == "pending" and probe_approved is False
          and bool(probe_conflicts),
          "planted preset=%s -> decision=%s conflicts=%s"
          % (probe_preset, probe_decision,
             [c["kind"] for c in probe_conflicts]))
    no_text, valid_text_no_metadata, positive_conflict = licence_evidence_probe()
    check("negative_test_missing_fixed_license_text_cannot_be_approved",
          no_text[0] == "pending" and no_text[1] is False,
          "MIT preset with no fixed text/metadata -> %s/%s" % (no_text[0], no_text[1]))
    check("positive_test_fixed_license_text_allows_missing_optional_metadata",
          valid_text_no_metadata[0] == "approved"
          and valid_text_no_metadata[1] is True
          and not valid_text_no_metadata[2],
          "MIT text with absent optional metadata -> %s/%s"
          % (valid_text_no_metadata[0], valid_text_no_metadata[1]))
    check("negative_test_fixed_license_text_metadata_conflict_stays_pending",
          positive_conflict[0] == "pending" and positive_conflict[1] is False
          and bool(positive_conflict[2]),
          "MIT preset against BSD text+metadata -> %s/%s"
          % (positive_conflict[0], positive_conflict[1]))

    same_file_patch = same_file_different_patch_probe()
    check("negative_test_same_file_with_different_patch_is_not_landing_evidence",
          same_file_patch.get("diff_text_equal") is False
          and same_file_patch.get("closed_under_the_landing_event") is False
          and same_file_patch.get("patch_correspondence") == "needs_review"
          and same_file_patch.get("file_level", [{}])[0].get("state") == "not_proven",
          "same path/different lines -> closed=%s state=%s"
          % (same_file_patch.get("closed_under_the_landing_event"),
             same_file_patch.get("file_level", [{}])[0].get("state")))
    no_landing = ancestor_without_landing_probe()
    check("negative_test_ancestor_without_merge_proof_has_no_landing_event",
          no_landing.get("adjudicated_landing_commit") is None
          and no_landing.get("landing_event_shape") == "unreconciled",
          "ancestor=True without merge proof -> landing=%s shape=%s"
          % (no_landing.get("adjudicated_landing_commit"),
             no_landing.get("landing_event_shape")))
    check("licence_decision_matches_a_fresh_re_run_of_the_same_predicate",
          all(repos[n]["license_review"]["decision"]
              == classify_license(
                  repos[n]["license_review"]["license_facts"]["preset_spdx"],
                  set(repos[n]["license_review"]["license_facts"]
                      ["detected_from_licence_text"]),
                  set(repos[n]["license_review"]["license_facts"]
                      ["detected_from_packaging_metadata"]))[0]
              and repos[n]["license_review"]["osi_permissive"]
              == classify_license(
                  repos[n]["license_review"]["license_facts"]["preset_spdx"],
                  set(repos[n]["license_review"]["license_facts"]
                      ["detected_from_licence_text"]),
                  set(repos[n]["license_review"]["license_facts"]
                      ["detected_from_packaging_metadata"]))[1]
              for n in repos),
          "decision is re-derived from the recorded facts, not trusted")
    dotenv_lr = repos["python-dotenv"]["license_review"]
    check("python_dotenv_licence_is_bsd_3_clause_not_mit",
          dotenv_lr["approved_spdx"] == "BSD-3-Clause"
          and dotenv_lr["license_facts"]["detected_from_licence_text"]
          == ["BSD-3-Clause"]
          and dotenv_lr["license_facts"]["detected_from_packaging_metadata"]
          == ["BSD-3-Clause"]
          and not dotenv_lr["license_conflicts"],
          "approved_spdx=%s text=%s" % (
              dotenv_lr["approved_spdx"],
              dotenv_lr["license_facts"]["detected_from_licence_text"]))
    check("all_licence_files_record_both_hash_bases",
          all(x.get("upstream_blob_sha256") and SHA256.match(x["upstream_blob_sha256"])
              and x.get("checkout_sha256") and SHA256.match(x["checkout_sha256"])
              and x.get("newline_transformation") in
              ("none", "lf_to_crlf_on_checkout", "other", "unknown")
              and x.get("sha256_basis")
              and x["sha256"] == x["checkout_sha256"]
              for group in load("license-files.json")["by_repo"].values()
              for x in group))
    missing_basis = [n for n, r in repos.items()
                     if set(r["license_review"]["hash_basis"]["per_file"])
                     != {x["path"] for x in r["license_review"]["evidence"]
                         ["all_license_file_hashes"]}]
    check("licence_review_hash_basis_covers_every_licence_file", not missing_basis,
          "offenders=%s" % missing_basis)
    # the python-dotenv pair the reviewer recomputed by hand: the checkout hash is
    # exactly the LF->CRLF transformation of the upstream blob, so the two hashes
    # cannot be read as a licence change
    dotenv_files = {x["path"]: x
                    for x in load("license-files.json")["by_repo"]["python-dotenv"]}
    lic_blob = dotenv_files["LICENSE"]
    check("checkout_hash_of_dotenv_licence_is_the_crlf_transformation_of_the_blob",
          lic_blob["newline_transformation"] == "lf_to_crlf_on_checkout"
          and lic_blob["upstream_blob_sha256"]
          == "80619b7049f08c81683ad0e01f08f257a840652dd71ee83146d36658c7d2c2b9"
          and lic_blob["checkout_sha256"]
          == "dd1c70c9434fdeb24d08f37fcc43a4eef9cf80db4cee3eb7755e0d888af67a19",
          "blob=%s/%s checkout=%s/%s" % (
              lic_blob["upstream_blob_sha256"][:16], lic_blob["upstream_blob_bytes"],
              lic_blob["checkout_sha256"][:16], lic_blob["checkout_bytes"]))

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
    # The gate no longer asserts "4/4 released" as a fact to be produced -- it
    # asserts that every real family's status axes are CONSISTENT with its evidence
    # and that any shortfall is recorded. Forcing the count would reward claiming a
    # family whose merge event could not be retrieved.
    released_real = [f for f in real if f["released"]]
    prepared_real = [f for f in real if f["source_preparation"]["status"] == "ready"]
    unqualified = [f for f in real
                   if f["source_preparation"]["status"] != "ready"]
    check("p0_has_four_real_and_four_variant_records",
          len(real) == 4 and len(var) == 4, "real=%d variant=%d" % (len(real), len(var)))
    check("released_real_families_all_carry_verified_merge_evidence",
          all(f["fix_time"]["qualifies_by_merge_event"] is True
              and f["merge_evidence"]["status"] == "verified"
              for f in released_real),
          "released=%d" % len(released_real))
    check("every_unprepared_real_family_records_a_reason_and_a_shortfall",
          all(f.get("release_decision", {}).get("blocking_checks")
              or f["source_preparation"]["status"] == "ready"
              for f in real)
          and all(f["source_preparation"]["status"] == "ready"
                  or any("source_preparation.status" in
                         (g if isinstance(g, str) else g.get("gap", ""))
                         for g in short["blocking_gaps"])
                  for f in unqualified),
          "unprepared=%s" % [f["family_id"] for f in unqualified])
    check("source_preparation_count_is_truthfully_carried_into_the_shortfall_report",
          short["p0"]["real_families_source_prepared"] == len(prepared_real)
          and short["p0"]["real_families_total"] == len(real)
          and [x["family_id"] for x in
               short["p0"]["real_families_source_preparation_shortfall"]]
          == [f["family_id"] for f in unqualified]
          and [x["status"] for x in
               short["p0"]["real_families_source_preparation_shortfall"]]
          == [f["source_preparation"]["status"] for f in unqualified],
          "shortfall says %d/%d prepared" % (
              short["p0"]["real_families_source_prepared"],
              short["p0"]["real_families_total"]))
    check("training_release_count_is_truthfully_carried_into_the_shortfall_report",
          short["p0"]["real_families_training_released"]
          == sum(1 for f in real
                 if f["training_release"]["approved_for_training"]),
          "shortfall says %d training-released"
          % short["p0"]["real_families_training_released"])
    check("p0_covers_at_least_two_repositories",
          len({f["repo"] for f in prepared_real}) >= 2,
          "repos=%s" % sorted({f["repo"] for f in prepared_real}))
    boltons = [f for f in real if f["repo"] == "boltons"]
    check("boltons_unproven_patch_correspondence_is_needs_review_and_shortfall",
          len(boltons) == 1
          and boltons[0]["source_preparation"]["status"] == "needs_review"
          and boltons[0]["checks"].get("landing_patch_correspondence_needs_review") is True
          and boltons[0]["family_id"]
          in short["p0"]["real_families_needs_review"]
          and any(x["family_id"] == boltons[0]["family_id"]
                  for x in short["p0"]["real_families_source_preparation_shortfall"]),
          "boltons=%s shortfall=%s" % (
              [(f["source_preparation"]["status"],
                f["checks"].get("landing_patch_correspondence_needs_review"))
               for f in boltons],
              short["p0"]["real_families_source_preparation_shortfall"]))

    # ---------------- 5b. the two status axes (Mika ruling 2026-10-06) -----------
    # The previous revision shipped one `released` boolean that mixed a static
    # artefact statement with a release decision. These checks exist so that cannot
    # come back: `released` may only ever mirror the source-preparation axis, and no
    # family may claim a training release without positive evidence on all four
    # required fronts.
    axis_missing = [f["family_id"] for f in fams
                    if not isinstance(f.get("source_preparation"), dict)
                    or not isinstance(f.get("training_release"), dict)
                    or not isinstance(f.get("status_axes"), dict)]
    check("every_family_carries_both_status_axes", not axis_missing,
          "offenders=%s" % axis_missing)

    scope_drift = [f["family_id"] for f in fams
                   if f.get("released_scope") != SOURCE_PREPARATION_SCOPE]
    check("every_family_scopes_its_released_alias_to_source_preparation",
          not scope_drift, "offenders=%s" % scope_drift)

    alias_mismatch = [f["family_id"] for f in fams
                      if f.get("released") !=
                      (f["source_preparation"]["status"] == "ready")]
    check("released_alias_mirrors_only_the_source_preparation_axis",
          not alias_mismatch, "offenders=%s" % alias_mismatch)

    unverified_release_claims = []
    for f in fams:
        tr = f["training_release"]
        if tr["status"] != "approved" and tr.get("approved_for_training"):
            unverified_release_claims.append("%s:approved_without_status" % f["family_id"])
        missing = sorted(k for k, v in tr["requires"].items() if not v["verified"])
        if tr.get("approved_for_training") and missing:
            unverified_release_claims.append("%s:missing=%s" % (f["family_id"], missing))
        if not tr.get("approved_for_training") and not tr.get("reason"):
            unverified_release_claims.append("%s:blocked_without_reason" % f["family_id"])
    check("no_family_is_training_released_without_every_evidence_item",
          not unverified_release_claims,
          "offenders=%s" % unverified_release_claims)

    gate_contract_drift = (
        ledger["status_contract"]["contract_id"] != STATUS_CONTRACT_ID
        or short["status_contract"]["contract_id"] != STATUS_CONTRACT_ID
        or pub["status_contract"]["contract_id"] != STATUS_CONTRACT_ID
        or ledger["status_contract"]["compatibility_field_scope"]
        != SOURCE_PREPARATION_SCOPE
        or short["status_contract"]["compatibility_field_scope"]
        != SOURCE_PREPARATION_SCOPE
        or pub["status_contract"]["compatibility_field_scope"]
        != SOURCE_PREPARATION_SCOPE)
    check("status_contract_id_and_scope_agree_across_ledger_shortfall_and_manifest",
          not gate_contract_drift,
          "ids=%s" % {ledger["status_contract"]["contract_id"],
                      short["status_contract"]["contract_id"],
                      pub["status_contract"]["contract_id"]})

    # NEGATIVE TEST: plant a family that claims a training release while one evidence
    # item is unverified. The predicate that decides admission must refuse it, so the
    # rule is exercised rather than merely asserted about the current data.
    planted = json.loads(json.dumps(fams[0]))
    planted["training_release"]["approved_for_training"] = True
    planted["training_release"]["status"] = "approved"
    planted_missing = sorted(k for k, v in planted["training_release"]["requires"].items()
                             if not v["verified"])
    check("negative_test_a_training_release_claim_without_evidence_cannot_be_admitted",
          bool(planted_missing)
          and not (planted["training_release"]["approved_for_training"]
                   and not planted_missing),
          "planted missing evidence=%s -> admitted=%s" % (
              planted_missing,
              planted["training_release"]["approved_for_training"]
              and not planted_missing))

    check("public_manifest_carries_the_same_two_axis_contract_as_the_ledger",
          pub["status_contract"]["axes"] == ledger["status_contract"]["axes"]
          and pub["training_release_gate"]["blocked_families"]
          == [f["family_id"] for f in fams
              if not f["training_release"]["approved_for_training"]]
          and pub["training_release_gate"]["approved_families"] == [],
          "approved=%s blocked=%d" % (
              pub["training_release_gate"]["approved_families"],
              len(pub["training_release_gate"]["blocked_families"])))

    csv_rows = list(csv.DictReader(open(os.path.join(OUT, "kaggle-23-d0-family-table.csv"),
                                        encoding="utf-8")))
    csv_offenders = [r["family_id"] for r in csv_rows
                     if "released" in r and r.get("released") != ""
                     or "source_preparation_status" not in r
                     or "training_release_status" not in r]
    check("csv_exposes_both_axes_and_never_a_bare_released_column",
          not csv_offenders and len(csv_rows) == len(fams),
          "rows=%d offenders=%s" % (len(csv_rows), csv_offenders))
    csv_alias_drift = [r["family_id"] for r, f in zip(csv_rows, fams)
                       if r["source_preparation_status"]
                       != f["source_preparation"]["status"]
                       or r["training_release_status"] != f["training_release"]["status"]]
    check("csv_axis_columns_agree_with_the_ledger", not csv_alias_drift,
          "offenders=%s" % csv_alias_drift)

    # ---------------- 6. structural checks per family ----------------
    # The gate asserts that each status is CONSISTENT with the family's own checks
    # and that an unprepared family records why. It must not assert that every family
    # is ready: that would reward publishing a family whose merge event was never
    # retrieved, which is exactly what this revision forbids.
    for f in real:
        failed = sorted(k for k, v in f["checks"].items() if not v)
        check("source_preparation_status_matches_the_structural_checks_%s" % f["family_id"],
              f["released"] == (not failed),
              "prep=%s failed=%s" % (f["source_preparation"]["status"], failed))
        check("unprepared_family_records_its_blocking_checks_%s" % f["family_id"],
              f["released"]
              or f["source_preparation"]["blocking_checks"] == failed,
              "blocking=%s" % f["source_preparation"]["blocking_checks"])

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
    check("time_axis_authority_is_the_original_merge_event",
          "MERGE event" in iso["rule"]["time_axis_authority"]
          and "audit corroboration ONLY" in iso["rule"]["time_axis_authority"]
          and set(iso["rule"]["rejected_time_substitutes"]) == REJECTED_TIME_SUBSTITUTES,
          "substitutes=%s" % iso["rule"]["rejected_time_substitutes"])
    check("merge_evidence_artifact_is_declared_with_its_rule",
          iso["rule"]["merge_event_evidence"]["artifact"]
          == "out/merge-evidence.json"
          and "unverified" in iso["rule"]["merge_event_evidence"]
          ["missing_evidence_rule"],
          "records=%d verified=%d unverified=%d" % (
              iso["rule"]["merge_event_evidence"]["records"],
              iso["rule"]["merge_event_evidence"]["verified"],
              iso["rule"]["merge_event_evidence"]["unverified"]))
    check("merge_evidence_records_carry_a_source_and_a_response_hash",
          all(r["commit_to_pr_lookup"]["url"].startswith("https://api.github.com/")
              and SHA256.match(r["commit_to_pr_lookup"]["raw_response_sha256"])
              for r in merge["records"])
          and all(r["pull_request"]["url"].startswith("https://api.github.com/")
                  and SHA256.match(r["pull_request"]["raw_response_sha256"])
                  for r in merge["records"] if r["pull_request"]))
    check("merge_evidence_is_labelled_unauthenticated_and_token_free",
          "no token" in merge["authentication"])
    check("author_and_committer_dates_are_downgraded_to_audit_fields",
          all(f["fix_time"]["author_and_committer_dates_role"].startswith(
              "audit corroboration ONLY") for f in real)
          and all(f["fix_time"]["basis"].startswith("merged_at_utc")
                  for f in released_real))
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
        check("fix_commit_inside_pinned_snapshot_%s" % f["family_id"],
              f["fix_commit_is_ancestor_of_pinned_revision"] is True)
        check("fix_commit_is_not_a_backport_or_cherry_pick_%s" % f["family_id"],
              f["backport_or_cherry_pick_marker"] is None)
        # audit-only corroboration: both commit dates must agree with the window,
        # but they are never the basis and this check alone cannot release anyone
        check("commit_dates_corroborate_the_window_audit_only_%s" % f["family_id"],
              parse(f["fix_time"]["author_date"]) < parse(TRAIN_END_EXCLUSIVE)
              and f["fix_time"]["both_dates_in_train_window"] is True,
              "author=%s committer=%s" % (f["fix_time"]["author_date"],
                                          f["fix_time"]["committer_date"]))
    for f in released_real:
        check("merge_event_in_train_window_%s" % f["family_id"],
              parse(f["fix_time"]["merge_event_utc"]) < parse(TRAIN_END_EXCLUSIVE)
              and f["merge_evidence"]["merged_at_utc"]
              == f["fix_time"]["merge_event_utc"]
              and f["merge_evidence"]["status"] == "verified",
              "merged_at=%s" % f["fix_time"]["merge_event_utc"])
        check("no_released_family_falls_in_dev_sealed_window_%s" % f["family_id"],
              not (parse(DEVSEALED_START) <= parse(f["fix_time"]["merge_event_utc"])
                   < parse(DEVSEALED_END_EXCLUSIVE)))
    # Mika's second time requirement as an executable negative test
    probe = merge_qualification_probe(merge["records"])
    check("negative_test_missing_merge_evidence_cannot_qualify",
          probe is not None and probe[1] is False,
          "planted fix_commit=%s status=%s -> qualified=%s"
          % (probe[0]["fix_commit"][:12], probe[0]["status"], probe[1]) if probe
          else "no unverified record to probe")
    check("unverified_records_are_counted_in_no_quota",
          all(not r["window_qualified_by_merge_event"]
              for r in merge["records"] if r["status"] != "verified")
          and all(f["fix_time"]["qualifies_by_merge_event"] is False
                  for f in real if f["merge_evidence"].get("status") != "verified"
                  or f["merge_evidence"].get("window_qualified_by_merge_event") is not True),
          "unverified=%s" % merge["summary"]["unverified_commits"])
    check("issue_reference_is_not_treated_as_merge_evidence",
          all(r.get("issue_reference_is_not_merge_evidence") is True
              for r in merge["records"]))

    # ---------------- 7b. landing-event reconciliation (Mika ruling 2026-10-06) ----
    # The review refused to accept "the associated pull request was merged" as proof
    # that a family is sound, because for several counted families GitHub's
    # merge_commit_sha is NOT the commit in the pinned snapshot -- and for boltons it is
    # absent from the snapshot entirely. These checks require every counted family to
    # carry a landing event re-derived from the pinned bytes, an adjudication of how the
    # PR head, the merge SHA and the snapshot relate, and a patch-equivalence result.
    landing_missing = [
        f["family_id"] for f in real
        if not (f["merge_evidence"].get("adjudicated_landing_commit")
                and f["merge_evidence"].get("landing_event_adjudication")
                and f["merge_evidence"].get("landing_event_shape"))]
    check("every_counted_family_records_an_adjudicated_landing_event",
          not landing_missing, "offenders=%s" % landing_missing)

    landing_not_ancestor = [
        f["family_id"] for f in real
        if f["merge_evidence"].get(
            "adjudicated_landing_is_ancestor_of_pinned_revision") is not True]
    check("every_adjudicated_landing_event_is_an_ancestor_of_the_pinned_revision",
          not landing_not_ancestor, "offenders=%s" % landing_not_ancestor)

    # The three shapes must be NAMED rather than collapsed into one vague statement.
    shapes = {f["family_id"]: f["merge_evidence"].get("landing_event_shape") for f in real}
    check("landing_event_shape_is_one_of_the_defined_values",
          all(s in ("two_parent_merge_commit", "fix_commit_is_the_landing_commit")
              for s in shapes.values()),
          "shapes=%s" % shapes)
    check("a_two_parent_landing_event_is_actually_a_two_parent_merge",
          all(f["merge_evidence"]["landing_event_evidence"]["landing_commit_is_a_two_parent_merge"]
              is True
              for f in real
              if f["merge_evidence"].get("landing_event_shape") == "two_parent_merge_commit"),
          "checked=%d" % sum(
              1 for f in real
              if f["merge_evidence"].get("landing_event_shape")
              == "two_parent_merge_commit"))

    eq_missing = [f["family_id"] for f in real
                  if (f["merge_evidence"].get("patch_equivalence") or {}).get(
                      "computed") is not True]
    check("patch_equivalence_was_computed_for_every_counted_family",
          not eq_missing, "offenders=%s" % eq_missing)

    not_covered = [
        (f["family_id"],
         (f["merge_evidence"]["patch_equivalence"] or {}).get("fix_files_not_covered"))
        for f in real
        if f["source_preparation"]["status"] == "ready"
        if (f["merge_evidence"].get("patch_equivalence") or {}).get(
            "closed_under_the_landing_event") is not True]
    check("every_fix_file_is_carried_by_the_landing_event",
          not not_covered, "offenders=%s" % not_covered)

    # boltons-specific: the discrepancy must be EXPLAINED, not silently aligned. The
    # requirement is a recorded landing event plus a reason the API SHA differs.
    boltons = [f for f in real if f["repo"] == "boltons"]
    boltons_ok = bool(boltons) and all(
        (f["merge_evidence"].get("merge_commit_geometry") or {})
        .get("merge_commit_present_in_pinned_clone") is False
        and "absent" in (f["merge_evidence"].get("landing_event_adjudication") or "")
        and (f["merge_evidence"].get("patch_equivalence") or {}).get(
            "fix_commit_is_ancestor_of_the_landing_merged_in_commit") is True
        for f in boltons)
    check("boltons_missing_merge_object_is_explained_with_ancestry_evidence",
          boltons_ok,
          "geometry=%s" % [
              ((f["merge_evidence"].get("merge_commit_geometry") or {})
               .get("merge_commit_present_in_pinned_clone"),
               (f["merge_evidence"].get("patch_equivalence") or {})
               .get("fix_commit_is_ancestor_of_the_landing_merged_in_commit"))
              for f in boltons])
    check("boltons_reconciliation_names_the_commit_the_pinned_merge_actually_brought_in",
          all((f["merge_evidence"].get("patch_equivalence") or {}).get(
              "landing_merged_in_commit") for f in boltons),
          "landing_merged_in=%s" % [
              (f["merge_evidence"].get("patch_equivalence") or {}).get(
                  "landing_merged_in_commit") for f in boltons])

    # NEGATIVE TEST: strip the landing evidence from a counted family and require the
    # adjudication predicate to stop calling it reconciled.
    planted_geo = json.loads(json.dumps(
        real[0]["merge_evidence"]["merge_commit_geometry"]))
    planted_geo["reconciliation"] = {"found": False, "matching_merge_commit": None}
    planted_present = planted_geo.get("merge_commit_present_in_pinned_clone")
    planted_fix_lands = planted_geo["reconciliation"].get(
        "fix_commit_is_ancestor_of_pinned_revision")
    planted_admitted = bool(planted_geo["reconciliation"].get("matching_merge_commit")) or (
        planted_present and planted_fix_lands)
    check("negative_test_a_family_without_landing_evidence_cannot_be_adjudicated",
          not planted_admitted,
          "planted found=False present=%s -> admitted=%s" % (planted_present, planted_admitted))

    check("replaced_family_left_a_retrievable_failure_record",
          any(r.get("status") == "replaced"
              and r.get("counted_in_no_quota") is True
              for r in merge.get("replaced_records", []))
          and any(f.get("replaces_fix_commit")
                  for f in real),
          "replaced_records=%d" % len(merge.get("replaced_records", [])))
    check("the_replaced_commit_is_counted_in_no_quota_and_is_not_a_family",
          all(r["status"] == "replaced" for r in merge.get("replaced_records", []))
          and not any(f.get("oracle_fix_commit") == r["fix_commit"]
                      for f in fams for r in merge.get("replaced_records", [])))
    check("no_variant_still_inherits_from_a_retired_parent",
          all(f["derived_from"] in {x["family_id"] for x in real} for f in var),
          "derived_from=%s" % {f["derived_from"] for f in var})
    check("retired_variant_family_is_gone_and_its_replacement_is_present",
          not any(f["family_id"] == "v3-train-click-001-var-rename" for f in fams)
          and any(f["family_id"] == "v3-train-click-001-var-predicate" for f in fams),
          "variants=%s" % [f["family_id"] for f in var])
    check("time_isolation_open_items_are_recorded", isinstance(iso["open_items"], list))
    check("dateutil_dev_family_shortfall_is_recorded_and_counted_zero",
          any(r["repo"] == "dateutil" and r["observed_family_candidates_in_window"] == 0
              for r in iso["open_items"])
          and iso["dev_sealed_inventory"]["dateutil"][
              "screened_candidate_commits_in_window"] == 0
          and short["dev_supply"]["screened_dev_candidates_found"]
          == sum(v["screened_candidate_commits_in_window"]
                 for v in iso["dev_sealed_inventory"].values() if v["role"] == "dev"))

    # ---------------- 8. alternative candidate stays a candidate ----------------
    alt = iso["alternative_dev_candidate"]
    alt_fi = alt["family_inventory"]
    check("alternative_candidate_is_recorded_for_python_dotenv",
          alt["repo"] == "python-dotenv"
          and SHA1.match(alt["pinned_commit"])
          and alt["license"]["decision"] == "approved"
          and alt["license"]["approved_spdx"] == "BSD-3-Clause"
          and alt_fi["merge_event_verified_and_in_window"] > 0,
          "screened=%d merge-verified=%d spdx=%s" % (
              alt_fi["screened_candidate_commits_in_window"],
              alt_fi["merge_event_verified_and_in_window"],
              alt["license"]["approved_spdx"]))
    check("alternative_candidate_families_separate_screened_from_qualified",
          all("counted_in_qualified_quota" in f for f in alt_fi["families"])
          and all(f["counted_in_qualified_quota"]
                  == (f["merge_evidence"]["status"] == "verified"
                      and f["merge_evidence"]["window_qualified_by_merge_event"] is True)
                  for f in alt_fi["families"])
          and sum(1 for f in alt_fi["families"] if f["counted_in_qualified_quota"])
          == alt_fi["merge_event_verified_and_in_window"],
          "one in-window commit has no merge event and is excluded")
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
          iso["evidence"]["train_candidate_total_screened_in_window"]
          == sum(v["screened_candidate_commits_in_window"]
                 for v in iso["train_inventory"].values()))
    check("inventory_counts_declare_themselves_as_screenings_not_quotas",
          all("screened_candidate_commits_in_window" in v
              and "merge_event_verified_and_in_window" in v
              and "COMMIT-DATE screening" in v["count_semantics"]
              for v in list(iso["train_inventory"].values())
              + list(iso["dev_sealed_inventory"].values())))
    check("merge_verified_inventory_count_never_exceeds_the_screened_count",
          all(v["merge_event_verified_and_in_window"]
              <= v["screened_candidate_commits_in_window"]
              for v in list(iso["train_inventory"].values())
              + list(iso["dev_sealed_inventory"].values())))

    # ---------------- 10. public / restricted separation ----------------
    # The public validator MUST be runnable by a reviewer who holds only the committed
    # files. restricted-oracle.json carries gold-adjacent material and is deliberately
    # NOT committed, so its checks are skipped here -- and a skip is recorded as a skip,
    # never as a pass and never as evidence of isolation.
    leaked = [f["family_id"] for f in pub["families"] if "oracle_assertions" in f]
    check("public_manifest_contains_no_oracle_assertions", not leaked, "offenders=%s" % leaked)
    check("public_manifest_keeps_sealed_repo_content_out",
          all(v.get("content_committed") is False
              for v in pub["sealed_repos_metadata_only"].values()))
    if ror is None:
        for nm, why in [
            ("restricted_file_covers_every_family",
             "restricted-oracle.json is deliberately not opened by the public validator; "
             "a reviewer with only the repository cannot evaluate its contents"),
            ("restricted_file_is_marked_not_for_training_authors",
             "input deliberately not opened by the public validator"),
            ("restricted_file_does_not_claim_an_unverified_split",
             "input deliberately not opened by the public validator"),
            ("restricted_oracle_split_is_really_isolated_from_the_actor",
             "no negative permission test exists; file presence/name/access path would "
             "not establish isolation"),
        ]:
            skip(nm, why)
    else:
        check("restricted_file_covers_every_family", len(ror["families"]) == len(fams))
        check("restricted_file_is_marked_not_for_training_authors",
              "training author" in ror["handling"])
        check("restricted_file_does_not_claim_an_unverified_split",
              ror.get("split_declaration_pending") is not None)
        # A file merely being named "restricted" or living in another directory is not
        # isolation. Without a negative permission test this stays UNPROVEN.
        skip("restricted_oracle_split_is_really_isolated_from_the_actor",
             "the file is present but no negative permission test (author cannot read it "
             "or self-authorise gold) has been produced, so isolation is unproven")

    # ---------------- 10b. gold derivability and actor-side isolation -----------
    # Mika's yellow items 6 and 7: state explicitly whether the train gold can be
    # derived from what the actor receives, and declare the actor-side isolation gaps
    # instead of leaving them implied.
    deriv = ledger["gold_derivability"]
    check("train_gold_derivability_is_stated_for_every_family",
          set(deriv["per_family"]) == {f["family_id"] for f in fams},
          "covered=%d of %d" % (len(deriv["per_family"]), len(fams)))
    check("gold_is_withheld_from_the_actor_in_every_family",
          all(v["gold_patch_reaches_the_actor"] is False
              for v in deriv["per_family"].values()))
    check("every_family_states_what_the_actor_receives_so_gold_is_not_reconstructible",
          all(v.get("actor_receives") and v.get("derivability_argument")
              for v in deriv["per_family"].values()))
    check("actor_side_isolation_gaps_are_declared_rather_than_implied",
          all(g.get("status") in ("unproven", "gap_declared")
              for g in ledger["actor_isolation"]["gaps"])
          and len(ledger["actor_isolation"]["gaps"]) >= 2,
          "gaps=%s" % [g["id"] for g in ledger["actor_isolation"]["gaps"]])
    check("actor_isolation_is_not_claimed_anywhere_in_the_deliverable",
          all(g["status"] == "unproven" for g in ledger["actor_isolation"]["gaps"])
          and deriv["summary"]["isolation_established"] is False
          and all(v["gold_patch_reaches_the_actor"] is False
                  for v in deriv["per_family"].values()),
          "isolation_established=%s" % deriv["summary"]["isolation_established"])

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
    for name, why in skipped:
        # SKIP is deliberately not PASS: a check that could not be evaluated is not
        # evidence, and the transcript has to make that unmistakable.
        print("SKIP  %-*s  %s" % (width, name, why))
    print("\n%d checks, %d failed, %d skipped (a skip is NOT a pass and is NOT "
          "isolation evidence)" % (len(results), failed, len(skipped)))
    if skipped:
        print("skipped checks: %s" % ", ".join(n for n, _ in skipped))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
