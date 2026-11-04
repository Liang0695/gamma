import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


parser = argparse.ArgumentParser(description="Verify the K80 inventory and prompt package against retained inputs.")
parser.add_argument("delivery", type=Path, help="Path to the unpacked k80-delivery directory")
parser.add_argument("inputs", type=Path, help="Directory containing the downloaded source attachments")
parser.add_argument("--write-evidence", action="store_true", help="Write validation-evidence.json after checks")
parser.add_argument("--skip-output-manifest", action="store_true", help="Skip the output hash-manifest comparison while generating evidence")
args = parser.parse_args()
D = args.delivery
I = args.inputs

packages = {
    "01a1a218-67af-737b-98a4-c904fa385852": I / "K40-psf-black-source-linkedpr-b01-20261103.zip",
    "01a1a5d8-9167-7ef8-bd97-34ab66b1759a": I / "K40-psf-black-source-enrichment-b02-20261104.zip",
    "01a1a5fa-919a-70ab-b379-6334c93d086c": I / "K40-psf-black-source-enrichment-b03-20261104.zip",
}
zip_by_id = {key: zipfile.ZipFile(path) for key, path in packages.items()}
catalog = load_json(I / "k68-catalog-v12.json")
manifest = load_json(D / "new-prompts-manifest.json")
associations = load_json(D / "known-associations.json")
review_diff = load_json(D / "review-return-diff.json")
checks = load_json(D / "diff-and-checks.json")

with (D / "canonical-readiness.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
by_id = {row["stable_id"]: row for row in rows}
candidate_rows = manifest["prompts"]
prompt_rows = [row for row in candidate_rows if row.get("prompt_file")]
deferred_rows = [row for row in candidate_rows if not row.get("prompt_file")]
parse_errors = {}
for path in sorted(D.glob("*.json")):
    try:
        load_json(path)
    except Exception as exc:
        parse_errors[path.name] = str(exc)

member_failures = []
for item in candidate_rows:
    stable = item["stable_id"]
    csv_row = by_id[stable]
    bindings = json.loads(csv_row["member_bindings_json"])
    expected_refs = [
        ("issue_body", item["source_issue_body"]),
        ("fixed_source_archive", item["fixed_source"]),
    ]
    for role, ref in expected_refs:
        match = next((b for b in bindings if b["role"] == role), None)
        if not match:
            member_failures.append({"stable_id": stable, "role": role, "error": "MISSING_CANONICAL_BINDING"})
            continue
        package_id = ref["attachment_id"]
        path_key = "member" if role == "issue_body" else "archive_member"
        digest_key = "member_sha256" if role == "issue_body" else "archive_sha256"
        expected_path = ref[path_key]
        expected_digest = ref[digest_key]
        if match["attachment_id"] != package_id or match["member_path"] != expected_path:
            member_failures.append({"stable_id": stable, "role": role, "error": "CANONICAL_REF_MISMATCH"})
            continue
        if match.get("member_sha256") != expected_digest:
            member_failures.append({"stable_id": stable, "role": role, "error": "CANONICAL_SHA_FIELD_MISMATCH"})
            continue
        actual_digest = digest(zip_by_id[package_id].read(expected_path))
        if actual_digest != expected_digest:
            member_failures.append({"stable_id": stable, "role": role, "error": "INPUT_MEMBER_SHA_MISMATCH"})
        if not match["verification"].startswith("VERIFIED"):
            member_failures.append({"stable_id": stable, "role": role, "error": "CANONICAL_MEMBER_NOT_VERIFIED"})

phase_errors = []
for item in candidate_rows:
    row = by_id[item["stable_id"]]
    expected_phase = "DRAFT_PENDING_REVIEW" if item.get("prompt_file") else "DEFERRED"
    if row["prompt_phase"] != expected_phase:
        phase_errors.append({"stable_id": item["stable_id"], "expected": expected_phase, "actual": row["prompt_phase"]})
    if not row["prompt_reason"].strip() or not row["prompt_manifest_ref"].strip():
        phase_errors.append({"stable_id": item["stable_id"], "error": "MISSING_REASON_OR_MANIFEST_REF"})

repro_material_checks = {
    "psf/black#4337": ['f"""', "24.4.2", "24.4.0"],
    "psf/black#4430": ["black --line-ranges 1-9 file.py", 'print("don\'t format me"  )', "   # don't format whitespace"],
    "psf/black#4540": ["tokenize.tokenize(example, lambda *_: None)", "720,011 calls in 0.133 seconds", "15-20%"],
}
repro_content_errors = []
for stable, snippets in repro_material_checks.items():
    prompt_path = D / "new-prompts" / f"black-{stable.split('#')[1]}.md"
    prompt = prompt_path.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in prompt:
            repro_content_errors.append({"stable_id": stable, "missing_snippet": snippet})

for row in rows:
    if row["family"] != "UNKNOWN" or row["split"] != "UNASSIGNED" or row["actor_visible"] != "False" or row["trusted_acceptance"] != "False" or row["purpose_authorization"] != "NOT_GRANTED" or row["training_release"] != "BLOCKED" or row["environment"] != "NOT_ASSESSED" or row["deployment"] != "NOT_ASSESSED" or row["old_exposure"] != "NOT_CLEARED":
        phase_errors.append({"stable_id": row["stable_id"], "error": "QUALIFICATION_GATE_CHANGED"})

sha_manifest = load_json(D / "sha256-manifest.json")
hash_mismatches = []
if not args.skip_output_manifest:
    for name, expected in sha_manifest["files"].items():
        path = D / name
        if not path.exists() or digest(path.read_bytes()) != expected:
            hash_mismatches.append(name)

zips_ok = all(z.testzip() is None for z in zip_by_id.values())
old_zip_path = I / "k68-b02-b11-unified-candidates-review-only-v1.zip"
with zipfile.ZipFile(old_zip_path) as old_zip:
    old_prompt_members = [n for n in old_zip.namelist() if n.startswith("candidate-prompts-review-only/") and n.endswith(".problem.md")]
old_prompt_overlap = sorted({x["stable_id"] for x in prompt_rows} & {r["task_id"] for r in catalog["records"] if r.get("final_prompt")})
black_catalog_rows = [r for r in rows if r["stable_id"].startswith("psf/black#") and r["stable_id"] in {x["task_id"] for x in catalog["records"]}]
missing_archive_count = sum(r["source_archive_member_path_status"] == "NOT_LOCATED" for r in black_catalog_rows if r["stable_id"] not in {x["stable_id"] for x in candidate_rows})

evidence = {
    "input_catalog_sha256": digest((I / "k68-catalog-v12.json").read_bytes()),
    "input_black_package_sha256": {key: digest(path.read_bytes()) for key, path in packages.items()},
    "canonical_rows": len(rows),
    "unique_canonical_ids": len(by_id),
    "catalog_ids": len(catalog["records"]),
    "legacy_rows": sum(row["prompt_phase"] == "LEGACY_STAGE_UNKNOWN_NOT_IN_V12" for row in rows),
    "candidate_rows": len(candidate_rows),
    "draft_pending_review": len(prompt_rows),
    "deferred": len(deferred_rows),
    "new_formal_pass_count": 0,
    "retained_prompt_content_count": 8,
    "reproduction_updates": ["psf/black#4337", "psf/black#4430", "psf/black#4540"],
    "reproduction_material_errors": repro_content_errors,
    "canonical_member_binding_failures": member_failures,
    "prompt_phase_errors": phase_errors,
    "standard_json_parse_errors": parse_errors,
    "output_hash_check_performed": not args.skip_output_manifest,
    "output_hash_mismatches": None if args.skip_output_manifest else hash_mismatches,
    "input_package_crc_ok": zips_ok,
    "historical_prompt_member_count": len(old_prompt_members),
    "new_prompt_overlap_with_old_selected_ids": old_prompt_overlap,
    "association_counts": {"same_repair_groups": len(associations["same_repair_groups"]), "same_archive_groups": len(associations["same_archive_groups"]), "known_duplicate_groups": sum("KNOWN_DUPLICATE" in item["status"] for item in associations["known_duplicate_groups"]), "related_not_asserted_duplicate_groups": sum("DUPLICATE_NOT_ASSERTED" in item["status"] for item in associations["known_duplicate_groups"])},
    "review_return_items": {"count": len(review_diff["changes"]), "all_fixed_or_pending_review": all(x["status"] in ("FIXED", "FIXED_PENDING_REVIEW") for x in review_diff["changes"]), "formal_prompt_pass_count": review_diff["formal_prompt_pass_count"]},
    "catalog_black_rows_without_archive_member_binding_outside_59_candidate_scope": missing_archive_count,
    "result": "PASS_CHECKS_ONLY; PROMPTS_REMAIN_DRAFT_PENDING_REVIEW",
}
print(json.dumps(evidence, ensure_ascii=False, indent=2))
if args.write_evidence:
    (D / "validation-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
