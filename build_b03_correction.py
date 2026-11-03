"""Build an append-only correction sidecar from saved UTF-8 response bytes."""
from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
ATT = ROOT / "attachments"
OUT = ROOT / "deliverables" / "k71-b03-encoding-correction-v1"
OUT.mkdir(parents=True, exist_ok=True)

INPUT = ATT / "K40-psf-requests-source-b03-corrected-20261020.zip"
APPEND = ATT / "k40-psf-requests-b03-append-v1.zip"
REVIEW_LEDGER = ATT / "k71-b03-review-ledger.json"
RETURN_COMMENT = "01a159e5-f1eb-791d-ae63-041c4944c617"
RETURN_LEDGER_ATTACHMENT = "01a159e5-eed5-71e3-a97f-a934e6697583"
INPUT_ATTACHMENT = "01a159d9-d60a-775e-952f-6cee01266991"
APPEND_ATTACHMENT = "01a159e0-2a6c-716d-98d5-7a51e2f6994d"
CATALOG_ATTACHMENT = "01a159e0-3326-7954-bae5-d53451b7099b"
RETURN_IDS = [6628, 6601, 6591, 6586, 6575, 6572, 6551, 6533, 6532, 6487, 6481]
RAW_MEMBER = "search-page3-response.json"
RAW_SHA_EXPECTED = "3e177f1b318e37b8a88f8d3b0e4cfdae86b0cc2e03582872b92344cb7bdd9fd4"
INPUT_SHA_EXPECTED = "3de350bdc6d6ec461b8d89c46f5cb1d3d229819cf81b13b111614be8f0578eda"
APPEND_SHA_EXPECTED = "2edcb5a0ab299e2d16b488cf79ae66cacadbbaacba3dca13c730b414d410d80d"
ISSUE_PATH = "screened-identities/psf-requests-issue-{number}.txt"
CORRECTION_FILENAME = "k71-b03-encoding-correction-v1.zip"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jbytes(obj: object) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read_zip_member(z: zipfile.ZipFile, name: str) -> bytes:
    return z.read(name)


input_bytes = INPUT.read_bytes()
append_bytes = APPEND.read_bytes()
assert sha(input_bytes) == INPUT_SHA_EXPECTED
assert sha(append_bytes) == APPEND_SHA_EXPECTED
review_ledger_bytes = REVIEW_LEDGER.read_bytes()
review_ledger = json.loads(review_ledger_bytes.decode("utf-8"))

with zipfile.ZipFile(INPUT) as source_zip:
    raw_response = read_zip_member(source_zip, RAW_MEMBER)
    assert sha(raw_response) == RAW_SHA_EXPECTED
    response = json.loads(raw_response.decode("utf-8"))
    assert len(response["items"]) == 100
    bodies = {item["number"]: item["body"].encode("utf-8") for item in response["items"]}
    assert len(bodies) == 100 and all(isinstance(item["body"], str) for item in response["items"])
    old_bodies = {n: read_zip_member(source_zip, ISSUE_PATH.format(number=n)) for n in bodies}
    mismatches = {n for n in bodies if bodies[n] != old_bodies[n]}
    assert mismatches == set(RETURN_IDS), f"unexpected mismatch IDs: {sorted(mismatches)}"

with zipfile.ZipFile(APPEND) as append_zip:
    nested_input = read_zip_member(append_zip, "source-batches/K40-PSF-REQUESTS-B03-original.zip")
    assert sha(nested_input) == INPUT_SHA_EXPECTED and nested_input == input_bytes
    catalog_bytes = read_zip_member(append_zip, "catalog.json")
    append_diff_bytes = read_zip_member(append_zip, "append-diff.json")
    screening_ledger_bytes = read_zip_member(
        append_zip, "source-batches/K40-PSF-REQUESTS-B03-screening-ledger.json"
    )
    catalog = json.loads(catalog_bytes.decode("utf-8"))
    append_diff = json.loads(append_diff_bytes.decode("utf-8"))
    screening_ledger = json.loads(screening_ledger_bytes.decode("utf-8"))

catalog_sha_before = sha(catalog_bytes)
append_diff_sha_before = sha(append_diff_bytes)
screening_ledger_sha_before = sha(screening_ledger_bytes)
assert len(catalog["records"]) == 300
b03_rows = [r for r in catalog["records"] if r.get("batch") == "K40-PSF-REQUESTS-B03"]
assert len(b03_rows) == 100

correction_rows = []
for number in RETURN_IDS:
    old = old_bodies[number]
    new = bodies[number]
    member = f"body-corrections/requests-issue-{number}.txt"
    catalog_row = next(r for r in b03_rows if r["issue_number"] == number)
    old_member = catalog_row["identity"]["body_member"]
    correction_rows.append({
        "repo_issue": f"psf/requests#{number}",
        "issue_number": number,
        "correction_member": member,
        "new_sha256": sha(new),
        "new_bytes": len(new),
        "supersedes": {
            "body_member": old_member,
            "sha256": sha(old),
            "bytes": len(old),
        },
        "correction_source": {
            "input_archive_sha256": INPUT_SHA_EXPECTED,
            "raw_response_member": RAW_MEMBER,
            "raw_response_sha256": RAW_SHA_EXPECTED,
            "json_field": f"items[number={number}].body",
            "encoding": "UTF-8",
            "write_method": "decoded body string encoded directly as UTF-8; no newline normalization",
        },
        "original_append_archive_sha256": APPEND_SHA_EXPECTED,
    })

correction_ledger = {
    "schema": "k71-b03-body-encoding-correction/1",
    "status": "CORRECTED_PENDING_INDEPENDENT_REVIEW",
    "supersedes_review": {
        "issue_id": "01a159da-1e4f-70ad-b3d0-c159e700ac08",
        "return_comment_id": RETURN_COMMENT,
        "return_ledger_attachment_id": RETURN_LEDGER_ATTACHMENT,
        "return_ledger_sha256": sha(review_ledger_bytes),
        "return_ledger_verdict": review_ledger["verdict"],
    },
    "immutable_inputs": {
        "source_zip": {"attachment_id": INPUT_ATTACHMENT, "sha256": INPUT_SHA_EXPECTED, "bytes": len(input_bytes)},
        "original_append_zip": {"attachment_id": APPEND_ATTACHMENT, "sha256": APPEND_SHA_EXPECTED, "bytes": len(append_bytes)},
        "prior_catalog_attachment_id": CATALOG_ATTACHMENT,
        "prior_catalog_sha256": catalog_sha_before,
        "prior_screening_ledger_sha256": screening_ledger_sha_before,
        "saved_raw_response_member": RAW_MEMBER,
        "saved_raw_response_sha256": RAW_SHA_EXPECTED,
    },
    "correction_method": "Decode saved raw response as UTF-8 JSON; use each items[].body string and encode directly to UTF-8 bytes. No network fetch, reverse mojibake conversion, or newline conversion.",
    "scope": {
        "corrected_issue_numbers": RETURN_IDS,
        "total_body_files_checked": 100,
        "exact_after_sidecar_overlay": 100,
        "unchanged_exact": 89,
        "correction_count": 11,
        "metadata_identity_count_before_and_after": 300,
        "stage_counts_unchanged": True,
        "qualification_gates_unchanged": True,
    },
    "rows": correction_rows,
}

screening_ledger_v2 = copy.deepcopy(screening_ledger)
screening_ledger_v2["body_correction_revision"] = {
    "supersedes_screening_ledger_sha256": screening_ledger_sha_before,
    "correction_filename": CORRECTION_FILENAME,
    "raw_response_member": RAW_MEMBER,
    "raw_response_sha256": RAW_SHA_EXPECTED,
    "correction_count": 11,
}
for item in correction_rows:
    row = next(r for r in screening_ledger_v2["rows"] if r["issue_number"] == item["issue_number"])
    row["body_file"] = f"{CORRECTION_FILENAME}!/{item['correction_member']}"
    row["body_sha256"] = item["new_sha256"]
    row["body_bytes"] = item["new_bytes"]
    row["supersedes"] = {
        "body_file": item["supersedes"]["body_member"],
        "body_sha256": item["supersedes"]["sha256"],
        "body_bytes": item["supersedes"]["bytes"],
    }
screening_ledger_v2_bytes = jbytes(screening_ledger_v2)
screening_ledger_v2_sha = sha(screening_ledger_v2_bytes)

catalog_v2 = copy.deepcopy(catalog)
catalog_v2["body_correction_revision"] = {
    "filename": CORRECTION_FILENAME,
    "supersedes_catalog_sha256": catalog_sha_before,
    "input_archive_sha256": INPUT_SHA_EXPECTED,
    "original_append_archive_sha256": APPEND_SHA_EXPECTED,
    "raw_response_sha256": RAW_SHA_EXPECTED,
    "correction_count": 11,
    "review_status": "PENDING_INDEPENDENT_REVIEW",
    "screening_ledger_filename": "screening-ledger-v2.json",
    "screening_ledger_sha256": screening_ledger_v2_sha,
}
catalog_v2["source_package_map"]["K40-PSF-REQUESTS-B03"]["body_correction_revision"] = {
    "filename": CORRECTION_FILENAME,
    "supersedes_catalog_sha256": catalog_sha_before,
    "raw_response_member": RAW_MEMBER,
    "raw_response_sha256": RAW_SHA_EXPECTED,
    "correction_count": 11,
    "screening_ledger_filename": "screening-ledger-v2.json",
    "screening_ledger_sha256": screening_ledger_v2_sha,
}
for item in correction_rows:
    row = next(r for r in b03_rows if r["issue_number"] == item["issue_number"])
    identity = next(r for r in catalog_v2["records"] if r["batch"] == row["batch"] and r["issue_number"] == item["issue_number"])["identity"]
    identity["body_member"] = f"{CORRECTION_FILENAME}!/{item['correction_member']}"
    identity["issue_body_sha256"] = item["new_sha256"]
    identity["body_correction"] = {
        "supersedes_body_member": item["supersedes"]["body_member"],
        "supersedes_sha256": item["supersedes"]["sha256"],
        "supersedes_bytes": item["supersedes"]["bytes"],
        "correction_member": f"{CORRECTION_FILENAME}!/{item['correction_member']}",
        "source_archive_sha256": INPUT_SHA_EXPECTED,
        "raw_response_member": RAW_MEMBER,
        "raw_response_sha256": RAW_SHA_EXPECTED,
    }

catalog_v2_bytes = jbytes(catalog_v2)
catalog_v2_sha = sha(catalog_v2_bytes)
correction_ledger["catalog_v2_sha256"] = catalog_v2_sha

diff_v2 = {
    "schema": "k68-catalog-append-diff/2",
    "kind": "append-only body encoding correction sidecar; no identity or qualification changes",
    "baseline": {
        "catalog_attachment_id": CATALOG_ATTACHMENT,
        "catalog_sha256": catalog_sha_before,
        "append_archive_attachment_id": APPEND_ATTACHMENT,
        "append_archive_sha256": APPEND_SHA_EXPECTED,
        "source_archive_attachment_id": INPUT_ATTACHMENT,
        "source_archive_sha256": INPUT_SHA_EXPECTED,
    },
    "correction": {
        "catalog_v2_sha256": catalog_v2_sha,
        "screening_ledger_v2_sha256": screening_ledger_v2_sha,
        "correction_count": 11,
        "body_files_checked": 100,
        "body_files_exact_after_overlay": 100,
        "unchanged_body_count": 89,
        "changed_issue_numbers": RETURN_IDS,
        "rows": correction_rows,
    },
    "invariants": {
        "identity_total": catalog["counts"]["identity_total"],
        "selected_review_only_prompts": catalog["counts"]["selected_review_only_prompts"],
        "deferred_identities": catalog["counts"]["deferred_identities"],
        "source_ready_inventory": catalog["counts"]["source_ready_inventory"],
        "deployable": catalog["counts"]["deployable"],
        "training_approved": catalog["counts"]["training_approved"],
        "b03_batch_counts_unchanged": True,
        "prior_200_identity_records_unchanged": True,
        "prior_66_prompt_records_unchanged": True,
        "source_role": "UNKNOWN",
        "split": "UNASSIGNED",
        "actor_visible": False,
        "trusted_acceptance": False,
        "training_release": "BLOCKED",
        "deployment_environment": "NOT_ASSESSED",
    },
    "package_members_added": [r["correction_member"] for r in correction_rows],
    "original_input_and_append_archives": "Preserved byte-for-byte as their existing K70 attachments; this package is a sidecar and does not rewrite either archive.",
}

audit_rows = []
for number in sorted(bodies):
    actual = bodies[number] if number in mismatches else old_bodies[number]
    audit_rows.append({
        "issue_number": number,
        "body_member": (
            f"{CORRECTION_FILENAME}!/body-corrections/requests-issue-{number}.txt"
            if number in mismatches
            else f"source-batches/K40-PSF-REQUESTS-B03-original.zip!/{ISSUE_PATH.format(number=number)}"
        ),
        "raw_response_utf8_bytes": len(bodies[number]),
        "raw_response_utf8_sha256": sha(bodies[number]),
        "delivered_overlay_bytes": len(actual),
        "delivered_overlay_sha256": sha(actual),
        "exact_byte_match": actual == bodies[number],
    })
assert all(r["exact_byte_match"] for r in audit_rows)
body_audit = {
    "schema": "k71-b03-100-body-byte-audit/1",
    "source_archive_sha256": INPUT_SHA_EXPECTED,
    "raw_response_member": RAW_MEMBER,
    "raw_response_sha256": RAW_SHA_EXPECTED,
    "checked_count": len(audit_rows),
    "exact_count": sum(r["exact_byte_match"] for r in audit_rows),
    "corrected_count": len(RETURN_IDS),
    "unchanged_count": 89,
    "rows": audit_rows,
}

files: dict[str, bytes] = {
    "catalog-v2.json": catalog_v2_bytes,
    "append-diff-v2.json": jbytes(diff_v2),
    "correction-ledger.json": jbytes(correction_ledger),
    "screening-ledger-v2.json": screening_ledger_v2_bytes,
    "body-byte-audit.json": jbytes(body_audit),
    "README.md": (
        "# K71 B03 body encoding correction\n\n"
        "This is an additive correction sidecar for the 11 RETURNed B03 issue bodies. "
        "It does not rewrite the original input ZIP or the original append ZIP; both are "
        "bound by their original SHA-256 values in `correction-ledger.json`.\n\n"
        "Each corrected body is copied from `items[].body` in the saved UTF-8 "
        "`search-page3-response.json`, then encoded directly as UTF-8 without newline "
        "normalization. `body-byte-audit.json` records the one-time comparison of all "
        "100 body members against that same response.\n\n"
        "The corrected catalog changes only the 11 B03 body pointers and hashes and adds "
        "supersession metadata. Identity totals, dispositions, source gates, prompt records, "
        "and stage counts are unchanged. Independent review is pending.\n"
    ).encode("utf-8"),
}
for item in correction_rows:
    files[item["correction_member"]] = bodies[item["issue_number"]]

checksums = "".join(f"{sha(data)}  {name}\n" for name, data in sorted(files.items())).encode("ascii")
files["SHA256SUMS.txt"] = checksums

for name, data in files.items():
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

for filename, data in {
    "k71-b03-catalog-v2.json": catalog_v2_bytes,
    "k71-b03-append-diff-v2.json": files["append-diff-v2.json"],
    "k71-b03-body-byte-audit.json": files["body-byte-audit.json"],
    "k71-b03-correction-ledger.json": files["correction-ledger.json"],
}.items():
    (ROOT / "deliverables" / filename).write_bytes(data)

zip_path = ROOT / "deliverables" / CORRECTION_FILENAME
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as outzip:
    for name, data in sorted(files.items()):
        info = zipfile.ZipInfo(name, date_time=(2026, 11, 3, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        outzip.writestr(info, data)
zip_sha = sha(zip_path.read_bytes())
zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(f"{zip_sha}  {CORRECTION_FILENAME}\n", encoding="ascii", newline="")

# Read the actual output ZIP back and verify checksums plus all 100 body bytes.
with zipfile.ZipFile(zip_path) as checkzip:
    assert set(checkzip.namelist()) == set(files)
    for name, data in files.items():
        assert checkzip.read(name) == data
    parsed_cat = json.loads(checkzip.read("catalog-v2.json").decode("utf-8"))
    assert len(parsed_cat["records"]) == 300
    assert parsed_cat["counts"] == catalog["counts"]
    for audit in audit_rows:
        if audit["issue_number"] in RETURN_IDS:
            actual = checkzip.read(f"body-corrections/requests-issue-{audit['issue_number']}.txt")
        else:
            actual = old_bodies[audit["issue_number"]]
        assert actual == bodies[audit["issue_number"]]

print(json.dumps({
    "input_sha256": INPUT_SHA_EXPECTED,
    "original_append_sha256": APPEND_SHA_EXPECTED,
    "prior_catalog_sha256": catalog_sha_before,
    "prior_append_diff_sha256": append_diff_sha_before,
    "prior_screening_ledger_sha256": screening_ledger_sha_before,
    "corrected_catalog_sha256": catalog_v2_sha,
    "correction_zip_sha256": zip_sha,
    "correction_zip_bytes": zip_path.stat().st_size,
    "body_files_checked": len(audit_rows),
    "exact_body_matches": sum(r["exact_byte_match"] for r in audit_rows),
    "corrected_issue_numbers": RETURN_IDS,
    "review_ledger_sha256": sha(review_ledger_bytes),
}, indent=2))
