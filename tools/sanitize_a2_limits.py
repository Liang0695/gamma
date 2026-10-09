"""Create a byte-distinct, path-sanitized derivative of the frozen A2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

RULE_VERSION = "k24-a2-path-sanitizer-1"
_ABS_PATH = re.compile(r"[A-Za-z]:[\\/](?:[^\s\"'<>|]+[\\/]?)+")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _walk(value, pointer: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, pointer + "/" + str(index))
    elif isinstance(value, str):
        yield pointer or "/", value


def sanitize(payload: object) -> tuple[object, list[str]]:
    changes: list[str] = []

    def visit(value, pointer: str = ""):
        if isinstance(value, dict):
            return {key: visit(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child, pointer + "/" + str(index)) for index, child in enumerate(value)]
        if isinstance(value, str):
            result, count = _ABS_PATH.subn("<LOCAL_PATH_REDACTED>", value)
            if count:
                changes.append(pointer or "/")
            return result
        return value

    return visit(payload), changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    source = Path(args.input)
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    derived, fields = sanitize(payload)
    if len(fields) != 2:
        raise SystemExit("expected exactly two path-bearing JSON string fields; refusing unexpected rewrite")
    derived_text = json.dumps(derived, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    derived_bytes = derived_text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
    if _ABS_PATH.search(derived_bytes.decode("utf-8")):
        raise SystemExit("derived evidence still contains an absolute path")

    output = Path(args.output)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(derived_bytes)
    manifest = {
        "schema_version": "k24-a2-limits-derived-manifest-1",
        "derivation_rule_version": RULE_VERSION,
        "source_filename": source.name,
        "source_file_bytes_sha256": _sha256(raw),
        "derived_filename": output.name,
        "derived_file_bytes_sha256": _sha256(derived_bytes),
        "transformation": "replace absolute path substrings in exactly two string fields with <LOCAL_PATH_REDACTED>; preserve all other parsed values",
        "changed_json_pointers": sorted(fields),
        "changed_field_count": len(fields),
        "source_modified": False,
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest_bytes = manifest_text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    print(json.dumps({"status": "derived", "changed_field_count": len(fields), "source_sha256": manifest["source_file_bytes_sha256"], "derived_sha256": manifest["derived_file_bytes_sha256"], "manifest_sha256": _sha256(manifest_bytes)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
