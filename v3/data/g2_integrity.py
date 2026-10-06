"""Pinned, source-controlled inputs for the narrowly scoped G2 fixture runner.

This module contains only public synthetic fixture baselines. It does not contain
or authorize real candidate/denylist data. Real mode requires a trusted provider
that is intentionally not configured in this source-only runtime.
"""

from __future__ import annotations

BASELINE_RELATIVE_PATH = "tests/fixtures/g2/integrity-baseline.json"
BASELINE_FILE_BYTES_SHA256 = "9df710d8edb918c5a1248a74864a428166706cfff23fd43eb043cf0829cc6bf7"
BASELINE_ARTIFACT_ID = "k24-g2-synthetic-input-baseline-1"
PROTOCOL_ARTIFACT_ID = "01a11242-bdc5-78ff-a522-4464369cd80f"
PROTOCOL_ZIP_SHA256 = "79d329f5dd50deb4d799d90659061b2717064cd285ed9c2180f90d9cb12ead72"
PROTOCOL_SCHEMA_VERSION = "v3-v2-exclusion-proof-protocol-4"
EXECUTION_MANIFEST_SCHEMA_VERSION = "v3-v2-exclusion-proof-execution-manifest-3"

# Filled only after the runtime closure stabilizes. A new implementation revision
# must intentionally refresh this map and its review evidence.
EXPECTED_EXECUTION_SOURCE_LF_SHA256: dict[str, str] = {
    'v3/__init__.py': 'f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5',
    'v3/cli.py': '5fc00bec3ee42ae34109ce14f8aa31f702222227a467a6813a285b6325b6c2b6',
    'v3/common/__init__.py': 'f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5',
    'v3/common/canonical.py': '75f5784190d25354bd02622aacde8761d990ee48434b7459c95b7054c0867e2e',
    'v3/common/errors.py': 'd8f907e528c5cd6e3b9807b6c061925a1852a34a106ac2d0c18e04fa657733d2',
    'v3/data/__init__.py': 'f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5',
    'v3/data/dedup.py': '2bae39b84c7ce7e78153aefcc56d6986a5e78de0e33f543280ffeefa4c0ae0c3',
    'v3/data/exporter.py': '5e6b709cc2a4b5e6d1d0a413d4a88dea1b2c741823b30d56768d749e4789d9a5',
    'v3/data/faces.py': '242a2fe06d0015cec3021c5ffe78e7c7e0707457f6a220c34b63cfcda05b14b1',
    'v3/data/oracle.py': 'be3cca98db68e5731f570bb720c4fd727ac15aadbe3bd89f4a98dfceca0c7945',
    'v3/data/source_lock.py': '809b9d315283db4a9a3d5710d0d2408d556adc31156f30c57e9629c0594fd3cb',
    'v3/exp/__init__.py': 'f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5',
    'v3/exp/exp1.py': 'db9fed9c8eb9a52a5aadcc4935e5a95b2122828aa58ed0d4bfd1205fdf6dcb02',
    'v3/search/__init__.py': 'f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5',
    'v3/search/lexsearch.py': '431af78e00d58e3790a2b1e368a1c642949d8862966d804f32395d92aa112765',
    'v3/t0/__init__.py': 'f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5',
    'v3/t0/deps.py': 'c4b12d9f6d3a8adcc5b277f4402235fdd83115e2a645f7eaac229646bc720525',
    'v3/train/__init__.py': 'f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5',
    'v3/train/template.py': 'fdd2fc2d7e00e094bd76e60e59dcd5f3ca1045671b6543aa4a61d8f5a2ecde80',
}
REQUIRED_EXECUTION_PATHS = (
    "v3/__init__.py",
    "v3/cli.py",
    "v3/common/__init__.py",
    "v3/common/canonical.py",
    "v3/common/errors.py",
    "v3/data/__init__.py",
    "v3/data/dedup.py",
    "v3/data/exporter.py",
    "v3/data/faces.py",
    "v3/data/g2_integrity.py",
    "v3/data/oracle.py",
    "v3/data/source_lock.py",
    "v3/exp/__init__.py",
    "v3/exp/exp1.py",
    "v3/search/__init__.py",
    "v3/search/lexsearch.py",
    "v3/t0/__init__.py",
    "v3/t0/deps.py",
    "v3/train/__init__.py",
    "v3/train/template.py",
)
