"""Run the whole D0 pipeline in order, from pinned checkouts to report.

Order matters: the licence ledger produces source-lock.json (which now carries
the machine-readable approval field), the miner produces the family inventory
for every role, the ledger builder derives each family's oracle material, the
extras builder derives the time-isolation / shortfall / manifest split, and the
gate re-asserts everything from the emitted JSON before the report is written
from those same files.

Usage:  python d0/run_all.py
Exit code is non-zero if the gate fails, so this can be the single CI entry
point for the D0 deliverable.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    "collect_licenses.py",   # source-lock.json + licence_review + per-file ledger
    "mine_families.py",      # family-candidates.json for every role
    "build_ledger.py",       # family-ledger.json (oracle material + time evidence)
    "build_extras.py",       # time isolation, shortfall, public/restricted split
]


def run(script, capture=None):
    print("=== %s ===" % script)
    if capture:
        p = subprocess.run([sys.executable, os.path.join(HERE, script)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with open(capture, "wb") as f:
            f.write(p.stdout)
        tail = p.stdout.decode("utf-8", "replace").strip().splitlines()
        for line in tail[-3:]:
            print("   " + line)
        return p.returncode
    p = subprocess.run([sys.executable, os.path.join(HERE, script)])
    return p.returncode


def main():
    for s in STEPS:
        rc = run(s)
        if rc != 0:
            print("pipeline aborted: %s exit=%d" % (s, rc))
            return rc
    # the gate writes its own transcript (UTF-8) so the report can quote it
    rc = run("validate_d0.py", capture=os.path.join(HERE, "out",
                                                    "validate_d0.output.txt"))
    if rc != 0:
        print("gate failed with exit=%d; report not regenerated" % rc)
        return rc
    return run("make_report.py")


if __name__ == "__main__":
    sys.exit(main())
