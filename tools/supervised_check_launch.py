"""Launch the supervised engineering check (thin wrapper around the adapter).

Exists so the *same* code path the tests exercise can be launched from a shell:
`python tools/supervised_check_launch.py --v6-dir tools/v6 --dest-dir ... --approved-budget-seconds ...`
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from v3.train import supervised_check as sc  # noqa: E402

if __name__ == "__main__":
    sys.exit(sc.main(sys.argv[1:]))
