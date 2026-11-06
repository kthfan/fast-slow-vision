"""Re-render reports/<name>/report.md from the JSON files in that directory.

    python scripts/make_report.py reports/c100_bvit_s
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsv.report import write_report  # noqa: E402

if __name__ == "__main__":
    for d in sys.argv[1:] or sys.exit(__doc__):
        print(write_report(d))
