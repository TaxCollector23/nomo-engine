"""Print the reproducible public serving comparison used by the audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nomo_planner.public_validation import run_sarathi_table4_replay


if __name__ == "__main__":
    print(json.dumps(run_sarathi_table4_replay().as_dict(), indent=2, sort_keys=True))
