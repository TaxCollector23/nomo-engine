"""Command-line entry point for the bounded Nomo cycle emulator.

Examples::

    python -m nomo.emulation --artifact artifact.json --config config.json --out result.json
    python -m nomo.emulation --schema result

The command intentionally reports simulated cycle/cache evidence only.  It
does not contact a board and does not invoke SystemC, Verilator, or Gem5.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .core import EmulationError, run_artifact
from .schema import get_schema


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", help="stable nomo.emulation.artifact/1 JSON file")
    parser.add_argument("--config", help="optional nomo.emulation.config/1 JSON file")
    parser.add_argument("--out", help="write the result JSON to this path; default is stdout")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON when writing stdout/file")
    parser.add_argument("--schema", choices=("artifact", "config", "result"),
                        help="print the requested stable JSON schema and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.schema:
        print(json.dumps(get_schema(args.schema), indent=None if args.compact else 2, sort_keys=True))
        return 0
    if not args.artifact:
        parser.error("--artifact is required unless --schema is used")
    try:
        result = run_artifact(args.artifact, args.config)
    except EmulationError as exc:
        print(f"nomo emulation: error: {exc}", file=sys.stderr)
        return 2
    payload = result.to_json(indent=None if args.compact else 2) + "\n"
    if args.out:
        if args.out == "-":
            sys.stdout.write(payload)
        else:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload)
            print(f"wrote {out} (simulated cycles={result.summary['cycles']})")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
