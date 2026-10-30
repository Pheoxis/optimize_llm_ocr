#!/usr/bin/env python3
"""Resume a partial ZIP job from storage/results/{job_id}/checkpoint.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.pipeline import resume_zip_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume ZIP OCR from checkpoint.json (skip completed members)",
    )
    parser.add_argument("job_id", help="Job UUID")
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable vision LLM escalation",
    )
    args = parser.parse_args()

    allow_vlm = None if not args.no_vlm else False
    try:
        summary = resume_zip_checkpoint(
            UUID(args.job_id),
            allow_vision_escalation=allow_vlm,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
