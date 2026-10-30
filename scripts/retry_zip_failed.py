#!/usr/bin/env python3
"""Retry ZIP members listed in storage/results/{job_id}/errors.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services.pipeline import retry_zip_failed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run OCR for failed members from errors.json",
    )
    parser.add_argument("job_id", help="Job UUID (same as results folder name)")
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable vision LLM escalation on retry",
    )
    args = parser.parse_args()

    try:
        job_uuid = UUID(args.job_id)
    except ValueError as exc:
        raise SystemExit(f"Invalid job_id: {args.job_id}") from exc

    settings.ensure_dirs()
    try:
        summary = retry_zip_failed(
            job_uuid,
            allow_vision_escalation=False if args.no_vlm else None,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(summary, indent=2))
    if summary.get("still_failed", 0) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
