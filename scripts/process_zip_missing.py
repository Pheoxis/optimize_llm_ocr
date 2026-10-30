#!/usr/bin/env python3
"""Recover partially processed ZIP jobs by re-OCRing members without results on disk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services.ingest import IngestError  # noqa: E402
from app.services.zip_missing import process_zip_missing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ZIP archive to result.json/result.txt/checkpoint.json "
            "and process only missing members"
        ),
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="Job UUID (storage/results/{job_id}/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze and write missing_report.json; do not run OCR",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable vision LLM escalation",
    )
    args = parser.parse_args()

    try:
        job_uuid = UUID(args.job_id)
    except ValueError as exc:
        print(f"Invalid --job-id: {args.job_id}", file=sys.stderr)
        return 1

    settings.ensure_dirs()
    allow_vlm = None if not args.no_vlm else False

    try:
        summary = process_zip_missing(
            job_uuid,
            dry_run=args.dry_run,
            allow_vision_escalation=allow_vlm,
        )
    except (FileNotFoundError, IngestError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if summary.get("still_missing", 0) > 0 and not args.dry_run:
        return 1
    if args.dry_run and summary.get("missing_count", 0) > 0:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
