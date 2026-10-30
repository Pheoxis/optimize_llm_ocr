#!/usr/bin/env python3
"""Process a local file through the pipeline (dev helper)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services.document_language import (  # noqa: E402
    DocumentLanguage,
    parse_document_language,
)
from app.services.jobs import job_store  # noqa: E402
from app.services.pipeline import run_job  # noqa: E402
from app.services.result_paths import get_result_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR pipeline on a local file")
    parser.add_argument("path", type=Path, help="Path to PDF or image")
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable vision LLM escalation (RapidOCR + Tesseract only)",
    )
    parser.add_argument(
        "--lang",
        choices=[lang.value for lang in DocumentLanguage],
        default=DocumentLanguage.AUTO.value,
        help="Document language hint: auto (default), pl, en",
    )
    parser.add_argument("--polski", action="store_true", help="Shortcut for --lang pl")
    parser.add_argument("--angielski", action="store_true", help="Shortcut for --lang en")
    args = parser.parse_args()
    if args.polski and args.angielski:
        raise SystemExit("Use only one of: --polski or --angielski")
    if args.polski:
        document_language = DocumentLanguage.PL
    elif args.angielski:
        document_language = DocumentLanguage.EN
    else:
        document_language = parse_document_language(args.lang)

    path = args.path.resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    settings.ensure_dirs()
    job = job_store.create()

    mode = "no-vlm" if args.no_vlm else "smart"
    print(f"Job ID: {job.job_id}")
    print(
        f"Processing {path.name} ({mode}, engine={settings.ocr_engine}, "
        f"lang={document_language.value}) …"
    )
    run_job(
        job.job_id,
        path,
        source_name=path.name,
        allow_vision_escalation=False if args.no_vlm else None,
        document_language=document_language.value,
    )
    record = job_store.get(job.job_id)
    if record is None:
        raise SystemExit("Job missing")

    result_dir = get_result_dir(job.job_id)
    print(f"Results: {result_dir}")

    print(f"Status: {record.status}")
    if record.error:
        print(f"Error: {record.error}")
        raise SystemExit(1)

    job_json = result_dir / "job.json"
    if job_json.is_file():
        import json
        meta = json.loads(job_json.read_text(encoding="utf-8"))
        print(f"\nDuration: {meta.get('duration_ms', '?')} ms")
        if meta.get("result_files"):
            print(f"Result files: {len(meta['result_files'])}")

    root_txt = result_dir / "result.txt"
    if root_txt.is_file():
        txt = root_txt.read_text(encoding="utf-8")
        print(f"\n--- result.txt ({len(txt)} chars) ---\n")
        print(txt[:3000] + ("…" if len(txt) > 3000 else ""))
    else:
        sub_results = sorted(result_dir.rglob("result.txt"))
        print(f"\nZIP: {len(sub_results)} member result(s)")
        for rp in sub_results[:5]:
            rel = rp.relative_to(result_dir)
            print(f"  {rel} ({rp.stat().st_size} B)")
        if len(sub_results) > 5:
            print(f"  … +{len(sub_results) - 5} more")

    print(f"\nFull output: {result_dir}")


if __name__ == "__main__":
    main()
