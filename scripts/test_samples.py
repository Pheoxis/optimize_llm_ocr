#!/usr/bin/env python3
"""Run OCR pipeline on every file in samples/ and print a summary table."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services.document_language import (  # noqa: E402
    DocumentLanguage,
    parse_document_language,
)
from app.services.fast_ocr import warmup as warmup_fast  # noqa: E402
from app.services.jobs import job_store  # noqa: E402
from app.services.pipeline import run_job  # noqa: E402
from app.services.result_paths import get_result_dir  # noqa: E402
from app.services.vision_llm import warmup as warmup_vision  # noqa: E402


def _page_summary(meta: dict) -> str:
    pages = meta.get("pages") or []
    if not pages:
        return "-"
    parts = []
    for p in pages:
        eng = p.get("engine", "?")
        t = p.get("elapsed_sec", "?")
        esc = p.get("escalation_reason")
        if esc:
            parts.append(f"p{p['page']}:{eng}({t}s,esc)")
        else:
            parts.append(f"p{p['page']}:{eng}({t}s)")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR pipeline on files from samples/")
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

    samples_dir = ROOT / "samples"
    files = sorted(
        p for p in samples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in settings.allowed_extensions
    )
    if not files:
        raise SystemExit(f"No sample files in {samples_dir}")

    settings.ensure_dirs()
    from app.services.fast_ocr import cuda_runtime_status
    from app.services.gpu import get_bulk_ocr_gpu_ids, invalidate_gpu_cache

    invalidate_gpu_cache()
    cuda = cuda_runtime_status()
    bulk = get_bulk_ocr_gpu_ids()
    print(f"OCR_ENGINE={settings.ocr_engine!r}")
    print(f"Document language hint: {document_language.value!r}")
    print(f"SMART gates: mean>={settings.smart_min_mean_conf}, "
          f"low_ratio<={settings.smart_min_low_conf_ratio}, "
          f"worst>={settings.smart_min_worst_score}")
    print(f"GPU bulk pool: {bulk}  (OCR_BULK_GPU_IDS={settings.ocr_bulk_gpu_ids!r})")
    print(
        f"CUDA: bootstrap={cuda.get('bootstrap_ok')} ort_device={cuda.get('ort_device')} "
        f"rapidocr={cuda.get('rapidocr_provider')} nvidia_count={cuda.get('nvidia_gpus')}"
    )
    if cuda.get("rapidocr_provider") != "CUDAExecutionProvider":
        print(
            "WARNING: RapidOCR not on CUDA — tests will be slow. "
            "Run outside sandbox / ensure nvidia driver + .venv nvidia libs."
        )
    print(f"\nWarming up engines…")
    warmup_fast(gpu_ids=bulk if len(bulk) > 1 else None)
    if settings.ocr_engine in ("smart", "vision", "auto"):
        warmup_vision()
    print()

    rows: list[dict] = []
    for path in files:
        print(f"{'─' * 72}\n▶ {path.name}")
        job = job_store.create()
        t0 = time.time()
        run_job(
            job.job_id,
            path,
            source_name=path.name,
            document_language=document_language.value,
        )
        wall = time.time() - t0
        record = job_store.get(job.job_id)
        if record is None or record.status != "done":
            err = record.error if record else "missing job"
            print(f"  FAILED: {err}\n")
            rows.append({"file": path.name, "status": "error", "error": err})
            continue

        out_dir = get_result_dir(job.job_id)
        result_json = out_dir / "result.json"
        data = json.loads(result_json.read_text(encoding="utf-8"))
        txt_path = out_dir / "result.txt"
        txt = txt_path.read_text(encoding="utf-8")
        meta = data.get("metadata") or {}
        pages = meta.get("pages") or []

        ocr_time = sum(p.get("elapsed_sec", 0) for p in pages if isinstance(p.get("elapsed_sec"), (int, float)))

        row = {
            "file": path.name,
            "status": "ok",
            "engine": data.get("engine", "?"),
            "pages": data.get("page_count", 1),
            "chars": len(txt),
            "lines": txt.count("\n") + 1 if txt else 0,
            "wall_sec": round(wall, 2),
            "ocr_sec": round(ocr_time, 2),
            "confidence": data.get("confidence"),
            "pages_detail": _page_summary(meta),
            "job_id": str(job.job_id),
        }
        rows.append(row)

        print(f"  engine:     {row['engine']}")
        print(f"  pages:      {row['pages']}")
        print(f"  chars:      {row['chars']}  lines: {row['lines']}")
        print(f"  wall time:  {row['wall_sec']}s  (OCR sum: {row['ocr_sec']}s)")
        if row["confidence"] is not None:
            print(f"  confidence: {row['confidence']}")
        print(f"  per-page:   {row['pages_detail']}")
        print(f"  output:     {txt_path.parent}")
        print(f"\n  --- preview (first 25 lines) ---")
        for line in txt.splitlines()[:25]:
            print(f"  {line}")
        if txt.count("\n") >= 25:
            print(f"  … (+{txt.count(chr(10)) - 24} more lines)")
        print()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'File':<40} {'Engine':<22} {'Pages':>5} {'Chars':>7} {'Wall':>7} {'OCR':>7}")
    print("-" * 72)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['file']:<40} ERROR: {r.get('error', '?')}")
            continue
        print(
            f"{r['file']:<40} {r['engine']:<22} {r['pages']:>5} "
            f"{r['chars']:>7} {r['wall_sec']:>6.1f}s {r['ocr_sec']:>6.1f}s"
        )
    print()


if __name__ == "__main__":
    main()
