#!/usr/bin/env python3
"""Smoke test A2: ZIP extract, sync OCR, optional Redis+worker path."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL {msg}", file=sys.stderr)
    sys.exit(1)


def test_zip_extract() -> None:
    from app.services.ingest import extract_zip

    tmp = Path(tempfile.mkdtemp())
    zpath = tmp / "t.zip"
    out = tmp / "out"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a/test.pdf", b"%PDF-1.4")
    files = extract_zip(zpath, out)
    if len(files) != 1:
        _fail(f"extract_zip expected 1 file, got {len(files)}")
    _ok("extract_zip")


def test_sync_run_job(sample: Path) -> None:
    from app.core.config import settings
    from app.core.constants import JobStatus
    from app.services.jobs import job_store
    from app.services.pipeline import run_job

    os.environ["USE_BACKGROUND_WORKER"] = "false"
    settings.ensure_dirs()
    job = job_store.create()
    dest = settings.uploads_dir / f"{job.job_id}_{sample.name}"
    dest.write_bytes(sample.read_bytes())
    t0 = time.time()
    run_job(job.job_id, dest, source_name=sample.name)
    rec = job_store.get(job.job_id)
    if rec is None or rec.status != JobStatus.DONE:
        _fail(f"sync run_job status={getattr(rec, 'status', None)} err={getattr(rec, 'error', None)}")
    from app.services.result_paths import get_result_dir

    meta = get_result_dir(job.job_id) / "job.json"
    if not meta.is_file():
        _fail("job.json missing after done")
    _ok(f"sync run_job [{sample.name}] {time.time() - t0:.1f}s")


def test_async_enqueue_and_worker(sample: Path) -> None:
    from app.core.config import settings
    from app.core.constants import JobStatus
    from app.services.jobs import job_store
    from app.services.queue import enqueue_document_job, ping_redis
    from app.workers.tasks import process_document_task

    if not ping_redis():
        print("  SKIP async (Redis not running)")
        return

    os.environ["USE_BACKGROUND_WORKER"] = "true"
    os.environ["DOCOCR_WORKER"] = "1"
    settings.ensure_dirs()
    job = job_store.create()
    dest = settings.uploads_dir / f"{job.job_id}_{sample.name}"
    dest.write_bytes(sample.read_bytes())
    enqueue_document_job(job.job_id, dest, sample.name)
    rec = job_store.get(job.job_id)
    if rec is None or rec.status != JobStatus.QUEUED:
        _fail("job not queued in Redis")

    t0 = time.time()
    process_document_task(str(job.job_id), str(dest), sample.name)
    rec = job_store.get(job.job_id)
    if rec is None or rec.status != JobStatus.DONE:
        _fail(f"async worker status={getattr(rec, 'status', None)} err={getattr(rec, 'error', None)}")
    _ok(f"async Redis+worker [{sample.name}] {time.time() - t0:.1f}s")


def main() -> None:
    print("A2 smoke tests")
    test_zip_extract()

    samples = ROOT / "samples"
    jpg = next(samples.glob("*.jpg"), None) if samples.is_dir() else None
    pdf = next(samples.glob("*.pdf"), None) if samples.is_dir() else None

    if jpg:
        test_sync_run_job(jpg)
        test_async_enqueue_and_worker(jpg)
    else:
        print("  SKIP OCR samples (no samples/*.jpg)")

    if pdf:
        test_sync_run_job(pdf)
    else:
        print("  SKIP PDF sample")

    print("All A2 smoke checks passed.")


if __name__ == "__main__":
    main()
