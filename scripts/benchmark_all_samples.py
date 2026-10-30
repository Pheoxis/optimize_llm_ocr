#!/usr/bin/env python3
"""Benchmark OCR on every .jpg/.png/.pdf under samples/ (recursive)."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("USE_BACKGROUND_WORKER", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services.fast_ocr import cuda_runtime_status, warmup  # noqa: E402
from app.services.gpu import get_bulk_ocr_gpu_ids, invalidate_gpu_cache  # noqa: E402
from app.services.jobs import job_store  # noqa: E402
from app.services.pipeline import run_job  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".pdf"}


def main() -> int:
    samples = ROOT / "samples"
    files = sorted(
        p for p in samples.rglob("*")
        if p.is_file() and p.suffix.lower() in EXT
    )
    # Images first (fast feedback), then PDFs alphabetically.
    def _sort_key(p: Path) -> tuple:
        ext = p.suffix.lower()
        if ext in (".jpg", ".jpeg", ".png"):
            return (0, p.name.lower())
        return (1, str(p.relative_to(samples)).lower())

    files = sorted(files, key=_sort_key)
    if not files:
        print("No files found", file=sys.stderr)
        return 1

    out_dir = settings.results_dir / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "benchmark_all_samples.json"
    log_path = out_dir / "benchmark_all_samples.log"

    settings.ensure_dirs()
    invalidate_gpu_cache()
    cuda = cuda_runtime_status()
    bulk = get_bulk_ocr_gpu_ids()

    with log_path.open("w", encoding="utf-8") as log:
        def log_line(msg: str) -> None:
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

        log_line(f"Files: {len(files)}  CUDA: {cuda.get('rapidocr_provider')}  GPUs: {bulk}")
        log_line("Warming up…")
        warmup(gpu_ids=bulk if len(bulk) > 1 else None)

        results: list[dict] = []
        total_t0 = time.perf_counter()

        for i, path in enumerate(files, 1):
            rel = str(path.relative_to(samples))
            job = job_store.create()
            t0 = time.perf_counter()
            status = "error"
            err: str | None = None
            try:
                run_job(job.job_id, path, source_name=path.name)
                rec = job_store.get(job.job_id)
                status = rec.status if rec else "missing"
                if rec and rec.status != "done":
                    err = rec.error
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            wall = round(time.perf_counter() - t0, 2)
            ext = path.suffix.lower()
            if ext == ".jpeg":
                ext = ".jpg"
            row = {
                "file": rel,
                "ext": ext,
                "wall_sec": wall,
                "status": status,
                "error": err,
            }
            results.append(row)
            log_line(f"[{i}/{len(files)}] {wall:7.1f}s  {status:<8}  {rel}")

            # Incremental save so a long run can be monitored.
            if i % 5 == 0 or i == len(files):
                partial = {
                    "progress": f"{i}/{len(files)}",
                    "elapsed_sec": round(time.perf_counter() - total_t0, 2),
                    "results": results,
                }
                report_path.write_text(
                    json.dumps(partial, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        total_sec = round(time.perf_counter() - total_t0, 2)

        by_ext: dict[str, list[float]] = defaultdict(list)
        ok = fail = 0
        for r in results:
            if r["status"] == "done":
                ok += 1
            else:
                fail += 1
            by_ext[r["ext"]].append(r["wall_sec"])

        summary = {
            "total_files": len(files),
            "ok": ok,
            "failed": fail,
            "total_wall_sec": total_sec,
            "total_wall_min": round(total_sec / 60, 2),
            "cuda": cuda,
            "by_extension": {
                ext: {
                    "count": len(times),
                    "total_sec": round(sum(times), 2),
                    "mean_sec": round(sum(times) / len(times), 2) if times else 0,
                    "max_sec": round(max(times), 2) if times else 0,
                }
                for ext, times in sorted(by_ext.items())
            },
            "results": results,
        }
        report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        log_line("")
        log_line("=" * 72)
        log_line(f"TOTAL: {len(files)} files in {total_sec:.1f}s ({total_sec/60:.1f} min)")
        log_line(f"OK: {ok}  FAILED: {fail}")
        for ext, stats in summary["by_extension"].items():
            log_line(
                f"  {ext}: n={stats['count']}  sum={stats['total_sec']}s  "
                f"mean={stats['mean_sec']}s  max={stats['max_sec']}s"
            )
        log_line(f"Report: {report_path}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
