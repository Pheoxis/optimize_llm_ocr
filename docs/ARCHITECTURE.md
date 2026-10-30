# Architektura modułów

Katalog `app/` — FastAPI + warstwa usług OCR.

```text
app/
├── main.py                 # FastAPI, lifespan warmup (RapidOCR, Ollama)
├── api/routes.py           # REST /api/v1/jobs
├── core/
│   ├── config.py           # Settings (.env), ensure_dirs()
│   └── constants.py        # JobStatus, SourceKind
├── models/schemas.py       # Pydantic: ProcessingResult, JobStatusResponse, OcrLine
├── workers/tasks.py        # RQ: process_document_task → run_job
└── services/               # logika OCR (poniżej)
```

---

## Warstwa API i jobów

| Moduł | Opis |
|--------|------|
| `jobs.py` | `InMemoryJobStore` / `RedisJobStore`, pole `results_folder` |
| `job_metadata.py` | Zapis `job.json` po statusie terminalnym |
| `queue.py` | Enqueue RQ `process_document_task` |
| `result_paths.py` | `allocate_result_dir`, `get_result_dir` — foldery po nazwie pliku |
| `ingest.py` | Walidacja rozmiaru/rozszerzenia, lista ZIP, `SourceKind` |

---

## Orchestracja

| Moduł | Opis |
|--------|------|
| **`pipeline.py`** | `run_job`, `_process_single_image`, `_process_bulk_document`, `_run_zip_job`, retry, ZIP errors |
| `pipeline_profile.py` | Profile OCR (PDF crop vs line) |
| `merge.py` | Łączenie tekstu stron (legacy helper) |

---

## PDF i obraz

| Moduł | Opis |
|--------|------|
| `pdf_render.py` | PDF → PNG per strona |
| `pdf_dpi_routing.py` | Heurystyki DPI (simple/dense/photo) |
| `scan_enhance.py` | Preprocess stron/cropów, **deskew** |
| **`document_segmentation.py`** | Regiony, orientacja, KSeF columns, resplit, merge kolumn |
| **`document_classifier.py`** | Typ dokumentu, `format_document_stream`, konsolidacja bloków |

---

## Silniki OCR

| Moduł | Opis |
|--------|------|
| **`fast_ocr.py`** | RapidOCR PP-OCRv5 Latin, GPU ONNX, warmup |
| **`line_smart_ocr.py`** | Linia: Rapid → Tess → warianty → VLM crop |
| **`ocr.py`** | Router stron/cropów, integracja engine |
| `vision_llm.py` | Ollama vision, prompty strict, `sanitize_ocr_output` |
| `minicpm_ocr.py` | Alternatywny crop VLM |
| `parallel_ocr.py` | `run_ocr_jobs`, kolejka jobów, multi-GPU |
| `crop_escalation.py` | Eskalacja trudnych cropów / linii |
| `ocr_preprocess_variants.py` | Warianty obrazu pod consensus |

---

## Jakość i język

| Moduł | Opis |
|--------|------|
| `ocr_quality.py` / `ocr_quality_pipeline.py` | Metryki i bramki jakości |
| `ocr_line_verification.py` | Rapid vs Tesseract, disagreement |
| `ocr_lines.py` | Parsowanie linii do `OcrLine` |
| `ocr_lang_routing.py` | `pol+eng` vs `eng` dla Tesseract |
| `ocr_postprocess.py` | Korekty PL po OCR |
| `ocr_lexicons.py` | Słowniki / wzorce |
| `ocr_reading_order.py` | Kolejność czytania |
| `ocr_validators.py` | Walidatory tekstu |
| `semantic_validators/` | PESEL, NIP, REGON, VIN, IBAN, daty |
| `ocr_noise_classifier.py` | Klasyfikacja szumu |
| `ocr_dedup.py` | Dedup geometryczny linii/bbox |
| `ocr_consensus.py` | Consensus wariantów |
| `ocr_router.py` | Routing trudności strony |
| `ocr_difficulty_analyzer.py` / `difficulty_analyzer.py` | Analiza trudności PDF |
| `ocr_cache.py` / `ocr_cache_integration.py` | Cache wyników OCR |

---

## Layout (rekonstrukcja)

| Moduł | Opis |
|--------|------|
| `layout/reconstruction.py` | Rekonstrukcja układu |
| `layout/columns.py` | Kolumny |
| `layout/reading_order.py` | Kolejność |
| `layout/form_detect.py` | Formularze |
| `layout/lines.py`, `barcode.py`, `handwriting.py`, `dedup.py`, `text_dedup.py` | Pomocnicze |

---

## ZIP i GPU

| Moduł | Opis |
|--------|------|
| `zip_streaming.py` | Planowanie memberów bez pełnego rozpakowania |
| `zip_queue.py` | Kolejka memberów, worker pool |
| `zip_worker.py` | Worker procesowy per member |
| `zip_checkpoint.py` | `checkpoint.json` atomowy |
| `zip_adaptive_queue.py` | Tierowanie trudności w ZIP |
| `zip_missing.py` | Analiza brakujących wyników |
| `gpu.py` / `gpu_pool.py` | Wykrywanie GPU |
| `gpu_scheduler.py` / `dynamic_gpu_scheduler.py` | Przydział GPU |
| `oom.py` | Wykrywanie OOM CUDA, retry |

---

## Przepływ zależności (skrót)

```text
routes → pipeline → { pdf_render, document_segmentation, parallel_ocr → fast_ocr, line_smart_ocr, vision_llm }
                 → document_classifier
                 → write_results
```

Nie importuj `pipeline` z modułów niskiego poziomu (unikaj cykli).

---

## Testy

`tests/` — pytest; nazwy `test_<moduł>.py` odpowiadają powyższym obszarom.

Uruchomienie: ` .venv/bin/python -m pytest tests/ -q`
