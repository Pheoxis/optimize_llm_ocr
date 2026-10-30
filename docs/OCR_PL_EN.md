# PL/EN exact-transcription OCR architecture

This pipeline is **OCR exact transcription only** — no semantic rewriting, autocorrect, or meaning inference.

## Engine tiers (target mix)

| Tier | Share (target) | Engine | Role |
|------|----------------|--------|------|
| 1 | ~95% | RapidOCR + PP-OCRv5 Latin | Primary OCR (GPU) |
| 2 | ~4% | Tesseract `pol` / `eng` | Line-crop verification |
| 3 | ~1% | MiniCPM-V / Qwen2.5-VL 7B | Crop-only vision escalation |

Full-page vision OCR is **off by default** (`OCR_PAGE_VISION_ESCALATION_ENABLED=false`).

## Modules

| Spec | Module |
|------|--------|
| Primary RapidOCR | `app/services/fast_ocr.py` |
| Language routing (`pol+eng` vs `eng`) | `app/services/ocr_lang_routing.py` |
| Line verification (Levenshtein) | `app/services/ocr_line_verification.py` |
| Polish validators (PESEL, NIP, REGON, IBAN, VIN, dates) | `app/services/semantic_validators/` |
| Crop vision escalation | `app/services/crop_escalation.py`, `app/services/vision_llm.py` |
| ZIP streaming / workers | `app/services/zip_streaming.py`, `zip_queue.py`, `zip_worker.py` |
| OCR cache | `app/services/ocr_cache.py` |
| Dynamic PDF DPI | `app/services/pdf_dpi_routing.py` |
| Universal duplicate suppression | `app/services/ocr_dedup.py` |

## Duplicate suppression (all document types)

One policy for photos, PDF crops, forms, receipts, and ZIP members:

- Remove only **geometric** double detections (overlap, containment, same-row touch gap).
- Text must be **exactly** the same (`casefold`; `ą` ≠ `a`).
- **Never** remove repeated labels at separate horizontal positions (form columns).
- When merging duplicates, prefer the reading with **more Polish diacritics** and higher OCR score.

Config: `OCR_DEDUP_OVERLAP_IOU`, `OCR_DEDUP_MAX_ADJACENT_GAP_PX`, etc.

## Language routing

After RapidOCR produces a line sample:

- Polish diacritics present → Tesseract `pol+eng`
- Otherwise → Tesseract `eng` only

RapidOCR language packs are **not** changed (PP-OCRv5 Latin covers PL+EN).

## Vision OCR rules

- Strict prompt in `vision_llm._PROMPT_OCR_STRICT`
- Deterministic sampling: `temperature=0`, `top_p=0`, `repeat_penalty=1`
- Default escalation model: `minicpm-v` (config: `VISION_ESCALATION_MODEL`)

## Forbidden behavior

- Autocorrecting OCR text (e.g. `Parstwo` → `Państwo`)
- Full-page vision by default
- Heavy models (`qwen3.6:35b`, gemma vision) for crop escalation
