# Stan projektu doc-ocr (LLM_Optimize)

**Data dokumentu:** 2026-05-29  
**Gałąź:** `main` (A2 scalone)  
**Repozytorium:** https://gitlab.local/Kamil/optimize_llm_ocr  
**Wersja docelowa:** przejście na **nową wersję pipeline’u OCR** (w trakcie — patrz §7)

**Indeks dokumentacji:** [README.md](README.md) · szczegóły techniczne: [PIPELINE.md](PIPELINE.md), [ARCHITECTURE.md](ARCHITECTURE.md), [API.md](API.md), [SCHEMA.md](SCHEMA.md), [ENV_REFERENCE.md](ENV_REFERENCE.md), [TROUBLESHOOTING.md](TROUBLESHOOTING.md), [SAMPLES.md](SAMPLES.md)

---

## 1. Cel systemu

Lokalne API (FastAPI) do OCR dokumentów biznesowych **PL + EN**:

- wejście: PDF, obrazy (JPG/PNG/…), archiwa ZIP (do 15 GB),
- wyjście: `result.txt` + `result.json` (jeden strumień tekstu + metadane OCR),
- priorytet: **dokładność transkrypcji** (1:1, bez „wymyślania” treści przez LLM na całej stronie),
- drugi priorytet: **prędkość** na GPU (RapidOCR PP-OCRv5, multi-GPU, równoległość).

Typowe przypadki: paragony, faktury, formularze (np. DOO), wizualizacje KSeF, skany wielostronicowe (seria SKMBT).

---

## 2. Architektura (obecny stan)

```text
POST /api/v1/jobs  ──►  job_store (pamięć lub Redis)
         │                    │
         ├─ sync ─────────────┼──► pipeline.run_job()
         └─ queue ──► RQ ──► worker ──► pipeline.run_job()
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
              IMAGE (1 plik)                  PDF (bulk)
         _process_single_image          _process_bulk_document
                    │                               │
                    │                    render → segmentacja → OCR cropów
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                         write_results → storage/results/
```

### Główne moduły

| Moduł | Rola |
|--------|------|
| `app/services/pipeline.py` | Orchestracja: typ pliku, retry (obrót, resplit, KSeF), ZIP |
| `app/services/document_segmentation.py` | Regiony na stronie, orientacja 0/90/180/270°, merge kolumn |
| `app/services/document_classifier.py` | Typ dokumentu, konsolidacja bloków, format wyjścia |
| `app/services/scan_enhance.py` | Preprocess + **deskew** (<15°) cropów |
| `app/services/fast_ocr.py` | RapidOCR GPU (ONNX CUDA) |
| `app/services/line_smart_ocr.py` | Linia po linii: RapidOCR → Tesseract → VLM crop |
| `app/services/vision_llm.py` | Ollama vision (eskalacja) |
| `app/services/result_paths.py` | Nazwy folderów wyników = stem pliku źródłowego |
| `app/services/zip_*` | ZIP: checkpoint, błędy per plik, adaptive queue, worker pool |

| Temat | Dokument |
|-------|----------|
| Przepływ krok po kroku | [PIPELINE.md](PIPELINE.md) |
| Wszystkie moduły | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Kontrakt JSON / pliki | [SCHEMA.md](SCHEMA.md) |
| API HTTP | [API.md](API.md) |
| Zmienne `.env` | [ENV_REFERENCE.md](ENV_REFERENCE.md) |
| Problemy / FAQ | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Próbki testowe | [SAMPLES.md](SAMPLES.md) |
| OCR PL/EN | [OCR_PL_EN.md](OCR_PL_EN.md) |
| Redis / worker | [A2.md](A2.md) |

---

## 3. Silniki OCR i tryby

| `OCR_ENGINE` | Zachowanie |
|--------------|------------|
| **`smart`** (domyślny) | RapidOCR; eskalacja line-smart / VLM przy niskiej pewności |
| `fast` | Tylko RapidOCR |
| `vision` | Zawsze Ollama |
| `tesseract` | Legacy |
| `auto` | Stary router (zdjęcie→vision, PDF→tesseract) |

Ważne flagi (`.env.example`):

- `PDF_SMART_ESCALATE=false` — **bez** full-page VLM na PDF (szybciej, mniej halucynacji).
- `OCR_LINE_LEVEL_ON_PDF_CROPS` — line-smart na trudnych cropach PDF.
- `PDF_DESKEW_CROPS=true` — korekta małego skosu skanera na cropach.
- `OCR_BULK_GPU_IDS=0,1` — pula GPU dla PDF/ZIP.

---

## 4. Wyniki na dysku

### Nazewnictwo folderów (od 2026-05-29)

Gdy `RESULTS_DIR_USE_SOURCE_NAME=true` (domyślnie):

```text
storage/results/paragon_1/
storage/results/SKMBT_C36026051111470-part-1/
```

- Powtórzenie tego samego pliku → `nazwa-2`, `nazwa-3`, …
- **Job ID (UUID)** nadal w API: `GET /api/v1/jobs/{uuid}`
- Mapowanie `uuid → folder`: pole `results_folder` w job store + `job.json`
- Wyłączenie: `RESULTS_DIR_USE_SOURCE_NAME=false` → z powrotem `storage/results/{uuid}/`

### Pliki w folderze joba

| Plik | Opis |
|------|------|
| `result.txt` | Tekst sformatowany (nagłówki typów dokumentów) |
| `result.json` | Pełny kontrakt + `metadata.pages`, linie OCR |
| `job.json` | Snapshot po `done`/`failed` (czas, status, `results_folder`) |
| `checkpoint.json` | ZIP: postęp crash-safe |
| `errors.json` | ZIP: lista nieudanych memberów |
| `{member}/result.*` | ZIP: wynik per plik w archiwum |

Katalog `storage/ocr_cache/` — cache hash→OCR (lokalny, w `.gitignore`).

---

## 5. Uruchomienie (skrót)

### Test jednego pliku (bez Redis, bez workera)

```bash
cd LLM_Optimize
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# W .env dla CLI:
#   USE_BACKGROUND_WORKER=false
#   JOB_EXECUTION_MODE=sync

export USE_BACKGROUND_WORKER=false
.venv/bin/python scripts/run_sample.py samples/paragon_1.jpg --no-vlm
# Faktury / SKMBT po polsku:
.venv/bin/python scripts/run_sample.py samples/SKMBT_C36026051111470-part-2.pdf --polski
# Wynik: storage/results/{nazwa_pliku}/result.txt
# Flagi języka: docs/JEZYK_OCR.md
```

### API + kolejka (produkcja)

```bash
redis-server
./scripts/run_rq_worker.sh          # terminal 2
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -F "file=@samples/paragon_1.jpg" http://localhost:8000/api/v1/jobs
```

### Wszystkie próbki z `samples/`

```bash
export USE_BACKGROUND_WORKER=false
.venv/bin/python scripts/test_samples.py
```

Wymagania: Python 3.12+, CUDA (zalecane), Ollama + `qwen3.6:35b` (tylko przy eskalacji VLM).

---

## 6. Co działa stabilnie (A2 + optymalizacje)

- Upload PDF / obraz / ZIP, limit 15 GB.
- Job sync lub Redis/RQ + worker.
- ZIP: równoległość multi-GPU, pin member→GPU, OOM retry per plik, checkpoint/resume/retry-failed.
- PDF wielostronicowy: segmentacja regionów, batch OCR, routing DPI.
- Zdjęcia: ścieżka bezpośrednia + line-smart.
- Deskew cropów, retry obrotu 0/90/180/270° przy „śmieciowym” OCR.
- Merge dwóch kolumn jednego dokumentu (faktury, formularze DOO).
- Foldery wyników po nazwie pliku źródłowego.

---

## 7. W trakcie zmian — nowa wersja pipeline’u

> **Ten dokument opisuje stan w gałęzi `main` po merge `A2`. Część zachowań jest jeszcze dopracowywana.**

### Znane ograniczenia / otwarte tematy

| Obszar | Status |
|--------|--------|
| Jedna strona PDF: KSeF + faktura + formularz DOO obok siebie | Czasem nadal jeden blok zamiast trzech logicznych sekcji |
| Kolejność kolumn w DOO (formularz opłat) | Tekst bywa przeplatany między kolumnami |
| Paragony pod dużym skosem (np. EMPIK) | Lepsze po deskew + rotation retry; nie zawsze 94%+ |
| Scalanie wielu paragonów na jednym skanie | Heurystyki `looks_like_merged_receipts` + resplit — w toku |
| Pełna dokumentacja API OpenAPI vs rzeczywiste pola `metadata` | Uzupełnianie w SCHEMA |
| Panel WWW / pola `structured` | Poza zakresem A2 |
| Mirror na GitHub | Repozytorium źródłowe: GitLab `Kamil/optimize_llm_ocr` |

### Kierunek „nowej wersji”

1. **Segmentacja strony** — lepsze rozdzielanie niezależnych dokumentów na jednym skanie (bez tnienia jednej faktury na pół).
2. **Klasyfikacja i merge** — spójny model bloków (KSeF / FV / paragon / formularz) przed eksportem TXT.
3. **Jakość PL** — postprocess + słowniki + line-verify bez nadmiernej eskalacji VLM.
4. **Metryki** — benchmark `scripts/benchmark_all_samples.py`, profile `pipeline_profile`.

Zmiany wprowadzane iteracyjnie na `A2` / `main`; ten plik należy aktualizować po większych merge’ach.

---

## 8. Repozytorium i gałęzie

| Gałąź | Opis |
|-------|------|
| `main` | Linia stabilna po merge A2 |
| `A2` | Rozwój funkcji klienta A2 + OCR (merge do `main`) |

```bash
git clone ssh://git@gitlab.local:2222/Kamil/optimize_llm_ocr.git
cd optimize_llm_ocr
git checkout main
```

---

## 9. Testy

```bash
.venv/bin/python -m pytest tests/ -q
# wybrane:
.venv/bin/python -m pytest tests/test_result_paths.py tests/test_zip_checkpoint.py -q
```

---

## 10. Skrócona mapa skryptów CLI

| Skrypt | Użycie |
|--------|--------|
| `scripts/run_sample.py` | Jeden plik lokalnie |
| `scripts/test_samples.py` | Cały katalog `samples/` |
| `scripts/smoke_a2.py` | Smoke A2 (sync + opcjonalnie queue) |
| `scripts/retry_zip_failed.py` | Ponów błędy z `errors.json` |
| `scripts/resume_zip_checkpoint.py` | Wznów po crashu |
| `scripts/process_zip_missing.py` | Brakujące wyniki ZIP |
| `scripts/run_rq_worker.sh` | Worker RQ |
| `scripts/benchmark_all_samples.py` | Pomiar czasów/jakości |

---

*Przy pytaniach o konkretny plik wynikowy — podaj ścieżkę `storage/results/{nazwa}/result.txt` i nazwę próbki ze `samples/`.*
