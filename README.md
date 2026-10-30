# doc-ocr — lokalny odczyt dokumentów (A1–A2)

API do przetwarzania PDF, obrazów i ZIP: OCR → jeden blok tekstu → `result.json` + `result.txt`.

> **Stan i WIP:** Projekt jest **w trakcie przejścia na nową wersję pipeline’u OCR** (segmentacja wielu dokumentów na stronie, jakość PL, merge KSeF/DOO).  
> **Dokumentacja:** [docs/README.md](docs/README.md) (indeks) · [STAN_PROJEKTU.md](docs/STAN_PROJEKTU.md) · [PIPELINE.md](docs/PIPELINE.md) · [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

**Repozytorium GitLab:** [Kamil/optimize_llm_ocr](https://100.127.177.52/Kamil/optimize_llm_ocr) · gałąź `main` · commit docs: `6c98b9f`+ ([weryfikacja](docs/GITLAB_SYNC.md))

```bash
# SSH (port 2222) — po dodaniu klucza w GitLab i ~/.ssh/config
git clone ssh://git@gitlab.local:2222/Kamil/optimize_llm_ocr.git
# albo bez aliasu DNS:
git clone ssh://git@100.127.177.52:2222/Kamil/optimize_llm_ocr.git

# HTTPS (wymaga PAT / loginu)
git clone https://100.127.177.52/Kamil/optimize_llm_ocr.git
```

## Struktura

```text
LLM_Optimize/
├── app/
│   ├── main.py
│   ├── api/routes.py           # REST /api/v1/jobs
│   ├── core/config.py          # Settings (.env)
│   ├── models/schemas.py       # ProcessingResult, JobStatusResponse
│   ├── services/
│   │   ├── pipeline.py         # orchestracja IMAGE / PDF / ZIP
│   │   ├── document_segmentation.py
│   │   ├── document_classifier.py
│   │   ├── fast_ocr.py         # RapidOCR GPU
│   │   ├── line_smart_ocr.py
│   │   ├── parallel_ocr.py
│   │   ├── result_paths.py     # foldery wyników po nazwie pliku
│   │   ├── zip_*.py            # ZIP, checkpoint, worker pool
│   │   └── layout/             # rekonstrukcja układu
│   └── workers/tasks.py        # RQ
├── docs/                       # pełna dokumentacja → docs/README.md
├── scripts/                    # run_sample, test_samples, worker, retry ZIP
├── storage/{uploads,results,temp,ocr_cache}/
├── samples/                    # lokalne próbki (gitignore)
├── requirements.txt
└── .env.example
```

## Silniki OCR

Trzy silniki, **smart router** (domyślny):

| Silnik | Model | Czas / obraz | Jakość |
|--------|-------|--------------|--------|
| **fast_ocr** | RapidOCR PP-OCRv5 Latin @ GPU (ONNX + CUDA) | **~0.25 s** | 95 %+ (drobne literówki PL diakrytyków) |
| **vision_llm** | Ollama `qwen3.6:35b` | ~8–13 s | **1:1** (eskalacja dla trudnych przypadków) |
| **tesseract** | klasyk | ~5 s | dobra dla czystych skanów |

Routing przez `OCR_ENGINE`:

| Tryb | Zachowanie | Średni czas | Rekomendacja |
|------|------------|-------------|--------------|
| **`smart` (domyślne)** | Zdjęcia: line-smart. PDF crop: RapidOCR + line-smart tylko gdy słaba jakość | 0.3–11 s | **exact transcription (bez halucynacji całej strony)** |
| `fast` | tylko RapidOCR, każdy wynik zaakceptowany | 0.25 s | gdy liczy się **wyłącznie czas** |
| `vision` | zawsze Ollama vision LLM | 8–13 s | gdy wymagane **gwarantowane 1:1** |
| `tesseract` | zawsze Tesseract | 5 s | legacy fallback |
| `auto` | zdjęcie → vision; skan PDF → tesseract | 5–13 s | starszy router (bez RapidOCR) |

### Line-level smart OCR (`OCR_LINE_LEVEL_ENABLED=true`)

1. RapidOCR → linie z bbox  
2. Tesseract na cropie każdej linii (disagreement / regex VIN·PESEL·IBAN)  
3. Trudne linie → warianty preprocess (CLAHE, threshold, sharpen…) + consensus  
4. Crop-only VLM (`VISION_ESCALATION_MODEL`, prompt exact, `temperature=0`, `top_p=0`)  
5. Full-page VLM **tylko** gdy &lt; `OCR_FULLPAGE_VLM_MIN_LINES` linii i brak tekstu  

Metadane per linia w `result.json` → `metadata.lines[]` (score, engine, escalated, disagreement).

### Legacy page gate (`OCR_LINE_LEVEL_ENABLED=false`)

Eskalacja do vision LLM gdy **dowolny** z warunków jest spełniony:

- mniej niż `SMART_MIN_LINES` (3) wykrytych bbox-ów,
- średnia pewność słów < `SMART_MIN_MEAN_CONF` (0.96),
- udział słów z pewnością < `SMART_WORD_THRESHOLD` (0.85) jest > `SMART_MIN_LOW_CONF_RATIO` (7 %),
- najgorsze słowo ma pewność < `SMART_MIN_WORST_SCORE` (0.80).

Domyślne progi są **zachowawcze** — preferują 1:1 dokładność. Jeśli zależy
głównie na szybkości, podnieś `SMART_MIN_LOW_CONF_RATIO` (np. 0.20) albo
ustaw `OCR_ENGINE=fast`.

Po starcie aplikacji oba silniki są wcześniej rozgrzewane (RapidOCR + Ollama).

### CUDA dla RapidOCR

`onnxruntime-gpu` 1.26 wymaga CUDA 12; nasz PyTorch ma 13. Wymagane biblioteki **już są zainstalowane** przez inne pakiety w `nvidia/*/lib/`. Moduł `fast_ocr` ładuje je automatycznie przez `ctypes.CDLL` — **nie trzeba ustawiać `LD_LIBRARY_PATH`**.

### Dwie karty graficzne (multi-GPU)

| Typ pliku | Ścieżka | GPU |
|-----------|---------|-----|
| **Zdjęcie / paragon** (`.jpg` …) | `image_direct` — OCR od razu na pliku, bez segmentacji | **tylko GPU 0** (`IMAGE_OCR_GPU=0`) |
| **PDF ≥2 stron** | strona N → GPU N; strony równolegle | `OCR_PAGE_PARALLEL_*` |
| **PDF 1 strona, wiele cropów** | regiony równolegle na 2 GPU | `OCR_PARALLEL_MIN_JOBS` (domyślnie 4) |
| **ZIP (obrazy / PDF / mix)** | **pin pliku do GPU** (round-robin), do 1 workera/GPU | `ZIP_PARALLEL_WORKERS=0` (auto) + `ZIP_PIN_MEMBER_TO_GPU=true` |
| **1× JPG** | tylko `IMAGE_OCR_GPU` (domyślnie 0) | — |

Jakość 1:1: `OCR_ENGINE=smart`, `OCR_LINE_LEVEL_ENABLED=true`, `PDF_SMART_ESCALATE=false` (PDF bez full-page VLM).  
Pula bulk: `OCR_BULK_GPU_IDS=0,1`. W `result.json`: `ocr_page_parallel`, `ocr_parallel`, `ocr_gpu_ids`, `ocr_pinned_gpu`.

### Stabilność na dużych archiwach (ZIP 1000+ plików)

- **Pin pliku do GPU** (`ZIP_PIN_MEMBER_TO_GPU=true`, domyślnie) — każdy plik trafia na `bulk_gpus[member_index % len(gpus)]`. 1000 jednostronicowych PDF rozkłada się równo na obie karty zamiast cisnąć GPU 0.
- **ZIP simple 1–2 strony** (`ZIP_SIMPLE_RECEIPT_MAX_PAGES=2`) — OpenPDF 2-stronicowe paragony bez segmentacji bulk (szybsza ścieżka niż pełny bulk).
- **Per-file error handling** — błąd jednego pliku **nie zabija** całego joba. Wyniki gotowe lądują na dysku; nieudane pliki opisuje `storage/results/{folder}/errors.json` (`member`, `kind`, `error`, `error_type`).
  - wszystko OK → `status: done`
  - część OK → `status: done` + `error: "N/M ZIP members failed (partial result, see errors.json)"`
  - wszystkie failed → `status: failed`
- **OOM retry** (`OCR_OOM_RETRY=true`) — wykrycie `CUDA out of memory` / `CUBLAS_STATUS_ALLOC_FAILED` / `CUDNN_*` → `torch.cuda.empty_cache()` + ponowienie raz na tej samej karcie. Drugi OOM → plik trafia do `errors.json`, reszta ZIP-a leci dalej.
- **Throttle dla samych PDF** (`ZIP_PDF_MAX_PARALLEL=N`, domyślnie 0) — gdy archiwum zawiera **wyłącznie PDF**, można odgórnie ograniczyć liczbę workerów (np. `=1` na słabszych kartach żeby uniknąć OOM przy gęstych skanach). Wartość `0` = brak dodatkowego ograniczenia.
- **Checkpoint (crash-safe)** — po każdym pliku: `storage/results/{folder}/checkpoint.json` (atomowy zapis). Po crashu: `POST /jobs/{id}/resume` lub `scripts/resume_zip_checkpoint.py <job_id>`.
- **Resume po błędach** — ponów tylko nieudane pliki z `errors.json` (oryginalny ZIP musi zostać w `storage/uploads/`):

  ```bash
  # CLI
  .venv/bin/python3 scripts/retry_zip_failed.py <job_id> [--no-vlm]
  .venv/bin/python3 scripts/resume_zip_checkpoint.py <job_id> [--no-vlm]
  .venv/bin/python3 scripts/process_zip_missing.py --job-id <job_id> [--dry-run] [--no-vlm]

  # API (job musi istnieć w job store)
  curl -X POST http://localhost:8000/api/v1/jobs/<job_id>/retry-failed
  ```

### Operacje — duży ZIP (OpenPDF) i GPU

| Zasada | Ustawienie / działanie |
|--------|-------------------------|
| **Jeden duży job naraz** | Nie uruchamiaj dwóch `openpdf-multireceipt-1k.zip` równolegle — konkurują o VRAM i spowalniają się nawzajem. |
| **Sync vs kolejka** | Zalecane: `JOB_EXECUTION_MODE=auto` (queue gdy Redis działa, sync fallback gdy nie). Produkcja: `JOB_EXECUTION_MODE=queue` + Redis + `scripts/run_rq_worker.sh`. |
| **GPU 0,1 — RapidOCR (bulk)** | `OCR_BULK_GPU_IDS=0,1`, `ZIP_PIN_MEMBER_TO_GPU=true`, `ZIP_PARALLEL_WORKERS=0` (auto = 2 przy 2 GPU). ZIP OCR w **osobnych procesach** (`spawn`); lock per GPU przez `Manager`; **recycle** co `MAX_DOCS_PER_WORKER` (domyślnie 200) → `worker_recycle.json`. Zdjęcia: `IMAGE_OCR_GPU=0`. |
| **Ollama (vision)** | Domyślnie **GPU inne niż bulk** lub ten sam host — full-page VLM jest wyłączone na PDF (`PDF_SMART_ESCALATE=false`). VLM używane głównie przy zdjęciach (line-smart / eskalacja). Nie uruchamiaj ciężkiego `qwen3.6:35b` równolegle z pełnym OpenPDF na tych samych kartach bez monitorowania `nvidia-smi`. |
| **Po częściowym ZIP** | `status: done` + `errors.json` → `retry_zip_failed.py <job_id>` dokańcza **tylko** wpisy z `failures[]`. |

### Pomiar (RTX PRO 6000)

| Plik | `fast` | `smart` (default, dokładnie) | `vision` (max jakość) |
|------|--------|------------------------------|------------------------|
| `paragon_1.jpg` zdjęcie sushi, jasne | 0.32 s | **0.32 s** (gate OK, mean=0.99) | ~8 s |
| `paragon_2.jpg` Poczta, słaby PL | 0.31 s | **15 s** (eskalacja: mean 0.95 < 0.96) | ~10 s |
| `SKMBT_…part-1.pdf` 5 str. faktura | **2.5 s** | ~45 s (4/5 stron eskalowane) | ~50 s |

Czyli:

- **Czytelne zdjęcia / paragony**: smart ≈ fast (< 1 s).
- **Trudne zdjęcia z polskimi diakrytykami**: smart automatycznie używa LLM (10–15 s, ale 1:1).
- **Wielostronicowe skany PDF (SKMBT)**: segmentacja + **RapidOCR na cropach** (`OCR_LINE_LEVEL_ON_PDF_CROPS=false`). Trudne cropy → line-smart (`OCR_LINE_LEVEL_ESCALATE_BAD_PDF_CROPS=true`). Bez full-page VLM (`PDF_SMART_ESCALATE=false`). 5 stron ≈ **11 s** na 2 GPU, ~8k znaków.

- **Wielostronicowe faktury PDF (stary problem z LLM)**: tryb `smart` z eskalacją do vision był wolny (~50 s) i ucinał tekst; dla PDF ustaw `PDF_SMART_ESCALATE=false` (domyślnie) lub `OCR_ENGINE=fast`. Dalsze poluzowanie:

```env
OCR_ENGINE=smart
SMART_MIN_MEAN_CONF=0.90         # mniej eskalacji → szybciej
SMART_MIN_LOW_CONF_RATIO=0.20
SMART_MIN_WORST_SCORE=0.60
```

## Uruchomienie (dev)

```bash
cd /root/LLM_Optimize
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# RapidOCR pobierze ~12 MB modeli (PP-OCRv5 det + cls + latin rec) przy
# pierwszym uruchomieniu i zachowa w .venv/lib/.../rapidocr/models/.

# Wymagane: Ollama uruchomiona z modelem qwen3.6:35b
systemctl status ollama
ollama list | grep qwen3.6

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Etap 1 — worker (osobny terminal, wymaga Redis):**

```bash
# Debian/Ubuntu: sudo apt install redis-server && sudo systemctl start redis-server
cp .env.example .env   # JOB_EXECUTION_MODE=auto (lub queue)
chmod +x scripts/run_rq_worker.sh
./scripts/run_rq_worker.sh
```

- `POST /api/v1/jobs` → natychmiast `{ "status": "queued", "job_id": "..." }`
- `GET /api/v1/jobs/{id}` → postęp / `done` / `failed`
- Sync (dev bez Redis): `JOB_EXECUTION_MODE=sync` (lub `auto` z wyłączonym Redisem)

- Dokumentacja API: `http://<tailscale-ip>:8000/docs` (po `./scripts/run_api.sh`)  
- Health: `http://<tailscale-ip>:8000/health`  

## API (`/api/v1`)

### Faktury propOS

| Metoda | Ścieżka | Opis |
|--------|---------|------|
| POST | `/invoice/extract` | **Sync (domyślnie):** POST wiszy → JSON faktury w odpowiedzi |
| POST | `/invoice/extract?async=true` | Async: 202 + `job_id` |
| GET | `/invoice/jobs/{id}` | Poll wyniku faktury (async) |

**Integracja:** [docs/PROPOS_API.md](docs/PROPOS_API.md) — checklist, przykłady curl/Python, konfiguracja.

### Ogólny OCR (PDF / ZIP)

| Metoda | Ścieżka | Opis |
|--------|---------|------|
| POST | `/jobs` | Upload → `queued` (async) lub sync |
| GET | `/jobs/{id}` | Status + `progress` |
| GET | `/jobs/{id}/results` | Lista ścieżek wyników (ZIP) |
| GET | `/jobs/{id}/result.json` | Wynik JSON (PDF/obraz) |
| GET | `/jobs/{id}/result.txt` | Wynik TXT |
| GET | `/jobs/{id}/files/{member}/result.json` | Wynik pliku z ZIP |
| GET | `/jobs/{id}/files/{member}/result.txt` | Tekst pliku z ZIP |
| POST | `/jobs/{id}/resume` | Wznów ZIP z `checkpoint.json` (pomiń ukończone) |
| POST | `/jobs/{id}/retry-failed` | Ponów tylko members z `errors.json` |

Po `done`: `storage/results/{nazwa_pliku}/job.json` (gdy `RESULTS_DIR_USE_SOURCE_NAME=true`) lub `storage/results/{job_id}/`.  
API nadal używa UUID w URL; folder wyników = stem nazwy uploadu (patrz [STAN_PROJEKTU.md](docs/STAN_PROJEKTU.md) §4).

**Pełny indeks:** [docs/README.md](docs/README.md)

## Test lokalny

```bash
source .venv/bin/activate
export USE_BACKGROUND_WORKER=false   # sync, bez Redis — zalecane do run_sample

# Paragon / szybki test
.venv/bin/python scripts/run_sample.py samples/paragon_1.jpg --no-vlm

# Polskie faktury PDF (SKMBT, VAT) — lepsze ą/ę/ł/ń
.venv/bin/python scripts/run_sample.py samples/SKMBT_C36026051111470-part-2.pdf --polski

# Angielski dokument
.venv/bin/python scripts/run_sample.py samples/invoice_en.pdf --angielski

# Wszystkie pliki w samples/ z hintem PL
.venv/bin/python scripts/test_samples.py --polski

python scripts/smoke_a2.py         # A2: ZIP + sync + Redis/worker (jeśli redis działa)
```

**Flagi języka:** `--polski` / `--angielski` / `--lang auto` (+ opcjonalnie `--no-vlm`).  
Pełny opis: [docs/JEZYK_OCR.md](docs/JEZYK_OCR.md).

## Harmonogram

- **A1:** `pdf_render`, `ocr`, `pipeline`, eksport wyników — **zrobione**  
- **A2:** ZIP, postęp, limit 15 GB, Redis/RQ, `job.json` — **zrobione** (optymalizacja kolejek → po A2)  
