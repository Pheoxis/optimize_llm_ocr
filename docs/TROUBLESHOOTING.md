# Rozwiązywanie problemów

## 1. `run_sample.py` kończy się od razu, bez komunikatów

**Przyczyny:**

| Problem | Rozwiązanie |
|---------|-------------|
| `python3` z systemu zamiast venv | `.venv/bin/python scripts/run_sample.py …` |
| Brak zależności | `pip install -r requirements.txt` w aktywnym venv |
| ImportError przed `print` | Uruchom z katalogu projektu: `cd LLM_Optimize` |
| Segfault ONNX/GPU | Sprawdź `nvidia-smi`, drivery; spróbuj `--no-vlm` |

**Oczekiwany output:**

```text
Job ID: …
Processing paragon_1.jpg …
Status: done
Results: …/storage/results/paragon_1/
```

---

## 2. Brak folderu w `storage/results/`

| Przyczyna | Co zrobić |
|-----------|-----------|
| Job `failed` | Przeczytaj `Error:` w konsoli lub `job.json` |
| `USE_BACKGROUND_WORKER=true` bez Redis | Ustaw `USE_BACKGROUND_WORKER=false` lub uruchom Redis |
| Zły katalog roboczy | `pwd` musi wskazywać korzeń repo |
| Szukasz UUID zamiast nazwy pliku | Przy `RESULTS_DIR_USE_SOURCE_NAME=true` folder = stem pliku, np. `paragon_1` |

Mapowanie UUID → folder: `GET /api/v1/jobs/{id}` → pole `results_folder` lub `job.json` w folderze wyników.

---

## 3. API zwraca `queued`, nic się nie dzieje

1. `redis-cli ping` → musi być `PONG`.
2. Worker: `./scripts/run_rq_worker.sh` w osobnym terminalu.
3. `.env`: `JOB_EXECUTION_MODE=queue` lub `auto` z działającym Redis.
4. Logi workera — czy `process_document_task` startuje.

**Dev bez kolejki:**

```env
USE_BACKGROUND_WORKER=false
JOB_EXECUTION_MODE=sync
```

---

## 4. Worker „pusty”, brak logów

Normalne, jeśli **nie ma jobów w kolejce** — worker czeka. `run_sample.py` **nie używa** workera (OCR w procesie skryptu).

Joby w kolejce tworzy tylko `POST /api/v1/jobs` przy trybie queue.

---

## 5. Redis `Connection refused`

```bash
sudo systemctl start redis-server
redis-cli ping
```

Albo przełącz na sync (patrz §3).

---

## 6. Ollama / VLM

| Objaw | Rozwiązanie |
|-------|-------------|
| Timeout eskalacji | `ollama serve`, `ollama pull qwen3.6` (lub model z `.env`) |
| Wolne PDF | `PDF_SMART_ESCALATE=false`, `run_sample --no-vlm` |
| OOM na GPU | Nie uruchamiaj ciężkiego modelu równolegle z pełnym ZIP; `nvidia-smi` |

---

## 7. CUDA / RapidOCR

| Objaw | Rozwiązanie |
|-------|-------------|
| Wolny OCR | Sprawdź w logu `rapidocr_provider` ≠ CPU; [README](../README.md) § CUDA |
| OOM na ZIP | `ZIP_PDF_MAX_PARALLEL=1`, jeden duży ZIP naraz |
| Drugi OOM | Plik w `errors.json`, reszta ZIP kontynuuje |

---

## 8. ZIP: `status: done` ale brakuje plików

1. Otwórz `storage/results/{folder}/errors.json`.
2. Ponów: `scripts/retry_zip_failed.py <job_id>`.
3. Po crashu: `scripts/resume_zip_checkpoint.py <job_id>`.
4. Braki na dysku: `scripts/process_zip_missing.py --job-id <job_id>`.

Oryginalny ZIP musi być w `storage/uploads/{job_id}_*.zip`.

---

## 9. Stary folder `storage/results/_index/`

Nieużywany — można usunąć:

```bash
rm -rf storage/results/_index
```

Mapowanie job → folder jest w **job store** (`results_folder`) i `job.json`.

---

## 10. Jakość OCR (biznesowe)

To **nie** błąd instalacji — znane ograniczenia v2: [STAN_PROJEKTU.md](STAN_PROJEKTU.md) §7.

Próbki: [SAMPLES.md](SAMPLES.md). Tuning: [ENV_REFERENCE.md](ENV_REFERENCE.md).

---

## 11. Przydatne komendy diagnostyczne

```bash
# venv OK?
.venv/bin/python -c "from app.core.config import settings; print(settings.ocr_engine)"

# CUDA RapidOCR
.venv/bin/python -c "from app.services.fast_ocr import cuda_runtime_status; print(cuda_runtime_status())"

# Test jednego pliku
export USE_BACKGROUND_WORKER=false
.venv/bin/python scripts/run_sample.py samples/paragon_1.jpg --no-vlm

# Testy jednostkowe
.venv/bin/python -m pytest tests/test_result_paths.py -q
```

Logi API: uruchom `uvicorn` z `--log-level debug`.
