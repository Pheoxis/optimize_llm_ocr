# Dokumentacja doc-ocr (LLM_Optimize)

**Repozytorium:** [Kamil/optimize_llm_ocr](https://gitlab.local/Kamil/optimize_llm_ocr) · gałąź `main`  
**Status:** pipeline OCR **w trakcie przejścia na nową wersję** — patrz [STAN_PROJEKTU.md](STAN_PROJEKTU.md) §7.

---

## Indeks dokumentów

| Dokument | Dla kogo | Zawartość |
|----------|----------|-----------|
| [STAN_PROJEKTU.md](STAN_PROJEKTU.md) | Wszyscy | Cel, stan, WIP, uruchomienie skrót, ograniczenia |
| [PIPELINE.md](PIPELINE.md) | Developerzy | Przepływ IMAGE / PDF / ZIP, retry, segmentacja, klasyfikacja |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Developerzy | Mapa modułów `app/services/`, zależności |
| [SCHEMA.md](SCHEMA.md) | Integratorzy | Kontrakt JSON, job API, pliki na dysku |
| [API.md](API.md) | Integratorzy | Endpointy, przykłady curl, progress, błędy |
| [PROPOS_API.md](PROPOS_API.md) | propOS / integratorzy | **API faktur** — jak działa, checklist, sync/async, Tailscale |
| [A2.md](A2.md) | Ops | Redis, RQ worker, wdrożenie produkcyjne |
| [ENV_REFERENCE.md](ENV_REFERENCE.md) | Ops / dev | Zmienne `.env` pogrupowane z opisem |
| [OCR_PL_EN.md](OCR_PL_EN.md) | OCR / ML | Zasady PL+EN, dedup, tiery silników |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Wszyscy | Typowe problemy (venv, Redis, brak wyniku) |
| [SAMPLES.md](SAMPLES.md) | QA / dev | Próbki testowe, benchmark, kryteria jakości |
| [JEZYK_OCR.md](JEZYK_OCR.md) | Wszyscy | Flagi `--polski` / `--angielski`, `OCR_DOCUMENT_LANGUAGE` |

**Szybki start:** [README.md](../README.md) w katalogu głównym.

**Konfiguracja pełna (komentarze):** [.env.example](../.env.example).

**API interaktywne:** `http://<host>:8000/docs` po `./scripts/run_api.sh` (Tailscale / LAN).
