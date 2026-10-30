# Język dokumentu (PL / EN / auto)

Pipeline obsługuje **hint języka** — informuje OCR, że skan jest po polsku lub angielsku.  
Dzięki temu lepiej radzi sobie z polskimi znakami (ą, ę, ł, ń, ó, ś, ź, ż) i nie myli ich z literami łacińskimi.

---

## Szybki start (jeden plik)

```bash
cd LLM_Optimize
source .venv/bin/activate
export USE_BACKGROUND_WORKER=false   # sync — bez Redis/workera

# Polski dokument (faktury, paragony PL) — ZALECANE dla SKMBT i faktur VAT
.venv/bin/python scripts/run_sample.py samples/faktura.pdf --polski

# Angielski dokument
.venv/bin/python scripts/run_sample.py samples/invoice_en.pdf --angielski

# Auto — heurystyka po tekście (domyślne)
.venv/bin/python scripts/run_sample.py samples/mixed.pdf --lang auto

# Bez eskalacji VLM (szybciej, tylko RapidOCR + Tesseract)
.venv/bin/python scripts/run_sample.py samples/paragon_1.jpg --polski --no-vlm
```

Wynik: `storage/results/{nazwa_pliku}/result.txt` i `result.json`.

---

## Flagi CLI

| Flaga | Skrót | Efekt |
|-------|-------|--------|
| `--lang pl` | `--polski` | Wymusza `pol+eng` w Tesseract, polski postprocessing, `language_detected: ["pl","en"]` |
| `--lang en` | `--angielski` | Wymusza `eng`, bez polskiego postprocessingu |
| `--lang auto` | *(domyślne)* | Routing po wykrytych znakach diakrytycznych w próbce OCR |
| `--no-vlm` | — | Wyłącza eskalację vision LLM (niezależne od języka) |

Skróty `--polski` / `--angielski` są równoważne `--lang pl` / `--lang en`.

**Nie używaj jednocześnie** `--polski` i `--angielski`.

### Wszystkie pliki w `samples/`

```bash
.venv/bin/python scripts/test_samples.py --polski
.venv/bin/python scripts/test_samples.py --angielski
.venv/bin/python scripts/test_samples.py --lang auto
```

---

## Zmienna `.env` (globalnie)

```env
# auto | pl | en — domyślny hint gdy CLI nie podaje flagi
OCR_DOCUMENT_LANGUAGE=auto
```

Przy `pl` w `.env` każdy job (także z API) dostaje polski routing bez flagi w CLI.  
CLI `--polski` / `--lang pl` **nadpisuje** na czas jednego uruchomienia `run_sample` / `test_samples`.

---

## Co robi hint `pl` w pipeline

1. **Tesseract** — od razu `pol+eng` (nie czeka na ą/ę/ł w pierwszej linii OCR).
2. **Weryfikacja linii** — nie pomija Tesseract przy wysokim confidence, gdy widać typowe błędy PL (`Jokale`, `llość`, `wystawienła`…).
3. **Postprocessing** — moduł `polish_text_recovery` + `ocr_postprocess` (np. `wystawienła`→`wystawienia`, `pfatnoścl`→`płatności`, `zt`→`zł`).
4. **Wynik** — pole `language_detected` w `result.json`.

Moduły: `app/services/document_language.py`, `app/services/polish_text_recovery.py`.

---

## Kiedy używać której flagi

| Typ dokumentu | Flaga |
|---------------|--------|
| Faktury VAT, noty, paragony PL, SKMBT | `--polski` |
| Umowy / faktury wyłącznie po angielsku | `--angielski` |
| Nie wiesz / mix PL+EN | `--lang auto` |
| Szybki test bez VLM | dowolna + `--no-vlm` |

---

## API (worker / sync)

Obecnie `POST /api/v1/jobs` przyjmuje sam plik — hint języka z CLI nie dotyczy API.  
Globalnie ustaw `OCR_DOCUMENT_LANGUAGE=pl` w `.env` na serwerze OCR, albo (przyszłość) pole `options.document_language` w metadanych uploadu.

Lokalne testy zawsze przez `scripts/run_sample.py` z flagą `--polski`.

---

## Przykłady regresji (SKMBT)

```bash
export USE_BACKGROUND_WORKER=false

for n in 2 3 4 5; do
  .venv/bin/python scripts/run_sample.py \
    "samples/SKMBT_C36026051111470-part-$n.pdf" --polski
done
```

Sprawdź: `storage/results/SKMBT_C36026051111470-part-*/result.txt` — pola typu *Data wystawienia*, *Termin płatności*, *Płatność*.

Więcej o próbkach: [SAMPLES.md](SAMPLES.md).  
Zmienne środowiskowe: [ENV_REFERENCE.md](ENV_REFERENCE.md) (`OCR_DOCUMENT_LANGUAGE`, `OCR_LANG_*`).
