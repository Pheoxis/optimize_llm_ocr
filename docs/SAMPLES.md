# Próbki testowe (`samples/`)

Katalog `samples/` jest w `.gitignore` (dane klienta nie trafiają do repo). W repozytorium jest tylko `samples/.gitkeep`.

Na maszynie dev **skopiuj** pliki testowe do `samples/` przed uruchomieniem skryptów.

---

## Zalecany zestaw (regresja SKMBT + paragony)

| Plik (przykład) | Typ | Co testuje |
|-----------------|-----|------------|
| `paragon_1.jpg` | Zdjęcie | Szybki smart, jasny paragon |
| `paragon_2.jpg` | Zdjęcie | Trudniejsze PL, eskalacja line-smart/VLM |
| `SKMBT_C36026051111470-part-1.pdf` | PDF 5 str. | Segmentacja, faktury, wiele regionów |
| `SKMBT_C36026051111470-part-3.pdf` | PDF | Paragony / wiele dokumentów na stronie |
| `SKMBT_C36026051111470-part-5.pdf` | PDF | KSeF + DOO + formularze na jednej stronie |
| `openpdf-multireceipt-1k.zip` | ZIP duży | Multi-GPU, checkpoint, errors.json |

Nazwy `SKMBT_*` — rzeczywiste skany klienta (wielostronicowe faktury i mieszane strony).

---

## Uruchomienie

```bash
export USE_BACKGROUND_WORKER=false

# Jeden plik — patrz docs/JEZYK_OCR.md
.venv/bin/python scripts/run_sample.py samples/paragon_1.jpg --no-vlm
.venv/bin/python scripts/run_sample.py samples/faktura.pdf --polski      # faktury PL
.venv/bin/python scripts/run_sample.py samples/invoice.pdf --angielski   # dokument EN
.venv/bin/python scripts/run_sample.py samples/mixed.pdf --lang auto   # domyślne

# SKMBT (wielostronicowe faktury) — zawsze --polski
.venv/bin/python scripts/run_sample.py samples/SKMBT_C36026051111470-part-2.pdf --polski

# Wszystkie pliki w samples/
.venv/bin/python scripts/test_samples.py --polski

# Benchmark (zapis do storage/results/benchmark/)
.venv/bin/python scripts/benchmark_all_samples.py
```

Szczegóły flag: **[JEZYK_OCR.md](JEZYK_OCR.md)**.

---

## Gdzie są wyniki

```text
storage/results/paragon_1/result.txt
storage/results/SKMBT_C36026051111470-part-1/result.json
```

Przy powtórnym uruchomieniu tego samego pliku: `paragon_1-2`, itd.

---

## Kryteria jakości (orientacyjne, WIP)

| Scenariusz | Oczekiwanie |
|------------|-------------|
| Czytelny paragon JPG | Tekst czytelny, NIP/kwoty poprawne w większości |
| PDF faktura 1 strona | Nagłówek Faktura, NIP, kwoty; bez losowego merge dwóch firm |
| Strona z 2 paragonami | Dwa bloki (nie jeden zlepiony EMPIK+AMIC) |
| DOO / formularz opłat | Jeden spójny tytuł dokumentu, nie pół strony ucięte |
| KSeF wizualizacja | Identyfikator KSeF lub „WIZUALIZACJA” w tekście |

**Docelowy KPI:** ~94%+ znaków na trudnych skanach — **nie gwarantowany** w obecnej wersji (patrz [STAN_PROJEKTU.md](STAN_PROJEKTU.md) §7).

---

## Porównanie wyników

1. Uruchom `run_sample` na pliku.
2. Otwórz `storage/results/{nazwa}/result.txt`.
3. Opcjonalnie diff względem wcześniejszego `result.txt` w archiwum.

`result.json` → `metadata.pages[]` pokazuje silnik i czas per strona (debug eskalacji).

---

## Brak plików w `samples/`

```text
No sample files in …/samples
```

Dodaj co najmniej jeden plik z dozwolonym rozszerzeniem (`.pdf`, `.jpg`, …).
