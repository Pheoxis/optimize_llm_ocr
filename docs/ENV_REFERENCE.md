# Referencja zmiennych środowiskowych

Pełna lista z komentarzami: [.env.example](../.env.example).  
Poniżej — **grupy** i typowe profile ustawień.

---

## Profile ustawień (szybki wybór)

### Dev — jeden plik, bez Redis

```env
JOB_EXECUTION_MODE=sync
USE_BACKGROUND_WORKER=false
RESULTS_DIR_USE_SOURCE_NAME=true
OCR_ENGINE=smart
PDF_SMART_ESCALATE=false
```

### Produkcja — API + worker

```env
JOB_EXECUTION_MODE=queue
USE_BACKGROUND_WORKER=true
REDIS_URL=redis://127.0.0.1:6379/0
OCR_BULK_GPU_IDS=0,1
ZIP_PIN_MEMBER_TO_GPU=true
```

### Maksymalna szybkość (akceptuj gorsze PL)

```env
OCR_ENGINE=fast
OCR_LINE_LEVEL_ENABLED=false
PDF_SMART_ESCALATE=false
```

### Maksymalna dokładność (wolno)

```env
OCR_ENGINE=smart
OCR_LINE_LEVEL_ENABLED=true
SMART_MIN_MEAN_CONF=0.98
VISION_ESCALATION_MODEL=minicpm-v
```

---

## Silnik i jakość OCR

| Zmienna | Domyślnie | Opis |
|---------|-----------|------|
| `OCR_ENGINE` | `smart` | `smart` \| `fast` \| `vision` \| `tesseract` \| `auto` |
| `OCR_LINE_LEVEL_ENABLED` | `true` | Line-smart na zdjęciach |
| `OCR_LINE_LEVEL_ON_PDF_CROPS` | `false` | Line-smart na każdym cropie PDF (wolne) |
| `OCR_LINE_LEVEL_ESCALATE_BAD_PDF_CROPS` | `true` | Line-smart tylko na słabych cropach PDF |
| `PDF_SMART_ESCALATE` | `false` | Full-page VLM na PDF |
| `OCR_PAGE_VISION_ESCALATION_ENABLED` | `false` | Legacy full-page VLM |
| `SMART_MIN_MEAN_CONF` | `0.96` | Próg eskalacji (page gate) |
| `SMART_MIN_LOW_CONF_RATIO` | `0.07` | Udział słów słabej pewności |
| `PDF_DESKEW_CROPS` | `true` | Deskew cropów &lt;15° |

---

## PDF — render i segmentacja

| Zmienna | Opis |
|---------|------|
| `PDF_DPI_ROUTING_ENABLED` | Dynamiczne DPI |
| `PDF_DPI_SIMPLE` / `DENSE` / `DEFAULT` | DPI dla klas stron |
| `PDF_SEGMENT_DOCUMENTS` | Wykrywanie wielu dokumentów na stronie |
| `PDF_AUTO_ROTATE` | Auto-obrót strony |
| `PDF_MIN_REGION_AREA_PCT` | Min. powierzchnia regionu |
| `PDF_MAX_REGIONS_PER_PAGE` | Limit regionów |
| `PDF_MERGE_RESPLIT_PARTS` | Resplit części strony (false = mniej agresywne) |
| `PDF_FULLPAGE_FALLBACK_*` | Kiedy OCR całej strony zamiast cropów |

---

## GPU i równoległość

| Zmienna | Opis |
|---------|------|
| `IMAGE_OCR_GPU` | GPU dla pojedynczych zdjęć |
| `OCR_BULK_GPU_IDS` | Lista GPU dla PDF/ZIP, np. `0,1` |
| `OCR_PAGE_PARALLEL_*` | Równoległe strony PDF |
| `OCR_PARALLEL_*` | Równoległe cropy |
| `FAST_USE_CUDA` | RapidOCR na GPU |

---

## ZIP

| Zmienna | Opis |
|---------|------|
| `ZIP_PARALLEL_WORKERS` | `0` = auto (≈ liczba GPU) |
| `ZIP_PIN_MEMBER_TO_GPU` | Round-robin GPU per plik |
| `ZIP_PDF_MAX_PARALLEL` | Limit workerów dla samych PDF (`0` = brak) |
| `ZIP_SIMPLE_RECEIPT_*` | Szybka ścieżka 1–2 strony |
| `MAX_DOCS_PER_WORKER` | Recycle procesu po N plikach |
| `ZIP_MEMBER_TIMEOUT_SEC` | Timeout na member |
| `OCR_OOM_RETRY` | Ponowienie po OOM CUDA |

---

## Kolejka i storage

| Zmienna | Opis |
|---------|------|
| `JOB_EXECUTION_MODE` | `auto` \| `queue` \| `sync` |
| `USE_BACKGROUND_WORKER` | Legacy; użyj z `JOB_EXECUTION_MODE` |
| `REDIS_URL` | Redis dla job store + RQ |
| `RESULTS_DIR_USE_SOURCE_NAME` | Folder wyników = stem pliku |
| `MAX_UPLOAD_BYTES` | Limit uploadu (15 GB domyślnie) |
| `OCR_CACHE_ENABLED` | Cache OCR na dysku |

---

## Vision / Ollama

| Zmienna | Opis |
|---------|------|
| `OLLAMA_URL` | URL Ollama |
| `VISION_ESCALATION_MODEL` | Model crop VLM (np. `minicpm-v`) |
| `VISION_MODEL` | Model full-page (rzadko używany przy PDF) |
| `VISION_TIMEOUT` | Timeout sekundy |

---

## Tesseract i język

| Zmienna | Opis |
|---------|------|
| `OCR_LANG` | Gdy routing wyłączony |
| `OCR_DOCUMENT_LANGUAGE` | Globalny hint języka (`auto` / `pl` / `en`); CLI: `--polski`, `--angielski` — patrz [JEZYK_OCR.md](JEZYK_OCR.md) |
| `OCR_LANG_ROUTING_ENABLED` | `pol+eng` vs `eng` per linia |
| `OCR_PSM_DOCUMENT` / `RECEIPT` | PSM Tesseract |

---

## Dedup i szum

| Zmienna | Opis |
|---------|------|
| `OCR_DEDUP_*` | Progi deduplikacji geometrycznej |
| `OCR_FILTER_SPURIOUS_FRAGMENTS` | Filtr śmieciowych fragmentów |
| `OCR_SEMANTIC_VALIDATION_*` | PESEL/NIP/… walidacja i eskalacja |

---

## Semantyczna walidacja

| Zmienna | Opis |
|---------|------|
| `OCR_SEMANTIC_VALIDATION_ENABLED` | Włącz walidatory |
| `OCR_SEMANTIC_VALIDATION_ESCALATE` | Eskalacja przy błędach checksum |

Szczegóły modułów: [OCR_PL_EN.md](OCR_PL_EN.md), [ARCHITECTURE.md](ARCHITECTURE.md).
