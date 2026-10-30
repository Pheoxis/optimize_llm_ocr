# API HTTP — `/api/v1`

Baza: `http://localhost:8000` (lub Tailscale IP maszyny, np. `http://100.x.x.x:8000`).  
OpenAPI: `/docs` (Swagger, assety hostowane lokalnie — bez CDN), `/redoc`, health: `/health`.

Uruchomienie pod Tailscale/LAN: `./scripts/run_api.sh` (`--host 0.0.0.0`).

**Ważne:** URL jobów używa **UUID**. Folder wyników na dysku może mieć **nazwę pliku** (`results_folder`) — patrz [SCHEMA.md](SCHEMA.md).

---

## Tworzenie joba

```http
POST /api/v1/jobs
Content-Type: multipart/form-data
file: <binary>
```

### Odpowiedź (sync — bez Redis)

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "done",
  "message": "Job completed (sync mode)"
}
```

### Odpowiedź (queue)

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "message": "Job queued — poll GET /jobs/{id}"
}
```

Upload zapisywany jako: `storage/uploads/{job_id}_{oryginalna_nazwa}`.

---

## Status joba

```http
GET /api/v1/jobs/{job_id}
```

### Przykład — PDF w trakcie

```json
{
  "job_id": "a1b2c3d4-…",
  "status": "processing",
  "progress": {
    "current": 3,
    "total": 5,
    "unit": "pages",
    "percent": 60.0
  },
  "created_at": "2026-05-29T12:00:00+00:00",
  "updated_at": "2026-05-29T12:00:15+00:00",
  "error": null,
  "results_folder": "SKMBT_C36026051111470-part-1",
  "result_files": []
}
```

### Przykład — ZIP w trakcie

```json
{
  "status": "processing",
  "progress": {
    "current": 120,
    "total": 1000,
    "unit": "files",
    "percent": 12.0
  },
  "results_folder": "openpdf-multireceipt-1k"
}
```

### Przykład — zakończony

```json
{
  "status": "done",
  "progress": {
    "current": 5,
    "total": 5,
    "unit": "pages",
    "percent": 100.0
  },
  "results_folder": "SKMBT_C36026051111470-part-1",
  "result_files": [
    "storage/results/SKMBT_C36026051111470-part-1/result.json",
    "storage/results/SKMBT_C36026051111470-part-1/result.txt"
  ],
  "error": null
}
```

### Częściowy sukces ZIP

`status` może być `done` z:

```json
"error": "42/1000 ZIP members failed (partial result, see errors.json)"
```

Szczegóły w `storage/results/{results_folder}/errors.json`.

---

## Pobieranie wyników

| Metoda | Ścieżka | Kiedy |
|--------|---------|--------|
| GET | `/jobs/{id}/results` | Lista `result_files` |
| GET | `/jobs/{id}/result.json` | Pojedynczy PDF/obraz |
| GET | `/jobs/{id}/result.txt` | Tekst |
| GET | `/jobs/{id}/files/{member_path}/result.json` | Plik w ZIP |
| GET | `/jobs/{id}/files/{member_path}/result.txt` | Tekst pliku w ZIP |

`member_path` — ścieżka względna w archiwum, np. `folder/doc.pdf`.

**409** gdy status ≠ `done`.

---

## ZIP — wznawianie

```http
POST /api/v1/jobs/{job_id}/resume
```

Wymaga `checkpoint.json` w folderze wyników i oryginalnego ZIP w `uploads/`.

```http
POST /api/v1/jobs/{job_id}/retry-failed
```

Wymaga `errors.json` z listą `failures`.

---

## Przykłady curl

```bash
# Upload
curl -s -X POST http://127.0.0.1:8000/api/v1/jobs \
  -F "file=@samples/paragon_1.jpg" | jq .

export JOB_ID="<uuid>"

# Poll
watch -n2 "curl -s http://127.0.0.1:8000/api/v1/jobs/$JOB_ID | jq .status,.progress"

# Wynik
curl -s "http://127.0.0.1:8000/api/v1/jobs/$JOB_ID/result.txt"

# ZIP — member
curl -s "http://127.0.0.1:8000/api/v1/jobs/$JOB_ID/files/subdir/faktura.pdf/result.txt"
```

---

## Kody błędów HTTP

| Kod | Znaczenie |
|-----|-----------|
| 400 | Zły plik / rozszerzenie / rozmiar |
| 404 | Nieznany `job_id` |
| 409 | Wynik niedostępny (job nie `done`) |
| 503 | Redis niedostępny (tryb queue) |

---

## Znalezienie folderu wyników po UUID

1. `GET /jobs/{id}` → `results_folder`
2. Dysk: `storage/results/{results_folder}/`
3. Fallback: `job.json` w tym folderze lub skan `storage/results/*/job.json`

Patrz też: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Faktura propOS

**Obsługiwane typy:** PDF, JPG, JPEG, PNG, TIFF, WEBP (max 20 MB).  
**Timeout:** `INVOICE_PROCESS_TIMEOUT_SEC` (domyślnie 40 s) — sync i async.

### Sync (domyślny) — POST wiszy, zwraca JSON

```http
POST /api/v1/invoice/extract
Content-Type: multipart/form-data
external_id: pi_12345   # opcjonalnie
file: <binary>
```

Połączenie pozostaje otwarte do końca OCR + ekstrakcji, potem **HTTP 200** z pełnym payloadem §4.1 / §4.2 (`status: done` lub `failed`).

### Async — `?async=true`

```http
POST /api/v1/invoice/extract?async=true
```

**HTTP 202 Accepted:**

```json
{ "job_id": "8f14e45f-ceea-467f-a8d4-91b8e6c1d2a3", "status": "queued" }
```

Poll:

```http
GET /api/v1/invoice/jobs/{job_id}
```

Statusy: `queued` → `processing` → `done` | `failed`.

### §4.1 Sukces (`status: "done"`)

```json
{
  "job_id": "8f14e45f-ceea-467f-a8d4-91b8e6c1d2a3",
  "external_id": "pi_12345",
  "document_type": "invoice",
  "status": "done",
  "error": null,
  "processing": {
    "engine": "rapidocr+vlm",
    "duration_ms": 14200,
    "pages": 2
  },
  "result": {
    "confidence": 0.93,
    "fields": {
      "invoice_number": { "value": "FV/2026/05/001", "confidence": 0.98 },
      "total_gross": { "value": 12300, "confidence": 0.98 },
      "seller": {
        "name": { "value": "Tauron Sprzedaż Sp. z o.o.", "confidence": 0.95 },
        "nip": { "value": "9542583988", "confidence": 0.99 }
      }
    },
    "line_items": [
      {
        "name": "Energia elektryczna — sprzedaż",
        "quantity": 1450.5,
        "unit": "kWh",
        "unit_price_net": 68,
        "total_net": 9863,
        "vat_rate": "23",
        "total_gross": 12131,
        "confidence": 0.91
      }
    ],
    "vat_summary": [
      { "rate": "23", "net": 10000, "vat": 2300, "gross": 12300 }
    ],
    "raw_text": "..."
  }
}
```

Pola nagłówka (`fields.*`): `{ value, confidence }`. Pozycje (`line_items`): wartości skalarne + `confidence` wiersza. Kwoty w **groszach**.

### §4.2 Błąd (`status: "failed"`)

```json
{
  "job_id": "...",
  "external_id": "pi_12345",
  "document_type": "invoice",
  "status": "failed",
  "error": {
    "code": "UNREADABLE",
    "message": "Dokument nieczytelny — zbyt niska rozdzielczość skanu."
  },
  "processing": { "engine": "rapidocr", "duration_ms": 3100, "pages": 1 },
  "result": null
}
```

### Błędy HTTP (upload)

| Kod | Znaczenie |
|-----|-----------|
| 400 | Zły plik / rozszerzenie |
| 401 | Brak lub zły Bearer |
| 413 | Plik > 20 MB |
| 404 | Nieznany `job_id` (poll) |

### Przykład curl

```bash
# Sync (domyślnie) — jeden POST, JSON w odpowiedzi
curl -s --max-time 60 -X POST "http://localhost:8000/api/v1/invoice/extract" \
  -F "external_id=pi_12345" \
  -F "file=@faktura.pdf" | jq .

# Async — upload + poll
curl -s -X POST "http://localhost:8000/api/v1/invoice/extract?async=true" \
  -F "external_id=pi_12345" -F "file=@faktura.pdf"
curl -s "http://localhost:8000/api/v1/invoice/jobs/JOB_ID"
```
