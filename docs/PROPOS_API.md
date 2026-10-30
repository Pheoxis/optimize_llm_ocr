# API faktur propOS — integracja

Dokument dla zespołu propOS / integratorów. Opisuje **jak działa** endpoint faktur, **jak z niego korzystać** oraz **co zostało zrealizowane** w ramach integracji.

**Repozytorium:** `Kamil/optimize_llm_ocr` · gałąź `main`  
**Powiązane:** [API.md](API.md) (pełna referencja HTTP), [ENV_REFERENCE.md](ENV_REFERENCE.md), [.env.example](../.env.example)

---

## 1. Co zostało zrobione (checklist)

| # | Wymaganie | Status | Uwagi |
|---|-----------|--------|-------|
| 1 | `POST` z plikiem faktury (PDF / obraz) | ✅ | `POST /api/v1/invoice/extract` |
| 2 | Połączenie wisi do końca procesu (timeout 30–40 s konfigurowalny) | ✅ | **Tryb domyślny (sync)** — patrz §3 |
| 3 | Gotowy, zmapowany JSON faktury w **tej samej odpowiedzi HTTP** | ✅ | HTTP 200, payload §4.1 / §4.2 |
| 4 | Format propOS: `fields`, `line_items[]`, `vat_summary[]` | ✅ | Kwoty w **groszach** (integer) |
| 5 | Każde pole nagłówka: `value` + `confidence` (0–1) | ✅ | `null` + `confidence: 0` gdy nieczytelne |
| 6 | `line_items` — wartości skalarne + `confidence` wiersza | ✅ | |
| 7 | `vat_summary` — `rate`, `net`, `vat`, `gross` | ✅ | Zawsze tablica (może być `[]`) |
| 8 | `external_id` — korelacja po stronie klienta | ✅ | Form field przy uploadzie |
| 9 | `job_id` w odpowiedzi | ✅ | UUID |
| 10 | Błąd biznesowy: `status: "failed"` + `error.code` | ✅ | Np. `UNREADABLE`, `EXTRACTION_FAILED` |
| 11 | Limit pliku 20 MB → HTTP 413 | ✅ | `INVOICE_MAX_UPLOAD_BYTES` |
| 12 | Opcjonalny Bearer (`OCR_SERVICE_TOKEN`) | ✅ | Puste = brak auth |
| 13 | Tryb async (202 + poll) — opcjonalnie | ✅ | `?async=true` |
| 14 | Swagger / docs pod Tailscale (LAN) | ✅ | Self-hosted UI, `./scripts/run_api.sh` |
| 15 | Izolacja pierwszej faktury z wielostronicowego PDF | ✅ | Bez paragonów / kolejnych FV w `raw_text` |
| 16 | Webhook / kolejka Redis dla faktur | ❌ | Poza zakresem — tylko sync lub poll |

---

## 2. Uruchomienie serwera

### Wymagania

- Python 3.12+, venv z `requirements.txt`
- **Ollama** z modelem tekstowym (domyślnie `INVOICE_TEXT_MODEL`, np. `qwen3.6:35b`)
- GPU opcjonalne (przyspiesza OCR)

### Start API (Tailscale / LAN)

```bash
cd LLM_Optimize
python3 -m venv .venv          # tylko przy pierwszym uruchomieniu
source .venv/bin/activate
pip install -r requirements.txt

# Zatrzymaj stary proces jeśli port zajęty:
pkill -f "uvicorn app.main:app" 2>/dev/null; sleep 1

# Uruchom (nasłuchuje na wszystkich interfejsach):
./scripts/run_api.sh
```

Domyślny port: **8000**. Inny port: `API_PORT=8001 ./scripts/run_api.sh`.

### Adresy

| Usługa | URL |
|--------|-----|
| Health | `http://<host>:8000/health` |
| Swagger UI | `http://<host>:8000/docs` |
| OpenAPI JSON | `http://<host>:8000/openapi.json` |

`<host>` = Tailscale IP serwera (np. `100.x.x.x`) lub `localhost` lokalnie.

> Swagger ładuje assety **z serwera** (`/static/swagger/`), nie z CDN — działa w sieci Tailscale bez internetu.

---

## 3. Jak działa API — przepływ

```
Plik faktury (PDF/JPG/…)
        │
        ▼
   OCR (RapidOCR / Tesseract)
        │
        ▼
   Izolacja tekstu pierwszej FAKTURY (wielostronicowe skany)
        │
        ▼
   LLM (Ollama, structured JSON)
        │
        ▼
   Normalizacja (NIP, grosze, confidence, sumy)
        │
        ▼
   JSON propOS (§4.1 done lub §4.2 failed)
```

### Tryb **sync** (domyślny — zalecany dla propOS)

1. Klient robi **jeden** `POST` z plikiem.
2. Połączenie **pozostaje otwarte** do końca OCR + ekstrakcji.
3. Serwer zwraca **HTTP 200** z pełnym JSON (`status: done` lub `failed`).

Timeout po stronie serwera: `INVOICE_PROCESS_TIMEOUT_SEC` (domyślnie **40 s**).  
**Klient** powinien ustawić własny timeout ≥ 60–120 s (pełny PDF + LLM może trwać ~90–110 s).

### Tryb **async** (opcjonalny)

1. `POST /api/v1/invoice/extract?async=true` → natychmiast **HTTP 202** + `{ job_id, status: "queued" }`.
2. Poll: `GET /api/v1/invoice/jobs/{job_id}` aż `status` = `done` | `failed`.

---

## 4. Endpointy

### 4.1 Ekstrakcja faktury (sync)

```http
POST /api/v1/invoice/extract
Content-Type: multipart/form-data
Authorization: Bearer <token>     # tylko gdy OCR_SERVICE_TOKEN ustawiony
external_id: pi_12345             # opcjonalnie
file: <binary>
```

**Odpowiedź:** HTTP **200**, body = payload §4.1 lub §4.2 (poniżej).

### 4.2 Ekstrakcja faktury (async)

```http
POST /api/v1/invoice/extract?async=true
```

**Odpowiedź:** HTTP **202**

```json
{
  "job_id": "8f14e45f-ceea-467f-a8d4-91b8e6c1d2a3",
  "status": "queued"
}
```

### 4.3 Poll wyniku (tylko async)

```http
GET /api/v1/invoice/jobs/{job_id}
```

Statusy pośrednie: `queued`, `processing`. Końcowe: `done`, `failed`.

---

## 5. Format odpowiedzi

### Sukces — `status: "done"`

```json
{
  "job_id": "8f14e45f-ceea-467f-a8d4-91b8e6c1d2a3",
  "external_id": "pi_12345",
  "document_type": "invoice",
  "status": "done",
  "error": null,
  "processing": {
    "engine": "rapidocr+tesseract",
    "duration_ms": 107000,
    "pages": 2
  },
  "result": {
    "confidence": 0.92,
    "fields": {
      "invoice_number": { "value": "FV/2026/05/001", "confidence": 0.98 },
      "issue_date": { "value": "2026-05-12", "confidence": 0.97 },
      "due_date": { "value": "2026-05-26", "confidence": 0.95 },
      "currency": { "value": "PLN", "confidence": 0.99 },
      "total_net": { "value": 10000, "confidence": 0.96 },
      "total_vat": { "value": 2300, "confidence": 0.96 },
      "total_gross": { "value": 12300, "confidence": 0.98 },
      "seller": {
        "name": { "value": "Tauron Sprzedaż Sp. z o.o.", "confidence": 0.95 },
        "nip": { "value": "9542583988", "confidence": 0.99 },
        "address": { "value": "ul. Łagiewnicka 60, 30-417 Kraków", "confidence": 0.85 }
      },
      "buyer": {
        "name": { "value": "Ogrody Włochy Sp. z o.o.", "confidence": 0.95 },
        "nip": { "value": "5252344078", "confidence": 0.99 },
        "address": { "value": "ul. Przykładowa 1, 02-123 Warszawa", "confidence": 0.85 }
      },
      "bank_account": { "value": "PL61109010140000071219812874", "confidence": 0.92 },
      "payment_method": { "value": "przelew", "confidence": 0.80 },
      "description": { "value": "Energia elektryczna 04/2026", "confidence": 0.85 }
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
    "raw_text": "TAURON Sprzedaż...\n..."
  }
}
```

**Zasady:**

- Kwoty w **groszach** (`12300` = 123,00 PLN).
- NIP: 10 cyfr, bez myślników w `value`.
- `line_items` i `vat_summary` **zawsze** są tablicami.
- `raw_text` obcięty do ~2500 znaków + `...` (pierwsza faktura ze skanu).

### Błąd — `status: "failed"`

```json
{
  "job_id": "...",
  "external_id": "pi_12345",
  "document_type": "invoice",
  "status": "failed",
  "error": {
    "code": "UNREADABLE",
    "message": "Dokument nieczytelny — brak tekstu OCR."
  },
  "processing": {
    "engine": "rapidocr",
    "duration_ms": 75,
    "pages": 1
  },
  "result": null
}
```

W trybie **sync** błąd biznesowy też wraca jako **HTTP 200** z `status: failed` (nie 4xx), zgodnie z kontraktem.

### Kody `error.code`

| Kod | Znaczenie |
|-----|-----------|
| `UNREADABLE` | Brak / zbyt słaby tekst OCR |
| `OCR_FAILED` | Wyjątek w pipeline OCR |
| `EXTRACTION_FAILED` | Błąd LLM (Ollama niedostępna, zły model, invalid JSON) |
| `TIMEOUT` | Przekroczony `INVOICE_PROCESS_TIMEOUT_SEC` |

### Błędy HTTP (transport)

| Kod | Znaczenie |
|-----|-----------|
| 400 | Złe rozszerzenie / brak nazwy pliku |
| 401 | Brak lub zły Bearer |
| 413 | Plik > 20 MB |
| 404 | Nieznany `job_id` (poll) |

---

## 6. Przykłady integracji

### curl — sync (propOS)

```bash
export API=http://100.x.x.x:8000   # Tailscale IP serwera OCR
export TOKEN=                      # opcjonalnie

curl -s --max-time 120 \
  -X POST "$API/api/v1/invoice/extract" \
  -H "Authorization: Bearer $TOKEN" \
  -F "external_id=pi_12345" \
  -F "file=@faktura.pdf" \
  | jq .
```

### curl — async + poll

```bash
JOB=$(curl -s -X POST "$API/api/v1/invoice/extract?async=true" \
  -F "external_id=pi_12345" -F "file=@faktura.pdf" | jq -r .job_id)

until [ "$STATUS" = "done" ] || [ "$STATUS" = "failed" ]; do
  sleep 2
  STATUS=$(curl -s "$API/api/v1/invoice/jobs/$JOB" | jq -r .status)
  echo "status=$STATUS"
done

curl -s "$API/api/v1/invoice/jobs/$JOB" | jq .
```

### Python (sync)

```python
import requests

resp = requests.post(
    "http://100.x.x.x:8000/api/v1/invoice/extract",
    files={"file": open("faktura.pdf", "rb")},
    data={"external_id": "pi_12345"},
    timeout=120,
)
data = resp.json()
if data["status"] == "done":
    fields = data["result"]["fields"]
    print(fields["invoice_number"]["value"], fields["total_gross"]["value"])
else:
    print(data["error"])
```

---

## 7. Konfiguracja (.env)

| Zmienna | Domyślnie | Opis |
|---------|-----------|------|
| `INVOICE_PROCESS_TIMEOUT_SEC` | `40` | Timeout procesu (zwiększ do 120 przy dużych PDF) |
| `INVOICE_MAX_UPLOAD_BYTES` | `20971520` | 20 MB |
| `INVOICE_TEXT_MODEL` | `qwen2.5:32b` | Model Ollama do ekstrakcji JSON |
| `INVOICE_RAW_TEXT_MAX_CHARS` | `2500` | Limit `raw_text` w odpowiedzi |
| `OCR_SERVICE_TOKEN` | puste | Bearer; puste = auth wyłączone |
| `API_CORS_ORIGINS` | `*` | CORS dla Swagger / klientów Tailscale |

Pełna lista: [.env.example](../.env.example).

---

## 8. Ograniczenia i uwagi produkcyjne

- **Czas:** typowy PDF 1–5 stron: **60–110 s** (OCR + LLM). Ustaw timeout klienta i serwera odpowiednio wyżej.
- **Wielostronicowe skany** z wieloma dokumentami: ekstrahowana jest **pierwsza faktura**; pozostałe strony nie trafiają do `result`.
- **Jakość:** przy niespójnych sumach VAT system **nie koryguje** kwot — obniża `confidence` nagłówka.
- **Model LLM:** musi być dostępny w Ollama (`ollama list`). Na serwerze testowym używany był `qwen3.6:35b`.
- **Osobne API jobów OCR** (`POST /api/v1/jobs`) nadal istnieje do ogólnego OCR/ZIP — **nie** jest to endpoint faktur propOS.

---

## 9. Testy w repozytorium

```bash
.venv/bin/python -m pytest tests/test_invoice_*.py -q
```

Przykładowe pliki: `samples/SKMBT_C36026051111470-part-1.pdf`.

---

## 10. Kontakt / dalsze kroki

- Dopasowanie pól 1:1 do zamrożonego schematu propOS (gdy klient dostarczy finalny JSON Schema).
- Webhook zamiast polla — poza obecnym zakresem.
- Paragony (`document_type: receipt`) — planowane później.

*Ostatnia aktualizacja dokumentu: czerwiec 2026.*
