from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "doc-ocr"
    debug: bool = False
    api_prefix: str = "/api/v1"
    # Behind reverse proxy / Tailscale Serve (optional)
    api_root_path: str = ""
    # OpenAPI servers[].url — empty = relative "/" (works with any host/IP)
    api_public_url: str = ""
    # CORS for Swagger + API from Tailscale/LAN ("*" = any origin)
    api_cors_origins: str = "*"

    def cors_origins_list(self) -> list[str]:
        raw = (self.api_cors_origins or "").strip()
        if not raw or raw == "*":
            return ["*"]
        return [part.strip() for part in raw.split(",") if part.strip()]

    # Storage (relative to project root)
    project_root: Path = Path(__file__).resolve().parents[2]
    storage_dir: Path = project_root / "storage"
    uploads_dir: Path = storage_dir / "uploads"
    results_dir: Path = storage_dir / "results"
    temp_dir: Path = storage_dir / "temp"
    # Name storage/results/{sanitized_source_stem}/ instead of {uuid}/ (index maps job_id → folder)
    results_dir_use_source_name: bool = True

    # Limits (A2)
    max_upload_bytes: int = 15 * 1024**3  # 15 GB
    max_zip_uncompressed_bytes: int = 15 * 1024**3
    allowed_extensions: frozenset[str] = frozenset(
        {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}
    )

    # PDF / OCR
    pdf_dpi: int = 250  # fallback when dynamic routing disabled
    # Dynamic PDF render DPI (heuristic routing)
    pdf_dpi_routing_enabled: bool = True
    pdf_dpi_simple: int = 180  # 1–2 page receipts, few regions
    pdf_dpi_dense: int = 280  # multi-page / large / multi-region scans
    pdf_dpi_default: int = 250
    pdf_dpi_photo_base: int = 220
    pdf_dpi_photo_max: int = 320
    pdf_dpi_photo_std_scale: float = 2.5  # extra DPI per luma std above threshold
    pdf_dpi_photo_luma_std_threshold: float = 28.0
    pdf_dpi_probe: int = 72  # low-res probe render for routing only
    pdf_dpi_routing_probe_render: bool = True
    pdf_dpi_routing_region_probe: bool = False  # costly; enable for dense page detection
    pdf_dpi_dense_min_pages: int = 3
    pdf_dpi_dense_min_bytes_per_page: int = 400_000
    pdf_dpi_dense_min_area_sq_in: float = 75.0
    pdf_dpi_simple_max_pages: int = 2
    pdf_dpi_simple_max_regions: int = 2
    pdf_dpi_simple_max_area_sq_in: float = 90.0
    # Tesseract verification langs (RapidOCR uses PP-OCRv5 Latin — unchanged)
    ocr_lang: str = "pol+eng"  # used when ocr_lang_routing_enabled=false
    # Per-run document language: auto | pl | en (CLI: --lang / --polski / --angielski)
    ocr_document_language: str = "auto"
    ocr_lang_routing_enabled: bool = True
    ocr_lang_tesseract_polish: str = "pol+eng"
    ocr_lang_tesseract_english: str = "eng"
    ocr_lang_polish_char_min: int = 2  # min ąęł… chars in sample to pick pol+eng
    ocr_psm_document: int = 6  # uniform block of text (PDF scans)
    ocr_psm_receipt: int = 6  # uniform block — best for photo receipts with x-cutoff filter
    ocr_receipt_scale: float = 1.5  # upscale phone photos
    ocr_receipt_strips: int = 3  # vertical tiles for long receipts
    ocr_receipt_overlap_ratio: float = 0.15
    ocr_upscale_documents: bool = False
    pdf_deskew_crops: bool = True  # correct small-angle scanner tilt (< 15°) per crop

    # Engine routing
    #   "smart"     → RapidOCR (GPU, ~0.25 s); escalate to vision LLM when
    #                 the OCR confidence falls below the gate (RECOMMENDED)
    #   "fast"      → RapidOCR only — accept its result regardless of confidence
    #   "vision"    → always vision LLM (slowest, best 1:1 quality)
    #   "tesseract" → always tesseract (legacy fallback)
    #   "auto"      → photos: vision LLM, clean PDF scans: tesseract (legacy)
    ocr_engine: str = "smart"

    # Smart-mode quality gate (RapidOCR → vision LLM escalation).
    #
    # Defaults bias toward 1:1 accuracy: if RapidOCR shows any sign of
    # uncertainty (e.g. blurry Polish diacritics), the page is re-OCRed by
    # the vision LLM.  Raise the thresholds for a "more permissive / faster"
    # behaviour, or set OCR_ENGINE=fast to skip escalation entirely.
    smart_min_mean_conf: float = 0.96      # global mean of word scores
    smart_min_low_conf_ratio: float = 0.07  # max fraction of words with score < smart_word_threshold
    smart_word_threshold: float = 0.85     # individual word "low confidence" cutoff
    smart_min_worst_score: float = 0.80    # if any single word is below this → escalate
    smart_min_lines: int = 3               # below this many detected boxes → escalate (legacy page gate)

    # Line-level smart OCR (exact transcription; crop-only VLM)
    ocr_line_level_enabled: bool = True  # pojedyncze zdjęcia / JPG w ZIP
    ocr_line_level_on_pdf_crops: bool = False  # False = RapidOCR na cropach PDF (szybko)
    ocr_line_level_escalate_bad_pdf_crops: bool = True  # crop PDF → line-smart tylko gdy Rapid słaby
    pdf_crop_line_escalate_mean_conf: float = 0.85
    pdf_crop_line_escalate_min_conf: float = 0.50
    pdf_crop_line_escalate_min_boxes: int = 2
    ocr_line_crop_padding_px: int = 8
    ocr_line_skip_tesseract_above: float = 0.94  # pewne linie → bez kosztu Tesseract
    ocr_line_disagreement_threshold: float = 0.40  # eskalacja przy dużym rozjazdzie Rapid↔Tess
    ocr_line_disagreement_light: float = 0.22  # miękki próg → tylko consensus, bez wariantów
    # Line verification: Rapid vs Tesseract on same crop (no full-page rerun).
    ocr_line_verification_enabled: bool = True
    ocr_line_suspicious_threshold: float = 0.40  # normalized Levenshtein → mark suspicious
    ocr_line_verification_skip_high_conf_tesseract: bool = True  # auto-skip Tess when Rapid is very confident
    ocr_line_tesseract_max_lines_per_page: int = 12  # 0 = unlimited; auto budget for mixed client inputs
    ocr_line_escalate_mean_conf: float = 0.82  # poniżej → eskalacja „hard”
    ocr_line_escalate_mean_conf_light: float = 0.93  # disagreement liczy się tylko poniżej
    ocr_line_escalate_min_conf: float = 0.55
    ocr_line_escalate_quality_below: float = 0.55
    ocr_line_multi_variant_on_light: bool = False  # bez 6× RapidOCR na lekkiej eskalacji
    ocr_line_vlm_without_multi_variant: bool = True  # VLM na crop zamiast pełnego multi-preprocess
    ocr_page_vision_escalation_enabled: bool = False  # legacy full-page VLM (discouraged)
    ocr_crop_merge_adjacent_lines: bool = True  # one vision call for consecutive bad lines
    ocr_crop_batch_size: int = 1  # chunk size for future batched VLM calls (1 = sequential)
    ocr_fullpage_vlm_min_lines: int = 3  # unused when page vision disabled
    ocr_consensus_enabled: bool = True
    line_score_weight_conf: float = 0.35
    line_score_weight_char: float = 0.25
    line_score_weight_regex: float = 0.20
    line_score_weight_agreement: float = 0.20

    # Semantic field validation (PESEL, NIP, REGON, VIN, IBAN, dates, invoices)
    ocr_semantic_validation_enabled: bool = True
    ocr_semantic_validation_escalate: bool = True  # mark lines for retry on checksum/format fail
    ocr_semantic_validation_escalate_warnings: bool = False  # near-miss PESEL etc.

    # Single-image fast path (paragony, pojedyncze zdjęcia)
    image_ocr_gpu: int = 0  # GPU 0 — zostaw GPU 1 dla Ollamy

    # Multi-GPU OCR — tylko duże PDF / ZIP (patrz progi poniżej)
    ocr_bulk_gpu_ids: str = ""  # puste = wszystkie karty; "0,1" = obie
    ocr_parallel_enabled: bool = True
    ocr_parallel_workers: int = 0  # 0 = po jednym workerze na kartę w puli bulk
    ocr_page_parallel_enabled: bool = True   # strona N → GPU N (równoległe strony)
    ocr_page_parallel_min_pages: int = 2     # od ilu stron włączyć page-parallel
    ocr_parallel_min_pages: int = 3       # region-parallel: PDF ≥ N stron
    ocr_parallel_min_jobs: int = 4        # region-parallel: ≥ N cropów (1-str. PDF)
    ocr_parallel_min_bytes: int = 2_000_000  # region-parallel: plik ≥ 2 MB

    # RapidOCR (fast engine)
    fast_use_cuda: bool = True
    fast_ocr_version: str = "PPOCRV5"   # PPOCRV5 = newest, multilingual (Latin)
    fast_lang_type: str = "latin"       # latin = polish, english, spanish, ...
    fast_model_type: str = "mobile"     # mobile (fast) / server (accurate)
    # Quality-driven retry on PP-OCRv5 *server* (more accurate, slower).
    # When the mobile model returns low-confidence results we re-OCR the
    # same image with the server model and pick the better of the two.
    # This adapts automatically — invoices & high-quality scans stay fast,
    # blurry photos / faint scans get an accuracy boost without manual tuning.
    fast_server_retry_enabled: bool = False
    fast_server_retry_min_score: float = 0.95
    fast_server_retry_low_conf_ratio: float = 0.10
    fast_server_retry_min_boxes: int = 6
    # Garbled-text heuristic: when mobile output looks Polish-noisy
    # (very low diacritic ratio for Polish-looking pages, many junk tokens,
    # repeated nonsense words) — also escalate to the server model.
    fast_server_retry_on_garbled_text: bool = True
    # Full-page Tesseract pol+eng fallback when RapidOCR returns Polish garble
    # (faint scans, phone photos of letters/invoices, multi-line forms with
    # poor diacritic recognition).  Adopted only when its Polish quality
    # score is at least N times the RapidOCR score.
    ocr_tesseract_polish_fullpage_fallback: bool = True
    # Tesseract needs to score at least this fraction of RapidOCR's quality
    # score to be adopted.  Lowered from 1.1 → 0.92 so that Tesseract pol+eng
    # wins whenever it produces correct Polish diacritics (which the quality
    # scoring rewards via diac_ratio), even if its clean_ratio is slightly lower.
    ocr_tesseract_polish_fallback_min_ratio: float = 0.92
    fast_line_y_tolerance: int = 10
    fast_line_separator: str = "    "   # used to join boxes on the same row
    # Universal OCR dedup (geometry + exact text; works for forms, receipts, PDF, ZIP)
    ocr_dedup_overlap_iou: float = 0.40
    ocr_dedup_max_adjacent_gap_px: int = 5       # same-row touching double-detect
    ocr_dedup_min_vertical_overlap: float = 0.50
    ocr_dedup_containment_ratio: float = 0.82    # small box inside larger, same text
    ocr_dedup_max_repeat_center_px: int = 180    # tile echo only within this X distance
    ocr_dedup_max_repeat_center_ratio: float = 0.08
    ocr_dedup_tile_echo_similarity: float = 0.90  # PL diacritics preserved in compare
    # Drop form chrome / empty-field OCR noise (zt, gr, 16 Z, FZ, …)
    ocr_filter_spurious_fragments: bool = True
    ocr_noise_short_max_score: float = 0.84
    ocr_noise_amount_max_score: float = 0.85
    ocr_noise_low_conf_threshold: float = 0.72
    ocr_noise_max_short_len: int = 4

    # PDF / multi-document scan handling
    # Scanner output (e.g. SKMBT) often packs several receipts on one A4
    # page, sometimes rotated 90°/180°.  When enabled, every rendered page
    # is segmented into separate documents and each one is OCRed individually.
    pdf_segment_documents: bool = True
    pdf_auto_rotate: bool = True
    pdf_min_region_area_pct: float = 0.008  # catch smaller receipts on dense pages
    pdf_min_region_side_pct: float = 0.03   # min width/height vs page (segmentation)
    pdf_max_regions_per_page: int = 24      # safety cap (batch scans)
    pdf_osd_min_confidence: float = 1.5     # below this → keep original orientation
    # When a PDF page is segmented into multiple document regions (receipts,
    # invoices) we still need separation in the output, but we intentionally
    # avoid noisy visual separators like "--- (dokument i/total) ---".
    # The caller can split by pages/paragony separately if needed; here we
    # use a simple blank-line join.
    pdf_document_separator: str = "\n\n"
    # PDFs: RapidOCR on scans is fast and complete; vision LLM is slow and can
    # truncate long receipts — do not escalate by default on PDF crops/pages.
    pdf_smart_escalate: bool = False
    # If segmented OCR captures less than this fraction of a full-page pass,
    # fall back to the full-page text (avoids missing receipts on dense pages).
    pdf_fullpage_fallback_ratio: float = 0.88
    # Full-page OCR mixes side-by-side receipts; only use when ≤ N regions found.
    pdf_fullpage_fallback_max_regions: int = 1
    # When segmented total is below this (chars), prefer full-page if not merged.
    pdf_fullpage_fallback_min_segmented_chars: int = 400
    # Below this seg/full ratio, use full-page even with multiple regions.
    pdf_fullpage_fallback_aggressive_ratio: float = 0.45
    pdf_resplit_garbled: bool = True
    # Przyciemnij jasny/szary druk na skanach batch (CLAHE + gamma) przed OCR.
    pdf_enhance_scans: bool = True
    pdf_enhance_faint_auto: bool = True  # auto-detect faint columns → strong darken
    # Po resplit garbled: osobne dokumenty dla każdego sub-regionu (EMPIK vs AMIC).
    pdf_merge_resplit_parts: bool = False
    # Tall crops / pages: vertical tiling in fast_ocr (px)
    fast_tile_min_height: int = 1800
    fast_tile_overlap_ratio: float = 0.12

    # Vision LLM (Ollama)
    ollama_url: str = "http://127.0.0.1:11434"
    # Crop-only vision OCR (local). Prefer small OCR VL models — NOT qwen3.6:35b / gemma vision.
    vision_escalation_model: str = "minicpm-v"  # crop escalation default
    vision_model: str = "qwen2.5vl:7b"  # OCR_ENGINE=vision only (full image; discouraged)
    vision_model_fast: str = "minicpm-v"  # legacy alias → escalation model
    # Legacy env keys — OCR uses fixed sampling in vision_llm._ocr_vision_options().
    vision_top_p: float = 0.0
    vision_repeat_penalty: float = 1.0
    vision_max_side: int = 1024  # px — sweet spot for qwen3.6:35b
    vision_jpeg_quality: int = 90
    vision_num_ctx: int = 4096
    vision_num_predict: int = 1500  # most receipts fit under 1.2k tokens
    vision_timeout: float = 300.0
    vision_keep_alive: str = "60m"  # keep model warm in VRAM

    # ── MiniCPM-V dedicated crop escalation ─────────────────────────────────
    # Direct transformers inference path — avoids Ollama HTTP overhead and
    # targets sub-second latency for single/few-line crop regions.
    #
    # Priority chain (crop escalation):
    #   1. minicpm_direct_enabled=True + model loads  → direct transformers
    #   2. minicpm_ollama_fallback=True               → Ollama (existing path)
    #   3. neither                                    → skip VLM
    #
    minicpm_direct_enabled: bool = True
    # HuggingFace model ID or absolute local path to model weights.
    # MiniCPM-V-2 (2.4 B) is the sweet spot for sub-second crops.
    minicpm_model_name: str = "openbmb/MiniCPM-V-2"
    # CUDA device for direct inference (should match escalation GPU).
    minicpm_device: str = "cuda:1"
    # Torch dtype: "bfloat16" | "float16" | "int4" (requires bitsandbytes).
    minicpm_dtype: str = "bfloat16"
    # Hard limit on crop longest side (px) before sending to the model.
    # Keeps vision-token count low — essential for sub-second latency.
    minicpm_max_crop_side: int = 512
    # Maximum tokens to generate per crop.
    # Single text line ≈ 15-40 tokens; 3 merged lines ≤ 120 tokens.
    minicpm_max_tokens: int = 128
    # Per-crop timeout (ms). Falls back to Ollama / RapidOCR if exceeded.
    minicpm_crop_timeout_ms: float = 2_000.0
    # Whether to allow Ollama as fallback when direct inference fails/unavailable.
    minicpm_ollama_fallback: bool = True
    # Crop-specific Ollama num_ctx/num_predict (lower than full-page values).
    minicpm_ollama_num_ctx: int = 1024
    minicpm_ollama_num_predict: int = 100
    # JPEG quality for crop encoding before VLM (can be lower than full-page).
    minicpm_jpeg_quality: int = 92

    # ZIP batch (A2)
    zip_order_images_first: bool = True  # szybkie pliki → postęp unit=files
    zip_parallel_workers: int = 0  # 0 = auto: 1 worker / GPU w puli bulk (semafory VRAM)
    # Pin each ZIP member (image/PDF) to one GPU via round-robin
    # (`member_index % len(bulk_gpus)`). Stops 2+ workers from stacking on GPU 0
    # for single-page PDF archives (OpenPDF-MultiReceipt).
    zip_pin_member_to_gpu: bool = True
    # Cap workers when the archive contains ONLY PDFs (VRAM-heavy).
    # 0 = no override (use ``ZIP_PARALLEL_WORKERS`` / GPU count).
    zip_pdf_max_parallel: int = 0
    # Fast path for 1-page receipt PDFs inside ZIP (OpenPDF-MultiReceipt).
    zip_simple_receipt_enabled: bool = True
    zip_simple_pdf_dpi: int = 200
    # 1–N page PDFs in ZIP can use zip_simple (no segmentation); 3+ → bulk (e.g. SKMBT).
    zip_simple_receipt_max_pages: int = 2
    # When true, render once and count segmented regions; > max → full bulk pipeline.
    zip_simple_receipt_probe_regions: bool = False
    zip_simple_receipt_max_regions: int = 2
    # One enhanced in-memory retry when RapidOCR returns no text (zip_simple only).
    zip_simple_empty_retry: bool = True
    # Per-member wall-clock limit in the ZIP process pool (seconds).
    zip_member_timeout_sec: float = 600.0
    # Grace period when shutting down the ZIP process pool after all work ends.
    zip_pool_shutdown_grace_sec: float = 30.0
    # Recycle ZIP OCR child processes after N documents (0 = no recycle).
    max_docs_per_worker: int = 200

    # ── Adaptive ZIP queue scheduling ────────────────────────────────────────
    # Three parallel process pools (FAST / MEDIUM / ESCALATION) replace the
    # single FIFO queue so that easy files are never blocked by hard ones.
    zip_adaptive_queue_enabled: bool = True
    # Per-tier process-pool worker counts (0 = auto: 1 per GPU).
    zip_aq_fast_workers: int = 2       # JPG, 1-page receipts
    zip_aq_medium_workers: int = 1     # dense scans, multi-page PDFs
    zip_aq_escalation_workers: int = 1 # VLM-heavy premium documents
    # Classification thresholds for PDF page-count routing.
    zip_aq_fast_max_pdf_pages: int = 1    # 1-page PDF → FAST
    zip_aq_medium_max_pdf_pages: int = 4  # 2-4 pages → MEDIUM; >4 → ESCALATION
    # Image size thresholds (bytes) for tier routing.
    zip_aq_fast_max_image_bytes: int = 400_000   # <400 KB → FAST
    zip_aq_medium_max_image_bytes: int = 2_000_000  # <2 MB → MEDIUM; ≥2 MB → ESCALATION
    # Whether to use the difficulty_analyzer for more accurate classification.
    # Adds ~12 ms per member; turn off for very large ZIPs (>500 files).
    zip_aq_use_difficulty_probe: bool = False
    # Difficulty score thresholds for tier routing (when probe enabled).
    zip_aq_difficulty_fast: float = 25.0
    zip_aq_difficulty_medium: float = 55.0
    # Per-tier per-member wall-clock timeouts (seconds; 0 = global default).
    zip_aq_fast_timeout_sec: float = 120.0
    zip_aq_medium_timeout_sec: float = 300.0
    zip_aq_escalation_timeout_sec: float = 0.0   # 0 = use zip_member_timeout_sec

    # ── GPU-specialised scheduling ────────────────────────────────────────────
    # Separates primary OCR (RapidOCR, ZIP throughput) from escalation work
    # (VLM crop calls, premium Tesseract) so they cannot block each other.
    gpu_scheduler_enabled: bool = True
    # GPU affinity for each role.  Set to the same value on single-GPU machines.
    gpu_scheduler_primary_gpu: int = 0    # RapidOCR + batch OCR + ZIP throughput
    gpu_scheduler_escalation_gpu: int = 1  # MiniCPM-V, premium crop escalation
    # Concurrent slots per queue.  > 1 allows several VLM calls in flight.
    gpu_scheduler_primary_workers: int = 2   # OCR worker threads on primary GPU
    gpu_scheduler_escalation_workers: int = 1  # VLM workers on escalation GPU
    # Hard limit on tasks waiting in the escalation queue.  When the queue is
    # full and ``gpu_scheduler_escalation_timeout_ms`` is exceeded, the call
    # falls back to the RapidOCR result immediately (no VLM).
    gpu_scheduler_escalation_queue_depth: int = 32
    # How long a primary-OCR thread waits for an escalation slot before giving
    # up and returning the RapidOCR baseline (milliseconds, 0 = wait forever).
    gpu_scheduler_escalation_timeout_ms: float = 60_000.0
    # Emit per-queue metrics to the log every N seconds (0 = disabled).
    gpu_scheduler_metrics_interval_sec: float = 30.0

    # ── Dynamic GPU scheduler (VRAM-aware, priority-based) ────────────────────
    # Wraps the base GpuScheduler with VRAM guards, escalation throttling,
    # queue priorities, and adaptive ZIP worker allocation.
    dynamic_gpu_scheduler_enabled: bool = True
    # Primary GPU (GPU0): defer non-critical jobs when free VRAM drops below this.
    dynamic_gpu_primary_min_free_mb: float = 512.0
    # Max wait for primary VRAM to recover (ms; 0 = never wait).
    dynamic_gpu_primary_vram_wait_ms: float = 200.0
    # Escalation GPU (GPU1): block VLM when free VRAM below this threshold.
    dynamic_gpu_escalation_min_free_mb: float = 1024.0
    # Block VLM escalation when GPU1 used VRAM exceeds this percentage.
    dynamic_gpu_escalation_max_used_pct: float = 85.0
    # Throttle escalation when primary queue depth reaches this count.
    dynamic_gpu_primary_busy_queue: int = 4
    # Throttle escalation when primary in_flight >= workers × this ratio.
    dynamic_gpu_primary_busy_inflight_ratio: float = 0.75

    # ── Adaptive OCR routing (difficulty-based pipeline selection) ────────────
    # When enabled, each image/page is first analysed by the lightweight
    # DifficultyAnalyzer (<20 ms) and routed to the cheapest pipeline tier
    # that can handle its complexity.  Falls back to the static pipeline when
    # the analyzer fails.
    adaptive_routing_enabled: bool = True
    # Score thresholds separating the four tiers (0–100).
    # Everything below `tier1` → RapidOCR-only (fastest).
    # tier1..tier2  → RapidOCR + Tesseract line verification (no VLM).
    # tier2..tier3  → RapidOCR + crop escalation (VLM allowed).
    # tier3..100    → premium escalation (all features, extra VLM budget).
    adaptive_tier1_threshold: float = 30.0
    adaptive_tier2_threshold: float = 60.0
    adaptive_tier3_threshold: float = 80.0
    # Maximum number of VLM calls per tier (0 = disabled).
    adaptive_tier0_vlm_calls: int = 0
    adaptive_tier1_vlm_calls: int = 0
    adaptive_tier2_vlm_calls: int = 3
    adaptive_tier3_vlm_calls: int = 8
    # Tesseract line-verification budget per tier (0 = unlimited).
    adaptive_tier0_tesseract_budget: int = 0
    adaptive_tier1_tesseract_budget: int = 6
    adaptive_tier2_tesseract_budget: int = 10
    adaptive_tier3_tesseract_budget: int = 0   # unlimited at premium tier
    # Probe DPI used by the analyzer (low → fast; 72 dpi is enough for heuristics).
    adaptive_probe_dpi: int = 72
    # Use fast=True for the difficulty probe (skips skew + handwriting estimate).
    # Saves ~6 ms per image at the cost of less skew information.
    adaptive_probe_fast: bool = False
    # Log routing decisions at DEBUG level (set True for verbose diagnostics).
    adaptive_routing_log_decisions: bool = True

    # ── 3-tier OCR difficulty classifier (ocr_difficulty_analyzer) ───────────
    # Simpler EASY / MEDIUM / HARD view on top of the 0-100 difficulty score.
    # When True, _process_single_image classifies each document before OCR and
    # selects a matching PipelineProfile.  EASY docs skip Tesseract, VLM and
    # heavy segmentation entirely — maximum throughput for clean documents.
    #   score < easy_max   → EASY   (RapidOCR only)
    #   score < hard_min   → MEDIUM (lightweight segmentation + optional Tesseract)
    #   score ≥ hard_min   → HARD   (full pipeline, VLM, validators)
    ocr_difficulty_enabled: bool = True
    ocr_difficulty_easy_max: float = 30.0   # scores below this → EASY
    ocr_difficulty_hard_min: float = 65.0   # scores from here up → HARD

    # ── Block-aware reading order (ocr_reading_order) ─────────────────────────
    # When True, replaces the legacy y-sort+x-sort in _reconstruct_lines with
    # a multi-stage pipeline: block detection → column detection → form mode →
    # adaptive line clustering → block-aware reading order.
    # Eliminates: cross-column merges, broken invoices, checkbox corruption,
    # tile-overlap duplicates, postal-notice garbling.
    advanced_reading_order_enabled: bool = True
    # Vertical gap (relative to local median text height) that splits one
    # text block from the next (e.g. section header → item list).
    # Increase to keep more content in one block; decrease to split sooner.
    reading_order_block_gap_ratio: float = 1.5
    reading_order_block_gap_min_px: int = 8
    # Adaptive y-tolerance for line clustering inside each block × column.
    # tol = max(min_px, local_median_h × ratio).  Smaller = stricter lines.
    reading_order_adaptive_y_tol_ratio: float = 0.50
    reading_order_adaptive_y_tol_min_px: int = 4
    # Multiply adaptive tolerance by this factor in form / table mode (tighter).
    reading_order_form_tol_factor: float = 0.70
    # Minimum horizontal gap (fraction of page width) between text columns.
    reading_order_column_gap_pct: float = 0.04

    # ── OCR noise classifier (ocr_noise_classifier) ─────────────────────────
    # Multi-score garbage detection BEFORE line merge.  Replaces simple
    # spurious-fragment heuristics when enabled.
    ocr_noise_classifier_enabled: bool = True
    # Adaptive noise_score thresholds per difficulty tier.
    # Lower = stricter (filter more).  EASY docs get stricter filtering.
    ocr_noise_threshold_easy: float = 0.42
    ocr_noise_threshold_medium: float = 0.55
    ocr_noise_threshold_hard: float = 0.68

    # ── Global OCR quality pipeline (ocr_quality_pipeline) ───────────────────
    # Computes a 0–100 quality_score after OCR and recommends recovery actions.
    ocr_quality_pipeline_enabled: bool = True
    ocr_quality_threshold_excellent: float = 90.0
    ocr_quality_threshold_acceptable: float = 75.0
    ocr_quality_threshold_suspicious: float = 50.0
    # Escalate to MiniCPM / recovery when SUSPICIOUS and score below this.
    ocr_quality_escalate_below: float = 60.0

    # OOM resilience — detect CUDA / cuBLAS / cuDNN OOM and retry once after
    # ``torch.cuda.empty_cache()``. Prevents one VRAM spike from killing a
    # 1000-file ZIP job.
    ocr_oom_retry: bool = True
    ocr_oom_retry_max: int = 1

    # Disk-backed OCR cache (SHA1 of image bytes + preprocessing variant)
    ocr_cache_enabled: bool = True
    ocr_cache_dir: Path = Path("storage/ocr_cache")
    ocr_cache_max_bytes: int = 2 * 1024**3  # 2 GB LRU eviction

    # Queue mode selection:
    #   auto  -> use queue when Redis is reachable, otherwise sync fallback
    #   queue -> always require Redis/RQ
    #   sync  -> always run OCR in request process
    # Empty / invalid value falls back to legacy ``use_background_worker``.
    job_execution_mode: str = ""

    # Queue (A2) — Redis + RQ; API enqueues, worker runs OCR
    redis_url: str = "redis://127.0.0.1:6379/0"
    rq_queue_name: str = "default"
    rq_job_timeout: int = 86400  # seconds (24h)
    use_background_worker: bool = False

    # ── propOS invoice API (sync POST → JSON) ───────────────────────────────
    invoice_max_upload_bytes: int = 20 * 1024 * 1024  # 20 MB → 413
    invoice_process_timeout_sec: float = 40.0  # whole request (OCR + LLM)
    invoice_text_model: str = "qwen2.5:32b"
    invoice_llm_timeout_sec: float = 90.0
    invoice_llm_num_ctx: int = 16384
    invoice_max_ocr_chars: int = 28_000
    invoice_raw_text_max_chars: int = 2500  # §4.1 raw_text cap in API response
    invoice_allow_vision_escalation: bool = False  # keep within 30–40 s budget
    ocr_service_token: str = ""  # optional Bearer; empty = auth disabled

    def ensure_dirs(self) -> None:
        for path in (self.uploads_dir, self.results_dir, self.temp_dir):
            path.mkdir(parents=True, exist_ok=True)
        cache = self.ocr_cache_dir
        if not cache.is_absolute():
            cache = self.project_root / cache
        cache.mkdir(parents=True, exist_ok=True)

    def effective_job_execution_mode(self) -> str:
        """
        Resolve runtime execution mode with backward compatibility.

        Priority:
          1) ``job_execution_mode`` env value (auto|sync|queue)
          2) legacy ``use_background_worker`` boolean
        """
        mode = (self.job_execution_mode or "").strip().lower()
        if mode in {"auto", "sync", "queue"}:
            return mode
        return "queue" if self.use_background_worker else "sync"


settings = Settings()
