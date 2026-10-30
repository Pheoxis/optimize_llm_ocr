"""PDF → page images (pypdfium2)."""

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def pdf_page_count(pdf_path: Path) -> int:
    """Fast page count without rendering (ZIP simple-receipt routing)."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def render_pdf_page_pil(
    pdf_path: Path,
    page_index: int = 0,
    *,
    dpi: int = 200,
) -> Image.Image:
    """Render a single PDF page to PIL (no PNG on disk)."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_index < 0 or page_index >= len(pdf):
            raise ValueError(f"Page index {page_index} out of range for {pdf_path.name}")
        page = pdf[page_index]
        bitmap = page.render(scale=dpi / 72.0)
        return bitmap.to_pil().convert("RGB")
    finally:
        pdf.close()


def render_pages(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    paths: list[Path] = []

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            out_path = output_dir / f"page_{index + 1:04d}.png"
            pil_image.save(out_path, format="PNG")
            paths.append(out_path)
    finally:
        pdf.close()

    if not paths:
        raise ValueError(f"PDF has no pages: {pdf_path}")
    return paths


def render_pages_routed(
    pdf_path: Path,
    output_dir: Path,
) -> tuple[list[Path], dict]:
    """
    Render all pages using heuristic DPI routing.

    Returns ``(paths, dpi_routing_metadata)``.
    """
    from app.services.pdf_dpi_routing import analyze_pdf_dpi

    analysis = analyze_pdf_dpi(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if analysis.uniform_dpi:
        paths = render_pages(
            pdf_path,
            output_dir,
            dpi=analysis.document_decision.dpi,
        )
    else:
        for outline in analysis.pages:
            dpi = analysis.page_decisions[outline.index].dpi
            pil = render_pdf_page_pil(
                pdf_path, page_index=outline.index, dpi=dpi,
            )
            out_path = output_dir / f"page_{outline.index + 1:04d}.png"
            pil.save(out_path, format="PNG")
            pil.close()
            paths.append(out_path)

    if not paths:
        raise ValueError(f"PDF has no pages: {pdf_path}")
    return paths, analysis.to_dict()
