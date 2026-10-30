"""Column-aware reading order reconstruction."""

from __future__ import annotations

from app.services.layout.types import LayoutBox


def reading_order_indices(boxes: list[LayoutBox]) -> list[int]:
    """Return original indices in column-major top-down, left-to-right order.

    Skipped:
      * boxes marked ``suppressed`` (duplicate)
      * boxes marked ``is_barcode`` (handled separately by the caller)
    """
    if not boxes:
        return []

    enumerated = [
        (i, b)
        for i, b in enumerate(boxes)
        if not b.suppressed and not b.is_barcode and b.text
    ]
    enumerated.sort(
        key=lambda item: (
            item[1].column,
            item[1].line,
            item[1].bbox[1],
            item[1].bbox[0],
        ),
    )

    order = 0
    for _, b in enumerated:
        b.reading_order = order
        order += 1
    return [i for i, _ in enumerated]


def lines_from_reading_order(
    boxes: list[LayoutBox],
    indices: list[int],
    *,
    separator: str = "    ",
    dedupe_intra_line: bool = False,
) -> list[str]:
    """Group ordered boxes by (column, line) and join with the separator.

    ``dedupe_intra_line`` defaults to False — form labels are often repeated
    on purpose at different x positions. Use overlap-based dedup at box merge
    time instead (see :mod:`text_dedup`).
    """
    if not indices:
        return []
    out: list[str] = []
    current_key: tuple[int, int] | None = None
    current_parts: list[str] = []
    for i in indices:
        b = boxes[i]
        key = (b.column, b.line)
        if key != current_key:
            if current_parts:
                out.append(separator.join(current_parts))
            current_parts = []
            current_key = key
        token = b.text.strip()
        if not token:
            continue
        if dedupe_intra_line and current_parts:
            from app.services.layout.text_dedup import texts_exact_same

            if texts_exact_same(current_parts[-1], token):
                continue
        current_parts.append(token)
    if current_parts:
        out.append(separator.join(current_parts))
    return out
