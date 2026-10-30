"""Merge page texts into one block."""

PAGE_SEPARATOR = "\n\n---\n\n"


def merge_page_texts(pages: list[str], include_separators: bool = True) -> str:
    if not pages:
        return ""
    if not include_separators:
        return "\n\n".join(p.strip() for p in pages if p.strip())
    return PAGE_SEPARATOR.join(p.strip() for p in pages if p.strip())
