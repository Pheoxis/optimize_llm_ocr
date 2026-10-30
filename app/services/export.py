"""Write ProcessingResult to disk."""

import json
from pathlib import Path

from app.models.schemas import ProcessingResult


def write_results(result: ProcessingResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "result.json"
    txt_path = output_dir / "result.txt"

    json_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    txt_path.write_text(result.text, encoding="utf-8")
    return json_path, txt_path
