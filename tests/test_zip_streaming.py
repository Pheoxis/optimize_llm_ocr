"""Streaming ZIP: plan from index, per-member extract and cleanup."""

from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image

from app.services.zip_streaming import (
    cleanup_zip_member_extract,
    extract_zip_member,
    plan_zip_members_from_archive,
    release_zip_member_temp,
)


def _make_zip(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            img = path.parent / name
            img.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img)
            zf.write(img, arcname=name)


def test_plan_without_extracting_bodies(tmp_path: Path) -> None:
    zip_path = tmp_path / "stream.zip"
    _make_zip(zip_path, ["a.png", "b.png"])
    extract_root = tmp_path / "extracted"
    members = plan_zip_members_from_archive(zip_path, extract_root)
    assert len(members) == 2
    assert not any(m.path.is_file() for m in members)


def test_extract_one_member_and_cleanup(tmp_path: Path) -> None:
    zip_path = tmp_path / "one.zip"
    _make_zip(zip_path, ["x.png", "y.png"])
    extract_root = tmp_path / "extracted"
    members = plan_zip_members_from_archive(zip_path, extract_root)
    target = next(m for m in members if m.rel_path == "x.png")

    path = extract_zip_member(zip_path, extract_root, "x.png")
    assert path.is_file()
    assert not (extract_root / "y.png").exists()

    release_zip_member_temp(extract_root, tmp_path / "work", target)
    assert not path.is_file()


def test_cleanup_prunes_empty_dirs(tmp_path: Path) -> None:
    zip_path = tmp_path / "nested.zip"
    _make_zip(zip_path, ["sub/doc.png"])
    extract_root = tmp_path / "extracted"
    members = plan_zip_members_from_archive(zip_path, extract_root)
    member = members[0]
    extract_zip_member(zip_path, extract_root, member.rel_path)
    sub = extract_root / "sub"
    assert sub.is_dir()
    cleanup_zip_member_extract(extract_root, member)
    assert not member.path.is_file()
    assert not sub.exists()
