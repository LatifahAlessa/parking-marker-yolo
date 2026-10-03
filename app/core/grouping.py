from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_ONEDRIVE_SUFFIX = re.compile(r"__OneDrive.*$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedName:
    group_id: str
    view_id: str
    canonical_stem: str


def canonical_stem(filename: str) -> str:
    stem = Path(filename).stem
    return _ONEDRIVE_SUFFIX.sub("", stem)


def parse_filename(filename: str) -> ParsedName:
    stem = canonical_stem(filename)
    if "_" not in stem:
        raise ValueError(
            f"Cannot infer group from '{filename}'. Expected a final numeric view token, "
            "for example VI_1448_121803_001.jpeg"
        )
    group_id, view_id = stem.rsplit("_", 1)
    if not view_id.isdigit() or not group_id:
        raise ValueError(
            f"Cannot infer group from '{filename}'. The final underscore-separated token "
            "must be numeric."
        )
    return ParsedName(group_id=group_id, view_id=view_id, canonical_stem=stem)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
