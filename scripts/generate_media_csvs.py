#!/usr/bin/env python
"""
Genera CSVs para imágenes y audios usando los contenidos de:
  - backend/data/images
  - backend/data/audios

Salida:
  - backend/data/csv/images.csv   (id,title,image_path,image_text)
  - backend/data/csv/audios.csv   (audio_id,file_name,file_path,tfidf_vector)
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "backend" / "data" / "images"
AUDIO_DIR = ROOT / "backend" / "data" / "audios"
OUT_DIR = ROOT / "backend" / "data" / "csv"


def gen_images_csv() -> Path:
    out = OUT_DIR / "images.csv"
    rows = []
    for i, f in enumerate(sorted(IMG_DIR.rglob("*")), start=1):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
            continue
        rows.append(
            {
                "id": i,
                "title": f.stem,
                "image_path": str(f.resolve()),
                "image_text": "",
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["id", "title", "image_path", "image_text"])
        w.writeheader()
        w.writerows(rows)
    print(f"[images] {len(rows)} filas -> {out}")
    return out


def gen_audios_csv() -> Path:
    out = OUT_DIR / "audios.csv"
    rows = []
    for f in sorted(AUDIO_DIR.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".mp3", ".wav", ".flac", ".ogg"}:
            continue
        rows.append(
            {
                "audio_id": len(rows) + 1,
                "file_name": f.name,
                "file_path": str(f.resolve()),
                "tfidf_vector": "",
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["audio_id", "file_name", "file_path", "tfidf_vector"])
        w.writeheader()
        w.writerows(rows)
    print(f"[audios] {len(rows)} filas -> {out}")
    return out


def main():
    gen_images_csv()
    gen_audios_csv()


if __name__ == "__main__":
    main()
