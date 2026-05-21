"""Sanity checks for the bundled synthetic PNGs.

These tests don't regenerate the images — they confirm that the files
committed alongside the JSONL fixtures exist, are non-empty, parse as
valid PNGs via PIL, and stay below the 5 KB-per-file budget.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

IMAGES_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "inferencebench_vision" / "datasets" / "images"
)

EXPECTED_IMAGES = [
    "ocr-01.png",
    "ocr-02.png",
    "ocr-03.png",
    "ocr-04.png",
    "ocr-05.png",
    "chart-01.png",
    "chart-02.png",
    "chart-03.png",
    "chart-04.png",
    "chart-05.png",
]


def test_all_bundled_images_exist() -> None:
    for name in EXPECTED_IMAGES:
        path = IMAGES_DIR / name
        assert path.exists(), f"missing bundled image: {path}"
        assert path.stat().st_size > 0, f"bundled image is empty: {path}"


@pytest.mark.parametrize("name", EXPECTED_IMAGES)
def test_bundled_image_is_valid_png(name: str) -> None:
    path = IMAGES_DIR / name
    with Image.open(path) as img:
        img.verify()
    # Re-open after verify() (verify consumes the stream).
    with Image.open(path) as img:
        assert img.format == "PNG"
        assert img.size[0] > 0
        assert img.size[1] > 0


def test_bundled_images_stay_under_size_budget() -> None:
    # Each bundled PNG should stay small — they ship in the wheel.
    for name in EXPECTED_IMAGES:
        path = IMAGES_DIR / name
        size = path.stat().st_size
        assert size < 5 * 1024, f"{name} is {size} bytes (>5 KB budget)"


def test_total_images_size_is_reasonable() -> None:
    total = sum((IMAGES_DIR / name).stat().st_size for name in EXPECTED_IMAGES)
    # 10 small PNGs should not exceed 50 KB combined.
    assert total < 50 * 1024
