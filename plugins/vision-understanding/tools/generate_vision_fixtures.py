"""Generate the synthetic PNGs that back the bundled vision fixtures.

Run once from the plugin root after editing the fixtures::

    python tools/generate_vision_fixtures.py

Re-running is safe and deterministic: same byte output every time given the
same PIL version. Keep the canvases SMALL (320x80 OCR, 280x180 charts) so
each PNG stays well under 5 KB on disk.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PLUGIN_ROOT / "src" / "inferencebench_vision" / "datasets" / "images"


# (filename, rendered-text) — keep strings short; they're aligned to a small canvas.
OCR_ROWS: list[tuple[str, str]] = [
    ("ocr-01.png", "APRIL 17"),
    ("ocr-02.png", "INVOICE 4421"),
    ("ocr-03.png", "TOTAL $89.50"),
    ("ocr-04.png", "ORDER #7732"),
    ("ocr-05.png", "DUE MAY 03"),
]

# (filename, [(label, value), ...]) — bar charts: labels sit under each bar
# and values sit just above each bar.
CHART_ROWS: list[tuple[str, list[tuple[str, int]]]] = [
    ("chart-01.png", [("A", 12), ("B", 25), ("C", 42), ("D", 18)]),
    ("chart-02.png", [("A", 20), ("B", 8), ("C", 30), ("D", 15)]),
    ("chart-03.png", [("A", 10), ("B", 15), ("C", 20), ("D", 25)]),
    ("chart-04.png", [("A", 14), ("B", 27), ("C", 19), ("D", 22)]),
    ("chart-05.png", [("A", 5), ("B", 10), ("C", 20), ("D", 25)]),
]


def _font(_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Pick a small bitmap font. The PIL default works without any system deps.

    ``_size`` is accepted for callers that ask for a specific point size but
    ignored — the bitmap default keeps the wheel free of font-file deps.
    """
    try:
        return ImageFont.load_default()
    except OSError:  # pragma: no cover - default font is always available
        return ImageFont.load_default()


def render_ocr(text: str, out_path: Path) -> None:
    """Render ``text`` centered on a 320x80 white canvas, save as PNG."""
    img = Image.new("L", (320, 80), color=255)
    draw = ImageDraw.Draw(img)
    font = _font(20)
    # Pillow >=10 uses textbbox for measurement.
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((320 - tw) // 2, (80 - th) // 2), text, fill=0, font=font)
    img.save(out_path, format="PNG", optimize=True)


def render_chart(bars: list[tuple[str, int]], out_path: Path) -> None:
    """Render a small 4-bar chart with labels + values, save as PNG."""
    width, height = 280, 180
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    font = _font(12)

    # Reserve room for axes / labels.
    left_pad = 25
    right_pad = 10
    top_pad = 20
    bottom_pad = 25
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    max_v = max(v for _, v in bars) or 1
    n = len(bars)
    bar_w = plot_w // (n * 2)

    # Y axis + X axis
    draw.line(
        [(left_pad, top_pad), (left_pad, height - bottom_pad)], fill=0, width=1
    )
    draw.line(
        [
            (left_pad, height - bottom_pad),
            (width - right_pad, height - bottom_pad),
        ],
        fill=0,
        width=1,
    )

    for i, (label, value) in enumerate(bars):
        bx = left_pad + bar_w + i * (2 * bar_w)
        bh = int((value / max_v) * (plot_h - 10))
        by_top = height - bottom_pad - bh
        draw.rectangle(
            [bx, by_top, bx + bar_w, height - bottom_pad], fill=80, outline=0
        )
        # Value label just above the bar.
        v_text = str(value)
        v_bbox = draw.textbbox((0, 0), v_text, font=font)
        v_w = v_bbox[2] - v_bbox[0]
        draw.text(
            (bx + (bar_w - v_w) // 2, by_top - 14), v_text, fill=0, font=font
        )
        # Category label below the bar.
        l_bbox = draw.textbbox((0, 0), label, font=font)
        l_w = l_bbox[2] - l_bbox[0]
        draw.text(
            (bx + (bar_w - l_w) // 2, height - bottom_pad + 4),
            label,
            fill=0,
            font=font,
        )

    img.save(out_path, format="PNG", optimize=True)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in OCR_ROWS:
        render_ocr(text, IMAGES_DIR / name)
        print(f"wrote {name}")
    for name, bars in CHART_ROWS:
        render_chart(bars, IMAGES_DIR / name)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
