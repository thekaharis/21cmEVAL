"""Build docs/collage.png, the README header: eval21 figures spread like printouts on a table.

    python docs/readme_collage.py FIGURE_DIR [--out docs/collage.png]

FIGURE_DIR holds the PNGs of an `eval21 plot all` run (the gallery page in FIGURE_DIR/gallery/).
Cards are listed back to front, axis-aligned (angle 0); positions are fractions of the canvas.
Needs Pillow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

CANVAS = (2800, 1800)
#        file (glob, first match)          width  centre x  centre y  angle
CARDS = [("cyl_error_tb.png",              1060,   0.24,     0.24,     0.0),
         ("bubble_size_distribution.png",   880,   0.77,     0.24,     0.0),
         ("ps_overlay_tb.png",             1000,   0.76,     0.53,     0.0),
         ("physical_pdfs.png",             1100,   0.27,     0.72,     0.0),
         ("ps_stage_curves_tb.png",         700,   0.80,     0.79,     0.0),
         ("gallery/cone_*.png",            1250,   0.48,     0.45,     0.0),
         ("parity_overlay*.png",            420,   0.58,     0.80,     0.0)]

PAPER_MARGIN = 0.022          # white border around each figure, fraction of the card width
SHADOW_OFFSET = (16, 24)
SHADOW_BLUR = 22
SHADOW_ALPHA = 0.42


def card(path: Path, width: int, angle: float) -> tuple[Image.Image, Image.Image]:
    """The figure on a sheet of paper, rotated; returns (card RGBA, shadow RGBA) of equal size."""
    fig = Image.open(path).convert("RGB")
    fig = fig.resize((width, round(fig.height * width / fig.width)), Image.LANCZOS)
    m = max(8, round(PAPER_MARGIN * width))
    sheet = Image.new("RGBA", (fig.width + 2 * m, fig.height + 2 * m), (255, 255, 255, 255))
    sheet.paste(fig, (m, m))
    ImageDraw.Draw(sheet).rectangle([0, 0, sheet.width - 1, sheet.height - 1], outline=(214, 214, 214, 255))
    pad = SHADOW_BLUR * 3 + max(SHADOW_OFFSET)
    framed = Image.new("RGBA", (sheet.width + 2 * pad, sheet.height + 2 * pad), (0, 0, 0, 0))
    framed.paste(sheet, (pad, pad))
    framed = framed.rotate(angle, resample=Image.BICUBIC, expand=False)
    alpha = framed.getchannel("A")
    shadow_alpha = Image.new("L", framed.size, 0)
    shadow_alpha.paste(alpha.point(lambda a: int(a * SHADOW_ALPHA)), SHADOW_OFFSET)
    shadow = Image.new("RGBA", framed.size, (20, 20, 28, 0))
    shadow.putalpha(shadow_alpha.filter(ImageFilter.GaussianBlur(SHADOW_BLUR)))
    return framed, shadow


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figures", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("collage.png"))
    ap.add_argument("--width", type=int, default=1600, help="output width in px")
    args = ap.parse_args(argv)

    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))     # transparent: works on light and dark pages
    for pattern, width, cx, cy, angle in CARDS:
        matches = sorted(args.figures.glob(pattern))
        if not matches:
            print(f"skipped (not found): {pattern}")
            continue
        front, shadow = card(matches[0], width, angle)
        xy = (round(cx * CANVAS[0] - front.width / 2), round(cy * CANVAS[1] - front.height / 2))
        _paste(canvas, shadow, xy)
        _paste(canvas, front, xy)
    bbox = canvas.getchannel("A").point(lambda a: 255 if a > 4 else 0).getbbox()
    canvas = canvas.crop(bbox)
    canvas = canvas.resize((args.width, round(canvas.height * args.width / canvas.width)), Image.LANCZOS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print("wrote", args.out, canvas.size)


def _paste(canvas: Image.Image, layer: Image.Image, xy) -> None:
    """alpha_composite that tolerates layers hanging over the canvas edge."""
    x, y = xy
    left, top = max(0, -x), max(0, -y)
    right = min(layer.width, canvas.width - x)
    bottom = min(layer.height, canvas.height - y)
    if right > left and bottom > top:
        canvas.alpha_composite(layer.crop((left, top, right, bottom)), dest=(x + left, y + top))


if __name__ == "__main__":
    main()
