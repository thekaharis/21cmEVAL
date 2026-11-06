"""Build docs/collage.png, the README header: eval21 figures spread like printouts on a table.

    python -m eval21.visualization.collage FIGURE_DIR [--out docs/collage.png]

FIGURE_DIR holds the PNGs of an `eval21 plot all` run (the gallery page in FIGURE_DIR/gallery/).
Cards are listed back to front; positions are fractions of the canvas, angles in degrees.
Needs Pillow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

CANVAS = (2800, 1750)
#        file (glob, first match)          width  centre x  centre y  angle
CARDS = [("cyl_error_tb.png",              1080,   0.28,     0.30,    -5.0),
         ("bubble_size_distribution.png",   920,   0.74,     0.29,     4.5),
         ("ps_overlay_tb.png",              720,   0.78,     0.70,    -3.5),
         ("physical_pdfs.png",             1140,   0.34,     0.71,     2.5),
         ("gallery/cone_*.png",            1300,   0.53,     0.47,    -1.0),
         ("parity_overlay*.png",            430,   0.635,    0.76,     5.5)]

PAPER_MARGIN = 0.022          # white border around each figure, fraction of the card width
SHADOW_OFFSET = (16, 24)
SHADOW_BLUR = 22
SHADOW_ALPHA = 0.42


def card(path: Path, width: int, angle: float) -> tuple:
    """The figure on a sheet of paper, rotated; returns (card RGBA, shadow RGBA) of equal size."""
    from PIL import Image, ImageDraw, ImageFilter

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
    ap.add_argument("--out", type=Path, default=Path("docs/collage.png"))
    ap.add_argument("--width", type=int, default=1600, help="output width in px")
    args = ap.parse_args(argv)
    from PIL import Image

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


def _paste(canvas, layer, xy) -> None:
    """alpha_composite that tolerates layers hanging over the canvas edge."""
    x, y = xy
    left, top = max(0, -x), max(0, -y)
    right = min(layer.width, canvas.width - x)
    bottom = min(layer.height, canvas.height - y)
    if right > left and bottom > top:
        canvas.alpha_composite(layer.crop((left, top, right, bottom)), dest=(x + left, y + top))


if __name__ == "__main__":
    main()
