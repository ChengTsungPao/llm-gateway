"""Generate the app icons in llm_gateway/static/icons/ (run once; the PNGs are committed).

usage: uv run --with pillow python scripts/make_icons.py

The mark: a 2x2 grid, bright on one diagonal and dim on the other — the same "▚" mark as
the UI's top-left logo — on a dark rounded tile. A "maskable" variant keeps the mark inside
the central safe zone so Android launchers can crop it to any shape.
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "llm_gateway" / "static" / "icons"
BG = (15, 20, 18, 255)        # matches the dark UI background
GREEN = (63, 191, 134, 255)   # accent
GREEN_FADED = tuple(round(g * 0.35 + b * 0.65) for g, b in zip(GREEN[:3], BG[:3])) + (255,)  # solid, no alpha
S = 1024                      # draw large, downscale for clean edges


def mark(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cell = (x1 - x0) / 2
    gap = cell * 0.12
    r = int(cell * 0.14)

    def sq(cx: int, cy: int, fill) -> None:
        draw.rounded_rectangle(
            (x0 + cx * cell + gap / 2, y0 + cy * cell + gap / 2,
             x0 + (cx + 1) * cell - gap / 2, y0 + (cy + 1) * cell - gap / 2),
            radius=r, fill=fill)

    # 2x2 grid, bright on one diagonal — the "▚" mark from the UI header
    sq(0, 0, GREEN)
    sq(1, 1, GREEN)
    sq(1, 0, GREEN_FADED)
    sq(0, 1, GREEN_FADED)


def render(maskable: bool) -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    if maskable:
        d.rectangle((0, 0, S, S), fill=BG)                   # full bleed; launcher crops
        pad = int(S * 0.27)                                  # mark stays in the 80% safe zone
    else:
        d.rounded_rectangle((0, 0, S - 1, S - 1), radius=int(S * 0.22), fill=BG)
        pad = int(S * 0.2)
    mark(d, (pad, pad, S - pad, S - pad))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    regular, maskable = render(False), render(True)
    for size in (32, 192, 512):
        regular.resize((size, size), Image.LANCZOS).save(OUT / f"icon-{size}.png")
    maskable.resize((512, 512), Image.LANCZOS).save(OUT / "icon-maskable-512.png")
    # iOS / many "add to home screen" tools: opaque square, they round the corners themselves
    maskable.resize((180, 180), Image.LANCZOS).convert("RGB").save(OUT / "apple-touch-icon.png")
    regular.save(OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
