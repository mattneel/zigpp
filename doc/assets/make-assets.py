#!/usr/bin/env python3
"""Derives Zig++'s brand assets from the logo.

    doc/assets/make-assets.py <logo.png> [--out-root .]

The logo is the only hand-made file: an RGBA drawing on a transparent
background. Everything else is generated from it, so a new logo means
re-running this script.

Writes:
    doc/book/theme/favicon.ico          16x16, 32x32, 48x48
    doc/book/theme/favicon-32x32.png
    doc/book/theme/apple-touch-icon.png 180x180, opaque (iOS shows alpha as black)
    doc/book/theme/icon-192.png         transparent, padded
    doc/book/theme/icon-512.png         transparent, padded
    doc/book/theme/logo.png             transparent, for pages in the book
    doc/book/theme/og.png               1200x630 Open Graph card
    doc/book/theme/site.webmanifest
    doc/assets/zigpp-logo.png           header image for README.md

It also prints the palette that doc/book/theme/zigpp.css uses: the logo's
dominant colours, and the contrast ratios with which the link colours are
readable as text on the light and the dark book themes.
"""

import argparse
import base64
import io
import json
import os
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

OG_SIZE = (1200, 630)
OG_BG = "#070b22"
LIGHT_BG = "#ffffff"
DARK_BG = "#0c122c"
FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def luminance(rgb):
    def channel(value):
        value /= 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def rgb_of(text):
    return tuple(int(text[i : i + 2], 16) for i in (1, 3, 5))


def hex_of(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


def readable(rgb, background, target=4.5):
    """Darkens or lightens a colour until it can be read on the background."""
    r, g, b = rgb
    up = luminance(background) < 0.5
    while contrast((r, g, b), background) < target:
        step = 4 if up else 6
        if up:
            r, g, b = min(255, r + step), min(255, g + step), min(255, b + step)
        else:
            r, g, b = max(0, r - step), max(0, g - step), max(0, b - step)
    return (r, g, b)


def palette(art, colors=24):
    """The logo's dominant colours, most common first, background colours out."""
    rgb = art.convert("RGB")
    alpha = art.getchannel("A")
    opaque = alpha.point(lambda a: 255 if a >= 200 else 0).histogram()[255]
    quantized = rgb.quantize(colors=colors)
    table = quantized.getpalette()
    data = quantized.get_flattened_data() if hasattr(quantized, "get_flattened_data") else quantized.getdata()
    found = []
    for index, count in Counter(data).most_common():
        r, g, b = table[index * 3 : index * 3 + 3]
        luma = luminance((r, g, b))
        if luma < 0.02 or luma > 0.9:  # the dark outlines and the white eyes
            continue
        found.append(((r, g, b), count / opaque))
    return found


def trim(art, threshold=16):
    """Crops the transparent margin, ignoring stray antialiasing pixels."""
    mask = art.getchannel("A").point(lambda a: 255 if a >= threshold else 0)
    return art.crop(mask.getbbox())


def square(art, size, margin):
    """Fits the art into a transparent square, `margin` of each side empty."""
    inner = int(size * (1 - 2 * margin))
    scale = inner / max(art.width, art.height)
    scaled = art.resize(
        (max(1, round(art.width * scale)), max(1, round(art.height * scale))),
        Image.LANCZOS,
    )
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(scaled, ((size - scaled.width) // 2, (size - scaled.height) // 2), scaled)
    return canvas


def save(image, path, palette_colors=None):
    if palette_colors:
        image = image.quantize(colors=palette_colors, method=Image.FASTOCTREE)
    image.save(path, optimize=True)
    return os.path.getsize(path)


def og_card(art, wordmark, accent):
    card = Image.new("RGB", OG_SIZE, OG_BG)
    draw = ImageDraw.Draw(card)
    draw.rectangle([0, 0, OG_SIZE[0], 10], fill=accent)
    logo = square(art, 380, 0.04)
    card.paste(logo, (760, (OG_SIZE[1] - 380) // 2), logo)
    font = ImageFont.truetype(next(path for path in FONTS if os.path.exists(path)), 150)
    draw.text((90, 245), "Zig++", font=font, fill=wordmark)
    return card


def favicon_svg(art, size=48):
    """mdbook writes its own <link rel="icon"> tags for theme/favicon.svg and
    theme/favicon.png, so both hold the logo as well, next to the tags that
    theme/head.hbs adds. The artwork is a drawing, not a vector, so the SVG
    carries it as a PNG."""
    buffer = io.BytesIO()
    square(art, size, 0.04).save(buffer, format="PNG", optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode()
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {0} {0}" width="{0}" height="{0}">\n'
        '  <image width="{0}" height="{0}" href="data:image/png;base64,{1}"/>\n'
        "</svg>\n"
    ).format(size, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logo")
    parser.add_argument("--out-root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    args = parser.parse_args()

    theme = os.path.join(args.out_root, "doc", "book", "theme")
    assets = os.path.join(args.out_root, "doc", "assets")
    source = Image.open(args.logo).convert("RGBA")
    art = trim(source)

    found = palette(art)
    accent = found[0][0]
    light_links = next((rgb for rgb, _ in found if contrast(rgb, rgb_of(LIGHT_BG)) >= 4.5), readable(found[0][0], rgb_of(LIGHT_BG)))
    dark_links = next((rgb for rgb, _ in found if contrast(rgb, rgb_of(DARK_BG)) >= 7), readable(found[0][0], rgb_of(DARK_BG), 7))

    sizes = {}
    favicon = square(art, 48, 0.04)
    favicon.save(os.path.join(theme, "favicon.ico"), sizes=[(16, 16), (32, 32), (48, 48)])
    sizes["favicon.png"] = save(favicon, os.path.join(theme, "favicon.png"))
    with open(os.path.join(theme, "favicon.svg"), "w") as file:
        file.write(favicon_svg(art))
    sizes["favicon.svg"] = os.path.getsize(os.path.join(theme, "favicon.svg"))
    sizes["favicon-32x32.png"] = save(square(art, 32, 0.04), os.path.join(theme, "favicon-32x32.png"))

    touch = square(art, 180, 0.08)
    opaque = Image.new("RGBA", touch.size, OG_BG)
    opaque.paste(touch, (0, 0), touch)
    sizes["apple-touch-icon.png"] = save(opaque, os.path.join(theme, "apple-touch-icon.png"))
    sizes["icon-192.png"] = save(square(art, 192, 0.08), os.path.join(theme, "icon-192.png"))
    sizes["icon-512.png"] = save(square(art, 512, 0.08), os.path.join(theme, "icon-512.png"), palette_colors=256)
    sizes["logo.png"] = save(square(art, 360, 0.02), os.path.join(theme, "logo.png"), palette_colors=256)
    sizes["og.png"] = save(og_card(art, (255, 255, 255), accent), os.path.join(theme, "og.png"), palette_colors=256)

    with open(os.path.join(theme, "site.webmanifest"), "w") as file:
        json.dump(
            {
                "name": "Zig++",
                "short_name": "Zig++",
                "description": "A Zig superset with private fields, LLVM forever, and std.gpu.",
                "start_url": "/",
                "display": "standalone",
                "theme_color": hex_of(accent),
                "background_color": OG_BG,
                "icons": [
                    {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                    {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
                ],
            },
            file,
            indent=2,
        )
        file.write("\n")

    header = art.resize((200, max(1, round(art.height * 200 / art.width))), Image.LANCZOS)
    sizes["doc/assets/zigpp-logo.png"] = save(header, os.path.join(assets, "zigpp-logo.png"), palette_colors=256)

    print("logo %s: %dx%d, art %dx%d" % (args.logo, source.width, source.height, art.width, art.height))
    print("dominant colours:")
    for rgb, share in found[:6]:
        print(
            "  %s  %5.1f%%  on white %5.2f:1  on %s %5.2f:1"
            % (hex_of(rgb), share * 100, contrast(rgb, rgb_of(LIGHT_BG)), DARK_BG, contrast(rgb, rgb_of(DARK_BG)))
        )
    print("accent, borders and gradients: %s" % hex_of(accent))
    print("links on light themes:         %s  %5.2f:1 on white" % (hex_of(light_links), contrast(light_links, rgb_of(LIGHT_BG))))
    print("links on dark themes:          %s  %5.2f:1 on %s" % (hex_of(dark_links), contrast(dark_links, rgb_of(DARK_BG)), DARK_BG))
    print("written:")
    for name, size in sizes.items():
        print("  %-28s %7.1f KiB" % (name, size / 1024))


if __name__ == "__main__":
    main()
