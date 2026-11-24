"""Diffuse colour, tangent normals and roughness for embossed polyethylene envelopes."""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
from materials import output, relief
from PIL import Image, ImageDraw, ImageFont


def shipping_print(w, h, ident):
    # Matte paper with a dense shipping manifest and two machine-readable fields.
    im = Image.new("RGB", (w, h), (227, 225, 221))
    d = ImageDraw.Draw(im)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    small = ImageFont.truetype(str(font_path), 12)
    big = ImageFont.truetype(str(font_path), 22)
    ink = (75, 72, 76)
    rng = np.random.default_rng(231 + ident)
    d.text((13, 9), "EXPRESS    " + ("P02", "B14", "W07")[ident], font=big, fill=ink)
    d.line((12, 40, w - 12, 40), fill=ink, width=2)
    for x, y, bw, bh in (
        (15, 51, w // 2 - 17, 40),
        (w // 2 + 14, 52, w // 2 - 30, 28),
        (14, h - 40, w - 29, 26),
    ):
        while bw > 4:
            bar = int(rng.choice([1, 2, 2, 3, 4]))
            d.rectangle((x, y, x + bar - 1, y + bh), fill=ink)
            step = bar + int(rng.integers(1, 4))
            x += step
            bw -= step
    for k, line in enumerate(
        (
            "SORTING STATION    ROUTE 03",
            "PARCEL ID    0842 5170 029",
            "STANDARD SERVICE       1 PC",
        )
    ):
        d.text((16, 98 + k * 16), line, font=small, fill=ink)
    d.rectangle((w - 84, 97, w - 19, 143), outline=ink, width=2)
    d.text((w - 75, 104), "21", font=big, fill=ink)
    return np.asarray(im)[:, :, ::-1].copy()


def make_white_maps():
    """Bake the white laminate. RGB rough maps pack roughness in R, metallic in G."""
    w, h = 1024, 800
    y, x = np.mgrid[:h, :w].astype(np.float32)
    rng = np.random.default_rng(70417)

    def field(rows, cols):
        knots = rng.standard_normal((rows, cols)).astype(np.float32)
        return cv.resize(knots, (w, h), interpolation=cv.INTER_CUBIC)

    swell = field(6, 8)
    mottle = field(20, 27)
    fine = field(140, 165)
    left = x - 13
    right = w - 14 - x
    top = y - 18
    bottom = h - 23 - y
    edge = np.minimum.reduce([left, right, top, bottom])
    seam = edge + 1.9 * swell + 0.9 * mottle
    body = np.clip(seam / 20, 0, 1)
    shoulder = np.exp(-np.maximum(seam, 0) / 123)

    # Shallow, distorted staggered embossing under the skin. Broader buckles
    # carry the reflection when the product occupies only ~180 camera pixels.
    u = (x + 5.4 * swell + 1.9 * mottle) / 28
    v = (y + 4.3 * field(7, 8) + 2.0 * mottle) / 23
    du = ((u + 0.5 * (np.floor(v) % 2)) % 1 - 0.5) / 0.48
    dv = (v % 1 - 0.5) / 0.47
    emboss = np.maximum(1 - du**2 - dv**2, 0) ** 1.6
    height = 4.0 * swell + 2.3 * mottle + (0.90 + 0.35 * shoulder) * emboss

    # Fans of folds emanate from pinches along each weld. Distances and depth
    # are in texels. An asymmetric shoulder lets a strip light catch one flank.
    meander = 4.0 * swell + 1.8 * mottle
    for oriented_height, oriented_meander, inset in (
        (height, meander, 18),
        (height[::-1], meander[::-1], 22),
        (height.T, meander.T, 13),
        (height.T[::-1], meander.T[::-1], 13),
    ):
        span = oriented_height.shape[1]
        for anchor in np.linspace(0.06, 0.94, 6) * span:
            anchor += rng.uniform(-35, 35)
            lean = rng.uniform(-0.9, 0.9)
            for branch in (-1, 0, 1):
                reach = rng.uniform(90, 165) * (1.24 if branch == 0 else 0.72)
                width = rng.uniform(6, 12) if branch == 0 else rng.uniform(3, 6)
                slope = lean + branch * rng.uniform(0.21, 0.49)
                bend = rng.uniform(-0.28, 0.28)
                # Work inside each fan's footprint only.
                d_max = 3.3 * reach
                extent = (abs(slope) + abs(bend)) * d_max + 7 * width + 32
                i0, i1 = (
                    max(0, int(anchor - extent)),
                    min(span, int(anchor + extent + 1)),
                )
                depth = min(oriented_height.shape[0], int(d_max + inset + 1))
                roi = np.s_[:depth, i0:i1]
                along = np.maximum(
                    np.arange(depth, dtype=np.float32)[:, None] - inset, 0
                )
                across = (
                    np.arange(i0, i1, dtype=np.float32)
                    - anchor
                    - along * slope
                    - bend * along * np.sin(along / reach)
                    - oriented_meander[roi]
                )
                q = across / (width * (0.85 + 0.45 * along / reach))
                envelope = np.exp(-0.5 * (along / reach) ** 2)
                fold = 0.60 * np.exp(-0.5 * ((q - 1.50) / 1.35) ** 2) - np.exp(
                    -0.5 * q**2
                )
                oriented_height[roi] += (
                    width * rng.uniform(0.68, 1.12) * fold * envelope
                )

    # Ribbon of compressed polymer and shallow puckers outside the seal.
    weld = np.exp(-0.5 * ((seam - 1.5) / 3.3) ** 2)
    perimeter = np.where(np.minimum(left, right) < np.minimum(top, bottom), y, x)
    crimp = np.sin(perimeter * 0.23 + 0.6 * mottle + seam * 0.09)
    height = body * height + (1 - body) * (1.55 * mottle + 0.74 * crimp) - 2.1 * weld
    # Roll both shoulders into the weld. The tilted normals leave the panel
    # reflection, while individual folds briefly turn back towards the light.
    shoulder_width = 63 + 6 * np.tanh(swell)
    for distance in (np.minimum(left, right), np.minimum(top, bottom)):
        height -= 26 * np.exp(-0.5 * (np.maximum(distance, 0) / shoulder_width) ** 2)
    gy, gx = np.gradient(height + 0.16 * fine)

    roughness = np.clip(
        0.265 + 0.031 * mottle + 0.019 * swell - 0.029 * emboss * body + 0.047 * weld,
        0.18,
        0.41,
    )
    metallic = np.clip(0.34 + 0.052 * swell + 0.032 * mottle - 0.06 * weld, 0.18, 0.49)

    # Fade from the reflective shoulder onto the flatter, whiter face.
    # Set the two insets independently because the UV rectangle is not square.
    def interior_weight(distance, inset, feather):
        t = np.clip((distance - inset) / feather, 0, 1)
        return t * t * (3 - 2 * t)

    core = interior_weight(np.minimum(left, right), 108, 90) * interior_weight(
        np.minimum(top, bottom), 112, 86
    )
    core_metal = np.clip(0.16 + 0.20 * (metallic - 0.34), 0.13, 0.19)
    metallic += core * (core_metal - metallic)
    roughness += 0.05 * core
    # Nearly uniform neutral-white pigment. No illumination encoded in albedo.
    pigment = np.array([235, 233, 236], dtype=np.float32) + (0.85 * swell)[..., None]
    bw, bh = 446, 232
    x0, y0 = (w - bw) // 2 - 13, (h - bh) // 2 - 5
    paper = np.zeros((h, w), np.float32)
    paper[y0 : y0 + bh, x0 : x0 + bw] = 1
    ink = np.zeros((h, w, 3), np.uint8)
    ink[y0 : y0 + bh, x0 : x0 + bw] = shipping_print(bw, bh, 2)
    # Keep the paper in the UV footprint used by parcel_mask's camera response.
    rot = cv.getRotationMatrix2D((w / 2, h / 2), -1.7, 1)
    paper = cv.warpAffine(paper, rot, (w, h))
    # The warped colour already includes coverage at the edge (black outside).
    ink = cv.warpAffine(ink, rot, (w, h))
    edge_y, edge_x = np.gradient(0.55 * cv.GaussianBlur(paper, (0, 0), 1.25))
    for labeled in (0, 1):
        name = f"pearl_white_{labeled}"
        cover = labeled * paper
        diffuse = pigment * (1 - cover[..., None]) + labeled * ink
        rough = roughness * (1 - cover) + (0.76 + 0.009 * mottle) * cover
        metal = metallic * (1 - cover)
        # Blend slopes, not absolute heights, to avoid a vertical wall around paper.
        slope = (1 - 0.94 * cover) - 0.24 * core * (1 - cover)
        nx = -gx * slope - labeled * edge_x
        ny = gy * slope + labeled * edge_y
        normals = np.dstack((np.ones_like(nx), ny, nx))  # OpenCV BGR; V points up.
        normals /= np.linalg.norm(normals, axis=2, keepdims=True)
        output(name, diffuse)
        output(name + "_normal", np.rint((normals * 0.5 + 0.5) * 255))
        output(
            name + "_rough",
            np.rint(np.dstack((np.zeros_like(rough), metal, rough)) * 255),
        )


def make_maps():
    w, h = 1024, 800
    y, x = np.mgrid[:h, :w].astype(np.float32)
    for kind, color in enumerate(((133, 81, 221), (212, 168, 65))):
        rng = np.random.default_rng(800 + kind)

        def field(rows, cols, rng=rng):
            return cv.resize(
                rng.normal(size=(rows, cols)).astype("float32"),
                (w, h),
                interpolation=cv.INTER_CUBIC,
            )

        broad = field(5, 7)
        mid = field(19, 26)
        fine = field(370, 510)
        sx = x + 4.2 * broad + 2.3 * mid
        sy = y + 4.0 * field(7, 9) + 1.4 * field(22, 28)
        pitch = 26.2
        row = np.floor(sy / (pitch * 0.89))
        dx = ((sx + row % 2 * pitch / 2) / pitch % 1 - 0.5) / 0.46
        dy = (sy / (pitch * 0.89) % 1 - 0.5) / 0.475
        r2 = dx**2 + dy**2
        bubbles = np.maximum(0, 1 - r2) ** 1.62
        seam = np.minimum.reduce([x - 12, w - 1 - x - 12, y - 14, h - 1 - y - 30])
        inner = np.clip(seam / 13, 0, 1)
        creases = np.zeros_like(x)
        for _ in range(26):
            x0, y0 = rng.uniform(-50, w + 50), rng.uniform(-20, h + 20)
            angle = rng.uniform(-1.1, 1.1)
            dx0, dy0 = x - x0, y - y0
            along = dx0 * np.cos(angle) + dy0 * np.sin(angle)
            perp = -dx0 * np.sin(angle) + dy0 * np.cos(angle) + 0.0010 * along**2
            length = rng.uniform(45, 170)
            gutter = rng.uniform(2.7, 6.8)
            creases += (
                rng.uniform(0.9, 3.2)
                * np.exp(-0.5 * (along / length) ** 2)
                * (
                    0.65 * np.exp(-0.5 * ((perp - 1.5 * gutter) / (1.5 * gutter)) ** 2)
                    - np.exp(-0.5 * (perp / gutter) ** 2)
                )
            )
        weld = np.exp(-(((seam - 2) / 3.6) ** 2))
        depth = (2.30, 1.70, 0.94)[kind]
        film_height = (
            (
                depth * bubbles * np.clip(0.83 + 0.19 * broad + 0.13 * mid, 0.28, 1.22)
                + creases
                + 0.52 * mid
            )
            * inner
            - 1.0 * weld
            + 0.12 * fine
        )
        base_rough = (
            (0.403, 0.438, 0.49)[kind]
            + 0.028 * mid
            + 0.025 * broad
            - 0.060 * bubbles * inner
        )
        base_rough += 0.044 * weld + 0.008 * fine
        tint = 1.9 * broad + 1.0 * mid + 2.1 * bubbles * inner + 0.7 * fine - 3.5 * weld
        rgb = np.clip(np.array(color) + tint[..., None], 0, 255).astype("uint8")
        for labeled in (0, 1):
            name = f"film{kind}_{labeled}"
            diffuse = rgb.copy()
            height = film_height.copy()
            rough = base_rough.copy()
            if labeled:
                bw, bh = 446, 232
                x0, y0 = (w - bw) // 2 - 13, (h - bh) // 2 - 5
                label = shipping_print(bw, bh, kind)
                paper = np.zeros((h, w), np.float32)
                cv.rectangle(paper, (x0 + 3, y0), (x0 + bw - 4, y0 + bh - 1), 1, -1)
                cv.rectangle(paper, (x0, y0 + 3), (x0 + bw - 1, y0 + bh - 4), 1, -1)
                printed = diffuse.copy()
                printed[y0 : y0 + bh, x0 : x0 + bw] = label
                rot = cv.getRotationMatrix2D((w / 2, h / 2), (-3.2, 2.4, -1.7)[kind], 1)
                paper = cv.warpAffine(paper, rot, (w, h))
                printed = cv.warpAffine(printed, rot, (w, h))
                diffuse = np.uint8(
                    diffuse * (1 - paper[..., None]) + printed * paper[..., None]
                )
                rough = rough * (1 - paper) + (0.64 + 0.016 * mid) * paper
                height = height * (1 - 0.86 * paper) + 0.50 * cv.GaussianBlur(
                    paper, (0, 0), 1
                )
            output(name, diffuse)
            relief(name, height, np.clip(rough, 0.23, 0.76))
    make_white_maps()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--white-only", action="store_true", help="Bake only pearl_white maps."
    )
    if parser.parse_args().white_only:
        make_white_maps()
    else:
        make_maps()
