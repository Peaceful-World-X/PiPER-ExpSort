"""Measured worktop microstructure and procedural PBR materials for padded parcels."""

from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cell import project
from wear import build_wear

A = Path(__file__).resolve().parents[1] / "assets"
T = A / "surfaces"
rng = np.random.default_rng(7281)


def noise(size, shape):
    return cv.resize(
        rng.standard_normal(shape).astype("float32"), size, interpolation=cv.INTER_CUBIC
    )


def output(name, image):
    cv.imwrite(str(T / (name + ".png")), np.uint8(np.clip(image, 0, 255)))


def relief(name, field, rough):
    gy, gx = np.gradient(field)
    normal = np.stack([-gx, gy, np.ones_like(gx)], axis=2)
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    output(name + "_normal", (normal[:, :, ::-1] * 0.5 + 0.5) * 255)
    output(name + "_rough", rough * 255)


def worktop(ref):
    size = (1536, 768)
    w, h = size
    u, v = np.meshgrid(np.linspace(0, 1, w), np.linspace(1, 0, h))
    points = np.stack(
        [-0.480 + 0.955 * u, -0.439 + 0.394 * v, np.zeros_like(u)], axis=2
    )
    uv = project(points).astype("float32")
    sample = cv.remap(
        ref, uv[:, :, 0], uv[:, :, 1], cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT
    )
    gray = cv.cvtColor(sample, cv.COLOR_BGR2GRAY).astype("float32")
    residual = np.clip(gray - cv.GaussianBlur(gray, (0, 0), 5), -12, 12)
    residual *= (
        (uv[:, :, 0] > 3)
        & (uv[:, :, 0] < 637)
        & (uv[:, :, 1] < 468)
        & (uv[:, :, 1] > 258)
    )
    coarse = noise(size, (15, 32))
    rub = noise(size, (69, 215))
    fine = noise(size, (530, 1024))
    lines = np.zeros((h, w), np.float32)
    for _ in range(290):
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        delta = rng.normal(0, [19, 13])
        cv.line(
            lines,
            (int(x), int(y)),
            (int(x + delta[0]), int(y + delta[1])),
            rng.uniform(-1, 1),
            1,
            cv.LINE_AA,
        )
    grain = rub * 0.85 + fine * 0.65 + coarse * 0.85 + residual * 0.48 + lines * 2.7
    contact = np.zeros((h, w), np.uint8)
    for _ in range(33):
        origin = (int(rng.uniform(0.18, 0.80) * w), int(rng.uniform(0.25, 0.84) * h))
        radius = (int(rng.uniform(32, 106)), int(rng.uniform(9, 48)))
        angle = float(rng.uniform(0, 360))
        cv.ellipse(
            contact,
            origin,
            radius,
            angle,
            0,
            float(rng.uniform(8, 45)),
            int(rng.uniform(43, 145)),
            1,
            cv.LINE_AA,
        )
    trail = cv.GaussianBlur(contact.astype("float32") / 255, (0, 0), 0.48)
    output(
        "mat",
        np.array([47, 82, 12]) + (grain + trail * 9.5 + coarse * 0.60)[:, :, None],
    )
    relief(
        "mat",
        (0.09 * rub + 0.042 * fine + 0.038 * residual + 0.12 * lines) * 0.105,
        np.clip(
            0.254 + 0.0045 * rub + 0.004 * coarse + 0.0013 * residual - 0.062 * trail,
            0.15,
            0.35,
        ),
    )


def parcels():
    size = (768, 600)
    w, h = size
    y, x = np.mgrid[:h, :w].astype(np.float32)
    grain = noise(size, (360, 490))
    for k, col in enumerate(
        [[147, 100, 228], [213, 173, 80], [209, 211, 211], [65, 111, 162]]
    ):
        for mark in [0, 1]:
            name = f"pack{k}_{mark}"
            coarse = noise(size, (10, 15))
            waves = noise(size, (23, 32))
            if k < 3:
                px, py = 20, 18
                u = x + 2.3 * coarse
                v = y + 2 * noise(size, (8, 10))
                nx = ((u + (np.floor(v / py) % 2) * px / 2) % px - px / 2) / (px * 0.47)
                ny = (v % py - py / 2) / (py * 0.47)
                bubbles = np.maximum(0, 1 - nx * nx - ny * ny) ** 1.8
                inset = np.minimum.reduce([x, w - 1 - x, y, h - 1 - y])
                grip = np.clip((inset - 14) / 12, 0, 1)
                folds = np.exp(-inset / 44) * np.sin(x * 0.19 + y * 0.27 + coarse * 2.5)
                height = (
                    bubbles * grip * (0.88 if k < 2 else 0.43)
                    + folds * 0.9
                    + waves * 0.26
                    + 0.05 * grain
                )
                vals = (
                    coarse * 2.5
                    + waves * 1.7
                    + grain * 0.95
                    + bubbles * 2.3
                    + folds * 4.2
                )
                rough = (0.43 if k == 2 else 0.38) + waves * 0.025 + grain * 0.009 - 0.045 * bubbles
            else:
                height = 0.14 * grain + 0.07 * waves
                vals = grain * 3.2 + coarse * 2
                rough = 0.64 + waves * 0.023
            color = np.clip(np.array(col) + vals[:, :, None], 0, 255).astype("uint8")
            if k == 3:
                color[:, w // 2 - 38 : w // 2 + 38] = np.clip(
                    color[:, w // 2 - 38 : w // 2 + 38].astype(float) * 1.12, 0, 255
                )
                rough[:, w // 2 - 38 : w // 2 + 38] = 0.32
            if mark:
                im = Image.fromarray(color[:, :, ::-1])
                d = ImageDraw.Draw(im)
                bx, by, bw, bh = 232, 207, 305, 177
                d.rounded_rectangle(
                    (bx, by, bx + bw, by + bh), radius=3, fill=(232, 231, 226)
                )
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
                bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 19)
                d.text(
                    (bx + 11, by + 6),
                    "SORT  /  " + str(17 + k),
                    fill=(45, 45, 41),
                    font=bold,
                )
                d.text(
                    (bx + 11, by + 35),
                    "ROBOTICS LAB     TRANSFER SAMPLE",
                    fill=(57, 54, 52),
                    font=font,
                )
                for yy in [58, 119, 149]:
                    d.line(
                        (bx + 8, by + yy, bx + bw - 8, by + yy),
                        fill=(86, 84, 80),
                        width=1,
                    )
                xx = bx + 12
                while xx < bx + bw - 17:
                    wid = int(rng.integers(1, 5))
                    d.rectangle((xx, by + 65, xx + wid, by + 106), fill=(46, 44, 44))
                    xx += wid + int(rng.integers(1, 4))
                d.text(
                    (bx + 12, by + 125),
                    "CELL 02       A-0" + str(k) + "      CONVEYOR",
                    fill=(61, 58, 56),
                    font=font,
                )
                color = np.asarray(im)[:, :, ::-1]
                rough[by : by + bh + 1, bx : bx + bw + 1] = 0.51
                height[by : by + bh + 1, bx : bx + bw + 1] *= 0.15
            output(name, color)
            relief(name, height * 0.32, np.clip(rough, 0.13, 0.82))


if __name__ == "__main__":
    T.mkdir(exist_ok=True)
    worktop(cv.imread(str(A / "empty-study.jpg")))
    parcels()
    from padded_film import make_maps

    make_maps()
    build_wear(output, relief)
