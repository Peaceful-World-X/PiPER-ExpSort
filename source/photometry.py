"""Fit broad illumination differences on fixed surfaces and a measured camera colour response."""

from pathlib import Path
from cell import unproject
import cv2 as cv
import numpy as np


# Lift the paper-white range measured on flat reference mailers. Deep ink and
# saturated highlights stay anchored so the barcode and creases retain contrast.
PAPER_LIFT = np.interp(
    np.arange(256),
    [0, 85, 125, 155, 183, 199, 212, 235, 255],
    [0, 0, 3, 13, 32, 44, 38, 16, 0],
).astype(np.float32)

# The white laminate returns a wide bright reflection between its darker folds.
# Fit display levels on the real bag shoulder; preserve deep shadows and printed ink.
WHITE_FILM_LIFT = np.interp(
    np.arange(256),
    [0, 90, 114, 143, 156, 174, 193, 208, 225, 241, 250, 255],
    [0, 0, 5, 16, 23, 35, 37, 35, 25, 13, 4, 0],
).astype(np.float32)


class CameraResponse:
    def __init__(self, real, rendered):
        h, w = rendered.shape[:2]
        self.initial = rendered.copy()
        size = (320, 240)
        real = cv.resize(real, size, interpolation=cv.INTER_AREA).astype(np.float32)
        model = cv.resize(rendered, size, interpolation=cv.INTER_AREA).astype(
            np.float32
        )
        y, x = np.indices((240, 320))
        self.gains = []
        green = lambda im: (
            (im[:, :, 1] > im[:, :, 2] + 17) & (im[:, :, 1] > im[:, :, 0] + 9)
        )
        # Regions require both views to show the work surface at calibration time.
        # Include the two reflected lights where either image loses saturation.
        clear_belt = ((x - 149) ** 2 / 32**2 + (y - 53) ** 2 / 33**2) < 1
        clear_mat = ((x - 192) ** 2 / 59**2 + (y - 212) ** 2 / 40**2) < 1
        mask = (green(real) & green(model)) | clear_belt | clear_mat
        regions = [
            ((y > 25) & (y < 95) & mask, 4.8),
            ((y > 125) & (y < 234) & mask, 14.0),
            ((y < 18) & (x > 24) & (x < 308), 6.5),
        ]
        for support, radius in regions:
            m = support.astype(np.float32)
            blur = lambda im: cv.GaussianBlur(im, (0, 0), radius)
            weight = blur(m)
            numerator = blur(real * m[:, :, None])
            denominator = blur(model * m[:, :, None])
            ratio = (numerator + 12 * weight[:, :, None] + 0.01) / (
                denominator + 12 * weight[:, :, None] + 0.01
            )
            lo, hi = (0.66, 1.50) if radius == 6.5 else (0.77, 1.27)
            if radius == 4.8:
                lo, hi = 0.48, 1.52
            ratio = np.clip(ratio, lo, hi)
            confidence = np.clip(weight / 0.32, 0, 1)[:, :, None]
            field = (ratio - 1) * confidence
            self.gains.append(cv.resize(field, (w, h), interpolation=cv.INTER_LINEAR))
        self.y = (np.arange(h)[:, None] / h * 480).astype(np.float32)
        self.blind_window = (np.clip((43 - self.y) / 7, 0, 1)).astype(np.float32)
        self.surface = np.where(
            (self.y < 197)[:, :, None], self.gains[0], self.gains[1]
        )
        self.surface *= np.where(
            self.y < 196,
            np.clip((self.y - 47) / 6, 0, 1),
            np.clip((self.y - 249) / 9, 0, 1) * np.clip((475 - self.y) / 7, 0, 1),
        )[:, :, None]

        # Recover material detail suppressed by the low-sample animation denoiser.
        # Sample the same UV-mapped mat used by the USD shader, in its physical plane.
        yy, xx = np.indices((h, w), dtype=np.float32)
        refx = (xx + 0.5) * 640 / w - 0.5
        refy = (yy + 0.5) * 480 / h - 0.5
        self.belt_glint = np.exp(
            -(((refx - 296) / 71) ** 4) - ((refy - 106) / 83) ** 4
        ).astype(np.float32)
        pts = unproject(np.stack([refx, refy], axis=-1), 0)
        folder = Path(__file__).resolve().parents[1] / "assets/surfaces"
        tex = cv.imread(str(folder / "mat.png"), cv.IMREAD_GRAYSCALE).astype(np.float32)
        th, tw = tex.shape
        u = ((pts[:, :, 0] + 0.48) / 0.955 * (tw - 1)).astype(np.float32)
        v = ((1 - (pts[:, :, 1] + 0.439) / 0.394) * (th - 1)).astype(np.float32)
        region = (u >= 0) & (u < tw) & (v >= 0) & (v < th)
        projected = cv.remap(tex, u, v, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
        detail = projected - cv.GaussianBlur(projected, (0, 0), 1.7 * w / 640)
        rms = np.sqrt(cv.GaussianBlur(detail * detail, (0, 0), 9 * w / 640) + 0.15)
        self.mat_grain = (np.clip(detail / rms, -2.8, 2.8) * region).astype(np.float32)

    def apply(self, frame, support_allow=1.0, film_masks=None, fixed_view_weight=1.0):
        fixed_view_weight = np.float32(fixed_view_weight)
        b, g, r = cv.split(frame.astype(np.float32))
        clamp = lambda a: np.clip(a, 0, 1)
        green = clamp((g - r - 5) / 22) * clamp((g - b + 1) / 16)
        # Specular surfaces have near-neutral high values. Limit correction to exposed sheet regions.
        spec = clamp((np.minimum(np.minimum(r, b), g) - 143) / 65)
        still = np.max(cv.absdiff(frame, self.initial), axis=2).astype(np.float32)
        spec *= clamp((48 - still) / 27)
        static = clamp((58 - still) / 35)
        surf = np.maximum(np.maximum(green, spec), self.belt_glint * static)
        # Gradual hue response for bright packaging, including highlights near the edge of saturation.
        pink = clamp((r - g - 9) / 33) * clamp((r - b - 4) / 23) * clamp((r - 90) / 74)
        blue = (
            clamp((b - r - 16) / 40) * clamp((b - g - 3) / 17) * clamp((b - 114) / 63)
        )
        surf *= support_allow * fixed_view_weight
        blind = (
            self.blind_window
            * fixed_view_weight
            * clamp((np.minimum(np.minimum(r, b), g) - 43) / 36)
        )
        mixed = frame.astype(np.float32) * (
            1 + self.surface * surf[:, :, None] + self.gains[2] * blind[:, :, None]
        )
        highlight = clamp((g - 102) / 111)
        amplitude = (0.85 + 4.8 * highlight) * surf * static
        mixed += (self.mat_grain * amplitude)[:, :, None]
        for weight, shift in [(pink, (9, 15, 35)), (blue, (34, 22, 28))]:
            mixed += weight[:, :, None] * np.array(shift, dtype=np.float32)
        if film_masks is not None:
            dyed, white, paper = film_masks
            # Sampled on settled real pink/blue envelopes and on their unprinted
            # paper areas. Retain the renderer's wrinkles, ink and moving shadows.
            for coverage, shift in [(pink, (29, 30, 15)), (blue, (17, 24, 30))]:
                mixed += (dyed * coverage)[:, :, None] * np.array(shift, np.float32)
            low = np.minimum(np.minimum(r, b), g)
            chroma = np.maximum(np.maximum(r, b), g) - low
            neutral = clamp((48 - chroma) / 24) * clamp((low - 102) / 58)
            y = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            lift = cv.LUT(y, PAPER_LIFT)
            gain = neutral * np.maximum(0.68 * white, paper)
            mixed += (lift * gain)[:, :, None]
            if np.any(white):
                # Work after the paper/film calibration, in display-referred luminance.
                luma = np.uint8(np.clip(cv.cvtColor(mixed, cv.COLOR_BGR2GRAY), 0, 255))
                mixed += (cv.LUT(luma, WHITE_FILM_LIFT) * white * neutral)[:, :, None]
        # Compress the top few display levels, emulating the video highlights.
        return np.uint8(np.clip(mixed - np.maximum(mixed - 250, 0) * 0.52, 0, 255))
