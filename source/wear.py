"""Workshop wear maps. All colour values are sRGB BGR; roughness and normals are linear."""

import cv2 as cv
import numpy as np


def build_wear(output, relief):
    rng = np.random.default_rng(24519)

    def field(w, h, rows, cols):
        patch = rng.normal(size=(rows, cols)).astype("float32")
        return cv.resize(patch, (w, h), interpolation=cv.INTER_CUBIC)

    def scratches(w, h, count, ratio, length):
        marks = np.zeros((h, w), np.uint8)
        for _ in range(count):
            p = rng.uniform([0, 0], [w, h])
            vec = rng.normal(size=2) * [length, length * ratio]
            cv.line(
                marks,
                tuple(p.astype(int)),
                tuple((p + vec).astype(int)),
                int(rng.uniform(42, 240)),
                1,
                cv.LINE_AA,
            )
        return marks.astype("float32") / 255

    # The belt travels in local X. Polish follows the parcel lanes and leaves the outer margins dull.
    w, h = 2048, 768
    yy, xx = np.mgrid[:h, :w].astype("float32")
    cross = yy / h
    broad = field(w, h, 11, 26)
    streak = field(w, h, 190, 22)
    micro = field(w, h, 510, 850)
    gloss = 0.7 * np.exp(-(((cross - 0.31) / 0.20) ** 2)) + 0.50 * np.exp(
        -(((cross - 0.70) / 0.15) ** 2)
    )
    edge = np.exp(-np.minimum(yy, h - 1 - yy) / (h * 0.050))
    abrasion = scratches(w, h, 1120, 0.075, 71)
    wipe = np.maximum(-0.1, field(w, h, 48, 34)) * np.clip(streak * 0.48 + 0.58, 0, 1)
    albedo = (
        np.array([42, 67, 18])
        + (
            (1.7 * broad + 1.65 * streak + 0.7 * micro + 6.2 * abrasion + 2.6 * wipe)[
                :, :, None
            ]
        )
    )
    albedo += edge[:, :, None] * np.array([3.0, 2.1, 3.7])
    rough = np.clip(
        0.190
        - 0.080 * gloss
        + 0.013 * broad
        + 0.009 * streak
        + 0.068 * edge
        + 0.08 * abrasion
        - 0.022 * wipe,
        0.11,
        0.40,
    )
    height = 0.018 * streak + 0.011 * micro - 0.034 * abrasion
    # Match both ends so the belt can wrap without a visible seam.
    for arr in (albedo, rough, height):
        n = 90
        s = np.linspace(0, 1, n, dtype="float32")
        if arr.ndim == 3:
            s = s[None, :, None]
        else:
            s = s[None, :]
        avg = (arr[:, :1] + arr[:, -1:]) * 0.5
        arr[:, :n] = s * arr[:, :n] + (1 - s) * avg
        arr[:, -n:] = (1 - s) * arr[:, -n:] + s * avg
    output("belt", albedo)
    relief("belt", height, rough)

    # Project each CAD face onto a stable local plane. Wear is strongest close to its
    # outside dimensions, interrupted by paint islands and straight polishing marks.
    w = h = 896
    yy, xx = np.mgrid[:h, :w].astype("float32")
    dist = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]) / w
    for name, bgr, baseline, is_metal in [
        ("usedGraphite", (71, 69, 68), 0.39, True),
        ("usedJoint", (43, 41, 40), 0.47, False),
        ("usedPlate", (43, 42, 40), 0.46, True),
        ("usedIvory", (201, 199, 199), 0.43, True),
        ("usedTape", (70, 160, 196), 0.55, False),
    ]:
        clouds = field(w, h, 11, 15)
        fine = field(w, h, 530, 590)
        rub = field(w, h, 90, 48)
        lines = scratches(w, h, 720, 0.49, 29)
        chips = np.clip(
            (field(w, h, 95, 109) + 0.34 * field(w, h, 210, 190) - 0.46) * 1.5, 0, 1
        )
        rim = np.exp(-dist / (0.010 if name != "usedPlate" else 0.017))
        bare = rim * chips * (0.70 if is_metal else 0.21) + lines * (0.09 + 0.36 * rim)
        # Short contiguous rubs on exposed lips are wide enough to read at the working-camera distance.
        damage = np.zeros((h, w), np.uint8)
        for j in range(45):
            a = int(rng.uniform(0.05, 0.95) * w)
            b = int(rng.uniform(0.001, 0.032) * h)
            length = int(rng.uniform(6, 44))
            end = b + int(rng.normal(0, 3))
            for dx in (0, 1):
                if j % 4 == 0:
                    pts = ((a, b + dx), (a + length, end + dx))
                elif j % 4 == 1:
                    pts = ((a, h - 1 - b - dx), (a + length, h - 1 - end - dx))
                elif j % 4 == 2:
                    pts = ((b + dx, a), (end + dx, a + length))
                else:
                    pts = ((w - 1 - b - dx, a), (w - 1 - end - dx, a + length))
                cv.line(
                    damage,
                    *pts,
                    int(rng.uniform(85, 200)),
                    int(rng.integers(1, 4)),
                    cv.LINE_AA,
                )
        bare = np.clip(
            bare + damage.astype("float32") / 255 * (0.87 if is_metal else 0.42), 0, 1
        )
        oil = np.clip((clouds + 0.4 * rub - 0.18) * 0.62, 0, 1)
        tone = 1.85 * clouds + 0.95 * fine + 1.20 * rub + lines * 5.0 - 2.8 * oil
        base = np.array(bgr, dtype="float32")
        tint = (
            np.array([121, 127, 129])
            if is_metal and name != "usedIvory"
            else np.array([145, 149, 146])
            if name == "usedIvory"
            else base + 21
        )
        color = base + tone[:, :, None] + bare[:, :, None] * (tint - base)
        rough = np.clip(
            baseline
            + 0.038 * clouds
            + 0.023 * rub
            + 0.018 * fine
            - 0.102 * oil
            - 0.13 * bare
            + 0.048 * lines,
            0.22,
            0.70,
        )
        normal = 0.019 * fine + 0.026 * rub - 0.085 * lines + 0.060 * bare
        output(name, color)
        relief(name, normal, rough)
