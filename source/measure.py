"""Measure the first minute of RGB footage: parcels, moving silhouettes and lighting."""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

from cell import unproject

A = Path(__file__).resolve().parents[1] / "assets"
RATE = 10


def masks(frame, bg):
    h, s, v = cv.split(cv.cvtColor(frame, cv.COLOR_BGR2HSV))
    y, x = np.indices(s.shape)
    belt = (y > 46 - 0.013 * x) & (y < 199 - 0.015 * x)
    # Compare white packages and kraft boxes against the empty conveyor to
    # exclude the fixed overhead reflections, trim and the specular worktop.
    delta = np.max(abs(frame.astype(float) - bg.astype(float)), axis=2)
    pink = ((h > 151) | (h < 5)) & (s > 47) & (v > 112)
    blue = (
        (h > 91)
        & (h < 113)
        & (s > 40)
        & (v > 140)
        & ((frame[:, :, 0].astype(int) - frame[:, :, 1]) > 8)
    )
    white = (s < 59) & (v > 158) & (delta > 28) & belt
    brown = (h > 4) & (h < 22) & (s > 61) & (v > 70) & (delta > 25) & belt
    for mask in (pink, blue, white, brown):
        mask = mask.astype("uint8") * 255
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 7), np.uint8))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        yield mask


def detect(frame, bg):
    out = []
    for k, mask in enumerate(masks(frame, bg)):
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv.contourArea(c)
            if area < (480 if k < 3 else 980):
                continue
            center, (w, h), angle = cv.minAreaRect(c)
            if max(w, h) < 36 or min(w, h) < 9 or max(w, h) > 247:
                continue
            if k == 2 and not 1500 < area < 12700:
                continue
            if area / (w * h) < 0.43:
                continue
            # Convert projected edges into world yaw on the belt plane.
            xy = cv.boxPoints(
                cv.minAreaRect(unproject(c[:, 0, :], 0.036)[:, :2].astype("float32"))
            )
            ew = np.roll(xy, -1, axis=0) - xy
            length = np.linalg.norm(ew, axis=1)
            j = np.argmax(length)
            yaw = np.arctan2(ew[j, 1], ew[j, 0])
            dims = [length[j], length[(j + 1) % 4]]
            if k == 2 and any(
                d[0] in (0, 1) and cv.pointPolygonTest(d[-1], center, True) > -7
                for d in out
            ):
                continue
            out.append((k, *center, yaw, *dims, area, c))
    return out


def main(ref):
    cap = cv.VideoCapture(str(ref))
    ok, bg = cap.read()
    assert ok
    cv.imwrite(str(A / "empty-study.jpg"), bg)
    tracks = []
    sil = []
    edges = []
    light = []
    frames = []
    for i in range(600):
        if i:
            for _ in range(2):
                cap.grab()
            ok, frame = cap.read()
            assert ok
        else:
            frame = bg
        obs = detect(frame, bg)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        yy, xx = np.indices(v.shape)
        dark = (v < 120) & ((s < 73) | (v < 48)) & (yy < 270)
        # The matte black manipulator and the pale yellow tape are separately observed.
        dark |= (h > 15) & (h < 39) & (s > 40) & (v > 119) & (yy < 240)
        dark = cv.morphologyEx(
            dark.astype("uint8"), cv.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        sil.append(cv.resize(dark, (320, 240), interpolation=cv.INTER_NEAREST))
        edges.append(
            cv.resize(cv.Canny(frame, 55, 135), (320, 240), interpolation=cv.INTER_AREA)
        )
        frames.append(cv.resize(frame, (320, 240)))
        # Robust channel ratios from multiple mat regions; reject packages, grippers and shadows.
        roi = (yy > 258) & (yy < 460) & (xx > 65) & (xx < 585)
        valid = roi & (s > 64) & (h > 56) & (h < 88) & (v > 62)
        ratio = (frame.astype(float) + 8) / (bg.astype(float) + 8)
        light.append(np.median(ratio[valid], axis=0))
        for kind in range(4):
            active = [tr for tr in tracks if tr["k"] == kind and i - tr["i"][-1] <= 5]
            det = [o for o in obs if o[0] == kind]
            used = set()
            if active and det:
                cost = np.empty((len(active), len(det)))
                for j, tr in enumerate(active):
                    a = np.array(tr["v"][-1])
                    dt = i - tr["i"][-1]
                    vel = np.zeros(2)
                    if len(tr["v"]) > 1:
                        vel = np.clip(
                            (a[:2] - np.asarray(tr["v"][-2])[:2])
                            / (tr["i"][-1] - tr["i"][-2]),
                            -40,
                            40,
                        )
                    for l, o in enumerate(det):
                        d = np.linalg.norm(a[:2] + vel * min(dt, 2) - o[1:3])
                        cost[j, l] = d + 6 * abs(np.log(o[6] / a[-1]))
                for j, l in zip(*linear_sum_assignment(cost)):
                    if cost[j, l] > (65 if i - active[j]["i"][-1] == 1 else 96):
                        continue
                    active[j]["i"].append(i)
                    active[j]["v"].append(det[l][1:-1])
                    used.add(l)
            for l, o in enumerate(det):
                if l not in used:
                    tracks.append(dict(k=kind, i=[i], v=[o[1:-1]]))
    cap.release()
    tracks = [tr for tr in tracks if len(tr["i"]) >= 4]
    raw = np.full((len(tracks), 600, 6), np.nan, np.float32)
    for j, tr in enumerate(tracks):
        raw[j, tr["i"]] = tr["v"]
    np.savez_compressed(
        A / "observations.npz",
        raw=raw,
        kind=[tr["k"] for tr in tracks],
        light=gaussian_filter1d(np.clip(light, 0.69, 1.16), 1.5, axis=0),
    )
    np.savez_compressed(A / "silhouettes.npz", mask=sil, edge=edges, frames=frames)
    print(
        "tracks",
        len(tracks),
        [sum(tr["k"] == i for tr in tracks) for i in range(4)],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    main(p.parse_args().reference)
