"""Measure transfer timing and settled bag outlines against the reference camera."""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
from finish import REAL, ROOT, SIM


def measurements(path, events):
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(path)
    fps = cap.get(cv.CAP_PROP_FPS)
    observed = []
    for t, kind, *_ in events:
        start = round((t - 0.72) * fps)
        cap.set(cv.CAP_PROP_POS_FRAMES, start)
        gate = []
        rect = None
        for frame in range(start, round((t + 0.52) * fps) + 1):
            ok, b = cap.read()
            if not ok:
                raise ValueError((path, frame))
            b = cv.resize(b, (640, 480), interpolation=cv.INTER_AREA)
            h, s, v = cv.split(cv.cvtColor(b, cv.COLOR_BGR2HSV))
            colored = (
                ((h > 146) | (h < 4)) & (s > 52) & (v > 110)
                if kind == 0
                else (h > 88) & (h < 117) & (s > 50) & (v > 119)
            )
            mask = np.uint8(colored)
            gate.append(mask[302:310, 150:500].sum() > 90)
            if frame == round((t + 0.52) * fps):
                mask[:229] = 0
                mask[:, :55] = 0
                mask[:, 550:] = 0
                cs = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
                kept = [c for c in cs if cv.contourArea(c) > 3200]
                if len(kept) == 1:
                    rect = cv.minAreaRect(kept[0])
        crossings = [
            (start + i) / fps
            for i in range(3, len(gate) - 1)
            if gate[i]
            and gate[i + 1]
            and not any(gate[i - 3 : i])
            and t - 0.35 <= (start + i) / fps <= t + 0.45
        ]
        observed.append((crossings[0] if len(crossings) == 1 else None, rect))
    cap.release()
    return observed


def compare(reference, simulation):
    events = np.loadtxt(ROOT / "assets/transfers.txt")
    events = events[events[:, 1] < 2]
    real = measurements(reference, events)
    sim = measurements(simulation, events)
    lag = []
    angles = []
    offset = []
    for (tr, rr), (ts, rs) in zip(real, sim):
        if tr is not None and ts is not None:
            lag.append(ts - tr)
        if rr is not None and rs is not None:
            angles.append(abs((rs[2] - rr[2] + 45) % 90 - 45))
            offset.append(np.linalg.norm(np.subtract(rs[0], rr[0])))
    if not lag or not angles:
        raise ValueError("No matched moving bags")
    print(
        f"{len(lag)} crossings, median signed lag {np.median(lag):.4f}s, median absolute {np.median(np.abs(lag)):.4f}s"
    )
    print(
        f"{len(angles)} settled outlines, median center error {np.median(offset):.2f}px, edge-angle error {np.median(angles):.2f} degrees"
    )
    return lag, offset, angles


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=REAL)
    p.add_argument("--simulation", type=Path, default=SIM)
    args = p.parse_args()
    compare(args.reference, args.simulation)
