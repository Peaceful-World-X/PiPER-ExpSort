"""Half-speed replay of two entire handoffs with a simultaneous two-times detail view."""

import argparse
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np
from cell import FPS, project
from finish import FFMPEG, ROOT, SIM
from parcel_routes import routes
from pxr import Usd, UsdGeom
from scipy.ndimage import gaussian_filter1d


def review(stage_path, source, destination):
    stage = Usd.Stage.Open(str(stage_path))
    cache = UsdGeom.XformCache()
    cap = cv.VideoCapture(str(source))
    w, h = (
        int(cap.get(k)) for k in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT)
    )
    tmp = destination.with_suffix(".partial.mp4")
    cmd = [
        FFMPEG,
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w * 2}x{h}",
        "-framerate",
        "50",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    items = routes()
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as pipe:
        for when in (9.0, 15.1):
            j = next(i for i, r in enumerate(items) if r.arrive == when)
            r = items[j]
            prim = stage.GetPrimAtPath(f"/World/Parcels/P{j:03}")
            a = round((r.times(1)[0] - 0.24) * FPS)
            b = round((r.times(0)[1] + 0.06) * FPS)
            centers = []
            for f in range(a, b):
                cache.SetTime(f)
                m = np.asarray(cache.GetLocalToWorldTransform(prim))
                side = 1 if f / FPS < (r.times(1)[1] + r.times(0)[0]) * 0.5 else 0
                pixel = project(0.58 * r.anchor(side) @ m[:3, :3] + m[3, :3]) * [
                    w / 640,
                    h / 480,
                ]
                centers.append(pixel)
            cap.set(cv.CAP_PROP_POS_FRAMES, a)
            for center in gaussian_filter1d(centers, 2, axis=0, mode="nearest"):
                ok, im = cap.read()
                if not ok:
                    raise RuntimeError("Incomplete review source")
                x = int(np.clip(round(center[0] - w / 4), 0, w // 2))
                y = int(np.clip(round(center[1] - h / 4), 0, h // 2))
                detail = cv.resize(
                    im[y : y + h // 2, x : x + w // 2],
                    (w, h),
                    interpolation=cv.INTER_LANCZOS4,
                )
                block = np.hstack([im, detail]).tobytes()
                pipe.stdin.write(block)
                pipe.stdin.write(block)
        pipe.stdin.close()
        cap.release()
        if pipe.wait():
            raise RuntimeError("Review encode failed")
    tmp.replace(destination)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=Path, default=ROOT / "ConveyorCell.usdc")
    p.add_argument("--source", type=Path, default=SIM)
    p.add_argument(
        "--output", type=Path, default=ROOT / "renders/conveyor-grasp-review.mp4"
    )
    args = p.parse_args()
    review(args.stage, args.source, args.output)
