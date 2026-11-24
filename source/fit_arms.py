"""Estimate rigid-arm animation from sparse hand observations, optical flow, and the PiPER FK."""

from pathlib import Path

import cv2 as cv
import mujoco as mj
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cell import BASES, FPS, FRAMES, project, unproject

A = Path(__file__).resolve().parents[1] / "assets"
M = mj.MjModel.from_xml_path(str(A / "agilex_piper/piper.xml"))
D = mj.MjData(M)
BASE = BASES + np.array([[-0.013, -0.023, 0], [0.016, -0.023, 0]])
YAW = np.array([0.0, np.pi])
RZ = Rotation.from_euler("z", YAW[:, None]).as_matrix()


def points(q, side):
    D.qpos[:6] = q
    mj.mj_kinematics(M, D)
    end = D.body("link6")
    pts = np.array(
        [D.body(j).xpos.copy() for j in ["link3", "link6"]]
        + [
            end.xpos + 0.137 * end.xmat.reshape(3, 3)[:, 2],
            D.body("link4").xpos.copy(),
            D.body("link2").xpos.copy(),
        ]
    )
    return pts @ RZ[side].T + BASE[side]


def observations(rgb, hand):
    g = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in rgb]
    sample = np.zeros((600, 2, 3, 2))
    for low in range(0, 600, 10):
        high = min(low + 10, 599)

        def follow(begin, end, start):
            out = [start.reshape(-1, 2).astype(np.float32)]
            step = 1 if end > begin else -1
            for k in range(begin, end, step):
                pos = out[-1]
                off = np.array([[-4, 0], [4, 0], [0, 0], [0, -4], [0, 4]], np.float32)
                cloud = (pos[:, None] + off).reshape(-1, 1, 2)
                nxt, status, _ = cv.calcOpticalFlowPyrLK(
                    g[k], g[k + step], cloud, None, winSize=(23, 23), maxLevel=3
                )
                delta = (nxt - cloud).reshape(-1, 5, 2)
                ok = status.reshape(-1, 5) > 0
                update = np.array(
                    [
                        np.median(a[m], axis=0) if m.any() else [0, 0]
                        for a, m in zip(delta, ok)
                    ]
                )
                out.append((pos + np.clip(update, -34, 34)).astype(np.float32))
            return np.array(out).reshape(-1, 2, 3, 2)

        l, r = hand[low // 10], hand[low // 10 + 1]
        forward = follow(low, high, l)
        back = follow(high, low, r)[::-1]
        u = np.linspace(0, 1, len(forward))[:, None, None, None]
        linear = l * (1 - u) + r * u
        # Anchor the optical flow at both manual observations to limit drift during occlusions.
        f = forward + u * (r - forward[-1])
        b = back + (1 - u) * (l - back[0])
        diff = (f * (1 - u) + b * u) - linear
        sample[low : high + 1] = linear + np.clip(diff, -45, 45)
    return gaussian_filter1d(sample, 0.43, axis=0, mode="nearest")


def fit():
    vis = np.load(A / "silhouettes.npz")
    hand = np.loadtxt(A / "landmarks.txt", encoding="utf-8-sig")[:, 1:].reshape(
        -1, 2, 3, 2
    )
    targets = observations(vis["frames"], hand)
    for row in np.loadtxt(A / "midpoints.txt", encoding="utf-8-sig"):
        center = round(row[0] * 10)
        seen = row[1:].reshape(2, 2, 2)
        offsets = np.where(seen > -900, seen - targets[center, :, 1:, :], 0)
        for k in range(max(0, center - 4), min(len(targets), center + 5)):
            targets[k, :, 1:, :] += np.cos((k - center) * np.pi / 10) ** 2 * offsets
    low = M.jnt_range[:5, 0] + 0.0001
    hi = M.jnt_range[:5, 1] - 0.0001
    parcel = np.load(A / "observations.npz")
    raw = parcel["raw"]
    masks = vis["mask"]
    full = []
    errors = []
    for side in (0, 1):
        poses = []
        prev = np.array([0, 0.48, -0.5, 0, 0.7])
        for i, target in enumerate(targets[:, side]):
            dist = cv.distanceTransform(1 - masks[i], cv.DIST_L2, 3)
            near = raw[:, i, :2] / 2
            parcel_dist = np.linalg.norm(near - target[2], axis=1)
            settled = i > 23 and np.any(parcel_dist < 29) and target[2, 1] > 37
            height = (
                0.064
                if settled
                else np.clip(0.085 + (80 - target[2, 1]) * 0.0014, 0.055, 0.26)
            )
            if i < 23:
                height = 0.14
            if settled:
                height = 0.060 if target[2, 1] < 98 else 0.027
            world = unproject(2 * target[2], height)
            while np.linalg.norm(world[:2] - BASE[side, :2]) > 0.55 and height < 0.32:
                height += 0.010
                world = unproject(2 * target[2], height)

            def residual(x):
                p = points(np.r_[x, 0], side)
                uv = project(p) * 0.5
                shaft = np.vstack(
                    [
                        uv[a] * w + uv[b] * (1 - w)
                        for a, b in [(0, 4), (0, 3), (3, 1), (1, 2)]
                        for w in [0.25, 0.5, 0.75]
                    ]
                )
                inside = (
                    (shaft[:, 0] > 1)
                    & (shaft[:, 0] < 318)
                    & (shaft[:, 1] > 1)
                    & (shaft[:, 1] < 238)
                )
                outside = (
                    map_coordinates(
                        dist, [shaft[:, 1], shaft[:, 0]], order=1, mode="nearest"
                    )
                    * inside
                )
                return np.r_[
                    ((uv[:3] - target) * np.array([[0.72], [0.85], [1.15]])).ravel(),
                    np.clip(outside, 0, 38) * 0.72,
                    (p[2] - world) * np.array([190.0, 190.0, 640.0]),
                    max(0, (0.030 if target[2, 1] < 98 else 0.010) - p[2, 2]) * 500,
                    (x - prev) * np.array([4, 4, 4, 5, 3]),
                    x[3] * 2.0,
                    max(0, p[2, 1] - 0.78) * 100,
                ]

            seeds = [prev]
            if i == 0 or (i % 10 == 0 and i > 10):
                direction = (
                    np.arctan2(world[1] - BASE[side, 1], world[0] - BASE[side, 0])
                    - YAW[side]
                )
                direction = (direction + np.pi) % (2 * np.pi) - np.pi
                seeds.extend(
                    [[direction, 1.1, -1.3, 0, 0.8], [direction, 1.9, -0.9, 0, 0.9]]
                )
            step = np.deg2rad([30.0, 26.0, 30.0, 34.0, 30.0])
            lo = np.maximum(low, prev - step) if i else low
            up = np.minimum(hi, prev + step) if i else hi
            res = min(
                [
                    least_squares(
                        residual,
                        np.clip(seed, lo, up),
                        bounds=(lo, up),
                        max_nfev=32,
                        diff_step=0.0005,
                    )
                    for seed in seeds
                ],
                key=lambda o: o.cost,
            )
            prev = res.x
            poses.append(np.r_[prev, 0.0])
            errors.append(
                np.linalg.norm(
                    project(points(poses[-1], side)[:3]) * 0.5 - target, axis=1
                )
            )
        # The 10 Hz evidence is regularised and resampled on the actual 50 Hz project clock.
        poses = gaussian_filter1d(poses, 0.74, axis=0, mode="nearest")
        q = CubicSpline(np.arange(600) / 10, poses, axis=0)(
            np.clip(np.arange(FRAMES) / FPS, 0, 59.9)
        )
        q = np.clip(q, M.jnt_range[:6, 0], M.jnt_range[:6, 1])
        full.append(q)
        print("arm", side, "fit", flush=True)
    full = np.array(full)
    tips = np.array([[points(q, s)[2] for q in full[s]] for s in (0, 1)])
    print(
        "median landmark deviations at half resolution",
        np.median(errors, axis=0),
        flush=True,
    )
    np.savez_compressed(
        A / "arms.npz", q=full, tip=tips, base=BASE, yaw=YAW, observed=targets
    )


if __name__ == "__main__":
    fit()
