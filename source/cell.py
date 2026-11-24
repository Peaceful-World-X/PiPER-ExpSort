"""Metric camera and stage coordinates. Front sorting mat is at negative Y."""

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

FPS = 50
FRAMES = 3000
BASES = np.array([[-0.325, 0, 0.009], [0.325, 0, 0.009]])
# Measured outer plate corners, clockwise starting at rear left, on reference frame 0.
CORNERS = np.array(
    [
        [60, 213],
        [186, 213],
        [179, 247],
        [50, 249],
        [468, 208],
        [585, 207],
        [602, 239],
        [475, 242],
    ],
    float,
)


def camera():
    world = np.array(
        [
            [cx + x, y, 0.0065]
            for cx in BASES[:, 0]
            for x, y in [
                (-0.097, 0.050),
                (0.097, 0.050),
                (0.097, -0.050),
                (-0.097, -0.050),
            ]
        ]
    )

    def solve(f):
        K = np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1.0]])
        _, r, t = cv.solvePnP(world, CORNERS, K, None)
        return K, r, t

    def resid(w):
        K, r, t = solve(w[0])
        return np.r_[
            (cv.projectPoints(world, r, t, K, None)[0][:, 0] - CORNERS).ravel(),
            (w[0] - 595) / 90,
        ]

    f = least_squares(resid, [595.0], bounds=(420, 900)).x[0]
    K, r, t = solve(f)
    R = cv.Rodrigues(r)[0]
    eye = (-R.T @ t).ravel()
    return K, R, eye


K, R, EYE = camera()


def project(world):
    xyz = (np.asarray(world) - EYE) @ R.T
    h = xyz @ K.T
    return h[..., :2] / h[..., 2:3]


def unproject(uv, z):
    uv = np.asarray(uv)
    ray = (
        np.concatenate((uv, np.ones(uv.shape[:-1] + (1,))), axis=-1)
        @ np.linalg.inv(K).T
        @ R
    )
    return EYE + ray * ((z - EYE[2]) / ray[..., 2])[..., None]
