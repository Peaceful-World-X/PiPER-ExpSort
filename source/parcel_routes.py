"""Identify real parcels once, in coordinates travelling with the conveyor."""

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from cell import EYE, FPS, FRAMES, project, unproject
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SPEED = 0.104
T = np.arange(FRAMES) / FPS
BELT = 0.035
TABLE = 0.0016
PINCH_Z = 0.014
PAD = 0.0018
PAD_Z = 0.135
INVERTED = np.diag([1.0, -1.0, -1.0])

# Centre crossing time and cross-belt coordinate of items that travel through the shot.
# Several upper and lower lanes are occupied at the same time.
THROUGH = [
    (3, 11.2, 0.595),
    (2, 15.65, 0.195),
    (2, 18.75, 0.55),
    (0, 21.0, 0.53),
    (1, 25.65, 0.485),
    (3, 28.18, 0.305),
    (2, 30.95, 0.58),
    (0, 39.75, 0.51),
    (0, 44.70, 0.25),
    (3, 47.05, 0.215),
    (2, 54.60, 0.17),
    (2, 59.94, 0.17),
    (0, 60.95, 0.51),
    (1, 63.3, 0.17),
]


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def rz(a):
    co, si = np.cos(a), np.sin(a)
    return np.array([[co, -si, 0.0], [si, co, 0.0], [0.0, 0.0, 1.0]])


def travelling_image(frames):
    tau = np.arange(0, 66, 0.0125)
    y = np.linspace(0.74, 0.025, 190)
    views = []
    for x in np.linspace(-0.40, -0.16, 11):
        times = tau + x / SPEED
        uv = np.rint(
            project(np.c_[np.full(len(y), x), y, np.full(len(y), 0.049)]) / 2
        ).astype(int)
        views.append(
            frames[
                np.clip(np.rint(times * 10).astype(int), 0, 599)[:, None],
                uv[None, :, 1],
                uv[None, :, 0],
            ].transpose(1, 0, 2)
        )
    return np.median(views, axis=0).astype("uint8"), tau, y


def hue_mask(im, k):
    h, s, v = cv.split(cv.cvtColor(im, cv.COLOR_BGR2HSV))
    return [
        (h > 147) & (s > 46) & (v > 105),
        (h >= 90) & (h < 117) & (s > 44) & (v > 128),
        (s < 62) & (v > 144),
        (h > 3) & (h < 24) & (s > 59) & (v > 63),
    ][k]


def fit_rect(points):
    corners = cv.boxPoints(cv.minAreaRect(np.asarray(points, dtype="float32")))
    edges = np.roll(corners, -1, axis=0) - corners
    lens = np.linalg.norm(edges, axis=1)
    j = np.argmax(lens)
    angle = np.arctan2(edges[j, 1], edges[j, 0])
    angle = (angle + np.pi / 2) % np.pi - np.pi / 2
    if abs(angle) > np.pi * 0.29:
        j = (j + 1) % 4
        angle = (angle + np.pi) % np.pi - np.pi / 2
    return (
        np.mean(corners, axis=0),
        float(lens[j]),
        float(lens[(j + 1) % 4]),
        float(angle),
    )


@dataclass
class Route:
    kind: int
    tau: float
    y: float
    length: float
    width: float
    yaw: float
    label: int
    arrive: float
    pull: float
    table_xy: np.ndarray
    table_yaw: float

    gate: float = 0.0
    carry_offsets: np.ndarray | None = None

    flip: bool = False
    time_edge: float | None = None

    def belt_center(self, t):
        return np.array(
            [
                (t - self.tau) * SPEED,
                self.y,
                BELT + (0.00015 if self.kind == 3 else 0.00155),
            ]
        )

    def times(self, side):
        pair = (
            (self.arrive - 0.54, self.arrive + 0.32)
            if side
            else (self.pull, self.pull + 0.70)
        )
        return tuple(round(v * FPS) / FPS for v in pair)

    def prepare_grips(self):
        self.release_pose = None
        self.local_anchor = {}
        self.reference = {}
        for side in (1, 0):
            a, _ = self.times(side)
            start = self.belt_center(a) if side else np.r_[self.table_xy, TABLE]
            end = (
                np.r_[self.table_xy, TABLE]
                if side
                else np.array([-0.605, -0.335, 0.116])
            )
            base = np.array([0.341, -0.023] if side else [-0.338, -0.023])
            angle = self.yaw if side else self.table_yaw
            d = rz(-angle)[:2, :2] @ (base - start[:2])
            lengths = np.array([self.length, self.width])
            axis = int(np.argmax(abs(d) / lengths))
            anchor = np.zeros(3)
            anchor[axis] = np.sign(d[axis]) * (lengths[axis] * 0.5 - 0.010)
            if (
                side == 0
                and np.linalg.norm(
                    (INVERTED @ anchor if self.flip else anchor)[:2]
                    - self.local_anchor[1][:2]
                )
                < 0.01
            ):
                anchor[1 - axis] = 0.035
            anchor[2] = PINCH_Z
            self.local_anchor[side] = anchor
            p = (start + rz(angle) @ anchor)[:2] - base
            az = np.arctan2(p[1], p[0]) - np.arcsin(0.0014165 / np.linalg.norm(p))
            offset = (rz(-az) @ rz(angle) @ anchor)[1]
            self.reference[side] = (base, angle, az, 0.0014165 - offset)
            if self.flip:
                self.local_anchor[side] = anchor.copy() if side else INVERTED @ anchor
                self.local_anchor[side][2] = 0
                if side:
                    self.prepare_turnover(base)
            elif side:
                natural = self.carried_rotation(side, end, 0, True)
                self.turn = (self.table_yaw - natural + np.pi / 2) % np.pi - np.pi / 2
                self.table_yaw = natural + self.turn

    def prepare_turnover(self, base):
        # A rectangle measured in one image has a 180 degree heading ambiguity.
        # Pick the landing that puts the held edge and the tool on the robot side.
        pick, down = self.times(1)
        start = self.belt_center(pick)[:2]
        best = None
        for yaw in (self.table_yaw, self.table_yaw + np.pi):
            # The observed hand stays at image-right while the free span swings.
            # A point on the roll axis remains on the same seam throughout.
            axis = (self.yaw + yaw) / 2
            if np.cos(axis) > 0:
                axis += np.pi
            along = -rz(axis - self.yaw)[:2, 0]
            radius = np.min(
                (np.array([self.length, self.width]) / 2 - 0.010)
                / np.maximum(abs(along), 1e-9)
            )
            anchor = np.r_[radius * along, 0.0]
            p0 = start + (rz(self.yaw) @ anchor)[:2] - base
            p1 = self.table_xy + (rz(yaw) @ INVERTED @ anchor)[:2] - base
            az0, az1 = (np.arctan2(v[1], v[0]) for v in (p0, p1))
            err = [(axis - az + np.pi) % (2 * np.pi) - np.pi for az in (az0, az1)]
            score = (
                0.7 * abs(np.sin(axis))
                + 0.55 * max(abs(e) for e in err)
                + np.linalg.norm(p1)
            )
            if best is None or score < best[0]:
                best = score, yaw, axis, anchor
        _, self.table_yaw, self.heading0, self.local_anchor[1] = best
        self.turn = 0.0

        edge = (
            self.time_edge
            if self.time_edge is not None
            else pick + 0.55 * (down - pick)
        )
        u = np.clip((edge - pick) / (down - pick), 0.14, 0.86)
        heading = self.heading0
        # The edge-on annotation is in image space. The tilted camera sees the
        # edge when its sight ray lies in the plane, not at 90 degrees of roll.
        edge = pick + u * (down - pick)
        self.edge_shift = np.zeros(2)
        if self.arrive == 10.9:
            # In source frame 326 the first turning pink spans x=206..362,
            # y=172..237. Anchor this one mid-stroke observation in the shot.
            sample = round(edge * FPS) / FPS
            center = self._transport(sample)[0] + [0, 0, PINCH_Z]
            self.edge_shift = (unproject([284, 204], center[2]) - center)[:2]
        center = self._transport(edge)[0] + [*self.edge_shift, PINCH_Z]
        view = EYE - center
        across = np.array([-np.sin(heading), np.cos(heading), 0])
        fraction = np.arctan2(view[2], -view @ across) / np.pi
        speed = 2 / (u / fraction + (1 - u) / (1 - fraction))
        self.roll_curve = CubicHermiteSpline([0, u, 1], [0, fraction, 1], [0, speed, 0])

    def roll(self, t):
        pick, down = self.times(1)
        return -np.pi * float(
            self.roll_curve(np.clip((t - pick) / (down - pick), 0, 1))
        )

    def carried_rotation(self, side, center, tilt, angle_only=False):
        base, angle, start, lateral = self.reference[side]
        delta = center[:2] - base
        az = np.arctan2(delta[1], delta[0]) - np.arcsin(
            np.clip(lateral / max(np.linalg.norm(delta), 0.001), -0.98, 0.98)
        )
        if angle_only:
            return angle + (az - start + np.pi) % (2 * np.pi) - np.pi
        co, si = np.cos(tilt), np.sin(tilt)
        bend = np.array([[co, 0, si], [0, 1, 0], [-si, 0, co]])
        return rz(az) @ bend @ rz(angle - start)

    def anchor(self, side):
        return self.local_anchor[side]

    def _transport(self, t):
        c = self.belt_center(t)
        angle = self.yaw
        tilt = 0.0
        carrier = None
        if self.arrive > 0:
            pick, down = self.times(1)
            pull, end = self.times(0)
            landing = np.r_[self.table_xy, TABLE]
            if t >= pick and t < down:
                u = (t - pick) / (down - pick)
                # Advance the crossing within the transfer without moving the belt pickup.
                bias = (
                    np.clip(0.155 + self.arrive - self.gate, 0.06, 0.24)
                    if self.gate
                    else 0.15
                )
                phase = np.clip(u + bias / (down - pick) * np.sin(np.pi * u), 0, 1)
                b = ease(phase)
                arc = np.sin(np.pi * phase) ** 1.28
                c = (1 - b) * self.belt_center(pick + min(t - pick, 0.12)) + b * landing
                c[2] += 0.163 * arc
                if self.carry_offsets is not None:
                    phase_samples = self.carry_offsets[:, 0]
                    c += np.array(
                        [
                            np.interp(u, phase_samples, col)
                            for col in self.carry_offsets[:, 1:].T
                        ]
                    )
                angle += b * (self.table_yaw - self.yaw)
                tilt = -0.43 * arc
                carrier = 1
            elif t >= down and t < pull:
                c = landing
                angle = self.table_yaw
            elif t >= pull:
                u = np.clip((t - pull) / (end - pull), 0, 1)
                b = ease(u)
                # The left arm lifts over the transparent side guard before leaving the camera.
                c = (1 - b) * landing + b * np.array([-0.605, -0.335, 0.116])
                c[1] -= 0.035 * np.sin(np.pi * u) ** 1.5
                c[2] += 0.055 * np.sin(np.pi * u) ** 1.35
                angle = self.table_yaw - 0.13 * b
                tilt = 0.20 * np.sin(np.pi * u)
                carrier = 0
                if t > end:
                    elapsed = t - end
                    c[2] = max(-0.277, 0.116 - 0.5 * 9.81 * elapsed**2)
                    c[0] -= 0.30 * ease((elapsed - 0.12) / 0.65)
        rot = (
            self.carried_rotation(carrier, c, tilt)
            if carrier is not None
            else rz(angle)
        )
        if carrier == 1 and not self.flip:
            rot = rz(self.turn * b) @ rot
        return c, rot, carrier

    def turnover_surface(self, t):
        c, rot, carrier = self._transport(t)
        if not self.flip:
            return c, rot
        c[2] += PINCH_Z
        if carrier == 1:
            if self.time_edge is not None:
                c[:2] += self.edge_shift * ease(1 - abs(t - self.time_edge) / 0.23)
            rot = (
                rz(self.heading0)
                @ Rotation.from_euler("x", self.roll(t)).as_matrix()
                @ rz(self.yaw - self.heading0)
            )
            # Raise the free corners over the belt rail as the sheet banks.
            corners = np.array(
                [
                    [x * self.length / 2, y * self.width / 2, 0]
                    for x in (-1, 1)
                    for y in (-1, 1)
                ]
            )
            world = corners @ rot.T + c
            floor = np.where(world[:, 1] >= 0.034, BELT, TABLE)
            c[2] += max(0, np.max(floor + PINCH_Z - world[:, 2]))
        elif t >= self.times(1)[1]:
            if carrier == 0:
                c += (rot[:, 2] - [0, 0, 1]) * PINCH_Z
            rot = rot @ INVERTED
        return c, rot

    def pose(self, t):
        c, rot = self.turnover_surface(t)
        if self.flip and self.release_pose is not None:
            down = self.times(1)[1]
            u = (t - down) / 0.14
            if 0 <= u < 1:
                # The core rests on the mat. Unload the inclined seam around
                # its last physical pinch, using the same root as the held FK.
                weight = 1 - ease(u)
                point, relative = self.release_pose
                grip = c + rot @ self.anchor(1)
                grip += weight * (point - grip)
                rot = rot @ Rotation.from_rotvec(relative * weight).as_matrix()
                c = grip - rot @ self.anchor(1)
        return c, rot


def projected_radius(item, axes):
    margins = np.array(
        [
            item.length * 0.5 + (0.010 if item.arrive > 0 else 0.003),
            item.width * 0.5 + 0.002,
        ]
    )
    return np.abs(axes @ rz(item.yaw)[:2, :2]) @ margins


def routes():
    frames = np.load(ASSETS / "silhouettes.npz")["frames"]
    events = np.loadtxt(ASSETS / "transfers.txt", encoding="utf-8-sig").tolist()
    # The final blue bag starts its pick at the very end of the original minute.
    events.append([60.45, 1, 153, 40, 61.37])
    seeds = []
    for arrive, k, u, v, pull in events:
        x, y, _ = unproject(np.array([u, v]) * 2, 0.049)
        seeds.append([int(k), arrive - 0.9 - x / SPEED, y, arrive, pull])
    seeds.extend([[k, t, y, -1, -1] for k, t, y in THROUGH])
    im, tau, ys = travelling_image(frames)
    xx, yy = np.meshgrid(tau, ys)
    landings = {round(row[0], 2): row for row in np.loadtxt(ASSETS / "landings.txt")}
    observed = np.loadtxt(ASSETS / "carry_observations.txt")
    result = []
    for k in range(4):
        subset = [s for s in seeds if s[0] == k]
        d = np.stack(
            [
                ((xx - t) * SPEED / 0.13) ** 2 + ((yy - y) / 0.13) ** 2
                for _, t, y, _, _ in subset
            ]
        )
        owner = np.argmin(d, axis=0)
        for j, (_, t, y, arrive, pull) in enumerate(subset):
            mask = (owner == j) & (d[j] < 2.3) & hue_mask(im, k)
            pts = np.c_[-xx[mask] * SPEED, yy[mask]]
            if len(pts) < 30:
                raise ValueError(f"No belt footprint for {k}, {t}")
            # Percentile cropping rejects isolated bright rail or marker pixels.
            lo, hi = np.quantile(pts, [0.014, 0.986], axis=0)
            pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
            ctr, l, w, ang = fit_rect(pts)
            tc = -ctr[0] / SPEED
            cx = np.clip(round(tc / 0.0125), 0, im.shape[1] - 1)
            cy = np.argmin(abs(ys - ctr[1]))
            patch = im[max(0, cy - 10) : cy + 11, max(0, cx - 16) : cx + 17].astype(
                float
            )
            label = int(
                k == 2
                or np.mean((patch.min(2) > 156) & (np.ptp(patch, axis=2) < 49)) > 0.13
            )
            if k == 3:
                l, w = 0.108, 0.078
            else:
                l = float(np.clip(l, 0.146, 0.222))
                w = float(np.clip(w, 0.126, 0.201))
            target = np.array([-0.035, -0.275])
            ta = ang
            gate = 0.0
            if arrive in landings:
                _, tx, ty, length, width, ta, mark, gate, _ = landings[arrive]
                target = np.array([tx, ty])
                l, w = (
                    float(np.clip(length, 0.14, 0.255)),
                    float(np.clip(width, 0.13, 0.242)),
                )
                label = int(mark)
                # Near-square upstream silhouettes have an ambiguous long side. Resolve
                # that quarter turn using the settled label and right-arm transport direction.
                natural_turn = np.deg2rad(74)
                diff = (ta - ang - natural_turn + np.pi / 2) % np.pi - np.pi / 2
                if abs(diff) > np.pi / 4:
                    ang += np.sign(diff) * np.pi / 2
            target = np.clip(target, [-0.16, -0.369], [0.09, -0.135])

            result.append(
                Route(
                    k,
                    tc,
                    float(ctr[1]),
                    l,
                    w,
                    ang,
                    label,
                    arrive,
                    pull,
                    target,
                    ta,
                    gate,
                    observed[observed[:, 0] == arrive, 1:]
                    if np.any(observed[:, 0] == arrive)
                    else None,
                )
            )
    # Correct small footprint measurement overlaps before planning either arm.
    for _ in range(12):
        moved = False
        for i, r in enumerate(result):
            for b in result[:i]:
                delta = np.array([(b.tau - r.tau) * SPEED, r.y - b.y])
                if np.linalg.norm(delta) > 0.36:
                    continue
                axes = np.column_stack((rz(r.yaw)[:2, :2], rz(b.yaw)[:2, :2])).T

                depth = (
                    projected_radius(r, axes)
                    + projected_radius(b, axes)
                    - np.abs(axes @ delta)
                )
                if np.min(depth) < 1e-5:
                    continue
                j = np.argmin(depth)
                push = axes[j] * np.sign(axes[j] @ delta) * (depth[j] + 0.001) * 0.5
                r.tau -= push[0] / SPEED
                b.tau += push[0] / SPEED
                r.y += push[1]
                b.y -= push[1]
                moved = True
        if not moved:
            break
    result.sort(key=lambda r: r.tau)
    table = ASSETS / "turnovers.txt"
    # The through-running silver P005 has no transfer row. Its visible face is blank.
    result[5].label = 0
    for ident, entry, landing, flip, edge in np.loadtxt(
        table, ndmin=2, encoding="utf-8-sig"
    ):
        if not 0 <= int(ident) < len(result) or ident != int(ident):
            raise ValueError(f"Invalid turnover ID {ident}")
        r = result[int(ident)]
        if landing >= 0 or entry >= 0:
            r.label = int(landing if landing >= 0 else entry)
        if flip == 1:
            if r.arrive <= 0 or r.kind == 3:
                raise ValueError(f"P{ident:03.0f} has no pouch transfer")
            r.flip = True
            r.time_edge = float(edge) if edge > 0 else None
    for r in result:
        if r.arrive > 0:
            r.prepare_grips()
    return result
