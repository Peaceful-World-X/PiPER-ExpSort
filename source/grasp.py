"""Synchronise the PiPER joints, opposing jaw pads, and the sealed parcel edges in metres."""

import itertools

import mujoco as mj
import numpy as np
from cell import FPS, FRAMES
from parcel_routes import ASSETS, PAD, PAD_Z, T, ease, routes, rz
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation

OPEN = 0.020


class Kinematics:
    def __init__(self, side, base, yaw):
        self.model = mj.MjModel.from_xml_path(str(ASSETS / "agilex_piper/piper.xml"))
        self.data = mj.MjData(self.model)
        self.base = base
        self.rot = Rotation.from_euler("z", yaw).as_matrix()
        self.hand = self.model.body("link6").id
        self.bounds = self.model.jnt_range[:6].T.copy()

    def pose(self, q):
        self.data.qpos[:6] = q
        mj.mj_kinematics(self.model, self.data)
        b = self.data.body(self.hand)
        self.local = b.xpos + PAD_Z * b.xmat.reshape(3, 3)[:, 2]
        return self.base + self.rot @ self.local, self.rot @ b.xmat.reshape(3, 3)

    def fit_pose(self, q, point, frame, steps=18):
        # Spatial Jacobian evaluated at the pad midpoint, in world coordinates.
        q = q.copy()
        jp = np.empty((3, self.model.nv))
        jr = np.empty_like(jp)
        for _ in range(steps):
            pos, R = self.pose(q)
            err = np.r_[
                point - pos, 0.13 * Rotation.from_matrix(frame @ R.T).as_rotvec()
            ]
            if np.linalg.norm(err) < 0.00004:
                break
            mj.mj_comPos(self.model, self.data)
            mj.mj_jac(self.model, self.data, jp, jr, self.local, self.hand)
            jac = np.vstack([self.rot @ jp[:, :6], 0.13 * self.rot @ jr[:, :6]])
            free = np.ones(6, bool)
            for _active in range(3):
                j = jac[:, free]
                dq = np.zeros(6)
                dq[free] = j.T @ np.linalg.solve(j @ j.T + 0.000018 * np.eye(6), err)
                outside = ((q < self.bounds[0] + 0.00001) & (dq < 0)) | (
                    (q > self.bounds[1] - 0.00001) & (dq > 0)
                )
                if not np.any(free & outside):
                    break
                free &= ~outside
            q = np.clip(
                q + dq * min(1, 0.22 / max(0.00001, np.max(abs(dq)))), *self.bounds
            )
        pos, R = self.pose(q)
        return q, np.linalg.norm(point - pos) + 0.13 * Rotation.from_matrix(
            frame @ R.T
        ).magnitude()

    def prepare_planar(self):
        self.pose(np.zeros(6))
        a = self.data.xanchor
        l1 = (a[2] - a[1])[[0, 2]]
        l2 = (a[4] - a[2])[[0, 2]]
        end = (self.local - a[4])[[0, 2]]
        self.lengths = np.array([np.linalg.norm(v) for v in (l1, l2, end)])
        self.angles = np.array([np.arctan2(v[1], v[0]) for v in (l1, l2, end)])
        self.offset = self.local[1]
        self.shoulder = a[1, 2]
        self.pitch = np.linspace(-1.45, 0.94, 1701)

    def planar(self, point, preferred=None):
        p = self.rot.T @ (point - self.base)
        d = np.linalg.norm(p[:2])
        az = np.arctan2(p[1], p[0]) - np.arcsin(self.offset / max(d, 0.01))
        pitch = self.pitch
        a, b, c = self.lengths
        x = np.sqrt(max(0, d * d - self.offset * self.offset)) - c * np.cos(pitch)
        z = p[2] - self.shoulder - c * np.sin(pitch)
        cosine = (x * x + z * z - a * a - b * b) / (2 * a * b)
        bend = -np.arccos(np.clip(cosine, -1, 1))
        q2 = (
            self.angles[0]
            - (np.arctan2(z, x) - np.arctan2(b * np.sin(bend), a + b * np.cos(bend)))
        ) % (2 * np.pi)
        q2 = np.where(q2 > 5.0, q2 - 2 * np.pi, q2)
        q3 = self.angles[1] - self.angles[0] - bend
        q5 = self.angles[2] - q2 - q3 - pitch
        qs = np.stack([q2, q3, q5])
        low, high = self.bounds[:, [1, 2, 4]]
        error = (
            np.maximum(low[:, None] - qs, 0).sum(0)
            + np.maximum(qs - high[:, None], 0).sum(0)
            + np.maximum(0, np.abs(cosine) - 1)
        )
        ideal = -np.deg2rad(np.interp(d, [0.18, 0.4, 0.58], [40, 24, 7]))
        if preferred is not None:
            ideal = preferred
        j = np.argmin(np.abs(pitch - ideal) + 2000 * error)
        return np.clip([az, q2[j], q3[j], 0.0, q5[j], np.pi / 2], *self.bounds)

    def aim_wrist(self, wrist, point):
        # With joints 4-6 fixed, the elbow-to-contact vector is one rigid
        # link with a lateral offset. Solve base yaw and the two planar joints.
        q = np.r_[np.zeros(3), wrist]
        self.pose(q)
        tip = self.local - self.data.xanchor[2]
        p = self.rot.T @ (point - self.base)
        d = np.linalg.norm(p[:2])
        az = np.arctan2(p[1], p[0]) - np.arcsin(np.clip(tip[1] / d, -1, 1))
        x = np.sqrt(max(0, d * d - tip[1] ** 2))
        z = p[2] - self.shoulder
        a, b = self.lengths[0], np.hypot(tip[0], tip[2])
        bend = -np.arccos(np.clip((x * x + z * z - a * a - b * b) / (2 * a * b), -1, 1))
        q[0] = (az + np.pi) % (2 * np.pi) - np.pi
        q[1] = (
            self.angles[0]
            - np.arctan2(z, x)
            + np.arctan2(b * np.sin(bend), a + b * np.cos(bend))
        )
        q[2] = (np.arctan2(tip[2], tip[0]) - self.angles[0] - bend + np.pi) % (
            2 * np.pi
        ) - np.pi
        return np.clip(q, *self.bounds)


def tool_frame(base, point):
    direction = point[:2] - base[:2]
    distance = np.linalg.norm(direction)
    direction /= max(distance, 1e-7)
    alpha = np.deg2rad(np.interp(distance, [0.20, 0.40, 0.57], [35, 23, 6]))
    forward = np.r_[direction * np.cos(alpha), -np.sin(alpha)]
    closing = np.r_[direction * np.sin(alpha), np.cos(alpha)]
    return np.column_stack([np.cross(closing, forward), closing, forward])


def grasp_point(item, t, side):
    center, rotation = item.pose(t)
    return center + rotation @ item.anchor(side)


def plan(side, items, base, initial):
    actions = sorted(
        [
            (r.times(side)[0], r.times(side)[1], j)
            for j, r in enumerate(items)
            if r.arrive > 0
        ]
    )
    owner = np.full(FRAMES, -1, int)
    pos = np.zeros((FRAMES, 3))
    jaw = np.full(FRAMES, OPEN)
    for a, b, j in actions:
        use = np.flatnonzero((T >= a - 1e-7) & (T <= b + 1e-7))
        for f in use:
            pos[f] = grasp_point(items[j], T[f], side)
            owner[f] = j
            jaw[f] = PAD
    previous_end = None
    for a, b, j in actions:
        if previous_end is None:
            previous_end = max(0, a - 1.15)
            previous_point = np.array(
                [0.06, 0.19, 0.20] if side else [-0.48, -0.275, 0.18]
            )
            first = round(previous_end * FPS)
            pos[:first] = initial[:first]
        target = grasp_point(items[j], a, side)
        lo = max(0, int(np.ceil(previous_end * FPS - 1e-7)))
        hi = min(FRAMES, int(np.ceil(a * FPS - 1e-7)))
        clear_start = (
            previous_point
            - tool_frame(base, previous_point)[:, 2] * 0.054
            + [0, 0, 0.068]
        )
        clear_end = target - tool_frame(base, target)[:, 2] * 0.061 + [0, 0, 0.061]
        span = a - previous_end
        withdraw = min(0.175, span * 0.21)
        approach = min(0.235, span * 0.26)
        for f in range(lo, hi):
            if owner[f] >= 0:
                continue
            t = T[f]
            elapsed = t - previous_end
            remain = a - t
            if elapsed < withdraw:
                u = ease((elapsed - 0.045) / (withdraw - 0.045))
                pos[f] = previous_point * (1 - u) + clear_start * u
                jaw[f] = PAD + (OPEN - PAD) * ease(elapsed / 0.085)
            elif remain < approach:
                u = ease((approach - remain) / (approach - 0.068))
                pred = grasp_point(items[j], t, side)
                pos[f] = (1 - u) * clear_end + u * pred
                jaw[f] = PAD + (OPEN - PAD) * ease(remain / 0.084)
            else:
                u = ease((elapsed - withdraw) / (span - withdraw - approach))
                pos[f] = (1 - u) * clear_start + u * clear_end
                pos[f, 2] += 0.069 * np.sin(np.pi * u)
                pos[f, 0] -= 0.022 * np.sin(np.pi * u)
                if not side:
                    pos[f, 1] -= 0.065 * np.sin(np.pi * u)
        previous_end = b
        previous_point = grasp_point(items[j], b, side)
    if b < T[-1]:
        for f in np.flatnonzero(T > b):
            u = ease((T[f] - b) / 0.25)
            pos[f] = previous_point + u * (
                -tool_frame(base, previous_point)[:, 2] * 0.055 + [0, 0, 0.085]
            )
            jaw[f] = PAD + (OPEN - PAD) * ease((T[f] - b) / 0.085)
    return pos, jaw, owner, first


def fit_wrist(robot, side, items, q, pos, owner):
    # Choose one grasp frame for the whole transfer. Test reach at pickup, lift
    # and release before committing to the wrist twist of a physical envelope.
    for j, r in enumerate(items):
        held = np.flatnonzero(owner == j)
        if not len(held) or (side and r.flip):
            continue
        a, b = held[0], held[-1]
        _, obj0 = r.pose(T[a])
        if len(held) < 8:
            continue
        _, neutral = robot.pose(q[a])
        best = None
        old_a, old_b = q[a].copy(), q[b].copy()
        _, obj1 = r.pose(T[b])
        correction = r.turn if side else 0.0
        for yaw in (
            -correction * 0.5,
            0.0,
            -correction,
            -correction * 0.8,
            -correction * 0.25,
        ):
            for pitch in (-0.54, 0.48, -0.26, 0.0, 0.76, -0.85):
                tool = rz(yaw) @ neutral @ Rotation.from_euler("x", pitch).as_matrix()
                start, e0 = robot.fit_pose(old_a, pos[a], tool, 34)
                end, e1 = robot.fit_pose(old_b, pos[b], obj1 @ obj0.T @ tool, 34)
                travel = np.max(abs(start - end))
                effort = np.linalg.norm(start - old_a) + np.linalg.norm(end - old_b)
                score = 1800 * (e0 + e1) + 0.15 * effort + 0.50 * travel
                if best is None or score < best[0]:
                    best = (score, start, end)
        _, start, end = best
        # Both endpoint configurations realize the same rigid grasp. The intermediate
        # wrist is free to pitch as the arm lifts; constrain position and joint continuity.
        for f in held:
            u = ease((f - a) / max(1, b - a))
            ref = q[f] + (1 - u) * (start - old_a) + u * (end - old_b)
            qi = np.clip(ref, *robot.bounds)
            for _ in range(24):
                point, _ = robot.pose(qi)
                if np.linalg.norm(point - pos[f]) < 0.00008:
                    break
                mj.mj_comPos(robot.model, robot.data)
                jp = np.empty((3, robot.model.nv))
                mj.mj_jac(robot.model, robot.data, jp, None, robot.local, robot.hand)
                jac = robot.rot @ jp[:, :6]
                dq = jac.T @ np.linalg.solve(
                    jac @ jac.T + 0.00012 * np.eye(3), pos[f] - point
                )
                qi = np.clip(qi + np.clip(dq, -0.12, 0.12), *robot.bounds)
            q[f] = qi
        q[a], q[b] = start, end
        # Blend the unconstrained approach and withdrawal into the two end frames.
        for target, sign, delta in [(a, -1, q[a] - old_a), (b, 1, q[b] - old_b)]:
            for distance in range(1, 13):
                f = target + sign * distance
                if f < 0 or f >= len(q) or owner[f] >= 0:
                    break
                q[f] = np.clip(q[f] + ease(1 - distance / 13) * delta, *robot.bounds)


def fit_turnover(robot, item, held, q, pos):
    # The sixth joint is coaxial with the midpoint between the pads. Roll it
    # during the lift, while steering the nose down into each supported seam.
    # The two jaw meshes exchange upper/lower roles without crossing the mat.
    pick, down = item.times(1)
    phase = (T[held] - pick) / (down - pick)
    # Use the full planned duration even for a pickup cut by the end of the shot.
    w = ease(phase)[:, None]
    roll = 0.40 * np.array([item.roll(T[f]) for f in held]) - 0.60 * np.pi * w[:, 0]
    best = None
    for pitch in (-0.56, -0.66, -0.74):
        nominal = np.array([robot.planar(pos[f], pitch) for f in held])
        nominal[:, 5] = np.deg2rad(36) + roll
        wrist_bend = gaussian_filter1d(nominal[:, 4], 2.8, mode="nearest")
        if len(held) > 1:
            progress = np.linspace(0, 1, len(held))
            wrist_bend += np.interp(
                progress, [0, 1], (nominal[:, 4] - wrist_bend)[[0, -1]]
            )
        for steer in (1.0, 0.65, 0.35, 0.0):
            ends = []
            for i in (0, -1):
                _, frame = robot.pose(nominal[i])
                az = np.arctan2(frame[1, 2], frame[0, 2])
                yaw = (item.heading0 - az + np.pi) % (2 * np.pi) - np.pi
                end, residual = robot.fit_pose(
                    nominal[i], pos[held[i]], rz(steer * yaw) @ frame, 36
                )
                if residual > 0.00012:
                    break
                ends.append(end[3:] - nominal[i, 3:])
            if len(ends) != 2:
                continue
            wrists = nominal[:, 3:].copy()
            wrists[:, 1] = wrist_bend
            wrists += (1 - w) * ends[0] + w * ends[1]
            trial = np.array(
                [robot.aim_wrist(wrist, pos[f]) for f, wrist in zip(held, wrists)]
            )
            reach = max(
                np.linalg.norm(robot.pose(row)[0] - pos[f])
                for f, row in zip(held, trial)
            )
            step = np.rad2deg(np.max(abs(np.diff(trial, axis=0)), initial=0.0))
            # Reject unreachable bent-wrist branches, then balance smoothness
            # against yaw and residual flex at the clamped edge.
            if reach > 0.00012:
                continue
            cost = 5 * max(0.0, step - 6.5) + (1 - steer) + abs(pitch + 0.56)
            if best is None or cost < best[0]:
                best = cost, trial
    if best is None:
        raise ValueError(f"No reachable right-hand turnover at {pick:.2f}s")
    best = best[1]
    q[held] = best
    if T[held[-1]] == down:
        _, obj0 = item.turnover_surface(pick)
        _, wrist0 = robot.pose(best[0])
        point, wrist = robot.pose(best[-1])
        _, flat = item.turnover_surface(down)
        item.release_pose = (
            point,
            Rotation.from_matrix(flat.T @ wrist @ wrist0.T @ obj0).as_rotvec(),
        )


def connect_turnovers(robot, items, owner, q, pos):
    held_runs = []
    for j, item in enumerate(items):
        held = np.flatnonzero(owner == j)
        if item.flip and held.size:
            fit_turnover(robot, item, held, q, pos)
        if held.size:
            held_runs.append((held[0], held[-1], item.flip))
    # Carry the wrist twist through withdrawal. Unwind while clear between
    # grasps and enter the next pinch with the same wrist frame on either side.
    held_runs.sort()
    for i, (a, b, flipped) in enumerate(held_runs):
        if not flipped:
            continue
        for end, direction in ((a, -1), (b, 1)):
            neighbour = i + direction
            other = (
                held_runs[neighbour][1 if direction < 0 else 0]
                if 0 <= neighbour < len(held_runs)
                else (0 if direction < 0 else len(q) - 1)
            )
            span = abs(end - other)
            count = min(
                40,
                (span - 1) // 2
                if 0 <= neighbour < len(held_runs) and held_runs[neighbour][2]
                else max(0, span - 8),
            )
            follow = min(8, count // 3)
            _, frame = robot.pose(q[end])
            last = q[end]
            for d in range(1, count + 1):
                f = end + direction * d
                if d <= follow:
                    trial, error = robot.fit_pose(last, pos[f], frame, 20)
                    if error > 0.00015 or max(abs(trial - last)) > 0.14:
                        # Keep the approach on the same branch even if the
                        # full endpoint orientation cannot reach the clear pose.
                        trial = np.clip(trial, last - 0.12, last + 0.12)
                    q[f] = last = trial
                else:
                    u = ease((d - follow) / (count - follow))
                    q[f] = (1 - u) * last + u * q[f]

    for (_, end, leaving), (begin, _, approaching) in itertools.pairwise(held_runs):
        if leaving or approaching:
            # Joint 6 rolls about the line through the pad midpoint. Preserve
            # that point and spread the reset across the entire empty-hand arc.
            lo = end + (1 if leaving else 3)
            hi = begin - (1 if approaching else 3)
            if hi > lo:
                u = np.linspace(0, 1, hi - lo + 1)
                ramp = 0.10
                w = (
                    u * u
                    - np.maximum(u - ramp, 0) ** 2
                    - np.maximum(u - 1 + ramp, 0) ** 2
                ) / (2 * ramp * (1 - ramp))
                q[lo : hi + 1, 5] = (1 - w) * q[lo, 5] + w * q[hi, 5]


def bank_blue_pouch(robot, items, owner, q):
    # The blue pouch arriving at 22.60 s tips its free edge down while the
    # printed face remains visible. Offsets were fit to its 30 Hz outline with
    # the original seam binding, PiPER limits and contact endpoints fixed.
    parcel = next(
        (i for i, r in enumerate(items) if r.kind == 1 and r.arrive == 22.6), None
    )
    frames = np.flatnonzero(owner == parcel) if parcel is not None else []
    if len(frames) < 3:
        return
    knots = [0, 3, 8, 13, 18, 23, 28, 33, 38, 43]  # 50 Hz frames since right closure
    offsets = np.deg2rad(
        [
            [0, 0, 0, 0, 0, 0],
            [-4.01, 5.23, -5.71, -6.31, -3.29, -14.85],
            [-2.99, 12.95, -5.54, 11.92, -18.77, -49.40],
            [0.15, 28.01, -12.31, 39.16, -46.72, -82.28],
            [11.54, 2.64, -3.14, 43.82, -38.15, -92.21],
            [29.70, -9.28, -18.65, 67.38, -21.58, -97.44],
            [37.31, -11.30, -24.27, 73.24, -10.22, -82.75],
            [20.53, -13.16, -10.80, 50.87, 9.49, -54.10],
            [7.31, -6.97, -14.27, 22.49, 25.09, -23.32],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    # Zero end slopes preserve the approach and withdrawal. Leave both
    # contact samples untouched, including the orientation used for binding.
    inside = frames[(frames > frames[0]) & (frames < frames[0] + knots[-1])]
    correction = CubicSpline(knots, offsets, bc_type="clamped")(inside - frames[0])
    q[inside] = np.clip(q[inside] + correction, *robot.bounds)


def clear_open_jaws(robot, q, jaws, owner):
    # Check the machined jaw vertices in world coordinates during approaches and returns.
    fingers = []
    for name in ("link7", "link8"):
        body = robot.model.body(name).id
        gi = next(
            i
            for i, b in enumerate(robot.model.geom_bodyid)
            if b == body and robot.model.geom_group[i] == 2
        )
        mid = robot.model.geom_dataid[gi]
        vi = robot.model.mesh_vertadr[mid]
        pts = robot.model.mesh_vert[vi : vi + robot.model.mesh_vertnum[mid]]
        fingers.append((gi, np.unique(pts, axis=0)))

    def deficit(f):
        robot.data.qpos[6:] = [jaws[f], -jaws[f]]
        robot.pose(q[f])
        bad = 0.0
        for gi, verts in fingers:
            pts = (
                verts @ robot.data.geom_xmat[gi].reshape(3, 3).T
                + robot.data.geom_xpos[gi]
            ) @ robot.rot.T + robot.base
            table = (
                (pts[:, 1] < -0.045) & (pts[:, 1] > -0.439) & (abs(pts[:, 0]) < 0.475)
            )
            belt = (pts[:, 1] >= 0.034) & (pts[:, 1] <= 0.768)
            floor = np.where(table, 0.0004, np.where(belt, 0.035, -2.0))
            bad = max(bad, np.max(floor + 0.0015 - pts[:, 2]))
        return bad

    free = np.flatnonzero(owner < 0)
    for _ in range(3):
        lift = np.zeros(len(q))
        for f in free:
            lift[f] = deficit(f)
        if lift.max() < 0.0003:
            break
        lift = np.maximum(lift, gaussian_filter1d(maximum_filter1d(lift, 5), 1.2))
        corrections = np.zeros_like(q)
        for f in free[lift[free] > 0.00015]:
            p, _ = robot.pose(q[f])
            target = p + [0, 0, lift[f] + 0.0005]
            trial = q[f].copy()
            for _step in range(12):
                point, _ = robot.pose(trial)
                mj.mj_comPos(robot.model, robot.data)
                jp = np.empty((3, robot.model.nv))
                mj.mj_jac(robot.model, robot.data, jp, None, robot.local, robot.hand)
                jac = robot.rot @ jp[:, :3]
                delta = jac.T @ np.linalg.solve(
                    jac @ jac.T + 0.0010 * np.eye(3), target - point
                )
                trial[:3] = np.clip(
                    trial[:3] + np.clip(delta, -0.05, 0.05), *robot.bounds[:, :3]
                )
            corrections[f] = trial - q[f]
        smooth = gaussian_filter1d(corrections, 1.0, axis=0, mode="nearest")
        q[free] = np.clip(
            q[free] + 0.65 * corrections[free] + 0.35 * smooth[free], *robot.bounds
        )


def solve(items):
    for item in items:
        if item.flip:
            item.release_pose = None
    old = np.load(ASSETS / "arms.npz")
    q = old["q"].copy()
    jaws = np.empty((2, FRAMES))
    owners = np.empty((2, FRAMES), int)
    xyz = np.empty((2, FRAMES, 3))
    basis = np.empty((2, FRAMES, 3, 3))
    for s in (0, 1):
        robot = Kinematics(s, old["base"][s], old["yaw"][s])
        for f in range(FRAMES):
            xyz[s, f], basis[s, f] = robot.pose(q[s, f])
        pos, jaws[s], owners[s], first = plan(s, items, old["base"][s], xyz[s].copy())
        robot.prepare_planar()
        for f in range(first, FRAMES):
            if np.linalg.norm(pos[f]) < 0.001:
                pos[f] = pos[min(f + 1, FRAMES - 1)]
            preferred = -0.93
            if owners[s, f] >= 0:
                item = items[owners[s, f]]
                aa, bb = item.times(s)
                u = (T[f] - aa) / (bb - aa)
                tilt = (
                    -0.43 * max(0, np.sin(np.pi * u)) ** 1.28
                    if s
                    else 0.20 * np.sin(np.pi * u)
                )
                preferred -= tilt
            q[s, f] = robot.planar(pos[f], preferred)
        fit_wrist(robot, s, items, q[s], pos, owners[s])
        actions = sorted([r.times(s) for r in items if r.arrive > 0])
        for (_, end), (begin, _) in itertools.pairwise(actions):
            start = round((end + 0.105) * FPS)
            stop = round((begin - 0.145) * FPS)
            if stop >= FRAMES or stop - start < 4:
                continue
            q0 = q[s, start].copy()
            q1 = q[s, stop].copy()
            yaw = (q0[0] + q1[0]) * 0.5
            mid = robot.planar(
                robot.base
                + robot.rot
                @ np.array([0.345 * np.cos(yaw), 0.345 * np.sin(yaw), 0.285])
            )
            for f in range(start + 1, stop):
                u = ease((f - start) / (stop - start))
                q[s, f] = (1 - u) ** 2 * q0 + 2 * u * (1 - u) * mid + u * u * q1
        recorded = gaussian_filter1d(old["q"][s], 2.25, axis=0, mode="nearest")
        for (_, end), (begin, _) in itertools.pairwise(actions):
            lo = max(0, round((end + 0.08) * FPS))
            hi = round((begin - 0.08) * FPS)
            if hi <= lo:
                continue
            for f in range(lo, min(FRAMES, hi + 1)):
                w = (0.64 if s == 0 else 0.47) * np.sin(
                    np.pi * (f - lo) / (hi - lo)
                ) ** 2
                q[s, f] = (1 - w) * q[s, f] + w * recorded[f]
        # Distribute sharp wrist turns over neighbouring samples inside the same
        # contact phase. Keep the pickup and release transforms exactly anchored.
        marks = sorted(
            set(
                [first, FRAMES - 1]
                + [
                    int(np.clip(round(t * FPS), first, FRAMES - 1))
                    for a, b in actions
                    for t in (a - 0.16, a, b, b + 0.16)
                ]
            )
        )
        for lo, hi in itertools.pairwise(marks):
            if hi - lo < 6:
                continue
            seg = q[s, lo : hi + 1].copy()
            weight = np.maximum(
                1, np.max(np.abs(np.diff(seg, axis=0)), axis=1) / np.deg2rad(5.5)
            )
            u = np.r_[0, np.cumsum(weight)]
            u *= (hi - lo) / u[-1]
            smooth = np.column_stack(
                [np.interp(np.arange(len(seg)), u, col) for col in seg.T]
            )
            filtered = gaussian_filter1d(smooth, 1.25, axis=0, mode="nearest")
            blend = ease(
                np.minimum(np.arange(len(seg)), np.arange(len(seg))[::-1]) / 4
            )[:, None]
            q[s, lo : hi + 1] = smooth + blend * (filtered - smooth)
        for f in range(first - 38, first):
            u = ease((f - first + 38) / 38)
            q[s, f] = (1 - u) * q[s, f] + u * q[s, first]
        if s == 1:
            connect_turnovers(robot, items, owners[s], q[s], pos)
        clear_open_jaws(robot, q[s], jaws[s], owners[s])
        if s == 1:
            bank_blue_pouch(robot, items, owners[s], q[s])
        if s == 0:
            pinch = np.flatnonzero(owners[s] == 30)
            if pinch.size:
                # During the lateral sweep toward P030, lift the open lower jaw
                # over the settled edge. Taper to zero before the closing approach.
                lead = pinch[0] - np.arange(FRAMES)
                sweep = (owners[s] < 0) & (lead > 9) & (lead < 22)
                phase = (22 - lead[sweep]) / 13
                q[s, sweep, 1] -= np.deg2rad(3.0) * np.sin(np.pi * phase) ** 2
        for f in range(first - 38, FRAMES):
            xyz[s, f], basis[s, f] = robot.pose(q[s, f])
        err = np.linalg.norm(xyz[s] - pos, axis=1)
        active = owners[s] >= 0
        print(
            "side",
            s,
            "displacement from nominal path after joint retiming (mm)",
            np.round(1000 * np.quantile(err[active], [0.5, 0.95, 1]), 2),
            "peak joint increment deg",
            round(np.rad2deg(np.max(abs(np.diff(q[s], axis=0)))), 2),
            flush=True,
        )
    return {
        "q": q,
        "jaw": jaws,
        "owner": owners,
        "xyz": xyz,
        "basis": basis,
        "base": old["base"],
        "yaw": old["yaw"],
    }


if __name__ == "__main__":
    solve(routes())
