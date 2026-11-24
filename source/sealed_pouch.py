"""Two sheets of embossed film with a heat-sealed rim and time-dependent local pinches."""

from functools import lru_cache

import numpy as np
from parcel_routes import ASSETS, FPS, PAD, PINCH_Z, TABLE, ease
from pxr import UsdGeom, UsdShade, Vt
from scipy.spatial.transform import Rotation


def vectors(points):
    return Vt.Vec3fArray.FromNumpy(
        np.ascontiguousarray(points.reshape(-1, 3), dtype="float32")
    )


def normals(points):
    dy, dx = np.gradient(points, axis=(1, 2))
    n = np.cross(dx, dy)
    n[1] *= -1
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


@lru_cache(maxsize=1)
def _profiles(path, mtime_ns):
    # The timestamp in the cache key also picks up replacements in a live session.
    rows = np.loadtxt(path, ndmin=2, encoding="utf-8-sig")
    if rows.shape[1] != 7 or not np.isfinite(rows).all():
        raise ValueError(f"{path}: expected seven finite numeric columns")
    profiles = {}
    for ident, peak, du, dv, su, sv, exponent in rows:
        if ident != int(ident) or ident in profiles:
            raise ValueError(f"{path}: invalid or duplicate parcel ID {ident}")
        if not (
            peak > 1
            and max(abs(du), abs(dv)) <= 1
            and 0.45 <= min(su, sv) <= max(su, sv) <= 0.95
            and 1.6 <= exponent <= 4
        ):
            raise ValueError(f"{path}: invalid payload profile for parcel {ident:g}")
        profiles[int(ident)] = (peak * 0.001, du, dv, su, sv, exponent)
    return profiles


def rest_shape(item, ident):
    path = ASSETS / "pouch_profiles.txt"
    peak, du, dv, su, sv, exponent = _profiles(path, path.stat().st_mtime_ns)[ident]
    # The reference fit supplies each bag's baseline peak and fill footprint.
    # Give shallow fills extra loft in proportion to the narrower physical
    # payload dimension. The fullest reference profiles keep their peak.
    shallow = ease((0.052 - peak) / 0.029)
    peak += shallow * min(0.010, 0.065 * min(item.length * su, item.width * sv))
    nx, ny = 49, 41
    u, v = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    rng = np.random.default_rng(1709 + ident)
    phase = rng.uniform(0, 2 * np.pi, 5)
    half_x, half_y = item.length / 2, item.width / 2
    x, y = half_x * u, half_y * v
    # Round just the cut corners, using a circular arc in physical units.
    radius = 0.0034
    cx = np.maximum(abs(x) - half_x + radius, 0)
    cy = np.maximum(abs(y) - half_y + radius, 0)
    scale = np.minimum(1, radius / np.maximum(np.hypot(cx, cy), 1e-9))
    x -= np.sign(x) * cx * (1 - scale)
    y -= np.sign(y) * cy * (1 - scale)
    x += 0.0011 * u**3 * np.cos(3.3 * v + phase[0])
    y += 0.0016 * v**3 * np.cos(3.9 * u + phase[1])
    ex, ey = (1 - abs(u)) * half_x, (1 - abs(v)) * half_y
    # A bulky fill opens the gusset over a wider strip. End welds and the
    # folded closure have independent cut widths in physical units.
    sx = 0.0040 + 0.0006 * np.sin(2.8 * v + phase[2])
    sy = np.where(v < 0, 0.0082, 0.0042)
    side = ease((ex - sx) / (0.010 + 0.29 * peak))
    end = ease((ey - sy) / (0.014 + 0.18 * peak))
    body = side * end
    # The superellipse describes the payload in material (UV) coordinates.
    radius_power = abs((u - du) / su) ** exponent + abs((v - dv) / sv) ** exponent
    payload = np.exp(-(1.45 - 0.32 * shallow) * radius_power)
    # Thin fills spread across more of the midbody. The bridge follows the
    # payload footprint and tapers to zero at the same heat-seal boundary.
    thin_fill = 1 - ease((peak - 0.013) / 0.042)
    shoulder = (0.19 + 0.045 * thin_fill + 0.14 * np.exp(-0.37 * radius_power)) * body
    crown = body * payload + shoulder * (1 - payload)
    crease = np.zeros_like(x)
    for sign in (-1, 1):
        axis = v * sign
        for _ in range(3):
            xu = rng.uniform(-0.88, 0.88)
            line = x - half_x * xu + rng.uniform(-0.62, 0.62) * (1 - axis) * half_y
            spread = rng.uniform(0.0018, 0.0036)
            crease -= (
                rng.uniform(0.00065, 0.0017)
                * np.exp(-0.5 * (line / spread) ** 2)
                * np.exp(-(((1 - axis) / 0.52) ** 2))
            )
    crown *= 1 + (1 - payload) * crease / peak
    if ident == 9:
        # At 21.167 s the middle of the near weld tucks back between two
        # shoulders. After landing this is the local -V edge of P009.
        tuck = np.exp(-0.5 * (u / 0.15) ** 2) * np.maximum(-v, 0) ** 3
        y += 0.009 * tuck
        crown *= 1 - 0.32 * tuck
    crown /= crown.max()

    # The sole lies flat under the contents, then rolls up under each
    # shoulder. The resulting overhang casts its own contact shadow.
    side_roll = (1 - side) * (0.075 + 0.025 * np.cos(3.1 * v + phase[0]))
    end_roll = (1 - end) * (0.060 + 0.020 * np.cos(2.8 * u + phase[1]))
    # Join intersecting rolls at the corners without a ridge along a diagonal.
    roll = side_roll + end_roll - 6 * side_roll * end_roll
    curl = (0.0007 + 0.0015 * (0.5 + 0.5 * np.cos(4.3 * v + phase[4]))) * (
        1 - ease(ex / 0.009)
    )
    flap = np.maximum(-v, 0) ** 4 * (1 - ease(ey / 0.013))
    curl += flap * (0.0015 + 0.0011 * np.sin(3.3 * u + phase[3]))
    clearance = (peak * (roll + 0.060 * (1 - payload)) + curl) * (
        1 - ease(crown / 0.62)
    )
    # The payload lifts the inner shoulder; slack film beyond it drapes back
    # onto the support. Some sections touch and others retain a curled lip.
    lifted_lip = ease((np.cos(3.3 * u + 4.3 * v + phase[4]) + 0.25) / 1.25)
    seam_height = (0.0012 + 0.5 * curl) * lifted_lip
    shoulder_blend = ease(np.minimum(ex / sx, ey / sy) * 0.5)
    clearance = shoulder_blend * clearance + (1 - shoulder_blend) * seam_height
    bottom = -0.00155 + clearance
    # Normalise on this sampled mesh, including both skins and the weld.
    # At the maximum the underside is exactly on the contact plane.
    weld = 0.00075 + flap * 0.00045
    top = bottom + weld + (peak - weld) * crown
    rest = np.stack([np.stack([x, y, top], -1), np.stack([x, y, bottom], -1)])
    return rest, u, v


def grip_patch(rest, anchor, binding):
    # Move the two skins at closure. The cut edge retains its unpinched XY outline.
    xy = rest[0, ..., :2]
    dist = np.linalg.norm(xy - anchor[:2], axis=-1)
    iy, ix = np.unravel_index(dist.argmin(), dist.shape)
    center = rest[:, iy, ix, :]
    across = binding.T[:, 1].copy()
    if across[2] < 0:
        across *= -1
    target = anchor + np.array([1, -1])[:, None] * PAD * across
    delta = target - center
    lateral = np.exp(-0.5 * (dist / 0.013) ** 2)
    # Taper tangential shifts to zero at the cut edge.
    rim = np.minimum.reduce(
        [
            abs(xy[..., 0] - xy[:, 0:1, 0]),
            abs(xy[..., 0] - xy[:, -1:, 0]),
            abs(xy[..., 1] - xy[0:1, :, 1]),
            abs(xy[..., 1] - xy[-1:, :, 1]),
        ]
    )
    lateral *= ease(rim / 0.007)
    lateral /= lateral[iy, ix]
    # The two pads slide the films oppositely only inside the locally
    # flattened pinch. On a steep shoulder that slide could cut through
    # the other skin if applied over the entire lifting patch.
    pad_edge = ease((dist - max(0.003, dist[iy, ix])) / 0.006)
    slide = lateral * (1 - pad_edge)
    lift = np.exp(-0.5 * (np.maximum(dist - 0.014, 0) / 0.017) ** 2)
    shift = np.empty_like(rest)
    shift[..., :2] = delta[:, :2].mean(axis=0) * lateral[..., None] + (
        (target[:, :2] - anchor[:2])[:, None, None, :] * slide[..., None]
    )
    # Blend the local thickness itself so the skins cannot cross at a thin weld.
    thickness = rest[0, ..., 2] - rest[1, ..., 2]
    squeeze = (target[0, 2] - target[1, 2] - thickness) * np.minimum(lateral, 1)
    # Turn the pinched strip toward the jaws. At a grazing grip the pad
    # separation is mostly tangential in the original sheet coordinates.
    tilt = -0.55 * ((xy - xy[iy, ix]) @ across[:2])
    mid = (anchor[2] - rest[..., 2].mean(axis=0) + tilt) * lift
    shift[..., 2] = mid + np.array([0.5, -0.5])[:, None, None] * squeeze
    return iy * xy.shape[1] + ix, shift, dist, lift


def chute_guard(stage):
    guard = stage.GetPrimAtPath("/World/Cell/Bench/OutsideGuard0")
    if not guard:
        return None
    bounds = (
        UsdGeom.Boundable(guard).ComputeWorldBound(0, "default").ComputeAlignedRange()
    )
    return np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())


def guard_crossing_height(world, faces, lo, hi, rim=0.0):
    low = world.min(axis=(0, 1, 2))
    high = world.max(axis=(0, 1, 2))
    if (low[:2] > hi[:2] + rim).any() or (high[:2] < lo[:2] - rim).any():
        return None
    # Include faces straddling the thin plate even if no vertex is inside it.
    quads = world.reshape(-1, 3)[faces]
    near = np.all(quads.max(axis=1)[:, :2] >= lo[:2] - rim, axis=1) & np.all(
        quads.min(axis=1)[:, :2] <= hi[:2] + rim, axis=1
    )
    if not near.any():
        return None
    return quads[near, :, 2].min() - hi[2]


def create(g, path, mat, item, motion, ident, bindings, poses):
    rest, u, v = rest_shape(item, ident)
    flip = getattr(item, "flip", False)
    if flip:
        # Both jaw anchors are on local z=0. Motion raises the origin by
        # PINCH_Z so that the belt and mat still see the measured bag height.
        rest[..., 2] -= PINCH_Z
        # Reflect the rest heights and exchange the load-bearing/free skins
        # without reversing their physical order or changing local thickness.
        landing_origin = TABLE + PINCH_Z
        inverted_offset = (
            landing_origin + rest[1, ..., 2].min() - rest[..., 2].sum(axis=0)
        )
    ny, nx = u.shape
    count = u.size
    corners = np.arange(count).reshape(ny, nx)[:-1, :-1].ravel()
    front = corners[:, None] + np.array([0, 1, nx + 1, nx])
    # Reverse the winding around the same first corner. Either diagonal then
    # spans the same physical edge on both skins, even when a weld twists.
    back = front[:, [0, 3, 2, 1]] + count
    border = np.r_[
        np.arange(nx),
        np.arange(2 * nx - 1, count, nx),
        np.arange(count - 2, count - nx - 1, -1),
        np.arange(count - 2 * nx, 0, -nx),
    ]
    following = np.roll(border, -1)
    wall = np.column_stack([following, border, border + count, following + count])
    faces = np.vstack([front, back, wall])
    uv = np.tile(np.c_[u.ravel() * 0.5 + 0.5, v.ravel() * 0.5 + 0.5], (2, 1))
    if flip:
        # Landing is a half-turn about X: -Y on the reverse reads as +V.
        uv[count:, 1] = 1 - uv[count:, 1]
    plain = f"film{item.kind}_0"
    mesh = g.mesh(
        path + "/Shell",
        rest.reshape(-1, 3),
        faces,
        plain if flip else mat,
        normals(rest),
        uv,
    )
    if flip:
        # label records the settled appearance. Walls always inherit plain film.
        first_face = len(front) if item.label else 0
        subset = UsdShade.MaterialBindingAPI(mesh).CreateMaterialBindSubset(
            "PrintedSkin",
            Vt.IntArray.FromNumpy(
                np.arange(first_face, first_face + len(front), dtype=np.int32)
            ),
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
            g.materials[f"film{item.kind}_1"]
        )
    elif item.label:
        subset = UsdShade.MaterialBindingAPI(mesh).CreateMaterialBindSubset(
            "UnprintedBack",
            Vt.IntArray.FromNumpy(np.arange(len(front), len(faces), dtype=np.int32)),
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(g.materials[plain])
    mesh.GetExtentAttr().Set(
        vectors(np.array([rest.min((0, 1, 2)) - 0.035, rest.max((0, 1, 2)) + 0.035]))
    )
    patches = {}
    pinch_lifts = {}
    bend_frames = range(0)
    keyframes = set()
    thickness = rest[0, ..., 2] - rest[1, ..., 2]
    peak = thickness.max()
    bulk = ease((peak - 0.013) / 0.042)
    # Under the contents both sheets move together. A narrow payload leaves
    # more free film to bend between the pinched point and the opposite corner.
    free_film = 1 - (0.30 + 0.50 * bulk) * ease(thickness / peak)
    for side, binding in enumerate(bindings):
        if binding is None:
            continue
        held = np.flatnonzero(motion["owner"][side] == ident)
        center, shift, dist, lift = grip_patch(rest, item.anchor(side), binding)
        if flip:
            # Reuse the grip patch's falloff to hold its centre at the pads
            # as the free sheets redistribute their crown during turnover.
            pinch_lifts[side] = lift
        mesh.GetPrim().SetCustomDataByKey(
            ("leftPadVertices", "rightPadVertices")[side],
            Vt.IntArray([int(center), int(center + count)]),
        )
        first, last = int(held[0]), int(held[-1])
        if flip and side == 1:
            bend_frames = range(first, last + 8)
            # The seam between the jaws follows FK. Over the next 3 cm the
            # flexible envelope can lay down while the wrist is still pitched.
            bend_weight = ease((dist - 0.010) / 0.030)
            right_anchor = item.anchor(1)
            keyframes.update(bend_frames)
        patches[side] = (
            first,
            last,
            shift,
            ease((dist - 0.010) / (0.90 * max(item.length, item.width))) * free_film,
        )
        keyframes.update(
            np.r_[held[::4], last, first - 7, first - 4, last + 4, last + 7].tolist()
        )
    # Track the acceleration of the free corners with damped elastic lag.
    # Integrate all four in world space; the clamped patch stays rigid.
    weights = (
        np.stack(
            [
                (1 - u) * (1 - v),
                (1 + u) * (1 - v),
                (1 + u) * (1 + v),
                (1 - u) * (1 + v),
            ],
            axis=-1,
        )
        * 0.25
    )
    anchors = rest[0, [0, 0, -1, -1], [0, -1, -1, 0]]
    spring = 235 + 385 * bulk
    damping = 1.35 * np.sqrt(spring)
    sag_limit = 0.026 - 0.015 * bulk
    lag = {}
    for first, last, shift, weight in patches.values():
        d = np.zeros((4, 3))
        vel = d.copy()
        for f in range(first - 2, last + 17):
            if any(i not in poses for i in (f - 1, f, f + 1)):
                continue
            world = [anchors @ poses[i][1].T + poses[i][0] for i in (f - 1, f, f + 1)]
            accel = (world[0] - 2 * world[1] + world[2]) * FPS**2
            # World drag and the resistance of the thicker core attenuate high acceleration.
            drive = -0.38 * np.clip(accel, -24, 24) if first <= f <= last + 2 else 0.0
            for _ in range(4):
                vel += (drive - damping * vel - spring * d) / (4 * FPS)
                d += vel / (4 * FPS)
            disp = (weights @ np.clip(d, -0.017, 0.017)) @ poses[f][1]
            lag[f] = weight[..., None] * disp
            keyframes.add(f)
        if last + 17 in poses:
            keyframes.add(last + 17)
    # USD interpolates between samples, so explicitly key the undeformed sheet either side of contact.
    guard_exit_plan = None
    guard = chute_guard(g.s) if 0 in patches else None
    if guard is not None:
        free_edge = ease(
            (np.linalg.norm(rest[0, ..., :2] - item.anchor(0)[:2], axis=-1) - 0.012)
            / 0.020
        )
    for f in sorted(keyframes):
        if f not in poses:
            continue
        shape = rest.copy()
        c, rot = poses[f]
        if flip:
            core_rot = item.turnover_surface(f / FPS)[1] if f in bend_frames else rot
            offset = ease((1 - core_rot[2, 2]) * 0.5) * inverted_offset
            shape[..., 2] += offset
        for side, (first, last, shift, weight) in patches.items():
            strength = ease((f - first + 7) / 7) * ease((last + 7 - f) / 7)
            shape += strength * shift
            if flip:
                shape[..., 2] -= strength * offset * pinch_lifts[side]
            if first < f < last:
                gravity = rot[2]
                h = np.min(shape.reshape(-1, 3) @ gravity + c[2])
                sag = min(
                    sag_limit * np.sin(np.pi * (f - first) / (last - first)) ** 1.45,
                    max(0, h - 0.036),
                )
                shape -= sag * weight[None, :, :, None] * gravity
        if f in lag:
            shape += lag[f][None, ...]
        if f in bend_frames:
            rv = Rotation.from_matrix(rot.T @ core_rot).as_rotvec()
            bend = Rotation.from_rotvec(bend_weight.reshape(-1, 1) * rv).as_matrix()
            shape = right_anchor + np.einsum(
                "vij,svj->svi", bend, (shape - right_anchor).reshape(2, count, 3)
            ).reshape(shape.shape)
        # During a turn either skin can hit a support first, including two
        # laterally separated vertices across the tilted pinch.
        lower = (shape if flip else shape[1]) @ rot.T + c
        if flip and 0 in patches and f > patches[0][1] and c[0] < -0.55:
            # The root descends into the chute after the left jaws open.
            # Slide the trailing weld past the bench lip during the initial
            # free flight; lifting it vertically would stretch it across the lip.
            exit_blend = ease((f - patches[0][1]) / 7)
            overhang = max(0.0, lower[..., 0].max() + 0.481)
            slide = overhang * exit_blend
            shape -= slide * rot[0]
            lower[..., 0] -= slide
        table = (
            (lower[..., 1] < -0.045)
            & (lower[..., 1] > -0.439)
            & (lower[..., 0] > (-0.480 if flip else -0.475))
            & (lower[..., 0] < 0.475)
        )
        belt = (lower[..., 1] >= 0.034) & (lower[..., 1] <= 0.766)
        support = np.where(table, 0.0 if flip else 0.00034, np.where(belt, 0.035, -2.0))
        # After the left claw releases beyond the workbench, the whole envelope
        # is falling into the chute. Its trailing corner cannot stay on the mat.
        if not flip and c[0] < -0.55 and c[1] < -0.10:
            support = np.minimum(support, -0.279)
        push = np.maximum(0, support - lower[..., 2])
        if flip:
            push = push.max(axis=0)
        # Both skins are lifted together to preserve the volume near first touch.
        shape += push[None, :, :, None] * rot[2]
        if f in bend_frames:
            # At touchdown the inclined seam can shear tangentially against
            # the flattened core. Slide the films apart along the support,
            # preserving both world heights and the midsurface.
            dy, dx = np.gradient(shape.mean(axis=0), axis=(0, 1))
            normal = np.cross(dx, dy)
            normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
            gap = ((shape[0] - shape[1]) * normal).sum(axis=-1)
            horizontal = normal - (normal @ rot[2])[..., None] * rot[2]
            slip = np.maximum(0, 0.0002 - gap) / np.maximum(
                (horizontal**2).sum(axis=-1), 0.0025
            )
            slide = 0.5 * slip[..., None] * horizontal
            shape[0] += slide
            shape[1] -= slide
        if guard is not None and patches[0][0] <= f <= patches[0][1]:
            lo, hi = guard
            world = shape @ rot.T + c
            released = f == patches[0][1]
            # Include the adjacent strip of film that sweeps over the thin
            # guard between USD ticks. At release test the actual plate.
            height = guard_crossing_height(world, faces, lo, hi)
            if not released and (height is None or height >= 0.002):
                adjacent = guard_crossing_height(world, faces, lo, hi, 0.008)
                if adjacent is not None and adjacent < 0:
                    height = adjacent
            if height is not None:
                if released:
                    ticks = max(1, int(FPS * np.sqrt(2 * max(0, height) / 9.81)))
                    guard_exit_plan = lo[0] - 0.0005, ticks
                elif height < 0.002:
                    # A held bag's remote corner can brush the guard first.
                    # Raise paired vertices along the sheet normal, keeping
                    # their local XY, separation and the clamped seam fixed.
                    xy = world[..., :2]
                    outside = (
                        np.maximum(lo[:2] - xy, xy - hi[:2]).max(axis=-1).min(axis=0)
                    )
                    weight = free_edge * (1 - ease((outside - 0.008) / 0.016))
                    lift = np.maximum(0, hi[2] + 0.002 - world[..., 2].min(axis=0))
                    shape[..., 2] += lift * weight / rot[2, 2]
        elif guard_exit_plan is not None:
            # The trailing corner must cross the chute-side face before it
            # falls to the guard top. Translate both skins as one volume.
            edge, ticks = guard_exit_plan
            overhang = (shape @ rot[0]).max() + c[0] - edge
            if overhang > 1e-8:
                shape -= ease((f - patches[0][1]) / ticks) * overhang * rot[0]
        mesh.GetPointsAttr().Set(vectors(shape), int(f))
        mesh.GetNormalsAttr().Set(vectors(normals(shape)), int(f))
        if flip:
            mesh.GetExtentAttr().Set(
                vectors(np.array([shape.min((0, 1, 2)), shape.max((0, 1, 2))])), int(f)
            )
    return mesh
