"""Verify actual USD geometry for conveyor entry, both pinch pads, rigid retention, and release."""

import argparse
from pathlib import Path

import numpy as np
from cell import FPS, FRAMES, project
from parcel_routes import PAD_Z, SPEED, routes
from pxr import Usd, UsdGeom


def matrix(cache, prim):
    return np.asarray(cache.GetLocalToWorldTransform(prim))


def inner_surfaces(stage, wrist, cache):
    # Intersect the actual triangles along the closing axis, at the pad centre.
    hits = []
    for link in ("link7", "link8"):
        finger = stage.GetPrimAtPath(wrist.GetPath().AppendChild(link))
        geom = next(p for p in finger.GetChildren() if p.IsA(UsdGeom.Mesh))
        mesh = UsdGeom.Mesh(geom)
        points = np.asarray(mesh.GetPointsAttr().Get())
        transform = matrix(cache, geom) @ np.linalg.inv(matrix(cache, wrist))
        local = points @ transform[:3, :3] + transform[3, :3]
        tri = local[np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        d = np.array([0, PAD_Z]) - tri[:, 0][:, [0, 2]]
        det = a[:, 0] * b[:, 2] - a[:, 2] * b[:, 0]
        safe = np.where(abs(det) > 1e-13, det, 1.0)
        u = (d[:, 0] * b[:, 2] - d[:, 1] * b[:, 0]) / safe
        v = (a[:, 0] * d[:, 1] - a[:, 2] * d[:, 0]) / safe
        inside = (abs(det) > 1e-13) & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.000001)
        ys = (tri[:, 0, 1] + u * a[:, 1] + v * b[:, 1])[inside]
        assert len(ys) >= 2, "Missing jaw pad triangle"
        hits.append(ys[np.argmin(abs(ys))])
    return np.sort(hits)


def audit(path):
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    items = routes()
    assert stage.GetTimeCodesPerSecond() == 50 and stage.GetEndTimeCode() == 2999
    parcels = stage.GetPrimAtPath("/World/Parcels").GetChildren()
    assert len(parcels) == len(items)
    wrists = [
        stage.GetPrimAtPath(
            "/World/" + s + "Arm/base_link/link1/link2/link3/link4/link5/link6"
        )
        for s in ("Left", "Right")
    ]
    maxima = np.zeros(3)
    held_samples = 0
    counts = [0, 0]
    arrivals = []
    departures = []
    for j, (prim, item) in enumerate(zip(parcels, items)):
        shell = prim.GetChild("Shell") or prim.GetChild("Carton")
        mesh = UsdGeom.Mesh(shell)
        pts = np.asarray(mesh.GetPointsAttr().Get())
        vis = UsdGeom.Imageable(prim).GetVisibilityAttr()
        # Use real vertices in the parent frame, including the carton height.
        cache.SetTime(0)
        shape = matrix(cache, shell) @ np.linalg.inv(matrix(cache, prim))
        p = pts @ shape[:3, :3] + shape[3, :3]
        extreme = np.array(
            [
                [x, y, z]
                for x in (p[:, 0].min(), p[:, 0].max())
                for y in (p[:, 1].min(), p[:, 1].max())
                for z in (p[:, 2].min(), p[:, 2].max())
            ]
        )
        visible = []
        belt_positions = []
        pickup = round(item.times(1)[0] * FPS) if item.arrive > 0 else FRAMES
        for f in range(FRAMES):
            if vis.Get(f) != "inherited":
                continue
            cache.SetTime(f)
            m = matrix(cache, prim)
            uv = project(extreme @ m[:3, :3] + m[3, :3])
            if (
                uv[:, 0].max() >= 0
                and uv[:, 0].min() < 640
                and uv[:, 1].max() >= 0
                and uv[:, 1].min() < 480
            ):
                visible.append((f, uv[:, 0].min(), uv[:, 0].max()))
                if f < pickup:
                    belt_positions.append(m[3, :3].copy())
        if not visible:
            raise AssertionError(f"Parcel {j} never appeared")
        # From first exposure to pickup or exit, require each 20 ms belt step.
        steps = np.diff(belt_positions, axis=0)
        assert len(steps) > 0 and np.allclose(steps, [SPEED / FPS, 0, 0], atol=2e-6), (
            j,
            "parcel speed changed on belt",
        )
        first, xlo, xhi = visible[0]
        arrivals.append(first)
        assert xlo < 0 and xhi < 12, (j, "not a left-edge entry", first, xlo, xhi)
        last, xlo, xhi = visible[-1]
        if last < FRAMES - 1:
            assert xhi < 42 or xlo > 620, (j, "interior exit", last, xlo, xhi)
            departures.append(last)
        for f in vis.GetTimeSamples():
            cache.SetTime(f)
            m = matrix(cache, prim)
            uv = project(extreme @ m[:3, :3] + m[3, :3])
            assert uv[:, 0].max() < 0 or uv[:, 0].min() > 640, (
                j,
                "visibility switch inside camera",
                f,
            )
        if item.arrive < 0:
            continue
        for s, side in enumerate(("left", "right")):
            begin, end = [round(v * FPS) for v in item.times(s)]
            if begin >= FRAMES:
                continue
            counts[s] += 1
            indices = list(shell.GetCustomDataByKey(side + "PadVertices"))
            assert len(indices) == 2
            cache.SetTime(begin)
            expected = inner_surfaces(stage, wrists[s], cache)
            assert expected[0] < -0.001 and expected[1] > 0.001
            joint = (
                wrists[s].GetChild("link7").GetAttribute("xformOp:translate:jawSlide")
            )
            assert np.linalg.norm(joint.Get(begin - 6)) > 0.007
            if end + 6 < FRAMES:
                assert np.linalg.norm(joint.Get(end + 6)) > 0.007
            rel0 = None
            for f in range(begin, min(FRAMES, end + 1)):
                cache.SetTime(f)
                relative = matrix(cache, shell) @ np.linalg.inv(
                    matrix(cache, wrists[s])
                )
                pp = (
                    np.asarray(mesh.GetPointsAttr().Get(f))[indices] @ relative[:3, :3]
                    + relative[3, :3]
                )
                if rel0 is None:
                    rel0 = relative.copy()
                drift = np.linalg.norm(relative[3, :3] - rel0[3, :3])
                spin = np.linalg.norm(relative[:3, :3] - rel0[:3, :3])
                miss = max(
                    np.abs(np.sort(pp[:, 1]) - expected).max(),
                    np.abs(pp[:, 0]).max(),
                    np.abs(pp[:, 2] - PAD_Z).max(),
                )
                maxima = np.maximum(maxima, [drift, spin, miss])
                held_samples += 1
                assert drift < 0.00025 and spin < 0.001 and miss < 0.00025, (
                    j,
                    s,
                    f,
                    drift,
                    spin,
                    miss,
                )
    assert counts == [25, 26]
    print(
        f"{len(arrivals)} left-edge entries, {len(departures)} edge exits; grips left/right {counts}; {held_samples} held frames",
        flush=True,
    )
    print(
        "max fixed-grasp translation mm",
        maxima[0] * 1000,
        "rotation matrix drift",
        maxima[1],
        "max pad/surface discrepancy mm",
        maxima[2] * 1000,
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", type=Path)
    audit(p.parse_args().stage)
