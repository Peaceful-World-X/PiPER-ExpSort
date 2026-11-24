"""Check the visible skins of the animated USD against the reference transfers."""

import argparse
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

from cell import EYE, FPS, FRAMES
from parcel_routes import ASSETS, routes


class Pouch:
    def __init__(self, mesh):
        self.mesh = mesh
        uv = np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get())
        self.columns = int(np.flatnonzero(uv[:, 1] != uv[0, 1])[0])
        self.rows = len(uv) // (2 * self.columns)
        plane = (self.rows - 1) * (self.columns - 1)
        center = (self.rows // 2) * (self.columns - 1) + self.columns // 2
        # Resolve the actual face bindings, including overrides on either skin.
        binding = UsdShade.MaterialBindingAPI(mesh)
        base, _ = binding.ComputeBoundMaterial()
        materials = [base, base]
        for part in binding.GetMaterialBindSubsets():
            faces = set(part.GetIndicesAttr().Get())
            for side in range(2):
                if center + side * plane in faces:
                    materials[side], _ = UsdShade.MaterialBindingAPI(
                        part
                    ).ComputeBoundMaterial()
        self.printed = [m.GetPrim().GetName().endswith("_1") for m in materials]

    def orientation(self, tick):
        points = np.asarray(self.mesh.GetPointsAttr().Get(tick)).reshape(
            2, self.rows, self.columns, 3
        )
        core = points[
            0, self.rows // 4 : -(self.rows // 4), self.columns // 4 : -(self.columns // 4)
        ]
        dy, dx = np.gradient(core, axis=(0, 1))
        normal = np.cross(dx, dy).mean(axis=(0, 1))
        matrix = np.asarray(self.mesh.ComputeLocalToWorldTransform(tick))
        normal = normal @ matrix[:3, :3]
        normal /= np.linalg.norm(normal)
        center = core.mean(axis=(0, 1)) @ matrix[:3, :3] + matrix[3, :3]
        view = EYE - center
        view /= np.linalg.norm(view)
        return normal, normal @ view


def audit(path):
    stage = Usd.Stage.Open(str(path))
    items = routes()
    observations = np.loadtxt(ASSETS / "turnovers.txt", ndmin=2, encoding="utf-8-sig")
    audited = set()
    angles = []
    edge_errors = []
    for ident, entry, landed, flip, edge_time, *_ in observations:
        ident = int(ident)
        item = items[ident]
        if item.arrive <= 0:
            continue
        assert ident not in audited, ident
        audited.add(ident)
        pouch = Pouch(UsdGeom.Mesh.Get(stage, f"/World/Parcels/P{ident:03}/Shell"))
        start, stop = (round(s * FPS) for s in item.times(1))
        first = max(0, start - 4)
        last = min(FRAMES - 1, stop + 5)
        samples = [pouch.orientation(t) for t in range(first, last + 1)]
        before = samples[0][0]
        after = samples[-1][0]
        if entry >= 0:
            assert pouch.printed[int(before[2] < 0)] == bool(entry), (ident, "belt face")
        if landed >= 0 and stop + 5 < FRAMES:
            assert pouch.printed[int(after[2] < 0)] == bool(landed), (ident, "landed face")
        if stop + 5 >= FRAMES:
            continue
        angle = np.rad2deg(np.arccos(np.clip(before @ after, -1, 1)))
        if flip == 0:
            assert angle < 45, (ident, "unobserved turnover", angle)
            continue
        if flip != 1:
            continue
        angles.append(angle)
        assert angle > 145, (ident, "incomplete turnover", angle)
        # Find the fractional source time when the camera sees the sheet edge-on.
        facing = np.array([v for _, v in samples])
        crossed = np.flatnonzero(facing[:-1] * facing[1:] <= 0)
        assert len(crossed), (ident, "never edge-on")
        if edge_time >= 0:
            times = (first + crossed + facing[crossed] / (facing[crossed] - facing[crossed + 1])) / FPS
            error = float(np.min(abs(times - edge_time)))
            edge_errors.append(error)
            assert error <= 2 / 30 + 1 / FPS / 2, (ident, "edge timing", error)
            print(f"P{ident:03}  normal change {angle:.1f} deg, edge timing error {error:.3f}s")
    assert audited == {n for n, r in enumerate(items) if r.arrive > 0}, "Missing pickup observations"
    assert angles, "No completed turnovers"
    print(f"{len(audited)} pickup routes, {len(angles)} completed turns. Minimum normal change {min(angles):.1f} deg.")
    if edge_errors:
        print(f"Median edge timing error {np.median(edge_errors):.3f}s. Maximum {max(edge_errors):.3f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    audit(parser.parse_args().stage)
