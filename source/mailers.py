"""Metric mailers with crimped edges, a puffed face, and a printed shipping-label variant."""

import numpy as np
from cell import FPS, FRAMES
from geometry import quat
from parcel_routes import SPEED, T
from pxr import Gf, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation
from sealed_pouch import create as create_pouch


def attachment(item, motion, parcel_id, side):
    frames = np.flatnonzero(motion["owner"][side] == parcel_id)
    if not len(frames):
        return None
    f = frames[0]
    _, bag_rot = item.pose(T[f])
    return motion["basis"][side, f].T @ bag_rot


def carried_pose(item, motion, parcel_id, frame, bindings):
    for side in (0, 1):
        if motion["owner"][side, frame] == parcel_id:
            rotation = motion["basis"][side, frame] @ bindings[side]
            return motion["xyz"][side, frame] - rotation @ item.anchor(side), rotation
    return item.pose(T[frame])


def add_packages(g, items, motion):
    for k in range(4):
        for label in (0, 1):
            name = f"film{k}_{label}" if k < 3 else f"pack{k}_{label}"
            maps = f"pearl_white_{label}" if k == 2 else name
            g.material(
                name,
                (0.5, 0.4, 0.4),
                0.34,
                0,
                "./assets/surfaces/" + maps + ".png",
                maps,
            )
            if k == 2:
                # The raw roughness sampler also carries the laminate fraction in G.
                path = g.materials[name].GetPath()
                shader = UsdShade.Shader.Get(g.s, path.AppendChild("Surface"))
                rough = UsdShade.Shader.Get(g.s, path.AppendChild("rough"))
                shader.GetInput("metallic").ConnectToSource(rough.ConnectableAPI(), "g")
    for j, r in enumerate(items):
        path = f"/World/Parcels/P{j:03}"
        node = g.group(path)
        material = f"film{r.kind}_{r.label}" if r.kind < 3 else f"pack3_{r.label}"
        if r.kind == 3:
            height = 0.048
            g.box(
                path + "/Carton",
                (0, 0, height / 2),
                (r.length, r.width, height),
                "pack3_0",
                0.002,
            )
            if r.label:
                l, w = r.length * 0.495, r.width * 0.495
                g.mesh(
                    path + "/TopPrint",
                    [
                        [-l, -w, height + 0.00035],
                        [l, -w, height + 0.00035],
                        [l, w, height + 0.00035],
                        [-l, w, height + 0.00035],
                    ],
                    [[0, 1, 2, 3]],
                    material,
                    uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
                )

        first = max(0, int(np.floor((r.tau - 1.36 / SPEED) * FPS)))
        if r.arrive > 0:
            last = min(FRAMES - 1, int(np.ceil((r.times(0)[1] + 0.95) * FPS)))
        else:
            last = min(FRAMES - 1, int(np.ceil((r.tau + 1.36 / SPEED) * FPS)))
        if first > last:
            raise ValueError("Empty parcel route")
        # Even in the alternate overview each new object is upstream of the belt inlet.
        vis = UsdGeom.Imageable(node).CreateVisibilityAttr()
        vis.Set("invisible")
        vis.Set("invisible", 0)
        vis.Set("inherited", first)
        if last + 1 < FRAMES:
            vis.Set("invisible", last + 1)
        frames = np.arange(first, last + 1)
        tr = node.AddTranslateOp()
        ori = node.AddOrientOp()
        bindings = [attachment(r, motion, j, s) for s in (0, 1)]
        poses = {}
        for f in frames:
            center, R = carried_pose(r, motion, j, f, bindings)
            poses[f] = (center, R)
            tr.Set(Gf.Vec3d(*center), int(f))
            ori.Set(quat(Rotation.from_matrix(R)), int(f))
        if r.kind != 3:
            create_pouch(g, path, material, r, motion, j, bindings, poses)
    print(len(items), "continuous parcel routes", flush=True)
