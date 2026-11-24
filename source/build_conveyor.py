"""Build a standalone editable USD cell, with both cameras and the full 50 Hz shot."""

import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

from cell import EYE, FPS, FRAMES, K, R
from comparison_camera import add_camera, extend_backdrop
from geometry import Scene
from mailers import add_packages
from parcel_routes import routes
from grasp import solve
from robots import build_model
from workcell import animate_belt, camera, furnish, lighting

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "assets"


def tape_wrap(g, path):
    a = np.array([-0.144, 0.051, 0])
    b = np.array([-0.164, 0.056, 0])
    normal = (b - a) / np.linalg.norm(b - a)
    y = np.cross(normal, [0.0, 0.0, 1.0])
    y /= np.linalg.norm(y)
    z = np.cross(normal, y)
    pts = []
    normals = []
    uv = []
    faces = []
    for i, t in enumerate(np.linspace(0, 2 * np.pi + 0.085, 66)):
        out = y * np.cos(t) + z * np.sin(t)
        lap = 0.00035 * np.sin(t) + (0.00048 if t >= 2 * np.pi else 0)
        for edge, base in enumerate([a, b]):
            irregular = 0.00064 * np.sin(5 * t + edge) + 0.00028 * np.sin(
                17 * t + 0.8 * edge
            )
            pts.append(base + normal * irregular + out * (0.0299 + lap))
            normals.append(out)
            uv.append((t / (2 * np.pi), float(edge)))
        if i:
            faces.append([2 * i - 2, 2 * i - 1, 2 * i + 1, 2 * i])
    g.mesh(path, pts, faces, "UsedTape", normals, uv)


def arm(g, model, path, base, yaw, qs, gap):
    data = mj.MjData(model)
    turn = Rotation.from_euler("z", yaw)
    g.group(path, base, turn)
    ops = build_model(g, path, model)
    upper = path + "/base_link/link1/link2"
    fore = upper + "/link3"
    tool = fore + "/link4/link5/link6"
    a = np.linspace(0, 2 * np.pi, 65)
    g.wire(
        upper + "/NylonTie",
        np.c_[0.145 + 0.002 * np.sin(a), -0.005 + 0.031 * np.cos(a), 0.024 * np.sin(a)],
        0.0017,
        "WhiteTie",
    )
    g.tube(
        upper + "/Tail",
        (0.145, 0.025, 0.009),
        (0.153, 0.061, 0.015),
        0.0013,
        "WhiteTie",
    )
    tape_wrap(g, fore + "/Tape")
    g.box(
        tool + "/WristCamera", (0, 0.029, 0.044), (0.043, 0.030, 0.049), "Mount", 0.004
    )
    g.tube(tool + "/Lens", (0, 0.029, 0.067), (0, 0.029, 0.074), 0.0103, "Socket")
    g.tube(tool + "/Plug", (0.019, 0.028, 0.039), (0.027, 0.028, 0.039), 0.0048, "Red")
    left = path.endswith("LeftArm")
    mount = np.array([-0.050, -0.006, 0.006] if left else [-0.046, -0.032, 0.041])
    g.box(
        tool + "/SensorBracket",
        (mount[0] * 0.54, mount[1], mount[2]),
        (0.038, 0.017, 0.020),
        "Mount",
        0.001,
    )
    g.box(tool + "/SensorHousing", mount, (0.046, 0.027, 0.030), "usedJoint", 0.0022)
    g.box(
        tool + "/ServiceSticker",
        mount + [-0.006, 0.0137, 0],
        (0.010, 0.0008, 0.012),
        "PowderIvory",
        0.0003,
    )
    g.tube(
        tool + "/SensorAperture",
        mount + [0, 0, 0.0145],
        mount + [0, 0, 0.020],
        0.008,
        "Socket",
    )
    for j, delta in enumerate([-0.015, 0.014]):
        pin = mount + [delta, 0.011, 0.0151]
        g.tube(
            tool + "/SensorScrew" + str(j), pin, pin + [0, 0, 0.0011], 0.00155, "Screw"
        )
    curves = []
    for title, r in [("DataHarness", 0.0030), ("WristLead", 0.0018)]:
        curves.append(g.wire(path + "/" + title, np.zeros((28, 3)), r, "Rubber"))
    splines = [
        make_interp_spline(np.arange(n), np.eye(n), k=2)(np.linspace(0, n - 1, 28))
        for n in (6, 5)
    ]

    def on(name, x):
        b = data.body(name)
        return b.xpos + b.xmat.reshape(3, 3) @ x

    for i, (q, jaw) in enumerate(zip(qs, gap)):
        data.qpos[:] = [*q, jaw, -jaw]
        mj.mj_kinematics(model, data)
        for jid, op in ops.items():
            d = data.qpos[model.jnt_qposadr[jid]]
            value = (
                Gf.Vec3d(*map(float, model.jnt_axis[jid] * d))
                if model.jnt_type[jid] == mj.mjtJoint.mjJNT_SLIDE
                else float(np.rad2deg(d) * model.jnt_axis[jid, 2])
            )
            if i == 0:
                op.Set(value)
            op.Set(value, i)
        elbow = on("link3", [-0.052, 0.061, -0.016])
        plug = on("link6", mount + [-0.021, 0, 0])
        ctrl = [
            np.array(
                [
                    [-0.042, -0.03, 0.012],
                    [-0.073, -0.045, 0.030],
                    on("link1", [-0.020, 0.030, 0.044]),
                    on("link2", [0.13, 0.034, 0.022]),
                    on("link2", [0.23, 0.060, 0.027]),
                    on("link3", [-0.012, 0.033, 0.034]),
                ]
            ),
            np.array(
                [
                    elbow,
                    0.78 * elbow
                    + 0.22 * plug
                    + ([-0.019, -0.017, 0.023] if left else [0, 0.026, 0.055]),
                    0.48 * elbow
                    + 0.52 * plug
                    + ([-0.052, -0.047, -0.034] if left else [0, 0.033, 0.059]),
                    0.20 * elbow
                    + 0.80 * plug
                    + ([-0.028, -0.028, -0.027] if left else [0, 0.010, 0.042]),
                    plug,
                ]
            ),
        ]
        for wire, interp, pts in zip(curves, splines, ctrl):
            coords = Vt.Vec3fArray.FromNumpy(
                np.ascontiguousarray(interp @ pts, dtype="float32")
            )
            wire.GetPointsAttr().Set(coords, i)
            if i == 0:
                wire.GetPointsAttr().Set(coords)


def build(output):
    s = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(s, "Z")
    UsdGeom.SetStageMetersPerUnit(s, 1)
    s.SetTimeCodesPerSecond(FPS)
    s.SetFramesPerSecond(FPS)
    s.SetStartTimeCode(0)
    s.SetEndTimeCode(FRAMES - 1)
    g = Scene(s)
    s.SetDefaultPrim(g.group("/World").GetPrim())
    furnish(g)
    extend_backdrop(s)
    cam = Rotation.from_matrix(R.T @ np.diag([1, -1, -1]))
    camera(g, "/World/Shot", EYE, None, K[0, 0] * 32 / 640, cam)
    add_camera(s)
    camera(g, "/World/Overview", np.array([-1.39, -0.99, 1.07]), [0, 0.16, 0.09], 30)
    s.GetRootLayer().customLayerData = {
        "cameraSettings": {"boundCamera": "/World/Shot"}
    }
    obs = np.load(A / "observations.npz")
    items = routes()
    arms = solve(items)
    # Undo display gamma approximately before modulating physical light power.
    gains = (
        0.72 * obs["light"][:, 1]
        + 0.20 * obs["light"][:, 2]
        + 0.08 * obs["light"][:, 0]
    ) ** 2.15
    lighting(g, gains)
    animate_belt(g, obs["raw"])
    add_packages(g, items, arms)
    gap = arms["jaw"]
    model = mj.MjModel.from_xml_path(str(A / "agilex_piper/piper.xml"))
    for i, side in enumerate(("Left", "Right")):
        arm(
            g,
            model,
            "/World/" + side + "Arm",
            arms["base"][i],
            arms["yaw"][i],
            arms["q"][i],
            gap[i],
        )
    s.GetRootLayer().Save()
    print(output, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "ConveyorCell.usdc")
    build(p.parse_args().output.resolve())


if __name__ == "__main__":
    main()
