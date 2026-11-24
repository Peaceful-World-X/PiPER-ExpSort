"""Camera move and source-frame mapping for the closing shot of the comparison."""

import math

import numpy as np
from pxr import Gf, UsdGeom

CAMERA = "/World/ComparisonCamera"
START = 39.0
SETTLE = 58.5
RATE = 50
FIRST_OUTPUT = int(START * RATE / 1.5)
ENDING_FRAMES = 2000 - FIRST_OUTPUT


def source_frame(index):
    """Match the existing comparison's 1.5x frame selection exactly."""
    return (3 * (FIRST_OUTPUT + index) + 1) // 2


def progress(seconds):
    u = min(1.0, max(0.0, (seconds - START) / (SETTLE - START)))
    return u * u * (3.0 - 2.0 * u)


def add_camera(stage):
    """Inherit the calibrated lens and move the camera in metres in the USD scene."""
    stage.GetEditTarget().GetLayer().timeCodesPerSecond = stage.GetTimeCodesPerSecond()
    base = UsdGeom.Camera.Get(stage, "/World/Shot")
    camera = UsdGeom.Camera.Define(stage, CAMERA)
    camera.GetPrim().GetReferences().AddInternalReference(base.GetPath())
    translate, orient = camera.GetOrderedXformOps()
    start = np.array(translate.Get())
    attitude = Gf.Quatd(orient.Get())
    end = np.array([0.23, -3.12, 1.97])
    target = np.array([0.0, 0.14, -0.09])
    look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*end), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    final = look.GetInverse().ExtractRotationQuat().GetNormalized()
    # Even perceived scale change across the move, with stationary endpoints.
    ratio = np.linalg.norm(end - target) / np.linalg.norm(start - target)
    for tick in range(round(START * RATE), round(SETTLE * RATE) + 1, 5):
        u = progress(tick / RATE)
        distance = math.expm1(math.log(ratio) * u) / (ratio - 1.0)
        translate.Set(Gf.Vec3d(*(start + distance * (end - start))), tick)
        orient.Set(Gf.Quatf(Gf.Slerp(distance, attitude, final)), tick)
    return camera


def extend_backdrop(stage):
    # Continue the existing partition across the wider frustum.
    building = stage.GetPrimAtPath("/World/Cell/Building")
    for prim in building.GetChildren():
        if prim.GetName().startswith(("Slat_", "BlindBacking")):
            UsdGeom.Xformable(prim).AddScaleOp(opSuffix="wide").Set((1.6, 1.0, 1.0))
