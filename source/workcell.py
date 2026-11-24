"""Conveyor bed, laminated inspection bench, mounts and louvred office partition."""

import numpy as np
from pxr import Sdf, UsdGeom, UsdLux, UsdShade, Vt
from scipy.spatial.transform import Rotation as Rot

from cell import EYE, unproject

P = "/World/Cell/"


def camera(g, path, eye, target, focal=None, rotation=None):
    cam = UsdGeom.Camera.Define(g.s, path)
    cam.CreateHorizontalApertureAttr(32.0)
    cam.CreateVerticalApertureAttr(24.0)
    cam.CreateFocalLengthAttr(focal or 29.0)
    cam.CreateClippingRangeAttr((0.015, 20.0))
    if rotation is None:
        f = np.asarray(target) - eye
        f = f / np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1])
        r /= np.linalg.norm(r)
        rotation = Rot.from_matrix(np.stack([r, np.cross(r, f), -f], axis=1))
    g.transform(cam, eye, rotation)


def furnish(g):
    mat = g.material
    for n, c, r, m in [
        ("Aluminium", (0.58, 0.56, 0.58), 0.31, 0.68),
        ("PowderIvory", (0.77, 0.72, 0.72), 0.32, 0.17),
        ("Screw", (0.12, 0.13, 0.15), 0.21, 0.88),
        ("Rubber", (0.012, 0.013, 0.014), 0.57, 0.04),
        ("Mount", (0.022, 0.023, 0.025), 0.37, 0.41),
        ("Underlay", (0.012, 0.160, 0.147), 0.44, 0.03),
        ("Blind", (0.42, 0.40, 0.43), 0.32, 0.57),
        ("GlassBack", (0.28, 0.30, 0.32), 0.26, 0.34),
        ("Trim", (0.075, 0.072, 0.080), 0.31, 0.68),
        ("Floor", (0.15, 0.139, 0.135), 0.71, 0.08),
        ("Yellow", (0.82, 0.59, 0.13), 0.48, 0.07),
        ("WhiteTie", (0.75, 0.76, 0.70), 0.44, 0.05),
        ("Red", (0.35, 0.021, 0.014), 0.34, 0.2),
        ("Socket", (0.025, 0.031, 0.043), 0.25, 0.44),
        ("Pine", (0.22, 0.12, 0.054), 0.64, 0.08),
        ("Furniture", (0.035, 0.043, 0.048), 0.65, 0.04),
    ]:
        worn = {"Mount": ("usedPlate", 0.19), "PowderIvory": ("usedIvory", 0.13)}
        if n in worn:
            texture, metal = worn[n]
            mat(n, c, r, metal, "./assets/surfaces/" + texture + ".png", texture)
        else:
            mat(n, c, r, m)
    mat(
        "UsedTape",
        (0.66, 0.44, 0.08),
        0.56,
        0.025,
        "./assets/surfaces/usedTape.png",
        "usedTape",
    )
    mat(
        "GreenMat",
        (0.03, 0.14, 0.07),
        0.235,
        0,
        "./assets/surfaces/mat.png",
        "mat",
        coat=0.22,
        coat_roughness=0.18,
    )
    mat(
        "ConveyorPVC",
        (0.014, 0.095, 0.045),
        0.12,
        0,
        "./assets/surfaces/belt.png",
        "belt",
        coat=0.18,
        coat_roughness=0.14,
    )
    belt_normal = UsdShade.Shader.Get(g.s, "/World/Materials/ConveyorPVC/normal")
    belt_normal.GetInput("scale").Set((5.0, 5.0, 2.0, 1.0))
    belt_normal.GetInput("bias").Set((-2.5, -2.5, -1.0, 0.0))
    mat("Guard", (0.51, 0.72, 0.69), 0.11, 0)
    guard = UsdShade.Shader.Get(g.s, "/World/Materials/Guard/Surface")
    guard.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.135)
    guard.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.49)

    def box(n, c, s, m, r=0.001, rot=None):
        return g.box(P + n, c, s, m, r, rot)

    def tube(n, a, b, r, m):
        return g.tube(P + n, a, b, r, m)

    box("Building/Floor", (0, 0, -0.80), (6, 5, 0.065), "Floor")
    box("Building/BlindBacking", (0, 1.045, 0.42), (4.1, 0.021, 2.6), "GlassBack")
    for j in range(195):
        z = -0.69 + j * 0.0105
        box(
            f"Building/Slat_{j:03}",
            (0, 1.018, z),
            (4.08, 0.013, 0.0011),
            "Blind",
            0.0005,
            Rot.from_euler("x", 50, degrees=True),
        )
    for j, x in enumerate([-1.73, -0.451, 0.135, 1.24, 1.94]):
        box(
            f"Building/Mullion{j}",
            (x, 1.002, 0.35),
            (0.019, 0.031, 2.12),
            "Trim",
            0.002,
        )
        box(
            f"Building/FrameEdge{j}",
            (x + 0.008, 0.983, 0.35),
            (0.002, 0.004, 2.12),
            "Aluminium",
        )
    box("Conveyor/Bed", (0, 0.400, 0.003), (2.45, 0.745, 0.049), "Trim", 0.006)
    box("Conveyor/Belt", (0, 0.393, 0.0285), (2.38, 0.679, 0.013), "ConveyorPVC", 0.006)
    for j, y in enumerate([0.047, 0.755]):
        box(
            f"Conveyor/Rail{j}",
            (0, y, 0.012),
            (2.46, 0.026, 0.046),
            "PowderIvory",
            0.002,
        )
        box(
            f"Conveyor/RailInset{j}",
            (0, y - 0.013, -0.001),
            (2.41, 0.0015, 0.004),
            "Aluminium",
        )
        for k, x in enumerate(np.linspace(-1.15, 1.15, 12)):
            tube(
                f"Conveyor/Rivet{j}_{k}", (x, y, 0.035), (x, y, 0.036), 0.0028, "Screw"
            )
    for j, x in enumerate([-1.195, 1.195]):
        tube(f"Conveyor/Drum{j}", (x, 0.051, 0.005), (x, 0.739, 0.005), 0.031, "Rubber")
        tube(
            f"Conveyor/Bearing{j}", (x, 0.021, 0.005), (x, 0.053, 0.005), 0.016, "Screw"
        )
    for j, x in enumerate([-1.08, 0.01, 1.10]):
        for k, y in enumerate([0.063, 0.729]):
            box(
                f"Conveyor/Extrusion{j}_{k}",
                (x, y, -0.37),
                (0.035, 0.035, 0.75),
                "Aluminium",
            )
            box(
                f"Conveyor/Tslot{j}_{k}",
                (x, y - 0.018, -0.37),
                (0.004, 0.001, 0.72),
                "Trim",
            )
            box(
                f"Conveyor/Foot{j}_{k}",
                (x, y, -0.752),
                (0.054, 0.053, 0.017),
                "Rubber",
                0.004,
            )
    box("Bench/Base", (0, -0.205, -0.019), (0.968, 0.570, 0.029), "Underlay", 0.003)
    box(
        "Bench/InspectionMat",
        (-0.0025, -0.242, -0.0023),
        (0.955, 0.394, 0.0046),
        "GreenMat",
        0.0018,
    )
    for j, x in enumerate([-0.440, 0.435]):
        for k, y in enumerate([-0.439, 0.036]):
            box(f"Bench/Leg{j}_{k}", (x, y, -0.397), (0.039, 0.040, 0.732), "Aluminium")
            box(
                f"Bench/Groove{j}_{k}",
                (x, y - 0.0206, -0.397),
                (0.005, 0.001, 0.706),
                "Trim",
            )
    for j, x in enumerate([-0.325, 0.325]):
        box(f"Bench/Plate{j}", (x, 0, 0.0025), (0.194, 0.102, 0.008), "Mount", 0.004)
        for k, (dx, dy) in enumerate(
            [(-0.082, -0.037), (0.083, -0.037), (-0.082, 0.037), (0.083, 0.037)]
        ):
            tube(
                f"Bench/Bolt{j}_{k}",
                (x + dx, dy, 0.006),
                (x + dx, dy, 0.008),
                0.0038,
                "Screw",
            )
    # Off-camera feeds hide the source and tail of the continuous belt in the overview.
    for idx, xx in enumerate((-1.39, 1.39)):
        box(
            f"Conveyor/FeedDeck{idx}",
            (xx, 0.393, 0.024),
            (0.43, 0.679, 0.020),
            "Rubber",
            0.002,
        )
        box(
            f"Conveyor/FeedCover{idx}",
            (xx, 0.393, 0.164),
            (0.49, 0.75, 0.014),
            "Trim",
            0.005,
        )
        for edge, y in enumerate((0.034, 0.752)):
            box(
                f"Conveyor/FeedSide{idx}_{edge}",
                (xx, y, 0.096),
                (0.49, 0.014, 0.15),
                "Trim",
            )
    # The sorted pieces fall into the return chute after the left gripper opens.
    box("Return/Floor", (-0.824, -0.326, -0.291), (0.68, 0.44, 0.024), "Rubber", 0.008)
    box(
        "Return/Cover",
        (-0.959, -0.326, -0.105),
        (0.40, 0.45, 0.018),
        "Furniture",
        0.009,
    )
    for edge, y in enumerate((-0.55, -0.104)):
        box(
            f"Return/Wall{edge}",
            (-0.824, y, -0.198),
            (0.68, 0.012, 0.20),
            "Furniture",
            0.003,
        )
    box(
        "Return/End", (-1.159, -0.326, -0.197), (0.014, 0.452, 0.21), "Furniture", 0.003
    )
    # Acrylic fences and the outboard overhead camera support from the overview photograph.
    for j, x in enumerate([-0.479, 0.477]):
        box(
            f"Bench/OutsideGuard{j}",
            (x, -0.256, 0.045),
            (0.003, 0.273, 0.09),
            "Guard",
            0.001,
        )
    for i, x in enumerate([-0.66, 0.66]):
        box(f"Support/Vertical{i}", (x, -0.49, 0.47), (0.018, 0.022, 1.05), "Trim")
        box(
            f"Support/Channel{i}",
            (x + 0.005, -0.502, 0.47),
            (0.003, 0.001, 1.04),
            "Screw",
        )
        box(f"Support/Base{i}", (x, -0.49, -0.049), (0.067, 0.068, 0.012), "Mount")
        box(
            f"Support/SideRail{i}",
            (x, -0.25, -0.072),
            (0.038, 0.62, 0.043),
            "PowderIvory",
        )
    box("Support/CrossRail", (0, -0.54, -0.067), (1.36, 0.035, 0.041), "PowderIvory")
    for x in [-0.64, 0.64]:
        tube(
            "Support/Bolt" + str(int(x > 0)),
            (x, -0.56, -0.065),
            (x, -0.562, -0.065),
            0.004,
            "Screw",
        )
    box("Office/Desk", (1.49, 0.62, 0.03), (0.61, 0.54, 0.03), "PowderIvory", 0.005)
    for i, x in enumerate([1.23, 1.73]):
        for j, y in enumerate([0.42, 0.82]):
            box(f"Office/Leg{i}{j}", (x, y, -0.37), (0.035, 0.035, 0.75), "Aluminium")
    tube(
        "Office/SteelBottle",
        (1.52, 0.62, 0.046),
        (1.52, 0.62, 0.234),
        0.036,
        "Aluminium",
    )
    tube("Office/Tumbler", (1.64, 0.61, 0.046), (1.64, 0.61, 0.202), 0.035, "Furniture")
    box("Office/ChairBack", (1.36, 0.94, 0.04), (0.35, 0.035, 0.38), "Furniture", 0.016)
    box(
        "Office/ChairSeat", (1.36, 0.79, -0.15), (0.37, 0.31, 0.044), "Furniture", 0.013
    )
    tube("Office/ChairPost", (1.36, 0.79, -0.73), (1.36, 0.79, -0.17), 0.026, "Screw")
    for i in range(5):
        a = 2 * np.pi * i / 5
        tube(
            f"Office/ChairFoot{i}",
            (1.36, 0.79, -0.70),
            (1.36 + 0.20 * np.cos(a), 0.79 + 0.20 * np.sin(a), -0.738),
            0.015,
            "Trim",
        )


def lighting(g, gains):
    def animate(attr, power):
        attr.Set(float(power))
        for i, r in enumerate(gains):
            attr.Set(float(power * r), i * 5)

    dome = UsdLux.DomeLight.Define(g.s, "/World/Lights/RoomBounce")
    dome.CreateColorAttr((0.92, 0.955, 1))
    animate(dome.CreateIntensityAttr(), 82)

    def panel(name, pos, target, w, h, power, col):
        light = UsdLux.RectLight.Define(g.s, "/World/Lights/" + name)
        light.CreateDiffuseAttr().Set(1.0 if name == "WorkshopFill" else 0.58)
        if name in ("SideBounce", "WindowReturn"):
            light.CreateDiffuseAttr().Set(0.86 if name == "SideBounce" else 0.24)
            light.CreateSpecularAttr().Set(0.12 if name == "SideBounce" else 0.74)
        light.CreateWidthAttr(w)
        light.CreateHeightAttr(h)
        light.CreateColorAttr(col)
        animate(light.CreateIntensityAttr(), power)
        f = np.asarray(target) - pos
        f = f / np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        if np.linalg.norm(r) < 1e-6:
            r = np.array([1.0, 0.0, 0.0])
        r /= np.linalg.norm(r)
        g.transform(
            light, pos, Rot.from_matrix(np.stack((r, np.cross(r, f), -f), axis=1))
        )

    # Position the highlighted fixtures by tracing a reflected camera ray through the two observed hot spots.
    def mirror(x, y, z, ceiling):
        hit = unproject(np.array([x, y]), z)
        ray = hit - EYE
        ray[2] *= -1
        return hit + ray * (ceiling - z) / ray[2], hit

    pos, hit = mirror(385, 422, 0, 1.18)
    panel("InspectionSoftbox", pos, hit, 0.292, 0.268, 16500, (1, 0.952, 0.930))
    pos, hit = mirror(296, 104, 0.035, 0.26)
    panel("PartitionWindow", pos, hit, 0.170, 0.136, 12100, (1, 0.966, 0.912))
    panel("PartitionCore", pos, hit, 0.139, 0.103, 13650, (1, 0.974, 0.934))
    panel(
        "WindowReturn",
        *mirror(346, 110, 0.035, 0.35),
        0.200,
        0.14,
        1950,
        (0.89, 0.94, 1),
    )
    panel(
        "SideBounce",
        np.array([0.82, -0.53, 0.54]),
        [-0.15, 0.10, 0.12],
        0.70,
        0.61,
        405,
        (0.91, 1, 0.965),
    )
    panel(
        "WorkshopFill",
        np.array([-0.66, -0.98, 1.48]),
        [0, 0.18, 0.04],
        1.40,
        1.03,
        850,
        (0.94, 0.95, 1),
    )


def animate_belt(g, raw):
    from parcel_routes import SPEED

    distance = np.arange(600) * SPEED / 10
    surface = UsdGeom.PrimvarsAPI(g.s.GetPrimAtPath(P + "Conveyor/Belt")).GetPrimvar(
        "st"
    )
    uv0 = np.array(surface.Get(), dtype="float32")
    for i, d in enumerate(distance):
        mapping = uv0.copy()
        mapping[:, 0] -= d / 2.38
        surface.Set(Vt.Vec2fArray.FromNumpy(mapping), i * 5)
    surface.Set(
        Vt.Vec2fArray.FromNumpy(
            uv0 - np.float32([float(distance[-1] + 0.08 * SPEED) / 2.38, 0])
        ),
        2999,
    )
