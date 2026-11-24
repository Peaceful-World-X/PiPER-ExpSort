"""Visible parcel footprints for keeping workbench-only display corrections off moving items."""

import cv2 as cv
import numpy as np
from cell import EYE, K, R, project
from pxr import Usd, UsdGeom, UsdShade


def paper_texture(plain, printed):
    # The two baked albedos differ only where the shipping paper was applied.
    images = []
    for mat in (plain, printed):
        image = UsdShade.Shader(mat.GetPrim().GetChild("Image"))
        path = image.GetInput("file").Get().resolvedPath
        rgb = cv.imread(path)
        if rgb is None:
            raise ValueError(f"Missing parcel albedo {path}")
        images.append(rgb)
    return np.any(images[0] != images[1], axis=-1)


def project_printed_shell(masks, kind, xy, depth, triangles, printed, st, paper):
    """Rasterize front-facing triangles, including opaque folds in front of the label."""
    h, w = masks.shape[1:]
    a, b, c = xy[triangles].transpose(1, 0, 2)
    ab, ac = b - a, c - a

    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    det = cross(ab, ac)
    # Projections are in image coordinates (+Y down). Pixel centres have
    # integer coordinates in the OpenCV camera model.
    lo = np.maximum(np.ceil(np.minimum(np.minimum(a, b), c)).astype(int), [0, 0])
    hi = np.minimum(
        np.floor(np.maximum(np.maximum(a, b), c)).astype(int), [w - 1, h - 1]
    )
    span = np.maximum(hi - lo + 1, 0)
    sizes = span.prod(axis=1) * (det < -1e-8)
    face = np.repeat(np.arange(len(triangles)), sizes)
    if not len(face):
        return
    pixel = np.arange(len(face)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    x = lo[face, 0] + pixel % span[face, 0]
    y = lo[face, 1] + pixel // span[face, 0]
    ap = np.column_stack((x, y)) - a[face]
    u, v = cross(ap, ac[face]) / det[face], cross(ab[face], ap) / det[face]
    bary = np.column_stack((1 - u - v, u, v))
    inside = (bary >= -1e-8).all(axis=1)
    face, x, y = face[inside], x[inside], y[inside]
    if not len(face):
        return
    weights = bary[inside] / depth[triangles[face]]
    inverse_z = weights.sum(axis=1)
    # Retain the nearest physical triangle at each pixel before sampling ST.
    key = y * w + x
    order = np.lexsort((-inverse_z, key))
    nearest = order[np.r_[True, np.diff(key[order]) != 0]]
    masks[int(kind == 2), y[nearest], x[nearest]] = 255
    label = nearest[printed[face[nearest]]]
    uv = (
        np.einsum("ij,ijk->ik", weights[label], st[triangles[face[label]]])
        / inverse_z[label, None]
    )
    # UsdUVTexture uses a lower-left origin; decoded PNG rows start at the top.
    i = np.clip((uv[:, 0] * paper.shape[1]).astype(int), 0, paper.shape[1] - 1)
    j = np.clip(((1 - uv[:, 1]) * paper.shape[0]).astype(int), 0, paper.shape[0] - 1)
    ink = label[paper[j, i]]
    masks[2, y[ink], x[ink]] = 255


class ParcelMask:
    def __init__(self, stage_path, moving_camera=False):
        self.stage = Usd.Stage.Open(str(stage_path))
        self.camera = None
        if moving_camera:
            from comparison_camera import CAMERA, add_camera

            self.camera = UsdGeom.Camera.Get(self.stage, CAMERA)
            if not self.camera:
                with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                    self.camera = add_camera(self.stage)
        self.cache = UsdGeom.XformCache()
        self.items = []
        self.films = []
        self.turnovers = []
        self.turnover_frame = None
        textures = {}
        for node in self.stage.GetPrimAtPath("/World/Parcels").GetChildren():
            surface = node.GetChild("Shell") or node.GetChild("Carton")
            subset = surface.GetChild("PrintedSkin")
            if subset:
                mesh = UsdGeom.Mesh(surface)
                plain, _ = UsdShade.MaterialBindingAPI(surface).ComputeBoundMaterial()
                marked, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
                kind = int(plain.GetPrim().GetName()[4])
                if kind not in textures:
                    textures[kind] = paper_texture(plain, marked)
                quads = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 4)
                tris = quads[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
                printed = np.zeros(len(quads), dtype=bool)
                printed[np.asarray(UsdGeom.Subset(subset).GetIndicesAttr().Get())] = (
                    True
                )
                st = np.asarray(UsdGeom.PrimvarsAPI(surface).GetPrimvar("st").Get())
                self.turnovers.append(
                    (
                        node.GetAttribute("visibility"),
                        mesh,
                        kind,
                        tris,
                        printed.repeat(2),
                        st,
                        textures[kind],
                    )
                )
                continue
            pts = np.asarray(UsdGeom.Mesh(surface).GetPointsAttr().Get())
            lo, hi = pts.min(0), pts.max(0)
            corners = np.array(
                [
                    [x, y, z]
                    for x in (lo[0], hi[0])
                    for y in (lo[1], hi[1])
                    for z in (lo[2] - 0.025, hi[2] + 0.002)
                ]
            )
            self.items.append((node.GetAttribute("visibility"), surface, corners))
            material, _ = UsdShade.MaterialBindingAPI(surface).ComputeBoundMaterial()
            name = material.GetPrim().GetName() if material else ""
            if not name.startswith("film"):
                continue
            # Both skins share UVs. Recover the front grid from the first, tagged skin.
            uv = np.asarray(UsdGeom.PrimvarsAPI(surface).GetPrimvar("st").Get())
            first_row = np.flatnonzero(uv[:, 1] != uv[0, 1])
            if not len(first_row):
                continue
            columns = int(first_row[0])
            rows = len(pts) // (2 * columns)
            if 2 * rows * columns != len(pts):
                continue
            kind = int(name[4])
            perimeter = np.arange(rows * columns).reshape(rows, columns)
            edge = np.r_[
                perimeter[0],
                perimeter[1:, -1],
                perimeter[-1, -2::-1],
                perimeter[-2:0:-1, 0],
            ]
            self.films.append(
                (
                    node.GetAttribute("visibility"),
                    surface,
                    kind,
                    edge,
                    rows,
                    columns,
                    name.endswith("_1"),
                )
            )

    def projection(self, frame):
        self.cache.SetTime(frame)
        if self.camera is None:
            return project
        camera = np.asarray(self.cache.GetLocalToWorldTransform(self.camera.GetPrim()))

        def in_frame(world):
            local = (world - camera[3, :3]) @ camera[:3, :3].T
            hom = (local * [1, -1, -1]) @ K.T
            return hom[..., :2] / hom[..., 2:3]

        return in_frame

    def turnover_masks(self, frame, w, h, in_frame):
        key = (frame, w, h)
        if key != self.turnover_frame:
            masks = np.zeros((3, h, w), np.uint8)
            eye, forward = EYE, R[2]
            if self.camera:
                camera = np.asarray(
                    self.cache.GetLocalToWorldTransform(self.camera.GetPrim())
                )
                eye, forward = camera[3, :3], -camera[2, :3]
            for vis, mesh, kind, tris, printed, st, paper in self.turnovers:
                if vis.Get(frame) != "inherited":
                    continue
                m = np.asarray(self.cache.GetLocalToWorldTransform(mesh.GetPrim()))
                pts = np.asarray(mesh.GetPointsAttr().Get(frame)) @ m[:3, :3] + m[3, :3]
                xy = in_frame(pts) * [w / 640, h / 480]
                if (xy.max(0) < 0).any() or (xy.min(0) >= [w, h]).any():
                    continue
                project_printed_shell(
                    masks, kind, xy, (pts - eye) @ forward, tris, printed, st, paper
                )
            self.turnover_coverage = masks
            self.turnover_frame = key
        return self.turnover_coverage

    def static_surface(self, frame, w, h):
        mask = np.ones((h, w), np.uint8) * 255
        in_frame = self.projection(frame)
        if self.turnovers:
            dyed, white, _ = self.turnover_masks(frame, w, h, in_frame)
            mask[dyed | white != 0] = 0
        for vis, geom, corners in self.items:
            if vis.Get(frame) != "inherited":
                continue
            m = np.asarray(self.cache.GetLocalToWorldTransform(geom))
            uv = in_frame(corners @ m[:3, :3] + m[3, :3])
            if uv[:, 0].max() < 0 or uv[:, 0].min() > 640:
                continue
            poly = cv.convexHull(np.int32(np.rint(uv * [w / 640, h / 480])))
            cv.fillConvexPoly(mask, poly, 0)
        return cv.GaussianBlur(mask, (0, 0), 1.3).astype("float32") / 255

    def film_masks(self, frame, w, h):
        """Project the visible face and its printed area for a local camera response."""
        masks = np.zeros((3, h, w), np.uint8)
        scale = np.array([w / 640, h / 480])
        in_frame = self.projection(frame)
        if self.turnovers:
            masks[:] = self.turnover_masks(frame, w, h, in_frame)
        for vis, geom, kind, edge, rows, cols, printed in self.films:
            if vis.Get(frame) != "inherited":
                continue
            m = np.asarray(self.cache.GetLocalToWorldTransform(geom))
            center = in_frame(m[3, :3])
            if center[0] < -155 or center[0] > 795:
                continue
            pts = np.asarray(UsdGeom.Mesh(geom).GetPointsAttr().Get(frame))
            xy = in_frame(pts[edge] @ m[:3, :3] + m[3, :3]) * scale
            cv.fillPoly(masks[int(kind == 2)], [np.int32(np.rint(xy))], 255)
            if not printed:
                continue
            # Include a margin for the slightly rotated paper. Pixel chroma rejects
            # the surrounding dyed film; the UV patch follows the animated top skin.
            front = pts[: rows * cols].reshape(rows, cols, 3)
            patch = front[
                round(0.30 * (rows - 1)) : round(0.70 * (rows - 1)) + 1,
                round(0.245 * (cols - 1)) : round(0.735 * (cols - 1)) + 1,
            ]
            corners = patch[[0, 0, -1, -1], [0, -1, -1, 0]]
            cp = in_frame(corners @ m[:3, :3] + m[3, :3]) * scale
            if cv.contourArea(np.float32(cp), oriented=True) < 0:
                cv.fillConvexPoly(masks[2], cv.convexHull(np.int32(np.rint(cp))), 255)
        return [
            cv.GaussianBlur(channel, (0, 0), 1.25).astype(np.float32) / 255
            for channel in masks
        ]
