"""Render native USD time samples through Isaac Sim RTX and encode an MP4."""

FPS, DURATION = 50, 60.0
import argparse, ctypes, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--width", type=int, default=960)
p.add_argument("--height", type=int, default=720)
p.add_argument("--fps", type=int, default=FPS)
p.add_argument("--start", type=float, default=0.0)
p.add_argument("--end", type=float, default=DURATION)
p.add_argument("--stills", action="store_true")
p.add_argument("--still-times", type=float, nargs="+", default=[0, 12, 34.5, 51, 59.98])
p.add_argument("--still-dir", type=Path, default=root / "renders")
p.add_argument("--stage", default=str(root / "ConveyorCell.usdc"))
p.add_argument("--movie", type=Path)
p.add_argument("--path-tracing", action="store_true")
p.add_argument(
    "--sharp",
    action="store_true",
    help="Disable finite shutter exposure for measurement stills or movies",
)
p.add_argument("--camera", default="/World/Shot")
p.add_argument(
    "--comparison-ending",
    action="store_true",
    help="Render the comparison's closing dolly at its exact output samples",
)
p.add_argument(
    "--ending-frames", type=int, nargs=2, metavar=("FIRST", "STOP"),
    help="Render a half-open range of output frames within the closing camera move",
)
args = p.parse_args()
if args.ending_frames and not args.comparison_ending:
    p.error("--ending-frames requires --comparison-ending.")
if args.fps < 1 or not 0 <= args.start < args.end <= DURATION:
    p.error("Select a positive frame rate and an interval inside the shot.")
first_frame = round(args.start * args.fps)
count = round(args.end * args.fps) - first_frame
if count < 1:
    p.error("The selected interval contains no output frames.")
from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "hide_ui": False,
        "renderer": "PathTracing" if args.path_tracing else "RayTracedLighting",
        "width": args.width,
        "height": args.height,
        "multi_gpu": False,
        "sync_loads": True,
        "extra_args": [
            "--/app/asyncRendering=false",
            "--/app/hydraEngine/waitIdle=true",
            "--/app/updateOrder/checkForHydraRenderComplete=1000",
            "--/app/runLoops/main/rateLimitEnabled=false",
            "--/app/player/useFixedTimeStepping=false",
            "--/rtx-transient/dlssg/enabled=false",
        ],
    }
)
import carb, omni.usd, omni.timeline
import omni.replicator.core as rep
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_buffer
import numpy as np
from PIL import Image

settings = carb.settings.get_settings()
ctx = omni.usd.get_context()
assert ctx.open_stage(args.stage)
stage = ctx.get_stage()
if args.comparison_ending:
    from comparison_camera import (
        CAMERA,
        ENDING_FRAMES,
        RATE,
        add_camera,
        extend_backdrop,
        source_frame,
    )
    from pxr import Usd

    if args.fps != RATE or args.start != 0 or args.end != DURATION:
        p.error("The comparison ending uses the preset 50 fps timeline.")
    if not stage.GetPrimAtPath(CAMERA):
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            add_camera(stage)
            extend_backdrop(stage)
    args.camera = CAMERA
    first_frame, stop = args.ending_frames or (0, ENDING_FRAMES)
    if not 0 <= first_frame < stop <= ENDING_FRAMES:
        p.error("The requested frame range falls outside the comparison ending.")
    count = stop - first_frame
settings.set("/rtx/post/tonemap/op", 4)
settings.set("/rtx/post/tonemap/filmIso", 600.0)
settings.set("/app/viewport/grid/enabled", False)
settings.set("/persistent/app/viewport/displayOptions", 0)
settings.set("/rtx/post/dlss/execMode", 2)
settings.set("/rtx/post/tonemap/whitepoint", 6500.0)
settings.set("/rtx/pathtracing/spp", 16)
settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
settings.set("/rtx/pathtracing/totalSpp", 64 if args.stills else 16)
settings.set("/rtx/post/motionblur/enabled", False)
viewport = get_active_viewport()
assert viewport is not None
viewport.camera_path = args.camera
viewport.resolution = (args.width, args.height)
viewport.updates_enabled = True
timeline = omni.timeline.get_timeline_interface()
timeline.pause()
for _ in range(30):
    app.update()
product = None
rgb = None


def prepare_annotator():
    global product, rgb
    rep.orchestrator.set_capture_on_play(False)
    product = rep.create.render_product(
        args.camera, (args.width, args.height), name="Shot"
    )
    rgb = rep.annotators.get("rgb")
    rgb.attach(product)
    viewport.updates_enabled = False
    for _ in range(12):
        rep.orchestrator.step(rt_subframes=2, delta_time=0.0)
        if np.asarray(rgb.get_data()).size:
            return
    raise RuntimeError("RGB annotator did not receive a frame")


def get(t):
    timeline.set_current_time(t)
    timeline.commit()
    rep.orchestrator.step(rt_subframes=2, delta_time=0.0)
    pixels = np.asarray(rgb.get_data())
    assert pixels.shape == (args.height, args.width, 4), pixels.shape
    return pixels[:, :, :3]


def capture(t, updates=1):
    timeline.set_current_time(t)
    timeline.commit()
    for _ in range(updates):
        app.update()
    result = []

    def receive(buffer, size, w, h, fmt):
        pointer = ctypes.pythonapi.PyCapsule_GetPointer
        pointer.restype = ctypes.c_void_p
        pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        raw = ctypes.string_at(pointer(buffer, None), size)
        result.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)[:, :, :3])

    request = capture_viewport_to_buffer(viewport, receive)
    for _ in range(250):
        app.update()
        if result:
            assert result[0].shape == (args.height, args.width, 3)
            return result[0]
    raise RuntimeError("Viewport capture timed out")


def pipelined(times):
    # Keep readback in flight while Hydra draws the next USD sample. Every request
    # owns its buffer and its sequence number. Preserve capture order at the output.
    ready = {}
    requests = {}
    next_out = 0
    pointer = ctypes.pythonapi.PyCapsule_GetPointer
    pointer.restype = ctypes.c_void_p
    pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    def callback(index):
        def receive(buffer, size, w, h, fmt):
            pixels = ctypes.string_at(pointer(buffer, None), size)
            assert (h, w) == (args.height, args.width)
            ready[index] = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 4)[
                :, :, :3
            ]

        return receive

    length = 0
    for index, t in enumerate(times):
        length = index + 1
        timeline.set_current_time(t)
        timeline.commit()
        requests[index] = capture_viewport_to_buffer(viewport, callback(index))
        app.update()
        for _ in range(25):
            if next_out in ready or length - next_out < 8:
                break
            app.update()
        while next_out in ready:
            yield ready.pop(next_out)
            requests.pop(next_out)
            next_out += 1
        if length - next_out >= 8:
            raise RuntimeError("Readback queue stopped advancing")
    for _ in range(30):
        if next_out == length:
            break
        app.update()
        while next_out in ready:
            yield ready.pop(next_out)
            requests.pop(next_out)
            next_out += 1
    if next_out != length:
        raise RuntimeError("Missing queued render")


def render_frames(count):
    if args.path_tracing and not args.sharp:
        prepare_annotator()
        settings.set("/omni/replicator/captureMotionBlur", True)
        settings.set("/omni/replicator/pathTracedMotionBlurSubSamples", 4)
        exposure = 0.020
        first_time = (
            source_frame(first_frame) if args.comparison_ending else first_frame
        ) / args.fps
        # Seed renderer motion history after seeking. Otherwise the first shutter
        # sample can contain the arms' frame-zero pose as a translucent ghost.
        if first_time > 0:
            for lead in (2, 1):
                timeline.set_current_time(max(0, first_time - (lead + 0.5) * exposure))
                timeline.commit()
                rep.orchestrator.step(delta_time=exposure)
        for i in range(count):
            t = (
                source_frame(first_frame + i) if args.comparison_ending else first_frame + i
            ) / args.fps
            timeline.set_current_time(max(0, t - exposure * 0.5))
            timeline.commit()
            rep.orchestrator.step(delta_time=exposure)
            pixels = np.asarray(rgb.get_data())
            if pixels.shape != (args.height, args.width, 4):
                raise RuntimeError("Incomplete shutter capture")
            yield pixels[:, :, :3]
        return
    if args.comparison_ending:
        prepare_annotator()
        for i in range(count):
            yield get(source_frame(first_frame + i) / args.fps)
        return
    # Compare widely separated poses against settled frames to identify the
    # viewport readback delay on this machine. Check both arms and the camera.
    times = [0.0, 12.0, 34.5, 51.0]
    refs = [capture(t, 8)[::12, ::12].astype(np.int16) for t in times]
    sequence = [0, 3, 2, 1, 0, 2, 3, 1, 3, 0]
    try:
        queued = [
            im[::12, ::12].astype(np.int16)
            for im in pipelined(times[j] for j in sequence)
        ]
    except RuntimeError as error:
        print("Readback probe", error, flush=True)
        queued = []
    for lag in range(4):
        if len(queued) != len(sequence):
            break
        scores = []
        for i in range(4, len(sequence)):
            difference = abs(queued[i] - refs[sequence[i - lag]])
            h, w = difference.shape[:2]
            scores.extend(
                [
                    difference.mean(),
                    difference[: 3 * h // 4, : w // 2].mean(),
                    difference[: 3 * h // 4, w // 2 :].mean(),
                ]
            )
        if max(scores) < 5.0:
            print(
                "Verified pipelined capture delay",
                lag,
                "max residual",
                max(scores),
                flush=True,
            )
            capture(first_frame / args.fps, 6)
            shot = (
                (first_frame + min(i, count - 1)) / args.fps for i in range(count + lag)
            )
            for i, image in enumerate(pipelined(shot)):
                if i >= lag:
                    yield image
            return
    print("Checking synchronous capture", flush=True)
    for settle in (0, 1):
        observed = [
            capture(times[j], updates=settle)[::12, ::12].astype(np.int16)
            for j in sequence
        ]
        for delay in range(4):
            for i in range(4, len(sequence)):
                reference = refs[sequence[i - delay]]
                residual = abs(observed[i] - reference)
                h, w = residual.shape[:2]
                regions = (
                    residual,
                    residual[: 3 * h // 4, : w // 2],
                    residual[: 3 * h // 4, w // 2 :],
                )
                if max(region.mean() for region in regions) > 5:
                    break
            else:
                print(
                    "Verified viewport warm-up and frame delay",
                    settle,
                    delay,
                    flush=True,
                )
                capture(first_frame / args.fps, 8)
                for i in range(count + delay):
                    image = capture(
                        (first_frame + min(i, count - 1)) / args.fps, updates=settle
                    )
                    if i >= delay:
                        yield image
                return
    print("Using synchronised render product", flush=True)
    prepare_annotator()
    for i in range(count):
        yield get((first_frame + i) / args.fps)


if args.stills:
    prepare_annotator()
    for t in args.still_times:
        img = get(t)
        Image.fromarray(img).save(
            args.still_dir / f"{args.camera.rsplit(chr(47), 1)[-1]}_{t:g}.jpg",
            quality=95,
        )
        print("CHECK", t, img.mean(), flush=True)
else:
    ffmpeg = next(
        (root / ".venv/Lib/site-packages/imageio_ffmpeg/binaries").glob("ffmpeg*.exe")
    )
    if args.movie is None:
        name = (
            ".comparison/ending_native.mp4"
            if args.comparison_ending
            else "conveyor-sim-60s.mp4"
        )
        destination = root / "renders" / name
    else:
        destination = args.movie
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.stem + ".partial.mp4")
    enc = subprocess.Popen(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{args.width}x{args.height}",
            "-framerate",
            str(args.fps),
            "-i",
            "-",
            "-map",
            "0:v",
            "-an",
            "-t",
            str(count / args.fps),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        stdin=subprocess.PIPE,
    )
    for i, img in enumerate(render_frames(count)):
        enc.stdin.write(img.tobytes())
        if i % 120 == 0:
            print("FRAME", i, "/", count, flush=True)
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("Encoding failed")
    tmp.replace(destination)
print("RENDER COMPLETE", flush=True)
if product is not None:
    rgb.detach(product)
    product.destroy()
app.close()
