"""Cut the reference without re-encoding, grade native renders, and combine both views."""

import argparse
import subprocess
from pathlib import Path

import cv2 as cv
from photometry import CameraResponse
from parcel_mask import ParcelMask
from comparison_camera import ENDING_FRAMES, RATE, START, progress, source_frame
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "renders"
REAL = OUT / "conveyor-reference-60s.mp4"
SIM = OUT / "conveyor-sim-60s.mp4"
PAIR = OUT / "PiPER-ExpSort.mp4"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CACHE = OUT / ".comparison"
ENDING = CACHE / "ending.mp4"
CALIBRATION = CACHE / "calibration.png"


def run(*args):
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True,
    )


def cut(source):
    temp = REAL.with_suffix(".partial.mp4")
    run(
        "-i",
        source,
        "-map",
        "0:v:0",
        "-frames:v",
        "1800",
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        temp,
    )
    temp.replace(REAL)


def exposure(source, stage=None, destination=SIM, *, ending=False):
    cap = cv.VideoCapture(str(source))
    count = ENDING_FRAMES if ending else 3000
    if not cap.isOpened() or cap.get(cv.CAP_PROP_FRAME_COUNT) != count:
        raise ValueError(f"Expected {count} rendered frames from {source}")
    width, height = (
        int(cap.get(v)) for v in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT)
    )
    masks = ParcelMask(stage or ROOT / "ConveyorCell.usdc", moving_camera=ending)
    CACHE.mkdir(exist_ok=True)
    cv.setNumThreads(3)
    tmp = destination.with_suffix(".exposure.mp4")
    args = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        "50",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    reference = cv.VideoCapture(str(REAL))
    ok, real0 = reference.read()
    reference.release()
    if not ok:
        raise ValueError("Missing reference first frame for camera calibration")
    response = None
    if ending:
        calibration = cv.imread(str(CALIBRATION))
        if calibration is None or calibration.shape[:2] != (height, width):
            raise ValueError(
                "The closing shot requires the main camera calibration frame"
            )
        response = CameraResponse(real0, calibration)
    with subprocess.Popen(args, stdin=subprocess.PIPE) as writer:
        try:
            for i in range(count):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Unexpected EOF at frame {i}")
                if response is None:
                    response = CameraResponse(real0, frame)
                    calibration_tmp = CALIBRATION.with_suffix(".partial.png")
                    if not cv.imwrite(str(calibration_tmp), frame):
                        raise OSError(CALIBRATION)
                    calibration_tmp.replace(CALIBRATION)
                tick = source_frame(i) if ending else i
                # The measured illumination field belongs to the original camera.
                # As the camera leaves that view, use the newly traced reflections.
                weight = (
                    np.exp(-((progress(tick / RATE) / 0.055) ** 2)) if ending else 1.0
                )
                coverage = (
                    masks.static_surface(tick, width, height) if weight > 0.001 else 1.0
                )
                out = response.apply(
                    frame, coverage, masks.film_masks(tick, width, height), weight
                )
                writer.stdin.write(out.tobytes())
                if (i + 1) % 500 == 0:
                    print("Graded native frames", i + 1, flush=True)
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("Exposure encode failed")
        finally:
            cap.release()
    tmp.replace(destination)


def pair():
    v = cv.VideoCapture(str(SIM))
    width, height = (
        int(v.get(k)) for k in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT)
    )
    if width == 0 or width * 3 != height * 4:
        raise ValueError("Expected 4:3 simulation")
    v.release()
    tmp = PAIR.with_suffix(".partial.mp4")
    graph = (
        f"[0:v]fps=50:start_time=0,scale={width}:{height}:flags=lanczos,setsar=1,"
        "select='not(eq(mod(n,3),1))',settb=1/50,setpts=N[l];"
        f"[1:v]trim=end_frame={round(START * RATE)},setsar=1,"
        "select='not(eq(mod(n,3),1))',settb=1/50,setpts=N[r0];"
        "[2:v]setsar=1,settb=1/50,setpts=N[r1];[r0][r1]concat=n=2:v=1:a=0[r];"
        "[l][r]hstack=shortest=1[v]"
    )
    run(
        "-i",
        REAL,
        "-i",
        SIM,
        "-i",
        ENDING,
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-an",
        "-r",
        "50",
        "-frames:v",
        "2000",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        tmp,
    )
    tmp.replace(PAIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cut", type=Path)
    p.add_argument("--exposure", type=Path, help="Grade the native 50 fps render")
    p.add_argument(
        "--exposure-final",
        action="store_true",
        help="Apply camera colour grading once after a fresh render",
    )
    p.add_argument("--compare", action="store_true")
    p.add_argument(
        "--grade-ending",
        type=Path,
        help="Grade the closing camera render for the comparison",
    )
    args = p.parse_args()
    if args.cut:
        cut(args.cut)
    if args.exposure:
        exposure(args.exposure)
    if args.exposure_final:
        exposure(SIM)
    if args.grade_ending:
        exposure(args.grade_ending, destination=ENDING, ending=True)
    if args.compare:
        pair()
