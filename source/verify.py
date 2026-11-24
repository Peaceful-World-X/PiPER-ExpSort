"""Decode the deliveries, check cadence and compare both panes to their sources."""

import argparse
import subprocess
from pathlib import Path
import cv2 as cv
import numpy as np
from comparison_camera import FIRST_OUTPUT
from finish import REAL, SIM, PAIR, ENDING, FFMPEG


def source_payloads(path, count=1800):
    result = subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-frames:v",
            str(count),
            "-f",
            "framehash",
            "-",
        ],
        capture_output=True,
        check=True,
        text=True,
        encoding="utf8",
    )
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in result.stdout.splitlines()
        if line.startswith("0,")
    ]


def verified_reference(requested):
    local = REAL.parent / "真机视频.mp4"
    source = requested if requested.is_file() else local
    if not source.is_file():
        raise FileNotFoundError(requested)
    payloads = source_payloads(source)
    assert len(payloads) == 1800 and payloads == source_payloads(REAL), (
        "Reference prefix changed"
    )
    print(f"Reference {source}: 1800/1800 compressed video packets match", flush=True)
    return source


def check_clock(path, rate, count):
    # A one-pixel grayscale frame preserves the presentation timestamps during full decode.
    result = subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-an",
            "-vf",
            "scale=1:1,format=gray",
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            f"1:{rate}",
            "-f",
            "framehash",
            "-",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    ticks = [
        list(map(int, line.split(",")[:5]))
        for line in lines
        if line and line[0].isdigit()
    ]
    assert any(f"1/{rate}" in s for s in lines if s.startswith("#tb")), path
    assert len(ticks) == count and all(
        row[2] == i and row[3] == 1 for i, row in enumerate(ticks)
    ), path
    # Confirm there are no audio streams by asking FFmpeg to map all optional audio streams alone.
    meta = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stderr
    assert "Audio:" not in meta, path
    print(
        f"{path.name}   {count} frames / {rate} fps = {count / rate:.2f}s, decoded, silent",
        flush=True,
    )


def at(cap, index):
    cap.set(cv.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    assert ok, index
    return frame


def check_panes(original):
    caps = [cv.VideoCapture(str(p)) for p in (original, REAL, SIM, PAIR, ENDING)]
    for f in [
        0,
        61,
        333,
        599,
        936,
        1150,
        FIRST_OUTPUT - 1,
        FIRST_OUTPUT,
        1512,
        1789,
        1999,
    ]:
        pair = at(caps[3], f)
        h, w = pair.shape[:2]
        source_frame = (3 * f + 1) // 2
        sim = (
            at(caps[2], source_frame)
            if f < FIRST_OUTPUT
            else at(caps[4], f - FIRST_OUTPUT)
        )
        assert sim.shape[:2] == (h, w // 2)
        r = np.abs(sim.astype(float) - pair[:, w // 2 :].astype(float)).mean()
        # FFmpeg fps places input frames on the nearest presentation boundary.
        candidates = []
        for j in set(
            np.clip(
                np.arange(int(source_frame * 0.6) - 1, int(source_frame * 0.6) + 3),
                0,
                1799,
            )
        ):
            ref = at(caps[1], j)
            assert np.array_equal(at(caps[0], j), ref), j
            enlarged = cv.resize(ref, (w // 2, h), interpolation=cv.INTER_LANCZOS4)
            candidates.append(
                (np.abs(enlarged.astype(float) - pair[:, : w // 2]).mean(), j)
            )
        l, j = min(candidates)
        print(
            f"{f / 50:.2f}s   left error {l:.2f}/255 at source frame {j}, right error {r:.2f}/255",
            flush=True,
        )
        assert (
            l < 4.5 and r < 3.0 and abs(j / 30 - source_frame / 50) <= 1 / 30 + 0.00001
        ), (
            f,
            l,
            j,
            r,
        )
    for cap in caps:
        cap.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    ref = verified_reference(p.parse_args().reference)
    for path, rate, count in [(REAL, 30, 1800), (SIM, 50, 3000), (PAIR, 50, 2000)]:
        check_clock(path, rate, count)
    check_panes(ref)
