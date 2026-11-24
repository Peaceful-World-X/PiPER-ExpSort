"""Launch an editable Isaac Sim viewport on the animated shot camera."""

DURATION, FPS = 60.0, 50
from pathlib import Path
import sys
from isaacsim import SimulationApp

verify = "--verify" in sys.argv
app = SimulationApp(
    {
        "headless": verify,
        "hide_ui": False,
        "renderer": "RayTracedLighting",
        "width": 1280,
        "height": 960,
        "multi_gpu": False,
    }
)
import carb, omni.usd, omni.timeline
from omni.kit.viewport.utility import get_active_viewport

root = Path(__file__).resolve().parents[1]
ctx = omni.usd.get_context()
if not ctx.open_stage(str(root / "ConveyorCell.usdc")):
    raise RuntimeError("Could not open ConveyorCell.usdc")
for _ in range(10):
    app.update()
s = carb.settings.get_settings()
s.set("/rtx/post/tonemap/op", 4)
s.set("/rtx/post/tonemap/filmIso", 600.0)
s.set("/rtx/post/tonemap/whitepoint", 6500.0)
s.set("/app/viewport/grid/enabled", False)
s.set("/rtx/post/dlss/execMode", 2)
vp = get_active_viewport()
view = "/World/Overview" if "--overview" in sys.argv else "/World/Shot"
vp.camera_path = view
vp.resolution = (960, 720)
timeline = omni.timeline.get_timeline_interface()
s.set("/app/player/useFixedTimeStepping", True)
s.set("/app/runLoops/main/rateLimitEnabled", True)
s.set("/app/runLoops/main/rateLimitFrequency", FPS)
timeline.set_time_codes_per_second(FPS)
timeline.set_ticks_per_frame(1)
timeline.set_start_time(0.0)
timeline.set_end_time(DURATION)
timeline.set_target_framerate(FPS)
timeline.set_current_time(0.0)
timeline.set_looping(True)
timeline.play()
if verify:
    times = []
    for _ in range(20):
        app.update()
        times.append(timeline.get_current_time())
    assert timeline.get_time_codes_per_seconds() == FPS
    steps = [b - a for a, b in zip(times, times[1:])]
    assert max(abs(step - 1 / FPS) for step in steps) < 1.0e-6, steps
    assert str(vp.camera_path) == view
    assert (
        ctx.get_stage()
        .GetPrimAtPath("/World/RightArm/base_link/link1/link2/link3/link4/link5/link6")
        .IsValid()
    )
    print(
        "VERIFIED playback",
        timeline.is_playing(),
        timeline.get_current_time(),
        "seconds, time codes per second",
        timeline.get_time_codes_per_seconds(),
        flush=True,
    )
else:
    while app.is_running():
        app.update()
app.close()
