#!/usr/bin/env python3
"""
corner_flicker_windows.py
Full-screen flicker test for a specific monitor on Windows (native Python).
Avoids GLFW vidmode attribute mismatches and uses high-resolution timing.
"""

import sys
import time
import argparse
import glfw
from OpenGL.GL import *

# ------------------------
# Flicker settings
# ------------------------
frequencies = {
    "topleft": 60.0,
    "topright": 60.0,
    "bottomleft": 60.0,
    "bottomright": 60.0,
}
patch_size = 100  # pixels

# ------------------------
# Window / monitor helpers
# ------------------------
def pick_monitor(index: int):
    monitors = glfw.get_monitors()
    if not monitors:
        raise RuntimeError("No monitors found")
    if index < 0 or index >= len(monitors):
        raise ValueError(f"Monitor index {index} out of range (0..{len(monitors)-1})")
    return monitors[index], monitors

def get_vidmode_info(monitor):
    mode = glfw.get_video_mode(monitor)
    # width/height available as mode.size.width/mode.size.height in many bindings
    try:
        w = mode.size.width
        h = mode.size.height
    except Exception:
        # fallback to direct attributes
        w = getattr(mode, "width", None)
        h = getattr(mode, "height", None)
    # refresh rate field name varies: try known names
    refresh = getattr(mode, "refresh_rate", None)
    if refresh is None:
        refresh = getattr(mode, "refreshRate", None)
    if refresh is None:
        # last resort: assume 60
        refresh = 60
    return int(w), int(h), int(refresh)

def init_window(monitor_index: int = 0):
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")

    monitor, all_monitors = pick_monitor(monitor_index)
    w, h, refresh = get_vidmode_info(monitor)

    print(f"Selected monitor index: {monitor_index} (found {len(all_monitors)} monitors)")
    # Print brief list for debugging
    for i, m in enumerate(all_monitors):
        mw, mh, mr = get_vidmode_info(m)
        print(f"  {i}: {mw}x{mh} @ {mr} Hz")

    print(f"Creating fullscreen window on monitor {monitor_index}: {w}x{h} @ {refresh}Hz")

    # Window hints: avoid setting color bit hints to prevent attribute errors
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    # We purposely don't set RED_BITS/GREEN_BITS/BLUE_BITS here to avoid attribute mismatch
    # Create a fullscreen window on the chosen monitor
    window = glfw.create_window(w, h, "Flicker Patches", monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")
    glfw.make_context_current(window)

    # Disable vsync
    glfw.swap_interval(0)

    # Setup projection to match pixel coordinates
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, w, 0, h, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # black background, opaque
    glClearColor(0.0, 0.0, 0.0, 1.0)

    return window, w, h, refresh

def draw_patch(x, y, w, h, on):
    if on:
        glColor4f(1.0, 1.0, 1.0, 1.0)
    else:
        # draw black quad (not transparent) so that fullscreen overlay shows proper toggles
        glColor4f(0.0, 0.0, 0.0, 1.0)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()

def run_flicker(monitor_index: int = 0, target_fps: float = None):
    window, W, H, refresh = init_window(monitor_index)

    # If target_fps not set, use the smallest distinct frequency among flickers (or 60)
    if target_fps is None:
        target_fps = 60.0

    # Frame pacing variables
    frame_time = 1.0 / target_fps
    start_t = time.perf_counter()
    last_t = start_t
    frame_counter = 0
    fps_print_interval = 1.0
    last_fps_print = start_t
    rendered_frames = 0

    try:
        while not glfw.window_should_close(window):
            now = time.perf_counter()
            # Only render when it's time for the next logical frame (frame pacing)
            if (now - last_t) >= frame_time:
                # Advance a consistent timestep (avoid drift accumulation)
                last_t += frame_time
                t = last_t - start_t

                # Render
                glClear(GL_COLOR_BUFFER_BIT)
                states = {}
                for key, f in frequencies.items():
                    # square wave at frequency f
                    phase = int(t * f) % 2
                    states[key] = (phase == 0)

                # Draw patches (corners)
                draw_patch(0, H - patch_size, patch_size, patch_size, states["topleft"])
                draw_patch(W - patch_size, H - patch_size, patch_size, patch_size, states["topright"])
                draw_patch(0, 0, patch_size, patch_size, states["bottomleft"])
                draw_patch(W - patch_size, 0, patch_size, patch_size, states["bottomright"])

                glfw.swap_buffers(window)
                glfw.poll_events()

                rendered_frames += 1
                frame_counter += 1

            else:
                # Sleep a tiny bit to avoid 100% CPU busy-wait; keep it tiny for accuracy
                time.sleep(0.0005)

            # Periodically print effective FPS measured locally
            if (now - last_fps_print) >= fps_print_interval:
                elapsed = now - last_fps_print
                fps = frame_counter / elapsed if elapsed > 0 else 0.0
                print(f"Effective logical FPS (target {target_fps:.1f}): {fps:.2f} | Monitor reports: {refresh} Hz")
                last_fps_print = now
                frame_counter = 0

    finally:
        glfw.terminate()

def parse_args():
    p = argparse.ArgumentParser(description="Windows-native flicker test (full-screen on chosen monitor)")
    p.add_argument("--monitor", type=int, default=0, help="Monitor index (0..N-1)")
    p.add_argument("--fps", type=float, default=60.0, help="Logical target FPS for flicker (e.g., 60)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_flicker(monitor_index=args.monitor, target_fps=args.fps)

