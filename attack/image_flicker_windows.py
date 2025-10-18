#!/usr/bin/env python3
"""
Windows-native fullscreen image flicker using GLFW + OpenGL.
"""

import glfw
from OpenGL.GL import *
from PIL import Image
import time
import math
import argparse
import pathlib

# ------------------------
# Flicker settings
# ------------------------
FIXED_FREQUENCY_HZ = 120.0  # change as needed

# ------------------------
# Window init
# ------------------------
def init_window():
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")

    #monitor = glfw.get_primary_monitor()
    monitor = glfw.get_monitors()[1]
    mode = glfw.get_video_mode(monitor)

    # Resolve width/height differences across bindings
    try:
        W, H = mode.size.width, mode.size.height
    except AttributeError:
        W, H = getattr(mode, "width", 800), getattr(mode, "height", 600)

    print(f"Monitor resolution: {W}x{H}, refresh rate: {getattr(mode, 'refresh_rate', 60)} Hz")

    # Fullscreen window
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)

    window = glfw.create_window(W, H, "Image Flicker", monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)  # disable vsync

    # Projection to pixel coordinates
    glViewport(0, 0, W, H)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, W, 0, H, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    return window, W, H

# ------------------------
# Texture helpers
# ------------------------
def load_texture(image_path: pathlib.Path):
    img = Image.open(image_path).convert("RGBA")
    img_data = img.tobytes("raw", "RGBA", 0, -1)
    w, h = img.size

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return tex_id, w, h

def draw_fullscreen_image(tex_id, W, H):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glColor4f(1.0, 1.0, 1.0, 1.0)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(W, 0)
    glTexCoord2f(1, 1); glVertex2f(W, H)
    glTexCoord2f(0, 1); glVertex2f(0, H)
    glEnd()

    glDisable(GL_TEXTURE_2D)

# ------------------------
# Main loop
# ------------------------
def run_flicker(self, image_path: pathlib.Path, flicker_type="fixed"):
    tex_id, img_w, img_h = load_texture(image_path)
    start_time = time.perf_counter()

    while not glfw.window_should_close(self.window):
        glClear(GL_COLOR_BUFFER_BIT)
        t = time.perf_counter() - start_time

        if flicker_type == "fixed":
            phase = int(t * FIXED_FREQUENCY_HZ) % 2
            if phase == 0:
                draw_fullscreen_image(tex_id, self.W, self.H)
        elif flicker_type == "lfm":
            # LFM-style square wave
            f_start, f_end = 1.0, 5.0
            chirp_duration = 10.0
            t_chirp = t % chirp_duration
            curr_freq = f_start + (f_end - f_start) * (t_chirp / chirp_duration)
            phase = math.sin(2 * math.pi * ((f_start * t_chirp) + 0.5 * ((f_end - f_start) / chirp_duration) * t_chirp**2))
            if phase > 0:
                draw_fullscreen_image(tex_id, self.W, self.H)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    glfw.terminate()

# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fullscreen image flicker (Windows-native GLFW)")
    parser.add_argument("-i", "--image_filename", type=str, required=True)
    parser.add_argument("-f", "--flicker_type", choices=["fixed", "lfm"], default="fixed")
    args = parser.parse_args()
    image_path = pathlib.Path(args.image_filename)
    run_flicker(image_path, flicker_type=args.flicker_type)
