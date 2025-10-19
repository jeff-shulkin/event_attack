import numpy as np
import scipy as sp
import imutils
import glfw
import cv2
from PIL import Image
from OpenGL.GL import *
import pathlib
import time


class EventAttack:
    def __init__(self, inject_img_path : pathlib.Path, carrier_img_path: pathlib.Path, attack_method : str, fps : int):
        # Initialize OpenGL window
        self.window, self.W, self.H = self._init_window()

        # Read in both the carrier and attack images
        self.inject_img = cv2.imread(inject_img_path, cv2.IMREAD_UNCHANGED)
        self.carrier_img = cv2.imread(carrier_img_path, cv2.IMREAD_UNCHANGED)

        # Resize images to min size
        h_c, w_c = self.carrier_img.shape[:2]
        h_i, w_i = self.inject_img.shape[:2]

        new_h = min(h_c, h_i)
        new_w = min(w_c, w_i)
        self.inject_img = cv2.resize(self.inject_img, (self.monitor_W, self.monitor_H), interpolation=cv2.INTER_LINEAR)
        self.carrier_img = cv2.resize(self.carrier_img, (self.monitor_W, self.monitor_H), interpolation=cv2.INTER_LINEAR)
        self.attack_method = attack_method
        self.fps = fps

        #cv2.imshow("Frame", self.inject_img)
        #cv2.waitKey(1)

    # Create injection mask
    def _create_injection_mask(self, injection_img):
        """
        Create a mask for embedding the injected image into the carrier.
        Combines alpha channel (if present) and letters/contours in the image.
        """
        # Start with alpha channel if it exists
        if injection_img.shape[2] == 4:
            print("Alpha channel detected!")
            alpha_mask = injection_img[:, :, 3] > 0
            rgb = injection_img[:, :, :3]
        else:
            print("No alpha channel detected.")
            alpha_mask = np.ones(injection_img.shape[:2], dtype=bool)
            rgb = injection_img

        # Detect letters by thresholding
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # Invert threshold so letters become True
        _, letter_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        letter_mask = letter_mask > 0

        # Combine alpha and letter masks
        injection_mask = np.logical_and(alpha_mask, letter_mask)

        # Optional: visualize mask
        # self._show_mask_cv(injection_mask, winname="Injection Mask")

        return injection_mask

    def _shift_mask(self, mask, dx, dy):
        """Shift a boolean mask by (dx, dy) with zero padding."""
        h, w = mask.shape
        shifted = np.zeros_like(mask)
        
        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx)
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)

        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return shifted

        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
        return shifted
    
    def _show_mask_cv(self, mask, winname="mask"):
        # mask: boolean HxW
        vis = (mask.astype(np.uint8) * 255)  # 0 or 255
        cv2.imshow(winname, vis)
        cv2.waitKey(0)
        cv2.destroyWindow(winname)

    # Inject attack img into carrier image
    def _inject_img(self, carrier_img, color_deltaE=3.0):
        # Convert images to LAB space
        #inject_lab = cv2.cvtColor(self.inject_img, cv2.COLOR_BGR2LAB)
        carrier_lab = cv2.cvtColor(carrier_img, cv2.COLOR_BGR2LAB)

        # Create injection mask to embed into carrier image
        injection_mask = self._create_injection_mask(self.inject_img)
        
        # Calculate the lightness change for the color difference color_deltaE
        carrier_L = carrier_lab[:, :, 0].astype(np.float32)
        k_L = 1.0
        S_L = 1 + ((0.015 * (carrier_L - 50) ** 2) / (np.sqrt(20 + (carrier_L - 50) ** 2)))
        delta_L = k_L * S_L * color_deltaE
        #self._show_mask_cv(injection_mask)
    
        # Create the negative and positive images
        pos_carrier_lab = carrier_lab.copy().astype(np.float32)
        neg_carrier_lab = carrier_lab.copy().astype(np.float32)
        
        pos_carrier_lab[:, :, 0][self._shift_mask(injection_mask, dx=1, dy=0)] += delta_L[self._shift_mask(injection_mask, dx=1, dy=0)]
        neg_carrier_lab[:, :, 0][injection_mask] -= delta_L[injection_mask]

        pos_carrier_lab = np.clip(pos_carrier_lab, 0, 255).astype(np.uint8)
        neg_carrier_lab = np.clip(neg_carrier_lab, 0, 255).astype(np.uint8)

        # Convert both images back to RGB space
        pos_carrier = cv2.cvtColor(pos_carrier_lab, cv2.COLOR_LAB2BGR)
        neg_carrier = cv2.cvtColor(neg_carrier_lab, cv2.COLOR_LAB2BGR)

        return pos_carrier, neg_carrier
    
    # Setup functions
    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to init GLFW")

        monitor = glfw.get_primary_monitor()
        #monitor = glfw.get_monitors()[1]
        mode = glfw.get_video_mode(monitor)

        # Resolve width/height differences across bindings
        try:
            self.monitor_W, self.monitor_H = mode.size.width, mode.size.height
        except AttributeError:
            self.monitor_W, self.monitor_H = getattr(mode, "width", 800), getattr(mode, "height", 600)

        print(f"Monitor resolution: {self.monitor_W}x{self.monitor_H}, refresh rate: {getattr(mode, 'refresh_rate', 60)} Hz")

        # Fullscreen window
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)

        window = glfw.create_window(self.monitor_W, self.monitor_H, "Image Flicker", monitor, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)
        glfw.swap_interval(0)  # disable vsync

        # Projection to pixel coordinates
        glViewport(0, 0, self.monitor_W, self.monitor_H)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.monitor_W, 0, self.monitor_H, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        return window, self.monitor_W, self.monitor_H

    def _load_texture_from_array(self, img_array):
        # Convert BGR to RGBA
        img_rgba = cv2.flip(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGBA), 0)
        h, w, _ = img_rgba.shape
        img_data = img_rgba.tobytes()

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        return tex_id, w, h
    
    def _create_masked_texture(self, frame, mask):
        # Find bounding box of mask
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        # Crop frame and mask
        cropped_frame = frame[y0:y1, x0:x1].copy()
        cropped_mask = mask[y0:y1, x0:x1]

        # Apply mask: zero out unmasked pixels (transparent)
        rgba = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = (cropped_mask.astype(np.uint8) * 255)

        tex_id, w, h = self._load_texture_from_array(rgba)
        return tex_id, x0, y0, w, h

    def _draw_fullscreen_image(self, tex_id, W, H):
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

    def _draw_masked_region(self, tex_id, x0, y0, w, h):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x0, y0)
        glTexCoord2f(1, 0); glVertex2f(x0 + w, y0)
        glTexCoord2f(1, 1); glVertex2f(x0 + w, y0 + h)
        glTexCoord2f(0, 1); glVertex2f(x0, y0 + h)
        glEnd()

        glDisable(GL_TEXTURE_2D)

    def flicker(self, duration=60, quit_key="ESC"):
        # Generate the positive and negative frames to flicker between
        pos_frame, neg_frame = self._inject_img(self.carrier_img, color_deltaE=3.0)

        # Load textures once instead of every frame for efficiency
        pos_tex, _, _ = self._load_texture_from_array(pos_frame)
        neg_tex, _, _ = self._load_texture_from_array(neg_frame)

        #injection_mask = self._create_injection_mask(self.inject_img)
        #pos_tex, x0, y0, w, h = self._create_masked_texture(pos_frame, injection_mask)
        #neg_tex, _, _, _, _ = self._create_masked_texture(neg_frame, injection_mask)
        key_map = {
            "ESC": glfw.KEY_ESCAPE,
            "Q": glfw.KEY_Q,
            "SPACE": glfw.KEY_SPACE,
            "ENTER": glfw.KEY_ENTER,
        }
        start_time = time.perf_counter()
        while not glfw.window_should_close(self.window):
            glClear(GL_COLOR_BUFFER_BIT)
            t = time.perf_counter() - start_time
            if duration is not None and (t >= duration):
                break
            phase = int(t * self.fps) % 2

            # Flash between positive and negative frames
            tex_id = pos_tex if phase == 0 else neg_tex
            self._draw_fullscreen_image(tex_id, self.monitor_W, self.monitor_H)
            #self._draw_masked_region(tex_id, x0, y0, w, h)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()
