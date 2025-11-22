import time
import threading
import queue
import pathlib

import numpy as np
import scipy as sp
import cv2
import yaml
from yaml.loader import FullLoader
from PIL import Image
import matplotlib.pyplot as plt

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

class EventAttack:
    def __init__(self, attack_config_path : pathlib.Path):
        self.attack_config = self._parse_attack_config(attack_config_path)
        print(self.attack_config)

        # Initialize global settings
        self.duration = self.attack_config["duration"]
        
        # Spatial settings
        self.dx = self.attack_config["dx"]
        self.dy = self.attack_config["dy"]
        self.scale = self.attack_config["scale"]
        
        # Injection settings
        self.injection_config = self.attack_config["injection_config"]

        # Temporal vibration settings
        self.vibration_config = self.attack_config["vibration_config"]
        self.monitor_id = self.attack_config["monitor_id"]
        self.attack_method = self.attack_config["attack_method"]

        # Initialize OpenGL window
        self.window, self.W, self.H = self._init_window(monitor_id=self.monitor_id)

        # Read in both the carrier and attack images
        self.inject_img = cv2.imread(self.attack_config["injected_image"], cv2.IMREAD_UNCHANGED)
        self.carrier_img = cv2.imread(self.attack_config["carrier_image"], cv2.IMREAD_UNCHANGED)
        self.inject_img = cv2.resize(self.inject_img, (self.monitor_W, self.monitor_H), interpolation=cv2.INTER_LINEAR)
        self.carrier_img = cv2.resize(self.carrier_img, (self.monitor_W, self.monitor_H), interpolation=cv2.INTER_LINEAR)

        # Initialize fullscreen quad
        self.vao = self._init_fullscreen_quad()

        # Compile simple shader
        self.vertex_src = """
            #version 330 core
            layout(location = 0) in vec2 in_pos;  // clip space [-1,1]
            layout(location = 1) in vec2 in_uv;   // texture coords [0,1]

            out vec2 frag_uv;

            void main() {
                frag_uv = in_uv;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
        """
        self.frag_src = """
            #version 330 core
            in vec2 frag_uv;
            out vec4 out_color;

            uniform sampler2D tex;          // flicker texture
            uniform sampler2D mask_tex;     // optional mask texture
            uniform int use_mask;           // 1 = masked_region, 0 = fullscreen

            void main() {
                vec4 color = texture(tex, frag_uv);

                if(use_mask == 1) {
                    float alpha = texture(mask_tex, frag_uv).r; // mask in red channel
                    color *= alpha;
                }

                out_color = color;
            }
        """
        self.shader = compileProgram(
            compileShader(self.vertex_src, GL_VERTEX_SHADER),
            compileShader(self.frag_src, GL_FRAGMENT_SHADER)
        )

        # Store tex ID in one location
        self.tex_uniform = glGetUniformLocation(self.shader, "tex")

    # Parse YAML config file
    def _parse_attack_config(self, config_path : pathlib.Path):
        with open(config_path, 'r') as stream:
            return yaml.load(stream, Loader=FullLoader)

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

    def _scale_mask(self, mask, scale_factor):
        """Scale a boolean mask around its center by a given factor."""
        h, w = mask.shape
        center = (w // 2, h // 2)

        # Convert boolean mask to uint8 image for resizing
        mask_uint8 = (mask.astype(np.uint8) * 255)

        # Compute new size
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Resize using OpenCV (preserves general shape)
        resized = cv2.resize(mask_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Create blank canvas and paste resized mask centered
        scaled = np.zeros_like(mask_uint8)
        x0 = max(center[0] - new_w // 2, 0)
        y0 = max(center[1] - new_h // 2, 0)
        x1 = min(x0 + new_w, w)
        y1 = min(y0 + new_h, h)

        resized_x0 = max(0, - (center[0] - new_w // 2))
        resized_y0 = max(0, - (center[1] - new_h // 2))
        resized_x1 = resized_x0 + (x1 - x0)
        resized_y1 = resized_y0 + (y1 - y0)

        scaled[y0:y1, x0:x1] = resized[resized_y0:resized_y1, resized_x0:resized_x1]
        return scaled > 0

    def _show_mask_cv(self, mask, winname="mask", key_time=0):
        # mask: boolean HxW
        vis = (mask.astype(np.uint8) * 255)  # 0 or 255
        cv2.imshow(winname, vis)
        cv2.waitKey(key_time)
        cv2.destroyWindow(winname)

    def _compute_brightness_gradient(self, gray_frame, ksize=3):
        grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=ksize)

        mag = np.hypot(grad_x, grad_y)
        mag = mag / mag.max()

        nx = grad_x / np.max(grad_x)
        ny = grad_y / np.max(grad_y)

        return mag.astype(np.float32), nx.astype(np.float32), ny.astype(np.float32)

    def _create_edge_gradients(self, inject_mask, color_deltaE, scale_area=3.0, edge_threshold=0.05, ksize=3):
        # Convert boolean mask to uint8
        mask_uint8 = (inject_mask.astype(np.uint8) * 255)

        # Dilate slightly to include some border
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)

        # Invert mask for distance transform (distance from outside the mask)
        inverted_mask = cv2.bitwise_not(dilated_mask)

        # Compute distance transform
        distance = cv2.distanceTransform(inverted_mask, distanceType=cv2.DIST_L2, maskSize=5)
        distance_clipped = np.clip(distance, 0, scale_area)

        # Normalize to [0,1] and invert (max at boundary)
        distance_norm = 1.0 - (distance_clipped / scale_area)

        # Apply color delta scaling
        delta_map_L = color_deltaE * distance_norm.astype(np.float32)

        # Optional: normals can be approximated by Sobel of mask
        grad_x = cv2.Sobel(mask_uint8.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask_uint8.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        mag_n = np.hypot(grad_x, grad_y) + 1e-6
        nx = grad_x / mag_n
        ny = grad_y / mag_n

        return delta_map_L, (nx, ny, mag_n)

    def _visualize_gradient(self, delta_map_L, normals_tuple, spacing=20, scale=5.0):
        nx, ny, mag_n = normals_tuple
        H,W = delta_map_L.shape

        plt.figure(figsize=(10,8))
        plt.imshow(delta_map_L, cmap='viridis')
        plt.colorbar(label='delta L (L* units)')
        plt.title('Delta Lightness Map')

        # quiver overlay: downsample for clarity
        ys = np.arange(0, H, spacing)
        xs = np.arange(0, W, spacing)
        X, Y = np.meshgrid(xs, ys)
        U = nx[Y, X]
        V = -ny[Y, X]   # flip y for plotting coordinate system
        M = mag_n[Y, X]
        plt.quiver(X, Y, U*scale*M, V*scale*M, M, cmap='coolwarm', scale=1, width=0.003)
        plt.gca().invert_yaxis()
        plt.show()

    def _vibrate_mask(self, mask, t):
        # Extract vibration-specific settings:
        vib_amp = self.vibration_config["amp"]
        vib_freq = self.vibration_config["freq"]
        vib_angle = self.vibration_config["angle"]

        print(f"vib_amp: {vib_amp}")
        print(f"vib_freq: {vib_freq}")
        print(f"vib_angle: {vib_angle}")

        if vib_freq in [0, None]:
            return mask
        
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        
        # Compute pixel displacement
        theta = np.deg2rad(vib_angle)
        vib_dx = vib_amp * np.cos(theta) * np.sin(2 * np.pi * vib_freq * t)
        vib_dy = vib_amp * np.sin(theta) * np.sin(2 * np.pi * vib_freq * t)

        print(f"vib_dx: {vib_dx}")
        print(f"vib_dy: {vib_dy}")
        
        # Shift only the mask
        M = np.float32([[1, 0, vib_dx], [0, 1, vib_dy]])
        shifted_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return shifted_mask > 0

    def _srgb_to_linear(self, srgb):
        # srgb assumed in [0,1] float
        a = 0.055
        linear = np.where(srgb <= 0.04045,
                        srgb / 12.92,
                        ((srgb + a) / (1.0 + a)) ** 2.4)
        return linear

    def _rgb_to_luminance_linear(self, img_bgr_uint8):
        """
        Input: img_bgr_uint8 -- OpenCV image in BGR uint8 (0..255)
        Returns: luminance_linear -- float32 array, linear irradiance (proportional to I)
        """
        # convert to float [0,1]
        img = img_bgr_uint8.astype(np.float32) / 255.0

        # BGR -> RGB ordering for formula weights
        b = img[..., 0]
        g = img[..., 1]
        r = img[..., 2]

        # undo sRGB gamma to get linear intensities for each channel
        r_lin = self._srgb_to_linear(r)
        g_lin = self._srgb_to_linear(g)
        b_lin = self._srgb_to_linear(b)

        # luminance (Rec. 709 / BT.601-like weights; match your camera/model)
        # Y = 0.2126 R + 0.7152 G + 0.0722 B  (Rec.709)
        # Or use ITU-R BT.601: Y = 0.299 R + 0.587 G + 0.114 B
        Y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

        return Y.astype(np.float32)

    def _log_intensity(self, img_bgr_uint8, eps=1e-6, use_natural_log=True):
        Y = self._rgb_to_luminance_linear(img_bgr_uint8)
        Y = np.maximum(Y, eps)
        if use_natural_log:
            return np.log(Y)
        else:
            return np.log10(Y)

    def _inject_img_lab(self, carrier_img, color_deltaE=1.5, t=0):
        # Grab mask pattern
        mask_pattern = self.injection_config["mask_pattern"]

        # Convert images to LAB space
        carrier_lab = cv2.cvtColor(carrier_img, cv2.COLOR_BGR2LAB)

        # Create injection mask to embed into carrier image
        injection_mask = self._create_injection_mask(self.inject_img)

        # Convert boolean mask to uint8
        im_uint8 = (injection_mask.astype(np.uint8) * 255)
            
        # Find all contours (outer + inner)
        contours, _ = cv2.findContours(im_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours onto a single-channel mask
        contour_mask = np.zeros_like(im_uint8)
        cv2.drawContours(contour_mask, contours, -1, color=255, thickness=5)

        # Convert back to boolean
        injection_mask = contour_mask > 0 if mask_pattern == "edges" else injection_mask > 0
        
        # Calculate the lightness change for the color difference color_deltaE
        carrier_L = carrier_lab[:, :, 0].astype(np.float32)
        k_L = 1.0
        S_L = 1 + ((0.015 * (carrier_L - 50) ** 2) / (np.sqrt(20 + (carrier_L - 50) ** 2)))
        delta_L = k_L * S_L * color_deltaE

        # Inject lightness change of 1-3 units along b* space
        
        # Spatial scaling/shifting
        scaled_mask = self._scale_mask(injection_mask, scale_factor=self.scale)
        translated_mask = self._shift_mask(mask=scaled_mask, dx=self.dx, dy=self.dy)

        # Vibrate mask
        total_mask = self._vibrate_mask(mask=translated_mask, t=t)
        delta_map_L, (nx, ny, mag_n) = self._create_edge_gradients(
            total_mask,
            color_deltaE=color_deltaE,
            scale_area=6.0,          # falloff radius (pixels)
            edge_threshold=0.04,     # edge detection threshold
            ksize=3
        )

        #delta_L *= (1 + 0.3 * delta_map_L / np.max(np.abs(delta_map_L) + 1e-6))

        # Create the negative and positive images
        pos_carrier_lab = carrier_lab.copy().astype(np.float32)
        neg_carrier_lab = carrier_lab.copy().astype(np.float32)

        # Apply delta only to masked regions, leave others unchanged
        mask_indices = np.where(total_mask)
        pos_carrier_lab[:, :, 0][mask_indices] = np.clip(
            pos_carrier_lab[:, :, 0][mask_indices] + delta_L[mask_indices], 0, 255
        )
        neg_carrier_lab[:, :, 0][mask_indices] = np.clip(
            neg_carrier_lab[:, :, 0][mask_indices] - delta_L[mask_indices], 0, 255
        )

        # Apply constant 1 unit difference among b* space
        pos_carrier_lab[:, :, 2][mask_indices] = np.clip(
            pos_carrier_lab[:, :, 2][mask_indices] + 0, 0, 255
        )
        neg_carrier_lab[:, :, 2][mask_indices] = np.clip(
            neg_carrier_lab[:, :, 2][mask_indices] - 0, 0, 255
        )

        pos_carrier_lab = np.clip(pos_carrier_lab, 0, 255).astype(np.uint8)
        neg_carrier_lab = np.clip(neg_carrier_lab, 0, 255).astype(np.uint8)

        # Convert both images back to RGB space
        pos_carrier = cv2.cvtColor(pos_carrier_lab, cv2.COLOR_LAB2RGB)
        neg_carrier = cv2.cvtColor(neg_carrier_lab, cv2.COLOR_LAB2RGB)

        return pos_carrier, neg_carrier, total_mask
    
    def _inject_img_log(self, carrier_img, contrast, t=0):
        # Convert images to log(I) space
        carrier_intensity = self._log_intensity(carrier_img, eps=1e-6, use_natural_log=True)
        inject_intensity = self._log_intensity(self.inject_img, eps=1e-6, use_natural_log=True)

        print(f"Max carrier intensity: {np.max(carrier_intensity)}")
        print(f"Max injected intensity: {np.max(inject_intensity)}")

        # Create injection mask
        injection_mask = self._create_injection_mask(self.inject_img)

        # Convert boolean mask to uint8
        im_uint8 = (injection_mask.astype(np.uint8) * 255)

        # Find all contours (outer + inner)
        contours, _ = cv2.findContours(im_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours onto a single-channel mask
        contour_mask = np.zeros_like(im_uint8)
        cv2.drawContours(contour_mask, contours, -1, color=255, thickness=2)

        # Convert back to boolean
        injection_mask = contour_mask > 0

        # Spatial scaling/shifting
        scaled_mask = self._scale_mask(injection_mask, scale_factor=self.scale)
        translated_mask = self._shift_mask(mask=scaled_mask, dx=self.dx, dy=self.dy)

        # Compute logI delta
        delta_logI = np.log10(1.0 + contrast)
        print(f"delta_logI: {delta_logI}")
        
        # Vibrating mask:
        total_mask = self._vibrate_mask(mask=translated_mask, t=t)

        # Create the negative and positive images
        pos_carrier_I = carrier_intensity.copy().astype(np.float32)
        neg_carrier_I = carrier_intensity.copy().astype(np.float32)

        # Apply delta only to masked regions, leave others unchanged
        mask_indices = np.where(total_mask)
        pos_carrier_I[mask_indices] += delta_logI
        neg_carrier_I[mask_indices] -= delta_logI

        pos_linear = 10 ** pos_carrier_I
        neg_linear = 10 ** neg_carrier_I

        pos_linear_uint8 = np.clip(pos_linear * 255, 0, 255).astype(np.uint8)
        neg_linear_uint8 = np.clip(neg_linear * 255, 0, 255).astype(np.uint8)

        # Convert both images back to RGB space
        pos_carrier = cv2.cvtColor(pos_linear_uint8, cv2.COLOR_GRAY2BGR)
        neg_carrier = cv2.cvtColor(neg_linear_uint8, cv2.COLOR_GRAY2BGR)

        #cv2.imshow("Positive carrier log", pos_carrier)
        #cv2.waitKey(0)

        #cv2.imshow("Negative carrier log", neg_carrier)
        #cv2.waitKey(0)

        return pos_carrier, neg_carrier, total_mask

    def _inject_img(self, carrier_img, t=0):
        mode = self.injection_config["injection_type"]
        if mode == "lab":
            color_deltaE = self.injection_config["color_deltaE"]
            return self._inject_img_lab(carrier_img, color_deltaE, t)
        
        elif mode == "log":
            contrast = self.injection_config["contrast"]
            return self._inject_img_log(carrier_img, contrast)

    # Setup functions
    def _init_window(self, monitor_id : int):
        if not glfw.init():
            raise RuntimeError("Failed to init GLFW")

        monitor = glfw.get_monitors()[monitor_id]
        mode = glfw.get_video_mode(monitor)

        # Resolve width/height differences across bindings
        try:
            self.monitor_W, self.monitor_H = mode.size.width, mode.size.height
            #self.monitor_W, self.monitor_H = 1920, 1080
        except AttributeError:
            self.monitor_W, self.monitor_H = getattr(mode, "width", 800), getattr(mode, "height", 600)

        print(f"Monitor resolution: {self.monitor_W}x{self.monitor_H}, refresh rate: {getattr(mode, 'refresh_rate', 60)} Hz")

        # Fullscreen window
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.FALSE)
        glfw.window_hint(glfw.SRGB_CAPABLE, glfw.FALSE)

        window = glfw.create_window(self.monitor_W, self.monitor_H, "Image Flicker", monitor, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)
        glfw.swap_interval(1)  # enable vsync

        # Projection to pixel coordinates
        glViewport(0, 0, self.monitor_W, self.monitor_H)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.monitor_W, 0, self.monitor_H, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #glEnable(GL_BLEND)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        return window, self.monitor_W, self.monitor_H

    def _init_fullscreen_quad(self):
        quad_vertices = np.array([
            # positions   # texcoords
            -1.0, -1.0,   0.0, 0.0,  # bottom-left
            1.0, -1.0,   1.0, 0.0,  # bottom-right
            1.0,  1.0,   1.0, 1.0,  # top-right

            -1.0, -1.0,   0.0, 0.0,  # bottom-left
            1.0,  1.0,   1.0, 1.0,  # top-right
            -1.0,  1.0,   0.0, 1.0,  # top-left
        ], dtype=np.float32)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

        # positions
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # uvs
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

        return vao

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

    def _draw_fullscreen_image(self, tex_id):
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shader)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glUniform1i(self.tex_uniform, 0)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

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

    def _draw_texture(self, tex_id):
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glUniform1i(self.tex_uniform, 0)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
    # Attack Functions
    def _precompute_vibration_textures(self, pos_frame, neg_frame, mask):
        """
        Precompute textures for the entire vibration cycle.
        Returns list of tuples: (pos_tex_info, neg_tex_info) for each time step.
        """
        def _apply_mask_to_frame(attack_frame, mask, carrier_frame):
            """
            Apply the vibrating mask to combine attack frame with carrier frame.
            Where mask is True, use attack_frame; where False, use carrier_frame.
            """
            result = carrier_frame.copy()
            result[mask] = attack_frame[mask]
            return result
        
        vib_freq = self.vibration_config["freq"]
        vib_amp = self.vibration_config["amp"]
        fps = self.attack_method["fps"]
        x0 = y0 = w = h = None
        textures = []

        # If no vibration, return single frame pair
        if vib_freq == 0 or vib_amp == 0:
            pos_tex, w, h = self._load_texture_from_array(pos_frame)
            neg_tex, _, _ = self._load_texture_from_array(neg_frame)
            textures.append((pos_tex, neg_tex))
            return textures, x0, y0, w, h

        # Compute number of frames per vibration period
        period = 1.0 / vib_freq
        num_frames = int(np.ceil(period * fps))
        print(f"Number of vibration frames: {num_frames}")

        for i in range(num_frames):
            t = i / fps
            vib_mask = self._vibrate_mask(mask, t)

            # Apply mask to create complete frames
            pos_complete = _apply_mask_to_frame(pos_frame, vib_mask, self.carrier_img)
            neg_complete = _apply_mask_to_frame(neg_frame, vib_mask, self.carrier_img)
            
            pos_full_tex, w, h = self._load_texture_from_array(pos_complete)
            neg_full_tex, _, _ = self._load_texture_from_array(neg_complete)
            textures.append((pos_full_tex, neg_full_tex))

        return textures, x0, y0, w, h

    def _fixed_flicker(self) -> None:
        fps = self.attack_method["fps"]

        # Generate both positive and negative frames
        pos_frame, neg_frame, base_mask = self._inject_img(self.carrier_img)

        # Precompute textures for vibration cycle
        texture_sequence, x0, y0, w, h = self._precompute_vibration_textures(
            pos_frame, neg_frame, base_mask
        )

        num_vibration_frames = len(texture_sequence)
        start_time = time.perf_counter()

        while not glfw.window_should_close(self.window):
            glClear(GL_COLOR_BUFFER_BIT)

            t = time.perf_counter() - start_time
            if self.duration is not None and t >= self.duration:
                break

            # Determine current vibration frame index
            vib_phase = (t * self.vibration_config["freq"]) % 1.0
            vib_index = int(vib_phase * num_vibration_frames) % num_vibration_frames
            pos_tex, neg_tex = texture_sequence[vib_index]

            # Alternate between positive and negative frames
            phase = int(t * fps) % 2
            tex_id = pos_tex if phase == 0 else neg_tex

            self._draw_texture(tex_id)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        # Cleanup textures
        try:
            for pos_tex, neg_tex in texture_sequence:
                glDeleteTextures([pos_tex, neg_tex])
            glfw.destroy_window(self.window)
            glfw.terminate()
        except Exception:
            pass

    def _lfm_flicker(self) -> None:
        # Extract LFM-specific parameters
        start_freq = self.attack_method["start_freq"]
        end_freq = self.attack_method["end_freq"]
        sweep_duration = self.attack_method["sweep_duration"]
        reverse_sweep = self.attack_method["reverse_sweep"]

        # Generate both positive and negative frames for flicker fusion
        pos_frame, neg_frame, mask = self._inject_img(self.carrier_img, color_deltaE=self.color_deltaE)

        # Generate OpenGL textures for both positive and negative frames
        pos_tex, x0, y0, w, h = self._create_masked_texture(pos_frame, mask)
        neg_tex, _, _, _, _ = self._create_masked_texture(neg_frame, mask)

        frame_count = 0
        start_time = time.perf_counter_ns()
        while not glfw.window_should_close(self.window):
            glClear(GL_COLOR_BUFFER_BIT)
            t = (time.perf_counter_ns() - start_time) / 1e9
            
            # Break if we have gone over duration
            if self.duration is not None and (t >= self.duration):
                break
            
            phase = min((t % sweep_duration) / sweep_duration, 1.0)
            if reverse_sweep:
                if phase > 0.5:
                    phase = 1.0 - (phase - 0.5) * 2
                else:
                    phase = phase * 2

            # Compute instantaneous frequency
            current_freq = start_freq + (end_freq - start_freq) * phase
            current_period = 1.0 / current_freq
            local_phase = (t % current_period) / current_period
            
            tex_id = pos_tex if local_phase < 0.5 else neg_tex
            self._draw_masked_region(tex_id, x0, y0, w, h)


            glfw.swap_buffers(self.window)
            glfw.poll_events()

            frame_count += 1

        glfw.terminate()

        cv2.destroyAllWindows()


    def _contrast_injection(self) -> None:
        # Extract Contrast Injection-specifc parameters
        injection_period = self.attack_method["injection_period"]
        fps = self.attack_method["fps"]
        frame_dur = 1.0 / float(fps)

        # Generate positive and negative frames
        pos_frame, neg_frame, mask = self._inject_img(self.carrier_img, color_deltaE=self.color_deltaE)
        
        # Generate black frame and normal frame for high contrast
        black_frame = np.zeros((self.monitor_H, self.monitor_W, 3), dtype=np.uint8)

        # Generate OpenGL textures for both positive and negative frames
        black_tex, _, _ = self._load_texture_from_array(black_frame)
        pos_tex, x0, y0, w, h = self._create_masked_texture(pos_frame, mask)
        neg_tex, _, _, _, _ = self._create_masked_texture(neg_frame, mask)
        normal_tex, _, _ = self._load_texture_from_array(self.carrier_img)


        start_time = time.perf_counter()
        while not glfw.window_should_close(self.window):
            glClear(GL_COLOR_BUFFER_BIT)
            t = time.perf_counter() - start_time

            # stop after requested duration, if any
            if self.duration is not None and t >= self.duration:
                break

            # Where are we in the injection cycle?
            cycle_t = t % injection_period

            # Sequence:
            # 0 .. frame_dur                -> black
            # frame_dur .. 2*frame_dur     -> pos (masked)
            # 2*frame_dur .. 3*frame_dur   -> neg (masked)
            # else                         -> normal (full)
            if cycle_t < frame_dur:
                # single black frame
                self._draw_fullscreen_image(black_tex, self.monitor_W, self.monitor_H)
            elif cycle_t < 2 * frame_dur:
                # single positive (masked)
                self._draw_masked_region(pos_tex, x0, y0, w, h)
            elif cycle_t < 3 * frame_dur:
                # single negative (masked)
                self._draw_masked_region(neg_tex, x0, y0, w, h)
            else:
                # normal full image the rest of the cycle
                self._draw_fullscreen_image(normal_tex, self.monitor_W, self.monitor_H)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()
          

    def flicker(self) -> None:
        # Determine which attack method we should use
        match self.attack_method["type"]:
            # Fixed frequency: flicker image at constant frequency
            case "fixed":
                self._fixed_flicker()

            # Variable frequency: flicker image at LFM between [start_freq, end_freq]
            case "lfm":
                self._lfm_flicker()

            # Contrast injection: inject a black image and then attack image at certain duration
            case "contrast_injection":
                self._contrast_injection()

            # Video injection: translate an mp4 video into flicker pattern
            case "video_injection":
                self._video_injection()
