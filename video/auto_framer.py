# video/auto_framer.py

import cv2
import numpy as np

from core.logger import get_logger
from core.config import Config

logger = get_logger("AUTO_FRAMER")


class AutoFramer:
    """Smart cropping for 9:16 vertical reels.

    Modes:
    - center_crop: crop center of 16:9 to fill 9:16 (no letterbox)
    - smart_crop: detect action area and crop around it
    - split_facecam: split screen with facecam on top, gameplay on bottom
    """

    TARGET_W = Config.TARGET_WIDTH   # 720
    TARGET_H = Config.TARGET_HEIGHT  # 1280

    def __init__(self, game_profile):
        self.profile = game_profile
        framing = self.profile.get_framing_config()
        self.mode = framing["mode"]
        self.facecam_position = framing["facecam_position"]
        self.facecam_ratio = framing["facecam_split_ratio"]

    def get_ffmpeg_vf(self, input_width, input_height, subtitle_path=None):
        """Build FFmpeg video filter string for the configured mode."""

        if self.mode == "split_facecam" and self.facecam_position:
            return self._split_facecam_vf(
                input_width, input_height, subtitle_path,
            )

        if self.mode == "smart_crop":
            return self._smart_crop_vf(
                input_width, input_height, subtitle_path,
            )

        # Default: center_crop
        return self._center_crop_vf(
            input_width, input_height, subtitle_path,
        )

    def _center_crop_vf(self, w, h, subtitle_path=None):
        """Crop center of widescreen to fill 9:16.

        For 1920x1080 input:
        - Target aspect = 9/16 = 0.5625
        - Crop width = h * 9/16 = 1080 * 0.5625 = 607.5 → 608
        - Crop from center horizontally
        - Then scale to 720x1280
        """
        target_aspect = 9 / 16
        src_aspect = w / h if h > 0 else 1.78

        if src_aspect > target_aspect:
            # Wider than target: crop horizontally (most common for gaming)
            crop_w = int(h * target_aspect)
            crop_h = h
            crop_x = f"(iw-{crop_w})/2"
            crop_y = "0"
        else:
            # Taller than target: crop vertically
            crop_w = w
            crop_h = int(w / target_aspect)
            crop_x = "0"
            crop_y = f"(ih-{crop_h})/2"

        vf = (
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={self.TARGET_W}:{self.TARGET_H}"
        )

        if subtitle_path:
            vf += self._subtitle_filter(subtitle_path)

        return vf

    def _smart_crop_vf(self, w, h, subtitle_path=None):
        """Crop focusing on center-bottom (crosshair/action area).

        Shifts the crop area slightly down from center to capture
        the crosshair and action area better.
        """
        target_aspect = 9 / 16
        crop_w = int(h * target_aspect)
        crop_h = h

        # Slightly below center (where crosshair and action usually is)
        crop_x = f"(iw-{crop_w})/2"
        crop_y = "0"

        vf = (
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={self.TARGET_W}:{self.TARGET_H}"
        )

        if subtitle_path:
            vf += self._subtitle_filter(subtitle_path)

        return vf

    def _split_facecam_vf(self, w, h, subtitle_path=None):
        """Split screen: facecam on top, gameplay on bottom.

        Layout (9:16 = 720x1280):
        - Top: facecam cropped from source (30% = 384px)
        - Bottom: gameplay cropped from source (70% = 896px)
        """
        fc = self.facecam_position  # (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
        top_h = int(self.TARGET_H * self.facecam_ratio)
        bottom_h = self.TARGET_H - top_h

        # Facecam source coordinates
        fc_x1 = int(w * fc[0])
        fc_y1 = int(h * fc[1])
        fc_x2 = int(w * fc[2])
        fc_y2 = int(h * fc[3])
        fc_w = fc_x2 - fc_x1
        fc_h = fc_y2 - fc_y1

        # Gameplay: center crop from full frame
        target_aspect = 9 / 16
        gp_crop_w = int(h * target_aspect)

        vf = (
            f"split[a][b];"
            f"[a]crop={fc_w}:{fc_h}:{fc_x1}:{fc_y1},"
            f"scale={self.TARGET_W}:{top_h}[top];"
            f"[b]crop={gp_crop_w}:{h}:(iw-{gp_crop_w})/2:0,"
            f"scale={self.TARGET_W}:{bottom_h}[bottom];"
            f"[top][bottom]vstack"
        )

        if subtitle_path:
            vf += self._subtitle_filter(subtitle_path)

        return vf

    def _subtitle_filter(self, subtitle_path):
        sub_fixed = subtitle_path.replace("\\", "/")
        return (
            f",subtitles='{sub_fixed}':"
            "force_style='"
            f"FontName={Config.SUBTITLE_FONT},"
            f"Fontsize={Config.SUBTITLE_FONTSIZE},"
            "PrimaryColour=&Hffffff&,"
            "OutlineColour=&H000000&,"
            "BackColour=&H66000000&,"
            "BorderStyle=3,"
            "Outline=2,"
            "Shadow=1,"
            f"MarginV={Config.SUBTITLE_MARGIN_V},"
            "Alignment=2'"
        )

    def detect_facecam(self, frame):
        """Auto-detect facecam position in a frame.

        Looks for a small rectangular region with a face,
        typically in a corner of the screen.

        Returns (x1_ratio, y1_ratio, x2_ratio, y2_ratio) or None.
        """
        h, w = frame.shape[:2]

        # Check each corner for a face-like region
        corners = [
            (0, 0, 0.25, 0.25),          # top-left
            (0.75, 0, 1.0, 0.25),         # top-right
            (0, 0.75, 0.25, 1.0),         # bottom-left
            (0.75, 0.75, 1.0, 1.0),       # bottom-right
        ]

        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades
                + "haarcascade_frontalface_default.xml"
            )
        except Exception:
            return None

        for cx1, cy1, cx2, cy2 in corners:
            x1, y1 = int(w * cx1), int(h * cy1)
            x2, y2 = int(w * cx2), int(h * cy2)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3,
            )

            if len(faces) > 0:
                logger.info(
                    f"Facecam detected at corner "
                    f"({cx1:.2f},{cy1:.2f})-({cx2:.2f},{cy2:.2f})"
                )
                return (cx1, cy1, cx2, cy2)

        return None
