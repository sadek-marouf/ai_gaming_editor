# vision/gaming_detector.py

import re

import cv2
import numpy as np
import easyocr
from core.logger import get_logger

logger = get_logger("GAMING_DETECTOR")


class GamingEventDetector:

    def __init__(self, use_gpu=False):

        self.reader = easyocr.Reader(
            ['en'],
            gpu=use_gpu
        )

        # sorted longest-first to avoid substring conflicts
        self.keywords = [
            ("triple kill", 1.5),
            ("double kill", 1.3),
            ("eliminated", 1.0),
            ("headshot", 1.2),
            ("victory", 1.8),
            ("knocked", 0.9),
            ("killed", 1.0),
            ("winner", 1.6),
            ("finish", 0.8),
            ("quadra", 1.7),
            ("kill", 1.0),
            ("ace", 2.0),
        ]

    def detect_gaming_text(self, frame):
        try:
            h, w = frame.shape[:2]
            roi = frame[0:int(h * 0.35), :]
            results = self.reader.readtext(roi, detail=0)

            if not results:
                return 0.0

            text = " ".join(results).lower()
            score = 0.0

            for key, weight in self.keywords:
                if re.search(r'\b' + re.escape(key) + r'\b', text):
                    score += weight

            return round(score, 4)

        except (RuntimeError, ValueError) as e:
            logger.warning(f"OCR detection failed: {e}")
            return 0.0

    def gaming_visual_score(self, frame, prev_frame=None):
        score = 0.0

        # =====================================
        # FRAME DIFFERENCE (motion)
        # =====================================
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            motion = np.mean(diff) / 255.0
            score += motion * 0.35

        # =====================================
        # BRIGHT FLASHES
        # =====================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright_ratio = np.mean(gray > 220)
        score += bright_ratio * 0.25

        # =====================================
        # COLOR SATURATION
        # =====================================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        score += saturation * 0.20

        # =====================================
        # CROSSHAIR DETECTION (HoughLines)
        # =====================================
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        crosshair_zone = gray[
            center_y - 40:center_y + 40,
            center_x - 40:center_x + 40,
        ]

        if crosshair_zone.size > 0:
            lines = cv2.HoughLinesP(
                crosshair_zone, 1, np.pi / 180,
                threshold=25, minLineLength=10, maxLineGap=3,
            )
            if lines is not None:
                vertical = 0
                horizontal = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dx, dy = abs(x2 - x1), abs(y2 - y1)
                    if dx < 4 and dy > 8:
                        vertical += 1
                    if dy < 4 and dx > 8:
                        horizontal += 1
                if vertical > 0 and horizontal > 0:
                    score += 0.25

        # =====================================
        # EDGE DENSITY
        # =====================================
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        score += edge_density * 0.20

        # =====================================
        # KILL FEED (top-right corner)
        # =====================================
        h, w = frame.shape[:2]
        kill_zone = frame[0:int(h * 0.25), int(w * 0.60):w]
        kill_gray = cv2.cvtColor(kill_zone, cv2.COLOR_BGR2GRAY)
        kill_edges = cv2.Canny(kill_gray, 150, 300)
        text_density = np.mean(kill_edges > 0)
        if text_density > 0.08:
            score += 0.20

        # =====================================
        # RED / DAMAGE FLASH
        # =====================================
        b, g, r = cv2.split(kill_zone)
        red_mask = (r > 170) & (g < 140)
        red_ratio = np.mean(red_mask)
        if red_ratio > 0.03:
            score += 0.15

        # =====================================
        # OCR GAMING EVENTS
        # =====================================
        ocr_score = self.detect_gaming_text(frame)
        score += ocr_score * 0.35

        return min(score, 1.0)

    def detect_frame_events(self, frame, prev_frame=None):
        score = self.gaming_visual_score(frame, prev_frame)
        events = []

        if score > 0.3:
            events.append({
                "type": "gaming_visual",
                "confidence": round(score, 4),
            })

        return {
            "frame_score": round(score, 4),
            "events": events,
        }

    def detect_video_events(self, frames_dict):
        results = []
        prev_frame = None

        for frame_id in sorted(frames_dict.keys()):
            frame = frames_dict[frame_id]
            resized = cv2.resize(frame, (640, 360))

            result = self.detect_frame_events(resized, prev_frame)
            prev_frame = resized

            if result["frame_score"] > 0.3:
                results.append({
                    "frame": frame_id,
                    "strength": result["frame_score"],
                    "events": result["events"],
                })

        logger.info(f"Detected {len(results)} gaming events")
        return results
