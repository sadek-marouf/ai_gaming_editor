# vision/hitmarker_detector.py

import os

import cv2
import numpy as np

from core.logger import get_logger

logger = get_logger("HITMARKER")


class HitmarkerDetector:
    """Detect hitmarkers/kill icons using OpenCV template matching.

    Falls back to color-based detection if no templates are available.
    """

    def __init__(self, game_profile):
        self.profile = game_profile
        self.region = self.profile.HITMARKER_REGION
        self.threshold = self.profile.HITMARKER_THRESHOLD
        self.templates = self._load_templates()

    def _load_templates(self):
        templates = []
        for path in self.profile.HITMARKER_TEMPLATES:
            abs_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                path,
            )
            if os.path.exists(abs_path):
                tmpl = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    templates.append(tmpl)
                    logger.info(f"Template loaded: {path}")
                else:
                    logger.warning(f"Failed to read template: {path}")
            else:
                logger.debug(f"Template not found: {abs_path}")

        if not templates:
            logger.info(
                "No templates available, using color-based detection"
            )

        return templates

    def _extract_region(self, frame):
        if self.region is None:
            return frame

        h, w = frame.shape[:2]
        x1 = int(w * self.region[0])
        y1 = int(h * self.region[1])
        x2 = int(w * self.region[2])
        y2 = int(h * self.region[3])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return frame

        return frame[y1:y2, x1:x2]

    def _template_match(self, gray_roi):
        """Match templates against ROI. Returns max confidence."""
        if not self.templates:
            return 0.0

        best = 0.0

        for tmpl in self.templates:
            # Scale template to multiple sizes for robustness
            for scale in [0.5, 0.75, 1.0, 1.25]:
                th, tw = tmpl.shape[:2]
                new_w = max(4, int(tw * scale))
                new_h = max(4, int(th * scale))

                if new_w >= gray_roi.shape[1] or new_h >= gray_roi.shape[0]:
                    continue

                scaled = cv2.resize(tmpl, (new_w, new_h))
                result = cv2.matchTemplate(
                    gray_roi, scaled, cv2.TM_CCOEFF_NORMED,
                )
                _, max_val, _, _ = cv2.minMaxLoc(result)
                best = max(best, max_val)

        return best

    def _color_hitmarker_detect(self, roi):
        """Detect red/white crosshair changes via color analysis.

        Hitmarkers in PUBG: small red/white X or cross that
        appears briefly at center screen on hit.
        """
        if roi is None or roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red hitmarker detection (low hue or high hue for red wrap)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        red_ratio = np.mean(red_mask > 0)

        # White crosshair/hitmarker flash
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        white_ratio = np.mean(gray > 240)

        # Combine: small bright/red marks in center = hitmarker
        score = 0.0
        if red_ratio > 0.01:
            score += min(red_ratio * 15, 0.6)
        if white_ratio > 0.02:
            score += min(white_ratio * 8, 0.4)

        return min(score, 1.0)

    def detect_frame(self, frame):
        """Detect hitmarker in a single frame.

        Returns:
            score: 0.0-1.0 confidence
            method: 'template' or 'color'
        """
        roi = self._extract_region(frame)

        # Try template matching first
        if self.templates:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            conf = self._template_match(gray_roi)
            if conf >= self.threshold:
                return {"score": round(conf, 4), "method": "template"}

        # Fall back to color-based detection
        color_score = self._color_hitmarker_detect(roi)
        return {"score": round(color_score, 4), "method": "color"}

    def detect_frames(self, frames_dict):
        """Detect hitmarkers across all frames.

        Returns per-second scores list.
        """
        sorted_keys = sorted(frames_dict.keys())
        scores = []

        for sec in sorted_keys:
            frame = frames_dict[sec]
            result = self.detect_frame(frame)
            scores.append(result["score"])

            if result["score"] > 0.5:
                logger.info(
                    f"Hitmarker at {sec}s: "
                    f"score={result['score']} ({result['method']})"
                )

        return scores
