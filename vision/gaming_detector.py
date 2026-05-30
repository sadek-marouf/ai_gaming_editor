# vision/gaming_detector.py

import re

import cv2
import numpy as np
import easyocr
from core.logger import get_logger

logger = get_logger("GAMING_DETECTOR")


class GamingEventDetector:

    def __init__(self, use_gpu=False):

        # OCR (قراءة النص من الشاشة)
        self.reader = easyocr.Reader(
            ['en'],
            gpu=use_gpu
        )

        # كلمات الأحداث المهمة داخل الألعاب (Game Events)
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
            ("quadra", 1.7),
            ("kill", 1.0),
            ("ace", 2.0),
        ]

    def detect_frame_events(self, frame):

        """
        تحليل frame واحد (صورة واحدة من الفيديو)
        واكتشاف هل فيه حدث مهم أو لا
        """

        score = 0.0
        events = []

        h, w = frame.shape[:2]

        # =====================================
        # 1. OCR TEXT DETECTION
        # =====================================

        try:

            # نركز على أعلى الشاشة (Kill feed غالبًا)
            roi = frame[0:int(h * 0.35), :]

            results = self.reader.readtext(
                roi,
                detail=0
            )

            text = " ".join(results).lower()

            # تحليل الكلمات (word-boundary matching)
            for key, weight in self.keywords:

                if re.search(r'\b' + re.escape(key) + r'\b', text):
                    score += weight
                    events.append({
                        "type": key,
                        "confidence": weight
                    })

        except Exception as e:
            logger.error(f"OCR failed: {e}")

        # =====================================
        # 2. VISUAL GAME SIGNALS
        # =====================================

        try:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Flash detection (وميض/انفجار)
            brightness = np.mean(gray)

            if brightness > 220:
                score += 0.3
                events.append({
                    "type": "flash",
                    "confidence": 0.3
                })

            # Red damage indicator (لون الدم / الضرر)
            b, g, r = cv2.split(frame)

            red_ratio = np.mean((r > 180) & (g < 120))

            if red_ratio > 0.05:
                score += 0.4
                events.append({
                    "type": "damage_flash",
                    "confidence": 0.4
                })

            # Crosshair presence (علامة التصويب)
            center = gray[
                h//2 - 30:h//2 + 30,
                w//2 - 30:w//2 + 30
            ]

            edges = cv2.Canny(center, 100, 200)

            if np.mean(edges > 0) > 0.1:
                score += 0.2
                events.append({
                    "type": "crosshair_activity",
                    "confidence": 0.2
                })

        except Exception as e:
            logger.error(f"Visual detection failed: {e}")

        return {
            "frame_score": round(score, 4),
            "events": events
        }

    def detect_video_events(self, frames_dict):

        """
        تحليل مجموعة فريمات كاملة
        frames_dict = {frame_id: frame}
        """

        results = []

        for frame_id, frame in frames_dict.items():

            result = self.detect_frame_events(frame)

            if result["frame_score"] > 0.3:

                results.append({
                    "frame": frame_id,
                    "strength": result["frame_score"],
                    "events": result["events"]
                })

        logger.info(
            f"Detected {len(results)} gaming events"
        )

        return results