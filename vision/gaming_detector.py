# vision/gaming_detector.py

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
        self.keywords = {
            "kill": 1.0,
            "killed": 1.0,
            "eliminated": 1.0,
            "headshot": 1.2,
            "double kill": 1.3,
            "triple kill": 1.5,
            "quadra": 1.7,
            "ace": 2.0,
            "knocked": 0.9,
            "victory": 1.8,
            "winner": 1.6
        }

    def detect_frame_events(self, frame):

        """
        تحليل frame واحد (صورة واحدة من الفيديو)
        واكتشاف هل فيه حدث مهم أو لا
        """

        score = 0.0
        events = []

        # =====================================
        # 1. OCR TEXT DETECTION
        # =====================================

        try:

            # نركز على أعلى الشاشة (Kill feed غالبًا)
            h, w = frame.shape[:2]

            roi = frame[0:int(h * 0.35), :]

            results = self.reader.readtext(
                roi,
                detail=0
            )

            text = " ".join(results).lower()

            # تحليل الكلمات
            for key, weight in self.keywords.items():

                if key in text:
                    score += weight
                    events.append({
                        "type": key,
                        "confidence": weight
                    })

        except (RuntimeError, ValueError) as e:
            logger.warning(f"OCR failed: {e}")

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

        except (cv2.error, ValueError) as e:
            logger.warning(f"Visual detection failed: {e}")

        return {
            "frame_score": min(score, 1.0),
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
                    "score": result["frame_score"],
                    "events": result["events"]
                })

        logger.info(
            f"Detected {len(results)} gaming events"
        )

        return results