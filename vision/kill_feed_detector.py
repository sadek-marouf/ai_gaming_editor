# vision/kill_feed_detector.py

import re

import cv2
import numpy as np
import easyocr

from core.logger import get_logger

logger = get_logger("KILL_FEED")


class KillFeedDetector:
    """Detects kill feed text in specific screen regions using OCR."""

    def __init__(self, game_profile, use_gpu=False):
        self.profile = game_profile
        self.use_gpu = use_gpu
        self._reader = None

        kf_config = self.profile.get_kill_feed_config()
        self.keywords = kf_config["keywords"]
        self.region = kf_config["region"]
        self.kill_log_region = kf_config.get("kill_log_region")
        self.instant_score = kf_config["instant_score"]

        # Sort keywords longest-first to avoid substring conflicts
        if self.keywords:
            self.keywords.sort(key=lambda x: len(x[0]), reverse=True)

    @property
    def reader(self):
        if self._reader is None:
            self._reader = easyocr.Reader(["en"], gpu=self.use_gpu)
        return self._reader

    def _extract_region(self, frame, region):
        if region is None:
            return None

        h, w = frame.shape[:2]
        x1 = int(w * region[0])
        y1 = int(h * region[1])
        x2 = int(w * region[2])
        y2 = int(h * region[3])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _ocr_region(self, roi):
        if roi is None or roi.size == 0:
            return ""

        try:
            results = self.reader.readtext(roi, detail=0)
            return " ".join(results).lower() if results else ""
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def _match_keywords(self, text):
        if not text:
            return 0.0, []

        score = 0.0
        matched = []

        for keyword, weight in self.keywords:
            pattern = re.escape(keyword)
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
                matched.append(keyword)

        return score, matched

    def detect_frame(self, frame):
        """Detect kill feed events in a single frame.

        Returns dict with:
            score: float (0.0 if nothing, high if kill detected)
            matched: list of matched keywords
            is_kill: bool (True if any keyword matched)
        """
        if not self.keywords:
            return {"score": 0.0, "matched": [], "is_kill": False}

        total_score = 0.0
        all_matched = []

        # Check primary kill feed region
        roi = self._extract_region(frame, self.region)
        text = self._ocr_region(roi)
        score, matched = self._match_keywords(text)
        total_score += score
        all_matched.extend(matched)

        # Check secondary kill log region (e.g. PUBG right side)
        if self.kill_log_region:
            roi2 = self._extract_region(frame, self.kill_log_region)
            text2 = self._ocr_region(roi2)
            score2, matched2 = self._match_keywords(text2)
            total_score += score2
            all_matched.extend(matched2)

        is_kill = len(all_matched) > 0

        return {
            "score": round(total_score, 4),
            "matched": all_matched,
            "is_kill": is_kill,
        }

    def detect_frames(self, frames_dict, sample_step=1):
        """Detect kill feed across all frames.

        Returns per-second scores list.
        """
        sorted_keys = sorted(frames_dict.keys())
        scores = []

        for sec in sorted_keys:
            if sec % sample_step != 0:
                scores.append(0.0)
                continue

            frame = frames_dict[sec]
            result = self.detect_frame(frame)

            if result["is_kill"]:
                logger.info(
                    f"Kill detected at {sec}s: "
                    f"{result['matched']} (score={result['score']})"
                )

            scores.append(result["score"])

        return scores
