# vision/motion_detector.py

import cv2
import numpy as np
from core.logger import get_logger

logger = get_logger("MOTION_DETECTOR")


class MotionDetector:

    def __init__(self):
        self.prev_frame = None

    def reset(self):
        self.prev_frame = None

    def compute_motion(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0

        diff = cv2.absdiff(self.prev_frame, gray)
        score = np.mean(diff) / 255.0

        self.prev_frame = gray

        return float(score)

    def batch_motion(self, frames):

        scores = []

        for frame in frames:
            scores.append(self.compute_motion(frame))

        logger.info(f"Motion computed for {len(scores)} frames")

        return scores