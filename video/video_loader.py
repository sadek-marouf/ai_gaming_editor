# video/video_loader.py

import math

import cv2
import os
from core.logger import get_logger

logger = get_logger("VIDEO_LOADER")


class VideoLoader:

    def __init__(self, video_path):

        self.video_path = os.path.abspath(video_path)

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video")

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = raw_fps if raw_fps and not math.isnan(raw_fps) else 30
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.duration = self.frame_count / self.fps

        logger.info(f"Loaded video: {self.video_path}")
        logger.info(f"FPS: {self.fps}, Duration: {self.duration:.2f}s")

    def release(self):
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def get_properties(self):
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
        }