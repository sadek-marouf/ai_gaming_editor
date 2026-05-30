# video/video_loader.py

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

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.duration = self.frame_count / self.fps

        logger.info(f"Loaded video: {self.video_path}")
        logger.info(f"FPS: {self.fps}, Duration: {self.duration:.2f}s")

    def release(self):
        self.cap.release()

    def get_properties(self):
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
        }