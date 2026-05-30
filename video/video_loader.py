# video/video_loader.py

import cv2
import os
from core.logger import get_logger

logger = get_logger("VIDEO_LOADER")


ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}


class VideoLoader:

    def __init__(self, video_path, allowed_dir=None):

        self.video_path = os.path.realpath(video_path)

        if allowed_dir is not None:
            allowed = os.path.realpath(allowed_dir)
            if not self.video_path.startswith(allowed + os.sep):
                raise ValueError("Video path is outside the allowed directory")

        ext = os.path.splitext(self.video_path)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")

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

        logger.info(f"Loaded video (duration={self.duration:.2f}s)")
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