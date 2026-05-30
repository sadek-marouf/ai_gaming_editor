# core/frame_manager.py

import cv2

from core.logger import get_logger

logger = get_logger("FRAME_MANAGER")


class FrameManager:

    def __init__(self, video_path):
        self.video_path = video_path
        self.cache = {}
        self.fps = None
        self.duration = None

    def load_frames(self, step=1):
        if self.cache:
            return self.cache

        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = int(total_frames / self.fps)

        for sec in range(0, self.duration, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * self.fps))
            ret, frame = cap.read()
            if ret:
                self.cache[sec] = frame

        cap.release()

        logger.info(f"Loaded {len(self.cache)} shared frames")
        return self.cache

    def get_frame(self, sec):
        return self.cache.get(sec)

    def clear(self):
        self.cache.clear()
