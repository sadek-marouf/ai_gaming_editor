# video/frame_sampler.py

import cv2
from core.logger import get_logger

logger = get_logger("FRAME_SAMPLER")


class FrameSampler:

    def __init__(self, video_loader):
        self.video_loader = video_loader

        self.cache = {}

    def sample_by_seconds(self, step=1):

        cap = self.video_loader.cap

        fps = self.video_loader.fps
        duration = self.video_loader.duration

        frames = {}

        for sec in range(int(duration)):

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ret, frame = cap.read()

            if ret:
                frames[sec] = frame

        self.cache = frames

        logger.info(f"Sampled {len(frames)} frames")

        return frames