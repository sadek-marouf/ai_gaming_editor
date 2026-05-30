# video/scene_detector.py

import json
import os

from core.logger import get_logger
from core.config import Config

logger = get_logger("SCENE_DETECTOR")


class SceneDetector:

    def __init__(self, video_path, cache_dir=None):
        self.video_path = video_path
        self.cache_dir = cache_dir

    def detect(self):
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "scenes_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    cached = json.load(f)
                logger.info(f"Loaded {len(cached)} scenes from cache")
                return cached

        try:
            from scenedetect import detect as sd_detect
            from scenedetect.detectors import ContentDetector

            scenes = sd_detect(
                self.video_path,
                ContentDetector(threshold=Config.SCENE_THRESHOLD),
            )

            times = [scene[0].get_seconds() for scene in scenes]

            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(times, f)

            logger.info(f"Detected {len(times)} scene changes")
            return times

        except ImportError:
            logger.warning("scenedetect not installed, skipping scene detection")
            return []
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return []
