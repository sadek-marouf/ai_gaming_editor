# output/reel_generator.py

import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.logger import get_logger
from core.config import Config
from audio.hype_detector import HypeDetector
from output.subtitle_generator import SubtitleGenerator
from scoring.smart_ranker import SmartRanker
from video.ffmpeg_manager import FFmpegManager

logger = get_logger("REEL_GENERATOR")


class ReelGenerator:

    def __init__(self, ffmpeg, temp_dir):
        self.ffmpeg = ffmpeg
        self.temp_dir = temp_dir
        self.subtitle_gen = SubtitleGenerator()
        self.hype_detector = HypeDetector()
        self.ranker = SmartRanker()

        os.makedirs(self.temp_dir, exist_ok=True)

    def generate(self, best_segments, reels_dir):
        os.makedirs(reels_dir, exist_ok=True)

        temp_clips = []
        clip_futures = {}
        profile = Config.GAMING_PROFILE

        with ThreadPoolExecutor(
            max_workers=Config.CLIP_ENCODING_WORKERS
        ) as executor:

            for idx, seg in enumerate(best_segments):
                hook_strength = self.hype_detector.hook_score(seg["text"])

                # =========================================
                # SMART START
                # =========================================
                if hook_strength > 0.7:
                    start = max(0, seg["start"] - 1)
                else:
                    start = max(0, seg["start"] - 3)

                # =========================================
                # CLIP SUBTITLE
                # =========================================
                clip_subtitle = os.path.join(
                    self.temp_dir, f"clip_{idx}.srt"
                )
                self.subtitle_gen.generate(
                    [seg], clip_subtitle, offset=start
                )

                # =========================================
                # SMART GAMING DURATION
                # =========================================
                base_duration = self.ranker.dynamic_duration(seg["score"])

                if seg.get("hype", 0) > 0.75:
                    base_duration *= 0.75
                if seg.get("motion", 0) > 0.8:
                    base_duration *= 0.80

                duration = float(np.clip(
                    base_duration,
                    profile["min_duration"],
                    profile["max_duration"],
                ))

                duration += Config.BREATHING_ROOM
                duration = max(duration, Config.MIN_CLIP_DURATION)

                out = os.path.join(self.temp_dir, f"clip_{idx}.mp4")

                future = executor.submit(
                    self.ffmpeg.cut_clip,
                    start, duration, out, clip_subtitle,
                )
                clip_futures[future] = (idx, out)

            # =========================================
            # COLLECT RESULTS
            # =========================================
            for future in as_completed(clip_futures):
                idx, out = clip_futures[future]
                try:
                    future.result()
                    temp_clips.append((idx, out))
                    logger.info(f"Clip {idx} encoded")
                except Exception as e:
                    logger.error(f"Clip {idx} failed: {e}")

        temp_clips.sort(key=lambda x: x[0])
        clip_paths = [path for _, path in temp_clips]

        if not clip_paths:
            raise RuntimeError("No clips generated")

        final_out = os.path.join(reels_dir, "viral_reel.mp4")
        self.ffmpeg.concat_clips(clip_paths, final_out)

        return final_out
