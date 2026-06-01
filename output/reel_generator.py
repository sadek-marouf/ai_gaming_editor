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
from video.effects_engine import EffectsEngine

logger = get_logger("REEL_GENERATOR")


class ReelGenerator:

    def __init__(self, ffmpeg, temp_dir, game_profile=None):
        self.ffmpeg = ffmpeg
        self.temp_dir = temp_dir
        self.subtitle_gen = SubtitleGenerator()
        self.hype_detector = HypeDetector()
        self.ranker = SmartRanker()

        self.game_profile = game_profile
        self.effects_engine = None
        self.effects_config = {}

        if game_profile:
            self.effects_config = game_profile.get_effects_config()
            self.effects_engine = EffectsEngine(game_profile)

        os.makedirs(self.temp_dir, exist_ok=True)

    def generate(self, best_segments, reels_dir):
        os.makedirs(reels_dir, exist_ok=True)

        temp_clips = []
        clip_futures = {}
        profile = Config.GAMING_PROFILE

        if self.game_profile:
            profile = self.game_profile.get_scoring_weights()

        with ThreadPoolExecutor(
            max_workers=Config.CLIP_ENCODING_WORKERS
        ) as executor:

            for idx, seg in enumerate(best_segments):
                triggers = seg.get("triggers", [])
                primary = self._find_primary_trigger(triggers)

                # =========================================
                # SMART TIMING (trigger-based)
                # =========================================
                if primary:
                    start, duration, trigger_offset = (
                        self._compute_trigger_timing(seg, primary)
                    )
                else:
                    start, duration, trigger_offset = (
                        self._compute_legacy_timing(seg, profile)
                    )

                # =========================================
                # CLIP SUBTITLE
                # =========================================
                clip_subtitle = os.path.join(
                    self.temp_dir, f"clip_{idx}.srt"
                )
                self.subtitle_gen.generate(
                    [seg], clip_subtitle, offset=start
                )

                out = os.path.join(self.temp_dir, f"clip_{idx}.mp4")

                future = executor.submit(
                    self.ffmpeg.cut_clip,
                    start, duration, out, clip_subtitle,
                    trigger_offset=trigger_offset,
                    effects_engine=self.effects_engine,
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

    def _find_primary_trigger(self, triggers):
        """Select strongest trigger event for timing anchor."""
        if not triggers:
            return None

        # Prefer kill_feed > hitmarker > peak
        type_priority = {"kill_feed": 3, "hitmarker": 2, "peak": 1}

        return max(
            triggers,
            key=lambda t: (
                type_priority.get(t["type"], 0),
                t["strength"],
            ),
        )

    def _compute_trigger_timing(self, seg, primary):
        """Smart padding: 3s before trigger, 2-3s after.

        Returns (start, source_duration, trigger_offset_in_clip).
        """
        pre_pad = self.effects_config.get("pre_trigger_pad", 3.0)
        post_pad = self.effects_config.get("post_trigger_pad", 2.5)
        clip_min = self.effects_config.get("trigger_clip_min", 6.0)
        clip_max = self.effects_config.get("trigger_clip_max", 8.0)
        slowmo_enabled = self.effects_config.get("slowmo_enabled", False)
        slowmo_dur = self.effects_config.get("slowmo_duration", 1.5)
        slowmo_speed = self.effects_config.get("slowmo_speed", 0.5)

        trigger_time = primary["time"]

        # Clip start: 3s before trigger, clamped to 0
        start = max(0, trigger_time - pre_pad)

        # Source end: trigger + post_pad
        source_end = trigger_time + post_pad

        # Clamp source duration to target range
        source_dur = source_end - start

        # If slowmo is enabled, the output is longer than source.
        # Slowmo adds (slowmo_dur / slowmo_speed - slowmo_dur) extra seconds.
        # Target output 6-8s → adjust source_dur down to compensate.
        if slowmo_enabled and slowmo_dur > 0:
            slowmo_extra = slowmo_dur * (1.0 / slowmo_speed - 1.0)
        else:
            slowmo_extra = 0.0

        # Expected output = source_dur + slowmo_extra
        expected_output = source_dur + slowmo_extra

        # If too short, extend post_pad
        if expected_output < clip_min:
            needed = clip_min - expected_output
            source_end += needed
            source_dur = source_end - start
            expected_output = source_dur + slowmo_extra

        # If too long, trim post section
        if expected_output > clip_max:
            excess = expected_output - clip_max
            source_end -= excess
            source_dur = source_end - start

        # Safety: ensure minimum source duration
        source_dur = max(source_dur, 3.0)

        trigger_offset = trigger_time - start

        logger.info(
            f"Trigger timing: t={trigger_time}s, "
            f"clip=[{start:.1f}-{start + source_dur:.1f}], "
            f"offset={trigger_offset:.1f}s, type={primary['type']}"
        )

        return start, source_dur, trigger_offset

    def _compute_legacy_timing(self, seg, profile):
        """Original timing logic for segments without triggers.

        Returns (start, duration, None).
        """
        hook_strength = self.hype_detector.hook_score(seg["text"])

        if hook_strength > 0.7:
            start = max(0, seg["start"] - 1)
        else:
            start = max(0, seg["start"] - 3)

        base_duration = self.ranker.dynamic_duration(seg["score"])

        if seg.get("hype", 0) > 0.75:
            base_duration *= 0.75
        if seg.get("motion", 0) > 0.8:
            base_duration *= 0.80

        duration = float(np.clip(
            base_duration,
            profile.get("min_duration", Config.MIN_CLIP_DURATION),
            profile.get("max_duration", 8),
        ))

        duration += Config.BREATHING_ROOM
        duration = max(duration, Config.MIN_CLIP_DURATION)

        return start, duration, None
