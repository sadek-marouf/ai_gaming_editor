# core/pipeline.py

import os
import json
import time
import shutil
import tempfile
import math

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.logger import get_logger
from core.config import Config
from core.frame_manager import FrameManager
from core.utils import check_gpu_available
from video.ffmpeg_manager import FFmpegManager
from video.scene_detector import SceneDetector
from audio.audio_engine import AudioEngine
from audio.transcriber import Transcriber
from vision.gaming_detector import GamingEventDetector
from vision.motion_detector import MotionDetector
from scoring.segment_scorer import SegmentScorer
from scoring.smart_ranker import SmartRanker
from output.reel_generator import ReelGenerator

logger = get_logger("PIPELINE")


class Pipeline:

    def __init__(
        self,
        video_path,
        output_dir=None,
        quality=None,
        parallel_workers=None,
    ):
        self.video_path = os.path.abspath(video_path)
        self.base_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        self.output_dir = os.path.abspath(
            os.path.join(
                output_dir or Config.OUTPUT_DIR,
                self.base_name,
            )
        )

        self.reels_dir = os.path.join(self.output_dir, "reels")
        self.audio_path = os.path.join(self.output_dir, "audio.wav")
        self.temp_dir = tempfile.mkdtemp(prefix="viral_")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reels_dir, exist_ok=True)

        self.use_gpu = check_gpu_available()
        self.quality = quality or Config.DEFAULT_QUALITY
        self.workers = parallel_workers or Config.PARALLEL_WORKERS

        self.frame_manager = FrameManager(self.video_path)
        self.ffmpeg = FFmpegManager(
            self.video_path,
            self.output_dir,
            quality=self.quality,
            use_gpu=self.use_gpu,
        )
        self.audio_engine = AudioEngine()
        self.transcriber = Transcriber()
        self.gaming_detector = GamingEventDetector(use_gpu=self.use_gpu)
        self.motion_detector = MotionDetector()
        self.scorer = SegmentScorer(
            self.audio_engine,
            cache_dir=self.output_dir,
        )
        self.ranker = SmartRanker()

        logger.info(
            f"Pipeline initialized - GPU: {self.use_gpu}, "
            f"Workers: {self.workers}"
        )

    # =====================================================
    # MOTION SCORES
    # =====================================================

    def _compute_motion_scores(self):
        frames = self.frame_manager.load_frames()
        duration = len(frames)
        self.motion_detector.reset()

        scores = []
        for sec in range(duration):
            frame = frames.get(sec)
            if frame is None:
                scores.append(0)
                continue

            score = self.motion_detector.compute_motion(frame)
            scores.append(score)

        if not scores:
            return []

        mx = max(scores)
        if mx <= 0:
            return [0.0] * len(scores)

        return [s / mx for s in scores]

    # =====================================================
    # VISUAL SCORES
    # =====================================================

    def _compute_visual_scores(self):
        frames = self.frame_manager.load_frames()
        duration = len(frames)

        scores = []
        prev_frame = None

        for sec in range(duration):
            frame = frames.get(sec)
            if frame is None:
                scores.append(0)
                continue

            resized = cv2.resize(frame, (640, 360))
            score = self.gaming_detector.gaming_visual_score(
                resized, prev_frame,
            )
            scores.append(score)
            prev_frame = resized

        return scores

    # =====================================================
    # GAMING PEAKS
    # =====================================================

    def _detect_gaming_peaks(self):
        audio_scores = self.audio_engine.energy_cache or []
        motion_scores = self._compute_motion_scores()

        length = min(len(audio_scores), len(motion_scores))
        peaks = []

        for i in range(length):
            audio = audio_scores[i] if i < len(audio_scores) else 0
            motion = motion_scores[i] if i < len(motion_scores) else 0

            combined = audio * 0.55 + motion * 0.45

            if combined > Config.COMBINED_PEAK_THRESHOLD:
                peaks.append({
                    "time": i,
                    "strength": round(combined, 3),
                })

        logger.info(f"Detected {len(peaks)} gaming peaks")
        return peaks

    # =====================================================
    # FACE SCORES (optional)
    # =====================================================

    def _compute_face_scores(self):
        try:
            from vision.face_detector import FaceDetector
            detector = FaceDetector(use_gpu=self.use_gpu)
            frames = self.frame_manager.load_frames()
            return detector.face_scores(frames)
        except Exception as e:
            logger.warning(f"Face detection skipped: {e}")
            return []

    # =====================================================
    # RUN PIPELINE
    # =====================================================

    def run(self):
        total_start = time.time()

        try:
            # =====================================================
            # STEP 1: EXTRACT AUDIO
            # =====================================================
            logger.info("[1/9] Extracting audio...")
            self.ffmpeg.extract_audio(self.audio_path)
            self.audio_engine.load_audio(self.audio_path)

            # =====================================================
            # STEP 2: TRANSCRIBE
            # =====================================================
            logger.info("[2/9] Transcribing audio...")
            segments = self.transcriber.transcribe(self.audio_path)

            if not segments:
                logger.error("No segments transcribed")
                return None

            # =====================================================
            # STEP 3: LOAD SHARED FRAMES
            # =====================================================
            logger.info("[3/9] Loading shared frames...")
            self.frame_manager.load_frames()

            # =====================================================
            # STEP 4-7: PARALLEL ANALYSIS
            # =====================================================
            logger.info("[4-7/9] Running parallel analysis...")

            executor = ThreadPoolExecutor(
                max_workers=self.workers
            )

            futures = {
                executor.submit(
                    self.audio_engine.compute_energy
                ): "audio_energy",
                executor.submit(
                    self._compute_motion_scores
                ): "motion",
                executor.submit(
                    self._compute_visual_scores
                ): "visual",
                executor.submit(
                    self._compute_face_scores
                ): "faces",
                executor.submit(
                    SceneDetector(
                        self.video_path,
                        cache_dir=self.output_dir,
                    ).detect
                ): "scenes",
            }

            results = {}

            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                    logger.info(f"  {key} completed")
                except Exception as e:
                    logger.error(f"  {key} failed: {e}")
                    results[key] = [] if key != "audio_energy" else None

            executor.shutdown(wait=False)

            audio_scores = results.get("audio_energy", [])
            motion_scores = results.get("motion", [])
            visual_scores = results.get("visual", [])
            face_scores = results.get("faces", [])
            scenes = results.get("scenes", [])

            # =====================================================
            # STEP 8: GAMING PEAKS + SCORING
            # =====================================================
            logger.info("[8/9] Scoring segments...")

            gaming_peaks = self._detect_gaming_peaks()

            scored = self.scorer.score_segments(
                segments,
                audio_scores,
                motion_scores,
                visual_scores,
                face_scores,
                scenes,
                gaming_peaks,
            )

            best = self.ranker.rank(scored)
            best.sort(key=lambda x: x["start"])

            # Save analysis
            json_path = os.path.join(self.output_dir, "segments.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)

            logger.info(f"Found {len(best)} best segments")
            for i, seg in enumerate(best, 1):
                logger.info(
                    f"  {i}. {seg['text'][:50]}... "
                    f"(score: {seg['score']:.3f})"
                )

            # =====================================================
            # STEP 9: GENERATE REEL
            # =====================================================
            logger.info("[9/9] Generating reel...")

            reel_gen = ReelGenerator(self.ffmpeg, self.temp_dir)
            reel = reel_gen.generate(best, self.reels_dir)

            elapsed = round(time.time() - total_start, 2)
            logger.info(f"FINAL REEL: {reel}")
            logger.info(f"TOTAL TIME: {elapsed} sec")

            return reel

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return None

        finally:
            self._cleanup()

    def _cleanup(self):
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.frame_manager.clear()
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
