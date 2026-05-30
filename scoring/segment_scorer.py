# scoring/segment_scorer.py

import numpy as np

from core.logger import get_logger
from core.config import Config
from audio.hype_detector import HypeDetector
from scoring.ai_scorer import AIScorer

logger = get_logger("SEGMENT_SCORER")


class SegmentScorer:

    def __init__(self, audio_engine, cache_dir=None):
        self.audio_engine = audio_engine
        self.hype_detector = HypeDetector()
        self.ai_scorer = AIScorer(cache_dir=cache_dir)
        self.profile = Config.GAMING_PROFILE

    def score_segments(
        self,
        segments,
        audio_scores,
        motion_scores,
        visual_scores,
        face_scores,
        scene_changes,
        gaming_peaks,
    ):
        results = []

        for seg in segments:
            s = int(seg["start"])
            e = int(seg["end"])
            if e <= s:
                continue

            text = seg["text"]

            # =========================================
            # SAFE SCORE EXTRACTION
            # =========================================
            a = self._safe_mean(audio_scores, s, e)
            m = self._safe_mean(motion_scores, s, e)
            v = self._safe_mean(visual_scores, s, e)
            f = self._safe_mean(face_scores, s, e) if face_scores else 0.0

            # =========================================
            # TEXT-BASED SCORES
            # =========================================
            h = self.hype_detector.hook_score(text)
            ai_data = self.ai_scorer.score(text)
            ai = ai_data["final"]

            gaming_hype = self.hype_detector.gaming_hype_score(text, a, m)

            silence_penalty = self.audio_engine.silence_penalty(
                seg["start"], seg["end"],
            )

            # =========================================
            # SCENE BONUS
            # =========================================
            scene_bonus = 0.0
            for sc in scene_changes:
                if s <= sc <= e:
                    scene_bonus = 0.15
                    break

            # =========================================
            # GAMING PEAK BONUS
            # =========================================
            peak_bonus = 0.0
            for peak in gaming_peaks:
                peak_time = peak["time"]
                if s <= peak_time <= e:
                    peak_bonus = max(
                        peak_bonus,
                        peak["strength"] * 0.25,
                    )

            # =========================================
            # FINAL SCORE
            # =========================================
            score = (
                (a * self.profile["audio_weight"])
                + (m * self.profile["motion_weight"])
                + (v * self.profile["visual_weight"])
                + (f * self.profile["face_weight"])
                + (h * self.profile["hook_weight"])
                + (ai * self.profile["ai_weight"])
                + (gaming_hype * self.profile["hype_weight"])
                + (silence_penalty * 0.05)
                + scene_bonus
                + peak_bonus
            )

            results.append({
                "start": seg["start"],
                "end": seg["end"],
                "score": round(score, 4),
                "text": text,
                "audio": round(a, 3),
                "motion": round(m, 3),
                "visual": round(v, 3),
                "faces": round(f, 3),
                "hook": round(h, 3),
                "ai": round(ai, 3),
                "hype": round(gaming_hype, 3),
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Scored {len(results)} segments")
        return results

    def _safe_mean(self, scores, s, e):
        if not scores:
            return 0.0
        safe_s = min(s, len(scores))
        safe_e = min(e, len(scores))
        if safe_s >= safe_e:
            return 0.0
        try:
            return float(np.mean(scores[safe_s:safe_e]))
        except Exception:
            return 0.0
