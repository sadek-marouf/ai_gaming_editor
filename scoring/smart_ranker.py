# scoring/smart_ranker.py

from core.logger import get_logger
from core.config import Config

logger = get_logger("SMART_RANKER")


class SmartRanker:

    def __init__(self, max_clips=None):
        self.max_clips = max_clips or Config.MAX_REEL_CLIPS

    def rank(self, scored_segments):
        if not scored_segments:
            return []

        final = []
        current = scored_segments[0]
        final.append(current)

        remaining = scored_segments[1:]

        while remaining and len(final) < self.max_clips:
            best_next = None
            best_score = -1

            for seg in remaining:
                transition = self._transition_score(current, seg)
                combined = seg["score"] * 0.7 + transition * 0.3

                if combined > best_score:
                    best_score = combined
                    best_next = seg

            if best_next is None:
                break

            final.append(best_next)
            remaining.remove(best_next)
            current = best_next

        logger.info(f"Ranked {len(final)} clips for reel")
        return final

    def _transition_score(self, seg1, seg2):
        score = 0.0

        score += max(0, 1 - abs(seg1.get("audio", 0) - seg2.get("audio", 0)))
        score += max(0, 1 - abs(seg1.get("motion", 0) - seg2.get("motion", 0)))
        score += max(0, 1 - abs(seg1.get("visual", 0) - seg2.get("visual", 0)))
        score += max(0, 1 - abs(seg1.get("faces", 0) - seg2.get("faces", 0)))

        return score / 4

    def dynamic_duration(self, score):
        if score >= 1.15:
            return 3.5
        if score >= 1.0:
            return 4.5
        if score >= 0.85:
            return 6.0
        if score >= 0.7:
            return 7.0
        return 8.0
