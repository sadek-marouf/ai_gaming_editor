# audio/hype_detector.py

import re

from core.logger import get_logger

logger = get_logger("HYPE_DETECTOR")


class HypeDetector:

    def __init__(self):
        self.hook_words = [
            "مستحيل", "لن تصدق", "صدمة", "كارثة", "سر",
            "فضيحة", "أخطر", "عاجل", "انتبه", "تحذير",
        ]

        self.emotional_patterns = [
            r"هههه+", r"اها+", r"واو+",
            r"اوو+", r"ييي+", r"ششش+",
        ]

    def gaming_hype_score(self, text, audio_score, motion_score):
        score = 0.0
        text = text.strip()

        # =========================================
        # AUDIO HYPE
        # =========================================
        if audio_score > 0.85:
            score += 0.35
        elif audio_score > 0.7:
            score += 0.20

        # =========================================
        # MOTION HYPE
        # =========================================
        if motion_score > 0.8:
            score += 0.30
        elif motion_score > 0.6:
            score += 0.15

        # =========================================
        # EXCLAMATIONS
        # =========================================
        exclamations = text.count("!") + text.count("！")
        if exclamations >= 2:
            score += 0.20
        elif exclamations == 1:
            score += 0.10

        # =========================================
        # QUESTION HYPE
        # =========================================
        questions = text.count("?") + text.count("؟")
        if questions >= 1:
            score += 0.08

        # =========================================
        # LETTER REPETITION (لاااااا، واااو)
        # =========================================
        repeated = re.findall(r"(.)\1{2,}", text)
        if repeated:
            score += 0.20

        # =========================================
        # CAPS / BIG ENERGY
        # =========================================
        letters = [c for c in text if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio > 0.4:
                score += 0.15

        # =========================================
        # SHORT FAST PHRASES
        # =========================================
        words = text.split()
        if 2 <= len(words) <= 6:
            score += 0.12

        # =========================================
        # FAST SPEECH BONUS
        # =========================================
        if len(words) >= 10:
            score += 0.08

        # =========================================
        # EMOTION DETECTION
        # =========================================
        for pattern in self.emotional_patterns:
            if re.search(pattern, text):
                score += 0.15
                break

        return min(score, 1.0)

    def hook_score(self, text):
        score = 0.0

        for word in self.hook_words:
            if word in text:
                score += 0.15

        if "؟" in text or "?" in text:
            score += 0.1

        if "!" in text:
            score += 0.1

        return min(score, 1.0)

    def excitement_score(self, text, audio_score):
        score = 0.0

        if audio_score > 0.7:
            score += 0.4

        exciting = ["لا", "مستحيل", "كارثة", "صدمة", "أخطر"]
        for word in exciting:
            if word in text:
                score += 0.15

        return min(score, 1.0)
