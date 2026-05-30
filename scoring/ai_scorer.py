# scoring/ai_scorer.py

import os
import re
import json
import hashlib

from core.logger import get_logger
from core.config import Config

logger = get_logger("AI_SCORER")

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        _client = OpenAI(
            api_key=api_key,
            base_url=Config.AI_BASE_URL,
        )
        logger.info("Groq AI client initialized")
        return _client
    except Exception as e:
        logger.error(f"Groq init error: {e}")
        return None


def default_ai_score():
    return {
        "hook": 0.5,
        "emotion": 0.5,
        "surprise": 0.5,
        "urgency": 0.5,
        "curiosity": 0.5,
        "cta": 0.5,
        "controversy": 0.5,
        "shareability": 0.5,
        "final": 0.5,
    }


class AIScorer:

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.enabled = Config.USE_AI or os.environ.get("GROQ_API_KEY")

    def score(self, text):
        if not self.enabled:
            return default_ai_score()

        client = _get_client()
        if client is None:
            return default_ai_score()

        if self.cache_dir:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_file = os.path.join(
                self.cache_dir, f"ai_cache_{text_hash}.json"
            )
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    return json.load(f)
        else:
            cache_file = None

        try:
            prompt = (
                "Analyze this text for SHORT-FORM VIRAL potential "
                "(TikTok/Reels style).\n\n"
                "Return ONLY a JSON object with scores 0-1:\n"
                "- hook, emotion, surprise, urgency, controversy, "
                "cta, curiosity, shareability\n\n"
                f"TEXT:\n{text[:800]}\n\n"
                "Return ONLY valid JSON, no markdown."
            )

            response = client.chat.completions.create(
                model=Config.AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=250,
                top_p=0.9,
            )

            raw = response.choices[0].message.content

            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = default_ai_score()

            final = (
                data.get("hook", 0.5) * 0.30
                + data.get("emotion", 0.5) * 0.20
                + data.get("surprise", 0.5) * 0.15
                + data.get("urgency", 0.5) * 0.15
                + data.get("curiosity", 0.5) * 0.10
                + data.get("cta", 0.5) * 0.10
            )

            data["final"] = min(final, 1.0)

            if cache_file:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(data, f)

            return data

        except Exception as e:
            logger.error(f"AI score error: {e}")
            return default_ai_score()
