# core/config.py

import os

class Config:

    # =========================
    # PATHS
    # =========================
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    CACHE_DIR = os.path.join(BASE_DIR, "cache")

    # =========================
    # VIDEO SETTINGS
    # =========================
    FPS_SAMPLE_RATE = 1  # كل كم ثانية نأخذ frame
    TARGET_WIDTH = 720
    TARGET_HEIGHT = 1280

    # =========================
    # QUALITY
    # =========================
    QUALITY_PRESETS = {
        "low": "2000k",
        "medium": "5000k",
        "high": "8000k",
    }

    DEFAULT_QUALITY = "medium"

    # =========================
    # AUDIO
    # =========================
    AUDIO_SAMPLE_RATE = 16000

    # =========================
    # GAMING THRESHOLDS
    # =========================
    MOTION_THRESHOLD = 0.6
    AUDIO_HYPE_THRESHOLD = 0.7
    VISUAL_FLASH_THRESHOLD = 0.5

    # =========================
    # AI SETTINGS
    # =========================
    USE_AI = False