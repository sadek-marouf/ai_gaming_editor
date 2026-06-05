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
    FPS_SAMPLE_RATE = 1
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

    DEFAULT_QUALITY = "high"

    # Premium NVENC encoding (GPU)
    NVENC_CODEC = "hevc_nvenc"
    NVENC_PRESET = "p7"
    NVENC_QP = 18

    # CPU fallback encoding
    CPU_CODEC = "libx264"
    CPU_CRF = 18
    CPU_PRESET = "slow"

    # Preserve native resolution / fps from source
    PRESERVE_NATIVE_RESOLUTION = True

    # =========================
    # AUDIO
    # =========================
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CODEC = "pcm_s16le"
    AUDIO_CHANNELS = 1

    # =========================
    # GAMING THRESHOLDS
    # =========================
    MOTION_THRESHOLD = 0.6
    AUDIO_HYPE_THRESHOLD = 0.7
    VISUAL_FLASH_THRESHOLD = 0.5
    COMBINED_PEAK_THRESHOLD = 0.75

    # =========================
    # SILENCE DETECTION
    # =========================
    SILENCE_RMS_LOW = 0.005
    SILENCE_RMS_MED = 0.01
    SILENCE_PENALTY_HEAVY = 0.2
    SILENCE_PENALTY_LIGHT = 0.5

    # =========================
    # SCENE DETECTION
    # =========================
    SCENE_THRESHOLD = 27.0

    # =========================
    # SUBTITLE
    # =========================
    SUBTITLE_WORDS_PER_CHUNK = 3
    SUBTITLE_FONT = "DejaVu Sans"
    SUBTITLE_FONTSIZE = 20
    SUBTITLE_MARGIN_V = 60

    # =========================
    # AI SETTINGS (Groq - legacy fallback)
    # =========================
    USE_AI = False
    AI_MODEL = "llama-3.1-8b-instant"
    AI_BASE_URL = "https://api.groq.com/openai/v1"

    # =========================
    # GEMINI AI DIRECTOR
    # =========================
    USE_GEMINI = True
    GEMINI_MODEL = "gemini-1.5-flash"

    # =========================
    # GAMING PROFILE
    # =========================
    GAMING_PROFILE = {
        "audio_weight": 0.30,
        "motion_weight": 0.35,
        "visual_weight": 0.08,
        "face_weight": 0.02,
        "hook_weight": 0.10,
        "ai_weight": 0.10,
        "hype_weight": 0.25,
        "min_duration": 3.5,
        "max_duration": 8,
        "fast_cuts": True,
        "aggressive_subtitles": True,
        "prefer_peaks": True,
    }

    # =========================
    # PARALLEL PROCESSING
    # =========================
    PARALLEL_WORKERS = 4
    CLIP_ENCODING_WORKERS = 2

    # =========================
    # REEL
    # =========================
    MAX_REEL_CLIPS = 4
    BREATHING_ROOM = 0.6
    MIN_CLIP_DURATION = 3.0
