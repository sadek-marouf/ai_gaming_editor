# games/pubg.py

import os

from games.base_profile import BaseGameProfile


class PUBGProfile(BaseGameProfile):
    """PUBG / PUBG Mobile game profile.

    Optimized for:
    - Kill feed detection (bottom-left / right side notifications)
    - Hitmarker and kill icon detection
    - Reduced motion weight (driving creates false positives)
    - Increased audio weight (gunfire, explosions, player reactions)
    - Vehicle scene filtering
    - Smart framing for vertical reels
    """

    NAME = "pubg"
    DISPLAY_NAME = "PUBG / PUBG Mobile"

    # =========================================
    # SCORING WEIGHTS
    # Reduced motion (driving = false positive)
    # Increased audio (gunfire, explosions)
    # High kill_feed_weight (actual combat)
    # =========================================
    AUDIO_WEIGHT = 0.40
    MOTION_WEIGHT = 0.10
    VISUAL_WEIGHT = 0.08
    FACE_WEIGHT = 0.02
    HOOK_WEIGHT = 0.08
    AI_WEIGHT = 0.07
    HYPE_WEIGHT = 0.20
    KILL_FEED_WEIGHT = 0.45

    # =========================================
    # CLIP SETTINGS
    # =========================================
    MIN_DURATION = 3.0
    MAX_DURATION = 10
    MAX_CLIPS = 5
    FAST_CUTS = True
    PREFER_PEAKS = True

    # =========================================
    # KILL FEED OCR
    # PUBG shows kill notifications on bottom-left
    # or right side of screen
    # Region: bottom 20% of screen, full width
    # Also check right side for kill log
    # =========================================
    KILL_FEED_KEYWORDS = [
        ("knocked out", 1.5),
        ("knocked", 1.3),
        ("finally killed", 2.0),
        ("killed", 1.5),
        ("headshot", 2.0),
        ("eliminated", 1.8),
        ("winner winner", 3.0),
        ("chicken dinner", 3.0),
        ("you killed", 2.0),
        ("got killed", 1.0),
    ]

    # Bottom-left region for kill feed notifications
    KILL_FEED_REGION = (0.0, 0.75, 0.55, 1.0)

    # Right side kill log
    KILL_LOG_REGION = (0.60, 0.05, 1.0, 0.45)

    KILL_FEED_INSTANT_SCORE = 1.0

    # =========================================
    # HITMARKER / KILL ICON
    # Center screen for crosshair hit feedback
    # =========================================
    HITMARKER_REGION = (0.40, 0.35, 0.60, 0.65)
    HITMARKER_THRESHOLD = 0.65

    # Template paths (relative to assets/)
    HITMARKER_TEMPLATES = [
        "assets/templates/pubg/hitmarker.png",
        "assets/templates/pubg/kill_icon.png",
    ]

    # =========================================
    # VEHICLE FILTERING
    # YOLO COCO classes: 2=car, 5=bus, 7=truck
    # If vehicle > 70% of screen for > 5s → penalize
    # =========================================
    VEHICLE_FILTER_ENABLED = True
    VEHICLE_CLASSES = [2, 5, 7]
    VEHICLE_AREA_THRESHOLD = 0.70
    VEHICLE_DURATION_THRESHOLD = 5.0

    # =========================================
    # AUTO-FRAMING
    # Smart crop focusing on crosshair area
    # =========================================
    FRAMING_MODE = "center_crop"
    FACECAM_POSITION = None
    FACECAM_SPLIT_RATIO = 0.30

    # =========================================
    # SMART TIMING
    # 3s build-up before kill, 2.5s aftermath
    # =========================================
    PRE_TRIGGER_PAD = 3.0
    POST_TRIGGER_PAD = 2.5
    TRIGGER_CLIP_MIN = 6.0
    TRIGGER_CLIP_MAX = 8.0

    # =========================================
    # POST-KILL EFFECTS
    # Flash, slow-mo, swoosh for cinematic reels
    # =========================================
    EFFECTS_ENABLED = True

    FLASH_ENABLED = True
    FLASH_DURATION = 0.2
    FLASH_CONTRAST = 1.5
    FLASH_SATURATION = 1.8

    SLOWMO_ENABLED = True
    SLOWMO_SPEED = 0.5
    SLOWMO_DURATION = 1.5

    SWOOSH_ENABLED = True
    SWOOSH_DURATION = 0.5
    SWOOSH_VOLUME = 0.2

    # =========================================
    # PEAK DETECTION
    # More audio, less motion for PUBG
    # =========================================
    AUDIO_PEAK_WEIGHT = 0.70
    MOTION_PEAK_WEIGHT = 0.30
    COMBINED_PEAK_THRESHOLD = 0.65

    def get_kill_feed_config(self):
        config = super().get_kill_feed_config()
        config["kill_log_region"] = self.KILL_LOG_REGION
        return config
