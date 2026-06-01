# games/base_profile.py

from core.logger import get_logger

logger = get_logger("GAME_PROFILE")


class BaseGameProfile:
    """Base class for game-specific profiles.

    Subclass this to add a new game. Override any attribute or method
    to customize detection and scoring for that game.

    To register a new game:
    1. Create games/<game_name>.py with a class that inherits BaseGameProfile
    2. Add the class to GAME_REGISTRY in games/registry.py
    """

    NAME = "generic"
    DISPLAY_NAME = "Generic Gaming"

    # =========================================
    # SCORING WEIGHTS
    # =========================================
    AUDIO_WEIGHT = 0.30
    MOTION_WEIGHT = 0.25
    VISUAL_WEIGHT = 0.10
    FACE_WEIGHT = 0.02
    HOOK_WEIGHT = 0.10
    AI_WEIGHT = 0.10
    HYPE_WEIGHT = 0.25
    KILL_FEED_WEIGHT = 0.0

    # =========================================
    # CLIP SETTINGS
    # =========================================
    MIN_DURATION = 3.5
    MAX_DURATION = 8
    MAX_CLIPS = 4
    FAST_CUTS = True
    PREFER_PEAKS = True

    # =========================================
    # KILL FEED OCR
    # =========================================
    KILL_FEED_KEYWORDS = []
    KILL_FEED_REGION = None  # (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
    KILL_FEED_INSTANT_SCORE = 1.0

    # =========================================
    # HITMARKER TEMPLATES
    # =========================================
    HITMARKER_TEMPLATES = []  # list of template image paths
    HITMARKER_REGION = None  # center region ratio
    HITMARKER_THRESHOLD = 0.7

    # =========================================
    # VEHICLE FILTERING
    # =========================================
    VEHICLE_FILTER_ENABLED = False
    VEHICLE_CLASSES = []  # YOLO class IDs
    VEHICLE_AREA_THRESHOLD = 0.70
    VEHICLE_DURATION_THRESHOLD = 5.0

    # =========================================
    # AUTO-FRAMING
    # =========================================
    FRAMING_MODE = "center_crop"  # center_crop | smart_crop | split_facecam
    FACECAM_POSITION = None  # (x1_ratio, y1_ratio, x2_ratio, y2_ratio) if known
    FACECAM_SPLIT_RATIO = 0.30  # top 30% for facecam

    # =========================================
    # SMART TIMING (trigger-based clipping)
    # =========================================
    PRE_TRIGGER_PAD = 3.0       # seconds before trigger event
    POST_TRIGGER_PAD = 2.5      # seconds after trigger (source time)
    TRIGGER_CLIP_MIN = 6.0      # min output clip duration
    TRIGGER_CLIP_MAX = 8.0      # max output clip duration

    # =========================================
    # POST-KILL EFFECTS
    # =========================================
    EFFECTS_ENABLED = False     # master switch for all effects

    FLASH_ENABLED = False
    FLASH_DURATION = 0.2        # seconds of contrast/saturation boost
    FLASH_CONTRAST = 1.5
    FLASH_SATURATION = 1.8

    SLOWMO_ENABLED = False
    SLOWMO_SPEED = 0.5          # playback speed (0.5 = half speed)
    SLOWMO_DURATION = 1.5       # source seconds to slow down

    SWOOSH_ENABLED = False
    SWOOSH_DURATION = 0.5       # seconds
    SWOOSH_VOLUME = 0.2

    # =========================================
    # PEAK DETECTION
    # =========================================
    AUDIO_PEAK_WEIGHT = 0.55
    MOTION_PEAK_WEIGHT = 0.45
    COMBINED_PEAK_THRESHOLD = 0.75

    def get_scoring_weights(self):
        return {
            "audio_weight": self.AUDIO_WEIGHT,
            "motion_weight": self.MOTION_WEIGHT,
            "visual_weight": self.VISUAL_WEIGHT,
            "face_weight": self.FACE_WEIGHT,
            "hook_weight": self.HOOK_WEIGHT,
            "ai_weight": self.AI_WEIGHT,
            "hype_weight": self.HYPE_WEIGHT,
            "kill_feed_weight": self.KILL_FEED_WEIGHT,
            "min_duration": self.MIN_DURATION,
            "max_duration": self.MAX_DURATION,
            "fast_cuts": self.FAST_CUTS,
            "prefer_peaks": self.PREFER_PEAKS,
        }

    def get_kill_feed_config(self):
        return {
            "keywords": self.KILL_FEED_KEYWORDS,
            "region": self.KILL_FEED_REGION,
            "instant_score": self.KILL_FEED_INSTANT_SCORE,
        }

    def get_vehicle_filter_config(self):
        return {
            "enabled": self.VEHICLE_FILTER_ENABLED,
            "classes": self.VEHICLE_CLASSES,
            "area_threshold": self.VEHICLE_AREA_THRESHOLD,
            "duration_threshold": self.VEHICLE_DURATION_THRESHOLD,
        }

    def get_framing_config(self):
        return {
            "mode": self.FRAMING_MODE,
            "facecam_position": self.FACECAM_POSITION,
            "facecam_split_ratio": self.FACECAM_SPLIT_RATIO,
        }

    def get_effects_config(self):
        return {
            "enabled": self.EFFECTS_ENABLED,
            "flash_enabled": self.FLASH_ENABLED,
            "flash_duration": self.FLASH_DURATION,
            "flash_contrast": self.FLASH_CONTRAST,
            "flash_saturation": self.FLASH_SATURATION,
            "slowmo_enabled": self.SLOWMO_ENABLED,
            "slowmo_speed": self.SLOWMO_SPEED,
            "slowmo_duration": self.SLOWMO_DURATION,
            "swoosh_enabled": self.SWOOSH_ENABLED,
            "swoosh_duration": self.SWOOSH_DURATION,
            "swoosh_volume": self.SWOOSH_VOLUME,
            "pre_trigger_pad": self.PRE_TRIGGER_PAD,
            "post_trigger_pad": self.POST_TRIGGER_PAD,
            "trigger_clip_min": self.TRIGGER_CLIP_MIN,
            "trigger_clip_max": self.TRIGGER_CLIP_MAX,
        }
