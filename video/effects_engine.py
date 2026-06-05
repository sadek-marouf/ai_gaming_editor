# video/effects_engine.py

from core.logger import get_logger
from core.config import Config

logger = get_logger("EFFECTS")


class EffectsEngine:
    """Builds FFmpeg filter_complex strings for post-kill cinematic effects.

    Effects:
    - Flash: brief contrast/saturation boost at kill moment
    - Swoosh: sub-bass transition sound at clip end

    Note: Slow-motion has been removed to preserve native playback speed
    for competitive gaming clips.
    """

    def __init__(self, game_profile):
        self.config = game_profile.get_effects_config()

    def has_effects(self):
        if not self.config.get("enabled", False):
            return False
        return (
            self.config.get("flash_enabled", False)
            or self.config.get("swoosh_enabled", False)
        )

    def build_filter_complex(
        self,
        crop_scale_vf,
        trigger_offset,
        clip_source_duration,
        subtitle_path=None,
    ):
        """Build FFmpeg filter_complex for a clip with effects.

        Args:
            crop_scale_vf: crop/scale filter string (from auto_framer)
            trigger_offset: seconds from clip start where kill happened
            clip_source_duration: source clip duration in seconds
            subtitle_path: optional subtitle file

        Returns:
            (filter_complex, video_label, audio_label, output_duration)
        """
        T = trigger_offset
        flash_enabled = self.config.get("flash_enabled", False)
        flash_dur = self.config.get("flash_duration", 0.2)
        flash_contrast = self.config.get("flash_contrast", 1.5)
        flash_sat = self.config.get("flash_saturation", 1.8)
        swoosh_enabled = self.config.get("swoosh_enabled", False)
        swoosh_dur = self.config.get("swoosh_duration", 0.5)
        swoosh_vol = self.config.get("swoosh_volume", 0.2)

        # Build subtitle filter segment
        sub_vf = ""
        if subtitle_path:
            sub_fixed = subtitle_path.replace("\\", "/").replace("'", "'\\''")
            sub_vf = (
                f",subtitles='{sub_fixed}':"
                "force_style='"
                f"FontName={Config.SUBTITLE_FONT},"
                f"Fontsize={Config.SUBTITLE_FONTSIZE},"
                "PrimaryColour=&Hffffff&,"
                "OutlineColour=&H000000&,"
                "BackColour=&H66000000&,"
                "BorderStyle=3,"
                "Outline=2,"
                "Shadow=1,"
                f"MarginV={Config.SUBTITLE_MARGIN_V},"
                "Alignment=2'"
            )

        base_vf = crop_scale_vf + sub_vf
        lines = []
        output_dur = clip_source_duration

        # ===========================
        # VIDEO CHAIN
        # ===========================
        if flash_enabled:
            lines.append(
                f"[0:v]{base_vf},"
                f"eq=contrast={flash_contrast}"
                f":saturation={flash_sat}"
                f":enable='between(t\\,{T:.3f}\\,{T + flash_dur:.3f})'[vout]"
            )
        else:
            lines.append(f"[0:v]{base_vf}[vout]")

        # Audio: copy directly (no speed change)
        lines.append("[0:a]acopy[amain]")

        # ===========================
        # SWOOSH AT END
        # ===========================
        if swoosh_enabled:
            delay_ms = max(0, int((output_dur - swoosh_dur) * 1000))
            lines.append(
                f"anoisesrc=d={swoosh_dur}:c=pink:s=44100,"
                f"lowpass=f=150,highpass=f=30,"
                f"afade=t=in:d=0.08,"
                f"afade=t=out:st={swoosh_dur * 0.5:.2f}:d={swoosh_dur * 0.5:.2f},"
                f"volume={swoosh_vol},"
                f"adelay={delay_ms}|{delay_ms}[swoosh]"
            )
            lines.append(
                "[amain][swoosh]amix=inputs=2"
                ":duration=first:normalize=0[aout]"
            )
            audio_label = "[aout]"
        else:
            audio_label = "[amain]"

        fc = ";\n".join(lines)
        return fc, "[vout]", audio_label, output_dur
