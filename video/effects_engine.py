# video/effects_engine.py

from core.logger import get_logger
from core.config import Config

logger = get_logger("EFFECTS")


class EffectsEngine:
    """Builds FFmpeg filter_complex strings for post-kill cinematic effects.

    Effects:
    - Flash: brief contrast/saturation boost at kill moment
    - Slow-mo: 0.5x time-ramp after kill with pitch-corrected audio
    - Swoosh: sub-bass transition sound at clip end
    """

    def __init__(self, game_profile):
        self.config = game_profile.get_effects_config()

    def has_effects(self):
        if not self.config.get("enabled", False):
            return False
        return (
            self.config.get("flash_enabled", False)
            or self.config.get("slowmo_enabled", False)
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
        slowmo_enabled = self.config.get("slowmo_enabled", False)
        slowmo_speed = self.config.get("slowmo_speed", 0.5)
        slowmo_dur = self.config.get("slowmo_duration", 1.5)
        swoosh_enabled = self.config.get("swoosh_enabled", False)
        swoosh_dur = self.config.get("swoosh_duration", 0.5)
        swoosh_vol = self.config.get("swoosh_volume", 0.2)

        # Clamp slowmo to available clip time
        if T + slowmo_dur > clip_source_duration:
            slowmo_dur = max(0.0, clip_source_duration - T)

        pts_factor = 1.0 / slowmo_speed  # 2.0 for 0.5x

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

        # ===========================
        # VIDEO + AUDIO CHAINS
        # ===========================
        if slowmo_enabled and slowmo_dur > 0.05:
            output_dur = self._build_slowmo_chain(
                lines, base_vf, T, slowmo_dur, pts_factor,
                flash_enabled, flash_dur, flash_contrast, flash_sat,
                clip_source_duration,
            )
        elif flash_enabled:
            output_dur = self._build_flash_only_chain(
                lines, base_vf, T, flash_dur,
                flash_contrast, flash_sat, clip_source_duration,
            )
        else:
            lines.append(f"[0:v]{base_vf}[vout]")
            lines.append("[0:a]acopy[amain]")
            output_dur = clip_source_duration

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

    def _build_slowmo_chain(
        self, lines, base_vf, T, slowmo_dur, pts_factor,
        flash_enabled, flash_dur, flash_contrast, flash_sat,
        clip_dur,
    ):
        """Build split/trim/concat chain for slow-motion effect."""
        post_start = T + slowmo_dur
        has_pre = T > 0.05
        has_post = (clip_dur - post_start) > 0.05

        # Determine split count and section names
        sections_v = []
        sections_a = []

        if has_pre and has_post:
            split_n = 3
        elif has_pre or has_post:
            split_n = 2
        else:
            split_n = 1

        # VIDEO: crop/scale then split
        split_labels = "".join(f"[v{i}]" for i in range(split_n))
        lines.append(f"[0:v]{base_vf},split={split_n}{split_labels}")

        idx = 0
        # Pre-trigger section (normal speed)
        if has_pre:
            lines.append(
                f"[v{idx}]trim=0:{T:.3f},"
                f"setpts=PTS-STARTPTS[pre]"
            )
            sections_v.append("[pre]")
            idx += 1

        # Slow-mo section with optional flash
        slowmo_vf = (
            f"[v{idx}]trim={T:.3f}:{post_start:.3f},"
            f"setpts={pts_factor}*(PTS-STARTPTS)"
        )
        if flash_enabled:
            slowmo_vf += (
                f",eq=contrast={flash_contrast}"
                f":saturation={flash_sat}"
                f":enable='lt(t\\,{flash_dur})'"
            )
        slowmo_vf += "[slowmo]"
        lines.append(slowmo_vf)
        sections_v.append("[slowmo]")
        idx += 1

        # Post-slowmo section (normal speed)
        if has_post:
            lines.append(
                f"[v{idx}]trim={post_start:.3f},"
                f"setpts=PTS-STARTPTS[post]"
            )
            sections_v.append("[post]")

        n_concat = len(sections_v)
        v_labels = "".join(sections_v)
        lines.append(f"{v_labels}concat=n={n_concat}:v=1:a=0[vout]")

        # AUDIO: split/trim/concat with atempo for slowmo
        a_split_labels = "".join(f"[a{i}]" for i in range(split_n))
        lines.append(f"[0:a]asplit={split_n}{a_split_labels}")

        idx = 0
        if has_pre:
            lines.append(
                f"[a{idx}]atrim=0:{T:.3f},"
                f"asetpts=PTS-STARTPTS[apre]"
            )
            sections_a.append("[apre]")
            idx += 1

        lines.append(
            f"[a{idx}]atrim={T:.3f}:{post_start:.3f},"
            f"asetpts=PTS-STARTPTS,"
            f"atempo={slowmo_speed_val(pts_factor)}[aslowmo]"
        )
        sections_a.append("[aslowmo]")
        idx += 1

        if has_post:
            lines.append(
                f"[a{idx}]atrim={post_start:.3f},"
                f"asetpts=PTS-STARTPTS[apost]"
            )
            sections_a.append("[apost]")

        a_labels = "".join(sections_a)
        lines.append(f"{a_labels}concat=n={n_concat}:v=0:a=1[amain]")

        # Calculate output duration
        pre_dur = T if has_pre else 0
        slowmo_out = slowmo_dur * pts_factor
        post_dur = (clip_dur - post_start) if has_post else 0
        output_dur = pre_dur + slowmo_out + post_dur

        return output_dur

    def _build_flash_only_chain(
        self, lines, base_vf, T, flash_dur,
        flash_contrast, flash_sat, clip_dur,
    ):
        """Flash effect without slow-motion."""
        lines.append(
            f"[0:v]{base_vf},"
            f"eq=contrast={flash_contrast}"
            f":saturation={flash_sat}"
            f":enable='between(t\\,{T:.3f}\\,{T + flash_dur:.3f})'[vout]"
        )
        lines.append("[0:a]acopy[amain]")
        return clip_dur


def slowmo_speed_val(pts_factor):
    """Convert PTS factor to atempo value.

    atempo accepts 0.5-100.0. For 0.5x playback (pts_factor=2.0),
    atempo=0.5. For values below 0.5, chain multiple atempo filters.
    """
    speed = 1.0 / pts_factor
    if speed >= 0.5:
        return f"{speed:.4f}"
    # Chain two atempo filters for very slow speeds
    half = speed ** 0.5
    return f"{half:.4f},atempo={half:.4f}"
