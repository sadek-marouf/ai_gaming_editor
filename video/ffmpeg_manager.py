# video/ffmpeg_manager.py

import os

import cv2

from core.logger import get_logger
from core.config import Config
from core.utils import run_cmd, get_best_codec

logger = get_logger("FFMPEG")


class FFmpegManager:

    def __init__(
        self, video_path, output_dir,
        quality=None, use_gpu=False, auto_framer=None,
    ):
        self.video_path = os.path.abspath(video_path)
        self.output_dir = os.path.abspath(output_dir)
        self.quality = quality or Config.DEFAULT_QUALITY
        self.use_gpu = use_gpu
        self.codec = get_best_codec(use_gpu)
        self.auto_framer = auto_framer
        self._input_size = None

        os.makedirs(self.output_dir, exist_ok=True)

    def _get_input_size(self):
        if self._input_size is not None:
            return self._input_size

        cap = cv2.VideoCapture(self.video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self._input_size = (w, h)
        return self._input_size

    def extract_audio(self, audio_path):
        cmd = [
            "ffmpeg", "-y",
            "-i", self.video_path,
            "-vn",
            "-acodec", Config.AUDIO_CODEC,
            "-ar", str(Config.AUDIO_SAMPLE_RATE),
            "-ac", str(Config.AUDIO_CHANNELS),
            audio_path,
        ]
        run_cmd(cmd, check=True)
        logger.info(f"Audio extracted to {audio_path}")

    def cut_clip(
        self, start, duration, output_path,
        subtitle_path=None, trigger_offset=None, effects_engine=None,
    ):
        # Route to effects pipeline if trigger present
        if (
            trigger_offset is not None
            and effects_engine is not None
            and effects_engine.has_effects()
        ):
            return self._cut_clip_with_effects(
                start, duration, output_path,
                subtitle_path, trigger_offset, effects_engine,
            )

        bitrate = Config.QUALITY_PRESETS.get(self.quality, "5000k")

        if self.auto_framer:
            w, h = self._get_input_size()
            vf = self.auto_framer.get_ffmpeg_vf(w, h, subtitle_path)
        else:
            vf = (
                f"scale={Config.TARGET_WIDTH}:{Config.TARGET_HEIGHT}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={Config.TARGET_WIDTH}:{Config.TARGET_HEIGHT}:"
                "(ow-iw)/2:(oh-ih)/2"
            )

            if subtitle_path:
                sub_fixed = subtitle_path.replace("\\", "/")
                vf += (
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

        cmd = ["ffmpeg", "-y"]

        if self.use_gpu:
            cmd += ["-hwaccel", "cuda"]

        cmd += [
            "-ss", str(start),
            "-t", str(duration),
            "-i", self.video_path,
            "-vf", vf,
            "-c:v", self.codec,
            "-preset", "fast",
            "-b:v", bitrate,
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        run_cmd(cmd, check=True)
        logger.info(f"Clip generated: {output_path}")

    def _cut_clip_with_effects(
        self, start, duration, output_path,
        subtitle_path, trigger_offset, effects_engine,
    ):
        """Cut clip with post-kill effects via filter_complex."""
        bitrate = Config.QUALITY_PRESETS.get(self.quality, "5000k")

        # Get base crop/scale VF without subtitles
        if self.auto_framer:
            w, h = self._get_input_size()
            base_vf = self.auto_framer.get_ffmpeg_vf(w, h, None)
        else:
            base_vf = (
                f"scale={Config.TARGET_WIDTH}:{Config.TARGET_HEIGHT}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={Config.TARGET_WIDTH}:{Config.TARGET_HEIGHT}:"
                "(ow-iw)/2:(oh-ih)/2"
            )

        fc, vlabel, alabel, out_dur = effects_engine.build_filter_complex(
            base_vf, trigger_offset, duration, subtitle_path,
        )

        cmd = ["ffmpeg", "-y"]

        if self.use_gpu:
            cmd += ["-hwaccel", "cuda"]

        cmd += [
            "-ss", str(start),
            "-t", str(duration),
            "-i", self.video_path,
            "-filter_complex", fc,
            "-map", vlabel,
            "-map", alabel,
            "-c:v", self.codec,
            "-preset", "fast",
            "-b:v", bitrate,
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        run_cmd(cmd, check=True)
        logger.info(
            f"Effects clip generated: {output_path} "
            f"(trigger@{trigger_offset:.1f}s, out={out_dur:.1f}s)"
        )

    def concat_clips(self, clip_paths, output_path):
        concat_file = os.path.join(self.output_dir, "concat.txt")

        with open(concat_file, "w") as f:
            for clip in clip_paths:
                clip_fixed = clip.replace("\\", "/")
                f.write(f"file '{clip_fixed}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path,
        ]

        run_cmd(cmd, check=True)
        logger.info(f"Final reel: {output_path}")

        try:
            os.remove(concat_file)
        except OSError:
            pass

        return output_path
