# video/ffmpeg_manager.py

import os

from core.logger import get_logger
from core.config import Config
from core.utils import run_cmd, get_best_codec

logger = get_logger("FFMPEG")


class FFmpegManager:

    def __init__(self, video_path, output_dir, quality=None, use_gpu=False):
        self.video_path = os.path.abspath(video_path)
        self.output_dir = os.path.abspath(output_dir)
        self.quality = quality or Config.DEFAULT_QUALITY
        self.use_gpu = use_gpu
        self.codec = get_best_codec(use_gpu)

        os.makedirs(self.output_dir, exist_ok=True)

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

    def cut_clip(self, start, duration, output_path, subtitle_path=None):
        bitrate = Config.QUALITY_PRESETS.get(self.quality, "5000k")

        vf = (
            f"scale={Config.TARGET_WIDTH}:{Config.TARGET_HEIGHT}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={Config.TARGET_WIDTH}:{Config.TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
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
