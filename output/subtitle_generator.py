# output/subtitle_generator.py

from core.logger import get_logger

logger = get_logger("SUBTITLES")


class SubtitleGenerator:

    @staticmethod
    def format_srt_time(seconds):
        ms = int((seconds % 1) * 1000)
        s = int(seconds % 60)
        m = int((seconds // 60) % 60)
        h = int(seconds // 3600)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def generate(self, segments, path, offset=0):
        with open(path, "w", encoding="utf-8") as f:
            for idx, seg in enumerate(segments, start=1):
                start_time = max(0, seg["start"] - offset)
                end_time = max(start_time + 0.1, seg["end"] - offset)

                start = self.format_srt_time(start_time)
                end = self.format_srt_time(end_time)

                f.write(f"{idx}\n")
                f.write(f"{start} --> {end}\n")
                f.write(seg["text"] + "\n\n")

        logger.info(f"Subtitles written: {path}")
