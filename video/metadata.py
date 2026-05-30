# video/metadata.py

from dataclasses import dataclass


@dataclass
class VideoMetadata:

    fps: float
    frame_count: int
    width: int
    height: int
    duration: float