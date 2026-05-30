# vision/moment_expander.py

from core.logger import get_logger

logger = get_logger("MOMENT_EXPANDER")


class MomentExpander:

    def __init__(
        self,
        pre_frames=60,
        post_frames=90,
        min_duration=120
    ):

        """
        pre_frames:
        frames before event

        post_frames:
        frames after event

        min_duration:
        minimum final duration
        """

        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.min_duration = min_duration

    def expand(
        self,
        clusters,
        total_frames
    ):

        expanded = []

        for cluster in clusters:

            start = max(
                0,
                cluster["start_frame"] -
                self.pre_frames
            )

            end = min(
                total_frames - 1,
                cluster["end_frame"] +
                self.post_frames
            )

            duration = end - start

            # =====================================
            # FORCE MIN DURATION
            # =====================================

            if duration < self.min_duration:

                extra = (
                    self.min_duration -
                    duration
                )

                end = min(
                    total_frames - 1,
                    end + extra
                )

            expanded.append({

                "start_frame": start,

                "end_frame": end,

                "duration_frames": (
                    end - start
                ),

                "peak_strength":
                cluster["peak_strength"],

                "avg_strength":
                cluster["avg_strength"],

                "events_count":
                cluster["events_count"],

                "type":
                "highlight_window"
            })

        logger.info(
            f"Expanded {len(expanded)} highlight windows"
        )

        return expanded