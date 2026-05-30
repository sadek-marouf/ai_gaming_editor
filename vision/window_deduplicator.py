# vision/window_deduplicator.py

from core.logger import get_logger

logger = get_logger("WINDOW_DEDUP")


class WindowDeduplicator:

    def __init__(self, overlap_threshold=0.5):

        """
        overlap_threshold:
        نسبة التداخل المسموح بها
        قبل اعتبار المقطعين مكررين
        """

        self.overlap_threshold = overlap_threshold

    def deduplicate(self, windows):

        if not windows:
            return []

        # =====================================
        # SORT BY STRENGTH
        # =====================================

        windows = sorted(
            windows,
            key=lambda x: x["peak_strength"],
            reverse=True
        )

        final_windows = []

        for candidate in windows:

            should_keep = True

            for existing in final_windows:

                overlap = self.calculate_overlap(
                    candidate,
                    existing
                )

                # =====================================
                # TOO MUCH OVERLAP
                # =====================================

                if overlap >= self.overlap_threshold:

                    should_keep = False
                    break

            if should_keep:
                final_windows.append(candidate)

        # =====================================
        # SORT TIMELINE
        # =====================================

        final_windows = sorted(
            final_windows,
            key=lambda x: x["start_frame"]
        )

        logger.info(
            f"Reduced windows: "
            f"{len(windows)} → {len(final_windows)}"
        )

        return final_windows

    def calculate_overlap(
        self,
        a,
        b
    ):

        start = max(
            a["start_frame"],
            b["start_frame"]
        )

        end = min(
            a["end_frame"],
            b["end_frame"]
        )

        # no overlap
        if end <= start:
            return 0.0

        intersection = end - start

        a_duration = (
            a["end_frame"] -
            a["start_frame"]
        )

        b_duration = (
            b["end_frame"] -
            b["start_frame"]
        )

        smaller = min(
            a_duration,
            b_duration
        )

        if smaller <= 0:
            if a_duration == 0 and b_duration == 0:
                return 1.0 if a["start_frame"] == b["start_frame"] else 0.0
            return 0.0

        return intersection / smaller