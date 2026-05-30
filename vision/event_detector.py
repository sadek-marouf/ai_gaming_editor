# vision/event_detector.py

import numpy as np

from core.logger import get_logger

logger = get_logger("EVENT_DETECTOR")


class MotionEventDetector:

    def __init__(
        self,
        window_size=5,
        spike_threshold=1.5,
        min_motion=0.08
    ):

        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.min_motion = min_motion

    def detect_spikes(self, motion_scores):

        """
        Detect sudden motion spikes
        inside gameplay footage.
        """

        if not motion_scores:
            return []

        events = []

        for i in range(
            self.window_size,
            len(motion_scores) - self.window_size
        ):

            current = motion_scores[i]

            window = motion_scores[
                i - self.window_size:
                i + self.window_size
            ]

            mean = np.mean(window)
            std = np.std(window)

            dynamic_threshold = (
                mean +
                (std * self.spike_threshold)
            )

            # =====================================
            # SPIKE DETECTION
            # =====================================

            if (
                current > dynamic_threshold and
                current > self.min_motion
            ):

                events.append({

                    "frame": i,

                    "type": "motion_spike",

                    "strength": round(
                        float(current),
                        4
                    ),

                    "mean": round(
                        float(mean),
                        4
                    ),

                    "threshold": round(
                        float(dynamic_threshold),
                        4
                    )
                })

        logger.info(
            f"Detected {len(events)} motion events"
        )

        return events