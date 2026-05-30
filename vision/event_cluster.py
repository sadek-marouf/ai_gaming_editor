# vision/event_cluster.py

from core.logger import get_logger

logger = get_logger("EVENT_CLUSTER")


class EventClusterer:

    def __init__(self, max_gap=20):

        """
        max_gap:
        maximum frame distance
        between events to merge them
        """

        self.max_gap = max_gap

    def cluster_events(self, events):

        if not events:
            return []

        # =====================================
        # SORT EVENTS
        # =====================================

        events = sorted(
            events,
            key=lambda x: x["frame"]
        )

        clusters = []

        current_cluster = [
            events[0]
        ]

        # =====================================
        # GROUP CLOSE EVENTS
        # =====================================

        for event in events[1:]:

            prev = current_cluster[-1]

            gap = (
                event["frame"] -
                prev["frame"]
            )

            # close enough → merge
            if gap <= self.max_gap:

                current_cluster.append(event)

            else:

                clusters.append(
                    self.build_cluster(
                        current_cluster
                    )
                )

                current_cluster = [event]

        # last cluster
        if current_cluster:

            clusters.append(
                self.build_cluster(
                    current_cluster
                )
            )

        logger.info(
            f"Generated {len(clusters)} gameplay moments"
        )

        return clusters

    def build_cluster(self, cluster_events):

        start = cluster_events[0]["frame"]
        end = cluster_events[-1]["frame"]

        strengths = [
            e["strength"]
            for e in cluster_events
        ]

        peak_strength = max(strengths)

        avg_strength = (
            sum(strengths) /
            len(strengths)
        )

        return {

            "start_frame": start,

            "end_frame": end,

            "duration_frames": (
                end - start
            ),

            "events_count": len(
                cluster_events
            ),

            "peak_strength": round(
                peak_strength,
                4
            ),

            "avg_strength": round(
                avg_strength,
                4
            ),

            "type": "gameplay_moment"
        }