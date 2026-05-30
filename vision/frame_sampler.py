from core.logger import get_logger

logger = get_logger("FRAME_SAMPLER")


class FrameSampler:

    def __init__(self, step=30):
        self.step = step

    def sample(self, frame_cache):

        sampled = {}

        for frame_id, frame in frame_cache.items():

            if frame_id % self.step == 0:

                sampled[frame_id] = frame

        logger.info(
            f"Sampled {len(sampled)} frames "
            f"from {len(frame_cache)}"
        )

        return sampled