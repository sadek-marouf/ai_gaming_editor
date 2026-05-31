# vision/vehicle_filter.py

from core.logger import get_logger

logger = get_logger("VEHICLE_FILTER")


class VehicleFilter:
    """Detect and penalize vehicle/driving scenes using YOLO.

    If vehicles occupy > area_threshold of the screen for
    > duration_threshold consecutive seconds, penalize the segment.
    """

    def __init__(self, game_profile, use_gpu=False):
        self.profile = game_profile
        self.use_gpu = use_gpu
        self._model = None

        vf_config = self.profile.get_vehicle_filter_config()
        self.enabled = vf_config["enabled"]
        self.vehicle_classes = vf_config["classes"]
        self.area_threshold = vf_config["area_threshold"]
        self.duration_threshold = vf_config["duration_threshold"]

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")

            if self.use_gpu:
                self._model.to("cuda")

            logger.info("YOLO model loaded for vehicle detection")
        except Exception as e:
            logger.warning(f"YOLO model load failed: {e}")
            self._model = None

        return self._model

    def _vehicle_area_ratio(self, frame, results):
        """Calculate total vehicle area as ratio of frame area."""
        if results is None or not results[0].boxes:
            return 0.0

        frame_area = frame.shape[0] * frame.shape[1]
        if frame_area <= 0:
            return 0.0

        vehicle_area = 0.0

        for i, cls_id in enumerate(results[0].boxes.cls.cpu().numpy()):
            if int(cls_id) in self.vehicle_classes:
                box = results[0].boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                vehicle_area += (x2 - x1) * (y2 - y1)

        return vehicle_area / frame_area

    def compute_vehicle_mask(self, frames_dict):
        """Compute per-second vehicle presence mask.

        Returns list of floats: 0.0 = no vehicle, 1.0 = vehicle dominant.
        """
        if not self.enabled:
            return []

        model = self._load_model()
        if model is None:
            return []

        sorted_keys = sorted(frames_dict.keys())
        ratios = []

        for sec in sorted_keys:
            frame = frames_dict[sec]
            try:
                device = 0 if self.use_gpu else "cpu"
                results = model.predict(
                    frame, verbose=False, device=device, conf=0.3,
                )
                ratio = self._vehicle_area_ratio(frame, results)
                ratios.append(ratio)
            except Exception as e:
                logger.error(f"Vehicle detection failed at {sec}s: {e}")
                ratios.append(0.0)

        return ratios

    def compute_penalties(self, vehicle_ratios):
        """Convert vehicle ratios to segment penalties.

        Returns per-second penalty multiplier:
        1.0 = no penalty, 0.0 = full penalty.
        """
        if not vehicle_ratios:
            return []

        penalties = [1.0] * len(vehicle_ratios)
        consecutive = 0

        for i, ratio in enumerate(vehicle_ratios):
            if ratio >= self.area_threshold:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= self.duration_threshold:
                # Penalize: set to 0 for this second and retroactively
                start = max(0, i - consecutive + 1)
                for j in range(start, i + 1):
                    penalties[j] = 0.0

        penalized = sum(1 for p in penalties if p == 0.0)
        if penalized > 0:
            logger.info(f"Vehicle filter: {penalized}s penalized")

        return penalties
