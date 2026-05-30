# vision/face_detector.py

import numpy as np

from core.logger import get_logger

logger = get_logger("FACE_DETECTOR")


class FaceDetector:

    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self._face_model = None
        self._person_model = None

    def _load_face_model(self):
        if self._face_model is not None:
            return self._face_model

        try:
            from ultralytics import YOLO
            self._face_model = YOLO("yolov8n-face.pt")

            if self.use_gpu:
                self._face_model.to("cuda")
                try:
                    self._face_model.model.half()
                    logger.info("Face model using FP16")
                except Exception:
                    pass

            logger.info("Face model loaded")

        except Exception as e:
            logger.error(f"Face model load failed: {e}")
            self._face_model = None

        return self._face_model

    def _load_person_model(self):
        if self._person_model is not None:
            return self._person_model

        try:
            from ultralytics import YOLO
            self._person_model = YOLO("yolov8n.pt")

            if self.use_gpu:
                self._person_model.to("cuda")
                try:
                    self._person_model.model.half()
                except Exception:
                    pass

            logger.info("Person model loaded")

        except Exception as e:
            logger.error(f"Person model load failed: {e}")
            self._person_model = None

        return self._person_model

    def face_scores(self, frames_dict):
        model = self._load_face_model()
        if model is None:
            logger.warning("Face model unavailable, returning zeros")
            return [0.0] * len(frames_dict)

        sorted_keys = sorted(frames_dict.keys())
        frames = [frames_dict[k] for k in sorted_keys]

        try:
            import torch
            device = 0 if self.use_gpu else "cpu"

            with torch.amp.autocast(
                device_type="cuda",
                enabled=self.use_gpu,
            ):
                results = model.predict(
                    frames,
                    verbose=False,
                    batch=32,
                    device=device,
                    conf=0.35,
                )

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return [0.0] * len(frames)

        scores = []
        for idx, r in enumerate(results):
            frame = frames[idx]
            frame_area = frame.shape[0] * frame.shape[1]
            max_score = 0.0

            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    face_area = (x2 - x1) * (y2 - y1)
                    ratio = face_area / frame_area
                    score = min(ratio * 12, 1.0)
                    max_score = max(max_score, score)

            scores.append(float(max_score))

        logger.info(f"Face scores computed for {len(scores)} frames")
        return scores

    def detect_subject_position(self, frame):
        model = self._load_person_model()
        if model is None:
            return 0.5

        try:
            results = model.predict(frame, verbose=False, conf=0.4)

            if not results or not results[0].boxes:
                return 0.5

            largest = None
            largest_area = 0

            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest = box

            if largest is None:
                return 0.5

            x1, _, x2, _ = largest
            center_x = (x1 + x2) / 2
            return center_x / frame.shape[1]

        except Exception as e:
            logger.error(f"Subject detection failed: {e}")
            return 0.5
