import os
import sys
import cv2
import gc
import re
import json
import math
import time
import shutil
import librosa
import logging
import whisperx
import tempfile
import subprocess
import numpy as np
import hashlib
import threading
import easyocr

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import queue

import torch

from ultralytics import YOLO

from scenedetect import detect
from scenedetect.detectors import ContentDetector

# =========================================================
# GROQ AI
# =========================================================

USE_AI = False
client = None

GROQ_API_KEY = "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"

if GROQ_API_KEY:

    try:

        from openai import OpenAI

        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        USE_AI = True

        print("Groq AI ENABLED")

    except Exception as e:

        print("Groq init error:", e)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# =========================================================
# TORCH
# =========================================================

def torch_available():

    try:

        return torch.cuda.is_available()

    except Exception:

        return False

# =========================================================
# GPU CHECK
# =========================================================

def check_nvidia_available():

    try:

        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )

        return result.returncode == 0

    except:

        return False

NVIDIA_AVAILABLE = check_nvidia_available()

QUALITY_PRESETS = {
    "low": "2000k",
    "medium": "5000k",
    "high": "8000k",
}

# =========================================================
# MAIN CLASS - OPTIMIZED
# =========================================================

class AdvancedViralProcessor:

    def __init__(
        self,
        video_path,
        output_dir="processed_data",
        temp_dir=None,
        quality="medium",
        parallel_workers=4
    ):

        self.video_path = os.path.abspath(video_path)

        self.base_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        self.output_path = os.path.abspath(
            os.path.join(output_dir, self.base_name)
        )

        self.temp_root = (
            temp_dir or tempfile.mkdtemp(prefix="viral_")
        )

        self.audio_path = os.path.join(
            self.output_path,
            "audio.wav"
        )

        self.reels_dir = os.path.join(
            self.output_path,
            "reels"
        )

        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.reels_dir, exist_ok=True)

        self.quality = quality
        self.use_gpu = NVIDIA_AVAILABLE
        
        # ✅ NEW: Parallel processing
        self.executor = ThreadPoolExecutor(max_workers=parallel_workers)
        self.results_queue = queue.Queue()

        # ✅ OPTIMIZED: Model caching

        self.face_model = None
        self.person_model = None

        # ✅ Cached audio for silence detection
        self.audio_data = None
        self.audio_sr = None
        self.frame_cache = {}

        self.importance_map = defaultdict(float)
        # OCR
        self.ocr_reader = easyocr.Reader(
            ['en'],
            gpu=self.use_gpu
        )

        self.hook_words = [
            "مستحيل",
            "لن تصدق",
            "صدمة",
            "كارثة",
            "سر",
            "فضيحة",
            "أخطر",
            "عاجل",
            "انتبه",
            "تحذير"
        ]

        logging.info(
            f"Processor initialized - GPU: {self.use_gpu}, "
            f"Parallel Workers: {parallel_workers}"
        )
    # =====================================================
    # =====================================================
    # OCR GAMING TEXT DETECTION
    # =====================================================

    def detect_gaming_text(self, frame):

        try:

            h, w = frame.shape[:2]

            # مناطق HUD المهمة
            roi = frame[
                0:int(h * 0.35),
                :
            ]

            results = self.ocr_reader.readtext(
                roi,
                detail=0
            )

            if not results:
                return 0.0

            text = " ".join(results).lower()

            keywords = [

                "kill",
                "killed",
                "headshot",
                "winner",
                "victory",
                "double kill",
                "triple kill",
                "eliminated",
                "knocked",
                "finish",
                "ace"

            ]

            score = 0.0

            for word in keywords:

                if word in text:
                    score += 0.25

            return min(score, 1.0)

        except Exception as e:

            logging.error(
                f"OCR detection failed: {e}"
            )

            return 0.0
    #============
    def gaming_visual_score(self, frame, prev_frame):

        score = 0.0

        # =====================================
        # FRAME DIFFERENCE
        # =====================================

        if prev_frame is not None:

            diff = cv2.absdiff(
                prev_frame,
                frame
            )

            motion = np.mean(diff) / 255.0

            score += motion * 0.35

        # =====================================
        # BRIGHT FLASHES
        # =====================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        bright_ratio = np.mean(gray > 220)

        score += bright_ratio * 0.25

        # =====================================
        # COLOR INTENSITY
        # =====================================

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        saturation = np.mean(hsv[:, :, 1]) / 255.0

        score += saturation * 0.20
        # =====================================
        # CROSSHAIR DETECTION
        # =====================================

        h, w = gray.shape

        center_x = w // 2
        center_y = h // 2

        crosshair_zone = gray[
            center_y - 40:center_y + 40,
            center_x - 40:center_x + 40
        ]

        lines = cv2.HoughLinesP(
            crosshair_zone,
            1,
            np.pi / 180,
            threshold=25,
            minLineLength=10,
            maxLineGap=3
        )

        if lines is not None:

            vertical = 0
            horizontal = 0

            for line in lines:

                x1, y1, x2, y2 = line[0]

                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                # Vertical line
                if dx < 4 and dy > 8:
                    vertical += 1

                # Horizontal line
                if dy < 4 and dx > 8:
                    horizontal += 1

            if vertical > 0 and horizontal > 0:
                score += 0.25

        # =====================================
        # EDGE DENSITY
        # =====================================

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        edge_density = np.mean(edges > 0)

        score += edge_density * 0.20
        # =====================================
        # KILL FEED DETECTION
        # =====================================

        h, w = frame.shape[:2]

        # المنطقة العلوية اليمنى
        kill_zone = frame[
            0:int(h * 0.25),
            int(w * 0.60):w
        ]

        kill_gray = cv2.cvtColor(
            kill_zone,
            cv2.COLOR_BGR2GRAY
        )

        # كشف النصوص والحواف
        kill_edges = cv2.Canny(
            kill_gray,
            150,
            300
        )

        text_density = np.mean(
            kill_edges > 0
        )

        # وجود نصوص كثيرة = احتمال Kill Feed
        if text_density > 0.08:
            score += 0.20

        # =====================================
        # RED / ORANGE FLASH
        # HEADSHOT / DAMAGE COLORS
        # =====================================

        b, g, r = cv2.split(kill_zone)

        red_mask = (
            (r > 170) &
            (g < 140)
        )

        red_ratio = np.mean(red_mask)

        if red_ratio > 0.03:
            score += 0.15
        # =====================================
        # OCR GAMING EVENTS
        # =====================================

        ocr_score = self.detect_gaming_text(
            frame
        )

        score += ocr_score * 0.35

        return min(score, 1.0)

# =====================================================
# DETECT AUDIO BEATS
# =====================================================

    def detect_audio_beats(self):

        """
        Detect strong audio beats / peaks
        for gaming sync editing
        """

        try:

            y, sr = librosa.load(
                self.audio_path,
                sr=16000
            )

            # =====================================
            # BEAT TRACK
            # =====================================

            tempo, beat_frames = librosa.beat.beat_track(
                y=y,
                sr=sr
            )

            beat_times = librosa.frames_to_time(
                beat_frames,
                sr=sr
            )

            beats = []

            for t in beat_times:

                beats.append(round(float(t), 2))

            logging.info(
                f"Detected {len(beats)} beats"
            )

            return beats

        except Exception as e:

            logging.error(
                f"Beat detection failed: {e}"
            )

            return []
    # =====================================================
    # DETECT GAMING PEAKS
    # =====================================================

    def detect_gaming_peaks(self):

        """
        يبحث عن اللحظات الحماسية في الفيديو
        اعتماداً على:
        - ارتفاع الصوت
        - الحركة
        - التغيرات المفاجئة
        """

        try:

            audio_scores = self.audio_energy()

            motion_scores = self.motion_scores()

            length = min(
                len(audio_scores),
                len(motion_scores)
            )

            peaks = []

            for i in range(length):

                audio = audio_scores[i]
                motion = motion_scores[i]

                # =====================================
                # COMBINED ENERGY
                # =====================================

                combined = (
                    audio * 0.55 +
                    motion * 0.45
                )

                # =====================================
                # PEAK DETECTION
                # =====================================

                if combined > 0.75:

                    peaks.append({
                        "time": i,
                        "strength": round(combined, 3)
                    })

            return peaks

        except Exception as e:

            logging.error(
                f"Gaming peaks detection failed: {e}"
            )

            return []
    # =====================================================
    # GAMING HYPE SCORE
    # =====================================================

    def gaming_hype_score(
        self,
        text,
        audio_score,
        motion_score
    ):

        score = 0.0

        text = text.strip()

        # =========================================
        # AUDIO HYPE
        # =========================================

        if audio_score > 0.85:
            score += 0.35

        elif audio_score > 0.7:
            score += 0.20

        # =========================================
        # MOTION HYPE
        # =========================================

        if motion_score > 0.8:
            score += 0.30

        elif motion_score > 0.6:
            score += 0.15

        # =========================================
        # EXCLAMATIONS
        # =========================================

        exclamations = text.count("!") + text.count("！")

        if exclamations >= 2:
            score += 0.20

        elif exclamations == 1:
            score += 0.10

        # =========================================
        # QUESTION HYPE
        # =========================================

        questions = text.count("?") + text.count("؟")

        if questions >= 1:
            score += 0.08

        # =========================================
        # LETTER REPETITION
        # مثال:
        # لاااااا
        # واااو
        # اهههه
        # =========================================

        repeated = re.findall(
            r"(.)\1{2,}",
            text
        )

        if repeated:
            score += 0.20

        # =========================================
        # CAPS / BIG ENERGY
        # =========================================

        upper_ratio = 0

        letters = [c for c in text if c.isalpha()]

        if letters:

            upper_letters = [
                c for c in letters
                if c.isupper()
            ]

            upper_ratio = (
                len(upper_letters) /
                len(letters)
            )

        if upper_ratio > 0.4:
            score += 0.15

        # =========================================
        # SHORT FAST PHRASES
        # =========================================

        words = text.split()

        if 2 <= len(words) <= 6:
            score += 0.12

        # =========================================
        # FAST SPEECH BONUS
        # =========================================

        if len(words) >= 10:
            score += 0.08

        # =========================================
        # EMOTION DETECTION
        # =========================================

        emotional_patterns = [
            r"هههه+",
            r"اها+",
            r"واو+",
            r"اوو+",
            r"ييي+",
            r"ششش+",
        ]

        for pattern in emotional_patterns:

            if re.search(pattern, text):
                score += 0.15
                break

        return min(score, 1.0)
    # =====================================================
    # GET PROFILE
    # =====================================================

    def get_profile(self, mode):

        profiles = {

            "gaming": {

                # =====================================
                # SCORE WEIGHTS
                # =====================================

                "audio_weight": 0.30,
                "motion_weight": 0.35,
                "visual_weight": 0.08,
                "face_weight": 0.02,
                "hook_weight": 0.10,
                "ai_weight": 0.10,
                "hype_weight": 0.25,

                # =====================================
                # DURATIONS
                # =====================================

                "min_duration": 3.5,
                "max_duration": 8,

                # =====================================
                # STYLE
                # =====================================

                "fast_cuts": True,
                "aggressive_subtitles": True,
                "prefer_peaks": True,
                "prioritize_motion": True,
                "prioritize_loud_audio": True,
            }
        }

        return profiles.get(
            mode,
            profiles["gaming"]
        )   
   
    # =====================================================
    # OPTIMIZED: Model Management
    # =====================================================

    def get_face_model(self):

        """
        Load and cache YOLO face model
        """

        if self.face_model is None:

            logging.info(
                "Loading YOLO face model..."
            )

            try:

                self.face_model = YOLO(
                    "yolov8n-face.pt"
                )

            except Exception as e:

                logging.error(
                    f"Face model load failed: {e}"
                )

                raise RuntimeError(
                    "yolov8n-face.pt not found"
                )

            if self.use_gpu:

                self.face_model.to("cuda")

                try:

                    self.face_model.model.half()

                    logging.info(
                        "Face model using FP16"
                    )

                except:
                    pass

        return self.face_model

    def get_person_model(self):
        """Load and cache person detection model"""
        if self.person_model is None:
            logging.info("Loading person detection model...")
            self.person_model = YOLO("yolov8n.pt")
            if self.use_gpu:
                self.person_model.to('cuda')
                try:
                    self.person_model.model.half()  # ✅ float16 precision
                    logging.info("Person model using half precision")
                except:
                    pass
        return self.person_model

    # =====================================================
    # UTILS
    # =====================================================

    def clean_text(self, text):

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _get_video_codec_advanced(self):
        """
        ✅ OPTIMIZED: Choose best available codec
        Priority: hevc_nvenc > h264_nvenc > libx265 > libx264
        """

        if self.use_gpu:

            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                timeout=5
            )

            codecs_priority = [
                "hevc_nvenc",
                "h264_nvenc",
                "libx265",
            ]

            for codec in codecs_priority:
                if codec in result.stdout:
                    logging.info(f"Using codec: {codec}")
                    return codec

        return "libx264"

    def run_cmd(self, cmd, check=False):

        logging.info(" ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print(result.stderr)

        if check and result.returncode != 0:
            raise RuntimeError("COMMAND FAILED")

        return result

    # =====================================================
    # AUDIO
    # =====================================================

    def extract_audio(self):

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            self.video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            self.audio_path
        ]

        self.run_cmd(cmd, check=True)

        # ✅ CACHE AUDIO FOR FAST REUSE
        self.audio_data, self.audio_sr = librosa.load(
            self.audio_path,
            sr=16000
        )

    # =====================================================
    # TRANSCRIBE
    # =====================================================

    def transcribe(self):

        """
        Optimized transcription
        to avoid CUDA OOM
        """

        try:

            # =====================================
            # FORCE CPU FOR WHISPER
            # =====================================

            device = "cpu"
            audio = whisperx.load_audio(
                self.audio_path
            )

            model = whisperx.load_model(
                "base",
                device=device,
                compute_type="int8"
            )

            result = model.transcribe(
                audio,
                batch_size=4
            )
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=device
            )

            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device
            )

            segments = []

            for seg in result["segments"]:

                words = seg.get("words", [])

                if not words:
                    continue

                chunk_words = []

                chunk_start = None

                for w in words:

                    word = w.get(
                        "word",
                        ""
                    ).strip()

                    if not word:
                        continue

                    if chunk_start is None:
                        chunk_start = float(
                            w["start"]
                        )

                    chunk_words.append(word)

                    # كل 3 كلمات subtitle
                    if len(chunk_words) >= 3:

                        text = " ".join(
                            chunk_words
                        )

                        text = self.clean_text(
                            text
                        )

                        if text:

                            split_words = text.split()

                            if len(split_words) > 1:

                                mid = len(
                                    split_words
                                ) // 2

                                text = (
                                    " ".join(
                                        split_words[:mid]
                                    )
                                    + "\\N" +
                                    " ".join(
                                        split_words[mid:]
                                    )
                                )

                            segments.append({

                                "start": chunk_start,

                                "end": float(
                                    w["end"]
                                ),

                                "text": text
                            })

                        chunk_words = []

                        chunk_start = None

                # باقي الكلمات
                if chunk_words:

                    text = " ".join(
                        chunk_words
                    )

                    text = self.clean_text(
                        text
                    )

                    if text:

                        segments.append({

                            "start": chunk_start,

                            "end": float(
                                words[-1]["end"]
                            ),

                            "text": text
                        })

            del model
            del model_a

            gc.collect()

            return segments

        except Exception as e:

            logging.error(
                f"Transcribe failed: {e}"
            )

            return []

    # =====================================================
    # AUDIO ENERGY
    # =====================================================

    def audio_energy(self):

        y, sr = librosa.load(self.audio_path, sr=16000)

        duration = int(
            math.ceil(librosa.get_duration(y=y, sr=sr))
        )

        energies = []

        for sec in range(duration):

            start = sec * sr
            end = min((sec + 1) * sr, len(y))

            frame = y[start:end]

            if len(frame) == 0:
                energies.append(0)
                continue

            energy = float(np.mean(frame ** 2))

            energies.append(energy)

        mx = max(energies) if energies else 1

        return [e / mx for e in energies]

    # =====================================================
    # SHARED FRAME CACHE
    # =====================================================

    def load_sampled_frames(self):

        """
        Load video frames once
        and share them across all analyzers
        """

        if self.frame_cache:
            return self.frame_cache

        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        duration = int(total_frames / fps)

        for sec in range(duration):

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(sec * fps)
            )

            ret, frame = cap.read()

            if not ret:
                continue

            self.frame_cache[sec] = frame

        cap.release()

        logging.info(
            f"Loaded {len(self.frame_cache)} shared frames"
        )

        return self.frame_cache
    # =====================================================
    # MOTION - OPTIMIZED FRAME SAMPLING
    # =====================================================

    def motion_scores(self):
        """
        ✅ OPTIMIZED:
        - Sample frames every second
        - Faster processing
        - Stable interpolation
        """
        frames = self.load_sampled_frames()
        duration = len(frames)

        scores = []

        prev = None

        for sec in range(duration):

            frame = frames.get(sec)

            if frame is None:
                scores.append(0)
                continue

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.GaussianBlur(
                gray,
                (21, 21),
                0
            )

            if prev is None:

                prev = gray

                scores.append(0)

                continue

            diff = cv2.absdiff(prev, gray)

            val = np.mean(diff)

            scores.append(float(val))

            prev = gray

        

        if not scores:
            return []

        mx = max(scores)

        if mx <= 0:
            return [0] * len(scores)

        return [
            s / mx
            for s in scores
        ]
    #=====================================================
    
    
    def detect_subject_position(self, frame):

        model = self.get_person_model()

        results = model.predict(
            frame,
            verbose=False,
            conf=0.4
        )

        if not results:
            return 0.5

        r = results[0]

        if not r.boxes:
            return 0.5

        largest = None
        largest_area = 0

        for box in r.boxes.xyxy.cpu().numpy():

            x1, y1, x2, y2 = box

            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:
                largest_area = area
                largest = box

        if largest is None:
            return 0.5

        x1, y1, x2, y2 = largest

        center_x = (x1 + x2) / 2

        width = frame.shape[1]

        normalized = center_x / width

        return normalized    
    #=====================================================
    # =====================================================
    # VISUAL SCORES - OPTIMIZED WITH SAMPLING
    # =====================================================

    def visual_scores(self):
        """
        ✅ OPTIMIZED: 
        - Sample frames every 0.5 seconds
        - Larger batch size (16 instead of 8)
        - Interpolate results
        """

        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        frames = []
        frame_indices = []
        
        # ✅ Sample every 0.5 seconds
        sample_rate = max(1, int(fps * 0.5))

        for sec in range(duration):

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))

            ret, frame = cap.read()

            if ret and sec % max(1, sample_rate // fps) == 0:
                frames.append(frame)
                frame_indices.append(sec)

        cap.release()

        if not frames:
            return [0] * duration

        sampled_scores = []

        prev_frame = None

        for frame in frames:

            frame = cv2.resize(
                frame,
                (640, 360)
            )

            score = self.gaming_visual_score(
                frame,
                prev_frame
            )

            sampled_scores.append(score)

            prev_frame = frame

        # ✅ Interpolate to full duration
        full_scores = []
        for sec in range(duration):
            if sec in frame_indices:
                idx = frame_indices.index(sec)
                full_scores.append(sampled_scores[idx])
            else:
                # Use nearest neighbor
                nearest_idx = min(
                    range(len(frame_indices)),
                    key=lambda i: abs(frame_indices[i] - sec)
                )
                full_scores.append(sampled_scores[nearest_idx])

        # ✅ Clean memory
        torch.cuda.empty_cache() if self.use_gpu else None

        return full_scores

    def face_scores(self):

        """
        YOLO Face Detection
        Fast GPU Batch Processing
        """

        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        duration = int(total_frames / fps)

        frames = []
        frame_indices = []

        # =====================================
        # SAMPLE FRAMES
        # =====================================

        for sec in range(duration):

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(sec * fps)
            )

            ret, frame = cap.read()

            if not ret:
                continue

            frames.append(frame)
            frame_indices.append(sec)

        cap.release()

        if not frames:
            return [0] * duration

        # =====================================
        # LOAD YOLO FACE MODEL
        # =====================================

        model = self.get_face_model()

        # =====================================
        # BATCH INFERENCE
        # =====================================

        with torch.amp.autocast(
            device_type="cuda",
            enabled=self.use_gpu
        ):

            results = model.predict(
                frames,
                verbose=False,
                batch=32,
                device=0 if self.use_gpu else "cpu",
                conf=0.35
            )

        sampled_scores = []

        # =====================================
        # PROCESS RESULTS
        # =====================================

        for idx, r in enumerate(results):

            frame = frames[idx]

            frame_area = (
                frame.shape[0] *
                frame.shape[1]
            )

            max_score = 0

            if r.boxes is not None and len(r.boxes) > 0:

                for box in r.boxes.xyxy.cpu().numpy():

                    x1, y1, x2, y2 = box

                    face_area = (
                        (x2 - x1) *
                        (y2 - y1)
                    )

                    ratio = (
                        face_area /
                        frame_area
                    )

                    score = min(
                        ratio * 12,
                        1.0
                    )

                    max_score = max(
                        max_score,
                        score
                    )

            sampled_scores.append(
                float(max_score)
            )

        # =====================================
        # INTERPOLATE FULL TIMELINE
        # =====================================

        full_scores = []

        for sec in range(duration):

            if sec in frame_indices:

                i = frame_indices.index(sec)

                full_scores.append(
                    sampled_scores[i]
                )

            else:

                nearest_idx = min(
                    range(len(frame_indices)),
                    key=lambda i:
                    abs(frame_indices[i] - sec)
                )

                full_scores.append(
                    sampled_scores[nearest_idx]
                )

        # =====================================
        # CLEAN GPU MEMORY
        # =====================================

        if self.use_gpu:
            torch.cuda.empty_cache()

        return full_scores
    # =====================================================
    # SCENE DETECT - WITH CACHING
    # =====================================================

    def detect_scenes(self):
        """
        ✅ OPTIMIZED: Cache scene detection results
        """

        cache_file = os.path.join(self.output_path, "scenes_cache.json")

        # ✅ Check cache
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cached = json.load(f)
                logging.info(f"Loaded {len(cached)} scenes from cache")
                return cached

        scenes = detect(
            self.video_path,
            ContentDetector(threshold=27.0)
        )

        times = []

        for scene in scenes:
            times.append(scene[0].get_seconds())

        # ✅ Save to cache
        with open(cache_file, 'w') as f:
            json.dump(times, f)

        return times

    # =====================================================
    # SILENCE PENALTY
    # =====================================================

    def detect_silence_penalty(self, start, end):

        # ✅ USE CACHED AUDIO
        y = self.audio_data
        sr = self.audio_sr

        if y is None:
            return 1.0

        s = int(start * sr)
        e = int(end * sr)

        segment = y[s:e]

        if len(segment) == 0:
            return 1.0

        rms = librosa.feature.rms(y=segment)[0]

        energy = float(np.mean(rms))

        if energy < 0.005:
            return 0.2

        if energy < 0.01:
            return 0.5

        return 1.0

    # =====================================================
    # HOOK SCORE
    # =====================================================

    def hook_score(self, text):

        score = 0

        for word in self.hook_words:

            if word in text:
                score += 0.15

        if "؟" in text or "?" in text:
            score += 0.1

        if "!" in text:
            score += 0.1

        return min(score, 1.0)

    # =====================================================
    # AI SCORE - ADVANCED WITH CACHING
    # =====================================================

    def ai_score(self, text):
        """
        ✅ OPTIMIZED: 
        - Advanced prompt with more factors
        - Result caching
        - Better error handling
        """

        if not USE_AI:
            return self._default_ai_score()

        # ✅ Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_file = os.path.join(
            self.output_path,
            f"ai_cache_{text_hash}.json"
        )

        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f)

        try:

            prompt = f"""
Analyze this text for SHORT-FORM VIRAL potential (TikTok/Reels style).

Consider these factors and return ONLY a JSON object with scores 0-1:
- hook: How engaging is the opening/hook
- emotion: Emotional resonance (funny, sad, angry, etc)
- surprise: Surprise/twist factor
- urgency: Sense of urgency or FOMO
- controversy: Controversial/debate potential
- cta: Call-to-action strength
- curiosity: Makes people want to know more
- shareability: How likely to be shared

TEXT:
{text[:800]}

Return ONLY valid JSON, no markdown, no explanation.
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=250,
                top_p=0.9,
            )

            raw = response.choices[0].message.content

            # ✅ Safe JSON extraction
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = self._default_ai_score()

            # ✅ Better weighting for viral content
            final = (
                data.get("hook", 0.5) * 0.30 +        # ✅ Hook is most important
                data.get("emotion", 0.5) * 0.20 +
                data.get("surprise", 0.5) * 0.15 +
                data.get("urgency", 0.5) * 0.15 +
                data.get("curiosity", 0.5) * 0.10 +
                data.get("cta", 0.5) * 0.10
            )

            data["final"] = min(final, 1.0)

            # ✅ Save to cache
            with open(cache_file, 'w') as f:
                json.dump(data, f)

            return data

        except Exception as e:
            logging.error(f"AI score error: {e}")
            return self._default_ai_score()

    def _default_ai_score(self):
        """Default AI score when AI is unavailable"""
        return {
            "hook": 0.5,
            "emotion": 0.5,
            "surprise": 0.5,
            "urgency": 0.5,
            "curiosity": 0.5,
            "cta": 0.5,
            "controversy": 0.5,
            "shareability": 0.5,
            "final": 0.5,
        }

    # =====================================================
    # EXCITEMENT SCORE
    # =====================================================

    def excitement_score(self, text, audio_score):

        score = 0

        if audio_score > 0.7:
            score += 0.4

        exciting = [
            "لا",
            "مستحيل",
            "كارثة",
            "صدمة",
            "أخطر"
        ]

        for e in exciting:

            if e in text:
                score += 0.15

        return min(score, 1.0)

    # =====================================================
    # DYNAMIC DURATION
    # =====================================================

    def dynamic_duration(self, score):

        # 🔥 Viral explosive moments
        if score >= 1.15:
            return 3.5

        # 🔥 Strong action
        if score >= 1.0:
            return 4.5

        # 🔥 Good gaming clip
        if score >= 0.85:
            return 6

        # متوسط
        if score >= 0.7:
            return 7

        return 8

    # =====================================================
    # SMART RANK
    # =====================================================

    def smart_rank(self, scored_segments):

        if not scored_segments:
            return []

        final = []

        # ابدأ بأقوى clip
        current = scored_segments[0]

        final.append(current)

        remaining = scored_segments[1:]

        while remaining and len(final) < 4:

            best_next = None
            best_score = -1

            for seg in remaining:

                transition = self.transition_score(
                    current,
                    seg
                )

                combined = (
                    seg["score"] * 0.7 +
                    transition * 0.3
                )

                if combined > best_score:
                    best_score = combined
                    best_next = seg

            if best_next is None:
                break

            final.append(best_next)

            remaining.remove(best_next)

            current = best_next

        return final

    # =====================================================
    # SCORE SEGMENTS
    # =====================================================

    def score_segments(
        self,
        segments,
        audio_scores,
        motion_scores,
        visual_scores,
        face_scores,
        scene_changes,
        gaming_peaks,
        
    ):
        profile = self.get_profile("gaming")

        results = []

        for seg in segments:

            s = int(seg["start"])
            e = int(seg["end"])

            if e <= s:
                continue

            text = seg["text"]

            # =========================================
            # SAFE AUDIO
            # =========================================

            try:
                a = float(np.mean(audio_scores[s:e]))
            except:
                a = 0

            # =========================================
            # SAFE MOTION
            # =========================================

            try:
                m = float(np.mean(motion_scores[s:e]))
            except:
                m = 0

            # =========================================
            # SAFE VISUAL
            # =========================================

            try:
                v = float(np.mean(visual_scores[s:e]))
            except:
                v = 0

            # =========================================
            # SAFE FACE SCORES
            # =========================================

            fs = face_scores or []

            if len(fs) == 0:

                f = 0

            else:

                safe_s = min(s, len(fs))
                safe_e = min(e, len(fs))

                if safe_s >= safe_e:
                    f = 0
                else:
                    f = float(
                        np.mean(fs[safe_s:safe_e])
                    )

            # =========================================
            # AI / HOOK
            # =========================================

            h = self.hook_score(text)

            ai_data = self.ai_score(text)

            ai = ai_data["final"]

            gaming_hype = self.gaming_hype_score(
                text,
                a,
                m
            )

            silence_penalty = self.detect_silence_penalty(
                seg["start"],
                seg["end"]
            )

            # =========================================
            # SCENE BONUS
            # =========================================

            scene_bonus = 0

            for sc in scene_changes:

                if s <= sc <= e:
                    scene_bonus = 0.15
                    break

            # =========================================
            # GAMING PEAK BONUS
            # =========================================

            peak_bonus = 0

            for peak in gaming_peaks:

                peak_time = peak["time"]

                if s <= peak_time <= e:

                    peak_bonus = max(
                        peak_bonus,
                        peak["strength"] * 0.25
                    )

            # =========================================
            # FINAL SCORE
            # =========================================

            score = (
                (a * profile["audio_weight"]) +
                (m * profile["motion_weight"]) +
                (v * profile["visual_weight"]) +
                (f * profile["face_weight"]) +
                (h * profile["hook_weight"]) +
                (ai * profile["ai_weight"]) +
                (gaming_hype * profile["hype_weight"]) +
                (silence_penalty * 0.05) +
                scene_bonus + peak_bonus
            )

            results.append({
                "start": seg["start"],
                "end": seg["end"],
                "score": round(score, 4),
                "text": text,
                "audio": round(a, 3),
                "motion": round(m, 3),
                "visual": round(v, 3),
                "faces": round(f, 3),
                "hook": round(h, 3),
                "ai": round(ai, 3),
                "hype": round(gaming_hype, 3),
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results
    
    # =====================================================
    # SUBTITLE GENERATION
    # =====================================================

    def format_srt_time(self, seconds):

        ms = int((seconds % 1) * 1000)
        s = int(seconds % 60)
        m = int((seconds // 60) % 60)
        h = int(seconds // 3600)

        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def generate_subtitles(
        self,
        segments,
        path,
        offset=0
    ):

        with open(path, "w", encoding="utf-8") as f:

            for idx, seg in enumerate(segments, start=1):

                # =====================================
                # RELATIVE TIMING
                # =====================================

                start_time = max(
                    0,
                    seg["start"] - offset
                )

                end_time = max(
                    start_time + 0.1,
                    seg["end"] - offset
                )

                start = self.format_srt_time(
                    start_time
                )

                end = self.format_srt_time(
                    end_time
                )

                f.write(f"{idx}\n")
                f.write(f"{start} --> {end}\n")
                f.write(seg["text"] + "\n\n")
    # =====================================================
    # GENERATE REEL - OPTIMIZED WITH PARALLEL ENCODING
    # =====================================================
    def generate_reel(self, best_segments):

        temp_clips = []

        bitrate = QUALITY_PRESETS.get(
            self.quality,
            "5000k"
        )



        codec = self._get_video_codec_advanced()

        clip_futures = {}

        with ThreadPoolExecutor(max_workers=2) as clip_executor:

            for idx, seg in enumerate(best_segments):

                hook_strength = self.hook_score(
                    seg["text"]
                )

                # =========================================
                # SMART START
                # =========================================

                if hook_strength > 0.7:
                    start = max(0, seg["start"] - 1)
                else:
                    start = max(0, seg["start"] - 3)

                # =========================================
                # CLIP SUBTITLE
                # =========================================

                clip_subtitle = os.path.join(
                    self.temp_root,
                    f"clip_{idx}.srt"
                )

                self.generate_subtitles(
                    [seg],
                    clip_subtitle,
                    offset=start
                )
                # =========================================
                # SMART START
                # =========================================

                if hook_strength > 0.7:
                    start = max(0, seg["start"] - 1)
                else:
                    start = max(0, seg["start"] - 3)

                profile = self.get_profile("gaming")

                # =========================================
                # SMART GAMING DURATION
                # =========================================

                base_duration = self.dynamic_duration(
                    seg["score"]
                )

                # hype clips تصبح أسرع
                if seg.get("hype", 0) > 0.75:
                    base_duration *= 0.75

                # الحركة العالية = قص أسرع
                if seg.get("motion", 0) > 0.8:
                    base_duration *= 0.80

                duration = np.clip(
                    base_duration,
                    profile["min_duration"],
                    profile["max_duration"]
                )

                # small breathing room after speech
                duration += 0.6

                # minimum safety duration
                duration = max(duration, 3)

                out = os.path.join(
                    self.temp_root,
                    f"clip_{idx}.mp4"
                )

                # =========================================
                # VIDEO FILTER
                # =========================================

                subtitle_fixed = clip_subtitle.replace("\\", "/")

                vf = (
                    "scale=720:1280:"
                    "force_original_aspect_ratio=decrease,"

                    "pad=720:1280:(ow-iw)/2:(oh-ih)/2,"

                    f"subtitles='{subtitle_fixed}':"
                    "force_style='"
                    "FontName=DejaVu Sans,"
                    "Fontsize=20,"
                    "PrimaryColour=&Hffffff&,"
                    "OutlineColour=&H000000&,"
                    "BackColour=&H66000000&,"
                    "BorderStyle=3,"
                    "Outline=2,"
                    "Shadow=1,"
                    "MarginV=60,"
                    "Alignment=2"
                    "'"
                )

                # =========================================
                # FFMPEG COMMAND
                # =========================================

                cmd = [
                    "ffmpeg",
                    "-y",
                ]

                if self.use_gpu:
                    cmd += [
                        "-hwaccel",
                        "cuda"
                    ]

                cmd += [
                    "-ss", str(start),
                    "-t", str(duration),

                    "-i", self.video_path,

                    "-vf", vf,

                    "-c:v", codec,

                    "-preset", "fast",

                    "-b:v", bitrate,

                    "-c:a", "aac",
                    "-b:a", "128k",

                    "-movflags",
                    "+faststart",

                    out
                ]

                future = clip_executor.submit(
                    self.run_cmd,
                    cmd,
                    True
                )

                clip_futures[future] = (
                    idx,
                    out
                )

            # =========================================
            # COLLECT RESULTS
            # =========================================

            for future in as_completed(
                clip_futures
            ):

                idx, out = clip_futures[future]

                try:

                    future.result()

                    temp_clips.append(
                        (idx, out)
                    )

                    logging.info(
                        f"✓ Clip {idx} encoded"
                    )

                except Exception as e:

                    logging.error(
                        f"✗ Clip {idx} failed: {e}"
                    )

        # =========================================
        # SORT CLIPS
        # =========================================

        temp_clips.sort(
            key=lambda x: x[0]
        )

        temp_clips = [
            path for _, path in temp_clips
        ]

        if not temp_clips:
            raise RuntimeError(
                "No clips generated"
            )

        # =========================================
        # CONCAT FILE
        # =========================================

        concat_file = os.path.join(
            self.temp_root,
            "concat.txt"
        )

        with open(concat_file, "w") as f:

            for clip in temp_clips:

                clip_fixed = clip.replace(
                    "\\",
                    "/"
                )

                f.write(
                    f"file '{clip_fixed}'\n"
                )

        final_out = os.path.join(
            self.reels_dir,
            "viral_reel.mp4"
        )

        # =========================================
        # FINAL CONCAT
        # =========================================

        cmd = [
            "ffmpeg",
            "-y",

            "-f", "concat",
            "-safe", "0",

            "-i", concat_file,

            "-c", "copy",

            final_out
        ]

        self.run_cmd(cmd, check=True)

        return final_out
   
    # =====================================================
    # CLEANUP
    # =====================================================

    def cleanup(self):

        try:
            shutil.rmtree(self.temp_root)
            logging.info("Cleaned up temporary files")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")
    #======================================================
    def transition_score(self, seg1, seg2):

        score = 0

        # ---------------------------------
        # Audio continuity
        # ---------------------------------

        audio_diff = abs(seg1["audio"] - seg2["audio"])

        score += max(0, 1 - audio_diff)

        # ---------------------------------
        # Motion continuity
        # ---------------------------------

        motion_diff = abs(seg1["motion"] - seg2["motion"])

        score += max(0, 1 - motion_diff)

        # ---------------------------------
        # Visual continuity
        # ---------------------------------

        visual_diff = abs(seg1["visual"] - seg2["visual"])

        score += max(0, 1 - visual_diff)

        # ---------------------------------
        # Face continuity
        # ---------------------------------

        face_diff = abs(seg1["faces"] - seg2["faces"])

        score += max(0, 1 - face_diff)

        return score / 4

    # =====================================================
    # PROCESS - PARALLEL EXECUTION
    # =====================================================

    def process(self):
        """
        ✅ OPTIMIZED: Parallel processing of independent tasks
        """

        total = time.time()

        try:

            logging.info("[1/9] Extracting audio...")
            self.extract_audio()

            # =====================================================
            # STEP 1: TRANSCRIBE FIRST
            # =====================================================

            logging.info("[2/9] Transcribing audio...")

            segments = self.transcribe()

            if not segments:
                logging.error("No segments transcribed")
                return None

            # =====================================================
            # STEP 2: PARALLEL LIGHT TASKS
            # =====================================================

            logging.info("[3-7/9] Running parallel analysis...")

            futures = {
                self.executor.submit(self.audio_energy): "audio_energy",
                self.executor.submit(self.motion_scores): "motion",
                self.executor.submit(self.visual_scores): "visual",
                self.executor.submit(self.face_scores): "faces",
                self.executor.submit(self.detect_scenes): "scenes",
                self.executor.submit(self.detect_gaming_peaks): "gaming_peaks",
            }

            results = {}
            completed_count = 0

            for future in as_completed(futures):

                key = futures[future]

                try:

                    results[key] = future.result()

                    completed_count += 1

                    logging.info(
                        f"✓ {key} completed ({completed_count}/6)"
                    )

                except Exception as e:

                    logging.error(f"✗ {key} failed: {e}")

                    results[key] = None

            # =====================================================
            # RESULTS
            # =====================================================

            audio_scores = results.get("audio_energy", [])
            motion_scores = results.get("motion", [])
            visual_scores = results.get("visual", [])
            face_scores = results.get("faces", [])
            scenes = results.get("scenes", [])
            gaming_peaks = results.get("gaming_peaks", [])
            if not segments:
                logging.error("No segments transcribed")
                return None

            logging.info("[8/9] AI scoring segments...")

            scored = self.score_segments(
                segments,
                audio_scores,
                motion_scores,
                visual_scores,
                face_scores,
                scenes,
                gaming_peaks , 
                
                
            )

            best = self.smart_rank(scored)

            best.sort(key=lambda x: x["start"])

            json_path = os.path.join(
                self.output_path,
                "segments.json"
            )

            with open(
                json_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    best,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            logging.info(f"Found {len(best)} best segments")
            for i, seg in enumerate(best, 1):
                logging.info(
                    f"  {i}. {seg['text'][:50]}... "
                    f"(score: {seg['score']:.3f})"
                )

            logging.info("[9/9] Generating reel...")

            reel = self.generate_reel(best)

            elapsed = round(time.time() - total, 2)
            logging.info(f"✅ FINAL REEL: {reel}")
            logging.info(f"⏱️  TOTAL TIME: {elapsed} sec")

            return reel

        except Exception as e:
            logging.error(f"Process failed: {e}", exc_info=True)
            return None

        finally:

            self.executor.shutdown(wait=False, cancel_futures=True)
            self.cleanup()

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced Viral Video Processor - Convert videos to optimized reels"
    )

    parser.add_argument(
        "video",
        help="Path to input video file"
    )

    parser.add_argument(
        "--out",
        default="processed_data",
        help="Output directory (default: processed_data)"
    )

    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Output quality (default: medium)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    args = parser.parse_args()

    # ✅ Validate input
    if not os.path.exists(args.video):
        logging.error(f"Video file not found: {args.video}")
        sys.exit(1)

    processor = AdvancedViralProcessor(
        args.video,
        output_dir=args.out,
        quality=args.quality,
        parallel_workers=args.workers
    )

    result = processor.process()

    if result:
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS: Reel generated")
        print(f"📁 Location: {result}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"❌ FAILED: Could not generate reel")
        print(f"{'='*60}")
        sys.exit(1)