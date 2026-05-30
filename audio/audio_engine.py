# audio/audio_engine.py

import librosa
import numpy as np
import math
from core.logger import get_logger
from core.config import Config

logger = get_logger("AUDIO_ENGINE")


class AudioEngine:

    def __init__(self, video_loader):

        self.video_loader = video_loader
        self.audio_path = None

        self.audio = None
        self.sr = None

        self.energy_cache = None
        self.beats_cache = None

    # =====================================================
    # LOAD AUDIO
    # =====================================================

    def load_audio(self, audio_path):

        self.audio_path = audio_path

        self.audio, self.sr = librosa.load(
            audio_path,
            sr=Config.AUDIO_SAMPLE_RATE
        )

        logger.info("Audio loaded successfully")

    # =====================================================
    # AUDIO ENERGY (per second)
    # =====================================================

    def compute_energy(self):

        if self.audio is None:
            raise RuntimeError("Audio not loaded")

        duration = int(
            math.ceil(librosa.get_duration(y=self.audio, sr=self.sr))
        )

        energies = []

        for sec in range(duration):

            start = sec * self.sr
            end = min((sec + 1) * self.sr, len(self.audio))

            chunk = self.audio[start:end]

            if len(chunk) == 0:
                energies.append(0)
                continue

            energy = float(np.mean(chunk ** 2))
            energies.append(energy)

        # normalize
        mx = max(energies) if energies else 1
        energies = [e / mx for e in energies]

        self.energy_cache = energies

        logger.info(f"Energy computed: {len(energies)} segments")

        return energies

    # =====================================================
    # BEAT DETECTION
    # =====================================================

    def detect_beats(self):

        if self.audio is None:
            raise RuntimeError("Audio not loaded")

        tempo, beat_frames = librosa.beat.beat_track(
            y=self.audio,
            sr=self.sr
        )

        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=self.sr
        )

        beats = [round(float(t), 2) for t in beat_times]

        self.beats_cache = beats

        logger.info(f"Detected {len(beats)} beats")

        return beats

    # =====================================================
    # SILENCE DETECTION
    # =====================================================

    def detect_silence(self, threshold=0.01):

        if self.audio is None:
            raise RuntimeError("Audio not loaded")

        duration = int(
            math.ceil(librosa.get_duration(y=self.audio, sr=self.sr))
        )

        silence_map = []

        for sec in range(duration):

            start = sec * self.sr
            end = min((sec + 1) * self.sr, len(self.audio))

            chunk = self.audio[start:end]

            if len(chunk) == 0:
                silence_map.append(1.0)
                continue

            rms = np.sqrt(np.mean(chunk ** 2))

            silence_map.append(1.0 if rms < threshold else 0.0)

        logger.info("Silence map computed")

        return silence_map

    # =====================================================
    # COMBINED AUDIO SCORE
    # =====================================================

    def get_audio_profile(self):

        if self.energy_cache is None:
            self.compute_energy()

        if self.beats_cache is None:
            self.detect_beats()

        return {
            "energy": self.energy_cache,
            "beats": self.beats_cache,
        }