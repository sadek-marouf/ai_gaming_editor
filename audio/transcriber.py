# audio/transcriber.py

import gc

from core.logger import get_logger
from core.config import Config
from core.utils import clean_text

logger = get_logger("TRANSCRIBER")


class Transcriber:

    def __init__(self, words_per_chunk=None):
        self.words_per_chunk = (
            words_per_chunk or Config.SUBTITLE_WORDS_PER_CHUNK
        )

    def transcribe(self, audio_path):
        try:
            import whisperx

            device = "cpu"
            audio = whisperx.load_audio(audio_path)

            model = whisperx.load_model(
                "base",
                device=device,
                compute_type="int8",
            )

            result = model.transcribe(audio, batch_size=4)

            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=device,
            )

            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
            )

            segments = self._chunk_segments(result["segments"])

            del model
            del model_a
            gc.collect()

            logger.info(f"Transcribed {len(segments)} segments")
            return segments

        except ImportError:
            logger.warning("whisperx not installed, skipping transcription")
            return []
        except Exception as e:
            logger.error(f"Transcribe failed: {e}")
            return []

    def _chunk_segments(self, raw_segments):
        segments = []

        for seg in raw_segments:
            words = seg.get("words", [])
            if not words:
                continue

            chunk_words = []
            chunk_start = None

            for w in words:
                word = w.get("word", "").strip()
                if not word:
                    continue

                if chunk_start is None:
                    chunk_start = float(w["start"])

                chunk_words.append(word)

                if len(chunk_words) >= self.words_per_chunk:
                    text = clean_text(" ".join(chunk_words))

                    if text:
                        split_words = text.split()
                        if len(split_words) > 1:
                            mid = len(split_words) // 2
                            text = (
                                " ".join(split_words[:mid])
                                + "\\N"
                                + " ".join(split_words[mid:])
                            )

                        segments.append({
                            "start": chunk_start,
                            "end": float(w["end"]),
                            "text": text,
                        })

                    chunk_words = []
                    chunk_start = None

            if chunk_words:
                text = clean_text(" ".join(chunk_words))
                if text:
                    segments.append({
                        "start": chunk_start,
                        "end": float(words[-1]["end"]),
                        "text": text,
                    })

        return segments
