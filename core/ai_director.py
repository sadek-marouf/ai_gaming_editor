# core/ai_director.py

import os
import time
import json

from core.logger import get_logger
from core.config import Config

logger = get_logger("AI_DIRECTOR")


def analyze_gaming_video_context(video_path, transcription_text=""):
    """Leverage Gemini 1.5 Flash long-context capability to understand
    game semantics and extract high-intent highlight blocks.

    Args:
        video_path: path to the full-length gameplay video
        transcription_text: raw text from WhisperX transcription

    Returns:
        dict with "highlights" list of {start, end, reason}

    Raises:
        ImportError: if google-genai SDK is not installed
        ValueError: if Gemini file processing fails
        RuntimeError: if GEMINI_API_KEY is missing
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai SDK not installed. "
            "Run: pip install google-genai"
        )

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. "
            "Export it as an environment variable before running."
        )

    model_name = Config.GEMINI_MODEL

    client = genai.Client(api_key=api_key)

    logger.info(f"Uploading video for structural analysis: {video_path}")
    video_file = client.files.upload(file=video_path)

    # Poll until Gemini finishes indexing the video
    while video_file.state.name == "PROCESSING":
        logger.info("Gemini is tokenizing and indexing video structures...")
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Google Gen AI Native File Processing Failed.")

    prompt = f"""
You are an expert Gaming Video Director. Analyze this PUBG Mobile gameplay video and its corresponding voice transcription text:

Transcription context: "{transcription_text}"

Tasks:
1. Scan the full video context to isolate the absolute best highlights (squad wipes, high-skill clutches, intensive kills).
2. Identify the precise start and end timestamps (in seconds).
3. Each highlight should be 6-15 seconds long.
4. Focus on actual combat moments, not driving, looting, or idle time.
5. Rank highlights by excitement level.

You must output strictly a valid raw JSON object matching the schema below. Do not wrap it in markdown blocks:
{{
  "highlights": [
    {{"start": int, "end": int, "reason": "string"}}
  ]
}}
"""

    logger.info(f"Executing Multimodal Model Inference ({model_name})...")
    response = client.models.generate_content(
        model=model_name,
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    result = json.loads(response.text)

    highlights = result.get("highlights", [])
    logger.info(f"Gemini identified {len(highlights)} highlights")
    for i, h in enumerate(highlights, 1):
        logger.info(
            f"  {i}. [{h['start']}s - {h['end']}s] {h.get('reason', '')}"
        )

    # Cleanup: delete the uploaded file
    try:
        client.files.delete(name=video_file.name)
        logger.info("Cleaned up uploaded video from Gemini")
    except Exception:
        pass

    return result


def is_gemini_available():
    """Check if Gemini AI Director can be used."""
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return bool(api_key)
