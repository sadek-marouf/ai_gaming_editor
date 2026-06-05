# core/utils.py

import re
import subprocess

from core.logger import get_logger

logger = get_logger("UTILS")


def check_gpu_available():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def torch_gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def run_cmd(cmd, check=False):
    logger.info(" ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        logger.error(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_best_codec(use_gpu=False):
    """Select best available video codec.

    Priority: hevc_nvenc (GPU) > h264_nvenc (GPU) > libx264 (CPU).
    """
    if use_gpu:
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for codec in ["hevc_nvenc", "h264_nvenc"]:
                if codec in result.stdout:
                    logger.info(f"Using codec: {codec}")
                    return codec
        except Exception:
            pass
    logger.info("Using CPU codec: libx264")
    return "libx264"


def get_codec_args(codec, use_gpu=False):
    """Return encoder-specific CLI arguments for premium quality.

    GPU (NVENC): hevc_nvenc -preset p7 -rc constqp -qp 18
    CPU: libx264 -preset slow -crf 18
    """
    from core.config import Config

    if "nvenc" in codec:
        return [
            "-c:v", codec,
            "-preset", Config.NVENC_PRESET,
            "-rc", "constqp",
            "-qp", str(Config.NVENC_QP),
        ]
    return [
        "-c:v", codec,
        "-preset", Config.CPU_PRESET,
        "-crf", str(Config.CPU_CRF),
    ]
