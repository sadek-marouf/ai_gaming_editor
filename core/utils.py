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
    if use_gpu:
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for codec in ["hevc_nvenc", "h264_nvenc", "libx265"]:
                if codec in result.stdout:
                    logger.info(f"Using codec: {codec}")
                    return codec
        except Exception:
            pass
    return "libx264"
