"""Bridge between the FastAPI backend and the SwachLens AI inference pipeline."""

from pathlib import Path
import sys

# Project root:
# C:\Users\ayush\OneDrive\Desktop\swachh_lens
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load the AI inference module from:
# project_root/ai/inference/analyze_image.py
AI_INFERENCE_DIR = PROJECT_ROOT / "ai" / "inference"

if str(AI_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_INFERENCE_DIR))

from analyze_image import analyze_image_bytes


def analyze_image(image_bytes: bytes) -> dict:
    """Run the trained SwachLens AI pipeline on uploaded image bytes."""
    return analyze_image_bytes(image_bytes)
