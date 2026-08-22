from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

# swachhLens-main/
BASE_DIR = Path(__file__).resolve().parents[2]

# swachhLens-main/ai/inference/
AI_INFERENCE_DIR = BASE_DIR / "ai" / "inference"

if str(AI_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_INFERENCE_DIR))


# ============================================================
# AI INFERENCE MODULE
# ============================================================

try:
    import analyze_image as _ai

except Exception as exc:
    _ai = None
    _IMPORT_ERROR = exc

else:
    _IMPORT_ERROR = None


# ============================================================
# MODEL CACHING
# ============================================================

if _ai is not None:

    _original_load_waste_gate = _ai.load_waste_gate
    _original_load_yolo = _ai.load_yolo
    _original_load_scene_model = _ai.load_scene_model

    _waste_gate = None
    _yolo_model = None
    _scene_model = None

    def _cached_load_waste_gate():
        global _waste_gate

        if _waste_gate is None:
            print("Loading Waste Gate V2...")
            _waste_gate = _original_load_waste_gate()

        return _waste_gate


    def _cached_load_yolo():
        global _yolo_model

        if _yolo_model is None:
            print("Loading YOLO waste detector...")
            _yolo_model = _original_load_yolo()

        return _yolo_model


    def _cached_load_scene_model():
        global _scene_model

        if _scene_model is None:
            print("Loading scene analysis model...")
            _scene_model = _original_load_scene_model()

        return _scene_model


    _ai.load_waste_gate = _cached_load_waste_gate
    _ai.load_yolo = _cached_load_yolo
    _ai.load_scene_model = _cached_load_scene_model


# ============================================================
# PUBLIC API
# ============================================================

def analyze_image(image_bytes: bytes) -> dict:
    """
    Analyze an uploaded image using the SwachhLens AI pipeline.

    Pipeline:

        1. Waste Gate V2
        2. YOLO waste detection
        3. Scene Analysis ResNet18

    The AI models are loaded once and reused for
    subsequent requests.
    """

    if _ai is None:
        return {
            "valid": False,
            "reason": f"AI inference unavailable: {_IMPORT_ERROR}",
            "wasteType": None,
            "severity": None,
            "confidence": None,
            "engine": "ai",
            "details": [],
            "summary": None,
        }

    try:
        return _ai.analyze_image_bytes(image_bytes)

    except Exception as exc:
        return {
            "valid": False,
            "reason": f"AI inference failed: {exc}",
            "wasteType": None,
            "severity": None,
            "confidence": None,
            "engine": "ai",
            "details": [],
            "summary": None,
        }