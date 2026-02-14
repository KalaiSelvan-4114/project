import base64
import io
import os
import re
from typing import Tuple, Dict, Any

import cv2
import gdown
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms


def _extract_drive_file_id(value: str) -> str:
    """Extract Google Drive file ID from a raw ID or full share URL."""
    if not value or not value.strip():
        return ""
    value = value.strip()
    # Already a short hex-like ID (e.g. 1DbMi52hwTyiSOYB2ERS4sL5SjHzEyOXJ)
    if re.match(r"^[\w\-]{20,}$", value) and "/" not in value:
        return value
    # Full URL: .../d/FILE_ID/view... or ...?id=FILE_ID
    m = re.search(r"/d/([\w\-]+)(?:/view|$)", value)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([\w\-]+)", value)
    if m:
        return m.group(1)
    return value


# ================= CONFIG =================
_raw_model_path = os.environ.get("MODEL_PATH", "efficientnet_b3_final.pth").strip()
# If MODEL_PATH was set to a Google Drive URL by mistake, use it as drive ID and default local path
if _raw_model_path.startswith("http") and "drive.google.com" in _raw_model_path:
    GOOGLE_DRIVE_FILE_ID = _extract_drive_file_id(_raw_model_path)
    MODEL_PATH = "efficientnet_b3_final.pth"
else:
    MODEL_PATH = _raw_model_path
    _drive_id_env = os.environ.get("GOOGLE_DRIVE_FILE_ID", "").strip()
    GOOGLE_DRIVE_FILE_ID = _extract_drive_file_id(_drive_id_env) if _drive_id_env else ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))

# IMPORTANT: keep this order exactly the same as used during training
# (ImageFolder usually orders classes alphabetically by folder name).
# New model: 5 disease classes only. If none is detected with high
# confidence, we treat the skin as healthy.
CLASS_NAMES = [
    "Acne",
    "Moles",
    "Psoriasis",
    "SkinCancer",
    "Vitiligo",
]

# If the highest softmax probability is below this threshold,
# we will treat the image as healthy / no disease.
HEALTHY_CONFIDENCE_THRESHOLD = float(
    os.environ.get("HEALTHY_CONFIDENCE_THRESHOLD", "0.60")
)


# ================= MODEL & TRANSFORMS =================
def get_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_model(num_classes: int) -> nn.Module:
    """
    Build EfficientNet-B3 model with the same classifier head used for training.
    The checkpoint in 'efficientnet_skin_disease.pth' was saved from a B3 backbone,
    which has 40 channels in the first conv layer and 1536 features before the
    classifier, matching the error log.
    """
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def ensure_model_downloaded(model_path: str) -> None:
    """Download model from Google Drive if file does not exist and GOOGLE_DRIVE_FILE_ID is set."""
    if os.path.isfile(model_path):
        return
    if not GOOGLE_DRIVE_FILE_ID:
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Either place the model file there, or set GOOGLE_DRIVE_FILE_ID to your Google Drive file ID."
        )
    # Download from Google Drive (e.g. share link: https://drive.google.com/file/d/FILE_ID/view)
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, model_path, quiet=False)


def load_model(model_path: str, device: str) -> nn.Module:
    ensure_model_downloaded(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}' after download attempt. "
            "Check GOOGLE_DRIVE_FILE_ID and that the file is shared as 'Anyone with the link can view'."
        )

    model = build_model(len(CLASS_NAMES))
    state = torch.load(model_path, map_location=device)
    # Support both pure state_dict and checkpoint dict formats
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


model = load_model(MODEL_PATH, DEVICE)
transform = get_test_transform()


# ================= IMAGE UTILITIES =================
def pil_from_upload(file_storage) -> Image.Image:
    img_bytes = file_storage.read()
    if not img_bytes:
        raise ValueError("Empty image file.")
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return pil_img


def adjust_resolution(
    pil_img: Image.Image,
    min_side: int = IMG_SIZE,
    max_side: int = 1024,
) -> Tuple[Image.Image, float]:
    """
    Scale image up if too small or down if extremely large so that
    the shortest side is at least `min_side` and the longest side
    is at most `max_side`. Returns (resized_image, scale_factor).
    """
    w, h = pil_img.size
    scale = 1.0

    short_side = min(w, h)
    long_side = max(w, h)

    if short_side < min_side:
        scale = min_side / float(short_side)
    elif long_side > max_side:
        scale = max_side / float(long_side)

    if abs(scale - 1.0) < 1e-3:
        return pil_img, 1.0

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
    return resized, scale


def is_skin_image(pil_img: Image.Image, min_ratio: float = 0.15) -> bool:
    """
    Very lightweight skin detection using YCrCb thresholds.
    This is not perfect, but enough to catch clearly non-skin images.
    """
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(img_ycrcb, lower, upper)
    skin_ratio = mask.mean() / 255.0
    return skin_ratio >= min_ratio


def tensor_from_pil(pil_img: Image.Image, device: str) -> torch.Tensor:
    t = transform(pil_img)
    return t.unsqueeze(0).to(device)


def segment_and_mark_disease(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV-based segmentation and disease highlighting that mirrors
    the standalone EfficientNet-B0 script you provided.
    Works on an RGB image of size (IMG_SIZE, IMG_SIZE).
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Gray conversion
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Segmentation (disease only) using Otsu threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    segmented_mask = np.zeros_like(mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(segmented_mask, [largest], -1, 255, -1)

    segmented_img = cv2.bitwise_and(
        img_rgb, img_rgb, mask=segmented_mask
    )

    # Disease area (circles on original)
    disease_marked = img_rgb.copy()

    for cnt in contours or []:
        if cv2.contourArea(cnt) > 300:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(
                disease_marked,
                (int(x), int(y)),
                int(radius),
                (0, 255, 0),
                2,
            )

    return segmented_img, disease_marked


def image_to_data_uri(img_rgb: np.ndarray) -> str:
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ================= FLASK APP =================
app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> Tuple[Any, int]:
    if "file" not in request.files:
        return (
            jsonify(
                {
                    "status": "error",
                    "error_code": "NO_FILE",
                    "message": "Please upload an image file with key 'file'.",
                }
            ),
            400,
        )

    file = request.files["file"]
    if file.filename == "":
        return (
            jsonify(
                {
                    "status": "error",
                    "error_code": "EMPTY_FILENAME",
                    "message": "No file selected. Please choose a skin image.",
                }
            ),
            400,
        )

    try:
        pil_img = pil_from_upload(file)
    except Exception:
        return (
            jsonify(
                {
                    "status": "error",
                    "error_code": "INVALID_IMAGE",
                    "message": "Unable to read image. Please upload a valid skin image.",
                }
            ),
            400,
        )

    original_size = pil_img.size

    # Skin content check
    if not is_skin_image(pil_img):
        return (
            jsonify(
                {
                    "status": "error",
                    "error_code": "NON_SKIN_IMAGE",
                    "message": "The uploaded image does not appear to be skin. "
                    "Please upload a clear photo of the affected skin area.",
                }
            ),
            400,
        )

    # Model inference (using the same preprocessing pipeline
    # as your standalone EfficientNet-B0 script)
    try:
        # Resize for model and OpenCV processing
        pil_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
        img_rgb = np.array(pil_resized)

        input_tensor = transform(pil_resized).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_idx = int(torch.argmax(probs).item())
        top_prob = float(probs[top_idx].item())

        # If the model is not confident enough about any disease class,
        # consider the skin healthy and skip Grad‑CAM.
        if top_prob < HEALTHY_CONFIDENCE_THRESHOLD:
            return (
                jsonify(
                    {
                        "status": "ok",
                        "is_healthy": True,
                        "prediction": "Healthy",
                        "confidence": round(top_prob * 100, 2),
                        "message": "Your skin appears healthy.",
                        "original_size": list(original_size),
                        "processed_size": list(pil_img.size),
                    }
                ),
                200,
            )

        predicted_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)

        # Disease detected: OpenCV-based segmentation and disease highlighting
        segmented_img, disease_marked = segment_and_mark_disease(img_rgb)
        seg_uri = image_to_data_uri(segmented_img)
        overlay_uri = image_to_data_uri(disease_marked)

        return (
            jsonify(
                {
                    "status": "ok",
                    "is_healthy": False,
                    "prediction": predicted_label,
                    "confidence": round(top_prob * 100, 2),
                    "original_image": image_to_data_uri(np.array(pil_img)),
                    "segmentation_image": seg_uri,
                    "disease_overlay_image": overlay_uri,
                    "original_size": list(original_size),
                    "processed_size": list(pil_img.size),
                    "message": "Disease detected. Highlighted areas show the affected skin.",
                }
            ),
            200,
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "error_code": "SERVER_ERROR",
                    "message": "An error occurred while processing the image.",
                    "details": str(e),
                }
            ),
            500,
        )


if __name__ == "__main__":
    # For local development
    app.run(host="0.0.0.0", port=5000, debug=False)

