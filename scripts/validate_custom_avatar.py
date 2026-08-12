#!/usr/bin/env python3
"""CPU-only preflight using the same MediaPipe detector as app.process_image."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app


def validate(image_path: Path, union_bbox_scale: float = 1.6) -> dict[str, int]:
    detector_module_path = Path(app.VIS_DIR) / "utils" / "face_detector.py"
    spec = importlib.util.spec_from_file_location(
        "custom_avatar_face_detector", detector_module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    detector = module.FaceDetector(
        mediapipe_model_asset_path=str(
            Path(app.VIS_DIR) / "utils" / "face_landmarker.task"
        ),
        delegate=0,
        face_detection_confidence=0.5,
        num_faces=1,
    )
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    cfg_path = Path(app.VIS_DIR) / "configs" / "audio_head_animator.yaml"
    cfg = app.OmegaConf.load(str(cfg_path))
    first = detector.get_face_xy_rotation_and_keypoints(
        image_np,
        cfg.data.mouth_bbox_scale,
        cfg.data.eye_bbox_scale,
    )
    if not first or len(first[6]) != 1:
        raise ValueError("the image must contain one clear detectable face")
    face_bbox = first[5][0]
    x1, y1 = face_bbox[0]
    x2, y2 = face_bbox[1]
    center = [int((y1 + y2) // 2), int((x1 + x2) // 2)]
    crop_size = int(max(x2 - x1, y2 - y1) * union_bbox_scale)
    if crop_size < 64:
        raise ValueError("the detected face is too small")
    crop_bbox = app.generate_crop_bounding_box(
        image_np.shape[0], image_np.shape[1], center, crop_size
    )
    cropped = app.crop_from_bbox(
        image_np, center, crop_bbox, size=crop_size
    )
    second = detector.get_face_xy_rotation_and_keypoints(
        cropped,
        cfg.data.mouth_bbox_scale,
        cfg.data.eye_bbox_scale,
    )
    if not second or len(second[6]) != 1:
        raise ValueError("face detection failed after the official 1.6x crop")
    return {
        "source_width": image.width,
        "source_height": image.height,
        "official_crop_size": crop_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.image.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
