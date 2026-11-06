"""
MetricVision: inference, physical measurement and ruler validation.

Examples
--------
Single image:
    python src/measure.py --input "path/to/photo.jpg"

Choose a model:
    python src/measure.py --input "path/to/photo.jpg" --model maskrcnn

Validate using a CSV:
    python src/measure.py --validate-csv "measurements.csv"

Compare raw and undistorted measurements:
    python src/measure.py --validate-csv "measurements.csv" --compare-distortion

Expected validation CSV columns:
    image,ground_truth_width_mm,ground_truth_height_mm

Assumptions
-----------
1. The card and ArUco marker are flat and coplanar.
2. The printed black ArUco square has a measured side length of 200 mm.
3. The camera is the same phone camera and 1x lens used for calibration.
4. Images use the same camera orientation convention as the training images.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# =========================================================
# PROJECT SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_FILE = ROOT / "calibration" / "camera_calibration.npz"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
INFERENCE_DIR = RESULTS_DIR / "measurements"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calibration raster dimensions.
CALIBRATION_WIDTH = 780
CALIBRATION_HEIGHT = 1040

# Model input dimensions used during training.
MODEL_SIZE = 256

# Your printed marker's measured black-square side length.
DEFAULT_MARKER_SIZE_MM = 200.0

# Your printed marker ID.
DEFAULT_MARKER_ID = 23

# Refusal thresholds. These are operational safeguards, not
# statistically calibrated probabilities.
MIN_SEGMENTATION_CONFIDENCE = 0.50
MIN_CARD_AREA_PIXELS = 100

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(1, 3, 1, 1)


# =========================================================
# MODEL CONSTRUCTION
# =========================================================

def make_semantic_model(name: str):
    if name == "unet":
        return smp.Unet(
            "resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

    if name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            "resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

    raise ValueError(f"Unknown semantic model: {name}")


def make_maskrcnn():
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    model.transform.min_size = (MODEL_SIZE,)
    model.transform.max_size = MODEL_SIZE

    box_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        box_features,
        2,
    )

    mask_features = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    hidden = (
        model.roi_heads.mask_predictor.conv5_mask.out_channels
    )

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_features,
        hidden,
        2,
    )

    return model


def load_model(name: str):
    checkpoint_path = WEIGHTS_DIR / f"{name}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    if name == "maskrcnn":
        model = make_maskrcnn()
    else:
        model = make_semantic_model(name)

    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded {name}; selected epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    return model


# =========================================================
# CAMERA CALIBRATION
# =========================================================

def load_calibration():
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_FILE}"
        )

    calibration = np.load(CALIBRATION_FILE)

    required = {"camera_matrix", "dist_coeffs"}

    if not required.issubset(set(calibration.files)):
        raise ValueError(
            "Unexpected calibration file keys. Found: "
            + ", ".join(calibration.files)
        )

    camera_matrix = calibration["camera_matrix"].astype(
        np.float64
    )

    distortion = calibration["dist_coeffs"].astype(
        np.float64
    )

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (CALIBRATION_WIDTH, CALIBRATION_HEIGHT),
        1.0,
        (CALIBRATION_WIDTH, CALIBRATION_HEIGHT),
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
        (CALIBRATION_WIDTH, CALIBRATION_HEIGHT),
        cv2.CV_16SC2,
    )

    return {
        "camera_matrix": camera_matrix,
        "distortion": distortion,
        "new_camera_matrix": new_camera_matrix,
        "map1": map1,
        "map2": map2,
        "roi": roi,
    }


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def prepare_camera_image(image_path: Path):
    """
    Read a photograph and convert it to the 780x1040
    portrait raster used by the calibration maps.

    Landscape images are rotated clockwise, matching the
    preprocessing used in the training experiment.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    if image.shape[0] < image.shape[1]:
        image = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

    image = cv2.resize(
        image,
        (CALIBRATION_WIDTH, CALIBRATION_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    return image


def undistort_image(image, calibration):
    return cv2.remap(
        image,
        calibration["map1"],
        calibration["map2"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def letterbox_image(image):
    """
    Reproduce the experiment.py letterbox operation.

    Returns:
        model_image: 256x256 RGB image
        metadata: scale and padding needed to restore the mask
    """

    height, width = image.shape[:2]

    scale = min(
        MODEL_SIZE / width,
        MODEL_SIZE / height,
    )

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (MODEL_SIZE, MODEL_SIZE, 3),
        dtype=np.uint8,
    )

    x_offset = (MODEL_SIZE - new_width) // 2
    y_offset = (MODEL_SIZE - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    rgb = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB,
    )

    metadata = {
        "new_width": new_width,
        "new_height": new_height,
        "x_offset": x_offset,
        "y_offset": y_offset,
    }

    return rgb, metadata


def restore_mask_to_camera_resolution(mask, metadata):
    """
    Remove letterbox padding and restore a 780x1040 mask.
    """

    x = metadata["x_offset"]
    y = metadata["y_offset"]

    width = metadata["new_width"]
    height = metadata["new_height"]

    cropped = mask[
        y:y + height,
        x:x + width,
    ]

    restored = cv2.resize(
        cropped.astype(np.uint8),
        (CALIBRATION_WIDTH, CALIBRATION_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )

    return restored > 0


# =========================================================
# SEGMENTATION INFERENCE
# =========================================================

def semantic_prediction(model, rgb):
    x = torch.from_numpy(
        rgb.copy()
    ).permute(2, 0, 1).float() / 255.0

    x = x.unsqueeze(0).to(DEVICE)

    mean = IMAGENET_MEAN.to(DEVICE)
    std = IMAGENET_STD.to(DEVICE)

    x = (x - mean) / std

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits)[0, 0]

    probability_map = probabilities.cpu().numpy()
    mask = probability_map >= 0.5

    if mask.any():
        confidence = float(
            probability_map[mask].mean()
        )
    else:
        confidence = float(
            probability_map.max()
        )

    return mask, confidence


def instance_prediction(model, rgb):
    x = torch.from_numpy(
        rgb.copy()
    ).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        output = model([
            x.to(DEVICE)
        ])[0]

    if len(output["scores"]) == 0:
        return (
            np.zeros(
                (MODEL_SIZE, MODEL_SIZE),
                dtype=bool,
            ),
            0.0,
        )

    # Select the highest-confidence predicted instance.
    index = int(
        torch.argmax(output["scores"]).item()
    )

    confidence = float(
        output["scores"][index].item()
    )

    mask = (
        output["masks"][index, 0].cpu().numpy()
        >= 0.5
    )

    return mask, confidence


def segment_card(model, model_name, camera_image):
    model_rgb, metadata = letterbox_image(camera_image)

    if model_name == "maskrcnn":
        small_mask, confidence = instance_prediction(
            model,
            model_rgb,
        )
    else:
        small_mask, confidence = semantic_prediction(
            model,
            model_rgb,
        )

    full_mask = restore_mask_to_camera_resolution(
        small_mask,
        metadata,
    )

    return full_mask, confidence


# =========================================================
# ARUCO REFERENCE DETECTION
# =========================================================

def create_aruco_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV ArUco module unavailable. "
            "Install opencv-contrib-python."
        )

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_6X6_250
    )

    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters,
        )

        return detector, dictionary, parameters

    return None, dictionary, parameters


def detect_reference_marker(image, marker_id):
    detector, dictionary, parameters = create_aruco_detector()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )

    if ids is None:
        raise RuntimeError(
            "REFERENCE_NOT_FOUND: No ArUco marker detected."
        )

    ids = ids.flatten()

    matches = np.where(ids == marker_id)[0]

    if len(matches) == 0:
        raise RuntimeError(
            f"REFERENCE_NOT_FOUND: ArUco ID {marker_id} "
            "was not detected."
        )

    marker_corners = corners[int(matches[0])].reshape(
        4, 2
    ).astype(np.float32)

    return marker_corners


# =========================================================
# CARD CONTOUR AND GEOMETRY
# =========================================================

def find_card_contour(mask):
    binary = (
        mask.astype(np.uint8) * 255
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError(
            "OBJECT_NOT_FOUND: No card segmentation found."
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    area = cv2.contourArea(contour)

    if area < MIN_CARD_AREA_PIXELS:
        raise RuntimeError(
            "OBJECT_NOT_FOUND: Segmented region is too small."
        )

    return contour


def marker_homography(marker_corners, marker_size_mm):
    """
    Map image coordinates to a planar coordinate system
    measured in millimetres.

    ArUco corner order:
        top-left, top-right, bottom-right, bottom-left

    The coordinate origin is the first detected marker corner.
    """

    physical_corners = np.array(
        [
            [0.0, 0.0],
            [marker_size_mm, 0.0],
            [marker_size_mm, marker_size_mm],
            [0.0, marker_size_mm],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(
        marker_corners.astype(np.float32),
        physical_corners,
    )

    if not np.isfinite(H).all():
        raise RuntimeError(
            "Could not calculate a valid perspective transform."
        )

    return H


def measure_card(contour, homography):
    """
    Fit a minimum-area rectangle to the segmented card contour,
    transform its corners into the marker plane, and calculate
    the physical edge lengths in millimetres.
    """

    rectangle = cv2.minAreaRect(contour)

    image_corners = cv2.boxPoints(
        rectangle
    ).astype(np.float32)

    transformed = cv2.perspectiveTransform(
        image_corners.reshape(1, -1, 2),
        homography,
    )[0]

    edge_lengths = []

    for i in range(4):
        p1 = transformed[i]
        p2 = transformed[(i + 1) % 4]

        distance = float(
            np.linalg.norm(p2 - p1)
        )

        edge_lengths.append(distance)

    # Opposite sides are averaged to reduce small contour errors.
    side_a = (
        edge_lengths[0] + edge_lengths[2]
    ) / 2.0

    side_b = (
        edge_lengths[1] + edge_lengths[3]
    ) / 2.0

    width = min(side_a, side_b)
    height = max(side_a, side_b)

    return {
        "width_mm": width,
        "height_mm": height,
        "image_corners": image_corners,
        "physical_corners": transformed,
        "edge_lengths_mm": edge_lengths,
    }


# =========================================================
# OVERLAY
# =========================================================

def create_overlay(
    image,
    contour,
    marker_corners,
    measurement,
    confidence,
    output_path,
):
    overlay = image.copy()

    cv2.drawContours(
        overlay,
        [contour],
        -1,
        (0, 220, 0),
        2,
    )

    cv2.polylines(
        overlay,
        [marker_corners.astype(np.int32)],
        True,
        (255, 150, 0),
        2,
    )

    corners = measurement["image_corners"].astype(
        np.int32
    )

    cv2.polylines(
        overlay,
        [corners],
        True,
        (0, 0, 255),
        2,
    )

    text_lines = [
        f"Width: {measurement['width_mm']:.1f} mm",
        f"Height: {measurement['height_mm']:.1f} mm",
        f"Segmentation confidence: {confidence:.3f}",
    ]

    y = 35

    for line in text_lines:
        cv2.putText(
            overlay,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            overlay,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 30

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        overlay,
    ):
        raise RuntimeError(
            f"Could not save overlay: {output_path}"
        )


# =========================================================
# COMPLETE MEASUREMENT PIPELINE
# =========================================================

def measure_image(
    image_path,
    model,
    model_name,
    calibration,
    marker_size_mm=DEFAULT_MARKER_SIZE_MM,
    marker_id=DEFAULT_MARKER_ID,
    use_undistortion=True,
    save_overlay=True,
):
    image_path = Path(image_path)

    raw_image = prepare_camera_image(
        image_path
    )

    if use_undistortion:
        processing_image = undistort_image(
            raw_image,
            calibration,
        )
    else:
        processing_image = raw_image.copy()

    # Segment the card.
    mask, confidence = segment_card(
        model,
        model_name,
        processing_image,
    )

    if confidence < MIN_SEGMENTATION_CONFIDENCE:
        raise RuntimeError(
            "LOW_CONFIDENCE: Segmentation confidence "
            f"{confidence:.3f} is below "
            f"{MIN_SEGMENTATION_CONFIDENCE:.2f}."
        )

    contour = find_card_contour(mask)

    # Detect the known physical reference.
    marker_corners = detect_reference_marker(
        processing_image,
        marker_id,
    )

    # Convert image coordinates to physical plane coordinates.
    H = marker_homography(
        marker_corners,
        marker_size_mm,
    )

    measurement = measure_card(
        contour,
        H,
    )

    if not all(
        math.isfinite(measurement[k])
        for k in ("width_mm", "height_mm")
    ):
        raise RuntimeError(
            "Invalid measurement produced."
        )

    if (
        measurement["width_mm"] <= 0
        or measurement["height_mm"] <= 0
    ):
        raise RuntimeError(
            "Invalid non-positive measurement produced."
        )

    output_path = (
        INFERENCE_DIR /
        f"{image_path.stem}_{model_name}_measurement.jpg"
    )

    if save_overlay:
        create_overlay(
            processing_image,
            contour,
            marker_corners,
            measurement,
            confidence,
            output_path,
        )

    result = {
        "image": image_path.name,
        "model": model_name,
        "width_mm": measurement["width_mm"],
        "height_mm": measurement["height_mm"],
        "confidence": confidence,
        "undistortion": use_undistortion,
        "overlay": str(output_path) if save_overlay else "",
    }

    return result


# =========================================================
# VALIDATION METRICS
# =========================================================

def absolute_error(predicted, actual):
    return abs(predicted - actual)


def percentage_error(predicted, actual):
    if actual == 0:
        return float("nan")

    return (
        abs(predicted - actual) / abs(actual)
    ) * 100.0


def validate_csv(
    csv_path,
    model,
    model_name,
    calibration,
    marker_size_mm,
    marker_id,
    compare_distortion=False,
):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    required_columns = {
        "image",
        "ground_truth_width_mm",
        "ground_truth_height_mm",
    }

    if not rows:
        raise ValueError(
            "Validation CSV contains no rows."
        )

    if not required_columns.issubset(
        set(rows[0].keys())
    ):
        raise ValueError(
            "CSV must contain columns: "
            "image, ground_truth_width_mm, "
            "ground_truth_height_mm"
        )

    if len(rows) < 10:
        raise ValueError(
            f"At least 10 validation instances are required; "
            f"found {len(rows)}."
        )

    results = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image"])

        if not image_path.is_absolute():
            image_path = csv_path.parent / image_path

        gt_width = float(
            row["ground_truth_width_mm"]
        )

        gt_height = float(
            row["ground_truth_height_mm"]
        )

        if gt_width <= 0 or gt_height <= 0:
            raise ValueError(
                f"Invalid ground truth in row {index}."
            )

        print(
            f"\nValidating {index}/{len(rows)}: "
            f"{image_path.name}"
        )

        # Standard pipeline: undistortion enabled.
        prediction = measure_image(
            image_path,
            model,
            model_name,
            calibration,
            marker_size_mm,
            marker_id,
            use_undistortion=True,
            save_overlay=True,
        )

        predicted_width = prediction["width_mm"]
        predicted_height = prediction["height_mm"]

        width_error = absolute_error(
            predicted_width,
            gt_width,
        )

        height_error = absolute_error(
            predicted_height,
            gt_height,
        )

        width_percentage = percentage_error(
            predicted_width,
            gt_width,
        )

        height_percentage = percentage_error(
            predicted_height,
            gt_height,
        )

        result = {
            "image": image_path.name,
            "ground_truth_width_mm": gt_width,
            "predicted_width_mm": predicted_width,
            "width_absolute_error_mm": width_error,
            "width_percentage_error": width_percentage,
            "ground_truth_height_mm": gt_height,
            "predicted_height_mm": predicted_height,
            "height_absolute_error_mm": height_error,
            "height_percentage_error": height_percentage,
            "confidence": prediction["confidence"],
        }

        if compare_distortion:
            raw_prediction = measure_image(
                image_path,
                model,
                model_name,
                calibration,
                marker_size_mm,
                marker_id,
                use_undistortion=False,
                save_overlay=False,
            )

            raw_width_error = absolute_error(
                raw_prediction["width_mm"],
                gt_width,
            )

            raw_height_error = absolute_error(
                raw_prediction["height_mm"],
                gt_height,
            )

            result.update({
                "raw_width_mm": raw_prediction["width_mm"],
                "raw_height_mm": raw_prediction["height_mm"],
                "raw_width_error_mm": raw_width_error,
                "raw_height_error_mm": raw_height_error,
            })

        results.append(result)

        print(
            f"  Predicted: "
            f"{predicted_width:.2f} x "
            f"{predicted_height:.2f} mm"
        )

        print(
            f"  Ground truth: "
            f"{gt_width:.2f} x "
            f"{gt_height:.2f} mm"
        )

    # Aggregate width and height errors separately.
    width_mae = float(np.mean([
        r["width_absolute_error_mm"]
        for r in results
    ]))

    height_mae = float(np.mean([
        r["height_absolute_error_mm"]
        for r in results
    ]))

    width_mpe = float(np.mean([
        r["width_percentage_error"]
        for r in results
    ]))

    height_mpe = float(np.mean([
        r["height_percentage_error"]
        for r in results
    ]))

    # Overall average across width and height measurements.
    all_absolute_errors = [
        r["width_absolute_error_mm"]
        for r in results
    ] + [
        r["height_absolute_error_mm"]
        for r in results
    ]

    all_percentage_errors = [
        r["width_percentage_error"]
        for r in results
    ] + [
        r["height_percentage_error"]
        for r in results
    ]

    summary = {
        "model": model_name,
        "instances": len(results),
        "width_MAE_mm": width_mae,
        "height_MAE_mm": height_mae,
        "overall_MAE_mm": float(
            np.mean(all_absolute_errors)
        ),
        "width_mean_percentage_error": width_mpe,
        "height_mean_percentage_error": height_mpe,
        "overall_mean_percentage_error": float(
            np.mean(all_percentage_errors)
        ),
    }

    if compare_distortion:
        summary.update({
            "raw_width_MAE_mm": float(np.mean([
                r["raw_width_error_mm"]
                for r in results
            ])),
            "raw_height_MAE_mm": float(np.mean([
                r["raw_height_error_mm"]
                for r in results
            ])),
            "undistorted_width_MAE_mm": width_mae,
            "undistorted_height_MAE_mm": height_mae,
        })

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        RESULTS_DIR /
        f"measurement_validation_{model_name}.csv"
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    summary_path = (
        RESULTS_DIR /
        f"measurement_summary_{model_name}.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary.keys()),
        )

        writer.writeheader()
        writer.writerow(summary)

    print("\n========== VALIDATION SUMMARY ==========")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nPer-instance results:", output_csv)
    print("Summary:", summary_path)

    if compare_distortion:
        print(
            "\nRaw versus undistorted errors are saved "
            "in the per-instance CSV."
        )

    return results, summary


# =========================================================
# COMMAND LINE
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Segment a card, detect its ArUco reference, "
            "and measure its dimensions in millimetres."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to one photograph.",
    )

    parser.add_argument(
        "--model",
        choices=[
            "unet",
            "deeplabv3plus",
            "maskrcnn",
        ],
        default="unet",
    )

    parser.add_argument(
        "--marker-size",
        type=float,
        default=DEFAULT_MARKER_SIZE_MM,
        help="Measured black ArUco square side length in mm.",
    )

    parser.add_argument(
        "--marker-id",
        type=int,
        default=DEFAULT_MARKER_ID,
    )

    parser.add_argument(
        "--validate-csv",
        type=str,
        help="CSV containing 10+ ruler-measured images.",
    )

    parser.add_argument(
        "--compare-distortion",
        action="store_true",
        help="Compare raw and undistorted measurement errors.",
    )

    args = parser.parse_args()

    if not args.input and not args.validate_csv:
        parser.error(
            "Provide --input or --validate-csv."
        )

    if args.marker_size <= 0:
        parser.error(
            "--marker-size must be positive."
        )

    print("Device:", DEVICE)
    print("Model:", args.model)
    print("Marker ID:", args.marker_id)
    print("Marker black-square size:", args.marker_size, "mm")

    calibration = load_calibration()

    print(
        "Calibration valid ROI:",
        calibration["roi"],
    )

    model = load_model(args.model)

    INFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.validate_csv:
        validate_csv(
            args.validate_csv,
            model,
            args.model,
            calibration,
            args.marker_size,
            args.marker_id,
            compare_distortion=args.compare_distortion,
        )

    if args.input:
        result = measure_image(
            args.input,
            model,
            args.model,
            calibration,
            args.marker_size,
            args.marker_id,
            use_undistortion=True,
            save_overlay=True,
        )

        print("\n========== MEASUREMENT ==========")
        print(f"Width: {result['width_mm']:.2f} mm")
        print(f"Height: {result['height_mm']:.2f} mm")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Overlay:", result["overlay"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)