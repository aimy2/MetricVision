"""
MetricVision: controlled segmentation experiment.

Models:
    1. U-Net (ResNet18 encoder)
    2. DeepLabV3+ (ResNet18 encoder)
    3. Mask R-CNN (ResNet50 FPN)

Protocol:
    - Fixed train/validation/test split: 49/14/7
    - 13 epochs per model by default
    - 256x256 input resolution by default
    - Camera undistortion applied to images AND masks
    - Validation IoU selects the best checkpoint
    - Test set is evaluated only with --test

Commands:
    python src/experiment.py
    python src/experiment.py --epochs 13 --size 256
    python src/experiment.py --test
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import segmentation_models_pytorch as smp


# ---------------------------------------------------------
# PATHS AND SETTINGS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "dataset"
SPLITS = DATA / "splits"

CALIBRATION_FILE = ROOT / "calibration" / "camera_calibration.npz"

WEIGHTS = ROOT / "weights"
RESULTS = ROOT / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42

EXPECTED_SPLITS = {
    "train": 49,
    "val": 14,
    "test": 7,
}

# ImageNet normalization for pretrained ResNet encoders.
IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------

def seed_everything(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# DATASET SPLITS
# ---------------------------------------------------------

def read_split(name):

    path = SPLITS / f"{name}.txt"

    if not path.exists():
        raise FileNotFoundError(path)

    entries = [
        x.strip()
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    expected = EXPECTED_SPLITS[name]

    if len(entries) != expected:
        raise ValueError(
            f"{name}: expected {expected} images, found {len(entries)}"
        )

    return entries


def resolve_image(name):

    candidates = [
        DATA / "images" / name,
        DATA / name,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(name)


# ---------------------------------------------------------
# CAMERA UNDISTORTION
# ---------------------------------------------------------

def create_undistortion_maps():

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_FILE}"
        )

    calibration = np.load(CALIBRATION_FILE)

    K = calibration["camera_matrix"].astype(np.float64)
    D = calibration["dist_coeffs"].astype(np.float64)

    # Calibration resolution: 780 x 1040.
    width = 780
    height = 1040

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        D,
        (width, height),
        1.0,
        (width, height),
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        K,
        D,
        None,
        new_K,
        (width, height),
        cv2.CV_16SC2,
    )

    return map1, map2, roi


# ---------------------------------------------------------
# IMAGE RESIZING
# ---------------------------------------------------------

def letterbox(image, mask, size):

    height, width = image.shape[:2]

    scale = min(
        size / width,
        size / height,
    )

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    mask = cv2.resize(
        mask,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

    image_canvas = np.zeros(
        (size, size, 3),
        dtype=np.uint8,
    )

    mask_canvas = np.zeros(
        (size, size),
        dtype=np.uint8,
    )

    x = (size - new_width) // 2
    y = (size - new_height) // 2

    image_canvas[
        y:y + new_height,
        x:x + new_width
    ] = image

    mask_canvas[
        y:y + new_height,
        x:x + new_width
    ] = mask

    return image_canvas, mask_canvas


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

class CardDataset(Dataset):

    def __init__(
        self,
        names,
        size=256,
        augment=False,
        undistortion_maps=None,
    ):

        self.names = names
        self.size = size
        self.augment = augment
        self.maps = undistortion_maps

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):

        name = self.names[index]

        image_path = resolve_image(name)

        mask_path = (
            DATA / "masks" /
            (Path(name).stem + ".png")
        )

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        # Decode JPEG at 1/4 resolution.
        # This avoids allocating a full-resolution RGB image.
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_REDUCED_COLOR_4,
        )

        # Also decode the mask at 1/4 resolution.
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_REDUCED_GRAYSCALE_4,
        )

        if image is None or mask is None:
            raise RuntimeError(
                f"Could not read image or mask: {name}"
            )

        if image.shape[:2] != mask.shape[:2]:

            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Normalize orientation to portrait.
        # Rotate the image and its mask together.
        if image.shape[0] < image.shape[1]:

            image = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )

            mask = cv2.rotate(
                mask,
                cv2.ROTATE_90_CLOCKWISE,
            )

        # Expected calibration-sized raster.
        if image.shape[:2] != (1040, 780):

            image = cv2.resize(
                image,
                (780, 1040),
                interpolation=cv2.INTER_AREA,
            )

            mask = cv2.resize(
                mask,
                (780, 1040),
                interpolation=cv2.INTER_NEAREST,
            )

        # Apply identical geometric remapping to image and mask.
        if self.maps is not None:

            map1, map2, _ = self.maps

            image = cv2.remap(
                image,
                map1,
                map2,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            mask = cv2.remap(
                mask,
                map1,
                map2,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        mask = (mask > 0).astype(np.uint8) * 255

        image, mask = letterbox(
            image,
            mask,
            self.size,
        )

        # Horizontal flip augmentation.
        if self.augment and random.random() < 0.5:

            image = np.ascontiguousarray(
                image[:, ::-1]
            )

            mask = np.ascontiguousarray(
                mask[:, ::-1]
            )

        # Mild brightness augmentation.
        if self.augment and random.random() < 0.35:

            factor = random.uniform(0.8, 1.2)

            image = np.clip(
                image.astype(np.float32) * factor,
                0,
                255,
            ).astype(np.uint8)

        x = torch.from_numpy(
            image.copy()
        ).permute(2, 0, 1).float() / 255.0

        y = torch.from_numpy(
            (mask > 0).astype(np.float32)
        )[None, :, :]

        return x, y, name


def collate_detection(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

def make_semantic(name, pretrained=True):

    encoder_weights = (
        "imagenet" if pretrained else None
    )

    if name == "unet":

        return smp.Unet(
            "resnet18",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )

    if name == "deeplabv3plus":

        return smp.DeepLabV3Plus(
            "resnet18",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )

    raise ValueError(name)


def make_maskrcnn(pretrained=True, size=256):

    weights = (
        MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        if pretrained else None
    )

    model = maskrcnn_resnet50_fpn(
        weights=weights
    )

    # Prevent Torchvision from resizing to 800 pixels.
    model.transform.min_size = (size,)
    model.transform.max_size = size

    # Two classes: background and card.
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


def normalize_semantic(x):

    return (
        x - IMAGENET_MEAN.to(x.device)
    ) / IMAGENET_STD.to(x.device)


# ---------------------------------------------------------
# MASK R-CNN TARGETS
# ---------------------------------------------------------

def detection_target(mask):

    ys, xs = torch.where(mask[0] > 0.5)

    if len(xs) == 0:
        raise ValueError("Empty training mask")

    box = torch.tensor(
        [[
            xs.min(),
            ys.min(),
            xs.max() + 1,
            ys.max() + 1,
        ]],
        dtype=torch.float32,
    )

    area = (
        (xs.max() - xs.min() + 1) *
        (ys.max() - ys.min() + 1)
    )

    return {
        "boxes": box,

        "labels": torch.ones(
            (1,),
            dtype=torch.int64,
        ),

        "masks": (mask > 0.5).to(torch.uint8),

        "image_id": torch.tensor([0]),

        "area": torch.tensor(
            [area],
            dtype=torch.float32,
        ),

        "iscrowd": torch.zeros(
            (1,),
            dtype=torch.int64,
        ),
    }


# ---------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------

def binary_stats(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    union = tp + fp + fn

    iou = tp / union if union else 1.0

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    return (
        float(iou),
        float(precision),
        float(recall),
    )


def mask_average_precision(records, threshold):

    """
    Image-level mask AP.

    Each image contains one ground-truth card and contributes
    one predicted mask. This is a custom single-instance AP
    protocol, not the official COCO evaluator.
    """

    if not records:
        return 0.0

    ranked = sorted(
        records,
        key=lambda x: x[0],
        reverse=True,
    )

    tp = []
    fp = []

    for confidence, iou in ranked:

        if iou >= threshold:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp, dtype=float)
    fp = np.cumsum(fp, dtype=float)

    recall = tp / len(records)

    precision = tp / np.maximum(
        tp + fp,
        1e-12,
    )

    # 101-point interpolated AP.
    recall_points = np.linspace(0, 1, 101)

    interpolated = []

    for r in recall_points:

        valid = precision[recall >= r]

        interpolated.append(
            valid.max() if len(valid) else 0.0
        )

    return float(np.mean(interpolated))


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

@torch.no_grad()
def predict_semantic(model, x):

    x = normalize_semantic(
        x.unsqueeze(0).to(DEVICE)
    )

    logits = model(x)

    probabilities = torch.sigmoid(
        logits
    )[0, 0]

    mask = probabilities >= 0.5

    if mask.any():
        confidence = float(
            probabilities[mask].mean().item()
        )
    else:
        confidence = float(
            probabilities.max().item()
        )

    return mask.cpu().numpy(), confidence


@torch.no_grad()
def predict_instance(model, x):

    model.eval()

    output = model([
        x.to(DEVICE)
    ])[0]

    if len(output["scores"]) == 0:

        return (
            np.zeros(
                x.shape[-2:],
                dtype=bool,
            ),
            0.0,
        )

    index = int(
        torch.argmax(output["scores"]).item()
    )

    confidence = float(
        output["scores"][index].item()
    )

    mask = (
        output["masks"][index, 0] >= 0.5
    )

    return mask.cpu().numpy(), confidence


# ---------------------------------------------------------
# VALIDATION / TEST EVALUATION
# ---------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, kind, split_name):

    model.eval()

    rows = []
    records = []

    for x, y, names in loader:

        x = x[0]

        gt = (
            y[0, 0].numpy() > 0.5
        )

        if kind == "maskrcnn":

            pred, confidence = predict_instance(
                model,
                x,
            )

        else:

            pred, confidence = predict_semantic(
                model,
                x,
            )

        iou, precision, recall = binary_stats(
            pred,
            gt,
        )

        rows.append(
            (iou, precision, recall)
        )

        records.append(
            (confidence, iou)
        )

    values = np.asarray(
        rows,
        dtype=float,
    )

    ap50 = mask_average_precision(
        records,
        0.50,
    )

    thresholds = np.arange(
        0.50,
        0.951,
        0.05,
    )

    aps = [
        mask_average_precision(records, t)
        for t in thresholds
    ]

    return {
        "split": split_name,
        "images": len(rows),

        "mean_image_iou": float(
            values[:, 0].mean()
        ),

        "pixel_precision": float(
            values[:, 1].mean()
        ),

        "pixel_recall": float(
            values[:, 2].mean()
        ),

        "custom_mask_AP50": ap50,

        "custom_mask_AP50_95": float(
            np.mean(aps)
        ),
    }


# ---------------------------------------------------------
# TRAIN SEMANTIC MODELS
# ---------------------------------------------------------

def train_semantic(
    name,
    train_loader,
    val_loader,
    epochs,
):

    model = make_semantic(
        name,
        pretrained=True,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    bce = nn.BCEWithLogitsLoss()

    dice = smp.losses.DiceLoss(
        mode="binary",
        from_logits=True,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=DEVICE.type == "cuda"
    )

    best_iou = -1.0

    history = []

    start = time.time()

    for epoch in range(1, epochs + 1):

        model.train()

        losses = []

        for x, y, names in train_loader:

            x = normalize_semantic(
                x.to(DEVICE)
            )

            y = y.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=DEVICE.type == "cuda"
            ):

                logits = model(x)

                loss = (
                    bce(logits, y) +
                    dice(logits, y)
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            losses.append(
                float(loss.detach().item())
            )

        metrics = evaluate(
            model,
            val_loader,
            "semantic",
            "val",
        )

        average_loss = float(
            np.mean(losses)
        )

        history.append({
            "model": name,
            "epoch": epoch,
            "train_loss": average_loss,
            **metrics,
        })

        print(
            f"{name} epoch {epoch:02d}/{epochs} "
            f"loss={average_loss:.4f} "
            f"val_IoU={metrics['mean_image_iou']:.4f}",
            flush=True,
        )

        if metrics["mean_image_iou"] > best_iou:

            best_iou = metrics["mean_image_iou"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": metrics,
                    "model_name": name,
                },
                WEIGHTS / f"{name}_best.pth",
            )

    elapsed = time.time() - start

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history, elapsed


# ---------------------------------------------------------
# TRAIN MASK R-CNN
# ---------------------------------------------------------

def train_maskrcnn(
    train_names,
    val_loader,
    epochs,
    size,
    maps,
):

    model = make_maskrcnn(
        pretrained=True,
        size=size,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        [
            p for p in model.parameters()
            if p.requires_grad
        ],
        lr=2e-4,
        weight_decay=1e-4,
    )

    dataset = CardDataset(
        train_names,
        size=size,
        augment=True,
        undistortion_maps=maps,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_detection,
    )

    best_iou = -1.0

    history = []

    start = time.time()

    for epoch in range(1, epochs + 1):

        model.train()

        losses = []

        for images, masks, names in loader:

            images = [
                image.to(DEVICE)
                for image in images
            ]

            targets = [
                detection_target(mask)
                for mask in masks
            ]

            targets = [
                {
                    key: value.to(DEVICE)
                    for key, value in target.items()
                }
                for target in targets
            ]

            loss_dict = model(
                images,
                targets,
            )

            loss = sum(
                loss_dict.values()
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            losses.append(
                float(loss.detach().item())
            )

        metrics = evaluate(
            model,
            val_loader,
            "maskrcnn",
            "val",
        )

        average_loss = float(
            np.mean(losses)
        )

        history.append({
            "model": "maskrcnn",
            "epoch": epoch,
            "train_loss": average_loss,
            **metrics,
        })

        print(
            f"maskrcnn epoch {epoch:02d}/{epochs} "
            f"loss={average_loss:.4f} "
            f"val_IoU={metrics['mean_image_iou']:.4f}",
            flush=True,
        )

        if metrics["mean_image_iou"] > best_iou:

            best_iou = metrics["mean_image_iou"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": metrics,
                    "model_name": "maskrcnn",
                },
                WEIGHTS / "maskrcnn_best.pth",
            )

    elapsed = time.time() - start

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history, elapsed


# ---------------------------------------------------------
# LOAD SAVED CHECKPOINTS
# ---------------------------------------------------------

def load_model(name, size):

    checkpoint_path = (
        WEIGHTS / f"{name}_best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    if name == "maskrcnn":

        model = make_maskrcnn(
            pretrained=False,
            size=size,
        )

    else:

        model = make_semantic(
            name,
            pretrained=False,
        )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(DEVICE)
    model.eval()

    return model, checkpoint


# ---------------------------------------------------------
# CSV OUTPUT
# ---------------------------------------------------------

def write_csv(path, rows):

    if not rows:
        return

    keys = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row.keys()
        )
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=13,
    )

    parser.add_argument(
        "--size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("Epochs must be positive")

    if args.size < 128:
        raise ValueError("Input size must be at least 128")

    seed_everything()

    WEIGHTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    train_names = read_split("train")
    val_names = read_split("val")
    test_names = read_split("test")

    # Verify split separation.
    if (
        set(train_names) & set(val_names)
        or set(train_names) & set(test_names)
        or set(val_names) & set(test_names)
    ):
        raise ValueError("Dataset splits overlap")

    print(
        f"Device: {DEVICE}; "
        f"train={len(train_names)}, "
        f"val={len(val_names)}, "
        f"test={len(test_names)}"
    )

    print("Input size:", args.size)

    maps = create_undistortion_maps()

    print(
        "Camera undistortion enabled."
    )

    print(
        "Calibration valid ROI:",
        maps[2],
    )

    # -----------------------------------------------------
    # TEST MODE
    # -----------------------------------------------------

    if args.test:

        test_loader = DataLoader(
            CardDataset(
                test_names,
                size=args.size,
                augment=False,
                undistortion_maps=maps,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        rows = []

        for name in (
            "unet",
            "deeplabv3plus",
            "maskrcnn",
        ):

            model, checkpoint = load_model(
                name,
                args.size,
            )

            kind = (
                "maskrcnn"
                if name == "maskrcnn"
                else "semantic"
            )

            metrics = evaluate(
                model,
                test_loader,
                kind,
                "test",
            )

            row = {
                "model": name,
                "selected_epoch": checkpoint["epoch"],
                **metrics,
            }

            rows.append(row)

            print(
                "\nTEST RESULT:",
                row,
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_csv(
            RESULTS / "test_metrics.csv",
            rows,
        )

        print(
            "\nSaved results/test_metrics.csv"
        )

        return

    # -----------------------------------------------------
    # TRAINING MODE
    # -----------------------------------------------------

    # Preserve old U-Net checkpoint.
    old_unet = WEIGHTS / "unet_best.pth"

    backup_unet = (
        WEIGHTS / "unet_best_preprocessing_v1.pth"
    )

    if old_unet.exists():

        if not backup_unet.exists():

            shutil.copy2(
                old_unet,
                backup_unet,
            )

        print(
            "Previous U-Net checkpoint preserved:",
            backup_unet,
        )

    train_loader = DataLoader(
        CardDataset(
            train_names,
            size=args.size,
            augment=True,
            undistortion_maps=maps,
        ),
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        CardDataset(
            val_names,
            size=args.size,
            augment=False,
            undistortion_maps=maps,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    all_history = []
    summaries = []

    # -----------------------------------------------------
    # U-NET: LOAD EXISTING BEST CHECKPOINT
    # -----------------------------------------------------

    print("\n=== Loading existing U-Net checkpoint ===")

    unet_model, unet_checkpoint = load_model(
        "unet",
        args.size,
    )

    unet_metrics = evaluate(
        unet_model,
        val_loader,
        "semantic",
        "val",
    )

    summaries.append({
        "model": "unet",
        "selected_epoch": unet_checkpoint["epoch"],
        "training_seconds": None,
        **unet_metrics,
    })

    print(
        "U-Net loaded from:",
        WEIGHTS / "unet_best.pth",
    )

    print(
        "U-Net selected epoch:",
        unet_checkpoint["epoch"],
    )

    print(
        "U-Net validation metrics:",
        unet_metrics,
    )

    del unet_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------
    # DEEPLABV3+
    # -----------------------------------------------------

    name = "deeplabv3plus"

    print(
        f"\n=== Training {name}: {args.epochs} epochs ==="
    )

    history, seconds = train_semantic(
        name,
        train_loader,
        val_loader,
        args.epochs,
    )

    all_history.extend(history)

    model, checkpoint = load_model(
        name,
        args.size,
    )

    metrics = evaluate(
        model,
        val_loader,
        "semantic",
        "val",
    )

    summaries.append({
        "model": name,
        "selected_epoch": checkpoint["epoch"],
        "training_seconds": round(seconds, 2),
        **metrics,
    })

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------
    # MASK R-CNN
    # -----------------------------------------------------

    print(
        f"\n=== Training maskrcnn: {args.epochs} epochs ==="
    )

    history, seconds = train_maskrcnn(
        train_names,
        val_loader,
        args.epochs,
        args.size,
        maps,
    )

    all_history.extend(history)

    model, checkpoint = load_model(
        "maskrcnn",
        args.size,
    )

    metrics = evaluate(
        model,
        val_loader,
        "maskrcnn",
        "val",
    )

    summaries.append({
        "model": "maskrcnn",
        "selected_epoch": checkpoint["epoch"],
        "training_seconds": round(seconds, 2),
        **metrics,
    })

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    write_csv(
        RESULTS / "training_history.csv",
        all_history,
    )

    write_csv(
        RESULTS / "validation_comparison.csv",
        summaries,
    )

    print("\n=== VALIDATION COMPARISON ===")

    for row in summaries:
        print(row)

    print("\nSaved:")
    print("results/training_history.csv")
    print("results/validation_comparison.csv")

    print("\nTest set has NOT been evaluated.")

    print("After reviewing validation results, run:")
    print("python src/experiment.py --test")


if __name__ == "__main__":
    main()