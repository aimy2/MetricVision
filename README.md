# MetricVision — Camera-Calibrated Object Measurement

MetricVision is a computer vision project that estimates the physical width and height of a planar object from a photograph. It combines camera calibration, image undistortion, deep-learning-based segmentation, ArUco reference detection, and planar geometric measurement.

The system is designed to take a new photograph containing an object and a known-size reference marker, detect the object and reference, estimate dimensions in millimetres, and generate a visual measurement overlay.

## Project objectives

- Calibrate a smartphone camera using a checkerboard pattern.
- Correct lens distortion before measurement.
- Build a manually annotated image segmentation dataset.
- Train and compare segmentation architectures without using YOLO or Roboflow.
- Detect a known-size ArUco reference marker.
- Estimate object dimensions in millimetres.
- Evaluate segmentation and physical measurement performance.
- Compare measurement errors with and without camera undistortion.

## System workflow

1. Capture a photograph containing the object and reference marker.
2. Apply the camera calibration and undistortion transformation.
3. Run the trained segmentation model to identify the object.
4. Detect the ArUco reference marker.
5. Use the known marker dimensions and planar geometry to estimate object dimensions.
6. Generate an annotated output image containing the detected object and measurement information.

## Technologies

- Python
- OpenCV
- PyTorch
- NumPy
- Pandas
- Segmentation models: U-Net, DeepLabV3+, and Mask R-CNN
- ArUco marker detection

## Camera calibration

Camera calibration was performed using 20 checkerboard photographs.

| Parameter | Value |
|---|---:|
| Checkerboard squares | 10 × 7 |
| Internal corners | 9 × 6 |
| Square size | 20 mm |
| Calibration image size | 780 × 1040 |
| Number of calibration images | 20 |
| RMS reprojection error | 1.0532 px |
| Mean per-image reprojection error | 0.9823 px |
| Highest per-image reprojection error | 1.6316 px |

The calibration parameters are saved in:

- `calibration/camera_calibration.npz`
- `calibration/camera_calibration.yaml`

The calibration investigation scripts and undistortion previews are also included in the `calibration/` directory.

## Dataset

The dataset contains 70 photographs captured using the same smartphone camera.

Each photograph has a corresponding manually created polygon annotation and binary segmentation mask.

| Split | Images |
|---|---:|
| Training | 49 |
| Validation | 14 |
| Test | 7 |
| **Total** | **70** |

The split assignments are stored in `dataset/splits/`.

Dataset structure:

```text
dataset/
├── images/
├── annotations/
├── masks/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

The segmentation target is the card. The reference marker is excluded from the card segmentation mask.

The photographs show the same physical card under different positions, angles, and lighting conditions. Consequently, the dataset does not establish generalization to different object designs.

## Segmentation experiments

Three segmentation architectures were evaluated:

- U-Net with a ResNet18 encoder
- DeepLabV3+ with a ResNet18 encoder
- Mask R-CNN with a ResNet50-FPN backbone

The dataset uses a fixed 49/14/7 training, validation, and test split. Validation performance was used for checkpoint selection. The test split was reserved for final model evaluation.

### Validation results

| Model | Mean image IoU | Precision | Recall | Custom mask AP@0.5 | Custom mask AP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.9185 | 0.9630 | 0.9536 | 1.0000 | 0.8797 |
| DeepLabV3+ | 0.8814 | 0.9169 | 0.9590 | 1.0000 | 0.7982 |
| Mask R-CNN | 0.9244 | 0.9537 | 0.9683 | 1.0000 | 0.8973 |

### Test results

| Model | Mean image IoU | Precision | Recall | Custom mask AP@0.5 | Custom mask AP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.9416 | 0.9660 | 0.9749 | 1.0000 | 0.9241 |
| DeepLabV3+ | 0.8896 | 0.9073 | 0.9786 | 1.0000 | 0.8243 |
| Mask R-CNN | 0.9344 | 0.9583 | 0.9741 | 1.0000 | 0.9037 |

**Metric interpretation:** The AP values above use the project's custom single-instance mask evaluation procedure. They are not official COCO evaluation results and should not be interpreted as COCO mAP.

The U-Net checkpoint used for the measurement workflow was selected at epoch 12.

## Physical measurement evaluation

The physical measurement workflow was evaluated using 10 photographs of the same physical card.

Reference dimensions:

- Width: 85 mm
- Height: 125 mm

### Undistorted-image results

| Metric | Width | Height |
|---|---:|---:|
| Mean absolute error | 5.66 mm | 6.33 mm |
| Mean percentage error | 6.66% | 5.07% |

**Overall MAE:** 6.00 mm

**Overall mean percentage error:** 5.86%

### Raw versus undistorted comparison

| Measurement | Raw MAE | Undistorted MAE |
|---|---:|---:|
| Width | 5.98 mm | 5.66 mm |
| Height | 7.13 mm | 6.33 mm |
| Overall | 6.56 mm | 6.00 mm |

Undistortion reduced overall MAE by approximately 8.5% in this evaluation.

### Evaluation limitation

This is an exploratory physical measurement evaluation using existing dataset photographs, not an independent 10-image validation.

The 10 photographs comprise:

- 6 training images
- 2 validation images
- 2 test images

Therefore, most photographs were not independent of model training or checkpoint selection. These results should not be interpreted as an unbiased estimate of performance on unseen photographs.

The evaluation also covers one physical card rather than multiple object designs and dimensions.

Segmentation confidence is not calibrated dimensional accuracy. A high segmentation confidence does not guarantee an accurate physical measurement.

Detailed evaluation files are stored in `results/`.

## Reference marker

The measurement workflow uses an ArUco marker:

| Parameter | Value |
|---|---|
| Dictionary | DICT_6X6_250 |
| Marker ID | 23 |
| Black-square side length | 200 mm |

The marker's black square is the physical reference. Its surrounding white margin is not included in the reference measurement.

The marker must be fully visible and positioned in the same plane as the measured object.

## Running the project

### Environment

Use Python with the dependencies required by the project scripts.

Activate your environment and install the required packages:

```powershell
conda activate llasa
pip install -r requirements.txt
```

### Camera calibration

```powershell
python calibration/calibrate_camera.py
```

### Train segmentation models

```powershell
python src/experiment.py --epochs 13 --size 256
```

### Evaluate trained models on the test split

```powershell
python src/experiment.py --test
```

### Measure an object in a new photograph

```powershell
python src/measure.py --input "path\to\your\image.jpg" --model unet
```

The measurement script uses the calibration parameters, trained segmentation checkpoint, and ArUco reference marker to produce the estimated dimensions and an annotated output image.

The default reference configuration is marker ID 23 with a 200 mm black-square side length.

### Run physical measurement evaluation

Prepare a CSV containing image paths and reference dimensions:

```csv
image,ground_truth_width_mm,ground_truth_height_mm
../dataset/images/example.jpg,85,125
```

Then run:

```powershell
python src/measure.py --validate-csv results/ground_truth.csv --model unet --compare-distortion
```

The evaluation produces per-image measurements and summary error metrics.

## Repository structure

```text
MetricVision/
├── calibration/
│   ├── images/
│   ├── calibrate_camera.py
│   ├── preview_undistortion.py
│   ├── inspect_undistortion.py
│   ├── compare_distortion_models.py
│   └── camera_calibration.npz
├── dataset/
│   ├── images/
│   ├── annotations/
│   ├── masks/
│   ├── splits/
│   └── create_masks.py
├── docs/
│   ├── Plan
│   ├── Objectives
│   └── Definition of Done
├── results/
│   ├── training_history.csv
│   ├── validation_comparison.csv
│   ├── test_metrics.csv
│   ├── ground_truth.csv
│   ├── measurement_validation_unet.csv
│   └── measurement_summary_unet.csv
├── src/
│   ├── experiment.py
│   └── measure.py
├── weights/
│   ├── unet_best.pth
│   └── deeplabv3plus_best.pth
├── requirements.txt
└── README.md
```

## Limitations and future improvements

- Expand physical measurement testing to independent photographs.
- Evaluate additional object designs and dimensions.
- Improve measurement accuracy under varying camera angles and distances.
- Investigate the remaining error introduced by reference-marker detection and planar geometry.
- Establish a calibrated relationship between segmentation confidence and measurement reliability.
- Add a confidence-based refusal mechanism for unreliable detections or missing references.

## Reproducibility

The repository includes the calibration artifacts, dataset annotations, fixed split files, experiment scripts, model checkpoints, and measurement evaluation outputs needed to inspect the workflow.

All reported results should be interpreted in the context of the dataset size, evaluation protocol, and limitations described above.