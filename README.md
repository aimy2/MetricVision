# MetricVision — Camera-Based Object Measurement

MetricVision is a computer vision system that measures an object's width and height in millimetres from a photograph.

It uses camera calibration, image undistortion, deep-learning segmentation, and an ArUco marker of known size to estimate physical dimensions.

## 1. How It Works

1. Capture an image containing the object and reference marker.
2. Correct camera distortion using calibration parameters.
3. Segment the object using a trained deep-learning model.
4. Detect the ArUco reference marker.
5. Use the marker's known dimensions and perspective geometry to calculate object dimensions.
6. Save an image showing the detected object and estimated measurements.

## 2. Technologies

- Python
- OpenCV and ArUco
- PyTorch and Torchvision
- NumPy
- Segmentation Models PyTorch

## 3. Camera Calibration

Calibration used 20 checkerboard photographs captured with the same smartphone camera.

| Parameter | Result |
|---|---:|
| Checkerboard | 10 × 7 squares |
| Internal corners | 9 × 6 |
| Square size | 20 mm |
| Image resolution | 780 × 1040 |
| Calibration images | 20 |
| RMS reprojection error | 1.0532 px |
| Mean reprojection error | 0.9823 px |

Calibration files:

- `calibration/camera_calibration.npz`
- `calibration/camera_calibration.yaml`

## 4. Dataset

The dataset contains 70 original photographs with manually created polygon annotations and binary segmentation masks.

| Split | Images |
|---|---:|
| Training | 49 |
| Validation | 14 |
| Test | 7 |
| **Total** | **70** |

The fixed split lists are stored in `dataset/splits/`.

All photographs show the same physical card under different positions, angles, and lighting conditions. Generalization to different object designs has not been established.

## 5. Segmentation Models

Three architectures were trained and evaluated:

- U-Net (ResNet18 encoder)
- DeepLabV3+ (ResNet18 encoder)
- Mask R-CNN (ResNet50-FPN)

### Validation results

| Model | Mean IoU | Precision | Recall | Custom AP@0.5 | Custom AP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.9185 | 0.9630 | 0.9536 | 1.0000 | 0.8797 |
| DeepLabV3+ | 0.8814 | 0.9169 | 0.9590 | 1.0000 | 0.7982 |
| Mask R-CNN | 0.9244 | 0.9537 | 0.9683 | 1.0000 | 0.8973 |

### Test results

| Model | Mean IoU | Precision | Recall | Custom AP@0.5 | Custom AP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.9416 | 0.9660 | 0.9749 | 1.0000 | 0.9241 |
| DeepLabV3+ | 0.8896 | 0.9073 | 0.9786 | 1.0000 | 0.8243 |
| Mask R-CNN | 0.9344 | 0.9583 | 0.9741 | 1.0000 | 0.9037 |

**Note:** AP values use a custom single-instance mask evaluation procedure. They are not official COCO mAP results.

U-Net is used in the measurement pipeline. Mask R-CNN achieved the highest validation IoU and custom AP@0.5:0.95.

## 6. Physical Measurement

The measurement pipeline was evaluated on 10 photographs of the same physical card.

Reference dimensions:

- Width: 85 mm
- Height: 125 mm

### Measurement results

| Metric | Width | Height |
|---|---:|---:|
| Mean absolute error | 5.66 mm | 6.33 mm |
| Mean percentage error | 6.66% | 5.07% |

**Overall MAE:** 6.00 mm

**Overall mean percentage error:** 5.86%

### Effect of undistortion

| Measurement | Raw image MAE | Undistorted image MAE |
|---|---:|---:|
| Width | 5.98 mm | 5.66 mm |
| Height | 7.13 mm | 6.33 mm |
| Overall | 6.56 mm | 6.00 mm |

Undistortion reduced overall MAE by approximately 8.5% in this evaluation.

**Evaluation limitation:** The 10 photographs include 6 training, 2 validation, and 2 test images. This is exploratory evidence, not an independent 10-image test. Results cover one physical card only.

Detailed results are stored in:

- `results/ground_truth.csv`
- `results/measurement_validation_unet.csv`
- `results/measurement_summary_unet.csv`

## 7. Reference Marker

The system uses an ArUco marker:

| Parameter | Value |
|---|---|
| Dictionary | DICT_6X6_250 |
| Marker ID | 23 |
| Black-square side length | 200 mm |

The measured reference is the black square, excluding its white margin. The marker and object must be flat and coplanar.

## 8. Failure Handling

The measurement pipeline refuses to produce a measurement when:

- No card segmentation is found.
- The segmented region is too small.
- The required ArUco marker is missing or its ID does not match.
- Segmentation confidence is below 0.50.
- The resulting dimensions are invalid.

The 0.50 threshold is an operational safeguard, not a calibrated probability of measurement accuracy.

## 9. Installation

Python 3.10 is recommended.

Create an environment:

```powershell
conda create -n metricvision python=3.10 -y
conda activate metricvision
```

Install PyTorch with CUDA 12.1 support:

```powershell
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 10. Running the Project

Run commands from the repository root.

**Camera calibration**

```powershell
python calibration/calibrate_camera.py
```

**Train segmentation models**

```powershell
python src/experiment.py --epochs 13 --size 256
```

**Evaluate test split**

```powershell
python src/experiment.py --test
```

**Measure a new photograph**

```powershell
python src/measure.py --input "path\to\image.jpg" --model unet
```

**Run measurement validation**

```powershell
python src/measure.py --validate-csv results/ground_truth.csv --model unet --compare-distortion
```

The measurement command outputs estimated width, height, segmentation confidence, and the annotated image path.

## 11. Project Structure

```text
MetricVision/
├── calibration/
│   ├── images/
│   ├── calibrate_camera.py
│   ├── preview_undistortion.py
│   ├── inspect_undistortion.py
│   ├── compare_distortion_models.py
│   ├── camera_calibration.npz
│   └── camera_calibration.yaml
├── dataset/
│   ├── images/
│   ├── annotations/
│   ├── masks/
│   ├── splits/
│   └── create_masks.py
├── docs/
│   ├── Plan.md
│   ├── Objectives.md
│   └── Definition_of_Done.md
├── results/
├── src/
│   ├── experiment.py
│   └── measure.py
├── weights/
│   ├── unet_best.pth
│   └── deeplabv3plus_best.pth
├── requirements.txt
└── README.md
```

## 12. Limitations and Future Work

- Test on more independent photographs and physical objects.
- Improve accuracy under different camera angles and lighting conditions.
- Evaluate generalization to different object designs.
- Calibrate segmentation confidence against actual measurement errors.
- Investigate remaining errors in marker detection and planar geometry.

## Reproducibility

The repository includes calibration parameters, original dataset images, annotations, fixed data splits, training scripts, model checkpoints, and measurement evaluation results.