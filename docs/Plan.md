# MetricVision — Project Plan

## 1. Project Goal

Develop a computer vision system that takes a photograph of a
physical card and estimates its real-world width and height in
millimetres.

The system will use camera calibration, image undistortion,
object segmentation and a known-size reference object to produce
measurements.

## 2. Experimental Setup

- Object: A physical card.
- Camera: A smartphone camera.
- Ground-truth measurement: A physical ruler.
- Camera calibration: OpenCV checkerboard calibration using at
  least 20 calibration images.
- Reference object: A printed ArUco marker with documented,
  physically measured dimensions.
- Dataset: 70 original photographs captured specifically for
  this project.
- Dataset split: 49 training, 14 validation and 7 test images.
- Annotation: Manually created card segmentation masks.
- Framework: Python, OpenCV and PyTorch.

Camera resolution, zoom and relevant capture settings will be
kept consistent. The card and reference marker will be placed
on the same plane.

## 3. Model Experiments

Three established segmentation architectures will be implemented
and compared:

1. U-Net
2. DeepLabV3+
3. Mask R-CNN with a ResNet-50 FPN backbone

The models will use the same dataset split and a consistent
evaluation protocol.

Pretrained weights may be used where appropriate. Training,
validation and checkpoint selection will be documented.

The final model will be selected using segmentation performance,
measurement accuracy, inference speed and computational cost.

## 4. Technical Approach

### Camera Calibration

- Capture at least 20 checkerboard calibration images.
- Detect checkerboard corners using OpenCV.
- Estimate the camera matrix and distortion coefficients.
- Calculate and report reprojection error.
- Save calibration parameters for reproducible inference.

### Dataset Collection and Preparation

- Capture 70 original photographs of the physical card.
- Include reasonable variation in position, orientation and
  lighting.
- Undistort images using the saved calibration parameters.
- Create manual segmentation masks.
- Create fixed training, validation and test file lists.
- Keep the test images separate from training and model selection.

### Model Training and Comparison

- Train U-Net, DeepLabV3+ and Mask R-CNN.
- Use training data for learning.
- Use validation data for model selection and tuning.
- Evaluate final selected checkpoints on the held-out test set.
- Apply a consistent evaluation protocol across architectures.

### Reference Detection and Measurement

- Detect the ArUco reference marker.
- Use its known physical dimensions and detected corners to
  estimate a planar perspective transformation.
- Rectify the card and reference plane where appropriate.
- Extract the segmented card boundary.
- Estimate card width and height in millimetres.

The reference marker and card must be coplanar for this
measurement approach to be valid.

### Inference Pipeline

A single inference command will:

1. Load the image and camera calibration parameters.
2. Undistort the input image.
3. Detect the reference marker.
4. Segment the card using the selected model.
5. Calculate width and height in millimetres.
6. Generate an annotated output image.
7. Report measurement confidence or refuse unreliable results.

The pipeline will handle missing card detections and missing
reference detections explicitly.

## 5. Evaluation Metrics

### Camera Calibration
- Reprojection error.

### Segmentation
- Mask IoU.
- Dice coefficient.
- Precision.
- Recall.
- mAP@0.5.
- mAP@0.5:0.95.

### Measurement
- Per-image predicted width and height.
- Ground-truth width and height.
- Mean absolute error in millimetres.
- Mean percentage error.

### System Performance
- Inference time.
- Model size.
- Computational requirements.

All models will use the same documented evaluation protocol.
For semantic segmentation models, mask-based AP will be calculated
using a consistent single-object prediction protocol.

## 6. Measurement Validation

- Measure at least 10 card instances manually using a ruler.
- Compare automated measurements with manual measurements.
- Report results for each instance.
- Calculate mean absolute error and mean percentage error.
- Compare measurements obtained from raw and undistorted images.

## 7. Accuracy and Reliability

Define one measurable speed or accuracy target.

Investigate at least one failure condition, such as:
- Camera tilt.
- Changing camera distance.
- Poor lighting.
- Partially obscured object boundaries.

Use confidence checks to avoid reporting unreliable measurements.

## 8. Risks and Limitations

- Changes in camera settings may invalidate calibration.
- Perspective and incorrect reference placement may introduce
  measurement errors.
- A small dataset may lead to overfitting.
- Manual annotation quality affects segmentation performance.
- Ruler precision limits ground-truth accuracy.
- A reference marker and card on different planes can produce
  incorrect measurements.

Unfinished experiments and limitations will be reported honestly.

## 9. Reproducibility

- Maintain a fixed dataset split.
- Record model configurations and training settings.
- Save calibration parameters and model checkpoints.
- Document dependencies and inference instructions.
- Keep the test set out of training, tuning and checkpoint selection.
- Document the final evaluation procedure and results.

