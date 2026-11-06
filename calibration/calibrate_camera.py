import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path("calibration/images")
OUTPUT_DIR = Path("calibration")

PATTERN_SIZE = (9, 6)
SQUARE_SIZE_MM = 20.0
IMAGE_SIZE = (780, 1040)  # width, height

# Prepare checkerboard coordinates in millimetres
objp = np.zeros(
    (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:PATTERN_SIZE[0],
    0:PATTERN_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE_MM

object_points = []
image_points = []
used_images = []

images = sorted(IMAGE_DIR.glob("*.jpeg"))

print(f"Found {len(images)} images")

for path in images:

    image = cv2.imread(str(path))

    if image is None:
        print(f"Unreadable: {path.name}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize all images to the same resolution
    if gray.shape[::-1] != IMAGE_SIZE:
        gray = cv2.resize(
            gray,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )

    success, corners = cv2.findChessboardCornersSB(
        gray,
        PATTERN_SIZE,
        flags=cv2.CALIB_CB_EXHAUSTIVE |
              cv2.CALIB_CB_ACCURACY
    )

    if success:
        object_points.append(objp.copy())
        image_points.append(corners)
        used_images.append(path.name)
        print(f"DETECTED: {path.name}")
    else:
        print(f"FAILED: {path.name}")

print(f"\nValid images: {len(object_points)}/{len(images)}")

if len(object_points) < 15:
    raise RuntimeError(
        "Not enough valid images. At least 15 are required."
    )

print("\nRunning camera calibration...")

rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        IMAGE_SIZE,
        None,
        None
    )
)

# Calculate per-image reprojection errors
errors = []

for i in range(len(object_points)):

    projected, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )
    observed = image_points[i].reshape(-1, 2).astype(np.float64)
    predicted = projected.reshape(-1, 2).astype(np.float64)

    error = np.sqrt(
        np.mean(np.sum((observed - predicted) ** 2, axis=1))
    )
    errors.append(error)

mean_error = float(np.mean(errors))

# Save calibration parameters
np.savez(
    OUTPUT_DIR / "camera_calibration.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    image_size=np.array(IMAGE_SIZE),
    square_size_mm=SQUARE_SIZE_MM,
    pattern_size=np.array(PATTERN_SIZE),
    rms=rms,
    mean_reprojection_error=mean_error
)

fs = cv2.FileStorage(
    str(OUTPUT_DIR / "camera_calibration.yaml"),
    cv2.FILE_STORAGE_WRITE
)

fs.write("camera_matrix", camera_matrix)
fs.write("distortion_coefficients", dist_coeffs)
fs.write("image_width", IMAGE_SIZE[0])
fs.write("image_height", IMAGE_SIZE[1])
fs.write("checkerboard_square_size_mm", SQUARE_SIZE_MM)
fs.write("opencv_rms", rms)
fs.write("mean_reprojection_error", mean_error)
fs.release()

print("\n========== CALIBRATION RESULTS ==========")
print("\nCamera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(dist_coeffs)

print(f"\nOpenCV RMS error: {rms:.4f} pixels")
print(f"Mean reprojection error: {mean_error:.4f} pixels")

print("\nPer-image errors:")

for name, error in zip(used_images, errors):
    print(f"{name}: {error:.4f} pixels")

print("\nSaved calibration files:")
print("calibration/camera_calibration.npz")
print("calibration/camera_calibration.yaml")
