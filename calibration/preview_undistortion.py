import cv2
import numpy as np
from pathlib import Path

data = np.load("calibration/camera_calibration.npz")

camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]
width, height = data["image_size"]

output_dir = Path("calibration/undistortion_preview")
output_dir.mkdir(exist_ok=True)

for filename in [
    "calib_001.jpeg",
    "calib_012.jpeg",
    "calib_020.jpeg"
]:
    image = cv2.imread(str(Path("calibration/images") / filename))

    image = cv2.resize(image, (int(width), int(height)))

    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (int(width), int(height)),
        1,
        (int(width), int(height))
    )

    undistorted = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs,
        None,
        new_matrix
    )

    comparison = np.hstack((image, undistorted))

    cv2.imwrite(
        str(output_dir / filename.replace(".jpeg", "_comparison.jpg")),
        comparison
    )

print("Saved three original-versus-undistorted comparisons.")
print("Open calibration/undistortion_preview/")
