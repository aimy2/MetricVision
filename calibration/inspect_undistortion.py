import cv2
import numpy as np

data = np.load("calibration/camera_calibration.npz")

K = data["camera_matrix"]
D = data["dist_coeffs"]
width, height = map(int, data["image_size"])

for filename in ["calib_001.jpeg", "calib_012.jpeg", "calib_020.jpeg"]:
    path = "calibration/images/" + filename
    image = cv2.imread(path)
    image = cv2.resize(image, (width, height))

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (width, height), 1, (width, height)
    )

    undistorted = cv2.undistort(image, K, D, None, new_K)

    x, y, w, h = roi
    retained = (w * h) / (width * height) * 100

    print("\n", filename)
    print("Original dimensions:", image.shape[1], "x", image.shape[0])
    print("Undistorted dimensions:", undistorted.shape[1], "x", undistorted.shape[0])
    print("Valid ROI:", roi)
    print(f"Retained rectangular area: {retained:.1f}%")

    cv2.imwrite(
        "calibration/undistortion_preview/" +
        filename.replace(".jpeg", "_undistorted.jpg"),
        undistorted
    )

print("\nInspection complete.")
