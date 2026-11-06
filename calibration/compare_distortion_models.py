import cv2
import numpy as np

data = np.load("calibration/camera_calibration.npz")

K = data["camera_matrix"]
D = data["dist_coeffs"]
width, height = map(int, data["image_size"])

models = {
    "Current model": D,
    "Simpler model": np.array(
        [D[0, 0], D[0, 1], 0, 0, 0],
        dtype=np.float64
    ).reshape(1, 5)
}

for name, distortion in models.items():

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, distortion,
        (width, height),
        1,
        (width, height)
    )

    x, y, w, h = roi
    retained = 100 * (w * h) / (width * height)

    print("\n" + name)
    print("Distortion:", distortion)
    print("Valid ROI:", roi)
    print(f"Retained area: {retained:.1f}%")

print("\nComparison complete.")
