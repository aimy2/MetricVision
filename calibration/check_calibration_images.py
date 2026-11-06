import cv2
from pathlib import Path

IMAGE_DIR = Path("calibration/images")
PATTERN_SIZE = (9, 6)  # Internal corners: columns, rows

images = sorted(IMAGE_DIR.glob("*.jpeg"))

print(f"Found {len(images)} images")
print("-" * 40)

detected = []
failed = []

for image_path in images:
    image = cv2.imread(str(image_path))

    if image is None:
        failed.append(image_path.name)
        print(f"FAILED (cannot read): {image_path.name}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    success, corners = cv2.findChessboardCornersSB(
        gray,
        PATTERN_SIZE,
        flags=cv2.CALIB_CB_EXHAUSTIVE |
              cv2.CALIB_CB_ACCURACY
    )

    if success:
        detected.append(image_path.name)
        print(f"DETECTED: {image_path.name}")
    else:
        failed.append(image_path.name)
        print(f"NOT DETECTED: {image_path.name}")

print("-" * 40)
print(f"Successfully detected: {len(detected)}/{len(images)}")
print(f"Failed: {len(failed)}")

if failed:
    print("Images needing attention:")
    for name in failed:
        print(name)
