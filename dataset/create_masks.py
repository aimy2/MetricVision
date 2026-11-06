import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = Path("dataset/images")
ANNOTATION_DIR = Path("dataset/annotations")
MASK_DIR = Path("dataset/masks")
MASK_DIR.mkdir(parents=True, exist_ok=True)

annotations = sorted(ANNOTATION_DIR.glob("*.json"))
success = 0
errors = []

for annotation_path in annotations:
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))

        image_path = IMAGE_DIR / f"{annotation_path.stem}.jpg"
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        shapes = data.get("shapes", [])
        if len(shapes) != 1 or shapes[0].get("label") != "card":
            raise ValueError("Expected exactly one polygon labelled 'card'")

        points = np.array(
            shapes[0]["points"], dtype=np.float32
        ).round().astype(np.int32)

        if len(points) < 3:
            raise ValueError("Polygon has fewer than 3 points")

        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)

        cv2.fillPoly(mask, [points], 255)

        output_path = MASK_DIR / f"{annotation_path.stem}.png"
        if not cv2.imwrite(str(output_path), mask):
            raise ValueError("Failed to save mask")

        success += 1

    except Exception as error:
        errors.append((annotation_path.name, str(error)))

print(f"Annotations: {len(annotations)}")
print(f"Masks generated: {success}")
print(f"Errors: {len(errors)}")

for name, error in errors:
    print(f"ERROR: {name}: {error}")

print("RESULT:", "PASS" if success == 70 and not errors else "CHECK REQUIRED")
