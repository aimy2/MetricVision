from pathlib import Path
import random

image_dir = Path("dataset/images")
split_dir = Path("dataset/splits")
split_dir.mkdir(parents=True, exist_ok=True)

images = sorted(
    p.name for p in image_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if len(images) != 70:
    raise ValueError(f"Expected 70 images, found {len(images)}")

rng = random.Random(42)
rng.shuffle(images)

train = images[:49]
val = images[49:63]
test = images[63:70]

for name, items in [
    ("train.txt", train),
    ("val.txt", val),
    ("test.txt", test)
]:
    (split_dir / name).write_text(
        "\n".join(items) + "\n",
        encoding="utf-8"
    )

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))
print("Seed: 42")
print("Split files updated.")
