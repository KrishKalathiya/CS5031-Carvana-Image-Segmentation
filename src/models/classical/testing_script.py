from pathlib import Path
from PIL import Image

data = Path("carvana-image-masking-challenge")

# Pick the first training image
image_path = sorted((data / "train").glob("*.jpg"))[0]

# Find its corresponding mask
mask_path = data / "train_masks" / (image_path.stem + "_mask.gif")

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

print("Image:", image_path)
print("Mask:", mask_path)
print("Sizes:", image.size, mask.size)

assert image.size == mask.size