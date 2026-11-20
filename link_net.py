import glob
import os
import zipfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Install segmentation-models-pytorch if not already installed
try:
  import segmentation_models_pytorch as smp
except ImportError:
  os.system("pip install -q segmentation-models-pytorch")
  import segmentation_models_pytorch as smp
# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "data_dir": "/kaggle/input/carvana-image-masking-challenge",
    "work_dir": "/kaggle/working",
    "img_size": (256, 256),  # (H, W) resized for fast training & memory efficiency
    "batch_size": 16,
    "epochs": 5,
    "lr": 1e-3,
    "encoder": "resnet34",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
# ==========================================
# 2. Auto-Detect / Extract Data
# ==========================================
input_root = "/kaggle/input"
work_dir = "/kaggle/working"
# 1. Search for existing image files first
found_jpgs = glob.glob(f"{input_root}/**/*.jpg", recursive=True)
if not found_jpgs:
  # Check for zip archives to extract
  zip_files = glob.glob(f"{input_root}/**/*.zip", recursive=True)
  print(f"Found zip files in input: {zip_files}")
  for zpath in zip_files:
    zname = os.path.basename(zpath)
    if "train" in zname.lower() and not "mask" in zname.lower():
      target = os.path.join(work_dir, "train")
      if not os.path.exists(target):
        print(f"Extracting {zname} -> {target}...")
        with zipfile.ZipFile(zpath, "r") as z:
          z.extractall(target)
    elif "mask" in zname.lower():
      target = os.path.join(work_dir, "train_masks")
      if not os.path.exists(target):
        print(f"Extracting {zname} -> {target}...")
        with zipfile.ZipFile(zpath, "r") as z:
          z.extractall(target)
# Locate training image and mask folders
train_img_candidates = glob.glob(f"{work_dir}/**/train", recursive=True) + [
    p for p in glob.glob(f"{input_root}/**/train*", recursive=True) if os.path.isdir(p)
]
train_mask_candidates = glob.glob(
    f"{work_dir}/**/train_masks*", recursive=True
) + [
    p
    for p in glob.glob(f"{input_root}/**/train_masks*", recursive=True)
    if os.path.isdir(p)
]
# Pick the best matches
train_img_dir = (
    train_img_candidates[0]
    if train_img_candidates
    else os.path.join(work_dir, "train")
)
train_mask_dir = (
    train_mask_candidates[0]
    if train_mask_candidates
    else os.path.join(work_dir, "train_masks")
)
all_images = sorted(glob.glob(f"{train_img_dir}/**/*.jpg", recursive=True))
if len(all_images) == 0:
  # Check if they are in work_dir directly
  all_images = sorted(glob.glob(f"{work_dir}/**/*.jpg", recursive=True))
assert (
    len(all_images) > 0
), f"No training images found! Checked {train_img_dir} and {work_dir}. Run step 1 above to check input paths."
print(
    f"Found {len(all_images)} training images in: {os.path.dirname(all_images[0])}"
)
# Resolve mask extension (.gif vs .png)
sample_id = os.path.splitext(os.path.basename(all_images[0]))[0]
mask_ext = ".gif"
sample_mask_path = os.path.join(train_mask_dir, f"{sample_id}_mask.gif")
if not os.path.exists(sample_mask_path):
  # Try finding sample mask recursively
  found_sample = glob.glob(
      f"{train_mask_dir}/**/{sample_id}_mask.*", recursive=True
  )
  if found_sample:
    train_mask_dir = os.path.dirname(found_sample[0])
    mask_ext = os.path.splitext(found_sample[0])[1]
  else:
    mask_ext = ".png"
all_masks = [
    os.path.join(
        train_mask_dir,
        os.path.splitext(os.path.basename(p))[0] + f"_mask{mask_ext}",
    )
    for p in all_images
]
print(f"Mask directory: {train_mask_dir} (format: *{mask_ext})")
print(f"Sample pair matched:")
print(f"  Image: {all_images[0]}")
print(f"  Mask:  {all_masks[0]} (exists: {os.path.exists(all_masks[0])})")

# ==========================================
# Model & Loss Function
# ==========================================
# LinkNet: Encoder features are bypassed directly into decoder stages via addition
model = smp.Linknet(
    encoder_name=CONFIG["encoder"],
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,  # Output raw logits
).to(CONFIG["device"])
class BCEDiceLoss(nn.Module):
  def __init__(self, smooth=1.0):
    super().__init__()
    self.bce = nn.BCEWithLogitsLoss()
    self.smooth = smooth
  def forward(self, logits, targets):
    bce_loss = self.bce(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    dice = (2.0 * intersection + self.smooth) / (
        probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
    )
    dice_loss = 1.0 - dice.mean()
    return bce_loss + dice_loss
def calculate_dice(preds, targets, threshold=0.5, smooth=1e-6):
  preds = (preds > threshold).float()
  intersection = (preds * targets).sum(dim=(2, 3))
  dice = (2.0 * intersection + smooth) / (
      preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
  )
  return dice.mean().item()
criterion = BCEDiceLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=1
)

# ==========================================
# Create DataLoaders
# ==========================================
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    all_images, all_masks, test_size=0.15, random_state=CONFIG["seed"]
)
train_loader = DataLoader(
    CarvanaDataset(train_imgs, train_masks, CONFIG["img_size"]),
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    CarvanaDataset(val_imgs, val_masks, CONFIG["img_size"]),
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
print(
    f"DataLoaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches."
)

# ==========================================
# 5. Training Loop
# ==========================================
best_val_dice = 0.0
for epoch in range(CONFIG["epochs"]):
  model.train()
  train_loss = 0.0
  pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
  for images, masks in pbar:
    images, masks = images.to(CONFIG["device"]), masks.to(CONFIG["device"])
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, masks)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
  # Validation
  model.eval()
  val_dice = 0.0
  with torch.no_grad():
    for images, masks in val_loader:
      images, masks = images.to(CONFIG["device"]), masks.to(CONFIG["device"])
      logits = model(images)
      probs = torch.sigmoid(logits)
      val_dice += calculate_dice(probs, masks) * images.size(0)
  val_dice /= len(val_imgs)
  scheduler.step(val_dice)
  print(
      f"Epoch {epoch+1} Summary | Train Loss: {train_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f}"
  )
  if val_dice > best_val_dice:
    best_val_dice = val_dice
    torch.save(model.state_dict(), "best_linknet.pth")
    print(f"--> Saved new best model checkpoint (Dice: {val_dice:.4f})")
      
# ==========================================
# 6. RLE Encoding & Submission Generation
# ==========================================
def rle_encode(mask):
  """Fast vectorized Run-Length Encoding.
  Order is top-to-bottom, left-to-right (column-major / Fortran order),
  1-indexed pairs of (start, length).
  """
  dots = mask.flatten(order="F")
  padded = np.pad(dots, (1, 1), mode="constant", constant_values=0)
  diff = padded[1:] - padded[:-1]
  starts = np.where(diff == 1)[0] + 1
  ends = np.where(diff == -1)[0] + 1
  lengths = ends - starts
  rle = []
  for s, l in zip(starts, lengths):
    rle.extend([str(s), str(l)])
  return " ".join(rle)
test_dir = extract_if_needed("test.zip", "test")
if not os.path.exists(test_dir):
  test_dir = os.path.join(CONFIG["data_dir"], "test")
test_images = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
if test_images:
  print(f"Generating submission for {len(test_images)} test images...")
  model.load_state_dict(torch.load("best_linknet.pth"))
  model.eval()
  test_dataset = CarvanaDataset(test_images, None, CONFIG["img_size"])
  test_loader = DataLoader(
      test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2
  )
  submission_records = []
  with torch.no_grad():
    for images, paths, (orig_h_batch, orig_w_batch) in tqdm(
        test_loader, desc="Inference"
    ):
      images = images.to(CONFIG["device"])
      logits = model(images)
      probs = torch.sigmoid(logits).cpu().numpy()
      for i in range(len(paths)):
        prob_mask = probs[i, 0]
        orig_h, orig_w = int(orig_h_batch[i]), int(orig_w_batch[i])
        # Resize predicted mask back to original resolution for evaluation
        prob_mask_orig = cv2.resize(
            prob_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        binary_mask = (prob_mask_orig > 0.5).astype(np.uint8)
        img_id = os.path.basename(paths[i])
        rle_str = rle_encode(binary_mask)
        submission_records.append({"img": img_id, "rle_mask": rle_str})
  sub_df = pd.DataFrame(submission_records)
  sub_df.to_csv("submission.csv", index=False)
  print("Saved submission.csv successfully!")
else:
  print("No test images found. Skipping submission generation.")