# Carvana Image Segmentation

Binary segmentation benchmark on the Kaggle Carvana Image Masking Challenge dataset. The objective is to produce a high-resolution binary mask separating each vehicle from its background.

We will be using Python 3.11 for a majority of the project. The dependencies will be stored in `requirements.txt`.
This repository includes both:
1. **Classical Vision Baselines:** Otsu thresholding and SLIC superpixel segmentation.
2. **Deep Learning Architectures:** Custom U-Net and LinkNet (ResNet-34 backbone) with vehicle-isolated cross-validation.

---

## 1. Project Structure

```text
├── carvana-image-masking-challenge/  # (Optional local dataset folder)
│   ├── train/                        # Raw training JPEG images
│   └── train_masks/                  # Ground-truth GIF masks
├── data/
│   ├── splits/                       # Leakage-free vehicle-level split CSVs
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── naive/                    # Image-level random splits (for leakage study)
│   ├── raw/
│   └── processed/
├── experiments/                      # Training history logs and metrics
│   ├── U_net/
│   └── Link_net/
├── outputs/                          # Saved checkpoints and artifacts
│   ├── U_net/checkpoints/best_unet_vehicle.pth
│   └── Link_net/checkpoints/best_linknet_vehicle.pth
├── scripts/                          # Dataset preparation & classical evaluation
│   ├── prepare_scripts.py
│   ├── visualize_dataset.py
│   ├── eval_otsu.py
│   └── eval_slic.py
├── src/
│   ├── data/                         # Data loaders and manifest readers
│   ├── evaluation/                   # Classical and superpixel metrics
│   └── models/
│       ├── classical/                # Otsu, SLIC, feature extraction
│       └── deep/                     # Deep learning framework
│           ├── common/               # Shared dataset, metrics, loss, and utilities
│           │   ├── config.py
│           │   ├── dataset.py
│           │   ├── evaluate.py
│           │   ├── metrics.py
│           │   └── submission.py
│           ├── U_net/                # Custom PyTorch U-Net
│           ├── Link_net/             # LinkNet (ResNet-34 encoder)
│           ├── train.py              # CLI training pipeline
│           └── predict.py            # CLI inference & visualization pipeline
└── requirements.txt
```

---

The "naive" split where images are split randomly is in ```data/splits/naive```, and the proper datasplit (where all 16 images of the same car are only in one of the sets and not the other) are in ```data/splits```.
## 2. Setup & Installation

### Environment Setup
Python 3.10+ or 3.11 is recommended. Create a virtual environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows

# Install dependencies
pip install -r requirements.txt
```

> **GPU Support:** If training with CUDA, ensure your PyTorch build matches your local CUDA version (see [pytorch.org](https://pytorch.org)).

### Dataset Preparation
1. Download data from Kaggle: [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge).
2. Extract the training images into `carvana-image-masking-challenge/train/` (or `data/train/`) and masks into `carvana-image-masking-challenge/train_masks/` (or `data/train_masks/`).
3. Generate the vehicle-level splits (avoiding data leakage across the 16 views per car):
   ```bash
   python -m scripts.prepare_scripts
   ```

---

## 3. Deep Learning CLI Usage

All deep learning entry points can be executed from the project root.

### A. Model Training (`train.py`)

Train a **U-Net** or **LinkNet** with vehicle-isolated train/val splitting:

```bash
# Train U-Net (default 256x256, 20 epochs, batch size 8)
python src/models/deep/train.py \
    --model-dir U_net \
    --images carvana-image-masking-challenge/train \
    --masks carvana-image-masking-challenge/train_masks \
    --output outputs/U_net \
    --epochs 20 \
    --batch-size 8 \
    --image-size 256

# Train LinkNet (ResNet-34 backbone)
python src/models/deep/train.py \
    --model-dir Link_net \
    --images carvana-image-masking-challenge/train \
    --masks carvana-image-masking-challenge/train_masks \
    --output outputs/Link_net \
    --epochs 20 \
    --batch-size 8 \
    --image-size 256
```

**Common Training Arguments:**
| Argument | Type | Default | Description |
|---|---|---|---|
| `--model-dir` | string | `U_net` | Architecture folder: `U_net` or `Link_net` |
| `--images` | path | `data/train` | Path to raw training images folder |
| `--masks` | path | `data/train_masks` | Path to ground truth masks folder |
| `--image-size` | int | `256` | Square resize dimension (must be divisible by 8) |
| `--epochs` | int | `20` | Number of training epochs |
| `--batch-size` | int | `8` | Mini-batch size |
| `--learning-rate`| float | `1e-3` | Initial Adam learning rate |
| `--split-mode` | string | `vehicle` | `vehicle` (group-safe) or `random` (naive) |
| `--output` | path | `outputs` | Target directory for checkpoints and metrics |

---

### B. Inference & Mask Generation (`predict.py`)

Run inference on a single image or an entire folder. Supports generating red-tinted overlays on the original input image.

```bash
# 1. Predict on a single image using the best U-Net checkpoint
python src/models/deep/predict.py \
    --image carvana-image-masking-challenge/train/00087a6bd4dc_01.jpg \
    --checkpoint outputs/U_net/best_unet_vehicle.pth \
    --model-dir U_net \
    --output predictions/ \
    --save-overlay

# 2. Batch predict on a folder of images using LinkNet
python src/models/deep/predict.py \
    --input-dir data/sample_images \
    --checkpoint outputs/Link_net/best_linknet_vehicle.pth \
    --model-dir Link_net \
    --output predictions/linknet_output \
    --threshold 0.5 \
    --save-overlay
```

**Output:**
- `<image_id>_predicted_mask.png`: Binary mask upscaled back to the original image resolution.
- `<image_id>_overlay.png` (optional): Visual overlay highlighting the predicted car body.

---

### C. Checkpoint Evaluation (`evaluate.py`)

Evaluate model metrics (loss, mean Dice, and mean IoU per image) against validation or test folders:

```bash
python src/models/deep/common/evaluate.py \
    --model-file src/models/deep/U_net/model.py \
    --model-class UNet \
    --checkpoint outputs/U_net/best_unet_vehicle.pth \
    --images carvana-image-masking-challenge/train \
    --masks carvana-image-masking-challenge/train_masks \
    --batch-size 8
```

---

### D. Kaggle Submission Generation (`submission.py`)

Run run-length encoding (RLE, Fortran column-major) on test images to produce a competition-ready `submission.csv`:

```bash
python src/models/deep/common/submission.py \
    --model-file src/models/deep/Link_net/model.py \
    --model-class LinkNet \
    --checkpoint outputs/Link_net/best_linknet_vehicle.pth \
    --input-dir carvana-image-masking-challenge/test \
    --output submission.csv
```

---

## 4. Classical Baseline CLI Usage

Run the baseline evaluation pipelines from the repository root:

```bash
# Evaluate Otsu thresholding baseline
python -m scripts.eval_otsu

# Evaluate SLIC superpixel compactness & regions
python -m scripts.eval_slic

# Visualize dataset samples and masks
python -m scripts.visualize_dataset
```
