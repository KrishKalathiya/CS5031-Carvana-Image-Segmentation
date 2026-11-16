"""Loss and per-image segmentation metrics for binary masks."""

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Weighted BCE-with-logits + soft Dice loss.

    This preserves the loss used by the existing Carvana U-Net train.py.
    The model must return raw logits, not sigmoid probabilities.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        if not 0.0 <= bce_weight <= 1.0:
            raise ValueError("bce_weight must be between 0 and 1")
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Logit/target shapes must match; got {tuple(logits.shape)} "
                f"and {tuple(targets.shape)}"
            )
        
        targets = targets.float()
        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits).flatten(start_dim=1)
        flat_targets = targets.float().flatten(start_dim=1)

        intersection = (probabilities * flat_targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + flat_targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def dice_iou_per_image(predictions: torch.Tensor,
                       targets: torch.Tensor,
                       threshold: float = 0.5,
                       from_logits: bool = True,
                       epsilon: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one Dice and IoU value per image in a batch.

    Competition-style validation should average these per-image values, not
    compute one global intersection over the entire validation set.
    ``predictions`` is [B, 1, H, W] (or [B, H, W]).
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction/target shapes must match; got {tuple(predictions.shape)} "
            f"and {tuple(targets.shape)}"
        )

    probabilities = torch.sigmoid(predictions) if from_logits else predictions
    predicted_binary = (probabilities >= threshold).to(dtype=torch.float32)
    target_binary = (targets >= 0.5).to(dtype=torch.float32)

    if predicted_binary.ndim == 3:
        dims = (1, 2)
    elif predicted_binary.ndim >= 4:
        dims = tuple(range(1, predicted_binary.ndim))
    else:
        raise ValueError("Expected a batched segmentation tensor")

    intersection = (predicted_binary * target_binary).sum(dim=dims)
    pred_sum = predicted_binary.sum(dim=dims)
    target_sum = target_binary.sum(dim=dims)
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + epsilon) / (pred_sum + target_sum + epsilon)
    iou = (intersection + epsilon) / (union + epsilon)
    return dice, iou


def mean_dice_iou(predictions: torch.Tensor,
                  targets: torch.Tensor,
                  threshold: float = 0.5,
                  from_logits: bool = True) -> tuple[float, float]:
    """Return mean per-image Dice and IoU for a batch."""
    dice, iou = dice_iou_per_image(
        predictions, targets, threshold=threshold, from_logits=from_logits
    )
    return float(dice.mean().item()), float(iou.mean().item())
