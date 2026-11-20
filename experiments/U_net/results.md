# U-Net Experimental Results

## Model Configuration

- Architecture: U-Net
- Dataset: Carvana Image Masking Challenge
- Image Size: 256 × 256
- Epochs: 20
- Batch Size: 8
- Learning Rate: 0.001
- Validation Split: Vehicle-based
- Split Seed: 42

## Best Validation Performance

| Metric | Result |
|---|---:|
| Validation Loss | 0.0060 |
| Dice Score | 0.9949 |
| IoU | 0.9899 |

## Generated Artifacts

- Best model checkpoint: `best_unet_vehicle.pth`
- Training history: `history_U_net_vehicle.csv`
- Validation split: `U_net_vehicle_split_seed42.csv`
- Kaggle predictions: `unet_submission.csv`

## Notes

The best checkpoint was saved at Epoch 20.

The submission contains predictions for 100,064 test images.