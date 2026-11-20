
# LinkNet Experimental Results

## Model Configuration

- Architecture: LinkNet
- Encoder: ResNet34 (ImageNet pretrained)
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
| Validation Loss | 0.0067 |
| Dice Score | 0.9933 |
| IoU | 0.9867 |

## Generated Artifacts

- Best model checkpoint: `best_linknet_vehicle.pth`
- Training history: `history_Link_net_vehicle.csv`
- Validation split: `Link_net_vehicle_split_seed42.csv`
- Kaggle predictions: `linknet_submission.csv`

## Notes

The best checkpoint was saved at Epoch 17 and tied at Epoch 19 (at displayed precision).

The submission contains predictions for 100,064 test images.
