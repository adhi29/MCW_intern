# Custom ResNet - 29th Dec

Built a ResNet-style architecture from scratch for CIFAR-10 classification.

## What's here

`model.py` - Custom ResNet implementation with evaluation code

## Architecture

Basic ResNet with residual blocks:
- Initial conv layer (3→64 channels)
- 4 residual layers with increasing channels (64→128→256→512)
- Each layer has 2 residual blocks
- Global average pooling + FC layer for classification

Total: 8 residual blocks, similar to ResNet-18 but adapted for CIFAR-10's 32x32 images.

## What it does

Loads a pretrained model (`custom_resnet_cifar10.pth`) and evaluates on CIFAR-10 test set.

Metrics:
- Test accuracy
- F1 score (macro-averaged across 10 classes)

## Running

```bash
python model.py
```

Needs the trained model file `custom_resnet_cifar10.pth` in the same directory. Will download CIFAR-10 test set automatically if not present.

## Notes

- Uses standard CIFAR-10 normalization
- Batch size 128 for testing
- Residual connections help with gradient flow during training
- Should get around 90%+ accuracy if trained properly
