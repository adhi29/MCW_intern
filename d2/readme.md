# Day 2 - CNN Baseline Model Training on CIFAR-10

##Date:December 26th

## Overview
Implemented and trained a baseline CNN model on CIFAR-10 dataset to establish performance benchmarks before applying quantization techniques.

## Tasks Completed

### 1. Dataset Preparation
- Loaded CIFAR-10 dataset (50,000 training images, 10,000 test images)
- Applied normalization transformations
- Set up DataLoaders with batch size of 128

### 2. Model Architecture
- Designed SimpleCNN with 3 convolutional layers
- Architecture:
  - Conv1: 3→32 channels
  - Conv2: 32→64 channels
  - Conv3: 64→128 channels
  - MaxPooling after each layer
  - 2 Fully connected layers
  - Dropout for regularization
- **Total Parameters**: 620,362

### 3. Training Process
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Trained for 50 epochs until saturation

## Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 77.41% |
| **Test F1 Score** | 77.51% |
| **Training Epochs** | 50 |

## Files
```
d2/
├── README.md
├── cifar10_baseline_training.ipynb
└── cifar10_cnn_baseline.pth
```
