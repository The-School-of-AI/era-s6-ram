# Session 5 Assignment - MNIST Classification

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification that meets specific architectural constraints.

## Model Architecture Summary

- Parameters: 8,932 (Less than 20k requirement ✓)
- Uses Batch Normalization ✓
- Uses Dropout (value: 0.05) ✓
- Uses Global Average Pooling (AvgPool2d) instead of Fully Connected layer ✓

## Training Logs

Best Training Accuracy: 98.93% (Epoch 12)
Best Test Accuracy: 99.50% (Epoch 14)

Detailed training logs:

EPOCH: 0
Loss=0.1125006154179573 Batch_id=468 Accuracy=90.12
Test set: Average loss: 0.0691, Accuracy: 9775/10000 (97.75%)

EPOCH: 1
Loss=0.0646856352686882 Batch_id=468 Accuracy=97.66
Test set: Average loss: 0.0386, Accuracy: 9882/10000 (98.82%)

...

EPOCH: 14
Loss=0.004074355121701956 Batch_id=468 Accuracy=98.92
Test set: Average loss: 0.0182, Accuracy: 9950/10000 (99.50%)

## Model Architecture Details

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 26, 26]              36
       BatchNorm2d-2            [-1, 4, 26, 26]               8
              ReLU-3            [-1, 4, 26, 26]               0
           Dropout-4            [-1, 4, 26, 26]               0
            Conv2d-5            [-1, 4, 24, 24]             144
       BatchNorm2d-6            [-1, 4, 24, 24]               8
              ReLU-7            [-1, 4, 24, 24]               0
           Dropout-8            [-1, 4, 24, 24]               0
            Conv2d-9            [-1, 8, 22, 22]             288
      BatchNorm2d-10            [-1, 8, 22, 22]              16
             ReLU-11            [-1, 8, 22, 22]               0
          Dropout-12            [-1, 8, 22, 22]               0
        MaxPool2d-13            [-1, 8, 11, 11]               0
           Conv2d-14            [-1, 8, 11, 11]              64
      BatchNorm2d-15            [-1, 8, 11, 11]              16
             ReLU-16            [-1, 8, 11, 11]               0
          Dropout-17            [-1, 8, 11, 11]               0
           Conv2d-18             [-1, 16, 9, 9]           1,152
      BatchNorm2d-19             [-1, 16, 9, 9]              32
             ReLU-20             [-1, 16, 9, 9]               0
          Dropout-21             [-1, 16, 9, 9]               0
           Conv2d-22             [-1, 16, 7, 7]           2,304
      BatchNorm2d-23             [-1, 16, 7, 7]              32
             ReLU-24             [-1, 16, 7, 7]               0
          Dropout-25             [-1, 16, 7, 7]               0
           Conv2d-26             [-1, 16, 7, 7]           2,304
      BatchNorm2d-27             [-1, 16, 7, 7]              32
             ReLU-28             [-1, 16, 7, 7]               0
          Dropout-29             [-1, 16, 7, 7]               0
           Conv2d-30             [-1, 16, 7, 7]           2,304
      BatchNorm2d-31             [-1, 16, 7, 7]              32
             ReLU-32             [-1, 16, 7, 7]               0
          Dropout-33             [-1, 16, 7, 7]               0
           Conv2d-34             [-1, 10, 7, 7]             160
        AvgPool2d-35             [-1, 10, 1, 1]               0
================================================================
Total params: 8,932
Trainable params: 8,932
Non-trainable params: 0
```

## Key Features

1. Data Augmentation:
   - Random rotation (-35° to 35°)
   - Normalization (mean=0.1307, std=0.3081)

2. Training Details:
   - Optimizer: SGD with momentum (lr=0.1, momentum=0.9)
   - Loss Function: Negative Log Likelihood
   - Batch Size: 128
   - Number of Epochs: 15

## Results Analysis

1. The model achieves consistent performance improvement over epochs
2. Final test accuracy of 99.50% exceeds the target of 99%
3. Training accuracy closely matches test accuracy, indicating good generalization
4. Model uses less than half of the maximum allowed parameters (8.9k vs 20k limit)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm
- matplotlib

## Note

The model successfully meets all architectural constraints while achieving excellent accuracy on the MNIST dataset.