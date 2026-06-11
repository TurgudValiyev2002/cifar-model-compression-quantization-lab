# One-Page Report: CIFAR-10 Linear Model Quantization

## Motivation

We use real CIFAR-10 to measure a credible compression trade-off between model size and accuracy.

## Dataset

We used the official CIFAR-10 Python archive. The experiment used 10,000 balanced training images and 2,000 balanced test images from the 10 CIFAR-10 classes.

## Method

We trained an SGD linear classifier on standardized flattened pixels. Then we quantized the learned weights and intercepts to 32, 16, 8, 4, and 2 bits and evaluated each version on the same held-out test set.

## Hyperparameters

The classifier used logistic loss, `alpha=0.0001`, `max_iter=100`, and random seed 42. The input features were 32x32x3 flattened pixel values.

## Results

The float64 model reached 0.3630 accuracy and 0.3667 macro F1. Accuracy stayed at 0.3630 for 32-bit, 16-bit, and 8-bit quantization. At 4-bit it dropped to 0.3475. At 2-bit it collapsed to 0.1080 accuracy.

## Interpretation

The result shows that 8-bit quantization preserved this model, while 2-bit quantization destroyed it. This is a realistic pattern: moderate quantization can work, but aggressive quantization can remove too much weight information.

## Conclusion

The project now reports an honest real-data compression experiment. A stronger follow-up should use a CNN, because linear pixels are not a strong CIFAR-10 model.
