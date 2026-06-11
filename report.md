# One-Page Report: CIFAR-10 CNN Quantization

## Motivation

We used a CNN because CIFAR-10 is an image dataset and convolution is a better model family than a linear pixel classifier.

## Dataset

The experiment used 6,000 CIFAR-10 training images and 1,500 test images from the official Python archive.

## Method

We trained a small PyTorch CNN for 6 epochs and then quantized the trained weights to 32, 16, 8, 4, and 2 bits. We evaluated each version on the same test set.

## Results

The float32 CNN achieved 0.4780 accuracy and 0.4539 macro F1. Int8 quantization achieved 0.4747 accuracy. Four-bit quantization dropped to 0.4407, and two-bit quantization collapsed to 0.1793.

## Interpretation

The CNN tolerates moderate quantization. Int8 keeps almost all accuracy, but 2-bit quantization removes too much information.

## Conclusion

CNN quantization gives a realistic compression result: int8 is useful, 4-bit is risky but possible, and 2-bit is too aggressive for this trained model.
