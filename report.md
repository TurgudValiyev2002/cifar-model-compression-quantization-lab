# Report: CIFAR-Style Model Compression and Quantization

## Motivation

We studied quantization because compressed models are important for edge AI and embedded AI.

## Dataset

The experiment used a controlled CIFAR-style RGB image dataset with 1200 samples, 16x16x3 pixels, and 3 classes. It is not the real CIFAR dataset.

## Method

We trained logistic regression on flattened image pixels, then quantized the learned weights to int8 and compared the original and quantized models.

## Hyperparameters

The test split was 25 percent. Logistic regression used `max_iter=1000` and `random_state=42`. Quantization used symmetric int8 scaling.

## Results

Both the original and quantized models achieved 1.0000 accuracy and 1.0000 macro F1. Weight storage decreased from 18432 bytes to 2304 bytes, an 8x reduction.

## Interpretation

Quantization did not hurt accuracy because this controlled image task has very clear class patterns. The result shows the mechanism of compression, not a claim about real CIFAR performance.

## Conclusion

Int8 quantization can strongly reduce storage. A stronger future version should evaluate a real CNN on real CIFAR-10.
