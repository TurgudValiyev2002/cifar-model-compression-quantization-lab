# CIFAR-Style Model Compression and Quantization Lab

![Quantization overview](assets/quantization_overview.png)

Figure: quantization stores the same model weights with fewer bits, reducing memory while trying to keep the model useful.

![Project overview](assets/readme_project_overview.png)

Figure: model-compression workflow from training to int8 weight quantization.


## Motivation

Edge AI often needs models that are small enough to run on limited hardware. Quantization is one common compression method: it stores model weights with fewer bits, usually with a small accuracy trade-off.

## Project Goal

We tested whether an image classifier can keep its accuracy after converting its learned weights from floating point values to int8 values.

## Dataset

The experiment uses a controlled CIFAR-style image dataset with 16x16 RGB images and three visual classes. The images are not the real CIFAR dataset. They are local image-like examples with simple color and shape patterns, used so the compression experiment runs without downloading external data.

## Tools

Python, NumPy, pandas, scikit-learn, and matplotlib.

## Method

We trained multinomial logistic regression on flattened RGB image pixels. After training, we quantized the learned weights to int8 and evaluated the quantized model using the same test set.

## Hyperparameters

- Samples: 1200
- Image size: 16x16x3
- Classes: 3
- Test split: 25 percent
- Model: `LogisticRegression(max_iter=1000, random_state=42)`
- Quantization: symmetric int8 weight quantization

## Results

| Model | Accuracy | Macro F1 | Weight Bytes | Compression Ratio |
|---|---:|---:|---:|---:|
| Float64 logistic regression | 1.0000 | 1.0000 | 18432 | 1.00 |
| Int8 weight quantized | 1.0000 | 1.0000 | 2304 | 8.00 |

The result files include `compression_metrics.csv`, `experiment_setup.csv`, `accuracy_comparison.png`, and `model_size_comparison.png`.

## Interpretation

The quantized model kept the same test accuracy while reducing weight storage by 8 times. This happened because the classification task is simple and the learned decision boundary is robust. On a real CIFAR model, the trade-off may be larger.

## Conclusion

Quantization can reduce model size with little or no accuracy loss when the task and model are stable. The next step should be testing a real CNN on real CIFAR-10 when PyTorch or TensorFlow and the dataset are available.

## How To Run

```bash
pip install -r requirements.txt
python 1_compression_quantization.py
```
