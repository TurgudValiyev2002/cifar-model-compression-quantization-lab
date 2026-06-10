from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results")
RNG = np.random.default_rng(42)

def make_cifar_style_data(n=1200, size=16):
    images, labels = [], []
    for i in range(n):
        label = i % 3
        img = RNG.normal(0.25, 0.08, (size, size, 3))
        if label == 0:
            img[:, : size // 2, 0] += 0.45
        elif label == 1:
            img[: size // 2, :, 1] += 0.45
        else:
            rr, cc = np.ogrid[:size, :size]
            mask = (rr - size // 2) ** 2 + (cc - size // 2) ** 2 <= (size // 4) ** 2
            img[mask, 2] += 0.55
        images.append(np.clip(img, 0, 1))
        labels.append(label)
    return np.array(images), np.array(labels)

def quantize_to_int8(weights):
    scale = np.max(np.abs(weights)) / 127
    q = np.round(weights / scale).astype(np.int8)
    recovered = q.astype(np.float32) * scale
    return recovered, scale

def main():
    RESULTS.mkdir(exist_ok=True)
    images, y = make_cifar_style_data()
    x = images.reshape(len(images), -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(x_train_s, y_train)
    pred = clf.predict(x_test_s)
    base_acc = accuracy_score(y_test, pred)
    base_f1 = f1_score(y_test, pred, average="macro")
    q_coef, scale = quantize_to_int8(clf.coef_)
    q_scores = x_test_s @ q_coef.T + clf.intercept_
    q_pred = q_scores.argmax(axis=1)
    q_acc = accuracy_score(y_test, q_pred)
    q_f1 = f1_score(y_test, q_pred, average="macro")
    float_bytes = clf.coef_.size * 8
    int8_bytes = clf.coef_.size * 1
    metrics = pd.DataFrame([
        {"model": "float64_logistic_regression", "accuracy": round(base_acc, 4), "macro_f1": round(base_f1, 4), "weight_bytes": float_bytes, "compression_ratio": 1.0},
        {"model": "int8_weight_quantized", "accuracy": round(q_acc, 4), "macro_f1": round(q_f1, 4), "weight_bytes": int8_bytes, "compression_ratio": round(float_bytes / int8_bytes, 2)},
    ])
    metrics.to_csv(RESULTS / "compression_metrics.csv", index=False)
    pd.DataFrame({"parameter": ["classes", "image_size", "train_samples", "test_samples", "int8_scale"], "value": [3, "16x16x3", len(x_train), len(x_test), scale]}).to_csv(RESULTS / "experiment_setup.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.bar(metrics["model"], metrics["accuracy"], color=["#3d6fb6", "#4a8f5a"])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Before and After Int8 Quantization")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_comparison.png", dpi=160)
    plt.figure(figsize=(6, 4))
    plt.bar(metrics["model"], metrics["weight_bytes"], color=["#b26a3b", "#4a8f5a"])
    plt.ylabel("Weight storage bytes")
    plt.title("Model Weight Storage")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "model_size_comparison.png", dpi=160)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
