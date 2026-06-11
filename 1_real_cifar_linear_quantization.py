from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE = DATA / "cifar-10-python.tar.gz"
EXTRACTED = DATA / "cifar-10-batches-py"
SEED = 42


def download_cifar10() -> None:
    DATA.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading CIFAR-10 from {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    if not EXTRACTED.exists():
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(DATA, filter="data")


def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    x = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(batch["labels"], dtype=np.int64)
    return x, y


def load_cifar10_subset(train_per_class: int = 1000, test_per_class: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download_cifar10()
    train_images = []
    train_labels = []
    for batch_id in range(1, 6):
        x, y = load_batch(EXTRACTED / f"data_batch_{batch_id}")
        train_images.append(x)
        train_labels.append(y)
    x_train_all = np.concatenate(train_images)
    y_train_all = np.concatenate(train_labels)
    x_test_all, y_test_all = load_batch(EXTRACTED / "test_batch")
    rng = np.random.default_rng(SEED)

    def balanced_subset(x: np.ndarray, y: np.ndarray, per_class: int) -> tuple[np.ndarray, np.ndarray]:
        indices = []
        for label in range(10):
            label_idx = np.where(y == label)[0]
            indices.extend(rng.choice(label_idx, size=per_class, replace=False))
        indices = np.array(indices)
        rng.shuffle(indices)
        return x[indices], y[indices]

    return (*balanced_subset(x_train_all, y_train_all, train_per_class), *balanced_subset(x_test_all, y_test_all, test_per_class))


def pixel_features(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32).reshape(len(images), -1) / 255.0


def quantize_symmetric(weights: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    if bits == 32:
        return weights.copy(), 1.0
    levels = 2 ** (bits - 1) - 1
    max_abs = np.max(np.abs(weights))
    scale = max_abs / levels if max_abs > 0 else 1.0
    q = np.round(weights / scale).clip(-levels, levels)
    return q * scale, scale


def evaluate_scores(scores: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    pred = scores.argmax(axis=1)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    x_train_img, y_train, x_test_img, y_test = load_cifar10_subset()
    x_train = pixel_features(x_train_img)
    x_test = pixel_features(x_test_img)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=100, random_state=SEED)),
        ]
    )
    model.fit(x_train, y_train)
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    x_test_scaled = scaler.transform(x_test)

    rows = []
    dense_float64_bytes = clf.coef_.size * 8 + clf.intercept_.size * 8
    for bits in [64, 32, 16, 8, 4, 2]:
        if bits == 64:
            coef = clf.coef_
            intercept = clf.intercept_
            scale = 1.0
        else:
            coef, scale = quantize_symmetric(clf.coef_, bits)
            intercept, _ = quantize_symmetric(clf.intercept_, bits)
        scores = x_test_scaled @ coef.T + intercept
        acc, f1 = evaluate_scores(scores, y_test)
        approx_bytes = (clf.coef_.size + clf.intercept_.size) * bits / 8
        rows.append(
            {
                "weight_precision_bits": bits,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "approx_model_bytes": round(approx_bytes, 1),
                "compression_ratio_vs_float64": round(dense_float64_bytes / approx_bytes, 2),
                "quantization_scale": scale,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "quantization_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"setting": "dataset", "value": "CIFAR-10 official Python archive"},
            {"setting": "train_images", "value": len(x_train)},
            {"setting": "test_images", "value": len(x_test)},
            {"setting": "classes", "value": 10},
            {"setting": "model", "value": "SGD linear classifier on standardized pixels"},
        ]
    ).to_csv(RESULTS / "experiment_setup.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(metrics["weight_precision_bits"], metrics["accuracy"], marker="o", linewidth=2)
    plt.gca().invert_xaxis()
    plt.ylim(0, 0.45)
    plt.xlabel("Weight precision bits")
    plt.ylabel("Test accuracy")
    plt.title("CIFAR-10 Accuracy After Weight Quantization")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_vs_precision.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(metrics["compression_ratio_vs_float64"], metrics["accuracy"], marker="o", linewidth=2)
    plt.xlabel("Compression ratio vs float64")
    plt.ylabel("Test accuracy")
    plt.ylim(0, 0.45)
    plt.title("Compression and Accuracy Trade-off")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "compression_accuracy_tradeoff.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("Real CIFAR-10\npixel features", 0.16),
        ("Train linear\nclassifier", 0.40),
        ("Quantize weights\n64 -> 2 bits", 0.64),
        ("Measure accuracy\nand size", 0.86),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.11, 0.55), xytext=(start[1] + 0.11, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Real-data weight quantization workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "readme_project_overview.png", dpi=180)
    plt.close(fig)

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
