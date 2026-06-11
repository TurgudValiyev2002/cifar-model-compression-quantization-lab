from __future__ import annotations

import copy
import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE = DATA / "cifar-10-python.tar.gz"
EXTRACTED = DATA / "cifar-10-batches-py"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class SmallCIFARCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def download_cifar10() -> None:
    DATA.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading CIFAR-10 from {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    if not EXTRACTED.exists():
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(DATA, filter="data")


def batch_path(name: str) -> Path:
    flat = DATA / name
    if flat.is_file():
        return flat
    standard = EXTRACTED / name
    if standard.is_file():
        return standard
    raise FileNotFoundError(name)


def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    x = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(batch["labels"], dtype=np.int64)
    return x, y


def load_subset(train_per_class: int = 600, test_per_class: int = 150) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download_cifar10()
    train_images = []
    train_labels = []
    for batch_id in range(1, 6):
        x, y = load_batch(batch_path(f"data_batch_{batch_id}"))
        train_images.append(x)
        train_labels.append(y)
    x_train_all = np.concatenate(train_images)
    y_train_all = np.concatenate(train_labels)
    x_test_all, y_test_all = load_batch(batch_path("test_batch"))
    rng = np.random.default_rng(SEED)

    def balanced(x: np.ndarray, y: np.ndarray, per_class: int) -> tuple[np.ndarray, np.ndarray]:
        idx = []
        for label in range(10):
            candidates = np.where(y == label)[0]
            idx.extend(rng.choice(candidates, size=per_class, replace=False))
        idx = np.array(idx)
        rng.shuffle(idx)
        return x[idx], y[idx]

    return (*balanced(x_train_all, y_train_all, train_per_class), *balanced(x_test_all, y_test_all, test_per_class))


def make_dataset(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    x = images.astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    x = ((x - mean) / std).transpose(0, 3, 1, 2)
    return TensorDataset(torch.tensor(x), torch.tensor(labels, dtype=torch.long))


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> tuple[float, float, float]:
    model.eval()
    losses = []
    labels = []
    preds = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            losses.append(loss_fn(logits, y_batch).item() * x_batch.size(0))
            labels.extend(y_batch.numpy())
            preds.extend(torch.argmax(logits, dim=1).numpy())
    return sum(losses) / len(loader.dataset), accuracy_score(labels, preds), f1_score(labels, preds, average="macro")


def train_model(train_loader: DataLoader, val_loader: DataLoader) -> tuple[nn.Module, pd.DataFrame, int]:
    set_seed()
    model = SmallCIFARCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, 7):
        model.train()
        labels = []
        preds = []
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            labels.extend(y_batch.numpy())
            preds.extend(torch.argmax(logits.detach(), dim=1).numpy())
        val_loss, val_acc, _ = evaluate(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / len(train_loader.dataset),
                "train_accuracy": accuracy_score(labels, preds),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 32:
        return tensor.clone()
    levels = 2 ** (bits - 1) - 1
    max_abs = tensor.abs().max()
    if max_abs == 0:
        return tensor.clone()
    scale = max_abs / levels
    return torch.round(tensor / scale).clamp(-levels, levels) * scale


def quantized_model(model: nn.Module, bits: int) -> nn.Module:
    q_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in q_model.named_parameters():
            if "weight" in name or "bias" in name:
                param.copy_(quantize_tensor(param, bits))
    return q_model


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def plot_outputs(metrics: pd.DataFrame, history: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(metrics["weight_precision_bits"], metrics["accuracy"], marker="o", linewidth=2)
    plt.gca().invert_xaxis()
    plt.ylim(0, max(0.65, metrics["accuracy"].max() + 0.08))
    plt.xlabel("Weight precision bits")
    plt.ylabel("Test accuracy")
    plt.title("CNN Accuracy After Post-training Quantization")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_vs_precision.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(metrics["compression_ratio_vs_float32"], metrics["accuracy"], marker="o", linewidth=2)
    plt.xlabel("Compression ratio vs float32")
    plt.ylabel("Test accuracy")
    plt.title("CNN Quantization Compression Trade-off")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "compression_accuracy_tradeoff.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("CNN loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="validation")
    axes[1].set_title("CNN accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "cnn_training_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("Real CIFAR-10", 0.15),
        ("Train small CNN", 0.40),
        ("Quantize weights\n32 -> 2 bits", 0.65),
        ("Measure accuracy\nand size", 0.88),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.12, 0.55), xytext=(start[1] + 0.12, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("CNN post-training quantization workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "readme_project_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    x_train, y_train, x_test, y_test = load_subset()
    rng = np.random.default_rng(SEED)
    train_idx = []
    val_idx = []
    for label in range(10):
        idx = np.where(y_train == label)[0]
        rng.shuffle(idx)
        split = int(0.85 * len(idx))
        train_idx.extend(idx[:split])
        val_idx.extend(idx[split:])
    train_loader = DataLoader(make_dataset(x_train[np.array(train_idx)], y_train[np.array(train_idx)]), batch_size=128, shuffle=True)
    val_loader = DataLoader(make_dataset(x_train[np.array(val_idx)], y_train[np.array(val_idx)]), batch_size=256, shuffle=False)
    test_loader = DataLoader(make_dataset(x_test, y_test), batch_size=256, shuffle=False)
    model, history, best_epoch = train_model(train_loader, val_loader)
    loss_fn = nn.CrossEntropyLoss()
    param_count = parameter_count(model)
    rows = []
    for bits in [32, 16, 8, 4, 2]:
        q_model = model if bits == 32 else quantized_model(model, bits)
        test_loss, acc, macro_f1 = evaluate(q_model, test_loader, loss_fn)
        approx_bytes = param_count * bits / 8
        rows.append(
            {
                "weight_precision_bits": bits,
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "test_loss": round(test_loss, 4),
                "approx_model_bytes": round(approx_bytes, 1),
                "compression_ratio_vs_float32": round((param_count * 4) / approx_bytes, 2),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "quantization_metrics.csv", index=False)
    history.to_csv(RESULTS / "cnn_training_history.csv", index=False)
    pd.DataFrame(
        [
            {"setting": "dataset", "value": "CIFAR-10 official Python archive"},
            {"setting": "model", "value": "small CNN with three convolution blocks"},
            {"setting": "train_images", "value": len(x_train)},
            {"setting": "test_images", "value": len(x_test)},
            {"setting": "parameters", "value": param_count},
            {"setting": "epochs", "value": 6},
            {"setting": "best_epoch", "value": best_epoch},
        ]
    ).to_csv(RESULTS / "experiment_setup.csv", index=False)
    torch.save(model.state_dict(), RESULTS / "float32_small_cifar_cnn.pt")
    plot_outputs(metrics, history)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
