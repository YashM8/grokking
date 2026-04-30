import json
import os
from typing import Any

import matplotlib.pyplot as plt


def plot_metrics(
    train_accs: list[float],
    train_losses: list[float],
    val_accs: list[float],
    val_losses: list[float],
    save_path: str = "figures/training_metrics.png",
) -> None:
    """Generate a 2x2 subplot grid with training and validation metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Grokking Training Metrics", fontsize=16, fontweight="bold")

    train_steps = list(range(1, len(train_accs) + 1))
    val_epochs = list(range(1, len(val_accs) + 1))

    # Top-left: Training accuracy vs steps
    axes[0, 0].plot(train_steps, train_accs, color="steelblue", linewidth=1.0)
    axes[0, 0].set_title("Training Accuracy")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Training loss vs steps
    axes[0, 1].plot(train_steps, train_losses, color="tomato", linewidth=1.0)
    axes[0, 1].set_title("Training Loss")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Validation accuracy vs epochs
    axes[1, 0].plot(val_epochs, val_accs, color="seagreen", linewidth=1.5)
    axes[1, 0].set_title("Validation Accuracy")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Validation loss vs epochs
    axes[1, 1].plot(val_epochs, val_losses, color="darkorange", linewidth=1.5)
    axes[1, 1].set_title("Validation Loss")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training metrics plot to {save_path}")


def plot_ascent_comparison(
    descent_history: dict[str, list[float]],
    ascent_history: dict[str, list[float]],
    save_path: str = "figures/ascent_comparison.png",
) -> None:
    """Visualize descent vs ascent dynamics side by side."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Descent vs Ascent Dynamics", fontsize=16, fontweight="bold")

    d_train_accs = descent_history.get("train_accs", [])
    d_train_losses = descent_history.get("train_losses", [])
    d_val_accs = descent_history.get("val_accs", [])
    d_val_losses = descent_history.get("val_losses", [])

    a_train_accs = ascent_history.get("train_accs", [])
    a_train_losses = ascent_history.get("train_losses", [])
    a_val_accs = ascent_history.get("val_accs", [])
    a_val_losses = ascent_history.get("val_losses", [])

    d_train_steps = list(range(1, len(d_train_accs) + 1))
    a_train_steps = list(range(1, len(a_train_accs) + 1))
    d_val_epochs = list(range(1, len(d_val_accs) + 1))
    a_val_epochs = list(range(1, len(a_val_accs) + 1))

    # Top-left: Training accuracy
    axes[0, 0].plot(d_train_steps, d_train_accs, color="steelblue", label="Descent", linewidth=1.0)
    axes[0, 0].plot(a_train_steps, a_train_accs, color="tomato", label="Ascent", linewidth=1.0)
    axes[0, 0].set_title("Training Accuracy")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Training loss
    axes[0, 1].plot(d_train_steps, d_train_losses, color="steelblue", label="Descent", linewidth=1.0)
    axes[0, 1].plot(a_train_steps, a_train_losses, color="tomato", label="Ascent", linewidth=1.0)
    axes[0, 1].set_title("Training Loss")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Validation accuracy
    axes[1, 0].plot(d_val_epochs, d_val_accs, color="steelblue", label="Descent", linewidth=1.5)
    axes[1, 0].plot(a_val_epochs, a_val_accs, color="tomato", label="Ascent", linewidth=1.5)
    axes[1, 0].set_title("Validation Accuracy")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Validation loss
    axes[1, 1].plot(d_val_epochs, d_val_losses, color="steelblue", label="Descent", linewidth=1.5)
    axes[1, 1].plot(a_val_epochs, a_val_losses, color="tomato", label="Ascent", linewidth=1.5)
    axes[1, 1].set_title("Validation Loss")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ascent comparison plot to {save_path}")


def load_metrics(metrics_file: str) -> dict[str, Any]:
    """Load metrics history from a JSON file."""
    with open(metrics_file) as f:
        return json.load(f)  # type: ignore[no-any-return]
