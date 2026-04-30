import json
import os
from argparse import Namespace
from math import ceil
from typing import Any, Sized

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data import get_data_loaders
from model import Transformer


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    """Load a model checkpoint from disk."""
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def evaluate_ascent(
    model: Transformer,
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model and return (accuracy, loss)."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total_loss = 0.0

    for batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch

        with torch.no_grad():
            output = model(inputs)[-1, :, :]
            correct += (torch.argmax(output, dim=1) == labels).sum().item()
            total_loss += (criterion(output, labels) * len(labels)).item()

    assert isinstance(val_loader.dataset, Sized)
    n = len(val_loader.dataset)
    return correct / n, total_loss / n


def train_ascent(
    model: Transformer,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
) -> dict[str, list[float]]:
    """Apply gradient ascent (reverse gradient direction) to test algorithm stability."""
    criterion = torch.nn.CrossEntropyLoss()

    train_accs: list[float] = []
    train_losses: list[float] = []
    val_accs: list[float] = []
    val_losses: list[float] = []

    step = 0
    num_epochs = ceil(num_steps / len(train_loader))

    for epoch in tqdm(range(num_epochs), desc="Ascent"):
        model.train()
        for batch in train_loader:
            if step >= num_steps:
                break

            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch

            optimizer.zero_grad()

            output = model(inputs)[-1, :, :]
            loss = criterion(output, labels)
            acc = (torch.argmax(output, dim=1) == labels).sum().item() / len(labels)

            loss.backward()

            # Reverse gradient direction (gradient ascent)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(-1)

            optimizer.step()

            train_accs.append(acc)
            train_losses.append(loss.item())

            if wandb.run is not None:
                wandb.log(
                    {
                        "ascent/training/accuracy": acc,
                        "ascent/training/loss": loss.item(),
                        "ascent_step": step,
                    }
                )

            step += 1

        val_acc, val_loss = evaluate_ascent(model, val_loader, device)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if wandb.run is not None:
            wandb.log(
                {
                    "ascent/validation/accuracy": val_acc,
                    "ascent/validation/loss": val_loss,
                    "ascent_epoch": epoch,
                }
            )

    return {
        "train_accs": train_accs,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "val_losses": val_losses,
    }


def main_ascent(args: Namespace) -> None:
    """Full ascent pipeline: load checkpoint, run gradient ascent, save results."""
    wandb.init(project="grokking", config=vars(args), job_type="ascent")
    assert wandb.run is not None

    device = torch.device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device)

    model = Transformer(
        num_layers=checkpoint["num_layers"],
        dim_model=checkpoint["dim_model"],
        num_heads=checkpoint["num_heads"],
        num_tokens=checkpoint["num_tokens"],
        seq_len=checkpoint["seq_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_loader, val_loader = get_data_loaders(
        checkpoint.get("operation", args.operation),
        checkpoint.get("prime", args.prime),
        args.training_fraction,
        args.batch_size,
    )

    ascent_history = train_ascent(
        model, train_loader, val_loader, optimizer, device, args.ascent_steps
    )

    os.makedirs("metrics", exist_ok=True)
    ascent_metrics_file = "metrics/ascent_history.json"

    with open(ascent_metrics_file, "w") as f:
        json.dump(ascent_history, f, indent=2)
    print(f"Saved ascent metrics to {ascent_metrics_file}")

    # Compare with training history if available
    descent_history: dict[str, list[float]] = {}
    if os.path.exists(args.metrics_file):
        with open(args.metrics_file) as f:
            descent_history = json.load(f)

    from plotting import plot_ascent_comparison

    plot_ascent_comparison(descent_history, ascent_history)
