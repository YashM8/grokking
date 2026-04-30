from argparse import Namespace
from math import ceil
from typing import Sized
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data import get_data_loaders
from model import Transformer


def main(args: Namespace) -> dict[str, list[float]]:
    wandb.init(project="grokking", config=vars(args))
    assert wandb.run is not None
    config = wandb.config
    device = torch.device(config.device)

    # Define time scales
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    # Define metrics
    wandb.define_metric("training/accuracy", step_metric="step")
    wandb.define_metric("training/loss", step_metric="step")
    wandb.define_metric("validation/accuracy", step_metric="epoch")
    wandb.define_metric("validation/loss", step_metric="epoch")

    train_loader, val_loader = get_data_loaders(
        config.operation, config.prime, config.training_fraction, config.batch_size
    )
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=config.prime + 2,
        seq_len=5,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=9
    )

    num_epochs = ceil(config.num_steps / len(train_loader))

    train_accs: list[float] = []
    train_losses: list[float] = []
    val_accs: list[float] = []
    val_losses: list[float] = []

    for epoch in tqdm(range(num_epochs)):
        train(model, train_loader, optimizer, scheduler, device, config.num_steps, train_accs, train_losses)
        val_acc, val_loss = evaluate(model, val_loader, device, epoch)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

    # Save checkpoint
    if hasattr(config, "checkpoint") and config.checkpoint:
        checkpoint_path = config.checkpoint
    else:
        checkpoint_path = "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_layers": config.num_layers,
            "dim_model": config.dim_model,
            "num_heads": config.num_heads,
            "num_tokens": config.prime + 2,
            "seq_len": 5,
            "operation": config.operation,
            "prime": config.prime,
        },
        checkpoint_path,
    )
    print(f"Saved model checkpoint to {checkpoint_path}")

    history: dict[str, list[float]] = {
        "train_accs": train_accs,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "val_losses": val_losses,
    }
    return history


def train(
    model: Transformer,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    num_steps: int,
    train_accs: list[float],
    train_losses: list[float],
) -> None:
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch

        optimizer.zero_grad()

        output = model(inputs)[-1, :, :]
        loss = criterion(output, labels)
        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        loss.backward()

        optimizer.step()
        scheduler.step()

        train_accs.append(acc.item())
        train_losses.append(loss.item())

        assert wandb.run is not None
        metrics = {
            "training/accuracy": acc,
            "training/loss": loss,
            "step": wandb.run.step,
        }
        wandb.log(metrics)

        if wandb.run.step == num_steps:
            return


def evaluate(
    model: Transformer,
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.0

    for batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch

        with torch.no_grad():
            output = model(inputs)[-1, :, :]
            correct += (torch.argmax(output, dim=1) == labels).sum()
            loss += criterion(output, labels) * len(labels)

    assert isinstance(val_loader.dataset, Sized)
    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)

    metrics = {"validation/accuracy": acc, "validation/loss": loss, "epoch": epoch}
    wandb.log(metrics, commit=False)

    return float(acc), float(loss)
