import json
import os
from argparse import ArgumentParser, Namespace

from data import ALL_OPERATIONS
from training import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["train", "ascent", "plot"], default="train"
    )
    parser.add_argument(
        "--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x+y"
    )
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--num_steps", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--checkpoint", type=str, default="model.pt")
    parser.add_argument("--ascent_steps", type=int, default=5000)
    parser.add_argument(
        "--metrics_file", type=str, default="metrics/training_history.json"
    )
    args: Namespace = parser.parse_args()

    if args.mode == "train":
        history = main(args)
        os.makedirs("metrics", exist_ok=True)
        with open(args.metrics_file, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved training metrics to {args.metrics_file}")

        from plotting import plot_metrics

        plot_metrics(
            history["train_accs"],
            history["train_losses"],
            history["val_accs"],
            history["val_losses"],
        )

    elif args.mode == "plot":
        from plotting import load_metrics, plot_metrics

        history = load_metrics(args.metrics_file)
        plot_metrics(
            history["train_accs"],
            history["train_losses"],
            history["val_accs"],
            history["val_losses"],
        )

    elif args.mode == "ascent":
        from ascent import main_ascent

        main_ascent(args)
