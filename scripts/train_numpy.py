#!/usr/bin/env python3
"""Train the NumPy reference MLP and save plots/metrics locally."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_playground import MLP, make_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["xor", "circles", "spiral", "moons", "blobs"], default="xor")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.13)
    parser.add_argument("--hidden", default="8,8", help="Comma-separated hidden widths, e.g. 8,8")
    parser.add_argument("--activation", choices=["tanh", "relu", "leaky-relu", "sigmoid"], default="tanh")
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="Directory for generated artifacts")
    args = parser.parse_args()
    hidden = [int(n.strip()) for n in args.hidden.split(",") if n.strip()]
    X, y = make_dataset(args.dataset, args.samples, args.seed, args.noise)
    model = MLP(hidden=hidden, activation=args.activation, optimizer=args.optimizer,
                learning_rate=args.learning_rate, seed=args.seed)
    start = time.perf_counter()
    history = model.fit(X, y, args.epochs)
    elapsed = time.perf_counter() - start
    metrics = model.evaluate(X, y)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)

    with (out / "training-history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "accuracy"]); writer.writeheader(); writer.writerows(history)
    with (out / "dataset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["x1", "x2", "label"]); writer.writerows(np.column_stack((X, y)))
    (out / "model.json").write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    summary = {"dataset": args.dataset, "samples": len(X), "seed": args.seed, "architecture": model.layer_sizes,
               "activation": args.activation, "optimizer": args.optimizer, "learning_rate": args.learning_rate,
               "epochs": args.epochs, "parameters": model.parameter_count(), "final_loss": metrics.loss,
               "training_accuracy": metrics.accuracy, "elapsed_seconds": elapsed,
               "warning": "Training accuracy on synthetic examples is not a measure of generalisation."}
    (out / "experiment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.plot([h["epoch"] for h in history], [h["loss"] for h in history], color="#159c78", linewidth=2)
    ax.set(title="Training loss", xlabel="Epoch", ylabel="Binary cross-entropy"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out / "loss-curve.png"); plt.close(fig)

    gx, gy = np.meshgrid(np.linspace(-1.3, 1.3, 180), np.linspace(-1.3, 1.3, 180))
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    zz = model.predict_proba(grid).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    ax.contourf(gx, gy, zz, levels=np.linspace(0, 1, 15), cmap="coolwarm", alpha=.28)
    ax.contour(gx, gy, zz, levels=[.5], colors=["#24364d"], linewidths=1.2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=15, c="#3975cf", edgecolors="white", linewidths=.25, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=15, c="#e58b45", edgecolors="white", linewidths=.25, label="Class 1")
    ax.set(title=f"{args.dataset.title()} · decision boundary", xlabel="Feature 1", ylabel="Feature 2", xlim=(-1.3, 1.3), ylim=(-1.3, 1.3)); ax.legend(frameon=False); ax.grid(alpha=.12); fig.tight_layout(); fig.savefig(out / "decision-boundary.png"); plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved experiment artifacts to: {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
