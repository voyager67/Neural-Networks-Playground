"""Readable NumPy implementation of a binary-classification MLP.

This is the research/CLI track that accompanies the browser visualizer. The
purpose is transparency and reproducibility, not production ML performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

Activation = Literal["sigmoid", "tanh", "relu", "leaky-relu"]
Optimizer = Literal["sgd", "momentum", "adam"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def _activate(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sigmoid":
        return _sigmoid(z)
    if kind == "tanh":
        return np.tanh(z)
    if kind == "relu":
        return np.maximum(0.0, z)
    if kind == "leaky-relu":
        return np.where(z > 0, z, 0.01 * z)
    raise ValueError(f"Unsupported activation: {kind}")


def _activation_derivative(z: np.ndarray, a: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sigmoid":
        return a * (1.0 - a)
    if kind == "tanh":
        return 1.0 - a * a
    if kind == "relu":
        return (z > 0).astype(float)
    if kind == "leaky-relu":
        return np.where(z > 0, 1.0, 0.01)
    raise ValueError(f"Unsupported activation: {kind}")


def make_dataset(kind: str = "xor", count: int = 240, seed: int = 42, noise: float = 0.13) -> tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible 2D binary-classification dataset.

    Returns ``X`` with shape ``(n, 2)`` and binary labels ``y`` with shape
    ``(n,)``. Every dataset is synthetic; no data is downloaded.
    """
    if kind not in {"xor", "circles", "spiral", "moons", "blobs"}:
        raise ValueError(f"Unknown dataset: {kind}")
    n = max(20, min(5000, int(count)))
    sigma = max(0.005, min(0.5, float(noise)))
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2), dtype=float)
    y = np.zeros(n, dtype=np.int64)
    if kind == "xor":
        x = rng.uniform(-1, 1, (n, 2))
        x += rng.normal(0, sigma * 0.35, x.shape)
        y = ((x[:, 0] < 0) != (x[:, 1] < 0)).astype(np.int64)
    elif kind == "circles":
        y = rng.integers(0, 2, n)
        angle = rng.uniform(0, 2 * np.pi, n)
        radius = np.where(y == 1, 0.69, 0.31) + rng.normal(0, sigma * 0.48, n)
        x[:, 0] = np.cos(angle) * radius
        x[:, 1] = np.sin(angle) * radius
    elif kind == "spiral":
        y = np.arange(n) % 2
        for i in range(n):
            arm_index = i // 2
            points_per_arm = max(1, int(np.ceil(n / 2)))
            t = arm_index / max(1, points_per_arm - 1)
            angle = t * np.pi * 3.5 + (np.pi if y[i] else 0)
            radius = 0.08 + t * 0.92
            x[i] = [np.cos(angle) * radius, np.sin(angle) * radius]
        x += rng.normal(0, sigma * 0.34, x.shape)
    elif kind == "moons":
        y = np.arange(n) % 2
        angles = rng.uniform(0, np.pi, n)
        x[y == 0, 0] = np.cos(angles[y == 0]) * 0.68 - 0.26
        x[y == 0, 1] = np.sin(angles[y == 0]) * 0.58 + 0.08
        x[y == 1, 0] = 1 - np.cos(angles[y == 1]) * 0.68 + 0.1
        x[y == 1, 1] = -np.sin(angles[y == 1]) * 0.58 - 0.12
        x += rng.normal(0, sigma * 0.55, x.shape)
    else:  # Gaussian blobs
        y = rng.integers(0, 2, n)
        centers = np.where(y[:, None] == 1, 0.46, -0.46)
        scales = np.array([0.16 + sigma * 0.35, 0.20 + sigma * 0.35])
        x = centers + rng.normal(0, 1, (n, 2)) * scales
    x = np.clip(x, -1.25, 1.25)
    order = rng.permutation(n)
    return x[order], y[order]


@dataclass
class Metrics:
    loss: float
    accuracy: float


class MLP:
    """Fully-connected MLP with backpropagation and SGD/Momentum/Adam."""
    def __init__(self, hidden: tuple[int, ...] | list[int] = (8, 8), activation: Activation = "tanh",
                 optimizer: Optimizer = "adam", learning_rate: float = 0.025, seed: int = 42):
        if not 1 <= len(hidden) <= 4 or any(int(n) < 2 or int(n) > 32 for n in hidden):
            raise ValueError("Use 1–4 hidden layers, each with 2–32 neurons.")
        if activation not in {"sigmoid", "tanh", "relu", "leaky-relu"}:
            raise ValueError(f"Unsupported activation: {activation}")
        if optimizer not in {"sgd", "momentum", "adam"}:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        if not 0.0001 <= float(learning_rate) <= 1:
            raise ValueError("learning_rate must be in [0.0001, 1].")
        self.layer_sizes = [2, *[int(n) for n in hidden], 1]
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = np.sqrt(2 / fan_in) if activation in {"relu", "leaky-relu"} else np.sqrt(2 / (fan_in + fan_out))
            self.weights.append(self.rng.normal(0, scale, (fan_out, fan_in + 1)))
        self.velocity = [np.zeros_like(w) for w in self.weights]
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.optimizer_steps = 0
        self.last_gradients = [np.zeros_like(w) for w in self.weights]

    def forward(self, X: np.ndarray, return_cache: bool = False):
        a = np.asarray(X, dtype=float)
        if a.ndim == 1:
            a = a[None, :]
        activations = [a]
        zs = []
        for i, weight in enumerate(self.weights):
            a_bias = np.concatenate([a, np.ones((len(a), 1))], axis=1)
            z = a_bias @ weight.T
            a = _sigmoid(z) if i == len(self.weights) - 1 else _activate(z, self.activation)
            zs.append(z)
            activations.append(a)
        if return_cache:
            return a[:, 0], (activations, zs)
        return a[:, 0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int64)

    @staticmethod
    def binary_cross_entropy(probabilities: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(np.asarray(probabilities).reshape(-1), 1e-7, 1 - 1e-7)
        labels = np.asarray(y).reshape(-1)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Metrics:
        p = self.predict_proba(X)
        return Metrics(self.binary_cross_entropy(p, y), float(np.mean((p >= 0.5) == np.asarray(y).reshape(-1))))

    def _batch_gradients(self, X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        predictions, (activations, zs) = self.forward(X, return_cache=True)
        labels = np.asarray(y, dtype=float).reshape(-1, 1)
        deltas: list[np.ndarray] = [np.empty((len(X), s), dtype=float) for s in self.layer_sizes[1:]]
        deltas[-1] = predictions.reshape(-1, 1) - labels  # sigmoid + BCE
        for layer in range(len(self.weights) - 2, -1, -1):
            # Ignore the bias column when propagating to the preceding layer.
            upstream = deltas[layer + 1] @ self.weights[layer + 1][:, :-1]
            deltas[layer] = upstream * _activation_derivative(zs[layer], activations[layer + 1], self.activation)
        gradients = []
        for layer, delta in enumerate(deltas):
            a_bias = np.concatenate([activations[layer], np.ones((len(X), 1))], axis=1)
            gradients.append((delta.T @ a_bias) / max(1, len(X)))
        return gradients

    def train_batch(self, X: np.ndarray, y: np.ndarray, *, momentum: float = 0.9, beta1: float = 0.9,
                    beta2: float = 0.999, epsilon: float = 1e-8) -> Metrics:
        gradients = self._batch_gradients(np.asarray(X), np.asarray(y))
        self.last_gradients = [g.copy() for g in gradients]
        self.optimizer_steps += 1
        for li, (weight, grad) in enumerate(zip(self.weights, gradients)):
            if self.optimizer == "momentum":
                self.velocity[li] = momentum * self.velocity[li] + grad
                weight -= self.learning_rate * self.velocity[li]
            elif self.optimizer == "adam":
                self.m[li] = beta1 * self.m[li] + (1 - beta1) * grad
                self.v[li] = beta2 * self.v[li] + (1 - beta2) * np.square(grad)
                m_hat = self.m[li] / (1 - beta1 ** self.optimizer_steps)
                v_hat = self.v[li] / (1 - beta2 ** self.optimizer_steps)
                weight -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            else:
                weight -= self.learning_rate * grad
        return self.evaluate(X, y)

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 24, shuffle: bool = True) -> Metrics:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=np.int64)
        indices = self.rng.permutation(len(X)) if shuffle else np.arange(len(X))
        for start in range(0, len(X), max(1, int(batch_size))):
            chosen = indices[start:start + batch_size]
            self.train_batch(X[chosen], y[chosen])
        return self.evaluate(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 24) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for epoch in range(1, max(0, int(epochs)) + 1):
            metrics = self.train_epoch(X, y, batch_size=batch_size)
            history.append({"epoch": float(epoch), "loss": metrics.loss, "accuracy": metrics.accuracy})
        return history

    def parameter_count(self) -> int:
        return sum(w.size for w in self.weights)

    def to_dict(self) -> dict:
        return {"format": "neural-playground-model", "formatVersion": 1, "layerSizes": self.layer_sizes,
                "activation": self.activation, "optimizer": self.optimizer, "learningRate": self.learning_rate,
                "seed": self.seed, "weights": [w.tolist() for w in self.weights]}
