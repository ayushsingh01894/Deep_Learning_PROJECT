"""
Shared utility functions used across all four modules of the
Smart Hospitality Management System.
"""

import json
import os
import random

import numpy as np


def set_global_seed(seed: int = 42):
    """Set random seeds for numpy, python's random, and (if available) TensorFlow,
    so training runs are as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_metrics_json(metrics: dict, out_path: str):
    """Save an evaluation metrics dictionary as pretty-printed JSON.

    Only ever call this with metrics that were actually computed from a
    trained model's predictions on real held-out data — never with
    placeholder or invented numbers.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[utils] Metrics saved to {out_path}")


def load_metrics_json(path: str):
    """Load a previously saved metrics JSON file. Returns None if it doesn't exist yet
    (e.g. because the model hasn't been trained/evaluated)."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_matplotlib_fig(fig, out_path: str, dpi: int = 150):
    """Save a matplotlib figure to disk, creating directories as needed."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[utils] Figure saved to {out_path}")
