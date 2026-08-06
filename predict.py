"""
Inference for the Stage 1 sentiment classifier (CYSE499/650 Assignment 2).

This module is the SINGLE SOURCE OF TRUTH for how a raw review string becomes a
prediction. `stage1_notebook.ipynb` imports these helpers so training and inference
cannot drift apart, and `stage2_notebook.ipynb` imports `load_model` / `predict` so the
hidden-test run provably uses the Stage 1 checkpoint and code path.

Only the `text` column is ever consumed. The `id` column of the released CSVs encodes
the gold label (e.g. "pos_cv696_29740"), so it is carried through for output formatting
only and is never shown to a model.

CLI
---
    python predict.py --input data/public_test.csv --output public_test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

CHECKPOINT_DIR = Path(__file__).resolve().parent / "model_checkpoint"


# --------------------------------------------------------------------------------------
# Neural classifier
# --------------------------------------------------------------------------------------


def build_mlp(input_dim: int, hidden_dim: int, dropout: float):
    """Feed-forward classifier over the reduced TF-IDF representation.

    Deliberately small. With 240 training documents a wide network memorises the
    training set within a couple of epochs, so capacity is kept low and regularised
    with dropout on both the input and the hidden layer.
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 2),
    )


def mlp_proba(model, features: np.ndarray) -> np.ndarray:
    """Positive-class probability from the neural model."""
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
        return torch.softmax(logits, dim=1)[:, 1].numpy().astype(np.float64)


# --------------------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------------------


@dataclass
class Bundle:
    """Everything needed to turn text into a probability, loaded from disk."""

    config: dict
    logreg_pipeline: object | None = None
    feature_pipeline: object | None = None
    mlp: object | None = None

    @property
    def threshold(self) -> float:
        return float(self.config["threshold"])


def load_model(checkpoint_dir: str | Path = CHECKPOINT_DIR) -> Bundle:
    """Load the frozen Stage 1 checkpoint. No network access, no retraining."""
    import joblib

    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json", encoding="utf-8") as fh:
        config = json.load(fh)

    bundle = Bundle(config=config)
    components = config["components"]

    if "logreg" in components:
        bundle.logreg_pipeline = joblib.load(checkpoint_dir / "logreg_pipeline.joblib")

    if "mlp" in components:
        import torch

        bundle.feature_pipeline = joblib.load(checkpoint_dir / "feature_pipeline.joblib")
        arch = config["mlp_architecture"]
        bundle.mlp = build_mlp(arch["input_dim"], arch["hidden_dim"], arch["dropout"])
        bundle.mlp.load_state_dict(
            torch.load(checkpoint_dir / "mlp_state_dict.pt", map_location="cpu")
        )
        bundle.mlp.eval()

    return bundle


# --------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------


def component_probabilities(bundle: Bundle, texts: Sequence[str]) -> dict[str, np.ndarray]:
    """Positive-class probability from each component of the checkpoint."""
    texts = [str(t) for t in texts]
    out: dict[str, np.ndarray] = {}

    if bundle.logreg_pipeline is not None:
        out["logreg"] = bundle.logreg_pipeline.predict_proba(texts)[:, 1]

    if bundle.mlp is not None:
        features = bundle.feature_pipeline.transform(texts)
        out["mlp"] = mlp_proba(bundle.mlp, np.asarray(features, dtype=np.float32))

    return out


def predict_proba(bundle: Bundle, texts: Sequence[str]) -> np.ndarray:
    """Blend the checkpoint's components into one positive-class probability."""
    parts = component_probabilities(bundle, texts)
    weights = bundle.config["weights"]
    total = sum(weights[name] for name in parts)
    blended = sum(parts[name] * weights[name] for name in parts) / total
    return np.asarray(blended, dtype=np.float64)


def predict(bundle: Bundle, texts: Sequence[str]) -> np.ndarray:
    """Hard 0/1 labels using the threshold chosen on cross-validated training folds."""
    return (predict_proba(bundle, texts) >= bundle.threshold).astype(int)


def write_predictions(input_csv: str | Path, output_csv: str | Path, checkpoint_dir=CHECKPOINT_DIR):
    """Read a CSV with `id` and `text`, write `id,predicted_label`."""
    import pandas as pd

    frame = pd.read_csv(input_csv)
    bundle = load_model(checkpoint_dir)
    labels = predict(bundle, frame["text"].astype(str).tolist())
    output = pd.DataFrame({"id": frame["id"], "predicted_label": labels})
    output.to_csv(output_csv, index=False, lineterminator="\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sentiment predictions.")
    parser.add_argument("--input", required=True, help="CSV with 'id' and 'text' columns")
    parser.add_argument("--output", required=True, help="where to write id,predicted_label")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    output = write_predictions(args.input, args.output, args.checkpoint)
    print(f"wrote {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
