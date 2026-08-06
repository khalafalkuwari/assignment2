"""
Model selection + final training for CYSE499/650 Assignment 2 (Stage 1).

Everything here reads `data/train.csv` only. `data/public_test.csv` is never used to
fit, tune, threshold, or early-stop anything -- it is scored once at the end purely for
reporting, which is what the assignment rules require.

    python train.py

Writes:
    model_checkpoint/          the frozen Stage 1 checkpoint
    public_test_predictions.csv
    results.json               CV table + public-test metrics, embedded in the notebook
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import Normalizer

from predict import build_mlp, mlp_proba

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "model_checkpoint"
SEED = 20260804

# Neural training hyper-parameters. Reported verbatim in the notebook writeup.
MLP_CONFIG = dict(
    svd_components=300,
    hidden_dim=256,
    dropout=0.5,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=16,
    epochs=30,
)


# --------------------------------------------------------------------------------------
# Feature builders
# --------------------------------------------------------------------------------------


def make_tfidf() -> FeatureUnion:
    """Word n-grams plus character n-grams over the whole review.

    A bag-of-n-grams representation reads the *entire* document. That matters here:
    reviews run to a median of ~730 words, so any fixed-window encoder would have to
    throw most of each review away. Character n-grams add robustness to the unseen
    vocabulary we expect at evaluation time, since only 240 documents are available to
    learn a word vocabulary from.
    """
    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"
    )
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    return FeatureUnion([("word", word), ("char", char)])


def make_logreg(C: float) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", make_tfidf()),
            (
                "clf",
                LogisticRegression(
                    C=C, class_weight="balanced", max_iter=5000, solver="liblinear"
                ),
            ),
        ]
    )


def make_feature_pipeline(n_components: int) -> Pipeline:
    """TF-IDF -> truncated SVD -> L2 norm; the dense input to the neural model."""
    return Pipeline(
        [
            ("tfidf", make_tfidf()),
            ("svd", TruncatedSVD(n_components=n_components, random_state=SEED)),
            ("norm", Normalizer()),
        ]
    )


# --------------------------------------------------------------------------------------
# Neural training
# --------------------------------------------------------------------------------------


def train_mlp(features: np.ndarray, labels: np.ndarray, cfg: dict, seed: int):
    """Train the feed-forward classifier with a class-weighted loss.

    The class weighting is the primary defence against the 3:1 positive skew: each
    negative example contributes 3x the gradient of a positive one, so the model cannot
    reach a low loss by simply predicting "positive" everywhere.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    counts = np.bincount(labels, minlength=2).astype(float)
    class_weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32)

    model = build_mlp(features.shape[1], cfg["hidden_dim"], cfg["dropout"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(cfg["epochs"]):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start : start + cfg["batch_size"]]
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return model


# --------------------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------------------


def tune_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Cutoff that maximises balanced accuracy on out-of-fold probabilities.

    Balanced accuracy, not raw accuracy, is the selection target: training is 75%
    positive while both evaluation sets are 50/50, so a cutoff tuned on the training
    prior would systematically over-predict the positive class at test time.
    """
    candidates = np.unique(np.round(prob, 4))
    grid = np.concatenate([[0.0], (candidates[:-1] + candidates[1:]) / 2.0, [1.0]])
    scores = [balanced_accuracy_score(y_true, (prob >= t).astype(int)) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def summarise(name: str, y: np.ndarray, prob: np.ndarray) -> dict:
    threshold, tuned = tune_threshold(y, prob)
    at_tuned = (prob >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "bal_acc@0.5": round(float(balanced_accuracy_score(y, (prob >= 0.5).astype(int))), 4),
        "bal_acc@tuned": round(tuned, 4),
        "macro_f1@tuned": round(float(f1_score(y, at_tuned, average="macro")), 4),
        "threshold": round(threshold, 4),
    }


def oof_probabilities(fit_predict, y: np.ndarray, n_repeats: int) -> np.ndarray:
    """Average out-of-fold probabilities over repeated stratified 5-fold CV.

    With 240 examples a single split is mostly noise, so fold assignments are repeated
    and each document's out-of-fold probabilities are averaged across repeats.
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    total = np.zeros(len(y))
    counts = np.zeros(len(y))
    for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
        total[test_idx] += fit_predict(train_idx, test_idx)
        counts[test_idx] += 1
    return total / counts


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    train_frame = pd.read_csv(ROOT / "data" / "train.csv")
    texts = np.array(train_frame["text"].astype(str).tolist(), dtype=object)
    y = train_frame["label"].to_numpy()
    print(f"train: {len(texts)} docs, class counts {np.bincount(y).tolist()}")

    results: dict = {
        "seed": SEED,
        "train_class_counts": np.bincount(y).tolist(),
        "cv_protocol": "RepeatedStratifiedKFold(n_splits=5); n_repeats=2 for logreg, 1 for the MLP",
        "mlp_config": MLP_CONFIG,
        "cv_table": [],
    }

    # ---- Model A: TF-IDF + logistic regression -----------------------------------
    print("\n[A] TF-IDF + logistic regression")
    best_logreg = None
    oof_logreg = None
    for C in (3.0, 10.0):
        start = time.time()

        def fit_predict(tr, te, C=C):
            model = make_logreg(C)
            model.fit(texts[tr].tolist(), y[tr])
            return model.predict_proba(texts[te].tolist())[:, 1]

        prob = oof_probabilities(fit_predict, y, n_repeats=2)
        row = summarise(f"logreg (C={C})", y, prob)
        row["C"] = C
        results["cv_table"].append(row)
        print(f"  C={C:<5} bal_acc={row['bal_acc@tuned']:.4f} auc={row['roc_auc']:.4f} ({time.time()-start:.0f}s)")
        if best_logreg is None or row["bal_acc@tuned"] > best_logreg["bal_acc@tuned"]:
            best_logreg, oof_logreg = row, prob

    # ---- Model B: TF-IDF -> SVD -> neural network --------------------------------
    print("\n[B] TF-IDF + SVD + PyTorch MLP")
    start = time.time()
    feature_pipeline_cv = make_feature_pipeline(MLP_CONFIG["svd_components"])

    def fit_predict_mlp(tr, te):
        pipeline = make_feature_pipeline(MLP_CONFIG["svd_components"])
        train_features = pipeline.fit_transform(texts[tr].tolist()).astype(np.float32)
        test_features = pipeline.transform(texts[te].tolist()).astype(np.float32)
        model = train_mlp(train_features, y[tr], MLP_CONFIG, seed=SEED + len(tr))
        return mlp_proba(model, test_features)

    oof_mlp = oof_probabilities(fit_predict_mlp, y, n_repeats=1)
    row_mlp = summarise("mlp (TF-IDF+SVD)", y, oof_mlp)
    results["cv_table"].append(row_mlp)
    print(f"  bal_acc={row_mlp['bal_acc@tuned']:.4f} auc={row_mlp['roc_auc']:.4f} ({time.time()-start:.0f}s)")

    # ---- Blend --------------------------------------------------------------------
    print("\n[C] Weighted blend of A and B")
    best_blend = None
    for w in np.linspace(0, 1, 21):
        prob = w * oof_logreg + (1 - w) * oof_mlp
        row = summarise(f"blend (logreg={w:.2f}, mlp={1-w:.2f})", y, prob)
        row["weights"] = {"logreg": round(float(w), 2), "mlp": round(float(1 - w), 2)}
        if best_blend is None or row["bal_acc@tuned"] > best_blend["bal_acc@tuned"]:
            best_blend = row
    results["cv_table"].append(best_blend)
    print(f"  best {best_blend['model']} bal_acc={best_blend['bal_acc@tuned']:.4f}")

    # ---- Pick the winner ----------------------------------------------------------
    candidates = [
        ("logreg", best_logreg, {"logreg": 1.0, "mlp": 0.0}),
        ("mlp", row_mlp, {"logreg": 0.0, "mlp": 1.0}),
        ("blend", best_blend, best_blend["weights"]),
    ]
    name, winner, weights = max(candidates, key=lambda c: c[1]["bal_acc@tuned"])
    components = [k for k, v in weights.items() if v > 0]
    print(f"\nSELECTED: {name} -> {winner['model']} (CV balanced accuracy {winner['bal_acc@tuned']:.4f})")
    results["selected"] = {"name": name, "detail": winner, "weights": weights}

    # ---- Retrain on all 240 examples and freeze -----------------------------------
    print("\nRetraining on the full training set and writing the checkpoint...")
    CHECKPOINT.mkdir(exist_ok=True)
    import joblib

    config = {
        "task": "binary sentiment classification (0=negative, 1=positive)",
        "components": components,
        "weights": weights,
        "threshold": winner["threshold"],
        "selection_metric": "balanced accuracy on repeated-stratified-CV out-of-fold predictions",
        "cv_balanced_accuracy": winner["bal_acc@tuned"],
        "trained_on": "data/train.csv (240 documents, 180 positive / 60 negative)",
        "seed": SEED,
    }

    if "logreg" in components:
        final_logreg = make_logreg(best_logreg["C"])
        final_logreg.fit(texts.tolist(), y)
        joblib.dump(final_logreg, CHECKPOINT / "logreg_pipeline.joblib", compress=3)
        config["logreg_C"] = best_logreg["C"]

    if "mlp" in components:
        feature_pipeline = make_feature_pipeline(MLP_CONFIG["svd_components"])
        features = feature_pipeline.fit_transform(texts.tolist()).astype(np.float32)
        final_mlp = train_mlp(features, y, MLP_CONFIG, seed=SEED)
        joblib.dump(feature_pipeline, CHECKPOINT / "feature_pipeline.joblib", compress=3)
        torch.save(final_mlp.state_dict(), CHECKPOINT / "mlp_state_dict.pt")
        config["mlp_architecture"] = {
            "input_dim": int(features.shape[1]),
            "hidden_dim": MLP_CONFIG["hidden_dim"],
            "dropout": MLP_CONFIG["dropout"],
        }
        config["mlp_training"] = MLP_CONFIG

    with open(CHECKPOINT / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    # ---- Score the public test set exactly once ------------------------------------
    print("\nScoring public_test.csv (first and only use)...")
    from predict import load_model, predict, predict_proba

    bundle = load_model(CHECKPOINT)
    test_frame = pd.read_csv(ROOT / "data" / "public_test.csv")
    test_texts = test_frame["text"].astype(str).tolist()
    test_y = test_frame["label"].to_numpy()

    prob = predict_proba(bundle, test_texts)
    pred = predict(bundle, test_texts)

    matrix = confusion_matrix(test_y, pred)
    results["public_test"] = {
        "accuracy": round(float((pred == test_y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test_y, pred)), 4),
        "macro_f1": round(float(f1_score(test_y, pred, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(test_y, prob)), 4),
        "confusion_matrix": matrix.tolist(),
        "n": int(len(test_y)),
    }
    print(json.dumps(results["public_test"], indent=2))

    pd.DataFrame({"id": test_frame["id"], "predicted_label": pred}).to_csv(
        ROOT / "public_test_predictions.csv", index=False, lineterminator="\n"
    )

    with open(ROOT / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nDone. Wrote model_checkpoint/, public_test_predictions.csv, results.json")


if __name__ == "__main__":
    main()
