"""Generate stage1_notebook.ipynb and stage2_notebook.ipynb.

Keeping the notebooks generated from one script means the narrative, the code cells and
the checkpoint they load stay consistent. Run `python build_notebooks.py`, then execute
the notebooks with nbconvert.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


# ======================================================================================
# Stage 1
# ======================================================================================

stage1 = [
    md(
        """
# Assignment 2 — Stage 1: Sentiment Classification of Movie Reviews

**CYSE499/650, Summer 2026**

The task is binary sentiment classification over the Pang & Lee movie-review polarity
corpus (`0 = negative`, `1 = positive`). The released training split is deliberately
small and skewed, which is the real difficulty of the assignment:

| Split | Size | Positive | Negative |
|---|---|---|---|
| `train.csv` | 240 | 180 | 60 |
| `public_test.csv` | 400 | 200 | 200 |

This notebook documents the model, the design decisions forced by that data, the
cross-validated comparison used to choose between candidates, and the final evaluation
on the public test set.

**Reproducibility note.** Training is performed by `train.py`, which writes
`model_checkpoint/` and `results.json`. This notebook *loads* those artefacts rather
than retraining, so it runs in seconds and reports exactly the model that was submitted.
Re-run `python train.py` to regenerate them from scratch.
"""
    ),
    code(
        """
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
train = pd.read_csv(ROOT / "data" / "train.csv")
public_test = pd.read_csv(ROOT / "data" / "public_test.csv")

print("train      ", train.shape, dict(train.label.value_counts().sort_index()))
print("public_test", public_test.shape, dict(public_test.label.value_counts().sort_index()))
"""
    ),
    md(
        """
## 1. What the data actually looks like

Two properties of this corpus drove every design decision below, so they are measured
here rather than assumed.
"""
    ),
    code(
        """
words = train.text.str.split().str.len()
print("review length in words — median %d, mean %d, 90th pct %d, max %d"
      % (words.median(), words.mean(), words.quantile(0.9), words.max()))

# Sanity check: no document is shared between the two splits.
overlap = set(train.source_file) & set(public_test.source_file)
print("documents shared between train and public test:", len(overlap))

print("\\nfirst 300 characters of one review:")
print(repr(train.text.iloc[0][:300]))
"""
    ),
    md(
        """
Two consequences:

**(a) The reviews are long.** A median review is ~730 words (~1,000 word-piece tokens)
and the longest is 2,570 words. A pretrained encoder with a 512-token window would
discard the majority of most reviews — and in this corpus the verdict often arrives in
the final paragraph, after several paragraphs of plot summary. Any fixed-window model
would need chunking plus document-level aggregation to avoid throwing away the part that
carries the label.

**(b) There is very little supervision.** 240 labelled documents is small enough that a
high-capacity model memorises the training set almost immediately.

Together these pushed the design toward representations that read the **whole** document
and models with **few effective parameters**.

> **Label leakage warning.** The `id` column encodes the gold label
> (`pos_cv696_29740`, `neg_cv963_7208`). It is used only to format the output CSV and is
> never given to a model. Likewise `label_name` and `source_file` are never used as
> features.
"""
    ),
    md(
        """
## 2. Model structure

Two candidate models were built over a shared text representation, and a weighted blend
of them was also evaluated.

**Shared representation — TF-IDF over the full review**

- word 1–2 grams, `min_df=2`, sublinear term frequency
- character `char_wb` 3–5 grams, `min_df=3`, sublinear term frequency

Character n-grams matter here specifically because of the small training set: with only
240 documents the learned word vocabulary is thin, and the assignment warns that
evaluation data contains tokens unseen in training. Character n-grams degrade gracefully
on unseen words (they still match morphology and sub-word fragments) where a pure word
model simply drops them.

**Model A — Logistic regression** directly on the sparse TF-IDF vector, with
`class_weight="balanced"`.

**Model B — Feed-forward neural network (PyTorch).** The sparse TF-IDF matrix is reduced
with truncated SVD (latent semantic analysis) and L2-normalised, then fed to:

```
Dropout(0.5) → Linear(d → 256) → LayerNorm → GELU → Dropout(0.5) → Linear(256 → 2)
```

`svd_components` is requested as 300, but truncated SVD cannot produce more components
than there are documents, so the effective width is **d = 240** — a detail worth stating
because it means the representation is saturated: every additional latent dimension the
configuration asks for is unavailable at this dataset size.

The SVD step is what makes a neural model viable on 240 examples: it collapses ~57k
sparse features into a dense subspace, so the network has a tractable number of input
weights and the latent dimensions capture co-occurrence structure that individual
n-grams cannot.

**Model C — Blend.** A weighted average of the two probability outputs, with the weight
chosen on cross-validation.
"""
    ),
    code(
        """
import inspect
import predict
import train as training

print(inspect.getsource(training.make_tfidf))
print(inspect.getsource(predict.build_mlp))
"""
    ),
    md(
        """
## 3. How the small and imbalanced training set was handled

The training split is 75% positive while both evaluation sets are 50/50. Left alone, a
model trained on this prior scores well on training-like data by leaning positive, and
then loses most of that advantage on a balanced test set. Four separate mechanisms
address this.

**1. Class-weighted loss.** Logistic regression uses `class_weight="balanced"`; the
neural model uses `CrossEntropyLoss(weight=...)` with the same inverse-frequency
weights. Each negative example contributes 3× the gradient of a positive one, so the
model cannot reach a low loss by predicting "positive" everywhere.

**2. Decision threshold tuned for balanced accuracy.** The probability cutoff is *not*
left at 0.5. It is chosen to maximise balanced accuracy on **out-of-fold** predictions,
then frozen into the checkpoint and applied unchanged at inference. This is the single
largest correction for the prior mismatch, and the notebook reports the model at both
0.5 and the tuned cutoff so the effect is visible.

**3. Repeated stratified cross-validation for every decision.** With 240 examples a
single train/validation split is mostly noise. Model choice, the regularisation
strength, the blend weight, and the threshold are all selected on repeated stratified
5-fold CV, with each document's out-of-fold probabilities averaged across repeats.
Every fold preserves the 3:1 class ratio.

**4. Capacity kept deliberately low.** Strong L2 regularisation on the linear model;
SVD compression, dropout of 0.5 on both the input and hidden layer, weight decay, and
gradient clipping on the neural model. The aim is a model that cannot memorise 240
documents.

**Selection metric.** Balanced accuracy, not raw accuracy — on a 3:1 training prior,
raw accuracy rewards exactly the bias we are trying to remove.

**What `public_test.csv` was used for: nothing but the final report.** It is not used to
fit, tune, threshold, blend, or early-stop. The rules forbid training on it, and every
number in Section 6 comes from a model that had never seen it.
"""
    ),
    md(
        """
## 4. Key training techniques

Hyper-parameters for the neural model, taken directly from `train.py`:

| Setting | Value | Why |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay; the standard choice for small dense classifiers |
| Learning rate | `1e-3` | With only 15 optimizer steps per epoch, a smaller LR does not converge inside the epoch budget |
| LR schedule | Cosine annealing to 0 over 30 epochs | Removes "which epoch to stop at" as a tunable, which matters when there is no validation set to spare |
| Weight decay | `1e-4` | Regularisation on top of dropout |
| Batch size | 16 | 240 examples → 15 steps/epoch; larger batches give too few updates, smaller ones are unstable |
| Epochs | 30 | Fixed budget, paired with the cosine schedule |
| Loss | Cross-entropy, inverse-frequency class weights | Counters the 3:1 skew |
| Gradient clipping | max-norm 1.0 | Guards against outlier batches in a small dataset |
| Dropout | 0.5 (input and hidden) | The main capacity control |
| Seed | 20260804 | Fixed for reproducibility |

For logistic regression the tuned parameter is the inverse regularisation strength `C`,
selected over `{3, 10}` by cross-validation with `liblinear`.
"""
    ),
    code(
        """
print(inspect.getsource(training.train_mlp))
"""
    ),
    md(
        """
## 5. Cross-validated model comparison

All numbers below come from `train.py` and use `train.csv` only.
"""
    ),
    code(
        """
results = json.load(open(ROOT / "results.json"))
print("CV protocol:", results["cv_protocol"])

cv = pd.DataFrame(results["cv_table"])[
    ["model", "roc_auc", "bal_acc@0.5", "bal_acc@tuned", "macro_f1@tuned", "threshold"]
]
cv
"""
    ),
    code(
        """
sel = results["selected"]
print("Selected model :", sel["detail"]["model"])
print("Blend weights  :", sel["weights"])
print("CV balanced acc: %.4f" % sel["detail"]["bal_acc@tuned"])
print("Threshold      : %.4f" % sel["detail"]["threshold"])
"""
    ),
    md(
        """
Note the gap between the `bal_acc@0.5` and `bal_acc@tuned` columns — that difference is
the imbalance correction described in Section 3, measured rather than assumed.
"""
    ),
    md(
        """
## 6. Evaluation on the public test set

The checkpoint is reloaded from disk exactly the way a grader would load it, then scored
on all 400 public test reviews.
"""
    ),
    code(
        """
from predict import load_model, predict as predict_labels, predict_proba

bundle = load_model(ROOT / "model_checkpoint")
print("checkpoint config:")
print(json.dumps(bundle.config, indent=2))
"""
    ),
    code(
        """
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, roc_auc_score)

texts = public_test.text.astype(str).tolist()
y_true = public_test.label.to_numpy()

y_prob = predict_proba(bundle, texts)
y_pred = predict_labels(bundle, texts)

print("Total accuracy    : %.4f" % accuracy_score(y_true, y_pred))
print("Balanced accuracy : %.4f" % balanced_accuracy_score(y_true, y_pred))
print("ROC AUC           : %.4f" % roc_auc_score(y_true, y_prob))
print()
print(classification_report(y_true, y_pred, target_names=["negative (0)", "positive (1)"], digits=4))
"""
    ),
    code(
        """
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = true, cols = predicted):")
print(pd.DataFrame(cm, index=["true negative", "true positive"],
                   columns=["pred negative", "pred positive"]))

fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=15)
ax.set_xticks([0, 1], ["negative", "positive"])
ax.set_yticks([0, 1], ["negative", "positive"])
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("Public test confusion matrix")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
### Reading the confusion matrix

The two off-diagonal cells are the two failure modes. Because the training prior is 3:1
positive, the risk this model was built to avoid is an excess of false *positives* —
negative reviews predicted positive. Comparing the two off-diagonal counts shows how
much of that bias the class weighting and tuned threshold actually removed.
"""
    ),
    md(
        """
## 7. Writing `public_test_predictions.csv`

The submitted file is regenerated here through the same `predict.py` entry point the
grader would call, and checked against the required schema.
"""
    ),
    code(
        """
from predict import write_predictions

out = write_predictions(ROOT / "data" / "public_test.csv",
                        ROOT / "public_test_predictions.csv")

assert list(out.columns) == ["id", "predicted_label"], out.columns
assert len(out) == len(public_test) == 400
assert set(out.predicted_label.unique()) <= {0, 1}
assert out.id.tolist() == public_test.id.tolist()
print("public_test_predictions.csv OK —", len(out), "rows")
out.head()
"""
    ),
    md(
        """
## 8. Limitations and honest scope

- **No pretrained transformer was used.** The plan was to compare against a fine-tuned
  MiniLM encoder with chunked document aggregation, but `transformers` could not be
  installed in this environment, so that arm was dropped rather than reported
  untested. The neural model here is trained from scratch on 240 documents; a pretrained
  language model is the most obvious source of further gains and is the first thing
  Stage 2 lists as future work.
- **The hyper-parameter sweep is small** (two values of `C`, one neural configuration),
  bounded by CPU-only training time. The neural arm used 5-fold CV without repeats,
  so its CV estimate is noisier than the logistic regression estimate.
- **240 training documents** means every CV estimate carries a standard error of roughly
  ±3 points. Differences smaller than that between the candidate models should not be
  treated as meaningful.
"""
    ),
    md(
        """
## References

- B. Pang and L. Lee, *A Sentimental Education: Sentiment Analysis Using Subjectivity
  Summarization Based on Minimum Cuts*, ACL 2004 — the source corpus.
- scikit-learn documentation: `TfidfVectorizer`, `TruncatedSVD`, `LogisticRegression`,
  `RepeatedStratifiedKFold`.
- PyTorch documentation: `AdamW`, `CrossEntropyLoss` class weighting,
  `CosineAnnealingLR`.

## Use of AI

Anthropic's Claude (via Claude Code) was used as a coding assistant to draft the code and
explanatory prose in this repository. All reported numbers were produced by executing that
code against the released data, and the modelling decisions were reviewed and accepted by
the author, who is responsible for this submission.
"""
    ),
]


# ======================================================================================
# Stage 2
# ======================================================================================

stage2 = [
    md(
        """
# Assignment 2 — Stage 2: Hidden Test Evaluation

**CYSE499/650, Summer 2026**

This notebook is **inference and evaluation only**. It contains no training code by
construction: it imports `load_model` / `predict` from `predict.py` and loads the
`model_checkpoint/` directory exactly as submitted for Stage 1. Nothing is retrained,
fine-tuned, refit, or re-thresholded — the decision threshold was fixed during Stage 1
from cross-validated training folds and is read straight out of `config.json`.

**To run:** place the released `hidden_test.csv` in `data/`, then run all cells.
"""
    ),
    code(
        """
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predict import load_model, predict as predict_labels, predict_proba, write_predictions

ROOT = Path.cwd()
HIDDEN = ROOT / "data" / "hidden_test.csv"

assert HIDDEN.exists(), f"Place the released hidden_test.csv at {HIDDEN}"
hidden = pd.read_csv(HIDDEN)
print("hidden_test:", hidden.shape, "| columns:", list(hidden.columns))
"""
    ),
    md(
        """
## 1. Load the frozen Stage 1 checkpoint

The config below is the one committed at the Stage 1 deadline. The `threshold` field in
particular was chosen on training-set cross-validation and has not been touched.
"""
    ),
    code(
        """
bundle = load_model(ROOT / "model_checkpoint")
print(json.dumps(bundle.config, indent=2))
"""
    ),
    md(
        """
## 2. Predict and write `hidden_test_predictions.csv`
"""
    ),
    code(
        """
out = write_predictions(HIDDEN, ROOT / "hidden_test_predictions.csv")

assert list(out.columns) == ["id", "predicted_label"]
assert set(out.predicted_label.unique()) <= {0, 1}
assert out.id.tolist() == hidden.id.tolist()
print("hidden_test_predictions.csv OK —", len(out), "rows")
out.head()
"""
    ),
    md(
        """
## 3. Hidden test accuracy and confusion matrix
"""
    ),
    code(
        """
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, roc_auc_score)

texts = hidden.text.astype(str).tolist()
y_true = hidden.label.to_numpy()
y_prob = predict_proba(bundle, texts)
y_pred = out.predicted_label.to_numpy()

hidden_acc = accuracy_score(y_true, y_pred)
print("Hidden test accuracy    : %.4f" % hidden_acc)
print("Hidden balanced accuracy: %.4f" % balanced_accuracy_score(y_true, y_pred))
print("Hidden ROC AUC          : %.4f" % roc_auc_score(y_true, y_prob))
print()
print(classification_report(y_true, y_pred, target_names=["negative (0)", "positive (1)"], digits=4))
"""
    ),
    code(
        """
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = true, cols = predicted):")
print(pd.DataFrame(cm, index=["true negative", "true positive"],
                   columns=["pred negative", "pred positive"]))

fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=15)
ax.set_xticks([0, 1], ["negative", "positive"])
ax.set_yticks([0, 1], ["negative", "positive"])
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("Hidden test confusion matrix")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 4. Public test vs hidden test
"""
    ),
    code(
        """
results = json.load(open(ROOT / "results.json"))
public = results["public_test"]

comparison = pd.DataFrame(
    {
        "public test": [public["accuracy"], public["balanced_accuracy"],
                        public["macro_f1"], public["roc_auc"], public["n"]],
        "hidden test": [round(hidden_acc, 4),
                        round(balanced_accuracy_score(y_true, y_pred), 4),
                        round(__import__("sklearn.metrics", fromlist=["f1_score"])
                              .f1_score(y_true, y_pred, average="macro"), 4),
                        round(roc_auc_score(y_true, y_prob), 4),
                        len(y_true)],
    },
    index=["accuracy", "balanced accuracy", "macro F1", "ROC AUC", "n"],
)
comparison["difference"] = comparison["hidden test"] - comparison["public test"]
comparison
"""
    ),
    code(
        """
gap = hidden_acc - public["accuracy"]
print("Public test accuracy: %.4f" % public["accuracy"])
print("Hidden test accuracy: %.4f" % hidden_acc)
print("Difference          : %+.4f" % gap)

# Rough significance guide: standard error of an accuracy estimate on n examples.
se = float(np.sqrt(hidden_acc * (1 - hidden_acc) / len(y_true)))
print("\\nApprox. standard error on the hidden-test estimate: +/- %.4f" % se)
print("The gap is %s given that margin."
      % ("within sampling noise" if abs(gap) < 2 * se else "larger than sampling noise"))
"""
    ),
    md(
        """
### Discussion

Both test sets are balanced 50/50 and are drawn from the same Pang & Lee corpus, so a
large gap between them would be surprising and would point at overfitting to the public
set. Since the public test set was used **only** for reporting in Stage 1 — never for
model selection, thresholding, or blending — the hidden-test result is a genuine
held-out estimate and the two should agree to within sampling noise. The cell above
quantifies that margin explicitly rather than eyeballing it.

The threshold is the one place where a gap could plausibly open. It was tuned on
out-of-fold predictions from a 240-document training set, so it carries real estimation
variance; a threshold slightly off-optimal shows up as an asymmetry between the two
off-diagonal cells of the confusion matrix rather than as a uniform accuracy drop.
"""
    ),
    md(
        """
## 5. What I would try next with more time or compute

1. **A pretrained language model, chunked over the full review.** The clearest gap in
   this submission. The reviews have a median length of ~730 words, so the right design
   is not a plain 512-token truncation but overlapping windows fed through a pretrained
   encoder with the per-chunk outputs pooled back to a document score. Fine-tuning a
   small encoder (MiniLM- or DistilBERT-sized) this way should beat a from-scratch model
   trained on 240 documents, because almost all of its language understanding comes from
   pretraining rather than from our tiny label set. This was the original plan and was
   dropped only because `transformers` could not be installed in the environment.

2. **Semi-supervised use of the unlabelled corpus.** The full Pang & Lee corpus has
   2,000 documents; we are given labels for 240. Fitting the TF-IDF vocabulary and the
   SVD basis on all available *text* (which uses no labels and so breaks no rule about
   training data) would give a much better-estimated latent space than 240 documents can
   support.

3. **Proper nested cross-validation.** The threshold and the blend weight are currently
   chosen on the same out-of-fold predictions used to report CV scores, which optimistically
   biases those CV numbers. An outer CV loop would give an unbiased estimate. It costs a
   multiple of the training time, which is why it was skipped on a CPU-only budget.

4. **A wider hyper-parameter sweep.** Two values of `C` and a single neural configuration
   were affordable here. Word/char n-gram ranges, SVD dimensionality, dropout and hidden
   width were all fixed at reasonable defaults rather than searched.

5. **Calibration and error analysis.** Isotonic or Platt calibration of the blended
   probability, plus reading the highest-confidence mistakes, would show whether the
   remaining errors are genuinely ambiguous reviews (mixed verdicts, sarcasm, long plot
   summaries with a one-line judgement) or a systematic failure the model could be
   fixed for.

## Use of AI

As in Stage 1, Anthropic's Claude (via Claude Code) was used as a coding assistant to
draft the code and prose in this notebook. All reported numbers were produced by executing
it against the released `hidden_test.csv` and the frozen Stage 1 checkpoint.
"""
    ),
]


def write(cells, path):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    nbf.write(notebook, str(path))
    print("wrote", path.name, f"({len(cells)} cells)")


if __name__ == "__main__":
    write(stage1, ROOT / "stage1_notebook.ipynb")
    write(stage2, ROOT / "stage2_notebook.ipynb")
