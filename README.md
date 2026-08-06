# Assignment 2 — Sentiment Classification of Movie Reviews

**CYSE499/650, Summer 2026**

Binary sentiment classification (`0 = negative`, `1 = positive`) on the Pang & Lee movie
review polarity corpus, trained on a deliberately small and imbalanced split.

## Result

| Metric | Public test (400 reviews) |
|---|---|
| **Total accuracy** | **0.7775** |
| Balanced accuracy | 0.7775 |
| Macro F1 | 0.7774 |
| ROC AUC | 0.8471 |

Confusion matrix (rows = true, columns = predicted):

|  | pred negative | pred positive |
|---|---|---|
| **true negative** | 151 | 49 |
| **true positive** | 40 | 160 |

Cross-validated balanced accuracy on the training set was 0.8111. The public test set
was scored exactly once, after the model was frozen; it was never used for model
selection, threshold tuning, or blending.

## The model

A weighted blend of two classifiers over a shared TF-IDF representation
(word 1–2 grams + character `char_wb` 3–5 grams, computed over the **entire** review):

| Component | Weight | Description |
|---|---|---|
| Logistic regression | 0.80 | `C=10`, `class_weight="balanced"`, on the sparse TF-IDF vector |
| Neural network (PyTorch) | 0.20 | TF-IDF → truncated SVD (240 dims) → `Dropout → Linear(240→256) → LayerNorm → GELU → Dropout → Linear(256→2)` |

A decision threshold of `0.7501`, chosen to maximise balanced accuracy on out-of-fold
cross-validation predictions, is stored in the checkpoint and applied at inference. That
it sits well above 0.5 is the imbalance correction made visible: a model fit on a 75%
positive training set assigns systematically inflated positive probabilities, so clearing
the positive bar requires more evidence than a naive 0.5 cutoff would demand.

**Why this shape.** Reviews here have a median length of ~730 words and a maximum of
2,570, so a fixed 512-token encoder window would discard most of each document — and in
this corpus the verdict often lands in the final paragraph. A bag-of-n-grams
representation reads the whole review. With only 240 labelled documents, keeping
effective capacity low mattered more than model size.

**Handling the 3:1 class imbalance** (180 positive / 60 negative in training, against
50/50 test sets): class-weighted losses in both components, a decision threshold tuned
for balanced accuracy rather than left at 0.5, balanced accuracy as the model-selection
metric, and repeated stratified cross-validation for every decision.

Full details, including the cross-validated comparison of all candidates, are in
`stage1_notebook.ipynb`.

## Repository layout

```
stage1_notebook.ipynb        Stage 1: development, design rationale, public-test evaluation
stage2_notebook.ipynb        Stage 2: hidden-test inference and evaluation (run when released)
predict.py                   Inference — single source of truth, imported by both notebooks
train.py                     Model selection + final training; writes the checkpoint
build_notebooks.py           Generates the two notebooks
model_checkpoint/            The frozen Stage 1 model (everything needed to reload it)
public_test_predictions.csv  id,predicted_label for the 400 public test reviews
results.json                 CV table + public-test metrics, read by the notebooks
requirements.txt             Pinned dependencies
data/                        train.csv, public_test.csv (as released)
```

## Setup

Requires Python 3.13 (developed on 3.13.14, Windows, CPU only — no GPU needed).

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` pins a CPU build of PyTorch. If `pip` cannot resolve it on your
platform, install PyTorch first from https://pytorch.org/get-started/locally/ (CPU
variant), then re-run the command above.

## Reproducing the predictions

Generate predictions from the committed checkpoint — no retraining, a few seconds:

```bash
python predict.py --input data/public_test.csv --output public_test_predictions.csv
```

Or open `stage1_notebook.ipynb` and run all cells. The notebook loads the checkpoint
rather than retraining, so it completes in seconds.

## Retraining from scratch (optional)

```bash
python train.py
```

Runs the cross-validated model selection, retrains the winner on all 240 training
examples, rewrites `model_checkpoint/`, `public_test_predictions.csv` and `results.json`.
Takes roughly 7 minutes on an 8-core CPU laptop. The random seed is fixed (`20260804`).

> Note: the committed checkpoint differs from what `train.py` emits in two storage-only
> respects, neither of which changes the model:
>
> 1. **The SVD basis is stored in float32**, not the float64 `train.py` writes. This takes
>    `feature_pipeline.joblib` from ~97 MB to a size that clears GitHub's 100 MB file
>    limit. Verified as a maximum probability shift of 2.4e-08 and zero label changes
>    across all 400 public test reviews.
> 2. **The joblib files are stored uncompressed.** Compression saved ~9 MB but cost 90
>    seconds of decompression on every load; uncompressed, the checkpoint loads in 0.4s.
>    Verified bit-identical — maximum probability change exactly 0, zero label changes.
>
> The largest file in `model_checkpoint/` is 56 MB.

## Stage 2

When `hidden_test.csv` is released, place it in `data/` and run all cells of
`stage2_notebook.ipynb`. It writes `hidden_test_predictions.csv` and reports hidden-test
accuracy, the confusion matrix, and a comparison against the public test result.

Stage 2 is inference only. The notebook imports `load_model` / `predict` from
`predict.py` and loads the Stage 1 `model_checkpoint/` unchanged — it contains no
training code, and the decision threshold is read from the checkpoint rather than
re-tuned.

## Limitations

- **No pretrained language model.** The intended comparison included a fine-tuned
  pretrained encoder with chunked document aggregation, but `transformers` could not be
  installed in the development environment, so that arm was dropped rather than reported
  untested. The neural component here is trained from scratch on 240 documents. This is
  the most likely source of further gains and is the first item in Stage 2's future-work
  list.
- **Small hyper-parameter sweep**, bounded by CPU-only training time: two values of `C`
  and a single neural configuration. The neural arm used 5-fold CV without repeats, so
  its estimate is noisier than the logistic regression estimate.
- With 240 training documents, cross-validation estimates carry a standard error of
  roughly ±3 points; differences smaller than that between candidates are not meaningful.

## References

- B. Pang and L. Lee, *A Sentimental Education: Sentiment Analysis Using Subjectivity
  Summarization Based on Minimum Cuts*, ACL 2004 — the source corpus.
- scikit-learn documentation: `TfidfVectorizer`, `TruncatedSVD`, `LogisticRegression`,
  `RepeatedStratifiedKFold`.
- PyTorch documentation: `AdamW`, `CrossEntropyLoss` class weighting, `CosineAnnealingLR`.

## Use of AI

Anthropic's Claude (via Claude Code) was used as a coding assistant to draft the code and
explanatory prose in this repository. All reported numbers were produced by executing that
code against the released data, and the modelling decisions were reviewed and accepted by
the author, who is responsible for this submission.
