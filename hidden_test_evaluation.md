# Stage 2 — Hidden Test Evaluation

**CYSE499/650 Assignment 2**

Predictions were generated from the Stage 1 checkpoint without retraining, fine-tuning,
or re-tuning the decision threshold. `stage2_notebook.ipynb` imports `load_model` and
`predict` from `predict.py` and reads the threshold out of
`model_checkpoint/config.json`, so the model that produced these numbers is byte-for-byte
the one submitted at Stage 1.

> **File name note.** The instructions refer to the release as `hidden_test.csv`; it was
> published as `hidden_test_with_labels.csv`. The notebook accepts either name.

## Result

| Metric | Hidden test (600 reviews) |
|---|---|
| **Total accuracy** | **0.7733** |
| Balanced accuracy | 0.7733 |
| Macro F1 | 0.7720 |
| ROC AUC | 0.8457 |

Confusion matrix (rows = true, columns = predicted):

|  | pred negative | pred positive |
|---|---|---|
| **true negative** | 209 | 91 |
| **true positive** | 45 | 255 |

Per class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative (0) | 0.8228 | 0.6967 | 0.7545 | 300 |
| positive (1) | 0.7370 | 0.8500 | 0.7895 | 300 |

The hidden set is balanced 300/300 and was verified disjoint from both `train.csv` and
`public_test.csv` (zero shared `source_file` entries).

## Public test vs hidden test

| Metric | Public test | Hidden test | Difference |
|---|---|---|---|
| Accuracy | 0.7775 | 0.7733 | −0.0042 |
| Balanced accuracy | 0.7775 | 0.7733 | −0.0042 |
| Macro F1 | 0.7774 | 0.7720 | −0.0054 |
| ROC AUC | 0.8471 | 0.8457 | −0.0014 |
| n | 400 | 600 | +200 |

**The two agree.** Accuracy differs by 0.42 points against a standard error of ±1.71
points on the hidden-test estimate, so the gap is comfortably inside sampling noise. All
four metrics move in the same direction by a negligible amount.

This is the outcome the Stage 1 protocol was built to produce. `public_test.csv` was used
**only** for reporting — never for model selection, threshold tuning, or blend weights,
all of which were decided on cross-validated training folds. That made the public set an
honest held-out estimate already, so the hidden set had nothing new to reveal. Had the
public set been used for tuning, the hidden number would be expected to fall noticeably
below it.

The hidden set is also 50% larger, so it is the more reliable of the two estimates.

## Where the errors remain

The residual bias runs toward **false positives**: 91 negative reviews called positive
against 45 positive reviews called negative, a 2.02:1 ratio. This is exactly the
direction the 3:1 positive training prior predicts, and it shows up as weaker recall on
the negative class (0.6967) than on the positive class (0.8500).

The imbalance handling removed most but not all of that skew. The model predicts positive
on 57.7% of the hidden set — much closer to the true 50.0% than to the training set's
75.0%, but still short of neutral. The decision threshold of 0.7501 was estimated from
out-of-fold predictions over just 240 documents and therefore carries real variance; a
cutoff slightly too low pushes borderline reviews positive, producing precisely this
asymmetry between the off-diagonal cells rather than a uniform accuracy drop.

Closing that gap is the most concrete improvement available: more labelled negatives, or
a nested cross-validation loop to estimate the threshold more stably.

## What I would try next

1. **A pretrained language model, chunked over the full review.** The clearest gap in
   this submission. Reviews here have a median length of ~706 words, so the right design
   is overlapping windows through a pretrained encoder with per-chunk outputs pooled to a
   document score — not a plain 512-token truncation. Fine-tuning a small encoder this
   way should beat a model trained from scratch on 240 documents, since almost all of its
   language understanding would come from pretraining rather than from our tiny label
   set. This was the original plan and was dropped only because `transformers` could not
   be installed in the development environment.

2. **Semi-supervised use of the unlabelled corpus.** The full Pang & Lee corpus holds
   2,000 documents; we have labels for 240. Fitting the TF-IDF vocabulary and the SVD
   basis on all available *text* uses no labels and breaks no rule about training data,
   but would give a far better-estimated latent space than 240 documents can support.
   Note that the SVD is currently saturated — it is capped at 240 components by the
   document count, not by the configured 300.

3. **Nested cross-validation for the threshold and blend weight.** Both are currently
   chosen on the same out-of-fold predictions used to report CV scores, which
   optimistically biases those CV numbers — visible here as CV balanced accuracy of
   0.8111 against 0.7733 on the hidden set. An outer loop would give an unbiased estimate
   and, more usefully, a more stable threshold. It costs a multiple of the training time,
   which is why it was skipped on a CPU-only budget.

4. **A wider hyper-parameter sweep.** Two values of `C` and a single neural configuration
   were affordable. The n-gram ranges, SVD dimensionality, dropout, and hidden width were
   all fixed at reasonable defaults rather than searched.

5. **Calibration and error analysis.** Isotonic or Platt calibration of the blended
   probability, plus reading the highest-confidence mistakes, would show whether the
   remaining errors are genuinely ambiguous reviews — mixed verdicts, sarcasm, long plot
   summaries with a one-line judgement — or a systematic failure that could be fixed.

## Use of AI

Anthropic's Claude (via Claude Code) was used as a coding assistant to draft the code and
explanatory prose in this repository. All reported numbers were produced by executing that
code against the released data, and the modelling decisions were reviewed and accepted by
the author, who is responsible for this submission.
