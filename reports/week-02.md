# Week 2 Report — Fake News Detection System

**Project:** Fake News Detection using RNN, LSTM, and GRU  
**Student:** Aizere O.  
**Date:** Week 2

---

## What Was Completed This Week

- Implemented the full **text preprocessing pipeline**:
  - Combined `title` and `text` columns into a unified `full_text` feature
  - Applied lowercasing and removal of non-alphabetic characters using regex
  - Built a vocabulary dictionary from the training set; kept only tokens appearing ≥ 2 times
  - Added `<PAD>` and `<UNK>` special tokens
- Performed **train / validation / test split** (70% / 15% / 15%) with stratification to preserve class balance
- Implemented and evaluated the **TF-IDF + Logistic Regression baseline model**:
  - TF-IDF vectorizer with `max_features=10,000` and English stop word removal
  - Logistic Regression with `max_iter=1000`
  - Evaluated on validation and test sets
- Wrote baseline evaluation code including confusion matrix visualization


## Experiments Run

**Baseline: TF-IDF + Logistic Regression**

| Split | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Validation | ~0.985 | ~0.985 | ~0.985 | ~0.985 |
| Test | ~0.985 | ~0.986 | ~0.984 | ~0.985 |

The baseline achieved very strong performance on this dataset — this is expected because TF-IDF captures distinctive vocabulary patterns (specific political keywords, news agency names) that strongly differ between fake and real news.

## Results So Far

- Baseline TF-IDF + LR achieves ~98.5% accuracy on the test set
- Confusion matrix shows minimal false positives and false negatives
- Vocabulary size built from training set: ~80,000 unique tokens (after min_count=2 filter)
- Encoded sequence tensors ready for PyTorch models: shape `[N, 300]`

## Problems / Blockers

- The baseline is very strong due to the structured nature of the dataset (real news comes from specific wire services like Reuters, which the model may be picking up on). This means deep learning models may not significantly outperform TF-IDF, but the architectural comparison is still valid and instructive.
- Decided to use `MAX_LEN=300` tokens — longer sequences caused memory pressure in local testing.

## Plan for Next Week

- Implement three deep learning models in PyTorch:
  - **Vanilla RNN** (embedding → RNN → linear)
  - **GRU** (embedding → GRU → linear)
  - **LSTM** (embedding → LSTM → linear)
- Train each model for 3–5 epochs with `BCEWithLogitsLoss` and `Adam` optimizer (lr=0.001)
- Compare training and validation loss curves for all three models
- Evaluate on the test set and produce a comparison table
