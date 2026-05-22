# Week 3 Report — Fake News Detection System

**Project:** Fake News Detection using RNN, LSTM, and GRU  
**Student:** Aizere O.  
**Date:** Week 3

---

## What Was Completed This Week

- Implemented all three deep learning models in PyTorch:
  - **RNNModel**: `Embedding(vocab_size, 64)` → `nn.RNN(64, 128)` → `Linear(128, 1)`
  - **GRUModel**: `Embedding(vocab_size, 64)` → `nn.GRU(64, 128)` → `Linear(128, 1)`
  - **LSTMModel**: `Embedding(vocab_size, 64)` → `nn.LSTM(64, 128)` → `Linear(128, 1)`
- Built a unified `train_model()` function that:
  - Trains for 3 epochs
  - Uses `BCEWithLogitsLoss` as the loss function (appropriate for binary classification)
  - Uses Adam optimizer with `lr=0.001`
  - Records train and validation loss per epoch
  - Plots loss curves after training
- Built an `evaluate_model()` function that returns accuracy, precision, recall, F1, confusion matrix, raw predictions, and true labels
- Ran all three models and recorded test-set results
- Produced a **model comparison table** and **confusion matrix plots** for RNN, GRU, and LSTM
- Began building **error analysis** — collected misclassified samples with their true labels, predicted labels, and confidence scores

## Important Files Changed / Committed

- `notebooks/03_deep_learning_models.ipynb` — RNN/GRU/LSTM training and evaluation
- `src/models.py` — model class definitions
- `src/train.py` — train_model() and evaluate_model() functions
- `results/rnn_loss_curve.png` — training/val loss for RNN
- `results/gru_loss_curve.png` — training/val loss for GRU
- `results/lstm_loss_curve.png` — training/val loss for LSTM
- `results/comparison_table.csv` — numerical comparison of all models
- `reports/week-03.md` — this report

## Experiments Run

**Training setup:** 3 epochs, batch size 32, Adam lr=0.001, BCEWithLogitsLoss, MAX_LEN=300

### Test Set Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| TF-IDF + LR (baseline) | ~0.985 | ~0.986 | ~0.984 | ~0.985 |
| RNN | ~0.978 | ~0.979 | ~0.977 | ~0.978 |
| GRU | ~0.985 | ~0.986 | ~0.984 | ~0.985 |
| LSTM | ~0.984 | ~0.985 | ~0.983 | ~0.984 |

### Key Observations

- **GRU** matched or slightly exceeded LSTM, likely because the dataset sequences are relatively short after truncation to 300 tokens and GRU's simpler gating is sufficient
- **RNN** performed slightly below LSTM/GRU — expected, as vanilla RNNs suffer from the vanishing gradient problem on longer sequences
- All three models converged within 3 epochs without obvious overfitting (validation loss tracked training loss closely)
- Deep learning models reached the same accuracy level as the TF-IDF baseline, confirming that the task is well-learned from raw token sequences

## Problems / Blockers

- Training on CPU was slow (~15 min per model for 3 epochs on 31,000 sequences); used Colab GPU to speed up
- The strong baseline makes the absolute improvement small in terms of numbers, but the architectural comparison and the interactive app still justify the deep learning approach
- Some misclassified articles are short (<50 words), which makes sense as context is limited for any model

## Plan for Next Week

- Perform full **error analysis**: inspect top misclassified samples, analyze patterns
- Write **final report** (markdown)
- Build working **Streamlit app** that loads saved models and vocab, accepts user text, and shows prediction + confidence
- Test app locally and document setup instructions
- Finalize GitHub repository: clean up notebooks, check README, commit all results
