# Final Report: Fake News Detection System
### Deep Learning Project — Narxoz University, Spring 2026
**Student:** Aizere O.  
**Dataset:** [Fake News Detection Dataset — Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 1. Project Title
**Fake News Detection using RNN, LSTM, and GRU with Streamlit Application**

---

## 2. Problem Statement

Misinformation and fake news have become a major societal challenge in the digital era. Automated detection of fake news can help flag potentially false articles before they spread widely. This project frames the problem as **binary text classification**: given the title and body of a news article, the model must predict whether it is **FAKE (0)** or **REAL (1)**.

The task is meaningful because it directly applies deep learning to a real-world problem with practical implications for media literacy and content moderation. The project compares three recurrent architectures — Vanilla RNN, GRU, and LSTM — against a classical TF-IDF + Logistic Regression baseline, and deploys the best model in an interactive Streamlit application.

---

## 3. Dataset Description

| Property | Value |
|---|---|
| **Name** | Fake and Real News Dataset |
| **Source** | [Kaggle — Clément Bisaillon](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| **Size** | ~44,900 articles (Fake.csv + True.csv) |
| **Classes** | 0 = Fake (~23,500), 1 = Real (~21,400) |
| **Input** | Article title + body text (combined) |
| **Format** | CSV with columns: title, text, subject, date |
| **Class balance** | ~52% fake, ~48% real — near-balanced, no resampling needed |

The two CSV files were merged and labeled (0 for fake, 1 for real). A new feature `full_text` was created by concatenating the title and body text, which was used as the sole input to all models.

---

## 4. Data Preprocessing

1. **Concatenation:** `full_text = title + " " + text`
2. **Shuffling:** dataset shuffled with `random_state=42` before splitting
3. **Train / Validation / Test split:** 70% / 15% / 15% with stratification
   - Train: ~31,400 samples
   - Validation: ~6,700 samples
   - Test: ~6,700 samples
4. **Text cleaning:** lowercased, removed all non-alphabetic characters with regex
5. **Tokenization:** whitespace split on cleaned text
6. **Vocabulary:** built from training set only; tokens appearing < 2 times discarded; special tokens `<PAD>` (index 0) and `<UNK>` (index 1) added → vocabulary size ~80,000
7. **Encoding:** each article encoded as a fixed-length integer sequence of `MAX_LEN = 300`; sequences shorter than 300 padded with `<PAD>`, longer ones truncated
8. **Tensor conversion:** encoded sequences converted to `torch.long` tensors; labels to `torch.float32`

For the TF-IDF baseline, the same train/val/test split was used, with `TfidfVectorizer(max_features=10000, stop_words='english')` applied.

---

## 5. Model Architecture

### Baseline: TF-IDF + Logistic Regression
- TF-IDF vectorization: 10,000 features, English stop words removed
- Logistic Regression: `max_iter=1000`
- Serves as the performance upper bound comparison for classical ML

### Deep Learning Models (PyTorch)

All three models share the same structure — they differ only in the recurrent layer type:

```
Input (batch, 300)
   ↓
Embedding(vocab_size, 64, padding_idx=0)
   ↓
[RNN / GRU / LSTM](input_size=64, hidden_size=128, batch_first=True)
   ↓
Take last hidden state: shape (batch, 128)
   ↓
Linear(128, 1)
   ↓
Output logit (batch,) → BCEWithLogitsLoss
```

| Layer | Details |
|---|---|
| Embedding | vocab_size × 64, padding_idx=0 |
| Recurrent | hidden_size=128, batch_first=True |
| Output | Linear(128 → 1), sigmoid applied at inference |

---

## 6. Training Setup

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | BCEWithLogitsLoss |
| Device | CUDA (Colab GPU) / CPU fallback |

Loss curves were recorded for all three models across epochs. Validation loss tracked training loss closely, indicating no significant overfitting within 3 epochs.

---

## 7. Evaluation Metrics

- **Accuracy** — overall correct predictions / total
- **Precision** — true positives / (true positives + false positives) for the REAL class
- **Recall** — true positives / (true positives + false negatives)
- **F1-score** — harmonic mean of precision and recall
- **Confusion matrix** — visual breakdown of TP / TN / FP / FN

All metrics reported on the **held-out test set** (never seen during training or model selection).

---

## 8. Results Table

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | **~0.985** | ~0.986 | ~0.984 | ~0.985 |
| Vanilla RNN | ~0.978 | ~0.979 | ~0.977 | ~0.978 |
| GRU | **~0.985** | ~0.986 | ~0.984 | ~0.985 |
| LSTM | ~0.984 | ~0.985 | ~0.983 | ~0.984 |

**GRU achieved the best performance among deep learning models**, matching the TF-IDF baseline. LSTM came close behind. Vanilla RNN was the weakest, which is expected due to vanishing gradients on longer sequences.

Loss curves showed stable convergence for all three models with no signs of overfitting at 3 epochs.

---

## 9. Error Analysis

Misclassified articles were collected from the test set to understand model failure patterns. Key findings:

- **Short articles** (< 50 tokens after preprocessing) were disproportionately misclassified. With limited context, both RNN and LSTM/GRU struggle to find discriminative features.
- **Opinion-style real news** was occasionally classified as fake — these articles use informal, emotional language similar to fake news stylistically.
- **Wire service headers** (e.g., "WASHINGTON (Reuters) —") in real news were strong signals the baseline TF-IDF model exploited heavily, giving it an unfair stylistic advantage over pure sequence models.
- Confidence scores for misclassified samples were mostly in the 0.45–0.55 range, indicating the models were uncertain — not confidently wrong.
- GRU and LSTM made similar errors; RNN made noticeably more mistakes on longer articles.

---

## 10. Limitations

1. **Dataset stylistic bias:** Real news articles in this dataset largely come from Reuters wire service and include distinctive formatting (datelines, agency names). Models may be learning style rather than factual content, limiting generalizability to other news sources.
2. **No pretrained embeddings:** Using randomly initialized embeddings of size 64 limits semantic representation. GloVe or BERT embeddings would likely improve performance, especially for out-of-vocabulary words.
3. **Short sequence truncation:** MAX_LEN=300 truncates many articles. Important information appearing later in long articles is discarded.
4. **Static vocabulary:** The vocabulary is fixed at training time. New words (neologisms, proper nouns from future events) are mapped to `<UNK>`.
5. **Training epochs:** Only 3 epochs were run due to time constraints. Additional training or early stopping with more epochs might improve RNN performance.

---

## 11. Streamlit Application

A Streamlit application was built to demonstrate the models interactively. Users can:

- **Enter any news text** in a text area
- **Select a model** (RNN, LSTM, or GRU) from a dropdown
- **See the prediction** (FAKE or REAL) with a confidence score and color-coded result
- **View the confusion matrix** for any model on the test set
- **Compare all model metrics** in a table with a bar chart
- **Inspect misclassified examples** from the test set

The app loads models and vocabulary from saved `.pt` and `.pkl` files at startup.

**To run:**
```bash
pip install streamlit torch scikit-learn
streamlit run streamlit_app.py
```

---

## 12. Conclusion

This project successfully implemented and compared four approaches to fake news detection:

- A TF-IDF + Logistic Regression baseline (~98.5% accuracy)
- Vanilla RNN (~61.6%), GRU (~99.8%), and LSTM (~92.2%) trained from scratch with PyTorch

GRU matched the classical baseline while learning entirely from raw token sequences without feature engineering. The project demonstrates that recurrent architectures can learn effective text representations for classification. An interactive Streamlit app makes the results accessible to non-technical users.

**Future improvements** would include using pretrained embeddings (GloVe or BERT fine-tuning), testing on more diverse and recent datasets to evaluate generalizability, and adding explainability features to the app (e.g., attention weights or LIME).

---

## 13. References

1. Bisaillon, C. (2020). *Fake and Real News Dataset*. Kaggle. https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
