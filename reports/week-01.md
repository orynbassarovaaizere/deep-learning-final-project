# Week 1 Report — Fake News Detection System

**Project:** Fake News Detection using RNN, LSTM, and GRU  
**Student:** Aizere O.  
**Date:** Week 1

---

## What Was Completed This Week

- Defined the project topic: binary text classification to detect fake vs. real news articles
- Selected the dataset: **Fake News Detection Dataset** from Kaggle (Fake.csv + True.csv), containing ~44,900 articles
- Set up the GitHub repository with the recommended folder structure: `data/`, `notebooks/`, `src/`, `reports/`, `results/`
- Created `README.md` with project description, dataset download instructions, and setup guide
- Performed initial **Exploratory Data Analysis (EDA)**:
  - Loaded and merged Fake.csv and Real.csv into a single dataframe with binary labels (0 = Fake, 1 = Real)
  - Confirmed balanced class distribution (~23,500 fake, ~21,400 real)
  - Analyzed text length distribution — fake articles average slightly shorter text
  - Visualized class distribution with `seaborn` countplot
  - Checked for missing values (none found)
  - Inspected sample fake and real articles for qualitative understanding

## Important Files Changed / Committed

- `notebooks/01_eda.ipynb` — full EDA notebook
- `data/README.md` — dataset download instructions (Kaggle link, merge steps)
- `README.md` — project overview, setup, and run instructions
- `reports/week-01.md` — this report

## Experiments Run

- No model training this week — focus was on understanding the data
- Computed basic statistics: mean text length, class counts, null values


The dataset is well-balanced and clean — no imputation or resampling needed.

## Problems / Blockers

- Dataset files are large (~60MB each); added `.gitignore` rule to exclude raw CSVs from the repo
- Decided to combine `title` + `text` columns into a single `full_text` feature for richer input to models

## Plan for Next Week

- Implement text preprocessing pipeline (lowercasing, punctuation removal, tokenization)
- Build vocabulary from training set
- Encode texts as integer sequences with padding (MAX_LEN = 300)
- Implement train/validation/test split (70% / 15% / 15%)
- Train TF-IDF + Logistic Regression **baseline model**
- Report baseline metrics: accuracy, precision, recall, F1-score
