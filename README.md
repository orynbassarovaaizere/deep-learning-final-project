Fake News Detection Using Deep Learning Models

Problem Statement:
The goal of this project is to build a model that can automatically detect whether a news article is fake or real based on its text.
This problem is important because misinformation spreads quickly online and can influence public opinion
The model will take the text of a news article as input and predict whether it is fake or real.

Dataset:
The dataset used in this project is the Fake and Real News Dataset from Kaggle
Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
The dataset consists of two files: Fake.csv and True.csv, containing fake and real news articles.
In total, the dataset includes approximately 40,000 news articles.
Input features:
title (news headline) 
text (main content of the article) 
Target label: 0 = fake, 1 = real 
Data format: CSV files

Planned Method 

Baseline model: Logistic Regression using simple text vectorization 

Deep learning model: Long Short-Term Memory (LSTM) network for sequence modeling. (RNN / LSTM / GRU)

Loss function: Binary Cross Entropy Loss

Evaluation metrics:
Confusion Matrix/Accuracy

Data split plan:
70% training
15% validation
15% test

Expected Challenges:
The dataset may contain noisy or inconsistent text, which can affect model performance.
Some news articles may be ambiguous and difficult to classify correctly.
The model may overfit if not properly regularized.
Training deep learning models may require significant computational resources.
Text preprocessing and sequence length handling may also be challenging.

Weekly Plan:

Week 1: Dataset preparation, repository setup, exploratory data analysis

Week 2: Text preprocessing, train/validation/test split, baseline model

Week 3: LSTM model training and experiments
Week 4: Model improvement, final evaluation, report writing and presentation

