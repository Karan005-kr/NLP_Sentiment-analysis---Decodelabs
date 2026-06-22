# Project 4: NLP & Sentiment Analysis

This project builds a machine learning pipeline that reads product reviews and predicts whether the sentiment is **Positive** or **Negative**.

## Project Goal

Program a machine to read and mathematically categorize unstructured human text using Natural Language Processing (NLP).

## Key Features

- Text preprocessing pipeline
  - HTML/URL cleaning
  - lowercasing
  - tokenization
  - stop-word removal
  - negation preservation such as `not`, `no`, `never`
  - POS-guided WordNet lemmatization
- TF-IDF vectorization using unigrams and bigrams
- Machine learning models
  - Multinomial Naive Bayes
  - Complement Naive Bayes
  - Linear SVM
- Model evaluation
  - accuracy
  - precision
  - recall
  - F1-score
  - confusion matrix
- Streamlit app for live prediction

## Folder Structure

```text
Project_4_NLP_Sentiment_Analysis/
├── app.py
├── data/
│   └── reviews.csv
├── models/
│   └── best_sentiment_model.pkl
├── outputs/
│   ├── cleaned_reviews.csv
│   ├── metrics.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── sample_predictions.csv
│   └── run_summary.json
├── reports/
│   └── Project_4_NLP_Sentiment_Analysis_Report.pdf
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── Project_4_NLP_Sentiment_Analysis.ipynb
├── requirements.txt
└── README.md
```

## How to Run

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

Windows PowerShell:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train_model.py
```

### 5. Test prediction from terminal

```bash
python src/predict.py "This product is not good and not worth the price"
```

### 6. Run Streamlit app

```bash
streamlit run app.py
```

## Dataset

The file `data/reviews.csv` contains product reviews with two columns:

- `review`: customer review text
- `sentiment`: positive or negative

You can replace this CSV with any real review dataset, but keep the same column names.

## Model Logic

Raw text cannot be used directly by machine learning models. First, text is cleaned and converted into tokens. Then, TF-IDF converts words and phrases into numerical feature weights. Finally, classifiers learn the relationship between the TF-IDF features and sentiment labels.

## Important NLP Point

Default stop-word lists often remove negation words like `not`. This is dangerous in sentiment analysis because:

- `good` = positive
- `not good` = negative

So this project keeps negation words during preprocessing.

## Best Model

After running the project, check:

```text
outputs/metrics.csv
```

The model with the highest F1-score is saved as:

```text
models/best_sentiment_model.pkl
```
