# Week 1 — Exploratory Data Analysis & Preprocessing in NLP

## Overview
End-to-end EDA and text preprocessing pipeline on the Twitter US Airline Sentiment dataset. The notebook explores class distributions, tweet length patterns, top bigrams, and constructs a robust 7-step preprocessing pipeline ready for downstream NLP modeling.

## Dataset
- **Source:** [Kaggle — Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Size:** ~14,000 tweets
- **Classes:** Negative (62%), Neutral (21%), Positive (17%)
- **Avg tweet length:** ~15 words

## What This Project Does
- Loads and inspects raw tweet data for quality issues
- Visualises sentiment class distribution and tweet length histograms
- Extracts top bigrams to identify dominant complaint patterns (e.g. "flight delay", "poor service")
- Builds a reusable 7-step preprocessing pipeline:
  1. Lowercasing
  2. URL & @mention removal (regex)
  3. Punctuation & special character removal
  4. Tokenization
  5. Stopword removal
  6. Lemmatization (preferred over stemming)
  7. Emoji-to-token conversion & slang expansion

## Key Findings
- **Strong class imbalance** — negative tweets dominate; SMOTE resampling recommended before modeling
- **High noise** — URLs, mentions, emojis require a robust preprocessing pipeline
- **Short text** (~15 words avg) favours TF-IDF with n-grams or transformer embeddings
- **Complaint bigrams** ("flight delay", "poor service") are strong classification features

## Project Structure
```
week1/
├── Week1_NLP_EDA_Preprocessing.ipynb   # Main notebook
├── requirements.txt
└── README.md
```

## Setup & Run
```bash
pip install -r requirements.txt
jupyter notebook Week1_NLP_EDA_Preprocessing.ipynb
```

## Requirements
See `requirements.txt`

## Skills Demonstrated
`pandas` · `NLTK` · `spaCy` · `matplotlib` · `seaborn` · `regex` · EDA · Text preprocessing
