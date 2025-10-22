import re
import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from wordfreq import zipf_frequency
from datasets import Dataset
from sklearn.utils import resample

def clean_review(text):
    if not isinstance(text, str): return ""
    sentences = sent_tokenize(text)
    cleaned = []
    threshold = 2.5
    for s in sentences:
        tokens = word_tokenize(s)
        valid = []
        for w in tokens:
            wl = w.lower()
            freq = zipf_frequency(wl, 'en')
            if ((re.match(r"[A-Za-z]+'[A-Za-z]+", w) and zipf_frequency(w.replace("'", ""), 'en') > threshold)
                or (w.isalpha() and freq > threshold)
                or w in [".", ",", "!", "?"]):
                valid.append(w)
        cleaned_sent = " ".join(valid)
        cleaned.append(re.sub(r'\s+([.,!?])', r'\1', cleaned_sent))
    return " ".join(cleaned).strip()

def preprocess_reviews(df):
    print("Cleaning and balancing")
    df["Cleaned_Reviews"] = df["Generated_Reviews"].fillna("").apply(clean_review)
    df = df[df["Cleaned_Reviews"].str.strip() != ""]
    df["Satisfaction_Score"] = df["Satisfaction_Score"].astype(int)

    min_class = df["Satisfaction_Score"].value_counts().min()
    df = (
        df.groupby("Satisfaction_Score", group_keys=False)
        .apply(lambda x: x.sample(min_class, replace=True))
        .sample(frac=1, random_state=42)
    )
    df["labels"] = df["Satisfaction_Score"].astype(int) - 1
    print("Balanced class distribution:\n", df["Satisfaction_Score"].value_counts())
    return df