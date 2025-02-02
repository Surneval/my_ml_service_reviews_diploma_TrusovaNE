# scripts/preprocess_data.py

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import html

def clean_text(text: str) -> str:
    """Clean the text by removing HTML tags, lowercasing, and removing punctuation."""
    # unescape HTML entities
    text = html.unescape(text)
    # remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # lowercase
    text = text.lower()
    # remove punctuation except for basic characters
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # strip extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    data_path = os.path.join("data", "IMDB Dataset.csv")
    df = pd.read_csv(data_path)
    # label encoding: positive -> 1, negative -> 0
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df['clean_review'] = df['review'].apply(clean_text)

    # drop rows with empty text after cleaning
    df = df[df['clean_review'].str.strip() != ""]

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    print("Data preprocessed and split into train and val sets.")

if __name__ == "__main__":
    main()

