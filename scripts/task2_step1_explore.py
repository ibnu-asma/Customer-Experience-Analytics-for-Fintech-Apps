# task2_step1_explore.py: Basic exploration for Task 2, Step 1
# Purpose: Verify input data and test sentiment/keyword libraries

import pandas as pd
import spacy
from transformers import pipeline
from utils import setup_logging
import logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def load_data(input_path="data/processed/reviews_clean.csv"):
    """Load and verify cleaned reviews, subsetting to expected columns."""
    try:
        df = pd.read_csv(input_path)
        expected_columns = ['review', 'rating', 'date', 'bank', 'source']
        available_columns = [col for col in expected_columns if col in df.columns]
        if len(available_columns) != len(expected_columns):
            missing = set(expected_columns) - set(available_columns)
            logger.warning("Missing columns: %s. Using available: %s", missing, available_columns)
        df = df[available_columns]
        logger.info("Loaded %d reviews from %s with columns %s", len(df), input_path, available_columns)
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Missing Values:\n", df.isna().sum())
        print("Bank Counts:\n", df['bank'].value_counts())
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        return None

def test_sentiment():
    """Test DistilBERT sentiment analysis on a sample review."""
    try:
        sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        sample_text = "Great app, fast transfers"
        result = sentiment_analyzer(sample_text[:512])[0]
        logger.info("Sentiment test: %s -> %s (%.2f)", sample_text, result['label'], result['score'])
        print(f"Sentiment Test: '{sample_text}' -> {result['label']} (Score: {result['score']:.2f})")
    except Exception as e:
        logger.error("Sentiment test failed: %s", e)

def test_keywords():
    """Test spaCy keyword extraction on a sample review."""
    try:
        nlp = spacy.load('en_core_web_sm')
        sample_text = "App crashes during login"
        doc = nlp(sample_text.lower())
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] or token.dep_ == 'compound']
        logger.info("Keyword test: %s -> %s", sample_text, keywords)
        print(f"Keyword Test: '{sample_text}' -> {keywords}")
    except Exception as e:
        logger.error("Keyword extraction failed: %s", e)

if __name__ == "__main__":
    logger.info("Starting Task 2, Step 1 exploration")
    df = load_data()
    if df is not None:
        test_sentiment()
        test_keywords()
    logger.info("Completed Task 2, Step 1 exploration")