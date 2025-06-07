"""
Sentiment analysis module for FinTech reviews.

This module provides functionality to perform sentiment analysis and
thematic analysis on FinTech app reviews.
"""

import pandas as pd
import spacy
import logging
from transformers import pipeline
from scripts.utils import setup_logging

class SentimentAnalyzer:
    """Class to perform sentiment and thematic analysis on reviews."""
    
    def __init__(self, input_path="data/processed/reviews_clean.csv"):
        """Initialize analyzer with data and models."""
        setup_logging()
        self.logger = logging.getLogger(__name__)
        try:
            self.df = pd.read_csv(input_path)
            self.df['review_id'] = self.df.index + 1
            self.nlp = spacy.load('en_core_web_sm')
            self.sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
            self.logger.info("Initialized SentimentAnalyzer with %d reviews", len(self.df))
        except Exception as e:
            self.logger.error("Initialization failed: %s", e)
            raise

    def compute_sentiment(self):
        """Compute sentiment scores for all reviews."""
        try:
            def get_sentiment(text):
                result = self.sentiment_analyzer(text[:512])[0]  # Truncate to 512 tokens
                label = 'neutral' if result['score'] < 0.6 else result['label'].lower()
                score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
                return label, score
            
            self.df['sentiment_label'], self.df['sentiment_score'] = zip(*self.df['review'].apply(get_sentiment))
            self.logger.info("Computed sentiment for %d reviews", len(self.df))
            return self.df
        except Exception as e:
            self.logger.error("Sentiment computation failed: %s", e)
            return None

    def extract_keywords(self):
        """Extract keywords using spaCy."""
        try:
            def get_keywords(text):
                doc = self.nlp(text.lower())
                keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] or token.dep_ == 'compound']
                return keywords if keywords else ['none']
            
            self.df['keywords'] = self.df['review'].apply(get_keywords)
            self.logger.info("Extracted keywords for %d reviews", len(self.df))
            return self.df
        except Exception as e:
            self.logger.error("Keyword extraction failed: %s", e)
            return None

    def assign_themes(self):
        """Assign themes based on keywords."""
        try:
            themes = {
                'Account Access Issues': ['login', 'password', 'authentication', 'access', 'lockout'],
                'Transaction Performance': ['transfer', 'payment', 'slow', 'delay', 'transaction'],
                'User Interface': ['ui', 'design', 'navigation', 'layout', 'interface'],
                'Customer Support': ['support', 'help', 'response', 'service', 'complaint'],
                'Feature Requests': ['feature', 'option', 'budgeting', 'request', 'add']
            }

            def get_themes(keywords):
                assigned = [theme for theme, kws in themes.items() if any(kw in keywords for kw in kws)]
                return assigned if assigned else ['Other']
            
            self.df['themes'] = self.df['keywords'].apply(get_themes)
            self.logger.info("Assigned themes to %d reviews", len(self.df))
            
            # Log themes per bank
            for bank in self.df['bank'].unique():
                bank_themes = self.df[self.df['bank'] == bank]['themes'].explode().value_counts()
                self.logger.info("Themes for %s: %s", bank, bank_themes.to_dict())
            return self.df
        except Exception as e:
            self.logger.error("Theme assignment failed: %s", e)
            return None

    def analyze(self, output_path="data/processed/reviews_analyzed.csv"):
        """Run full analysis pipeline and save results."""
        try:
            self.compute_sentiment()
            self.extract_keywords()
            self.assign_themes()
            output_df = self.df[['review_id', 'review', 'sentiment_label', 'sentiment_score', 'themes']]
            output_df.to_csv(output_path, index=False)
            self.logger.info("Saved %d analyzed reviews to %s", len(output_df), output_path)
            return output_df
        except Exception as e:
            self.logger.error("Analysis failed: %s", e)
            return None

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.analyze()