"""
Sentiment and Thematic Analysis for FinTech Reviews

This module provides functionality to perform sentiment analysis using both
DistilBERT and VADER, along with thematic analysis using TF-IDF and rule-based
theming for FinTech app reviews.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

# Sentiment Analysis
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Text Processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Utils
from scripts.utils import setup_logging, timeit

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Constants
THEMES_CONFIG = {
    'Account Access Issues': ['login', 'password', 'authentication', 'access', 'lockout', 'account'],
    'Transaction Performance': ['transfer', 'payment', 'slow', 'delay', 'transaction', 'failed', 'pending'],
    'User Interface': ['ui', 'design', 'navigation', 'layout', 'interface', 'screen', 'button', 'menu'],
    'Customer Support': ['support', 'help', 'response', 'service', 'complaint', 'contact', 'assistance'],
    'App Performance': ['crash', 'freeze', 'bug', 'error', 'performance', 'speed', 'lag'],
    'Feature Requests': ['feature', 'option', 'budgeting', 'request', 'add', 'include', 'want']
}

SENTIMENT_THRESHOLDS = {
    'positive': 0.6,
    'negative': 0.4,
    'neutral': (0.4, 0.6)
}

class SentimentAnalyzer:
    """Class to perform sentiment and thematic analysis on reviews with VADER fallback."""
    
    def __init__(self, input_path="data/processed/reviews_clean.csv"):
        """Initialize analyzer with data and models."""
        self.logger = logging.getLogger(__name__)
        try:
            # Load and prepare data
            self.df = pd.read_csv(input_path)
            self.df['review_id'] = self.df.index + 1
            
            # Initialize NLP components
            self.nlp = spacy.load('en_core_web_sm')
            
            # Initialize sentiment analyzers
            self.distilbert_analyzer = pipeline(
                'sentiment-analysis',
                model='distilbert-base-uncased-finetuned-sst-2-english',
                device=-1  # Use CPU for compatibility
            )
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize TF-IDF Vectorizer
            self.tfidf = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            self.logger.info("Initialized SentimentAnalyzer with %d reviews", len(self.df))
            
        except Exception as e:
            self.logger.error("Initialization failed: %s", e, exc_info=True)
            raise

    @timeit
    def compute_sentiment(self, use_vader_fallback=True):
        """
        Compute sentiment scores for all reviews using DistilBERT with VADER fallback.
        
        Args:
            use_vader_fallback: Whether to use VADER if DistilBERT fails
            
        Returns:
            DataFrame: DataFrame with added sentiment columns
        """
        def distilbert_sentiment(text):
            """Get sentiment using DistilBERT."""
            try:
                result = self.distilbert_analyzer(text[:512])[0]  # Truncate to 512 tokens
                label = 'positive' if result['label'] == 'POSITIVE' else 'negative'
                score = result['score']
                return label, score, 'distilbert'
            except Exception as e:
                self.logger.warning("DistilBERT failed: %s", str(e))
                return None, None, 'error'
                
        def vader_sentiment(text):
            """Get sentiment using VADER as fallback."""
            try:
                scores = self.vader_analyzer.polarity_scores(text)
                compound = scores['compound']
                
                if compound >= SENTIMENT_THRESHOLDS['positive']:
                    return 'positive', compound, 'vader'
                elif compound <= SENTIMENT_THRESHOLDS['negative']:
                    return 'negative', compound, 'vader'
                else:
                    return 'neutral', compound, 'vader'
            except Exception as e:
                self.logger.warning("VADER failed: %s", str(e))
                return 'neutral', 0.0, 'error'
        
        try:
            self.logger.info("Computing sentiment for %d reviews...", len(self.df))
            
            # Initialize result columns
            self.df['sentiment_label'] = None
            self.df['sentiment_score'] = 0.0
            self.df['sentiment_source'] = 'none'
            
            # Process in batches to handle memory
            batch_size = 50
            for i in tqdm(range(0, len(self.df), batch_size), desc="Processing sentiment"):
                batch = self.df.iloc[i:i + batch_size]
                
                # First try DistilBERT
                results = batch['review'].apply(distilbert_sentiment)
                
                # Fall back to VADER if needed
                if use_vader_fallback:
                    for j, (label, score, source) in enumerate(results):
                        if label is None or source == 'error':
                            idx = i + j
                            if idx < len(self.df):
                                vader_label, vader_score, _ = vader_sentiment(self.df.at[idx, 'review'])
                                self.df.at[idx, 'sentiment_label'] = vader_label
                                self.df.at[idx, 'sentiment_score'] = vader_score
                                self.df.at[idx, 'sentiment_source'] = 'vader'
                
                # Update successful DistilBERT results
                for j, (label, score, source) in enumerate(results):
                    if label is not None and source != 'error':
                        idx = i + j
                        if idx < len(self.df):
                            self.df.at[idx, 'sentiment_label'] = label
                            self.df.at[idx, 'sentiment_score'] = score
                            self.df.at[idx, 'sentiment_source'] = source
            
            # Fill any remaining None values with neutral
            self.df['sentiment_label'] = self.df['sentiment_label'].fillna('neutral')
            
            self.logger.info("Sentiment distribution:\n%s", 
                           self.df['sentiment_label'].value_counts())
            self.logger.info("Sentiment sources:\n%s", 
                           self.df['sentiment_source'].value_counts())
            
            return self.df
            
        except Exception as e:
            self.logger.error("Sentiment computation failed: %s", str(e), exc_info=True)
            raise

    @timeit
    def extract_keywords(self, max_keywords=10):
        """
        Extract keywords using TF-IDF and spaCy.
        
        Args:
            max_keywords: Maximum number of keywords to extract per review
            
        Returns:
            DataFrame: DataFrame with added keywords column
        """
        try:
            self.logger.info("Extracting keywords using TF-IDF...")
            
            # Get TF-IDF features
            tfidf_matrix = self.tfidf.fit_transform(self.df['review'])
            feature_names = self.tfidf.get_feature_names_out()
            
            # Convert to dense matrix for easier manipulation
            dense = tfidf_matrix.todense()
            
            # Get top keywords for each document
            keywords_list = []
            for i in range(len(self.df)):
                # Get indices of top N TF-IDF scores
                top_indices = np.argsort(dense[i].A1)[-max_keywords:][::-1]
                # Get corresponding feature names
                keywords = [feature_names[idx] for idx in top_indices if dense[i, idx] > 0]
                keywords_list.append(keywords)
            
            # Add to dataframe
            self.df['keywords'] = keywords_list
            self.logger.info("Extracted keywords for %d reviews", len(self.df))
            
            # Also extract named entities for additional context
            self.logger.info("Extracting named entities...")
            self.df['entities'] = self.df['review'].apply(
                lambda x: [(ent.text, ent.label_) 
                          for ent in self.nlp(x).ents]
            )
            
            return self.df
            
        except Exception as e:
            self.logger.error("Keyword extraction failed: %s", str(e), exc_info=True)
            raise

    @timeit
    def assign_themes(self, min_theme_score=0.2):
        """
        Assign themes based on keywords and entities using a scoring system.
        
        Args:
            min_theme_score: Minimum score required to assign a theme
            
        Returns:
            DataFrame: DataFrame with added themes column
        """
        try:
            self.logger.info("Assigning themes to reviews...")
            
            # Enhanced theme configuration with weights
            theme_weights = {
                'Account Access Issues': {
                    'login': 0.8, 'password': 0.9, 'authentication': 0.7, 
                    'access': 0.6, 'lockout': 0.8, 'account': 0.5
                },
                'Transaction Performance': {
                    'transfer': 0.8, 'payment': 0.7, 'slow': 0.9, 
                    'delay': 0.9, 'transaction': 0.6, 'failed': 0.8, 'pending': 0.7
                },
                'User Interface': {
                    'ui': 0.9, 'design': 0.7, 'navigation': 0.8, 
                    'layout': 0.7, 'interface': 0.8, 'screen': 0.6, 'button': 0.5
                },
                'Customer Support': {
                    'support': 0.9, 'help': 0.8, 'response': 0.7, 
                    'service': 0.6, 'complaint': 0.9, 'contact': 0.5
                },
                'App Performance': {
                    'crash': 1.0, 'freeze': 0.9, 'bug': 0.8, 
                    'error': 0.7, 'performance': 0.6, 'speed': 0.5, 'lag': 0.8
                },
                'Feature Requests': {
                    'feature': 0.9, 'option': 0.7, 'budgeting': 0.8, 
                    'request': 0.9, 'add': 0.6, 'include': 0.7, 'want': 0.5
                }
            }
            
            def score_themes(keywords, entities):
                """Calculate theme scores based on keywords and entities."""
                scores = {theme: 0.0 for theme in theme_weights}
                
                # Score based on keywords
                for word in keywords:
                    for theme, weights in theme_weights.items():
                        if word in weights:
                            scores[theme] += weights[word]
                
                # Boost scores for named entities
                for entity, label in entities:
                    entity = entity.lower()
                    for theme, weights in theme_weights.items():
                        if entity in weights:
                            scores[theme] += weights[entity] * 1.5  # Higher weight for entities
                
                # Normalize scores
                max_score = max(scores.values()) if scores else 0
                if max_score > 0:
                    scores = {k: v/max_score for k, v in scores.items()}
                
                # Get themes above threshold
                return [
                    theme for theme, score in scores.items() 
                    if score >= min_theme_score
                ] or ['Other']
            
            # Apply theme scoring
            self.df['themes'] = self.df.apply(
                lambda row: score_themes(row['keywords'], row.get('entities', [])), 
                axis=1
            )
            
            # Log theme distribution
            self.logger.info("Assigned themes to %d reviews", len(self.df))
            theme_counts = self.df['themes'].explode().value_counts()
            self.logger.info("Overall theme distribution:\n%s", theme_counts)
            
            # Log themes per bank
            for bank in self.df['bank'].unique():
                bank_themes = self.df[self.df['bank'] == bank]['themes'].explode().value_counts()
                self.logger.info("Themes for %s:\n%s", bank, bank_themes)
            
            return self.df
            
        except Exception as e:
            self.logger.error("Theme assignment failed: %s", str(e), exc_info=True)
            raise

    @timeit
    def analyze(self, output_dir="data/processed"):
        """
        Run full analysis pipeline and save results.
        
        Args:
            output_dir: Directory to save output files
            
        Returns:
            dict: Analysis results and metrics
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Run analysis pipeline
            self.logger.info("Starting analysis pipeline...")
            
            # 1. Sentiment Analysis
            self.compute_sentiment(use_vader_fallback=True)
            
            # 2. Keyword and Entity Extraction
            self.extract_keywords(max_keywords=15)
            
            # 3. Theme Assignment
            self.assign_themes(min_theme_score=0.2)
            
            # Prepare output data
            output_columns = [
                'review_id', 'review', 'cleaned_review', 'rating', 'date', 'bank',
                'sentiment_label', 'sentiment_score', 'sentiment_source',
                'keywords', 'entities', 'themes'
            ]
            
            # Save full results
            output_path = os.path.join(output_dir, 'reviews_analyzed.csv')
            self.df[output_columns].to_csv(output_path, index=False)
            
            # Save summary statistics
            summary = {
                'total_reviews': len(self.df),
                'sentiment_distribution': self.df['sentiment_label'].value_counts().to_dict(),
                'sentiment_sources': self.df['sentiment_source'].value_counts().to_dict(),
                'theme_distribution': self.df['themes'].explode().value_counts().to_dict(),
                'bank_breakdown': {
                    bank: {
                        'count': len(group),
                        'sentiment': group['sentiment_label'].value_counts().to_dict(),
                        'top_themes': group['themes'].explode().value_counts().head(5).to_dict()
                    }
                    for bank, group in self.df.groupby('bank')
                }
            }
            
            # Save summary as JSON
            summary_path = os.path.join(output_dir, 'analysis_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info("Analysis complete. Results saved to %s", output_dir)
            self.logger.info("Sentiment distribution: %s", summary['sentiment_distribution'])
            
            return summary
            
        except Exception as e:
            self.logger.error("Analysis failed: %s", str(e), exc_info=True)
            raise

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.analyze()