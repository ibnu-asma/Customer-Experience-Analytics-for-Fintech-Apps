"""
Sentiment analysis module for FinTech reviews.

This module provides functionality to perform sentiment analysis and
thematic analysis on FinTech app reviews.
"""

import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from textblob import TextBlob
import spacy
from collections import Counter
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Class for performing sentiment and thematic analysis on reviews."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """Initialize the sentiment analyzer with configuration.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self.nlp = self._load_spacy_model()
        self.sentiment_thresholds = self.config['sentiment']
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            dict: Configuration dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise
    
    def _load_spacy_model(self):
        """Load spaCy model for NLP tasks."""
        try:
            model_name = self.config['sentiment'].get('model', 'en_core_web_sm')
            return spacy.load(model_name)
        except OSError:
            logger.warning(f"{model_name} not found. Downloading...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            return spacy.load(model_name)
    
    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment of a given text.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            tuple: (polarity, sentiment_label)
        """
        if not text or not isinstance(text, str):
            return 0.0, "neutral"
            
        # Use TextBlob for sentiment analysis
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        # Categorize sentiment
        if polarity > self.sentiment_thresholds.get('threshold_positive', 0.2):
            sentiment = "positive"
        elif polarity < self.sentiment_thresholds.get('threshold_negative', -0.2):
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return polarity, sentiment
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """Extract key phrases from text using spaCy.
        
        Args:
            text: Input text.
            top_n: Number of key phrases to return.
            
        Returns:
            list: List of key phrases.
        """
        if not text or not isinstance(text, str):
            return []
            
        doc = self.nlp(text)
        
        # Extract noun chunks and named entities
        key_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        key_phrases.extend([ent.text.lower() for ent in doc.ents])
        
        # Count and return most common phrases
        return [phrase for phrase, _ in Counter(key_phrases).most_common(top_n)]
    
    def analyze_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform sentiment analysis on a DataFrame of reviews.
        
        Args:
            df: DataFrame containing reviews with a 'cleaned_text' column.
            
        Returns:
            pd.DataFrame: DataFrame with added sentiment analysis columns.
        """
        logger.info("Starting sentiment analysis...")
        
        # Make a copy to avoid modifying the original
        df_analyzed = df.copy()
        
        # Analyze sentiment
        if 'cleaned_text' in df_analyzed.columns:
            # Calculate sentiment scores
            sentiment_results = df_analyzed['cleaned_text'].apply(
                lambda x: self.analyze_sentiment(x) if pd.notnull(x) else (0.0, "neutral")
            )
            
            # Unpack results
            df_analyzed['sentiment_score'], df_analyzed['sentiment'] = zip(*sentiment_results)
            
            # Extract key phrases
            df_analyzed['key_phrases'] = df_analyzed['cleaned_text'].apply(
                lambda x: self.extract_key_phrases(x) if pd.notnull(x) else []
            )
        
        logger.info(f"Sentiment analysis complete. Analyzed {len(df_analyzed)} reviews.")
        return df_analyzed
    
    def save_analysis_results(self, df: pd.DataFrame, filename: str) -> str:
        """Save analysis results to a CSV file.
        
        Args:
            df: Analyzed DataFrame to save.
            filename: Name for the output file.
            
        Returns:
            str: Path to the saved file.
        """
        try:
            # Create analysis results directory if it doesn't exist
            analysis_dir = Path(self.config['paths']['processed_data']) / 'analysis'
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filepath
            filepath = analysis_dir / f"{filename}_analyzed.csv"
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            logger.info(f"Saved analysis results to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            raise

def main():
    """Main function to run the sentiment analysis pipeline."""
    try:
        analyzer = SentimentAnalyzer()
        
        # Example: Process preprocessed data files
        processed_dir = Path(analyzer.config['paths']['processed_data'])
        
        # Process each preprocessed file
        for proc_file in processed_dir.glob('*_processed.csv'):
            try:
                logger.info(f"Analyzing {proc_file.name}...")
                df = pd.read_csv(proc_file)
                
                # Perform sentiment analysis
                df_analyzed = analyzer.analyze_reviews(df)
                
                # Save the analyzed data
                output_filename = proc_file.stem.replace('_processed', '')
                analyzer.save_analysis_results(df_analyzed, output_filename)
                
            except Exception as e:
                logger.error(f"Error analyzing {proc_file.name}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Fatal error in sentiment analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
