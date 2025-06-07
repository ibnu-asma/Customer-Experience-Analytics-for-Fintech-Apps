"""
Data preprocessing module for FinTech reviews.

This module provides functionality to clean, transform, and prepare
raw review data for analysis.
"""

import logging
import re
import string
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewPreprocessor:
    """Class for preprocessing FinTech app reviews."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """Initialize the preprocessor with configuration.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
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
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data.
        
        Args:
            text: Input text to clean.
            
        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the reviews DataFrame.
        
        Args:
            df: Input DataFrame containing raw reviews.
            
        Returns:
            pd.DataFrame: Processed DataFrame with additional features.
        """
        logger.info("Starting review preprocessing...")
        
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Clean text
        if 'review_text' in df_processed.columns:
            df_processed['cleaned_text'] = df_processed['review_text'].apply(self.clean_text)
        
        # Convert date columns to datetime if they exist
        date_columns = ['review_date', 'date', 'timestamp']
        for col in date_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
        
        # Extract additional features if possible
        if 'review_date' in df_processed.columns:
            df_processed['review_year'] = df_processed['review_date'].dt.year
            df_processed['review_month'] = df_processed['review_date'].dt.month
        
        logger.info(f"Preprocessing complete. Processed {len(df_processed)} reviews.")
        return df_processed
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """Save processed data to a CSV file.
        
        Args:
            df: Processed DataFrame to save.
            filename: Name for the output file.
            
        Returns:
            str: Path to the saved file.
        """
        try:
            # Create processed data directory if it doesn't exist
            processed_dir = Path(self.config['paths']['processed_data'])
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filepath
            filepath = processed_dir / f"{filename}_processed.csv"
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            logger.info(f"Saved processed data to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

def main():
    """Main function to run the preprocessing pipeline."""
    try:
        preprocessor = ReviewPreprocessor()
        
        # Example: Process raw data files
        raw_data_dir = Path(preprocessor.config['paths']['raw_data'])
        processed_data_dir = Path(preprocessor.config['paths']['processed_data'])
        
        # Create processed directory if it doesn't exist
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each raw data file
        for raw_file in raw_data_dir.glob('*.csv'):
            try:
                logger.info(f"Processing {raw_file.name}...")
                df = pd.read_csv(raw_file)
                
                # Preprocess the data
                df_processed = preprocessor.preprocess_reviews(df)
                
                # Save the processed data
                output_filename = raw_file.stem.replace('_raw', '')
                preprocessor.save_processed_data(df_processed, output_filename)
                
            except Exception as e:
                logger.error(f"Error processing {raw_file.name}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Fatal error in preprocessing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
