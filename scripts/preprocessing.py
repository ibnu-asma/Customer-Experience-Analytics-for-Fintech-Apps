"""
This script preprocesses reviews with the following steps:
1. Removes duplicates and handles missing data
2. Normalizes dates and text data
3. Cleans and processes review text
4. Adds additional text-based features
"""
import os
import re
import string
import pandas as pd
import logging
import emoji
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download
from typing import Optional, List, Dict, Any



# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")
    raise

from utils import setup_logging

class TextPreprocessor:
    """Handles text cleaning and preprocessing for review data."""
    
    def __init__(self):
        """Initialize the text preprocessor with required components."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.english_words = set(nltk.corpus.words.words())
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Convert emojis to text
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove punctuation and special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Remove non-English words and single characters
        tokens = [word for word in tokens if word in self.english_words and len(word) > 1]
        
        return ' '.join(tokens)
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """
        Extract various features from text.
        
        Args:
            text: Input text
            
        Returns:
            dict: Dictionary containing text features
        """
        if not isinstance(text, str):
            text = str(text)
            
        words = text.split()
        char_count = len(text)
        word_count = len(words)
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'avg_word_length': round(avg_word_length, 2),
            'has_question': 1 if '?' in text else 0,
            'has_exclamation': 1 if '!' in text else 0,
            'is_all_caps': 1 if text.isupper() else 0
        }


class ReviewPreprocessor:
    """Class to preprocess Google Play Store review data."""
    
    def __init__(self):
        """Initialize preprocessor with logging and text processor."""
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.text_processor = TextPreprocessor()

    def preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess review text and add text-based features.
        
        Args:
            df: Input DataFrame with review text
            
        Returns:
            DataFrame: DataFrame with processed text and new features
        """
        self.logger.info("Starting text preprocessing...")
        
        # Clean review text
        df['cleaned_review'] = df['review'].apply(self.text_processor.clean_text)
        
        # Extract text features
        text_features = df['cleaned_review'].apply(
            lambda x: pd.Series(self.text_processor.extract_text_features(x))
        )
        
        # Add text features to DataFrame
        df = pd.concat([df, text_features], axis=1)
        
        # Add sentiment polarity (placeholder - could use VADER or TextBlob)
        df['sentiment_polarity'] = 0.0  # Will be implemented in future
        
        self.logger.info("Text preprocessing completed")
        return df
    
    def preprocess(self, input_path="data/raw/reviews.csv", output_path="data/processed/reviews_clean.csv"):
        """
        Preprocess reviews with comprehensive cleaning and feature engineering.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save processed data
            
        Returns:
            DataFrame: Processed DataFrame or None if error occurs
        """
        try:
            # Load raw data
            df = pd.read_csv(input_path)
            self.logger.info("Loaded %d reviews from %s", len(df), input_path)

            # Remove duplicates
            initial_len = len(df)
            df = df.drop_duplicates(subset=['review', 'bank', 'date'], keep='first')
            self.logger.info("Removed %d duplicates, %d reviews remain", 
                           initial_len - len(df), len(df))

            # Handle missing data
            df['review'] = df['review'].fillna('').astype(str)
            df['rating'] = df['rating'].fillna(0).astype(int)
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            
            # Drop rows with missing dates
            missing_dates = df['date'].isna().sum()
            df = df.dropna(subset=['date'])
            self.logger.info("Dropped %d reviews with missing dates", missing_dates)
            
            # Process text data
            df = self.preprocess_text(df)

            # Validate data quality
            missing_data_percentage = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            if missing_data_percentage > 5:
                self.logger.warning("Missing data percentage (%.2f%%) exceeds 5%% threshold", 
                                  missing_data_percentage)

            # Save processed data
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_path, index=False)
            self.logger.info("Saved %d processed reviews to %s", len(df), output_path)
            
            # Save a sample for inspection
            sample_path = os.path.join(output_dir, 'reviews_sample.csv')
            df.sample(min(100, len(df))).to_csv(sample_path, index=False)
            self.logger.info("Saved sample of %d reviews to %s", 
                           min(100, len(df)), sample_path)
            
            return df
        except Exception as e:
            self.logger.error("Error in preprocessing: %s", e)
            return None

if __name__ == "__main__":
    preprocessor = ReviewPreprocessor()
    preprocessor.preprocess()
