# This script preprocesses reviews by removing duplicates, 
# handling missing data, and normalizing dates.
import pandas as pd
import logging
from utils import setup_logging

class ReviewPreprocessor:
    """Class to preprocess Google Play Store review data."""
    
    def __init__(self):
        """Initialize preprocessor with logging."""
        setup_logging()
        self.logger = logging.getLogger(__name__)

    def preprocess(self, input_path="data/raw/reviews.csv", output_path="data/processed/reviews_clean.csv"):
        """Preprocess reviews: remove duplicates, handle missing data, normalize dates."""
        try:
            # Load raw data
            df = pd.read_csv(input_path)
            self.logger.info("Loaded %d reviews from %s", len(df), input_path)

            # Remove duplicates
            initial_len = len(df)
            df = df.drop_duplicates(subset=['review', 'bank', 'date'], keep='first')
            self.logger.info("Removed %d duplicates, %d reviews remain", initial_len - len(df), len(df))

            # Handle missing data
            df['review'] = df['review'].fillna('').astype(str)
            df['rating'] = df['rating'].fillna(0).astype(int)
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            
            # Drop rows with missing dates
            missing_dates = df['date'].isna().sum()
            df = df.dropna(subset=['date'])
            self.logger.info("Dropped %d reviews with missing dates", missing_dates)

            # Validate data quality
            missing_data_percentage = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            if missing_data_percentage > 5:
                self.logger.warning("Missing data percentage (%.2f%%) exceeds 5%% threshold", missing_data_percentage)

            # Save cleaned data
            df.to_csv(output_path, index=False)
            self.logger.info("Saved %d cleaned reviews to %s", len(df), output_path)
            return df
        except Exception as e:
            self.logger.error("Error in preprocessing: %s", e)
            return None

if __name__ == "__main__":
    preprocessor = ReviewPreprocessor()
    preprocessor.preprocess()
