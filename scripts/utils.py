# This shared utility module sets up logging, which both data_collection.py and preprocessing.py will use. It’s included here as it’s a dependency for the scripts below.
import logging

def setup_logging():
    """Configure logging for the project."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fintech-reviews-analytics.log'),
            logging.StreamHandler()
        ]
    )