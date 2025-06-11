# utils.py: Utility functions for logging

import logging
import os

def setup_logging(log_file="fintech-reviews-analytics.log"):
    """Set up logging with UTF-8 encoding."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),  # Use UTF-8 for file
                logging.StreamHandler()  # Console output
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info("Logging initialized with UTF-8 encoding")
    except Exception as e:
        print(f"Failed to set up logging: {e}")