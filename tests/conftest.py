"""
Pytest configuration and fixtures for testing the FinTech Reviews Analytics project.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Sample test data
SAMPLE_REVIEWS = [
    {
        "review_id": "1",
        "app_name": "TestApp",
        "review_text": "Great app, very useful!",
        "cleaned_text": "great app very useful",
        "rating": 5,
        "sentiment_score": 0.8,
        "sentiment": "positive",
        "review_date": "2023-01-01T12:00:00",
        "author": "user1"
    },
    {
        "review_id": "2",
        "app_name": "TestApp",
        "review_text": "Not working properly. Very disappointed.",
        "cleaned_text": "not working properly very disappointed",
        "rating": 2,
        "sentiment_score": -0.7,
        "sentiment": "negative",
        "review_date": "2023-01-02T12:00:00",
        "author": "user2"
    },
    {
        "review_id": "3",
        "app_name": "TestApp",
        "review_text": "It's okay, could be better.",
        "cleaned_text": "its okay could be better",
        "rating": 3,
        "sentiment_score": 0.1,
        "sentiment": "neutral",
        "review_date": "2023-01-03T12:00:00",
        "author": "user3"
    }
]

@pytest.fixture
def sample_reviews_df():
    """Return a DataFrame with sample review data."""
    return pd.DataFrame(SAMPLE_REVIEWS)

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory structure for testing."""
    # Create directories
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "data_dir": data_dir,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir
    }

@pytest.fixture
def mock_config(tmp_path, temp_data_dir):
    """Create a mock configuration for testing."""
    config = {
        "app": {
            "name": "test_app",
            "version": "1.0.0",
            "log_level": "INFO"
        },
        "paths": {
            "raw_data": str(temp_data_dir["raw_dir"]),
            "processed_data": str(temp_data_dir["processed_dir"]),
            "visualizations": str(temp_data_dir["data"] / "visualizations"),
            "reports": str(temp_data_dir["data"] / "reports")
        },
        "scraping": {
            "user_agent": "test_agent",
            "request_delay": 1,
            "max_retries": 3,
            "timeout": 10
        },
        "sentiment": {
            "model": "en_core_web_sm",
            "threshold_positive": 0.2,
            "threshold_negative": -0.2
        },
        "database": {
            "host": "localhost",
            "port": 1521,
            "service_name": "XE",
            "username": "test_user",
            "password": "test_pass",
            "pool_size": 5,
            "max_overflow": 10
        }
    }
    
    # Save config to a temporary file
    import yaml
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)
