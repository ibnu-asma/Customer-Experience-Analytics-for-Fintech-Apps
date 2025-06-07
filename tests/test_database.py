"""
Unit tests for the database storage module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import the module to test
from scripts import database_storage

# Sample test data
SAMPLE_REVIEW = {
    "review_id": "test_review_123",
    "app_name": "TestApp",
    "app_version": "1.0.0",
    "review_text": "This is a test review.",
    "cleaned_text": "this is a test review",
    "rating": 4,
    "sentiment_score": 0.8,
    "sentiment": "positive",
    "review_date": "2023-01-01T12:00:00",
    "author": "test_user"
}

class TestDatabaseManager:
    """Test cases for the DatabaseManager class."""
    
    @pytest.fixture
    def mock_db_manager(self, mock_config, tmp_path):
        """Create a DatabaseManager instance with a mock config and in-memory SQLite DB."""
        # Override database config to use SQLite in-memory for testing
        with patch.dict(mock_config['database'], {
            'dialect': 'sqlite',
            'database': ':memory:',
            'username': '',
            'password': '',
            'host': '',
            'port': None,
            'service_name': ''
        }):
            # Create the database manager
            db_manager = database_storage.DatabaseManager(mock_config)
            
            # Create tables
            database_storage.Base.metadata.create_all(db_manager.engine)
            
            yield db_manager
            
            # Clean up
            database_storage.Base.metadata.drop_all(db_manager.engine)
    
    def test_initialization(self, mock_db_manager):
        """Test database initialization and table creation."""
        # Check that the engine was created
        assert mock_db_manager.engine is not None
        
        # Check that tables were created
        with mock_db_manager.engine.connect() as conn:
            tables = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
            
            table_names = {t[0] for t in tables}
            assert 'fintech_reviews' in table_names
    
    def test_save_reviews(self, mock_db_manager):
        """Test saving reviews to the database."""
        # Create test data
        test_review = SAMPLE_REVIEW.copy()
        test_df = pd.DataFrame([test_review])
        
        # Save the review
        saved_count = mock_db_manager.save_reviews(test_df, test_review['app_name'])
        
        # Check the result
        assert saved_count == 1
        
        # Verify the data was saved
        with mock_db_manager.Session() as session:
            result = session.query(database_storage.Review).filter_by(
                review_id=test_review['review_id']
            ).first()
            
            assert result is not None
            assert result.app_name == test_review['app_name']
            assert result.rating == test_review['rating']
            assert result.sentiment == test_review['sentiment']
    
    def test_save_duplicate_review(self, mock_db_manager):
        """Test that duplicate reviews are updated rather than inserted."""
        # Create and save initial review
        test_review = SAMPLE_REVIEW.copy()
        test_df = pd.DataFrame([test_review])
        
        # Save first time
        mock_db_manager.save_reviews(test_df, test_review['app_name'])
        
        # Update the review and save again
        updated_review = test_review.copy()
        updated_review['rating'] = 1  # Change the rating
        updated_review['sentiment'] = 'negative'
        updated_review['sentiment_score'] = -0.5
        
        updated_df = pd.DataFrame([updated_review])
        saved_count = mock_db_manager.save_reviews(updated_df, updated_review['app_name'])
        
        # Should still only have one record (updated)
        assert saved_count == 1
        
        # Verify the update
        with mock_db_manager.Session() as session:
            result = session.query(database_storage.Review).filter_by(
                review_id=test_review['review_id']
            ).first()
            
            assert result.rating == updated_review['rating']
            assert result.sentiment == updated_review['sentiment']
    
    def test_get_reviews(self, mock_db_manager):
        """Test retrieving reviews from the database."""
        # Add test data
        test_review = SAMPLE_REVIEW.copy()
        test_df = pd.DataFrame([test_review])
        mock_db_manager.save_reviews(test_df, test_review['app_name'])
        
        # Retrieve reviews
        result_df = mock_db_manager.get_reviews(limit=10)
        
        # Check the result
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1
        assert result_df.iloc[0]['review_id'] == test_review['review_id']
        assert result_df.iloc[0]['app_name'] == test_review['app_name']
    
    def test_get_reviews_with_filter(self, mock_db_manager):
        """Test retrieving reviews with app name filter."""
        # Add test data for two different apps
        review1 = SAMPLE_REVIEW.copy()
        review2 = SAMPLE_REVIEW.copy()
        review2['review_id'] = 'test_review_456'
        review2['app_name'] = 'DifferentApp'
        
        test_df = pd.DataFrame([review1, review2])
        mock_db_manager.save_reviews(test_df, "TestApp")  # app_name parameter is overridden by the data
        
        # Filter for one app
        result_df = mock_db_manager.get_reviews(app_name='TestApp')
        
        # Should only get reviews for the specified app
        assert len(result_df) == 1
        assert result_df.iloc[0]['app_name'] == 'TestApp'

class TestDatabaseEdgeCases:
    """Test edge cases and error handling in the database module."""
    
    @pytest.fixture
    def db_manager(self, mock_config):
        with patch('sqlalchemy.create_engine') as mock_engine:
            # Mock the database engine
            mock_engine.return_value = create_engine('sqlite:///:memory:')
            
            # Create the database manager
            dbm = database_storage.DatabaseManager(mock_config)
            
            # Create tables
            database_storage.Base.metadata.create_all(dbm.engine)
            
            yield dbm
            
            # Clean up
            database_storage.Base.metadata.drop_all(dbm.engine)
    
    def test_save_empty_dataframe(self, db_manager):
        """Test saving an empty DataFrame."""
        empty_df = pd.DataFrame()
        saved_count = db_manager.save_reviews(empty_df, "TestApp")
        assert saved_count == 0
    
    def test_save_invalid_data(self, db_manager):
        """Test saving data with missing required fields."""
        # Missing required 'review_id' field
        invalid_review = {"app_name": "TestApp", "rating": 5}
        test_df = pd.DataFrame([invalid_review])
        
        with pytest.raises(Exception):
            db_manager.save_reviews(test_df, "TestApp")
    
    def test_connection_error(self, mock_config):
        """Test behavior when database connection fails."""
        with patch('sqlalchemy.create_engine') as mock_engine:
            # Make engine creation raise an exception
            mock_engine.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception):
                database_storage.DatabaseManager(mock_config)

class TestDatabaseIntegration:
    """Integration tests for the database module."""
    
    @pytest.fixture
    def db_manager(self, mock_config, tmp_path):
        # Use a temporary SQLite database file
        db_path = tmp_path / "test.db"
        with patch.dict(mock_config['database'], {
            'dialect': 'sqlite',
            'database': str(db_path),
            'username': '',
            'password': '',
            'host': '',
            'port': None,
            'service_name': ''
        }):
            # Create the database manager
            dbm = database_storage.DatabaseManager(mock_config)
            
            # Create tables
            database_storage.Base.metadata.create_all(dbm.engine)
            
            yield dbm
            
            # Clean up
            database_storage.Base.metadata.drop_all(dbm.engine)
    
    def test_end_to_end_workflow(self, db_manager, sample_reviews_df):
        """Test the complete database workflow with sample data."""
        # Ensure we have cleaned text and sentiment for the test
        if 'cleaned_text' not in sample_reviews_df.columns:
            sample_reviews_df['cleaned_text'] = sample_reviews_df['review_text'].str.lower()
        if 'sentiment' not in sample_reviews_df.columns:
            sample_reviews_df['sentiment'] = 'neutral'
            sample_reviews_df['sentiment_score'] = 0.0
        
        # Save reviews to the database
        saved_count = db_manager.save_reviews(sample_reviews_df, "TestApp")
        assert saved_count == len(sample_reviews_df)
        
        # Retrieve the reviews
        retrieved_df = db_manager.get_reviews(limit=100)
        
        # Check that we got the same number of reviews back
        assert len(retrieved_df) == len(sample_reviews_df)
        
        # Check that the data matches
        for _, original_row in sample_reviews_df.iterrows():
            review_id = original_row['review_id']
            retrieved_row = retrieved_df[retrieved_df['review_id'] == review_id].iloc[0]
            
            # Check that the data was saved and retrieved correctly
            assert retrieved_row['app_name'] == original_row['app_name']
            assert retrieved_row['rating'] == original_row['rating']
            
            # Check that dates were properly handled
            if pd.notna(original_row.get('review_date')):
                assert pd.to_datetime(retrieved_row['review_date']) == pd.to_datetime(original_row['review_date'])
        
        # Test updating a review
        updated_review = sample_reviews_df.iloc[0].copy()
        updated_review['rating'] = 1  # Change the rating
        updated_review['sentiment'] = 'negative'
        updated_review['sentiment_score'] = -0.8
        
        # Save the update
        updated_count = db_manager.save_reviews(
            pd.DataFrame([updated_review]), 
            updated_review['app_name']
        )
        assert updated_count == 1
        
        # Verify the update
        updated_retrieved = db_manager.get_reviews(
            app_name=updated_review['app_name']
        )
        updated_row = updated_retrieved[
            updated_retrieved['review_id'] == updated_review['review_id']
        ].iloc[0]
        
        assert updated_row['rating'] == updated_review['rating']
        assert updated_row['sentiment'] == updated_review['sentiment']
