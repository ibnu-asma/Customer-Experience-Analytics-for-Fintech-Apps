"""
Unit tests for the data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module to test
from scripts import preprocessing

class TestReviewPreprocessor:
    """Test cases for the ReviewPreprocessor class."""
    
    @pytest.fixture
    def mock_preprocessor(self, mock_config):
        """Create a ReviewPreprocessor instance with a mock config."""
        return preprocessing.ReviewPreprocessor(mock_config)
    
    def test_clean_text(self, mock_preprocessor):
        """Test text cleaning functionality."""
        # Test cases
        test_cases = [
            ("Great app! It's awesome.", "great app its awesome"),
            ("Check out https://example.com", "check out"),
            ("Hello123 world!", "hello world"),
            ("<html>Test</html>", "test"),
            ("  extra   spaces  ", "extra spaces"),
            ("", ""),
            (None, ""),
            (123, "")
        ]
        
        for input_text, expected in test_cases:
            assert mock_preprocessor.clean_text(input_text) == expected
    
    def test_preprocess_reviews(self, mock_preprocessor):
        """Test the main preprocessing pipeline."""
        # Create test data
        test_data = pd.DataFrame([
            {"review_text": "Great app! It's awesome.", "rating": 5, "review_date": "2023-01-01"},
            {"review_text": "Not good at all!", "rating": 2, "review_date": "2023-01-02"},
            {"review_text": "", "rating": 4, "review_date": None}  # Edge case with empty text
        ])
        
        # Process the data
        processed_df = mock_preprocessor.preprocess_reviews(test_data)
        
        # Assertions
        assert isinstance(processed_df, pd.DataFrame)
        assert "cleaned_text" in processed_df.columns
        assert len(processed_df) == len(test_data)
        
        # Check text cleaning
        assert "awesome" in processed_df.iloc[0]["cleaned_text"]
        assert "good" in processed_df.iloc[1]["cleaned_text"]
        assert processed_df.iloc[2]["cleaned_text"] == ""  # Empty input should remain empty
        
        # Check date conversion
        assert pd.api.types.is_datetime64_any_dtype(processed_df["review_date"])
        
        # Check additional features
        if "review_year" in processed_df.columns:
            assert processed_df["review_year"].iloc[0] == 2023
    
    def test_save_processed_data(self, mock_preprocessor, temp_data_dir):
        """Test saving processed data to a file."""
        # Create test data
        test_data = pd.DataFrame([
            {"review_text": "Test 1", "cleaned_text": "test 1", "rating": 5},
            {"review_text": "Test 2", "cleaned_text": "test 2", "rating": 4}
        ])
        
        # Test saving
        output_path = mock_preprocessor.save_processed_data(test_data, "test_output")
        
        # Assertions
        assert Path(output_path).exists()
        assert "test_output_processed.csv" in output_path
        
        # Verify the saved data
        saved_data = pd.read_csv(output_path)
        assert len(saved_data) == 2
        assert "cleaned_text" in saved_data.columns
        assert "test 1" in saved_data["cleaned_text"].values

class TestPreprocessingEdgeCases:
    """Test edge cases and error handling in preprocessing."""
    
    @pytest.fixture
    def preprocessor(self, mock_config):
        return preprocessing.ReviewPreprocessor(mock_config)
    
    def test_empty_dataframe(self, preprocessor):
        """Test with an empty DataFrame."""
        empty_df = pd.DataFrame()
        result = preprocessor.preprocess_reviews(empty_df)
        assert result.empty
    
    def test_missing_columns(self, preprocessor):
        """Test with missing required columns."""
        # DataFrame missing 'review_text' column
        df = pd.DataFrame({"rating": [1, 2, 3]})
        with pytest.raises(KeyError):
            preprocessor.preprocess_reviews(df)
    
    def test_save_to_nonexistent_dir(self, preprocessor, tmp_path):
        """Test saving to a non-existent directory."""
        test_data = pd.DataFrame([{"test": "data"}])
        non_existent_dir = tmp_path / "nonexistent" / "subdir"
        
        with patch.object(preprocessor.config['paths'], 'processed_data', str(non_existent_dir)):
            output_path = preprocessor.save_processed_data(test_data, "test")
            assert Path(output_path).parent.exists()  # Should create the directory
            assert Path(output_path).exists()

class TestPreprocessingIntegration:
    """Integration tests for the preprocessing module."""
    
    def test_end_to_end_processing(self, mock_preprocessor, sample_reviews_df):
        """Test the complete preprocessing pipeline with sample data."""
        # Process the sample data
        processed_df = mock_preprocessor.preprocess_reviews(sample_reviews_df)
        
        # Basic assertions
        assert not processed_df.empty
        assert len(processed_df) == len(sample_reviews_df)
        assert "cleaned_text" in processed_df.columns
        
        # Check that all reviews were processed
        assert processed_df["cleaned_text"].notna().all()
        
        # Check that the text was cleaned
        for idx, row in sample_reviews_df.iterrows():
            if pd.notna(row["review_text"]):
                # Text should be lowercased
                assert row["review_text"].lower() in ' '.join(processed_df["cleaned_text"].dropna())
        
        # Check that the output can be saved
        output_path = mock_preprocessor.save_processed_data(processed_df, "integration_test")
        assert Path(output_path).exists()
        
        # Clean up
        Path(output_path).unlink()
