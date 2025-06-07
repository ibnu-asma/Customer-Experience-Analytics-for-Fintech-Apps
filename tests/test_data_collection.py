"""
Unit tests for the data collection module.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path

# Import the module to test
from scripts import data_collection

class TestReviewScraper:
    """Test cases for the ReviewScraper class."""
    
    @pytest.fixture
    def mock_scraper(self, mock_config):
        """Create a ReviewScraper instance with a mock config."""
        return data_collection.ReviewScraper(mock_config)
    
    def test_init(self, mock_scraper):
        """Test initialization of the ReviewScraper."""
        assert mock_scraper.base_delay == 1
        assert mock_scraper.max_retries == 3
        assert mock_scraper.timeout == 10
    
    @patch('scripts.data_collection.requests.get')
    def test_scrape_reviews_success(self, mock_get, mock_scraper):
        """Test successful review scraping."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "reviews": [
                {"id": "1", "text": "Great app!", "rating": 5, "date": "2023-01-01"},
                {"id": "2", "text": "Not good", "rating": 2, "date": "2023-01-02"}
            ]
        }
        mock_get.return_value = mock_response
        
        # Test the method
        result = mock_scraper.scrape_reviews("test.app", max_reviews=2)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "review_text" in result.columns
        assert "rating" in result.columns
    
    @patch('scripts.data_collection.time.sleep')
    @patch('scripts.data_collection.requests.get')
    def test_scrape_reviews_retry(self, mock_get, mock_sleep, mock_scraper):
        """Test retry mechanism on failed requests."""
        # Mock failed responses
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"reviews": []}
        
        # First two calls fail, third succeeds
        mock_get.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success
        ]
        
        # Test the method
        result = mock_scraper.scrape_reviews("test.app")
        
        # Assertions
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # Should sleep between retries
        assert isinstance(result, pd.DataFrame)
    
    def test_save_reviews(self, mock_scraper, temp_data_dir):
        """Test saving reviews to a CSV file."""
        # Create test data
        test_reviews = pd.DataFrame([
            {"review_text": "Great!", "rating": 5, "review_date": "2023-01-01"},
            {"review_text": "Bad!", "rating": 1, "review_date": "2023-01-02"}
        ])
        
        # Test saving
        output_path = mock_scraper.save_reviews(test_reviews, "TestApp")
        
        # Assertions
        assert Path(output_path).exists()
        assert "TestApp_reviews_" in output_path
        assert output_path.endswith(".csv")
        
        # Verify the saved data
        saved_data = pd.read_csv(output_path)
        assert len(saved_data) == 2
        assert "review_text" in saved_data.columns
        assert "rating" in saved_data.columns

class TestMainFunction:
    """Test cases for the main function and entry points."""
    
    @patch('scripts.data_collection.ReviewScraper')
    def test_main_success(self, mock_scraper_class, mock_config):
        """Test the main function with successful execution."""
        # Setup mock
        mock_scraper = MagicMock()
        mock_scraper.scrape_reviews.return_value = pd.DataFrame([{"text": "test"}])
        mock_scraper_class.return_value = mock_scraper
        
        # Call the main function
        result = data_collection.main()
        
        # Assertions
        assert result == 0
        assert mock_scraper.scrape_reviews.called
        assert mock_scraper.save_reviews.called
    
    @patch('scripts.data_collection.ReviewScraper')
    def test_main_exception(self, mock_scraper_class, mock_config):
        """Test the main function with an exception."""
        # Setup mock to raise an exception
        mock_scraper = MagicMock()
        mock_scraper.scrape_reviews.side_effect = Exception("Test error")
        mock_scraper_class.return_value = mock_scraper
        
        # Call the main function
        with pytest.raises(Exception):
            data_collection.main()
            
        # Assert that save_reviews was not called after exception
        assert not mock_scraper.save_reviews.called
