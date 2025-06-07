"""
Unit tests for the sentiment analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY

# Import the module to test
from scripts import sentiment_analysis

class TestSentimentAnalyzer:
    """Test cases for the SentimentAnalyzer class."""
    
    @pytest.fixture
    def mock_analyzer(self, mock_config):
        """Create a SentimentAnalyzer instance with a mock config."""
        with patch('spacy.load') as mock_spacy_load:
            # Mock spaCy model
            mock_nlp = MagicMock()
            mock_spacy_load.return_value = mock_nlp
            
            # Create analyzer with mocked spaCy
            analyzer = sentiment_analysis.SentimentAnalyzer(mock_config)
            
            # Mock NLP pipeline
            mock_doc = MagicMock()
            mock_nlp.return_value = mock_doc
            mock_doc.noun_chunks = [MagicMock(text="user experience"), MagicMock(text="app")]
            mock_doc.ents = [MagicMock(text="FinTech")]
            
            return analyzer
    
    def test_analyze_sentiment(self, mock_analyzer):
        """Test sentiment analysis on different texts."""
        # Test cases: (text, expected_sentiment)
        test_cases = [
            ("I love this app!", "positive"),
            ("This is terrible!", "negative"),
            ("It's okay, I guess.", "neutral"),
            ("", "neutral"),  # Empty string
            (None, "neutral")   # None input
        ]
        
        for text, expected_sentiment in test_cases:
            _, sentiment = mock_analyzer.analyze_sentiment(text)
            assert sentiment == expected_sentiment
    
    def test_extract_key_phrases(self, mock_analyzer):
        """Test key phrase extraction."""
        # Test with valid text
        text = "The user experience of this FinTech app is amazing."
        key_phrases = mock_analyzer.extract_key_phrases(text, top_n=3)
        
        # Should return lowercase phrases
        assert isinstance(key_phrases, list)
        assert len(key_phrases) <= 3
        assert any(phrase in ["user experience", "fintech", "app"] for phrase in key_phrases)
        
        # Test with empty text
        assert mock_analyzer.extract_key_phrases("") == []
        assert mock_analyzer.extract_key_phrases(None) == []
    
    def test_analyze_reviews(self, mock_analyzer, sample_reviews_df):
        """Test the complete review analysis pipeline."""
        # Add a cleaned_text column if not present
        if 'cleaned_text' not in sample_reviews_df.columns:
            sample_reviews_df['cleaned_text'] = sample_reviews_df['review_text'].str.lower()
        
        # Analyze the reviews
        analyzed_df = mock_analyzer.analyze_reviews(sample_reviews_df)
        
        # Assertions
        assert isinstance(analyzed_df, pd.DataFrame)
        assert len(analyzed_df) == len(sample_reviews_df)
        assert "sentiment_score" in analyzed_df.columns
        assert "sentiment" in analyzed_df.columns
        assert "key_phrases" in analyzed_df.columns
        
        # Check sentiment values
        valid_sentiments = ["positive", "neutral", "negative"]
        assert all(sentiment in valid_sentiments for sentiment in analyzed_df["sentiment"].unique())
        
        # Check key_phrases is a list
        assert all(isinstance(phrases, list) for phrases in analyzed_df["key_phrases"])
    
    def test_save_analysis_results(self, mock_analyzer, tmp_path):
        """Test saving analysis results to a file."""
        # Create test data
        test_data = pd.DataFrame([
            {
                "review_text": "Great app!",
                "sentiment_score": 0.8,
                "sentiment": "positive",
                "key_phrases": ["great app"]
            }
        ])
        
        # Test saving
        output_path = mock_analyzer.save_analysis_results(test_data, "test_analysis")
        
        # Assertions
        assert "test_analysis_analyzed.csv" in output_path
        assert Path(output_path).exists()
        
        # Verify the saved data
        saved_data = pd.read_csv(output_path)
        assert len(saved_data) == 1
        assert "sentiment" in saved_data.columns
        assert saved_data.iloc[0]["sentiment"] == "positive"

class TestSentimentEdgeCases:
    """Test edge cases and error handling in sentiment analysis."""
    
    @pytest.fixture
    def analyzer(self, mock_config):
        with patch('spacy.load'):
            return sentiment_analysis.SentimentAnalyzer(mock_config)
    
    def test_empty_dataframe(self, analyzer):
        """Test with an empty DataFrame."""
        empty_df = pd.DataFrame()
        result = analyzer.analyze_reviews(empty_df)
        assert result.empty
    
    def test_missing_columns(self, analyzer):
        """Test with missing required columns."""
        # DataFrame missing 'cleaned_text' column
        df = pd.DataFrame({"review_text": ["test"]})
        with pytest.raises(KeyError):
            analyzer.analyze_reviews(df)
    
    def test_invalid_sentiment_thresholds(self, mock_config):
        """Test with invalid sentiment thresholds."""
        # Modify config to have invalid thresholds
        mock_config["sentiment"]["threshold_positive"] = -1.0
        mock_config["sentiment"]["threshold_negative"] = 1.0  # Invalid: negative > positive
        
        with patch('spacy.load'):
            analyzer = sentiment_analysis.SentimentAnalyzer(mock_config)
            
            # Test that analyze_sentiment still works (should handle invalid thresholds)
            score, sentiment = analyzer.analyze_sentiment("test")
            assert sentiment in ["positive", "neutral", "negative"]

class TestSentimentIntegration:
    """Integration tests for the sentiment analysis module."""
    
    def test_end_to_end_analysis(self, mock_analyzer, sample_reviews_df):
        """Test the complete sentiment analysis pipeline with sample data."""
        # Ensure we have cleaned text
        if 'cleaned_text' not in sample_reviews_df.columns:
            sample_reviews_df['cleaned_text'] = sample_reviews_df['review_text'].str.lower()
        
        # Analyze the reviews
        analyzed_df = mock_analyzer.analyze_reviews(sample_reviews_df)
        
        # Basic assertions
        assert not analyzed_df.empty
        assert len(analyzed_df) == len(sample_reviews_df)
        
        # Check sentiment distribution
        sentiment_counts = analyzed_df["sentiment"].value_counts().to_dict()
        assert sum(sentiment_counts.values()) == len(analyzed_df)
        
        # Check that key phrases were extracted
        assert "key_phrases" in analyzed_df.columns
        assert all(isinstance(phrases, list) for phrases in analyzed_df["key_phrases"])
        
        # Test saving results
        output_path = mock_analyzer.save_analysis_results(analyzed_df, "integration_test")
        assert Path(output_path).exists()
        
        # Clean up
        Path(output_path).unlink()
