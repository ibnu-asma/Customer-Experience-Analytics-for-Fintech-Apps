"""
Unit tests for the visualization module.
"""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Import the module to test
from scripts import visualization

class TestReviewVisualizer:
    """Test cases for the ReviewVisualizer class."""
    
    @pytest.fixture
    def mock_visualizer(self, mock_config):
        """Create a ReviewVisualizer instance with a mock config."""
        return visualization.ReviewVisualizer(mock_config)
    
    @pytest.fixture
    def sample_analysis_data(self):
        """Create sample analysis data for testing."""
        return pd.DataFrame({
            'review_date': pd.date_range('2023-01-01', periods=30, freq='D'),
            'rating': np.random.randint(1, 6, 30),
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 30, p=[0.6, 0.3, 0.1]),
            'app_name': ['TestApp'] * 30,
            'cleaned_text': ['test review text ' + str(i) for i in range(30)]
        })
    
    def test_plot_rating_distribution(self, mock_visualizer, sample_analysis_data, tmp_path):
        """Test plotting rating distribution."""
        # Create the plot
        fig = mock_visualizer.plot_rating_distribution(sample_analysis_data)
        
        # Check the return type
        assert isinstance(fig, plt.Figure)
        
        # Test saving the figure
        output_path = tmp_path / "rating_distribution.png"
        mock_visualizer.save_plot(fig, str(output_path))
        
        # Check that the file was created
        assert output_path.exists()
        assert os.path.getsize(output_path) > 0
        
        # Clean up
        output_path.unlink()
    
    def test_plot_sentiment_distribution(self, mock_visualizer, sample_analysis_data, tmp_path):
        """Test plotting sentiment distribution."""
        # Create the plot
        fig = mock_visualizer.plot_sentiment_distribution(sample_analysis_data)
        
        # Check the return type
        assert isinstance(fig, plt.Figure)
        
        # Test saving the plot
        output_path = tmp_path / "sentiment_distribution.png"
        mock_visualizer.save_plot(fig, str(output_path))
        
        # Check that the file was created
        assert output_path.exists()
        assert os.path.getsize(output_path) > 0
        
        # Clean up
        output_path.unlink()
    
    def test_plot_rating_trend(self, mock_visualizer, sample_analysis_data, tmp_path):
        """Test plotting rating trends over time."""
        # Create the plot
        fig = mock_visualizer.plot_rating_trend(sample_analysis_data)
        
        # Check the return type
        assert isinstance(fig, go.Figure)
        
        # Test saving the plot
        output_path = tmp_path / "rating_trend.html"
        mock_visualizer.save_plot(fig, str(output_path))
        
        # Check that the file was created
        assert output_path.exists()
        assert os.path.getsize(output_path) > 0
        
        # Clean up
        output_path.unlink()
    
    def test_generate_wordcloud(self, mock_visualizer, sample_analysis_data, tmp_path):
        """Test generating a word cloud."""
        # Create the word cloud
        fig = mock_visualizer.generate_wordcloud(sample_analysis_data, max_words=20)
        
        # Check the return type
        assert fig is not None
        
        # Test saving the word cloud
        output_path = tmp_path / "wordcloud.png"
        mock_visualizer.save_plot(fig, str(output_path))
        
        # Check that the file was created
        assert output_path.exists()
        assert os.path.getsize(output_path) > 0
        
        # Clean up
        output_path.unlink()
    
    def test_create_dashboard(self, mock_visualizer, sample_analysis_data, tmp_path):
        """Test creating a dashboard with multiple visualizations."""
        # Create the dashboard
        dashboard_path = mock_visualizer.create_dashboard(
            sample_analysis_data, 
            output_dir=str(tmp_path),
            filename="test_dashboard.html"
        )
        
        # Check that the dashboard file was created
        dashboard_path = Path(dashboard_path)
        assert dashboard_path.exists()
        assert os.path.getsize(dashboard_path) > 0
        
        # Clean up
        dashboard_path.unlink()

class TestVisualizationEdgeCases:
    """Test edge cases and error handling in the visualization module."""
    
    @pytest.fixture
    def visualizer(self, mock_config):
        return visualization.ReviewVisualizer(mock_config)
    
    def test_empty_dataframe(self, visualizer):
        """Test with an empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Test each plotting function with empty DataFrame
        with pytest.raises(ValueError):
            visualizer.plot_rating_distribution(empty_df)
        
        with pytest.raises(ValueError):
            visualizer.plot_sentiment_distribution(empty_df)
        
        with pytest.raises(ValueError):
            visualizer.plot_rating_trend(empty_df)
        
        with pytest.raises(ValueError):
            visualizer.generate_wordcloud(empty_df)
    
    def test_missing_columns(self, visualizer):
        """Test with DataFrames missing required columns."""
        # DataFrame missing 'rating' column
        df = pd.DataFrame({
            'sentiment': ['positive', 'negative'],
            'review_date': pd.date_range('2023-01-01', periods=2)
        })
        
        with pytest.raises(KeyError):
            visualizer.plot_rating_distribution(df)
        
        # DataFrame missing 'sentiment' column
        df = pd.DataFrame({
            'rating': [4, 5],
            'review_date': pd.date_range('2023-01-01', periods=2)
        })
        
        with pytest.raises(KeyError):
            visualizer.plot_sentiment_distribution(df)
        
        # DataFrame missing 'review_date' column
        df = pd.DataFrame({
            'rating': [4, 5],
            'sentiment': ['positive', 'negative']
        })
        
        with pytest.raises(KeyError):
            visualizer.plot_rating_trend(df)
    
    def test_invalid_save_path(self, visualizer, sample_analysis_data):
        """Test saving to an invalid path."""
        # Create a plot
        fig = visualizer.plot_rating_distribution(sample_analysis_data)
        
        # Try to save to a non-existent directory
        with pytest.raises(OSError):
            visualizer.save_plot(fig, "/non/existent/path/plot.png")

class TestVisualizationIntegration:
    """Integration tests for the visualization module."""
    
    @pytest.fixture
    def visualizer(self, mock_config):
        return visualization.ReviewVisualizer(mock_config)
    
    def test_end_to_end_visualization(self, visualizer, sample_analysis_data, tmp_path):
        """Test the complete visualization workflow with sample data."""
        # Create output directory
        output_dir = tmp_path / "visualizations"
        output_dir.mkdir()
        
        # Generate all visualizations
        rating_dist = visualizer.plot_rating_distribution(sample_analysis_data)
        sentiment_dist = visualizer.plot_sentiment_distribution(sample_analysis_data)
        rating_trend = visualizer.plot_rating_trend(sample_analysis_data)
        wordcloud = visualizer.generate_wordcloud(sample_analysis_data)
        
        # Save all visualizations
        visualizer.save_plot(rating_dist, str(output_dir / "rating_distribution.png"))
        visualizer.save_plot(sentiment_dist, str(output_dir / "sentiment_distribution.png"))
        visualizer.save_plot(rating_trend, str(output_dir / "rating_trend.html"))
        visualizer.save_plot(wordcloud, str(output_dir / "wordcloud.png"))
        
        # Create a dashboard
        dashboard_path = visualizer.create_dashboard(
            sample_analysis_data,
            output_dir=str(output_dir),
            filename="test_dashboard.html"
        )
        
        # Check that all files were created
        assert (output_dir / "rating_distribution.png").exists()
        assert (output_dir / "sentiment_distribution.png").exists()
        assert (output_dir / "rating_trend.html").exists()
        assert (output_dir / "wordcloud.png").exists()
        assert Path(dashboard_path).exists()
        
        # Check file sizes (should be greater than 0)
        for file in output_dir.glob("*"):
            assert os.path.getsize(file) > 0
        
        # Clean up
        for file in output_dir.glob("*"):
            file.unlink()
        output_dir.rmdir()
    
    def test_visualization_with_missing_values(self, visualizer, tmp_path):
        """Test visualization with DataFrame containing missing values."""
        # Create test data with missing values
        test_data = pd.DataFrame({
            'review_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', None]),
            'rating': [5, None, 3, 4],
            'sentiment': ['positive', 'negative', None, 'neutral'],
            'app_name': ['TestApp'] * 4,
            'cleaned_text': ['good', 'bad', None, 'okay']
        })
        
        # Test each visualization function
        try:
            # Rating distribution should handle missing ratings
            fig1 = visualizer.plot_rating_distribution(test_data)
            assert isinstance(fig1, plt.Figure)
            
            # Sentiment distribution should handle missing sentiments
            fig2 = visualizer.plot_sentiment_distribution(test_data)
            assert isinstance(fig2, plt.Figure)
            
            # Rating trend should handle missing dates and ratings
            fig3 = visualizer.plot_rating_trend(test_data)
            assert isinstance(fig3, go.Figure)
            
            # Word cloud should handle missing text
            fig4 = visualizer.generate_wordcloud(test_data)
            assert fig4 is not None
            
            # Dashboard creation should work with all visualizations
            dashboard_path = visualizer.create_dashboard(
                test_data,
                output_dir=str(tmp_path),
                filename="test_dashboard_missing.html"
            )
            assert Path(dashboard_path).exists()
            
            # Clean up
            Path(dashboard_path).unlink()
            
        except Exception as e:
            pytest.fail(f"Visualization failed with missing values: {str(e)}")
