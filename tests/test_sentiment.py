import pytest
import pandas as pd
from scripts.sentiment_analysis import SentimentAnalyzer

@pytest.fixture
def sample_data(tmp_path):
    """Create a sample CSV for testing."""
    data = pd.DataFrame({
        'review': ['Great app, fast transfers', 'Crashes often', 'Good UI'],
        'rating': [5, 2, 4],
        'date': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'bank': ['CBE', 'BOA', 'Dashen'],
        'source': ['Google Play', 'Google Play', 'Google Play']
    })
    input_path = tmp_path / "test_reviews.csv"
    data.to_csv(input_path, index=False)
    return input_path

@pytest.fixture
def analyzer(sample_data):
    """Initialize SentimentAnalyzer with sample data."""
    return SentimentAnalyzer(input_path=str(sample_data))

def test_compute_sentiment(analyzer):
    """Test sentiment computation."""
    df = analyzer.compute_sentiment()
    assert 'sentiment_label' in df.columns
    assert 'sentiment_score' in df.columns
    assert df['sentiment_label'].isin(['positive', 'negative', 'neutral']).all()
    assert (df['sentiment_score'] >= 0).all() and (df['sentiment_score'] <= 1).all()

def test_extract_keywords(analyzer):
    """Test keyword extraction."""
    df = analyzer.extract_keywords()
    assert 'keywords' in df.columns
    assert all(isinstance(kw, list) for kw in df['keywords'])
    assert 'transfer' in df.iloc[0]['keywords']  # From 'fast transfers'

def test_assign_themes(analyzer):
    """Test theme assignment."""
    analyzer.extract_keywords()
    df = analyzer.assign_themes()
    assert 'themes' in df.columns
    assert all(isinstance(themes, list) for themes in df['themes'])
    assert 'Transaction Performance' in df.iloc[0]['themes']  # From 'transfers'

def test_analyze(analyzer, tmp_path):
    """Test full analysis pipeline."""
    output_path = tmp_path / "reviews_analyzed.csv"
    df = analyzer.analyze(output_path=str(output_path))
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {'review_id', 'review', 'sentiment_label', 'sentiment_score', 'themes'}
    assert len(df) == 3
    assert output_path.exists()