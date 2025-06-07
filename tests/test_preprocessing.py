
import pytest
import pandas as pd
from scripts.preprocessing import ReviewPreprocessor

@pytest.fixture
def preprocessor():
    return ReviewPreprocessor()

@pytest.fixture
def sample_data(tmp_path):
    """Create a sample CSV for testing."""
    data = pd.DataFrame({
        'review': ['Great app', 'Crashes often', None, 'Good UI'],
        'rating': [5, 2, 3, None],
        'date': ['2023-01-01', 'invalid', '2023-02-01', '2023-03-01'],
        'bank': ['CBE', 'BOA', 'Dashen', 'CBE'],
        'source': ['Google Play', 'Google Play', 'Google Play', 'Google Play']
    })
    input_path = tmp_path / "test_reviews.csv"
    data.to_csv(input_path, index=False)
    return input_path

def test_preprocess(preprocessor, sample_data, tmp_path):
    """Test preprocessing functionality."""
    output_path = tmp_path / "reviews_clean.csv"
    df = preprocessor.preprocess(input_path=str(sample_data), output_path=str(output_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # One row dropped due to invalid date
    assert set(df.columns) == {'review', 'rating', 'date', 'bank', 'source'}
    assert df['review'].isna().sum() == 0
    assert df['rating'].isna().sum() == 0
    assert df['date'].isna().sum() == 0
    assert output_path.exists()
