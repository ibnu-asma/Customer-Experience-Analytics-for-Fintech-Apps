
import pytest
import pandas as pd
from scripts.data_collection import ReviewScraper

@pytest.fixture
def scraper():
    """Fixture to initialize ReviewScraper."""
    return ReviewScraper()

def test_scrape_reviews(scraper):
    """Test scraping reviews for a single app."""
    reviews = scraper.scrape_reviews("com.cbe.mobile", "CBE", count=10)
    assert isinstance(reviews, list)
    assert len(reviews) > 0
    assert all(key in reviews[0] for key in ['review', 'rating', 'date', 'bank', 'source'])
    assert reviews[0]['bank'] == "CBE"
    assert reviews[0]['source'] == "Google Play"

def test_collect_all_reviews(scraper, tmp_path):
    """Test collecting reviews for all apps and saving to CSV."""
    output_path = tmp_path / "test_reviews.csv"
    df = scraper.collect_all_reviews(output_path=str(output_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert set(df.columns) == {'review', 'rating', 'date', 'bank', 'source'}
    assert output_path.exists()
