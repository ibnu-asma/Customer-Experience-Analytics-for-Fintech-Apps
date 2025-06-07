import yaml
import pandas as pd
from google_play_scraper import reviews
from datetime import datetime
import logging
from scripts.utils import setup_logging

class ReviewScraper:
    """Class to scrape Google Play Store reviews for specified apps."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize scraper with configuration."""
        setup_logging()
        self.logger = logging.getLogger(__name__)
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            self.apps = self.config['apps']
            self.logger.info("Initialized ReviewScraper with config: %s", config_path)
        except Exception as e:
            self.logger.error("Failed to load config: %s", e)
            raise

    def scrape_reviews(self, app_id, app_name, count=400):
        """Scrape reviews for a given app."""
        try:
            result, _ = reviews(app_id, count=count, lang='en', country='et')
            self.logger.info("Scraped %d reviews for %s", len(result), app_name)
            return [
                {
                    'review': r['content'] if r['content'] else '',
                    'rating': r['score'],
                    'date': r['at'].strftime('%Y-%m-%d') if r['at'] else None,
                    'bank': app_name,
                    'source': 'Google Play'
                }
                for r in result
            ]
        except Exception as e:
            self.logger.error("Error scraping %s: %s", app_name, e)
            return []

    def collect_all_reviews(self, output_path="data/raw/reviews.csv"):
        """Scrape reviews for all banks and save to CSV."""
        try:
            all_reviews = []
            for app_key, app_info in self.apps.items():
                reviews_data = self.scrape_reviews(app_info['id'], app_info['name'])
                all_reviews.extend(reviews_data)
            
            if not all_reviews:
                self.logger.error("No reviews collected")
                return None
                
            df = pd.DataFrame(all_reviews)
            df.to_csv(output_path, index=False)
            self.logger.info("Saved %d reviews to %s", len(df), output_path)
            return df
        except Exception as e:
            self.logger.error("Error collecting reviews: %s", e)
            return None

if __name__ == "__main__":
    scraper = ReviewScraper()
    scraper.collect_all_reviews()