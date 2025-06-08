# Week 2 Mobile Banking Analysis

This repository contains code and data for analyzing mobile banking app reviews for Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank.

## Task 1: Data Collection and Preprocessing

### Methodology
- **Web Scraping**: Used `google-play-scraper` to collect 400+ reviews per bank (1,200+ total) from the Google Play Store. Reviews include text, ratings, dates, and app names.
- **Preprocessing**: Removed duplicates, handled missing data, and normalized dates to YYYY-MM-DD format. Saved as `cleaned_reviews.csv`.
- **Tools**: Python with `pandas` and `google-play-scraper`.

### Files
- `scrape_reviews.py`: Script to scrape reviews.
- `preprocess_reviews.py`: Script to clean data.
- `raw_reviews.csv`: Raw scraped data.
- `cleaned_reviews.csv`: Processed data.
- `requirements.txt`: List of dependencies.

### KPIs
- Collected 1,200+ reviews with <5% missing data.
- Organized Git repo with clear commits.

### Next Steps
- Proceed to Task 2 for sentiment and theme analysis.