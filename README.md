# FinTech Reviews Analytics

## Overview

This project analyzes Google Play Store reviews for three Ethiopian banks (CBE, BOA, Dashen) to provide data-driven insights on customer satisfaction and app improvements. The project includes comprehensive text preprocessing, sentiment analysis, and thematic analysis of user reviews.

## Project Status

✅ **Task 1: Data Collection & Preprocessing** - Completed  
🔄 **Task 2: Sentiment & Thematic Analysis** - In Progress  
⏳ **Task 3: Database Integration** - Pending  
⏳ **Task 4: Visualization & Reporting** - Pending

## Project Structure

```
fintech-reviews-analytics/
├── .vscode/                   # VS Code settings
├── .github/                  # GitHub Actions workflows
├── data/                     # Data storage
│   ├── raw/                 # Raw scraped reviews
│   ├── processed/           # Cleaned and processed data
│   │   ├── reviews_clean.csv  # Main processed dataset
│   │   └── reviews_sample.csv # Sample for inspection
│   └── visualizations/      # Generated plots and charts
├── scripts/                 # Python modules
│   ├── data_collection.py   # Scrapes app store reviews
│   ├── preprocessing.py     # Text cleaning and feature engineering
│   ├── sentiment_analysis.py # Sentiment and thematic analysis
│   ├── database_storage.py   # Database operations
│   └── utils.py             # Shared utilities
├── tests/                   # Unit and integration tests
│   ├── test_preprocessing.py
│   └── conftest.py
├── notebooks/               # Jupyter notebooks
│   ├── 01_eda_preprocessing.ipynb
│   └── explore_sentiment.ipynb
├── sql/                     # Database schema and migrations
├── reports/                 # Project reports and documentation
├── .gitignore
├── requirements.txt         # Python dependencies
└── config.yaml              # Configuration settings
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Enhanced Text Preprocessing

The preprocessing pipeline includes advanced text cleaning and feature engineering:

- **Text Cleaning**:
  - URL and HTML tag removal
  - Emoji conversion to text
  - Punctuation and special character removal
  - Stopword removal and lemmatization
  - Non-English word filtering

- **Feature Engineering**:
  - Character and word counts
  - Average word length
  - Sentiment indicators
  - Question/exclamation detection
  - Text complexity metrics

## Usage

1. Configure your settings in `config.yaml`
2. Run the data collection pipeline:
   ```bash
   python -m scripts.data_collection
   ```
3. Process and enhance the data:
   ```bash
   python -m scripts.preprocessing
   ```
4. Run sentiment and thematic analysis:
   ```bash
   python -m scripts.sentiment_analysis
   ```

## Output Files

- `data/processed/reviews_clean.csv`: Full processed dataset
- `data/processed/reviews_sample.csv`: Random sample for inspection
- `data/visualizations/`: Generated plots and charts
- `reports/`: Analysis reports and findings

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.





