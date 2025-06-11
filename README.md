# FinTech Reviews Analytics

## Overview

This project analyzes Google Play Store reviews for three Ethiopian banks (Commercial Bank of Ethiopia, Bank of Abyssinia, and Dashen Bank) to provide data-driven insights on customer satisfaction and app improvements. The project includes comprehensive text preprocessing, sentiment analysis, thematic analysis, database integration, and visualization/reporting of user reviews.

## Project Status

✅ **Task 1: Data Collection & Preprocessing** - Completed  
✅ **Task 2: Sentiment & Thematic Analysis** - Completed  
✅ **Task 3: Database Integration** - Completed  
✅ **Task 4: Visualization & Reporting** - Completed  

## Project Structure
fintech-reviews-analytics/
├── .vscode/                   # VS Code settings
├── .github/                  # GitHub Actions workflows
├── data/                     # Data storage
│   ├── raw/                 # Raw scraped reviews
│   ├── processed/           # Cleaned and processed data
│   │   ├── reviews_clean.csv  # Main processed dataset
│   │   └── reviews_analyzed.csv # Analyzed dataset with sentiment and themes
│   └── visualizations/      # Generated plots and charts (e.g., sentiment_trends_monthly.png)
├── scripts/                 # Python modules
│   ├── data_collection.py   # Scrapes app store reviews
│   ├── preprocessing.py     # Text cleaning and feature engineering
│   ├── sentiment_analysis.py # Sentiment and thematic analysis
│   ├── task2_step3_pipeline.py # Production pipeline for Task 2, Step 3
│   └── utils.py             # Shared utilities
├── tests/                   # Unit and integration tests
│   ├── test_preprocessing.py
│   └── conftest.py
├── notebooks/               # Jupyter notebooks
│   ├── 01_eda_preprocessing.ipynb
│   └── explore_sentiment.ipynb
├── sql/                     # Database schema and migrations
├── reports/                 # Project reports and documentation (e.g., task4_report.pdf)
├── .gitignore
├── requirements.txt         # Python dependencies
└── config.yaml              # Configuration settings

## Setup

1. Clone the repository:
  
   git clone <repository-url>
   cd fintech-reviews-analytics
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
   Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On Unix/MacOS: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Enhanced Text Preprocessing

The preprocessing pipeline includes advanced text cleaning and feature engineering:

### Text Cleaning:
- URL and HTML tag removal
- Emoji conversion to text
- Punctuation and special character removal
- Stopword removal and lemmatization
- Non-English word filtering

### Feature Engineering:
- Character and word counts
- Average word length
- Sentiment indicators (polarity scores)
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

4. Run sentiment and thematic analysis pipeline:
   ```bash
   python scripts/task2_step3_pipeline.py
   ```

5. Generate visualizations and report (via Jupyter notebook):
   ```bash
   jupyter notebook notebooks/explore_sentiment.ipynb
   ```
   Then execute all cells in the notebook.

## Output Files

- `data/processed/reviews_clean.csv`: Full processed dataset with cleaned reviews.
- `data/processed/reviews_analyzed.csv`: Analyzed dataset with sentiment labels, scores, keywords, and themes.
- `data/visualizations/`:
  - `sentiment_trends_monthly.png`: Monthly sentiment trends.
  - `rating_distributions.png`: Box plot of ratings by bank.
  - `keyword_cloud.png`: Word cloud of frequent keywords.
- `reports/task4_report.pdf`: Comprehensive report with insights and recommendations.

## Key Insights

- **Sentiment Analysis**: Dashen Bank leads with 297 positive reviews, while Bank of Abyssinia has the highest negative sentiment (226 reviews).

- **Thematic Analysis**:
  - Dominant theme: "App Usability" (488 reviews across all banks).
  - Bank-specific pain points: 
    - "Transaction Problems" (60 reviews for Dashen)
    - "Login Issues" (17 reviews for Abyssinia)

- **Recommendations**:
  - Add budgeting tools
  - Implement transaction recovery for CBE and Dashen
  - Address login issues for Abyssinia
  - Optimize performance for Dashen

## Testing

Run tests using pytest:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.