# FinTech Reviews Analytics

A comprehensive analytics pipeline for collecting, processing, and analyzing FinTech application reviews.

## Project Structure

```
fintech-reviews-analytics/
├── .vscode/               # VS Code settings
├── .github/              # GitHub Actions workflows
├── data/                 # Data storage
│   ├── raw/             # Raw scraped reviews
│   ├── processed/       # Cleaned data
│   └── visualizations/  # Generated plots
├── scripts/             # Python scripts for processing
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for analysis
├── sql/                 # Database schema
├── reports/             # Project reports
├── .gitignore          # Git ignore rules
├── requirements.txt    # Python dependencies
└── config.yaml         # Configuration settings
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

## Usage

1. Configure your settings in `config.yaml`
2. Run the data collection pipeline:
   ```bash
   python -m scripts.data_collection
   ```
3. Process the data:
   ```bash
   python -m scripts.preprocessing
   ```
4. Run the analysis:
   ```bash
   python -m scripts.sentiment_analysis
   ```

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
