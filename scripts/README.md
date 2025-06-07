Scripts Directory
This directory contains Python scripts for the core functionality of the Fintech Reviews Analytics project, including data collection, preprocessing, analysis, database storage, and visualization.
Scripts

data_collection.py: Scrapes reviews from Google Play Store.
preprocessing.py: Cleans and normalizes review data.
sentiment_analysis.py: Performs sentiment and thematic analysis.
database_storage.py: Sets up and populates Oracle database.
visualization.py: Generates plots and insights.
utils.py: Shared utility functions.

Usage
Run scripts in sequence:
python scripts/data_collection.py
python scripts/preprocessing.py
python scripts/sentiment_analysis.py
python scripts/database_storage.py
python scripts/visualization.py

Ensure dependencies are installed (pip install -r requirements.txt) and config.yaml is configured.
