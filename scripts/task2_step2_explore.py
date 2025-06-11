# task2_step2_explore.py: Exploratory analysis for Task 2, Step 2
# Purpose: Analyze 50 reviews for sentiment and keywords, propose themes

import pandas as pd
import spacy
from transformers import pipeline
from utils import setup_logging
import logging
import os
from unidecode import unidecode
from spacy.language import Language
from spacy.tokens import Doc

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def load_data(input_path="data/processed/reviews_clean.csv"):
    """Load and subset data to expected columns."""
    try:
        df = pd.read_csv(input_path)
        expected_columns = ['review', 'rating', 'date', 'bank', 'source']
        available_columns = [col for col in expected_columns if col in df.columns]
        if len(available_columns) != len(expected_columns):
            missing = set(expected_columns) - set(available_columns)
            logger.warning("Missing columns: %s. Using available: %s", missing, available_columns)
        df = df[available_columns]
        logger.info("Loaded %d reviews from %s", len(df), input_path)
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        return None

def sample_reviews(df, n=50):
    """Sample ~50 reviews, balanced across banks."""
    try:
        n_per_bank = n // 3
        remainder = n % 3
        samples = []
        for bank in df['bank'].unique():
            bank_df = df[df['bank'] == bank]
            sample_size = n_per_bank + (1 if remainder > 0 else 0)
            remainder -= 1
            sampled = bank_df.sample(n=min(sample_size, len(bank_df)), random_state=42)
            samples.append(sampled)
        sample_df = pd.concat(samples).reset_index(drop=True)
        sample_df['review_id'] = range(1, len(sample_df) + 1)
        logger.info("Sampled %d reviews: %s", len(sample_df), sample_df['bank'].value_counts().to_dict())
        return sample_df
    except Exception as e:
        logger.error("Failed to sample reviews: %s", e)
        return None

def analyze_sentiment(reviews):
    """Compute sentiment using DistilBERT."""
    try:
        sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        sentiments = []
        for review in reviews:
            try:
                result = sentiment_analyzer(review[:512])[0]
                label = 'NEUTRAL' if result['score'] < 0.6 else result['label']
                sentiments.append((label, result['score']))
            except Exception as e:
                logger.warning("Sentiment failed for review: %s", e)
                sentiments.append(('UNKNOWN', 0.0))
        logger.info("Computed sentiment for %d reviews", len(reviews))
        return sentiments
    except Exception as e:
        logger.error("Sentiment analysis failed: %s", e)
        return [('UNKNOWN', 0.0)] * len(reviews)

def extract_keywords(reviews):
    """Extract keywords using spaCy, including bigrams, adverbs, entities."""
    try:
        nlp = spacy.load('en_core_web_sm')
        
        # Add bigram component to spaCy pipeline
        @Language.component("bigram_component")
        def bigram_component(doc):
            bigrams = []
            for i in range(len(doc) - 1):
                if doc[i].pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and doc[i+1].pos_ in ['NOUN', 'VERB']:
                    bigrams.append(f"{doc[i].text}_{doc[i+1].text}")
            doc._.bigrams = bigrams
            return doc
        
        if not nlp.has_pipe("bigram_component"):
            Doc.set_extension("bigrams", default=[], force=True)
            nlp.add_pipe("bigram_component", last=True)

        keywords_list = []
        for review in reviews:
            try:
                doc = nlp(review[:1000])
                keywords = []
                # Extract tokens
                for token in doc:
                    if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']:
                        keywords.append(unidecode(token.text))
                # Extract entities
                for ent in doc.ents:
                    keywords.append(unidecode(ent.text))
                # Extract bigrams
                keywords.extend(doc._.bigrams)
                keywords = list(dict.fromkeys(keywords))[:10]  # Max 10, remove duplicates
                keywords_list.append(keywords)
            except Exception as e:
                logger.warning("Keyword extraction failed for review: %s", e)
                keywords_list.append([])
        logger.info("Extracted keywords for %d reviews", len(reviews))
        return keywords_list
    except Exception as e:
        logger.error("Keyword extraction failed: %s", e)
        return [[]] * len(reviews)

def propose_themes(sample_df):
    """Propose 3-5 themes per bank based on keywords."""
    themes = {}
    try:
        for bank in sample_df['bank'].unique():
            bank_df = sample_df[sample_df['bank'] == bank]
            all_keywords = [kw for keywords in bank_df['keywords'] for kw in keywords]
            keyword_counts = pd.Series(all_keywords).value_counts().head(20).index.tolist()
            bank_themes = []
            # Specific theme mappings (case-insensitive)
            kw_lower = [kw.lower() for kw in keyword_counts]
            if any(kw in ['login', 'crash', 'error', 'sign', 'authentication', 'log_in', 'sign_in'] for kw in kw_lower):
                bank_themes.append(('Login Issues', [kw for kw in keyword_counts if kw.lower() in ['login', 'crash', 'error', 'sign', 'authentication', 'log_in', 'sign_in']]))
            if any(kw in ['transfer', 'payment', 'deposit', 'transaction', 'send', 'pay', 'transact'] for kw in kw_lower):
                bank_themes.append(('Transaction Problems', [kw for kw in keyword_counts if kw.lower() in ['transfer', 'payment', 'deposit', 'transaction', 'send', 'pay', 'transact']]))
            if any(kw in ['app', 'interface', 'ui', 'design', 'navigation', 'usability'] for kw in kw_lower):
                bank_themes.append(('App Usability', [kw for kw in keyword_counts if kw.lower() in ['app', 'interface', 'ui', 'design', 'navigation', 'usability']]))
            if any(kw in ['support', 'service', 'help', 'customer', 'response', 'assist'] for kw in kw_lower):
                bank_themes.append(('Customer Support', [kw for kw in keyword_counts if kw.lower() in ['support', 'service', 'help', 'customer', 'response', 'assist']]))
            if any(kw in ['slow', 'lag', 'performance', 'speed', 'fast', 'quick', 'slowly'] for kw in kw_lower):
                bank_themes.append(('Performance Issues', [kw for kw in keyword_counts if kw.lower() in ['slow', 'lag', 'performance', 'speed', 'fast', 'quick', 'slowly']]))
            # Fallback themes
            if len(bank_themes) < 3:
                other_keywords = [kw for kw in keyword_counts if kw.lower() not in sum([t[1] for t in bank_themes], [])][:5]
                if other_keywords:
                    bank_themes.append(('App Feedback', other_keywords))
            if len(bank_themes) < 3:
                bank_themes.append(('Miscellaneous', keyword_counts[:5]))
            themes[bank] = bank_themes[:5]
            logger.info("Proposed themes for %s: %s", bank, bank_themes)
        return themes
    except Exception as e:
        logger.error("Theme proposal failed: %s", e)
        return {}

def save_results(sample_df, output_path="data/processed/sample_50_reviews.csv"):
    """Save analysis results to CSV."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sample_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info("Saved results to %s", output_path)
    except Exception as e:
        logger.error("Failed to save results: %s", e)

if __name__ == "__main__":
    logger.info("Starting Task 2, Step 2 exploratory analysis")
    df = load_data()
    if df is not None:
        sample_df = sample_reviews(df)
        if sample_df is not None:
            sentiments = analyze_sentiment(sample_df['review'])
            sample_df['sentiment_label'], sample_df['sentiment_score'] = zip(*sentiments)
            sample_df['keywords'] = extract_keywords(sample_df['review'])
            themes = propose_themes(sample_df)
            print("Sample Analysis Results:\n", sample_df[['review_id', 'bank', 'sentiment_label', 'sentiment_score', 'keywords']].head(5))
            print("\nProposed Themes:")
            for bank, bank_themes in themes.items():
                print(f"{bank}:")
                for theme, keywords in bank_themes:
                    print(f"  - {theme}: {keywords}")
            save_results(sample_df)
    logger.info("Completed Task 2, Step 2 exploratory analysis")