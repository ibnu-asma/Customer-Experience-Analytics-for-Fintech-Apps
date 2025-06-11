# task2_step3_pipeline.py: Production pipeline for Task 2, Step 3
# Purpose: Analyze all reviews for sentiment, keywords, themes

import pandas as pd
import spacy
import logging
import os
import re
from unidecode import unidecode
from spacy.language import Language
from spacy.tokens import Doc
from utils import setup_logging
from sentiment_analysis import compute_sentiment  # Assumed module

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
        df['review_id'] = range(1, len(df) + 1)
        logger.info("Loaded %d reviews from %s", len(df), input_path)
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        return None

def extract_keywords(reviews):
    """Extract keywords using spaCy, including bigrams, negations, entities."""
    try:
        nlp = spacy.load('en_core_web_sm')
        
        # Add bigram component
        @Language.component("bigram_component")
        def bigram_component(doc):
            bigrams = []
            for i in range(len(doc) - 1):
                if (doc[i].pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] or doc[i].text.lower() in ['not', 'no']) and doc[i+1].pos_ in ['NOUN', 'VERB', 'ADJ']:
                    bigrams.append(f"{doc[i].text}_{doc[i+1].text}")
            doc._.bigrams = bigrams
            return doc
        
        if not nlp.has_pipe("bigram_component"):
            Doc.set_extension("bigrams", default=[], force=True)
            nlp.add_pipe("bigram_component", last=True)

        keywords_list = []
        stop_words = {'anede', 'one', 'very', 'many', 'yebazaabataale', '', 'star', 'recomend', 'good', 'nice', 'poor', 'best', 'happy'}
        for review in reviews:
            try:
                doc = nlp(review[:1000])
                keywords = []
                # Extract tokens
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'] or token.text.lower() in ['not', 'no']) and token.text.lower() not in stop_words:
                        kw = unidecode(token.text)
                        if kw and re.match(r'[a-zA-Z0-9_]+$', kw):
                            keywords.append(kw)
                # Extract entities
                for ent in doc.ents:
                    kw = unidecode(ent.text)
                    if kw.lower() not in stop_words and re.match(r'[a-zA-Z0-9_]+$', kw):
                        keywords.append(kw)
                # Extract bigrams
                keywords.extend([bg for bg in doc._.bigrams if not any(sw in bg.lower() for sw in stop_words) and re.match(r'[a-zA-Z0-9_]+$', bg)])
                keywords = list(dict.fromkeys(keywords))[:10]
                # Fallback for short reviews
                if not keywords and review.strip():
                    first_word = unidecode(review.split()[0].lower())
                    if first_word not in stop_words and re.match(r'[a-zA-Z0-9_]+$', first_word):
                        keywords = [first_word]
                keywords_list.append(keywords)
            except Exception as e:
                logger.warning("Keyword extraction failed for review: %s", e)
                keywords_list.append([])
        logger.info("Extracted keywords for %d reviews", len(reviews))
        return keywords_list
    except Exception as e:
        logger.error("Keyword extraction failed: %s", e)
        return [[]] * len(reviews)

def propose_themes(df):
    """Propose 3-5 themes per bank based on keywords."""
    themes = {}
    try:
        for bank in df['bank'].unique():
            bank_df = df[df['bank'] == bank]
            all_keywords = [kw for keywords in bank_df['keywords'] for kw in keywords]
            keyword_counts = pd.Series(all_keywords).value_counts().head(20).index.tolist()
            bank_themes = []
            kw_lower = [kw.lower() for kw in keyword_counts]
            # Specific theme mappings with regex
            if any(re.search(r'log(in|_)|crash|error|sign|auth', kw) for kw in kw_lower):
                matched = [kw for kw in keyword_counts if re.search(r'log(in|_)|crash|error|sign|auth', kw.lower())]
                bank_themes.append(('Login Issues', matched))
            if any(re.search(r'trans|pay|depos|send|cash', kw) for kw in kw_lower):
                matched = [kw for kw in keyword_counts if re.search(r'trans|pay|depos|send|cash', kw.lower())]
                bank_themes.append(('Transaction Problems', matched))
            if any(re.search(r'app|ui|design|navig|usab', kw) for kw in kw_lower):
                matched = [kw for kw in keyword_counts if re.search(r'app|ui|design|navig|usab', kw.lower())]
                bank_themes.append(('App Usability', matched))
            if any(re.search(r'support|serv|help|cust|resp|assist', kw) for kw in kw_lower):
                matched = [kw for kw in keyword_counts if re.search(r'support|serv|help|cust|resp|assist', kw.lower())]
                bank_themes.append(('Customer Support', matched))
            if any(re.search(r'slow|lag|perf|speed|fast|quick', kw) for kw in kw_lower):
                matched = [kw for kw in keyword_counts if re.search(r'slow|lag|perf|speed|fast|quick', kw.lower())]
                bank_themes.append(('Performance Issues', matched))
            # Fallback theme
            used_keywords = sum([t[1] for t in bank_themes], [])
            other_keywords = [kw for kw in keyword_counts if kw not in used_keywords and kw.lower() not in ['easy', 'cool']][:5]
            if other_keywords and len(bank_themes) < 3:
                bank_themes.append(('Other Feedback', other_keywords))
            themes[bank] = bank_themes[:5]
            logger.info("Proposed themes for %s: %s", bank, bank_themes)
        return themes
    except Exception as e:
        logger.error("Theme proposal failed: %s", e)
        return {}

def save_results(df, themes, output_path="data/processed/reviews_analyzed.csv"):
    """Save analysis results to CSV."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Add themes to dataframe
        df['themes'] = df['bank'].map(lambda b: str(themes.get(b, [])))
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info("Saved results to %s", output_path)
    except Exception as e:
        logger.error("Failed to save results: %s", e)

if __name__ == "__main__":
    logger.info("Starting Task 2, Step 3 production pipeline")
    df = load_data()
    if df is not None:
        # Compute sentiment
        sentiments = compute_sentiment(df['review'])
        df['sentiment_label'], df['sentiment_score'] = zip(*sentiments)
        # Extract keywords
        df['keywords'] = extract_keywords(df['review'])
        # Propose themes
        themes = propose_themes(df)
        # Save results
        save_results(df, themes)
        # Print summary
        print("Sample Analysis Results:\n", df[['review_id', 'bank', 'sentiment_label', 'sentiment_score', 'keywords', 'themes']].head(5))
        print("\nProposed Themes:")
        for bank, bank_themes in themes.items():
            print(f"{bank}:")
            for theme, keywords in bank_themes:
                print(f"  - {theme}: {keywords}")
    logger.info("Completed Task 2, Step 3 production pipeline")