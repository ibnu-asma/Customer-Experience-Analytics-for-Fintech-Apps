# task2_step3_pipeline.py: Production pipeline for Task 2, Step 3 with oracledb
# Purpose: Analyze reviews and save to CSV and Oracle XE

import pandas as pd
import spacy
import logging
import os
import re
import oracledb
from unidecode import unidecode
from spacy.language import Language
from spacy.tokens import Doc
from utils import setup_logging
from sentiment_analysis import compute_sentiment

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def load_data(input_path="data/processed/reviews_clean.csv"):
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
    try:
        nlp = spacy.load('en_core_web_sm')
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
        stop_words = {'anede', 'one', 'yebazaabataale', '', 'recomend', 'super', 'many', 'very'}
        for review in reviews:
            try:
                doc = nlp(review[:1000])
                keywords = []
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'] or token.text.lower() in ['not', 'no']) and token.text.lower() not in stop_words:
                        kw = unidecode(token.text)
                        if kw and re.match(r'[a-zA-Z0-9_]+$', kw):
                            keywords.append(kw)
                for ent in doc.ents:
                    kw = unidecode(ent.text)
                    if kw.lower() not in stop_words and re.match(r'[a-zA-Z0-9_]+$', kw):
                        keywords.append(kw)
                keywords.extend([bg for bg in doc._.bigrams if not any(sw in bg.lower() for sw in stop_words) and re.match(r'[a-zA-Z0-9_]+$', bg)])
                keywords = list(dict.fromkeys(keywords))[:10]
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
    themes = {}
    try:
        for bank in df['bank'].unique():
            bank_df = df[df['bank'] == bank]
            all_keywords = [kw for keywords in bank_df['keywords'] for kw in keywords]
            keyword_counts = pd.Series(all_keywords).value_counts()
            keyword_counts = keyword_counts[keyword_counts >= 2].index.tolist()
            bank_themes = []
            kw_lower = [kw.lower() for kw in keyword_counts]
            if any(re.search(r'log(in|_).*|crash|error|sign|auth', kw) for kw in kw_lower):
                matched = [kw for kw in keyword_counts if re.search(r'log(in|_).*|crash|error|sign|auth', kw.lower())]
                bank_themes.append(('Login Issues', matched))
            if any(re.search(r'trans.*|pay.*|depos.*|send|cash', kw) for kw in kw_lower):
                matched = [kw for kw in keyword_counts if re.search(r'trans.*|pay.*|depos.*|send|cash', kw.lower())]
                bank_themes.append(('Transaction Problems', matched))
            if any(re.search(r'app|ui|design|navig|usab', kw) for kw in kw_lower):
                matched = [kw for kw in keyword_counts if re.search(r'app|ui|design|navig|usab', kw.lower())]
                bank_themes.append(('App Usability', matched))
            if any(re.search(r'support|serv.*|help|cust.*|resp|assist', kw) for kw in kw_lower):
                matched = [kw for kw in keyword_counts if re.search(r'support|serv.*|help|cust.*|resp|assist', kw.lower())]
                bank_themes.append(('Customer Support', matched))
            if any(re.search(r'slow|lag|perf.*|speed|fast|quick', kw) for kw in kw_lower):
                matched = [kw for kw in keyword_counts if re.search(r'slow|lag|perf.*|speed|fast|quick', kw.lower())]
                bank_themes.append(('Performance Issues', matched))
            used_keywords = sum([t[1] for t in bank_themes], [])
            other_keywords = [kw for kw in keyword_counts if kw not in used_keywords and kw.lower() not in ['easy', 'cool', 'good', 'nice', 'happy', 'poor', 'best']][:5]
            if other_keywords and len(bank_themes) < 3:
                bank_themes.append(('Other Feedback', other_keywords))
            themes[bank] = bank_themes[:5]
            logger.info("Proposed themes for %s: %s", bank, bank_themes)
        return themes
    except Exception as e:
        logger.error("Theme proposal failed: %s", e)
        return {}

def save_results(df, themes, output_path="data/processed/reviews_analyzed.csv"):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df['themes'] = df['bank'].map(lambda b: str(themes.get(b, [])))
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info("Saved results to %s", output_path)
        # Save to Oracle XE with oracledb
        dsn = oracledb.makedsn('localhost', 1521, service_name='XEPDB1')
        with oracledb.connect(user='sys', password='admin', dsn=dsn, mode=oracledb.SYSDBA) as connection:
            cursor = connection.cursor()
            # Insert banks
            banks = df['bank'].unique()
            for bank_id, bank_name in enumerate(banks, 1):
                cursor.execute("INSERT INTO banks (bank_id, bank_name) VALUES (:1, :2)", (bank_id, bank_name))
            # Insert reviews
            for _, row in df.iterrows():
                bank_id = list(banks).index(row['bank']) + 1
                cursor.execute("""
                    INSERT INTO reviews (review_id, bank_id, review_text, rating, review_date, source, sentiment_label, sentiment_score, keywords, themes)
                    VALUES (:1, :2, :3, :4, TO_DATE(:5, 'YYYY-MM-DD'), :6, :7, :8, :9, :10)
                """, (row['review_id'], bank_id, row['review'], row['rating'], row['date'], row['source'], row['sentiment_label'], row['sentiment_score'], str(row['keywords']), row['themes']))
            connection.commit()
            logger.info("Saved %d reviews to Oracle XE", len(df))
    except Exception as e:
        logger.error("Failed to save results: %s", e)

if __name__ == "__main__":
    logger.info("Starting Task 2, Step 3 production pipeline")
    df = load_data()
    if df is not None:
        sentiments = compute_sentiment(df['review'])
        df['sentiment_label'], df['sentiment_score'] = zip(*sentiments)
        df['keywords'] = extract_keywords(df['review'])
        themes = propose_themes(df)
        save_results(df, themes)
        print("Sample Analysis Results:\n", df[['review_id', 'bank', 'sentiment_label', 'sentiment_score', 'keywords', 'themes']].head(5))
        print("\nProposed Themes:")
        for bank, bank_themes in themes.items():
            print(f"{bank}:")
            for theme, keywords in bank_themes:
                print(f"  - {theme}: {keywords}")
    logger.info("Completed Task 2, Step 3 production pipeline")