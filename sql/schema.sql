-- SQL Schema for FinTech Reviews Analytics Database
-- This file contains the DDL for creating the database tables

-- Create sequence for auto-incrementing IDs
CREATE SEQUENCE review_id_seq
    START WITH 1
    INCREMENT BY 1
    NOCACHE
    NOCYCLE;

-- Main reviews table
CREATE TABLE fintech_reviews (
    id NUMBER DEFAULT review_id_seq.NEXTVAL PRIMARY KEY,
    review_id VARCHAR2(100) NOT NULL,
    app_name VARCHAR2(100) NOT NULL,
    app_version VARCHAR2(50),
    review_text CLOB,
    cleaned_text CLOB,
    rating NUMBER(3, 1),
    sentiment_score NUMBER(5, 3),
    sentiment VARCHAR2(20),
    review_date TIMESTAMP,
    author VARCHAR2(100),
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT uq_review_id UNIQUE (review_id)
);

-- Create index on frequently queried columns
CREATE INDEX idx_fintech_reviews_app_name ON fintech_reviews(app_name);
CREATE INDEX idx_fintech_reviews_rating ON fintech_reviews(rating);
CREATE INDEX idx_fintech_reviews_sentiment ON fintech_reviews(sentiment);
CREATE INDEX idx_fintech_reviews_review_date ON fintech_reviews(review_date);

-- Create a view for basic analytics
CREATE OR REPLACE VIEW vw_review_analytics AS
SELECT
    app_name,
    COUNT(*) AS total_reviews,
    ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(MEDIAN(rating), 2) AS median_rating,
    ROUND(STDDEV(rating), 2) AS rating_stddev,
    SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) AS positive_reviews,
    SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) AS neutral_reviews,
    SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) AS negative_reviews,
    ROUND(AVG(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) * 100, 2) AS percent_positive,
    ROUND(AVG(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) * 100, 2) AS percent_neutral,
    ROUND(AVG(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) * 100, 2) AS percent_negative
FROM
    fintech_reviews
GROUP BY
    app_name;

-- Create a table for storing app metadata
CREATE TABLE fintech_apps (
    app_id VARCHAR2(50) PRIMARY KEY,
    app_name VARCHAR2(100) NOT NULL,
    package_name VARCHAR2(200) NOT NULL,
    category VARCHAR2(100),
    description CLOB,
    play_store_url VARCHAR2(500),
    app_store_url VARCHAR2(500),
    website_url VARCHAR2(500),
    is_active NUMBER(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

-- Create a table for storing key phrases extracted from reviews
CREATE TABLE review_key_phrases (
    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    review_id VARCHAR2(100) NOT NULL,
    app_name VARCHAR2(100) NOT NULL,
    phrase VARCHAR2(500) NOT NULL,
    phrase_type VARCHAR2(50),
    sentiment_score NUMBER(5, 3),
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
    FOREIGN KEY (review_id) REFERENCES fintech_reviews(review_id) ON DELETE CASCADE
);

-- Create a table for storing visualization data
CREATE TABLE visualization_data (
    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    visualization_name VARCHAR2(100) NOT NULL,
    app_name VARCHAR2(100),
    data_type VARCHAR2(50) NOT NULL,
    data_json CLOB NOT NULL,
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

-- Create a table for tracking data collection runs
CREATE TABLE collection_runs (
    run_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    app_name VARCHAR2(100) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR2(20) NOT NULL,
    reviews_collected NUMBER DEFAULT 0,
    error_message VARCHAR2(4000),
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

-- Create a table for storing configuration
CREATE TABLE app_config (
    config_key VARCHAR2(100) PRIMARY KEY,
    config_value CLOB,
    description VARCHAR2(500),
    is_active NUMBER(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

-- Insert default configuration
INSERT INTO app_config (config_key, config_value, description)
VALUES 
    ('scraping.rate_limit', '2', 'Delay in seconds between requests'),
    ('scraping.max_retries', '3', 'Maximum number of retry attempts'),
    ('sentiment.threshold_positive', '0.2', 'Threshold for positive sentiment'),
    ('sentiment.threshold_negative', '-0.2', 'Threshold for negative sentiment'),
    ('visualization.theme', 'darkgrid', 'Default visualization theme'),
    ('visualization.palette', 'viridis', 'Default color palette');

-- Create a view for sentiment analysis by date
CREATE OR REPLACE VIEW vw_sentiment_by_date AS
SELECT
    app_name,
    TRUNC(review_date) AS review_date,
    COUNT(*) AS total_reviews,
    ROUND(AVG(sentiment_score), 3) AS avg_sentiment,
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) AS positive_count,
    SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) AS neutral_count,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_count
FROM
    fintech_reviews
WHERE
    review_date IS NOT NULL
GROUP BY
    app_name, TRUNC(review_date)
ORDER BY
    app_name, TRUNC(review_date) DESC;

-- Create a view for version analysis
CREATE OR REPLACE VIEW vw_version_analysis AS
SELECT
    app_name,
    app_version,
    COUNT(*) AS review_count,
    ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(AVG(sentiment_score), 3) AS avg_sentiment,
    ROUND(AVG(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) * 100, 2) AS percent_positive,
    MIN(review_date) AS first_review_date,
    MAX(review_date) AS last_review_date
FROM
    fintech_reviews
WHERE
    app_version IS NOT NULL
GROUP BY
    app_name, app_version
ORDER BY
    app_name, first_review_date DESC;
