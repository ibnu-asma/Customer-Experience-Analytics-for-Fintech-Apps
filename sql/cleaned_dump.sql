-- Cleaned SQL Dump
-- Removed unnecessary metadata and storage parameters

-- Drop tables if they exist
DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS banks CASCADE;

-- Create Banks table
CREATE TABLE banks (
    bank_id NUMBER PRIMARY KEY,
    bank_name VARCHAR2(100) NOT NULL
);

-- Create Reviews table
CREATE TABLE reviews (
    review_id NUMBER PRIMARY KEY,
    bank_id NUMBER,
    review_text CLOB,
    rating NUMBER,
    review_date DATE,
    source VARCHAR2(100),
    sentiment_label VARCHAR2(20),
    sentiment_score NUMBER,
    keywords CLOB,
    themes CLOB,
    FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
);

-- Insert data into banks table
INSERT INTO banks (bank_id, bank_name) VALUES (1, 'Commercial Bank of Ethiopia');
INSERT INTO banks (bank_id, bank_name) VALUES (2, 'Bank of Abyssinia');
INSERT INTO banks (bank_id, bank_name) VALUES (3, 'Dashen Bank');

-- Insert data into reviews table
-- (The original INSERT statements should follow here, but without the metadata)
-- For example:
-- INSERT INTO reviews (review_id, bank_id, review_text, rating, review_date, source, sentiment_label, sentiment_score, keywords, themes) VALUES (...);
