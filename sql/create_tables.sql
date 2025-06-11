-- sql/create_tables.sql: Create tables for bank reviews data

-- Drop tables if they exist
DROP TABLE banks CASCADE CONSTRAINTS;
DROP TABLE reviews CASCADE CONSTRAINTS;

-- Create Banks table
CREATE TABLE banks (
    bank_id NUMBER PRIMARY KEY,
    bank_name VARCHAR2(100) NOT NULL UNIQUE
);

-- Create Reviews table
CREATE TABLE reviews (
    review_id NUMBER PRIMARY KEY,
    bank_id NUMBER,
    review_text VARCHAR2(1000),
    rating NUMBER,
    review_date DATE,
    source VARCHAR2(100),
    sentiment_label VARCHAR2(20),
    sentiment_score NUMBER,
    keywords VARCHAR2(500),
    themes VARCHAR2(1000),
    FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
);

-- Initial data can be inserted via Python