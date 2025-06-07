"""
Database storage module for FinTech reviews analytics.

This module provides functionality to store and retrieve review data
from an Oracle database.
"""

import logging
import os
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SQLAlchemy base class
Base = declarative_base()

class Review(Base):
    """Database model for storing reviews."""
    __tablename__ = 'fintech_reviews'
    
    id = Column(Integer, primary_key=True)
    review_id = Column(String(100), unique=True, nullable=False)
    app_name = Column(String(100), nullable=False)
    app_version = Column(String(50))
    review_text = Column(Text)
    cleaned_text = Column(Text)
    rating = Column(Float)
    sentiment_score = Column(Float)
    sentiment = Column(String(20))
    review_date = Column(DateTime)
    author = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Review(id={self.id}, app={self.app_name}, rating={self.rating}, sentiment={self.sentiment})>"

class DatabaseManager:
    """Class for managing database operations."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """Initialize the database manager with configuration.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self.engine = self._create_engine()
        self.Session = sessionmaker(bind=self.engine)
        self._create_tables()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            dict: Configuration dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise
    
    def _create_engine(self):
        """Create a SQLAlchemy engine using the configuration."""
        db_config = self.config['database']
        
        # Get credentials from environment variables if not in config
        username = db_config.get('username') or os.getenv('DB_USERNAME')
        password = db_config.get('password') or os.getenv('DB_PASSWORD')
        
        if not username or not password:
            raise ValueError("Database username and password must be provided in config or environment variables")
        
        # Build connection string
        dsn = f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={db_config['host']})(PORT={db_config['port']}))(CONNECT_DATA=(SERVICE_NAME={db_config['service_name']})))"
        connection_string = f"oracle+cx_oracle://{username}:{password}@{dsn}"
        
        try:
            engine = create_engine(
                connection_string,
                pool_size=db_config.get('pool_size', 5),
                max_overflow=db_config.get('max_overflow', 10),
                echo=True  # Set to False in production
            )
            return engine
        except Exception as e:
            logger.error(f"Error creating database engine: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created/verified")
        except SQLAlchemyError as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def save_reviews(self, reviews_df: pd.DataFrame, app_name: str) -> int:
        """Save reviews to the database.
        
        Args:
            reviews_df: DataFrame containing reviews to save.
            app_name: Name of the app the reviews belong to.
            
        Returns:
            int: Number of reviews saved.
        """
        session = self.Session()
        saved_count = 0
        
        try:
            for _, row in reviews_df.iterrows():
                try:
                    # Check if review already exists
                    existing = session.query(Review).filter_by(
                        review_id=str(row.get('review_id', ''))
                    ).first()
                    
                    if existing:
                        # Update existing review
                        for key, value in row.items():
                            if hasattr(existing, key) and key != 'id':
                                setattr(existing, key, value)
                        existing.updated_at = datetime.utcnow()
                    else:
                        # Create new review
                        review_data = {
                            'review_id': str(row.get('review_id', '')),
                            'app_name': app_name,
                            'app_version': row.get('app_version'),
                            'review_text': row.get('review_text'),
                            'cleaned_text': row.get('cleaned_text'),
                            'rating': row.get('rating'),
                            'sentiment_score': row.get('sentiment_score'),
                            'sentiment': row.get('sentiment'),
                            'review_date': row.get('review_date'),
                            'author': row.get('author')
                        }
                        review = Review(**review_data)
                        session.add(review)
                    
                    saved_count += 1
                    
                    # Commit in batches of 100
                    if saved_count % 100 == 0:
                        session.commit()
                        
                except Exception as e:
                    logger.error(f"Error saving review {row.get('review_id')}: {e}")
                    session.rollback()
                    continue
            
            # Final commit for any remaining records
            session.commit()
            logger.info(f"Successfully saved/updated {saved_count} reviews for {app_name}")
            
        except Exception as e:
            logger.error(f"Error in save_reviews: {e}")
            session.rollback()
            raise
            
        finally:
            session.close()
            
        return saved_count
    
    def get_reviews(self, app_name: str = None, limit: int = 1000) -> pd.DataFrame:
        """Retrieve reviews from the database.
        
        Args:
            app_name: Filter by app name. If None, returns all reviews.
            limit: Maximum number of reviews to return.
            
        Returns:
            pd.DataFrame: DataFrame containing the reviews.
        """
        session = self.Session()
        
        try:
            query = session.query(Review)
            
            if app_name:
                query = query.filter(Review.app_name == app_name)
            
            reviews = query.order_by(Review.review_date.desc()).limit(limit).all()
            
            # Convert to DataFrame
            reviews_list = [{
                'id': r.id,
                'review_id': r.review_id,
                'app_name': r.app_name,
                'app_version': r.app_version,
                'review_text': r.review_text,
                'cleaned_text': r.cleaned_text,
                'rating': r.rating,
                'sentiment_score': r.sentiment_score,
                'sentiment': r.sentiment,
                'review_date': r.review_date,
                'author': r.author,
                'created_at': r.created_at,
                'updated_at': r.updated_at
            } for r in reviews]
            
            return pd.DataFrame(reviews_list)
            
        except Exception as e:
            logger.error(f"Error retrieving reviews: {e}")
            raise
            
        finally:
            session.close()

def main():
    """Main function to demonstrate database operations."""
    try:
        # Example usage
        db = DatabaseManager()
        
        # Example: Load some data and save to database
        # df = pd.read_csv('path/to/analyzed_reviews.csv')
        # db.save_reviews(df, 'Example App')
        
        # Example: Query reviews
        # reviews = db.get_reviews('Example App')
        # print(reviews.head())
        
        logger.info("Database operations completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in database operations: {e}")
        return 1

if __name__ == "__main__":
    main()
