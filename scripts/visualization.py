"""
Visualization module for FinTech reviews analytics.

This module provides functionality to create visualizations and generate
insights from the analyzed review data.
"""

import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.dates as mdates
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewVisualizer:
    """Class for creating visualizations from review data."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """Initialize the visualizer with configuration.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self._setup_plotting()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise
    
    def _setup_plotting(self):
        """Set up plotting style and parameters."""
        # Set the style
        plt.style.use(self.config['visualization'].get('style', 'whitegrid'))
        sns.set_theme(style=self.config['visualization'].get('theme', 'darkgrid'))
        
        # Set default color palette
        self.palette = self.config['visualization'].get('palette', 'viridis')
        sns.set_palette(self.palette)
        
        # Set DPI for saved figures
        self.dpi = self.config['visualization'].get('dpi', 300)
        self.fig_format = self.config['visualization'].get('format', 'png')
    
    def save_figure(self, fig, filename: str, subfolder: str = '') -> str:
        """Save a figure to the visualizations directory.
        
        Args:
            fig: Matplotlib or Plotly figure to save.
            filename: Name for the output file (without extension).
            subfolder: Optional subfolder within the visualizations directory.
            
        Returns:
            str: Path to the saved figure.
        """
        try:
            # Create visualizations directory if it doesn't exist
            vis_dir = Path(self.config['paths']['visualizations'])
            if subfolder:
                vis_dir = vis_dir / subfolder
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filepath
            filepath = vis_dir / f"{filename}.{self.fig_format}"
            
            # Save the figure
            if hasattr(fig, 'savefig'):  # Matplotlib figure
                fig.savefig(
                    filepath,
                    dpi=self.dpi,
                    bbox_inches='tight',
                    format=self.fig_format
                )
            elif hasattr(fig, 'write_image'):  # Plotly figure
                fig.write_image(filepath)
            
            plt.close(fig)
            logger.info(f"Saved figure to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving figure {filename}: {e}")
            raise
    
    def plot_rating_distribution(self, df: pd.DataFrame, app_name: str = '') -> plt.Figure:
        """Create a distribution plot of review ratings.
        
        Args:
            df: DataFrame containing review data with a 'rating' column.
            app_name: Name of the app (for the title).
            
        Returns:
            plt.Figure: The created figure.
        """
        if 'rating' not in df.columns:
            raise ValueError("DataFrame must contain a 'rating' column")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create countplot
        sns.countplot(
            x='rating',
            data=df,
            ax=ax,
            palette=self.palette,
            order=sorted(df['rating'].unique())
        )
        
        # Customize the plot
        title = f"Distribution of Ratings{f' for {app_name}' if app_name else ''}"
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points'
            )
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"{app_name.lower().replace(' ', '_')}_ratings" if app_name else "ratings_distribution"
        self.save_figure(fig, filename, 'distributions')
        
        return fig
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, app_name: str = '') -> plt.Figure:
        """Create a distribution plot of sentiment analysis results.
        
        Args:
            df: DataFrame containing review data with a 'sentiment' column.
            app_name: Name of the app (for the title).
            
        Returns:
            plt.Figure: The created figure.
        """
        if 'sentiment' not in df.columns:
            raise ValueError("DataFrame must contain a 'sentiment' column")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create countplot
        sentiment_order = ['positive', 'neutral', 'negative']
        sns.countplot(
            x='sentiment',
            data=df,
            ax=ax,
            palette=self.palette,
            order=sentiment_order
        )
        
        # Customize the plot
        title = f"Sentiment Distribution{f' for {app_name}' if app_name else ''}"
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points'
            )
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"{app_name.lower().replace(' ', '_')}_sentiment" if app_name else "sentiment_distribution"
        self.save_figure(fig, filename, 'sentiment')
        
        return fig
    
    def plot_ratings_over_time(self, df: pd.DataFrame, app_name: str = '', time_period: str = 'M') -> plt.Figure:
        """Plot average ratings over time.
        
        Args:
            df: DataFrame containing review data with 'review_date' and 'rating' columns.
            app_name: Name of the app (for the title).
            time_period: Time period to group by ('D' for day, 'W' for week, 'M' for month).
            
        Returns:
            plt.Figure: The created figure.
        """
        required_cols = ['review_date', 'rating']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {', '.join(required_cols)}")
        
        # Make a copy and convert date if needed
        df_plot = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_plot['review_date']):
            df_plot['review_date'] = pd.to_datetime(df_plot['review_date'])
        
        # Set date as index and resample
        df_plot.set_index('review_date', inplace=True)
        df_resampled = df_plot['rating'].resample(time_period).mean()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the data
        df_resampled.plot(ax=ax, marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Customize the plot
        title = f"Average Rating Over Time{f' for {app_name}' if app_name else ''}"
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average Rating', fontsize=12)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        period_map = {'D': 'daily', 'W': 'weekly', 'M': 'monthly'}
        period_str = period_map.get(time_period, time_period)
        filename = f"{app_name.lower().replace(' ', '_')}_ratings_over_time_{period_str}" if app_name else f"ratings_over_time_{period_str}"
        self.save_figure(fig, filename, 'trends')
        
        return fig
    
    def create_word_cloud(self, df: pd.DataFrame, text_col: str = 'cleaned_text', 
                         app_name: str = '', max_words: int = 100) -> plt.Figure:
        """Create a word cloud from review text.
        
        Args:
            df: DataFrame containing the text data.
            text_col: Name of the column containing the text.
            app_name: Name of the app (for the title).
            max_words: Maximum number of words to include in the word cloud.
            
        Returns:
            plt.Figure: The created figure.
        """
        if text_col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{text_col}' column")
        
        # Combine all text
        text = ' '.join(df[text_col].dropna().astype(str))
        
        if not text.strip():
            logger.warning("No text data available for word cloud")
            return None
        
        # Create and generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            contour_width=3,
            contour_color='steelblue',
            colormap='viridis'
        ).generate(text)
        
        # Plot the word cloud
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Add title
        title = f"Word Cloud{f' for {app_name}' if app_name else ''}"
        plt.title(title, fontsize=16, pad=20)
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"{app_name.lower().replace(' ', '_')}_wordcloud" if app_name else "wordcloud"
        self.save_figure(fig, filename, 'wordclouds')
        
        return fig
    
    def plot_sentiment_trends(self, df: pd.DataFrame, app_name: str = '', 
                             time_period: str = 'M') -> plt.Figure:
        """Plot sentiment trends over time.
        
        Args:
            df: DataFrame containing review data with 'review_date' and 'sentiment' columns.
            app_name: Name of the app (for the title).
            time_period: Time period to group by ('D' for day, 'W' for week, 'M' for month).
            
        Returns:
            plt.Figure: The created figure.
        """
        required_cols = ['review_date', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {', '.join(required_cols)}")
        
        # Make a copy and convert date if needed
        df_plot = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_plot['review_date']):
            df_plot['review_date'] = pd.to_datetime(df_plot['review_date'])
        
        # Set date as index and resample
        df_plot.set_index('review_date', inplace=True)
        
        # Create a pivot table with sentiment counts over time
        sentiment_counts = pd.crosstab(
            index=df_plot.index,
            columns=df_plot['sentiment'],
            dropna=False
        )
        
        # Resample to the desired time period
        sentiment_resampled = sentiment_counts.resample(time_period).sum()
        
        # Calculate percentages
        sentiment_pct = sentiment_resampled.div(sentiment_resampled.sum(axis=1), axis=0) * 100
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the stacked area chart
        sentiment_pct.plot.area(ax=ax, alpha=0.7, linewidth=1, color=['#2ecc71', '#f1c40f', '#e74c3c'])
        
        # Customize the plot
        title = f"Sentiment Trends Over Time{f' for {app_name}' if app_name else ''}"
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Percentage of Reviews', fontsize=12)
        ax.legend(title='Sentiment', loc='upper left')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save the figure
        period_map = {'D': 'daily', 'W': 'weekly', 'M': 'monthly'}
        period_str = period_map.get(time_period, time_period)
        filename = f"{app_name.lower().replace(' ', '_')}_sentiment_trends_{period_str}" if app_name else f"sentiment_trends_{period_str}"
        self.save_figure(fig, filename, 'trends')
        
        return fig

def main():
    """Main function to generate visualizations."""
    try:
        visualizer = ReviewVisualizer()
        
        # Example: Load some analyzed data
        # processed_dir = Path(visualizer.config['paths']['processed_data'])
        # analysis_file = next(processed_dir.glob('*_analyzed.csv'), None)
        
        # if analysis_file:
        #     df = pd.read_csv(analysis_file)
        #     
        #     # Generate visualizations
        #     visualizer.plot_rating_distribution(df, 'Example App')
        #     visualizer.plot_sentiment_distribution(df, 'Example App')
        #     visualizer.plot_ratings_over_time(df, 'Example App', 'M')
        #     visualizer.plot_sentiment_trends(df, 'Example App', 'M')
        #     visualizer.create_word_cloud(df, 'Example App')
        
        logger.info("Visualization generation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return 1

if __name__ == "__main__":
    main()
