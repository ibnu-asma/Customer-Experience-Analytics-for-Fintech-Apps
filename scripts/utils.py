"""
Utility functions for the FinTech Reviews Analytics project.

This module contains helper functions used across different parts of the project.
"""

import os
import logging
import yaml
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "../config.yaml") -> dict:
    """Load configuration from a YAML file.
    
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

def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory: Directory path to check/create.
        
    Returns:
        Path: Path object of the directory.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_hash(filepath: Union[str, Path], algorithm: str = 'md5', 
                 chunk_size: int = 8192) -> str:
    """Calculate the hash of a file.
    
    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm to use (e.g., 'md5', 'sha1', 'sha256').
        chunk_size: Size of chunks to read from the file.
        
    Returns:
        str: The hexadecimal digest of the file.
    """
    hash_func = hashlib.new(algorithm)
    filepath = Path(filepath)
    
    with filepath.open('rb') as f:
        while chunk := f.read(chunk_size):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def save_json(data: Any, filepath: Union[str, Path], indent: int = 4) -> Path:
    """Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON-serializable).
        filepath: Path to save the JSON file.
        indent: Indentation level for pretty-printing.
        
    Returns:
        Path: Path to the saved file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with filepath.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    logger.debug(f"Saved JSON data to {filepath}")
    return filepath

def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        The loaded data.
    """
    filepath = Path(filepath)
    
    with filepath.open('r', encoding='utf-8') as f:
        return json.load(f)

def format_timestamp(timestamp: Optional[Union[str, float, int]] = None, 
                    fmt: str = '%Y-%m-%d_%H-%M-%S') -> str:
    """Format a timestamp as a string.
    
    Args:
        timestamp: Timestamp to format. If None, uses current time.
        fmt: Format string for the output.
        
    Returns:
        str: Formatted timestamp string.
    """
    if timestamp is None:
        dt = datetime.now()
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    else:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    return dt.strftime(fmt)

def setup_logging(log_file: Optional[Union[str, Path]] = None, 
                 log_level: str = 'INFO') -> None:
    """Set up logging configuration.
    
    Args:
        log_file: Path to the log file. If None, logs only to console.
        log_level: Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of a specified size.
    
    Args:
        lst: List to split.
        chunk_size: Maximum size of each chunk.
        
    Returns:
        List of chunks (lists).
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def safe_get(dictionary: Dict, *keys, default: Any = None) -> Any:
    """Safely get a value from a nested dictionary.
    
    Args:
        dictionary: The dictionary to search in.
        *keys: Keys to traverse in the dictionary.
        default: Default value to return if any key is not found.
        
    Returns:
        The value at the specified path or the default value.
    """
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that a DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.
        
    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing required columns: {', '.join(missing_columns)}")
        return False
    return True

def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path: Path to the project root directory.
    """
    return Path(__file__).parent.parent

def get_data_dir() -> Path:
    """Get the data directory path from config.
    
    Returns:
        Path: Path to the data directory.
    """
    config = load_config()
    return Path(config['paths']['data'])

def get_processed_data_dir() -> Path:
    """Get the processed data directory path from config.
    
    Returns:
        Path: Path to the processed data directory.
    """
    config = load_config()
    return Path(config['paths']['processed_data'])

def get_visualizations_dir() -> Path:
    """Get the visualizations directory path from config.
    
    Returns:
        Path: Path to the visualizations directory.
    """
    config = load_config()
    return Path(config['paths']['visualizations'])

def get_reports_dir() -> Path:
    """Get the reports directory path from config.
    
    Returns:
        Path: Path to the reports directory.
    """
    config = load_config()
    return Path(config['paths']['reports'])
