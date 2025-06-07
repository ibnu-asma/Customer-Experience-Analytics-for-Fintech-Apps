"""
Unit tests for the utility functions module.
"""

import pytest
import os
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open

# Import the module to test
from scripts import utils

class TestConfigLoading:
    """Test cases for configuration loading utilities."""
    
    def test_load_config_default(self, tmp_path):
        """Test loading default config when no file is provided."""
        # Create a temporary config file
        config_path = tmp_path / "config.yaml"
        test_config = {"test_key": "test_value"}
        config_path.write_text(yaml.dump(test_config))
        
        # Test loading the config
        with patch('scripts.utils.CONFIG_PATH', str(config_path)):
            config = utils.load_config()
            
        assert config == test_config
    
    def test_load_config_custom_path(self, tmp_path):
        """Test loading config from a custom path."""
        # Create a temporary config file
        custom_path = tmp_path / "custom_config.yaml"
        test_config = {"custom_key": "custom_value"}
        custom_path.write_text(yaml.dump(test_config))
        
        # Test loading the config from custom path
        config = utils.load_config(str(custom_path))
        assert config == test_config
    
    def test_load_config_missing_file(self):
        """Test behavior when config file is missing."""
        with pytest.raises(FileNotFoundError):
            utils.load_config("/non/existent/config.yaml")
    
    def test_load_config_invalid_yaml(self, tmp_path):
        """Test behavior with invalid YAML content."""
        # Create a file with invalid YAML
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("invalid: yaml: file")
        
        with pytest.raises(yaml.YAMLError):
            utils.load_config(str(bad_config))

class TestDirectoryUtils:
    """Test cases for directory utility functions."""
    
    def test_create_directory_new(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "new_directory"
        
        # Directory should not exist yet
        assert not new_dir.exists()
        
        # Create the directory
        result = utils.create_directory(str(new_dir))
        
        # Check results
        assert result is True
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_create_directory_exists(self, tmp_path):
        """Test behavior when directory already exists."""
        existing_dir = tmp_path / "existing_dir"
        existing_dir.mkdir()
        
        # Try to create the directory again
        result = utils.create_directory(str(existing_dir))
        
        # Should return True (success) without error
        assert result is True
    
    def test_create_directory_permission_error(self, tmp_path):
        """Test behavior when there's a permission error."""
        # Mock os.makedirs to raise a PermissionError
        with patch('os.makedirs') as mock_makedirs:
            mock_makedirs.side_effect = PermissionError("Permission denied")
            
            # Try to create a directory
            result = utils.create_directory("/root/no_permission")
            
            # Should return False on error
            assert result is False

class TestJSONUtils:
    """Test cases for JSON utility functions."""
    
    def test_save_json(self, tmp_path):
        """Test saving data to a JSON file."""
        test_data = {"key": "value", "nested": {"number": 123, "bool": True}}
        output_file = tmp_path / "test.json"
        
        # Save the data
        result = utils.save_json(test_data, str(output_file))
        
        # Check results
        assert result is True
        assert output_file.exists()
        
        # Verify the content
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
            
        assert saved_data == test_data
    
    def test_load_json(self, tmp_path):
        """Test loading data from a JSON file."""
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(test_data))
        
        # Load the data
        loaded_data = utils.load_json(str(json_file))
        
        # Check results
        assert loaded_data == test_data
    
    def test_load_json_invalid_file(self, tmp_path):
        """Test loading from a non-existent or invalid JSON file."""
        # Non-existent file
        assert utils.load_json("/non/existent/file.json") is None
        
        # Invalid JSON file
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{invalid: json}")
        
        with pytest.raises(json.JSONDecodeError):
            utils.load_json(str(bad_json))

class TestDataValidation:
    """Test cases for data validation utilities."""
    
    def test_validate_dataframe_basic(self):
        """Test basic DataFrame validation."""
        # Valid DataFrame
        valid_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10.5, 20.0, 30.5]
        })
        
        assert utils.validate_dataframe(valid_df) is True
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        assert utils.validate_dataframe(empty_df) is False
        
        # Non-DataFrame input
        assert utils.validate_dataframe("not a dataframe") is False
    
    def test_validate_dataframe_required_columns(self):
        """Test DataFrame validation with required columns."""
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['A', 'B']
        })
        
        # Should pass with correct required columns
        assert utils.validate_dataframe(df, required_columns=['id', 'name']) is True
        
        # Should fail with missing required columns
        assert utils.validate_dataframe(df, required_columns=['id', 'name', 'missing']) is False
    
    def test_validate_dataframe_no_nans(self):
        """Test DataFrame validation checking for NaNs."""
        df_with_nans = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, None, 30]  # Contains a NaN
        })
        
        # Should pass when not checking for NaNs
        assert utils.validate_dataframe(df_with_nans, check_nans=False) is True
        
        # Should fail when checking for NaNs
        assert utils.validate_dataframe(df_with_nans, check_nans=True) is False

class TestMiscUtils:
    """Test cases for miscellaneous utility functions."""
    
    def test_get_timestamp(self):
        """Test timestamp generation."""
        timestamp = utils.get_timestamp()
        
        # Should be a string
        assert isinstance(timestamp, str)
        
        # Should be in the expected format (YYYY-MM-DD_HH-MM-SS)
        import re
        assert re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', timestamp)
    
    def test_chunk_list(self):
        """Test chunking a list into smaller chunks."""
        test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Test exact division
        chunks = list(utils.chunk_list(test_list, 5))
        assert len(chunks) == 2
        assert chunks[0] == [1, 2, 3, 4, 5]
        assert chunks[1] == [6, 7, 8, 9, 10]
        
        # Test with remainder
        chunks = list(utils.chunk_list(test_list, 3))
        assert len(chunks) == 4  # 3 full chunks + 1 partial
        assert chunks[-1] == [10]
        
        # Test with chunk size larger than list
        chunks = list(utils.chunk_list(test_list, 20))
        assert len(chunks) == 1
        assert chunks[0] == test_list
    
    def test_safe_get(self):
        """Test safe dictionary access with default values."""
        test_dict = {
            'top': {
                'middle': {
                    'bottom': 'value'
                },
                'list': [1, 2, 3]
            }
        }
        
        # Test existing nested key
        assert utils.safe_get(test_dict, ['top', 'middle', 'bottom']) == 'value'
        
        # Test non-existent key with default
        assert utils.safe_get(test_dict, ['nonexistent'], 'default') == 'default'
        
        # Test partial path
        assert utils.safe_get(test_dict, ['top', 'nonexistent']) is None
        
        # Test with list index
        assert utils.safe_get(test_dict, ['top', 'list', 1]) == 2
        
        # Test invalid path
        assert utils.safe_get(test_dict, ['top', 'middle', 'bottom', 'too_deep']) is None

class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    def test_end_to_end_workflow(self, tmp_path):
        """Test a complete workflow using multiple utility functions."""
        # 1. Create a directory
        data_dir = tmp_path / "data"
        assert utils.create_directory(str(data_dir)) is True
        
        # 2. Create some test data
        test_data = {
            'ids': list(range(1, 101)),
            'values': [i * 10 for i in range(1, 101)]
        }
        
        # 3. Save as JSON
        json_file = data_dir / "test_data.json"
        assert utils.save_json(test_data, str(json_file)) is True
        
        # 4. Load the JSON
        loaded_data = utils.load_json(str(json_file))
        assert loaded_data == test_data
        
        # 5. Convert to DataFrame and validate
        df = pd.DataFrame(loaded_data)
        assert utils.validate_dataframe(df, required_columns=['ids', 'values'], check_nans=True) is True
        
        # 6. Use chunk_list to process in batches
        batches = list(utils.chunk_list(df.to_dict('records'), 25))
        assert len(batches) == 4  # 100 items / 25 per batch = 4 batches
        
        # 7. Test safe_get with the loaded data
        assert utils.safe_get(loaded_data, ['ids', 0]) == 1
        assert utils.safe_get(loaded_data, ['nonexistent'], 'default') == 'default'
        
        # 8. Test timestamp in filenames
        timestamp = utils.get_timestamp()
        new_file = data_dir / f"processed_{timestamp}.json"
        assert utils.save_json({"status": "success"}, str(new_file)) is True
        assert new_file.exists()
        
        # Clean up
        json_file.unlink()
        new_file.unlink()
        data_dir.rmdir()
